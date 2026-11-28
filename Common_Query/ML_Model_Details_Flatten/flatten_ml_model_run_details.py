#!/usr/bin/env python3
"""
Flatten `calcFeature` of  prj-prod-dataplatform.audit_balance.ml_model_run_details

Output grain : one row per source row (customerId x digitalLoanAccountId x model run)
Output cols  : every source column (calcFeature is replaced by its expanded features)
               + calcFeature_parse_status (ok / empty / null / error)
               + feature_set_id        id of the (modelDisplayName, modelVersionId, trenchCategory, deviceOs)
                                       combination of that row
               + associated_features   comma-separated feature columns that belong to that combination
               + one column per feature found in ANY model's calcFeature
Also written : <output>_feature_map.parquet/.csv  one row per (combination, feature column) -> join on
               feature_set_id to see exactly which columns a combination uses
               <output>_feature_dictionary.csv    one row per feature column

Fill rule    : feature not present in that row's calcFeature (i.e. not related to the row's
               modelDisplayName / modelVersionId / trenchCategory / deviceOs)
                   numeric feature     -> -1
                   categorical feature -> Not_Applicable

Why the parser is not just json.loads()
    calcFeature was observed in these shapes (all handled):
      1. Python-repr dict          {'a': 1, 'b': None}
      2. Python-repr with numpy    {'a': np.float64(0.5), 'b': np.str_('x'), 'c': Timestamp('2026-11-27 07:39:42')}
      3. Real JSON                 {"a": 1, "b": null, "nested": {"x": 1.0}}   (nested dicts are flattened)
      4. List wrapping a dict      [{'customerId': '123', 'a': 1}]             (single item is unwrapped)
      5. Empty                     {}  /  NULL  / ''
    Parsing uses json first, then a *safe* AST walker (no eval), so nan/inf/None/null/true/false,
    np.* wrappers, Timestamp/datetime and double-encoded JSON all work.

Typical usage
    # dry-run to see bytes scanned
    python flatten_ml_model_run_details.py --start-date 2026-11-01 --end-date 2026-11-26 --dry-run
    # build local parquet
    python flatten_ml_model_run_details.py --start-date 2026-11-01 --end-date 2026-11-26 \
        --output flat.parquet
    # build + load into BigQuery
    python flatten_ml_model_run_details.py --start-date 2026-11-01 --end-date 2026-11-26 \
        --output flat.parquet \
        --dest-table prj-prod-dataplatform.audit_balance.ml_model_run_details_flat \
        --write-disposition WRITE_TRUNCATE
    # offline test on the exported sample
    python flatten_ml_model_run_details.py --input-csv sample.csv --output flat.parquet

Auth: Application Default Credentials  (gcloud auth application-default login)
Deps: pip install google-cloud-bigquery pyarrow
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import decimal
import hashlib
import json
import logging
import math
import os
import pickle
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

log = logging.getLogger("flatten_calc_feature")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
PROJECT_ID = "prj-prod-dataplatform"
SOURCE_TABLE = "prj-prod-dataplatform.audit_balance.ml_model_run_details"

BASE_COLUMNS = [
    "customerId", "digitalLoanAccountId", "prediction", "model", "modelDisplayName",
    "modelVersionId", "calcFeature", "start_time", "end_time", "publish_time",
    "trenchCategory", "deviceOs",
]
BASE_INDEX = {c: i for i, c in enumerate(BASE_COLUMNS)}

NUMERIC_DEFAULT = -1
CATEGORICAL_DEFAULT = "Not_Applicable"
STATUS_COLUMN = "calcFeature_parse_status"

# A "feature set" = the feature columns that actually occur in calcFeature for one combination of
# (modelDisplayName, modelVersionId, trenchCategory, deviceOs).
COMBO_COLUMNS = ["modelDisplayName", "modelVersionId", "trenchCategory", "deviceOs"]
COMBO_INDEX = tuple(BASE_INDEX[c] for c in COMBO_COLUMNS)
ID_COLUMN = "feature_set_id"            # stable id of the combination, key into the *_feature_map table
ASSOC_COLUMN = "associated_features"    # comma-separated feature column names of that combination
MAP_SUFFIX = "_feature_map"

NESTED_SEP = "_"            # deviceV2NewFeatures -> GeneralInfo_model  =>  deviceV2NewFeatures_GeneralInfo_model
MAX_DEPTH = 8               # deeper structures are stored as a JSON string
MAX_LIST_ELEMENTS = 20      # longer lists are stored as a JSON string instead of key_0 ... key_N
MAX_COLUMN_NAME_LEN = 300   # BigQuery limit
RESERVED_PREFIXES = ("_TABLE_", "_FILE_", "_PARTITION", "_ROW_TIMESTAMP", "__ROOT__", "_COLIDENTIFIER")
INT64_MAX = 2 ** 63 - 1
# strict decimal literal (no leading zeros, no whitespace, no nan/inf/hex) -> safe to treat as a number
_NUM_STR_RE = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?$")
_NUM_STR_MAX_LEN = 40

PAGE_SIZE = 50_000          # BigQuery result page size
SPILL_BATCH = 5_000         # records per pickle batch in the local spill file
DEFAULT_CHUNK_ROWS = 20_000 # rows per arrow chunk written to the output

BQ_TO_ARROW = {
    "STRING": pa.string(), "INTEGER": pa.int64(), "INT64": pa.int64(),
    "FLOAT": pa.float64(), "FLOAT64": pa.float64(), "BOOLEAN": pa.bool_(), "BOOL": pa.bool_(),
    "TIMESTAMP": pa.timestamp("us", tz="UTC"), "DATETIME": pa.timestamp("us"),
    "DATE": pa.date32(),
}
CSV_MODE_TYPES = {c: (pa.int64() if c == "customerId" else pa.string()) for c in BASE_COLUMNS}


# --------------------------------------------------------------------------------------
# 1. Robust parsing of the calcFeature string
# --------------------------------------------------------------------------------------
_NAME_CONSTANTS = {
    "None": None, "null": None, "NaT": None, "NA": None,
    "nan": math.nan, "NaN": math.nan, "inf": math.inf, "Infinity": math.inf,
    "true": True, "false": False, "True": True, "False": False,
}
_FLOAT_CALLS = {"float", "float16", "float32", "float64", "float_", "double", "longdouble", "Decimal"}
_INT_CALLS = {"int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
              "intp", "intc", "longlong"}
_BOOL_CALLS = {"bool", "bool_", "bool8"}
_STR_CALLS = {"str", "str_", "unicode_", "UUID"}
_TIME_CALLS = {"Timestamp", "datetime", "date", "time", "Timedelta", "timedelta", "to_datetime"}
_SEQ_CALLS = {"list", "tuple", "set", "frozenset", "array", "ndarray", "asarray"}


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _eval(node: ast.AST, depth: int = 0) -> Any:
    """Evaluate a restricted subset of Python literals. Anything unknown becomes its source text."""
    if depth > 200:
        raise RecursionError("expression nested too deeply")
    d = depth + 1

    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        return None if v is Ellipsis else v

    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if k is None:       # {**other} is not supported -> skip
                continue
            out[str(_eval(k, d))] = _eval(v, d)
        return out

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_eval(e, d) for e in node.elts]

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, d)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
        return ast.unparse(node)

    if isinstance(node, ast.Name):
        return _NAME_CONSTANTS[node.id] if node.id in _NAME_CONSTANTS else ast.unparse(node)

    if isinstance(node, ast.Attribute):          # np.nan, np.inf, pd.NaT, pd.NA ...
        return _NAME_CONSTANTS[node.attr] if node.attr in _NAME_CONSTANTS else ast.unparse(node)

    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        try:
            args = [_eval(a, d) for a in node.args]
            kwargs = {kw.arg: _eval(kw.value, d) for kw in node.keywords if kw.arg}
            if name in _FLOAT_CALLS:
                return float(args[0])
            if name in _INT_CALLS:
                return int(args[0])
            if name in _BOOL_CALLS:
                return bool(args[0])
            if name in _STR_CALLS:
                return str(args[0]) if args else ""
            if name in _TIME_CALLS:
                if args and isinstance(args[0], str):
                    return args[0]
                if args and all(isinstance(a, int) and not isinstance(a, bool) for a in args):
                    if name == "date":
                        return dt.date(*args[:3]).isoformat()
                    if name in ("datetime", "Timestamp"):
                        return dt.datetime(*args[:7]).isoformat(sep=" ")
                return ast.unparse(node)
            if name in _SEQ_CALLS:
                if not args:
                    return []
                return args[0] if isinstance(args[0], list) else [args[0]]
            if name == "dict":
                out = dict(args[0]) if args and isinstance(args[0], dict) else {}
                out.update(kwargs)
                return out
        except RecursionError:
            raise
        except Exception:
            pass
        return ast.unparse(node)

    return ast.unparse(node)


def _decode(value: Any) -> Any:
    """str/bytes/dict/list -> python object. Raises ValueError when the text cannot be understood."""
    obj = value
    if isinstance(obj, bytes):
        obj = obj.decode("utf-8", "replace")
    for _ in range(3):                       # a few passes to undo double-encoded JSON
        if not isinstance(obj, str):
            return obj
        s = obj.strip()
        if s == "" or s.lower() in {"null", "none", "nan"}:
            return None
        try:
            obj = json.loads(s)              # fast path (also accepts NaN / Infinity)
            continue
        except ValueError:
            pass
        try:
            obj = _eval(ast.parse(s, mode="eval").body)
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
            raise ValueError(f"unparseable: {exc}") from exc
    return obj


def parse_blob(value: Any) -> tuple[Any, str]:
    """Return (container_or_None, status) where status is ok / empty / null / error."""
    if value is None:
        return None, "null"
    try:
        obj = _decode(value)
    except ValueError:
        return None, "error"
    if obj is None:
        return None, "null"
    if isinstance(obj, (dict, list)):
        return obj, ("ok" if obj else "empty")
    return None, "error"


# --------------------------------------------------------------------------------------
# 2. Flattening
# --------------------------------------------------------------------------------------
Path = tuple  # tuple[str, ...]


def _norm_scalar(v: Any, diag: Counter | None) -> Any:
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        if diag is not None:
            diag["non_finite_values_treated_as_null"] += 1
        return None
    if isinstance(v, decimal.Decimal):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    return str(v)


def flatten(obj: Any, path: Path = (), out: dict | None = None,
            diag: Counter | None = None, depth: int = 0) -> dict[Path, Any]:
    """Flatten dicts / lists recursively into {path_tuple: scalar}."""
    if out is None:
        out = {}

    if isinstance(obj, dict):
        if depth >= MAX_DEPTH:
            out[path or ("calcFeature_json",)] = json.dumps(obj, default=str)
            return out
        for k, v in obj.items():
            flatten(v, path + (str(k),), out, diag, depth + 1)

    elif isinstance(obj, list):
        if not obj:
            return out
        if depth == 0 and len(obj) == 1:                 # [{...}]  -> unwrap
            flatten(obj[0], path, out, diag, depth)
        elif len(obj) > MAX_LIST_ELEMENTS or depth >= MAX_DEPTH:
            out[path or ("calcFeature_json",)] = json.dumps(obj, default=str)
        else:
            if depth == 0 and diag is not None:
                diag["top_level_lists_with_multiple_items"] += 1
            label = "item" if depth == 0 else ""
            for i, e in enumerate(obj):
                flatten(e, path + (f"{label}{i}",), out, diag, depth + 1)

    else:
        out[path or ("value",)] = _norm_scalar(obj, diag)
    return out


def prediction_features(raw: Any, diag: Counter) -> dict[Path, Any]:
    """Optional: expand the `prediction` column ('0.25' or "{'cb_score': .., 'dl_score': ..}")."""
    try:
        obj = _decode(raw)
    except ValueError:
        return {}
    if isinstance(obj, (dict, list)):
        return flatten(obj, ("prediction",), None, diag, 1)
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return {("prediction", "value"): _norm_scalar(float(obj), diag)}
    return {}


# --------------------------------------------------------------------------------------
# 3. Schema discovery (pass 1) -> final column specs
# --------------------------------------------------------------------------------------
@dataclass
class _Stat:
    n: int = 0
    n_null: int = 0
    has_int: bool = False
    has_float: bool = False
    has_bool: bool = False
    has_str: bool = False
    n_str: int = 0
    n_str_numeric: int = 0          # strings that look like a plain decimal number ('0.1374', '42')
    has_float_str: bool = False     # ... and that are decimal/exponent formatted ('0.1374', '1e-5')
    max_abs_int: int = 0
    max_abs_str_int: int = 0
    models: set = field(default_factory=set)


@dataclass
class ColumnSpec:
    name: str
    path: Path
    kind: str                 # numeric | categorical
    arrow_type: pa.DataType
    default: Any
    null_value: Any
    n: int
    n_null: int
    models: list


def sanitize(raw: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]", "_", raw) or "f"
    if name[0].isdigit():
        name = "f_" + name
    if name.upper().startswith(RESERVED_PREFIXES):
        name = "f_" + name
    return name[:MAX_COLUMN_NAME_LEN]


class SchemaBuilder:
    def __init__(self) -> None:
        self.stats: dict[Path, _Stat] = {}

    def observe(self, path: Path, v: Any, model: str) -> None:
        s = self.stats.get(path)
        if s is None:
            s = self.stats[path] = _Stat()
        s.n += 1
        s.models.add(model)
        if v is None:
            s.n_null += 1
        elif isinstance(v, bool):
            s.has_bool = True
        elif isinstance(v, int):
            s.has_int = True
            s.max_abs_int = max(s.max_abs_int, abs(v))
        elif isinstance(v, float):
            s.has_float = True
        else:
            s.has_str = True
            s.n_str += 1
            if len(v) <= _NUM_STR_MAX_LEN and _NUM_STR_RE.match(v):
                s.n_str_numeric += 1
                if "." in v or "e" in v or "E" in v:
                    s.has_float_str = True
                else:
                    s.max_abs_str_int = max(s.max_abs_str_int, abs(int(v)))

    def finalize(self, reserved: Iterable[str], fill_null_values: bool,
                 force_numeric: Iterable[str] = (), force_categorical: Iterable[str] = ()) -> tuple[list[ColumnSpec], dict]:
        used = {r.lower() for r in reserved}
        force_num = {x.lower() for x in force_numeric}
        force_cat = {x.lower() for x in force_categorical}
        specs: list[ColumnSpec] = []
        info = {"renamed": [], "all_null": [], "coerced_from_strings": [], "mixed_text_and_numbers": []}

        for path in sorted(self.stats, key=lambda p: tuple(map(str, p))):
            s = self.stats[path]
            raw = NESTED_SEP.join(path)
            name = sanitize(raw)
            if name.lower() in used:                       # clashes with a table column (e.g. customerId)
                name = sanitize("calc_" + raw)
            base, i = name, 1
            while name.lower() in used:
                i += 1
                suffix = f"_{i}"
                name = base[: MAX_COLUMN_NAME_LEN - len(suffix)] + suffix
            used.add(name.lower())
            if name != raw:
                info["renamed"].append((raw, name))

            # Type rule:
            #   numeric      : only numbers/bools, OR every string is a plain decimal number AND the column
            #                  also holds real numbers or the strings are decimal-formatted (scores stored as text).
            #   categorical  : anything else. Integer-looking strings only (IDs / codes such as '886685284')
            #                  deliberately stay categorical. Override with --force-numeric / --force-categorical.
            real_num = s.has_int or s.has_float or s.has_bool
            if not s.has_str:
                numeric = real_num
            else:
                numeric = (s.n_str == s.n_str_numeric) and (real_num or s.has_float_str)
                if real_num and not numeric:
                    info["mixed_text_and_numbers"].append(name)
            keys = {name.lower(), raw.lower()}
            if keys & force_num:
                numeric = True
            elif keys & force_cat:
                numeric = False
            if numeric and s.has_str:
                info["coerced_from_strings"].append(name)
            if numeric:
                is_int = (not s.has_float) and (not s.has_float_str) and \
                         max(s.max_abs_int, s.max_abs_str_int) <= INT64_MAX
                kind, atype, default = "numeric", (pa.int64() if is_int else pa.float64()), NUMERIC_DEFAULT
            else:
                kind, atype, default = "categorical", pa.string(), CATEGORICAL_DEFAULT
                if s.n == s.n_null:
                    info["all_null"].append(name)
            specs.append(ColumnSpec(
                name=name, path=path, kind=kind, arrow_type=atype, default=default,
                null_value=(default if fill_null_values else None),
                n=s.n, n_null=s.n_null, models=sorted(s.models),
            ))
        return specs, info


# --------------------------------------------------------------------------------------
# 4. Reading the source (BigQuery or an exported CSV)
# --------------------------------------------------------------------------------------
def build_query(args) -> tuple[str, list]:
    from google.cloud import bigquery

    cols = ", ".join(f"`{c}`" for c in BASE_COLUMNS)
    where, params = ["TRUE"], []
    if args.start_date:
        where.append("CAST(start_time AS DATE) >= @start_date")
        params.append(bigquery.ScalarQueryParameter("start_date", "DATE", dt.date.fromisoformat(args.start_date)))
    if args.end_date:
        where.append("CAST(start_time AS DATE) <= @end_date")
        params.append(bigquery.ScalarQueryParameter("end_date", "DATE", dt.date.fromisoformat(args.end_date)))

    sql = f"SELECT {cols}\nFROM `{args.source_table}`\nWHERE " + " AND ".join(where)
    if args.dedupe_latest:
        sql += (
            "\nQUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId, "
            "modelDisplayName, modelVersionId ORDER BY start_time DESC, end_time DESC) = 1"
        )
    if args.limit:
        sql += f"\nLIMIT {int(args.limit)}"
    return sql, params


def read_bigquery(args):
    """Returns (None, None) for --dry-run, else (base_types, row_iterator)."""
    from google.cloud import bigquery

    client = bigquery.Client(project=args.project)
    sql, params = build_query(args)
    log.info("Query:\n%s", sql)
    cfg = bigquery.QueryJobConfig(query_parameters=params)

    if args.dry_run:
        cfg.dry_run, cfg.use_query_cache = True, False
        job = client.query(sql, job_config=cfg)
        log.info("DRY RUN: query would process %.2f GB", job.total_bytes_processed / 1e9)
        return None, None

    result = client.query(sql, job_config=cfg).result(page_size=PAGE_SIZE)
    base_types = {}
    for f in result.schema:
        base_types[f.name] = BQ_TO_ARROW.get(f.field_type.upper(), pa.string())
    rows = (tuple(row[c] for c in BASE_COLUMNS) for row in result)
    return base_types, rows


def read_csv(path: str):
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

    def gen():
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing = [c for c in BASE_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(f"CSV is missing columns: {missing}")
            for row in reader:
                out = []
                for c in BASE_COLUMNS:
                    v = row[c]
                    if v == "":
                        v = None
                    elif c == "customerId":
                        try:
                            v = int(v)
                        except ValueError:
                            v = None
                    out.append(v)
                yield tuple(out)

    return dict(CSV_MODE_TYPES), gen()


# --------------------------------------------------------------------------------------
# 5. Build the flat table (pass 2) and write it
# --------------------------------------------------------------------------------------
def _to_base(v: Any, t: pa.DataType) -> Any:
    if v is None:
        return None
    if pa.types.is_string(t):
        if isinstance(v, str):
            return v
        return json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
    if pa.types.is_int64(t):
        return int(v)
    if pa.types.is_float64(t):
        return float(v)
    return v


def _convert(spec: ColumnSpec, v: Any) -> Any:
    if v is None:
        return spec.null_value
    if spec.kind == "numeric":
        try:
            return float(v) if pa.types.is_float64(spec.arrow_type) else int(v)
        except (ValueError, TypeError, OverflowError):      # text that is not a number (only with --force-numeric)
            return spec.null_value
    return v if isinstance(v, str) else str(v)


def _iter_spill(f) -> Iterator[tuple]:
    f.seek(0)
    while True:
        try:
            batch = pickle.load(f)
        except EOFError:
            return
        yield from batch


def _open_writer(path: str, schema: pa.Schema):
    if path.lower().endswith(".csv"):
        return pacsv.CSVWriter(path, schema)
    return pq.ParquetWriter(path, schema, compression="snappy")


def feature_set_id(combo: tuple) -> str:
    """Deterministic id for a (modelDisplayName, modelVersionId, trenchCategory, deviceOs) combination."""
    return "FS_" + hashlib.sha1(json.dumps(list(combo), default=str).encode("utf-8")).hexdigest()[:10]


def _combo_text(x: Any) -> str | None:
    return None if x is None else str(x)


def _col_type_name(s: ColumnSpec) -> str:
    if pa.types.is_int64(s.arrow_type):
        return "INT64"
    return "FLOAT64" if pa.types.is_float64(s.arrow_type) else "STRING"


def build_feature_sets(combo_rows: dict, combo_paths: dict, specs: list[ColumnSpec], output: str):
    """Per combination: id + feature column list. Writes <output>_feature_map.parquet/.csv (long form)."""
    order = {s.path: j for j, s in enumerate(specs)}
    name_by_path = {s.path: s.name for s in specs}
    spec_by_path = {s.path: s for s in specs}
    combo_info, rows = {}, []
    for combo in sorted(combo_rows, key=lambda c: tuple("" if x is None else str(x) for x in c)):
        fid = feature_set_id(combo)
        paths = sorted(combo_paths[combo], key=order.__getitem__)
        combo_info[combo] = (fid, ",".join(name_by_path[p] for p in paths))
        for p in paths:
            s = spec_by_path[p]
            rows.append((fid, *map(_combo_text, combo), combo_rows[combo], s.name, _col_type_name(s), str(s.default)))
    ids = [v[0] for v in combo_info.values()]
    if len(set(ids)) != len(ids):
        raise RuntimeError("feature_set_id hash collision - please report")

    names = ["feature_set_id", *COMBO_COLUMNS, "n_rows", "column_name", "column_type", "default_value"]
    types = [pa.string()] * 5 + [pa.int64(), pa.string(), pa.string(), pa.string()]
    table = pa.table({n: pa.array([r[i] for r in rows], type=t) for i, (n, t) in enumerate(zip(names, types))})
    stem = os.path.splitext(output)[0] + MAP_SUFFIX
    pq.write_table(table, stem + ".parquet", compression="snappy")
    pacsv.write_csv(table, stem + ".csv")
    return combo_info, stem + ".parquet", stem + ".csv"


def _build_chunk(records: list, out_base: list[str], base_types: dict, specs: list[ColumnSpec],
                 path_idx: dict, schema: pa.Schema, combo_info: dict, inline_list: bool) -> pa.Table:
    n = len(records)
    arrays = []
    for c in out_base:
        i, t = BASE_INDEX[c], base_types[c]
        arrays.append(pa.array([_to_base(r[0][i], t) for r in records], type=t))
    arrays.append(pa.array([r[2] for r in records], type=pa.string()))
    infos = [combo_info[tuple(r[0][i] for i in COMBO_INDEX)] for r in records]
    arrays.append(pa.array([x[0] for x in infos], type=pa.string()))
    if inline_list:
        arrays.append(pa.array([x[1] for x in infos], type=pa.string()))

    cols = [[s.default] * n for s in specs]
    for row_i, (_, feats, _) in enumerate(records):
        for path, v in feats.items():
            j = path_idx[path]
            cols[j][row_i] = _convert(specs[j], v)
    for s, values in zip(specs, cols):
        arrays.append(pa.array(values, type=s.arrow_type))
    return pa.Table.from_arrays(arrays, schema=schema)


def process(base_types: dict, rows: Iterable[tuple], args) -> dict:
    display_i, version_i, calc_i, pred_i = (BASE_INDEX[c] for c in
                                            ("modelDisplayName", "modelVersionId", "calcFeature", "prediction"))
    schema_builder, diag = SchemaBuilder(), Counter()
    combo_rows: Counter = Counter()          # combination -> number of rows
    combo_paths: dict = {}                   # combination -> set of feature paths present in its calcFeature
    inline_list = getattr(args, "inline_feature_list", True)
    intern: dict = {}   # share one tuple object per feature path -> pickle memoises it (spill ~3x smaller)
    out_base = [c for c in BASE_COLUMNS if c != "calcFeature" or args.keep_raw_calcfeature]

    # ---- pass 1: parse + flatten every row once, spill to local disk, learn the schema ----
    with tempfile.TemporaryFile() as spill:
        buf, total = [], 0
        for base in rows:
            obj, status = parse_blob(base[calc_i])
            feats = flatten(obj, diag=diag) if status == "ok" else {}
            if args.expand_prediction:
                feats.update(prediction_features(base[pred_i], diag))
            feats = {intern.setdefault(p, p): v for p, v in feats.items()}
            diag[f"calcFeature_status_{status}"] += 1
            model = f"{base[display_i]} {base[version_i]}"
            for p, v in feats.items():
                schema_builder.observe(p, v, model)
            combo = tuple(base[i] for i in COMBO_INDEX)
            combo_rows[combo] += 1
            cp = combo_paths.get(combo)
            if cp is None:
                cp = combo_paths[combo] = set()
            cp.update(feats)                                   # keys = feature paths present in this row
            buf.append((base, feats, status))
            total += 1
            if len(buf) >= SPILL_BATCH:
                pickle.dump(buf, spill, pickle.HIGHEST_PROTOCOL)
                buf = []
            if total % 100_000 == 0:
                log.info("pass 1: %s rows parsed", f"{total:,}")
        if buf:
            pickle.dump(buf, spill, pickle.HIGHEST_PROTOCOL)
        log.info("pass 1 done: %s rows, %s distinct feature paths", f"{total:,}", f"{len(schema_builder.stats):,}")

        specs, info = schema_builder.finalize(out_base + [STATUS_COLUMN, ID_COLUMN, ASSOC_COLUMN],
                                              args.fill_null_values, args.force_numeric, args.force_categorical)
        path_idx = {s.path: j for j, s in enumerate(specs)}
        combo_info, map_parquet, map_csv = build_feature_sets(combo_rows, combo_paths, specs, args.output)

        fields = [pa.field(c, base_types[c]) for c in out_base]
        fields.append(pa.field(STATUS_COLUMN, pa.string()))
        fields.append(pa.field(ID_COLUMN, pa.string()))
        if inline_list:
            fields.append(pa.field(ASSOC_COLUMN, pa.string()))
        fields += [pa.field(s.name, s.arrow_type) for s in specs]
        schema = pa.schema(fields)

        # ---- pass 2: build typed chunks with the canonical schema and write them ----
        written, chunk = 0, []
        writer = _open_writer(args.output, schema)
        try:
            for rec in _iter_spill(spill):
                chunk.append(rec)
                if len(chunk) >= args.chunk_rows:
                    writer.write_table(_build_chunk(chunk, out_base, base_types, specs, path_idx, schema,
                                                    combo_info, inline_list))
                    written += len(chunk)
                    chunk = []
                    log.info("pass 2: %s rows written", f"{written:,}")
            if chunk:
                writer.write_table(_build_chunk(chunk, out_base, base_types, specs, path_idx, schema,
                                                combo_info, inline_list))
                written += len(chunk)
        finally:
            writer.close()

    # ---- feature dictionary: which model(s) each column belongs to ----
    dict_path = os.path.splitext(args.output)[0] + "_feature_dictionary.csv"
    with open(dict_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["column_name", "source_path", "type", "default_when_not_applicable",
                    "rows_present", "rows_null", "models_with_feature"])
        for s in specs:
            w.writerow([s.name, NESTED_SEP.join(s.path), str(s.arrow_type), s.default,
                        s.n, s.n_null, "; ".join(s.models)])

    # ---- summary ----
    n_num = sum(s.kind == "numeric" for s in specs)
    log.info("rows written: %s | feature columns: %d (numeric %d, categorical %d) | total columns: %d",
             f"{written:,}", len(specs), n_num, len(specs) - n_num, len(schema))
    for k, v in sorted(diag.items()):
        log.info("  %s = %s", k, f"{v:,}")
    if info["renamed"]:
        log.info("  columns renamed (raw -> name): %s", info["renamed"][:15])
    if info["coerced_from_strings"]:
        log.info("  typed numeric although some/all values arrive as text: %s", info["coerced_from_strings"])
    if info["mixed_text_and_numbers"]:
        log.warning("  numbers mixed with non-numeric text -> typed STRING (check or use --force-numeric): %s",
                    info["mixed_text_and_numbers"])
    if info["all_null"]:
        log.warning("  features that were NULL in every row (typed STRING): %s", info["all_null"])
    if diag.get("calcFeature_status_error"):
        log.warning("  %s rows had an unparseable calcFeature (all features defaulted); "
                    "filter %s = 'error' to inspect them", f"{diag['calcFeature_status_error']:,}", STATUS_COLUMN)
    sizes = [len(v[1].split(",")) if v[1] else 0 for v in combo_info.values()]
    log.info("feature sets: %d combinations of %s | %d-%d associated feature columns each",
             len(combo_info), "/".join(COMBO_COLUMNS), min(sizes, default=0), max(sizes, default=0))
    log.info("outputs: %s | %s | %s", args.output, dict_path, map_csv)
    return {"rows": written, "specs": specs, "schema": schema, "map_parquet": map_parquet,
            "map_csv": map_csv, "combos": len(combo_info)}


def _load_parquet(client, path: str, dest: str, disposition: str) -> None:
    from google.cloud import bigquery

    cfg = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET, write_disposition=disposition)
    with open(path, "rb") as f:
        job = client.load_table_from_file(f, dest, job_config=cfg)
    job.result()
    table = client.get_table(dest)
    log.info("loaded %s rows x %d columns into %s", f"{table.num_rows:,}", len(table.schema), dest)


def load_to_bigquery(args, map_parquet: str | None = None) -> None:
    """Load the flat parquet to --dest-table and the feature map to <dest-table>_feature_map."""
    from google.cloud import bigquery

    if not args.output.lower().endswith(".parquet"):
        raise SystemExit("--dest-table requires a .parquet --output")
    client = bigquery.Client(project=args.project)
    _load_parquet(client, args.output, args.dest_table, args.write_disposition)
    if map_parquet:
        _load_parquet(client, map_parquet, args.dest_table + MAP_SUFFIX, args.write_disposition)


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=PROJECT_ID)
    p.add_argument("--source-table", default=SOURCE_TABLE)
    p.add_argument("--start-date", help="YYYY-MM-DD, filters on CAST(start_time AS DATE)")
    p.add_argument("--end-date", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--limit", type=int, help="LIMIT rows (testing)")
    p.add_argument("--dedupe-latest", action="store_true",
                   help="keep only the latest run per customerId/digitalLoanAccountId/modelDisplayName/modelVersionId")
    p.add_argument("--input-csv", help="read an exported CSV instead of BigQuery (offline testing)")
    p.add_argument("--output", default="ml_model_run_details_flat.parquet", help=".parquet or .csv")
    p.add_argument("--dest-table", help="also load the parquet into this BigQuery table (project.dataset.table)")
    p.add_argument("--write-disposition", default="WRITE_EMPTY",
                   choices=["WRITE_EMPTY", "WRITE_TRUNCATE", "WRITE_APPEND"])
    p.add_argument("--keep-raw-calcfeature", action="store_true", help="keep the original calcFeature string column")
    p.add_argument("--fill-null-values", action="store_true",
                   help="also replace features present-but-NULL with -1 / Not_Applicable (default: keep NULL)")
    p.add_argument("--expand-prediction", action="store_true",
                   help="add prediction_value / prediction_<key> columns parsed from `prediction`")
    p.add_argument("--force-numeric", nargs="*", default=[], metavar="COL",
                   help="treat these features as numeric (final column name or raw path)")
    p.add_argument("--force-categorical", nargs="*", default=[], metavar="COL",
                   help="treat these features as categorical, e.g. an id-like code")
    p.add_argument("--no-inline-feature-list", dest="inline_feature_list", action="store_false",
                   help="do not add the `associated_features` column (feature_set_id + *_feature_map table remain)")
    p.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    p.add_argument("--dry-run", action="store_true", help="BigQuery dry run only (prints bytes scanned)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    base_types, rows = read_csv(args.input_csv) if args.input_csv else read_bigquery(args)
    if args.dry_run:
        return
    result = process(base_types, rows, args)
    if args.dest_table:
        load_to_bigquery(args, result["map_parquet"])


if __name__ == "__main__":
    main()
