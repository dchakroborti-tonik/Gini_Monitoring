"""
BigQuery -> GCS (parquet) -> DuckDB vector table pipeline
==========================================================

Flow
----
1. EXPORT DATA runs the BigQuery SQL and writes parquet straight to GCS
   (server side: nothing is pulled into pandas / local memory).
2. Parquet shards are downloaded from GCS (compressed, so cheap).
3. DuckDB flattens `calcFeature` JSON (any nesting depth, arrays included) with
   json_tree(), infers Data_type, adds the Prediction row, and writes the table
   `dd_ml_models_vector_details` into a local .duckdb file.

Modified version: Prod `calcFeature` is a Python repr (single quotes, None/True/False,
enum / Timestamp objects), not JSON, so the original kept only the Prediction row for
those models. Such rows are converted to JSON before flattening; apps-score `prediction`
dicts use `combined_score`; Prod wrapper keys are unwrapped to match Dev feature names.
See "Step 3" below. `prediction` is now exported as STRING (was NUMERIC) so dicts survive,
which means a fresh export is needed for the apps-score Prediction fix.

Install:  pip install duckdb google-cloud-bigquery google-cloud-storage
Auth   :  gcloud auth application-default login   (or a service account)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import duckdb

log = logging.getLogger("ml_vector_pipeline")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PROJECT_ID = "prj-prod-dataplatform"
BUCKET = "prod-asia-southeast1-tonik-report-analytics"
GCS_BASE_PREFIX = "DC/Parquet_dump"
GCS_EXPORT_SUBFOLDER = "ml_model_run_details"   # cleaned on every export
BQ_LOCATION = "asia-southeast1"                 # must match the bucket's location
TABLE_NAME = "dd_ml_models_vector_details"
DUCKDB_PATH = "dd_ml_models_vector_details.duckdb"

DTYPE_FLOAT, DTYPE_DATE, DTYPE_STRING = "Float64", "Date", "String"

SQL_QUERY = """
with b1 as
(SELECT
  safe_cast(`customerId` as int64) customerId,
  `digitalLoanAccountId`,
  cast(`prediction` as string) prediction,
  case when `modelDisplayName` like '%Alpha-Cash-CIC-Model%' then 'cic_model_cash'
       when `modelDisplayName` like '%Alpha Cash CIC Model%' then 'cic_model_cash'
       when `modelDisplayName` like '%cic_model_sil%' then 'cic_model_sil'
       when `modelDisplayName` like '%Alpha - CIC-SIL-Model%' then 'cic_model_sil'
       when `modelDisplayName` like '%Sil-Alpha-CIC-SIL-Model%' then 'cic_model_sil'
       when `modelDisplayName` like '%Alpha-Cash-Stack-Model%' then 'alpha_stack_model_cash'
       when `modelDisplayName` like '%alpha_stack_model_cash%' then 'alpha_stack_model_cash'
       when `modelDisplayName` like '%Alpha - StackingModel%' then 'alpha_stack_model_sil'
       when `modelDisplayName` like '%alpha_stack_model_sil%' then 'alpha_stack_model_sil'
       when `modelDisplayName` like '%Beta-Cash-AppScore-Model%' then 'apps_score_cash'
       when `modelDisplayName` like '%Beta-Cash-Demo-Model%' then 'beta_demo_model_cash'
       when `modelDisplayName` like '%Beta-Cash-Stack-Model%'then 'beta_stack_model_cash'
       when `modelDisplayName` like '%Beta - AppsScoreModel%'then 'apps_score_model_sil'
       when `modelDisplayName` like '%Beta - DemoScoreModel%' then 'beta_demo_model_sil'
       when `modelDisplayName` like '%Beta - StackScoreModel%'then 'beta_stack_model_sil'
       else `modelDisplayName` end modelDisplayName,      
  `modelVersionId`,
  `calcFeature`,
  `trenchCategory`,
  `deviceOs`,
  'Prod' Data_selection
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
union all
SELECT
  safe_cast(`customerId` as int64) customerId,
  `digitalLoanAccountId`,
  cast(`prediction` as string) prediction,
  case when `modelDisplayName` like '%Alpha-Cash-CIC-Model%' then 'cic_model_cash'
       when `modelDisplayName` like '%Alpha Cash CIC Model%' then 'cic_model_cash'
       when `modelDisplayName` like '%cic_model_sil%' then 'cic_model_sil'
       when `modelDisplayName` like '%Alpha - CIC-SIL-Model%' then 'cic_model_sil'
       when `modelDisplayName` like '%Sil-Alpha-CIC-SIL-Model%' then 'cic_model_sil'
       when `modelDisplayName` like '%Alpha-Cash-Stack-Model%' then 'alpha_stack_model_cash'
       when `modelDisplayName` like '%alpha_stack_model_cash%' then 'alpha_stack_model_cash'
       when `modelDisplayName` like '%Alpha - StackingModel%' then 'alpha_stack_model_sil'
       when `modelDisplayName` like '%alpha_stack_model_sil%' then 'alpha_stack_model_sil'
       when `modelDisplayName` like '%Beta-Cash-AppScore-Model%' then 'apps_score_cash'
       when `modelDisplayName` like '%Beta-Cash-Demo-Model%' then 'beta_demo_model_cash'
       when `modelDisplayName` like '%Beta-Cash-Stack-Model%'then 'beta_stack_model_cash'
       when `modelDisplayName` like '%Beta - AppsScoreModel%'then 'apps_score_model_sil'
       when `modelDisplayName` like '%Beta - DemoScoreModel%' then 'beta_demo_model_sil'
       when `modelDisplayName` like '%Beta - StackScoreModel%'then 'beta_stack_model_sil'
       else `modelDisplayName` end modelDisplayName,  
  `modelVersionId`,
  `calcFeature`,
  `trenchCategory`,
  `deviceOs`,
  `Data_selection`
FROM
  `prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116`
)
select
    b1.customerId,
    b1.digitalLoanAccountId,
    b1.prediction,
    b1.modelDisplayName,
    b1.modelVersionId,
    b1.calcFeature ,
    b1.trenchCategory,
    b1.deviceOs,
    b1.Data_selection,
  date(lmt.startApplyDateTime) application_date,
  lmt.new_loan_type
from b1
Left join `risk_credit_mis.loan_master_table` lmt on b1.digitalLoanAccountId = lmt.digitalLoanAccountId
;
"""

# --------------------------------------------------------------------------- #
# Step 1 + 2: BigQuery -> GCS -> local
# --------------------------------------------------------------------------- #
def export_query_to_gcs(
    sql: str,
    project: str = PROJECT_ID,
    bucket: str = BUCKET,
    prefix: str = f"{GCS_BASE_PREFIX}/{GCS_EXPORT_SUBFOLDER}/",
    location: Optional[str] = BQ_LOCATION,
    calc_feature_is_json_type: bool = False,
    clean_prefix: bool = True,
) -> str:
    """Run `sql` in BigQuery and export the result as parquet shards to GCS.

    calc_feature_is_json_type: set True if calcFeature is a BigQuery JSON column
    (it is converted to its JSON text). Leave False if it is a STRING column.
    Returns the gs:// prefix that holds the parquet shards.
    """
    from google.cloud import bigquery, storage

    prefix = prefix.strip("/") + "/"
    gcs = storage.Client(project=project)
    bq = bigquery.Client(project=project, location=location)

    if clean_prefix:  # avoid mixing stale shards from an older run
        stale = list(gcs.list_blobs(bucket, prefix=prefix))
        for blob in stale:
            blob.delete()
        log.info("Deleted %d old object(s) under gs://%s/%s", len(stale), bucket, prefix)

    inner = sql.strip().rstrip(";").strip()
    select_list = (
        "* REPLACE (TO_JSON_STRING(calcFeature) AS calcFeature)"
        if calc_feature_is_json_type
        else "*"
    )
    export_sql = f"""
EXPORT DATA OPTIONS (
  uri = 'gs://{bucket}/{prefix}part-*.parquet',
  format = 'PARQUET',
  compression = 'SNAPPY',
  overwrite = true
) AS
SELECT {select_list} FROM (
{inner}
)
"""
    log.info("Running BigQuery EXPORT DATA -> gs://%s/%s", bucket, prefix)
    job = bq.query(export_sql)
    job.result()
    log.info("Export finished. Bytes billed: %s", job.total_bytes_billed)
    return f"gs://{bucket}/{prefix}"


def download_parquet_from_gcs(
    bucket: str,
    prefix: str,
    local_dir: Path,
    project: str = PROJECT_ID,
    workers: int = 4,
    chunk_mb: int = 8,
    attempts: int = 5,
    timeout: int = 300,
) -> list[Path]:
    """Download every parquet shard under gs://bucket/prefix into local_dir.

    Built for flaky connections: each file is fetched in `chunk_mb` ranged
    requests (a dropped connection only re-sends one chunk), written to a
    .part file and renamed only once its size matches GCS, and the whole file is
    retried up to `attempts` times. Files already complete in local_dir are skipped.
    """
    from google.cloud import storage

    prefix = prefix.strip("/") + "/"
    client = storage.Client(project=project)
    blobs = [b for b in client.list_blobs(bucket, prefix=prefix) if b.name.endswith(".parquet")]
    if not blobs:
        raise FileNotFoundError(f"No parquet files under gs://{bucket}/{prefix}")

    local_dir.mkdir(parents=True, exist_ok=True)

    def _dl(blob) -> Path:
        dest = local_dir / Path(blob.name).name
        if dest.exists() and dest.stat().st_size == blob.size:
            return dest  # finished by an earlier attempt / run
        part = dest.with_name(dest.name + ".part")
        blob.chunk_size = chunk_mb * 1024 * 1024  # multiple of 256 KiB
        for attempt in range(1, attempts + 1):
            try:
                blob.download_to_filename(str(part), timeout=timeout)
                got = part.stat().st_size
                if got != blob.size:
                    raise IOError(f"size mismatch: got {got}, expected {blob.size}")
                part.replace(dest)
                return dest
            except Exception as exc:
                part.unlink(missing_ok=True)
                if attempt == attempts:
                    raise
                wait = 2 ** attempt
                log.warning("%s: download attempt %d/%d failed (%s); retrying in %ds",
                            dest.name, attempt, attempts, exc, wait)
                time.sleep(wait)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        files = list(ex.map(_dl, blobs))
    log.info("Downloaded %d parquet file(s) to %s", len(files), local_dir)
    return sorted(files)


# --------------------------------------------------------------------------- #
# Step 3: DuckDB processing
# --------------------------------------------------------------------------- #
# What is different from Ml_Vector_Pipeline.py
# --------------------------------------------
# * Prod `calcFeature` (audit_balance.ml_model_run_details) is a Python repr of a dict
#   (single quotes, None/True/False, <Enum: 'x'>, Timestamp('...')), not JSON, so
#   json_valid() rejected it and only the Prediction row survived. Rows that are not
#   valid JSON are now converted to JSON by `pyrepr_to_json` (a tolerant tokenizer,
#   run in a process pool) and flattened exactly like valid-JSON rows.
# * Apps-score models store `prediction` as a dict ({'cb_score':..,'combined_score':..});
#   the score in PREDICTION_DICT_KEY becomes the Prediction row.
# * Prod wraps features in one extra key for some models ({"stackScoreModel": {...}}),
#   Dev data does not. Wrapper keys in UNWRAP_KEYS (and 1-element top-level arrays)
#   are removed so Prod and Dev feature names match.
# * Rows that still cannot be parsed are reported per model/version/Data_selection and a
#   few raw samples are written to a CSV, instead of one warning line per file.

UNWRAP_KEYS = ("stackScoreModel", "demoScoreModel", "appScoreModel")
UNWRAP_SINGLE_ELEMENT_ARRAY = True
PREDICTION_DICT_KEY = "combined_score"

# Numeric text: 12, -3.5, .5, 1e-5  (leading-zero strings like '007' stay String: ids/zip codes)
_NUM_RE = r'^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$'
_LEADING_ZERO_RE = r'^\s*[-+]?0\d'
# ISO-like dates / timestamps: 2024-01-31, 2024/01/31, 2024-01-31T10:20:30.123Z, 2024-01-31 10:20
_DATE_RE = (r'^\s*\d{4}[-/]\d{2}[-/]\d{2}'
            r'([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?\s*(Z|[+-]\d{2}:?\d{2})?)?\s*$')


# --- Python repr -> JSON ---------------------------------------------------- #
# A quote is a string delimiter only when followed by , : } ] or end of text; any other
# quote of the same kind is an apostrophe inside the string ('Internet Cafe's' is one string).
_ENDS = r"(?!\s*(?:[,:}\]]|$))"
_TOKEN = re.compile(
    r"(?P<sq>'(?:[^'\\]|\\.|'" + _ENDS + r")*')"
    r'|(?P<dq>"(?:[^"\\]|\\.|"' + _ENDS + r')*")'
    r"|(?P<enum><[A-Za-z_][\w.]*(?:\s*:\s*(?P<ev>'[^']*'|\"[^\"]*\"|[^<>\s,]+))?>)"
    r"|(?P<call>[A-Za-z_][\w.]*\((?P<args>[^()]*)\))"
    r"|(?P<spec>[-+]?(?:inf|Infinity|nan|NaN)\b)"
    r"|(?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"|(?P<word>[A-Za-z_][\w.]*)",
    re.S,
)
_ESC = re.compile(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)", re.S)
_SIMPLE_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "0": "\0"}
_WORDS = {"None": "null", "null": "null", "True": "true", "true": "true",
          "False": "false", "false": "false"}
_NUM_ONLY = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _unescape(m: "re.Match") -> str:
    c = m.group(1)
    return chr(int(c[1:], 16)) if len(c) > 1 else _SIMPLE_ESC.get(c, c)


def _json_str(inner: str) -> str:
    if "\\" not in inner and '"' not in inner and inner.isprintable():
        return '"' + inner + '"'
    text = _ESC.sub(_unescape, inner)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # JSON-style 😀 surrogate pairs
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return json.dumps(text, ensure_ascii=False)


def _json_num(t: str) -> str:
    t = t.lstrip("+")
    if t.endswith("."):
        t += "0"
    sign = "-" if t.startswith("-") else ""
    t = t.lstrip("-")
    return sign + ("0" + t if t.startswith(".") else t)


def _quoted(t: str) -> bool:
    return len(t) >= 2 and t[0] == t[-1] and t[0] in "'\""


def _call_repl(m: "re.Match") -> str:
    text, args = m.group(), m.group("args").strip()
    base = text[:text.index("(")].rsplit(".", 1)[-1]
    inner = args[1:-1] if _quoted(args) else args
    if base == "Decimal":
        return _json_num(inner) if _NUM_ONLY.fullmatch(inner) else "null"
    if _quoted(args):                        # Timestamp('2026-04-08 12:17:53'), UUID('..')
        return _json_str(inner)
    if _NUM_ONLY.fullmatch(args):
        return _json_num(args)
    parts = [p.strip() for p in args.split(",")]
    if base in ("datetime", "date") and 3 <= len(parts) <= 7 and all(p.isdigit() for p in parts):
        y, mo, d, *rest = (int(p) for p in parts)
        out = f"{y:04d}-{mo:02d}-{d:02d}"
        if rest:
            rest += [0] * (3 - len(rest))
            out += f" {rest[0]:02d}:{rest[1]:02d}:{rest[2]:02d}"
        return '"' + out + '"'
    return json.dumps(text)


def _token_repl(m: "re.Match") -> str:
    kind = m.lastgroup
    if kind in ("sq", "dq"):
        return _json_str(m.group()[1:-1])
    if kind == "num":
        return _json_num(m.group())
    if kind == "word":
        return _WORDS.get(m.group()) or json.dumps(m.group())
    if kind == "spec":
        return "null"
    if kind == "call":
        return _call_repl(m)
    ev = m.group("ev")                       # <TrenchCategoryEnum.TRENCH_1: 'Trench 1'>
    if ev is None:
        return json.dumps(m.group()[1:-1])
    if _quoted(ev):
        return _json_str(ev[1:-1])
    return _json_num(ev) if _NUM_ONLY.fullmatch(ev) else json.dumps(ev)


def pyrepr_to_json(s: str) -> str:
    """Convert a Python-repr / loosely-quoted dict text into JSON text.

    Valid JSON passes through unchanged in meaning. Not validated here; callers
    check the result with DuckDB json_valid().
    """
    return _TOKEN.sub(_token_repl, s)


def _convert_chunk(chunk: list) -> list:
    out = []
    for s in chunk:
        try:
            out.append(pyrepr_to_json(s))
        except Exception:
            out.append(None)
    return out


def _extract_prediction_chunk(chunk: list) -> list:
    out = []
    for s in chunk:
        val = None
        try:
            obj = json.loads(pyrepr_to_json(s))
            if isinstance(obj, dict):
                obj = obj.get(PREDICTION_DICT_KEY)
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                if math.isfinite(obj):
                    val = repr(float(obj))
            elif isinstance(obj, str):
                val = obj
        except Exception:
            pass
        out.append(val)
    return out


def _map_chunks(fn, chunks: list, pool) -> list:
    if pool is not None:
        try:
            return [x for part in pool.map(fn, chunks) for x in part]
        except BrokenProcessPool:
            log.warning("Process pool died; converting the rest of this batch in-process")
    return [x for chunk in chunks for x in fn(chunk)]


# --- SQL -------------------------------------------------------------------- #
_SRC_SQL = r"""
CREATE OR REPLACE TEMP TABLE src AS
SELECT
  CAST(row_number() OVER () - 1 AS BIGINT)  AS rid,
  TRY_CAST(customerId AS BIGINT)            AS customerId,
  CAST(digitalLoanAccountId AS VARCHAR)     AS digitalLoanAccountId,
  CAST(modelDisplayName AS VARCHAR)         AS modelDisplayName,
  CAST(modelVersionId AS VARCHAR)           AS modelVersionId,
  CAST(trenchCategory AS VARCHAR)           AS trenchCategory,
  CAST(deviceOs AS VARCHAR)                 AS deviceOs,
  CAST(new_loan_type AS VARCHAR)            AS new_loan_type,
  CAST(application_date AS DATE)            AS application_date,
  CAST(Data_selection AS VARCHAR)           AS Data_selection,
  trim(CAST(prediction AS VARCHAR))         AS pred_raw,
  trim(CAST(calcFeature AS VARCHAR))        AS cf_raw
FROM read_parquet('__PATH__')
"""

# Rows the DuckDB-native path cannot use; they go through the Python converter.
_CF_NEEDS_FIX = "cf_raw IS NOT NULL AND cf_raw <> '' AND NOT json_valid(cf_raw)"
_PRED_NEEDS_FIX = "pred_raw IS NOT NULL AND pred_raw <> '' AND TRY_CAST(pred_raw AS DOUBLE) IS NULL"

# cf_json: valid JSON (or double-encoded JSON) as-is, else the Python-converted text;
# then 1-element top-level arrays and known wrapper keys are unwrapped.
_SJ_SQL = r"""
CREATE OR REPLACE TEMP TABLE sj AS
WITH a AS (
  SELECT s.*,
    COALESCE(
      CASE WHEN json_valid(s.cf_raw) THEN
             CASE WHEN json_type(s.cf_raw) = 'VARCHAR'            -- double-encoded JSON string
                  THEN CASE WHEN json_valid(json_extract_string(s.cf_raw, '$'))
                            THEN json_extract_string(s.cf_raw, '$') END
                  ELSE s.cf_raw END
      END,
      CASE WHEN json_valid(f.cf_json) THEN f.cf_json END
    ) AS cf_json0,
    COALESCE(TRY_CAST(s.pred_raw AS DOUBLE), TRY_CAST(p.pred AS DOUBLE)) AS pred_num
  FROM src s
  LEFT JOIN cf_fix f ON f.rid = s.rid
  LEFT JOIN pred_fix p ON p.rid = s.rid
),
b AS (
  SELECT *,
    CASE WHEN __UNWRAP_ARRAY__ AND json_type(cf_json0) = 'ARRAY' AND json_array_length(cf_json0) = 1
         THEN CAST(json_extract(cf_json0, '$[0]') AS VARCHAR)
         ELSE cf_json0 END AS cf_json1
  FROM a
)
SELECT * EXCLUDE (cf_json0, cf_json1),
  CASE WHEN __WRAP_COND__
       THEN CAST(json_extract(cf_json1, '$."' || json_keys(cf_json1)[1] || '"') AS VARCHAR)
       ELSE cf_json1 END AS cf_json
FROM b
"""

_STATS_SQL = r"""
SELECT modelDisplayName, modelVersionId, Data_selection,
       count(*)                                                          AS n_rows,
       count(*) FILTER (WHERE cf_json IS NOT NULL AND cf_raw IS NOT NULL
                          AND NOT json_valid(cf_raw))                    AS n_repaired,
       count(*) FILTER (WHERE cf_json IS NULL AND cf_raw IS NOT NULL
                          AND cf_raw <> '')                              AS n_unparseable,
       count(*) FILTER (WHERE cf_raw IS NULL OR cf_raw = '')             AS n_empty,
       count(*) FILTER (WHERE pred_num IS NULL OR NOT isfinite(pred_num)) AS n_no_pred
FROM sj GROUP BY 1, 2, 3
"""

_SAMPLES_SQL = r"""
SELECT modelDisplayName, modelVersionId, Data_selection, left(cf_raw, 300)
FROM sj
WHERE cf_json IS NULL AND cf_raw IS NOT NULL AND cf_raw <> ''
QUALIFY row_number() OVER (PARTITION BY modelDisplayName, modelVersionId, Data_selection
                           ORDER BY rid) <= 3
"""

_INSERT_SQL = r"""
INSERT INTO "__TABLE__"
-- (a) Prediction row
SELECT customerId, digitalLoanAccountId, modelDisplayName, modelVersionId, trenchCategory,
       deviceOs, new_loan_type, application_date, Data_selection,
       'Prediction' AS Features,
       CAST(pred_num AS VARCHAR) AS "Values",
       '__DTYPE_FLOAT__' AS Data_type,
       FALSE AS is_delete, TRUE AS is_active, DATE '__RUN_DATE__' AS run_date
FROM sj
WHERE pred_num IS NOT NULL AND isfinite(pred_num)

UNION ALL

-- (b) one row per leaf of calcFeature (objects and arrays flattened to any depth)
SELECT customerId, digitalLoanAccountId, modelDisplayName, modelVersionId, trenchCategory,
       deviceOs, new_loan_type, application_date, Data_selection,
       feature_name AS Features,
       v            AS "Values",
       CASE
         WHEN v IS NULL                                          THEN '__DTYPE_STRING__'
         WHEN jtype IN ('BIGINT', 'UBIGINT', 'DOUBLE')           THEN '__DTYPE_FLOAT__'
         WHEN jtype = 'BOOLEAN'                                  THEN '__DTYPE_STRING__'
         WHEN regexp_matches(v, '__DATE_RE__')
              AND TRY_CAST(substr(trim(v), 1, 10) AS DATE) IS NOT NULL
                                                                 THEN '__DTYPE_DATE__'
         WHEN regexp_matches(v, '__NUM_RE__')
              AND NOT regexp_matches(v, '__LEADING_ZERO_RE__')   THEN '__DTYPE_FLOAT__'
         ELSE '__DTYPE_STRING__'
       END AS Data_type,
       FALSE AS is_delete, TRUE AS is_active, DATE '__RUN_DATE__' AS run_date
FROM (
  SELECT s.customerId, s.digitalLoanAccountId, s.modelDisplayName, s.modelVersionId,
         s.trenchCategory, s.deviceOs, s.new_loan_type, s.application_date, s.Data_selection,
         -- '$.a.b[0].c' -> 'a.b[0].c' ; '$."my key"' -> 'my key'
         replace(regexp_replace(jt.fullkey, '^\$\.?', ''), '"', '') AS feature_name,
         json_extract_string(jt.atom, '$')                          AS v,
         jt.type                                                    AS jtype
  FROM (SELECT * FROM sj WHERE cf_json IS NOT NULL) s,
       json_tree(s.cf_json) jt
  WHERE jt.type NOT IN ('OBJECT', 'ARRAY')   -- leaves only (incl. JSON null)
    AND jt.fullkey <> '$'
) f
"""

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KEY = re.compile(r"^[A-Za-z0-9_]+$")


def _fill(template: str, path: str = "", table: str = "", run_date: str = "",
          unwrap_array: bool = False, wrap_cond: str = "FALSE") -> str:
    repl = {
        "__PATH__": path.replace("'", "''"),
        "__TABLE__": table,
        "__RUN_DATE__": run_date,
        "__NUM_RE__": _NUM_RE,
        "__LEADING_ZERO_RE__": _LEADING_ZERO_RE,
        "__DATE_RE__": _DATE_RE,
        "__DTYPE_FLOAT__": DTYPE_FLOAT,
        "__DTYPE_DATE__": DTYPE_DATE,
        "__DTYPE_STRING__": DTYPE_STRING,
        "__UNWRAP_ARRAY__": "TRUE" if unwrap_array else "FALSE",
        "__WRAP_COND__": wrap_cond,
    }
    sql = template
    for k, v in repl.items():
        sql = sql.replace(k, v)
    return sql


def _wrap_condition(keys: Iterable[str]) -> str:
    keys = list(keys)
    for k in keys:
        if not _KEY.match(k):
            raise ValueError(f"Unsafe unwrap key: {k!r}")
    if not keys:
        return "FALSE"
    in_list = ", ".join(f"'{k}'" for k in keys)
    return (f"json_type(cf_json1) = 'OBJECT' AND len(json_keys(cf_json1)) = 1 "
            f"AND json_keys(cf_json1)[1] IN ({in_list})")


def _create_table_sql(table: str, if_not_exists: bool = False) -> str:
    ine = "IF NOT EXISTS " if if_not_exists else ""
    return f"""
CREATE TABLE {ine}"{table}" (
  customerId BIGINT,
  digitalLoanAccountId VARCHAR,
  modelDisplayName VARCHAR,
  modelVersionId VARCHAR,
  trenchCategory VARCHAR,
  deviceOs VARCHAR,
  new_loan_type VARCHAR,
  application_date DATE,
  Data_selection VARCHAR,
  Features VARCHAR,
  "Values" VARCHAR,
  Data_type VARCHAR,
  is_delete BOOLEAN,
  is_active BOOLEAN,
  run_date DATE
)"""


def _repair(con, raw_col: str, fix_table: str, predicate: str, fn, pool,
            batch_rows: int = 100_000, chunk: int = 5_000) -> int:
    """Push rows matching `predicate` through `fn` (Python) into temp table `fix_table`."""
    import pandas as pd

    n = con.execute(f"SELECT count(*) FROM src WHERE {predicate}").fetchone()[0]
    if not n:
        return 0
    max_rid = con.execute("SELECT max(rid) FROM src").fetchone()[0]
    for lo in range(0, max_rid + 1, batch_rows):
        rows = con.execute(
            f"SELECT rid, {raw_col} FROM src "
            f"WHERE rid >= {lo} AND rid < {lo + batch_rows} AND {predicate}"
        ).fetchall()
        if not rows:
            continue
        raws = [r[1] for r in rows]
        chunks = [raws[i:i + chunk] for i in range(0, len(raws), chunk)]
        fixed = _map_chunks(fn, chunks, pool)
        df = pd.DataFrame({"rid": [r[0] for r in rows], "v": fixed})
        con.register("_fix_df", df)
        con.execute(f"INSERT INTO {fix_table} SELECT rid, v FROM _fix_df")
        con.unregister("_fix_df")
    return n


def _default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def build_duckdb_table(
    parquet_files: Iterable[Path],
    duckdb_path: str = DUCKDB_PATH,
    table_name: str = TABLE_NAME,
    run_date: Optional[date] = None,
    mode: str = "replace",
    memory_limit: Optional[str] = None,
    threads: Optional[int] = None,
    convert_workers: Optional[int] = None,
    unwrap_keys: Iterable[str] = UNWRAP_KEYS,
    unwrap_single_element_array: bool = UNWRAP_SINGLE_ELEMENT_ARRAY,
    failure_sample_path: Optional[str] = None,
) -> str:
    """Flatten parquet files into the vector table inside a local DuckDB file.

    mode="replace": table is rebuilt in a staging table and swapped in only if
                    every file succeeded (a failed run never leaves a half table).
    mode="append" : rows are added to the existing table (created if missing).
    convert_workers: processes used to convert Python-repr calcFeature to JSON
                    (default: min(8, cores-2); 1 = in-process).
    failure_sample_path: CSV for raw samples of rows that still fail to parse
                    (default: calcFeature_parse_failures.csv next to the .duckdb file).
    """
    if not _IDENT.match(table_name):
        raise ValueError(f"Unsafe table name: {table_name!r}")
    if mode not in ("replace", "append"):
        raise ValueError("mode must be 'replace' or 'append'")

    files = [Path(f) for f in parquet_files]
    if not files:
        raise ValueError("No parquet files supplied")
    run_date_s = (run_date or date.today()).isoformat()
    staging = f"{table_name}__staging"
    wrap_cond = _wrap_condition(unwrap_keys)
    workers = convert_workers or _default_workers()
    out_dir = Path(duckdb_path).resolve().parent
    sample_csv = Path(failure_sample_path) if failure_sample_path else out_dir / "calcFeature_parse_failures.csv"

    pool = None
    if workers > 1:
        try:
            pool = ProcessPoolExecutor(max_workers=workers)
        except Exception as exc:  # e.g. restricted environment
            log.warning("Could not start %d converter processes (%s); running in-process", workers, exc)

    con = duckdb.connect(duckdb_path)
    stats: dict = {}
    samples: dict = {}
    try:
        if memory_limit:
            con.execute(f"SET memory_limit='{memory_limit}'")
        if threads:
            con.execute(f"SET threads={int(threads)}")
        spill = out_dir / ".duckdb_spill"
        con.execute(f"SET temp_directory='{spill}'")

        con.execute(f'DROP TABLE IF EXISTS "{staging}"')
        con.execute(_create_table_sql(staging))

        for i, f in enumerate(files, 1):
            path = str(f)
            con.execute(_fill(_SRC_SQL, path=path))
            con.execute("CREATE OR REPLACE TEMP TABLE cf_fix (rid BIGINT, cf_json VARCHAR)")
            con.execute("CREATE OR REPLACE TEMP TABLE pred_fix (rid BIGINT, pred VARCHAR)")
            n_cf = _repair(con, "cf_raw", "cf_fix", _CF_NEEDS_FIX, _convert_chunk, pool)
            n_pred = _repair(con, "pred_raw", "pred_fix", _PRED_NEEDS_FIX,
                             _extract_prediction_chunk, pool)
            con.execute(_fill(_SJ_SQL, unwrap_array=unwrap_single_element_array,
                              wrap_cond=wrap_cond))

            for m, v, sel, n_rows, n_rep, n_bad, n_empty, n_nopred in con.execute(_STATS_SQL).fetchall():
                acc = stats.setdefault((m, v, sel), [0, 0, 0, 0, 0])
                for j, x in enumerate((n_rows, n_rep, n_bad, n_empty, n_nopred)):
                    acc[j] += x
            for m, v, sel, raw in con.execute(_SAMPLES_SQL).fetchall():
                bucket = samples.setdefault((m, v, sel), [])
                if len(bucket) < 3:
                    bucket.append(raw)

            con.execute(_fill(_INSERT_SQL, table=staging, run_date=run_date_s))
            con.execute("DROP TABLE sj")
            con.execute("DROP TABLE src")
            log.info("[%d/%d] processed %s (converted from Python repr: calcFeature=%d, prediction=%d)",
                     i, len(files), f.name, n_cf, n_pred)

        if mode == "replace":
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(f'ALTER TABLE "{staging}" RENAME TO "{table_name}"')
        else:
            con.execute(_create_table_sql(table_name, if_not_exists=True))
            con.execute(f'INSERT INTO "{table_name}" SELECT * FROM "{staging}"')
            con.execute(f'DROP TABLE "{staging}"')

        n = con.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
        dist = con.execute(
            f'SELECT Data_type, count(*) FROM "{table_name}" '
            f"WHERE run_date = DATE '{run_date_s}' GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        log.info("Table %s rows=%d | this run by Data_type=%s", table_name, n, dist)
        _report_coverage(stats, samples, sample_csv)
        con.execute("CHECKPOINT")
    finally:
        con.close()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(out_dir / ".duckdb_spill", ignore_errors=True)
    return duckdb_path


def _report_coverage(stats: dict, samples: dict, sample_csv: Path) -> None:
    """Log what was repaired / what still has no features, per model/version/Data_selection."""
    import pandas as pd

    cols = ["modelDisplayName", "modelVersionId", "Data_selection",
            "rows", "repaired", "unparseable", "empty_calcFeature", "no_prediction"]
    df = pd.DataFrame([(*k, *v) for k, v in sorted(stats.items(), key=lambda kv: tuple(map(str, kv[0])))],
                      columns=cols)
    tot = df[["rows", "repaired", "unparseable", "empty_calcFeature", "no_prediction"]].sum()
    log.info("calcFeature: %d rows | converted from Python repr: %d | still unparseable: %d | "
             "empty/NULL: %d | rows without a Prediction: %d",
             tot["rows"], tot["repaired"], tot["unparseable"], tot["empty_calcFeature"], tot["no_prediction"])
    bad = df[(df["unparseable"] > 0) | (df["empty_calcFeature"] > 0) | (df["no_prediction"] > 0)]
    if bad.empty:
        return
    log.warning("Groups with rows missing features/prediction:\n%s", bad.to_string(index=False))
    if samples:
        rows = [(m, v, sel, raw) for (m, v, sel), raws in samples.items() for raw in raws]
        pd.DataFrame(rows, columns=["modelDisplayName", "modelVersionId", "Data_selection",
                                    "calcFeature_first_300_chars"]).to_csv(sample_csv, index=False)
        log.warning("Raw samples of unparseable calcFeature written to %s", sample_csv)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_pipeline(
    sql: str = SQL_QUERY,
    project: str = PROJECT_ID,
    bucket: str = BUCKET,
    gcs_prefix: str = f"{GCS_BASE_PREFIX}/{GCS_EXPORT_SUBFOLDER}/",
    location: Optional[str] = BQ_LOCATION,
    duckdb_path: str = DUCKDB_PATH,
    table_name: str = TABLE_NAME,
    calc_feature_is_json_type: bool = False,
    skip_export: bool = False,
    mode: str = "replace",
    run_date: Optional[date] = None,
    local_dir: Optional[str] = None,
    keep_local_parquet: bool = False,
    memory_limit: Optional[str] = None,
    threads: Optional[int] = None,
    convert_workers: Optional[int] = None,
) -> str:
    """BigQuery SQL -> parquet in GCS -> DuckDB table `table_name` in `duckdb_path`.

    skip_export=True re-uses parquet already sitting in GCS (no BigQuery cost),
    handy when only the DuckDB logic changes. Note: parquet exported by the older
    script holds a NUMERIC `prediction` (dict predictions are already NULL there), so
    the apps-score Prediction fix needs one fresh export with this script.
    """
    if not skip_export:
        export_query_to_gcs(sql, project, bucket, gcs_prefix, location,
                            calc_feature_is_json_type)

    tmp_root = Path(local_dir) if local_dir else Path(tempfile.mkdtemp(prefix="ml_vec_"))
    try:
        files = download_parquet_from_gcs(bucket, gcs_prefix, tmp_root, project)
        return build_duckdb_table(files, duckdb_path, table_name, run_date, mode,
                                  memory_limit, threads, convert_workers)
    finally:
        if not keep_local_parquet:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # `python Ml_Vector_Pipeline_Modified.py --skip-export` re-uses parquet already in GCS (no BigQuery cost)
    out = run_pipeline(skip_export="--skip-export" in sys.argv)
    print(f"Done -> {out} (table: {TABLE_NAME})")