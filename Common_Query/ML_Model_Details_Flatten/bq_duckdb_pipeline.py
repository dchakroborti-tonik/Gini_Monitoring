#!/usr/bin/env python3
"""
BigQuery -> GCS parquet -> local DuckDB -> flat table (DuckDB table and/or BigQuery table)

Source : prj-prod-dataplatform.audit_balance.ml_model_run_details   (stays untouched - this COPIES it)
Bucket : gs://prod-asia-southeast1-tonik-report-analytics/DC/Parquet_dump/ml_model_run_details/

Stages (run individually or all together):
    export    BigQuery extract job -> parquet shards in the bucket          (extract jobs are free)
    download  bucket -> ./ml_model_run_details_dump/  (parallel, resumable)
    load      parquet shards -> DuckDB file ./ml_model_run_details.duckdb, table ml_model_run_details
    cleanup   DELETE the exported parquet files from the bucket - only after every file is verified
              (size + crc32c) against the local copy AND the DuckDB row count matches the local files
    flatten   DuckDB raw table -> flat table, written to --sink duckdb | bigquery | both
              (+ a <flat_table>_feature_map table: which feature columns belong to which combination)
    select    pull one model/version/trench/OS combination with ONLY its associated feature columns
    all       export + download + load + cleanup + flatten      (--keep-gcs skips the cleanup)

Examples
    python bq_duckdb_pipeline.py export
    python bq_duckdb_pipeline.py download
    python bq_duckdb_pipeline.py load --expect-rows 9850856
    python bq_duckdb_pipeline.py cleanup --cleanup-dry-run      # shows what would be deleted
    python bq_duckdb_pipeline.py cleanup
    python bq_duckdb_pipeline.py flatten --sink duckdb
    python bq_duckdb_pipeline.py flatten --sink bigquery --dest-table prj-prod-dataplatform.audit_balance.ml_model_run_details_flat --write-disposition WRITE_TRUNCATE
    python bq_duckdb_pipeline.py flatten --sink both  --dest-table ... --start-date 2026-11-01
    python bq_duckdb_pipeline.py select --model-display-name apps_score_cash --model-version-id v1 \
        --trench-category "Trench 1" --device-os android --select-output apps_v1_t1_android.parquet

The flatten logic lives in flatten_ml_model_run_details.py - keep both files in the same folder.

Costs
    * BigQuery extract job, local DuckDB work, BigQuery load job : no query/processing charge.
    * Not zero: the parquet files sit in the bucket (GCS storage, cents/month for a few GB) until the
      cleanup stage deletes them, and the download is billed as GCS network egress (cents per GB).
    * Deleted objects can keep costing if the bucket has object versioning or soft delete (default 7 days
      on newer buckets) - the cleanup stage prints a warning when it detects either.

Auth: Application Default Credentials (gcloud auth application-default login).
Deps: pip install google-cloud-bigquery google-cloud-storage duckdb pyarrow
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa

import flatten_ml_model_run_details as fl

log = logging.getLogger("bq_duckdb_pipeline")

PROJECT_ID = "prj-prod-dataplatform"
SOURCE_TABLE = "prj-prod-dataplatform.audit_balance.ml_model_run_details"
BUCKET = "prod-asia-southeast1-tonik-report-analytics"
GCS_PREFIX = "DC/Parquet_dump/ml_model_run_details"

LOCAL_DUMP_DIR = Path("ml_model_run_details_dump")      # relative to the working directory
DUCKDB_FILE = Path("ml_model_run_details.duckdb")
RAW_TABLE = "ml_model_run_details"
FLAT_TABLE = "ml_model_run_details_flat"
FLAT_PARQUET = Path("ml_model_run_details_flat.parquet")
DEFAULT_DEST_TABLE = "prj-prod-dataplatform.audit_balance.ml_model_run_details_flat"

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise SystemExit(f"invalid table name: {name!r}")
    return name


def _gb(n: float) -> str:
    return f"{n / 1e9:,.2f} GB"


def _list_shards(gcs, bucket: str, prefix: str) -> list:
    """Parquet files directly inside gs://bucket/prefix/ (not in sub-folders), sorted by name."""
    start = len(prefix) + 1
    blobs = [b for b in gcs.list_blobs(bucket, prefix=prefix + "/")
             if b.name.endswith(".parquet") and "/" not in b.name[start:]]
    return sorted(blobs, key=lambda b: b.name)


# ---------------------------------------------------------------------------------------------
# 1. export : BigQuery -> GCS parquet
# ---------------------------------------------------------------------------------------------
def stage_export(args) -> None:
    from google.cloud import bigquery, storage

    prefix = args.gcs_prefix.strip("/")
    bq = bigquery.Client(project=args.project)
    table = bq.get_table(args.source_table)
    log.info("source: %s | %s rows | %s logical | location=%s | partitioning=%s",
             args.source_table, f"{table.num_rows:,}", _gb(table.num_bytes or 0), table.location,
             getattr(table.time_partitioning, "type_", None) or getattr(table.range_partitioning, "field", None))

    gcs = storage.Client(project=args.project)
    bucket = gcs.bucket(args.bucket)
    try:
        bucket.reload()
        log.info("bucket: %s | location=%s | class=%s", args.bucket, bucket.location, bucket.storage_class)
        if bucket.location.upper() != (table.location or "").upper():
            log.warning("dataset location (%s) != bucket location (%s). BigQuery only exports to a bucket in the "
                        "same region (or inside the dataset's multi-region); the job will fail otherwise.",
                        table.location, bucket.location)
    except Exception as exc:                                    # missing storage.buckets.get permission etc.
        log.warning("could not read bucket metadata (%s) - continuing", exc)

    existing = list(gcs.list_blobs(args.bucket, prefix=prefix + "/", max_results=1))
    if existing:
        raise SystemExit(
            f"gs://{args.bucket}/{prefix}/ already contains files (e.g. {existing[0].name}).\n"
            "Refusing to export into a non-empty folder - leftover shards from an older export would be "
            "downloaded together with the new ones and duplicate rows.\n"
            "Use --gcs-prefix with a new folder name, or delete the old folder yourself first.")

    destination = f"gs://{args.bucket}/{prefix}/part-*.parquet"
    cfg = bigquery.ExtractJobConfig(
        destination_format=bigquery.DestinationFormat.PARQUET,
        compression=bigquery.Compression.ZSTD,                  # smaller files -> smaller download
    )
    log.info("extract -> %s", destination)
    t0 = time.time()
    job = bq.extract_table(table, destination, job_config=cfg, location=table.location)
    job.result()
    blobs = _list_shards(gcs, args.bucket, prefix)
    log.info("export done in %.0fs: %d files, %s", time.time() - t0, len(blobs), _gb(sum(b.size for b in blobs)))
    log.info("(BigQuery's extract is a point-in-time copy of %s rows; the original table is unchanged)",
             f"{table.num_rows:,}")
    Path(args.manifest).write_text(json.dumps({
        "source_table": args.source_table,
        "source_rows_at_export_start": table.num_rows,
        "bucket": args.bucket, "prefix": prefix,
        "exported_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "files": [{"name": b.name, "size": b.size, "crc32c": b.crc32c} for b in blobs],
    }, indent=2))
    log.info("manifest written: %s", Path(args.manifest).resolve())


# ---------------------------------------------------------------------------------------------
# 2. download : GCS -> local folder
# ---------------------------------------------------------------------------------------------
def stage_download(args) -> None:
    from google.cloud import storage

    prefix = args.gcs_prefix.strip("/")
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    gcs = storage.Client(project=args.project)
    blobs = _list_shards(gcs, args.bucket, prefix)
    if not blobs:
        raise SystemExit(f"no parquet files under gs://{args.bucket}/{prefix}/ - run the export stage first")
    total = sum(b.size for b in blobs)
    free = shutil.disk_usage(local_dir).free
    log.info("%d files, %s to download; free disk %s", len(blobs), _gb(total), _gb(free))
    if free < 3 * total:
        log.warning("less than 3x the download size is free - the DuckDB file needs roughly the same space again")
    log.info("note: GCS network egress for this download is billed to the bucket's project (cents per GB)")

    def fetch(blob):
        target = local_dir / Path(blob.name).name
        if target.exists() and target.stat().st_size == blob.size:
            return target.name, blob.size, True                  # already downloaded -> resume
        tmp = target.with_suffix(".part")
        blob.download_to_filename(str(tmp))
        if tmp.stat().st_size != blob.size:
            raise IOError(f"size mismatch for {blob.name}")
        os.replace(tmp, target)
        return target.name, blob.size, False

    done = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(fetch, b) for b in blobs]):
            _, _, was_skipped = fut.result()
            done += 1
            skipped += was_skipped
            if done % 10 == 0 or done == len(blobs):
                log.info("downloaded %d/%d files (%d already present)", done, len(blobs), skipped)
    log.info("download complete -> %s", local_dir.resolve())


# ---------------------------------------------------------------------------------------------
# 3. load : local parquet -> DuckDB table
# ---------------------------------------------------------------------------------------------
def _connect(path: Path, read_only: bool = False):
    import duckdb
    con = duckdb.connect(str(path), read_only=read_only)
    con.execute("SET TimeZone='UTC'")                           # keeps timestamp tz as plain 'UTC' for pyarrow
    return con


def _scalar(con, sql: str):
    """First value of the first row; consumes the whole result (DuckDB 2.x blocks other work on the connection
    while a result from execute() is unfinished)."""
    return con.execute(sql).fetchall()[0][0]


def stage_load(args) -> None:
    local_dir = Path(args.local_dir)
    files = sorted(local_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {local_dir.resolve()} - run the download stage first")
    raw = _ident(args.raw_table)
    glob = (local_dir / "*.parquet").as_posix().replace("'", "''")
    con = _connect(Path(args.duckdb_file))
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TABLE {raw} AS SELECT * FROM read_parquet('{glob}', union_by_name=true)")
    n = _scalar(con, f"SELECT count(*) FROM {raw}")
    cols = [r[0] for r in con.execute(f"DESCRIBE {raw}").fetchall()]
    missing = [c for c in fl.BASE_COLUMNS if c not in cols]
    log.info("DuckDB table %s: %s rows, %d columns in %.0fs -> %s", raw, f"{n:,}", len(cols),
             time.time() - t0, Path(args.duckdb_file).resolve())
    if missing:
        log.error("expected columns missing from the dump: %s", missing)
    if args.expect_rows and n != args.expect_rows:
        log.warning("row count %s differs from --expect-rows %s", f"{n:,}", f"{args.expect_rows:,}")
    elif args.expect_rows:
        log.info("row count matches --expect-rows")
    con.close()


# ---------------------------------------------------------------------------------------------
# 4. cleanup : delete the bucket copy, but only once DuckDB provably holds everything
# ---------------------------------------------------------------------------------------------
def _local_crc32c(path: Path) -> str | None:
    """crc32c of a local file in the same base64 form GCS reports; None if the library is unavailable."""
    try:
        import google_crc32c
    except ImportError:
        return None
    h = google_crc32c.Checksum()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


def stage_cleanup(args) -> None:
    """Delete the exported parquet files from the bucket.

    Nothing is deleted unless ALL of these hold:
      1. every parquet file in the bucket folder exists locally with identical size and crc32c
      2. the DuckDB raw table exists and its row count equals the row count of those local files
    Only the .parquet files directly inside the export folder are touched.
    """
    from google.cloud import storage

    prefix = args.gcs_prefix.strip("/")
    if not prefix or "/" not in prefix:
        raise SystemExit(f"refusing to clean up with gcs prefix {prefix!r} - too broad (expected e.g. DC/Parquet_dump/x)")

    gcs = storage.Client(project=args.project)
    blobs = _list_shards(gcs, args.bucket, prefix)
    if not blobs:
        log.info("nothing to delete: no parquet files under gs://%s/%s/", args.bucket, prefix)
        return

    local_dir = Path(args.local_dir)
    problems, verified = [], []
    for b in blobs:
        lf = local_dir / Path(b.name).name
        if not lf.exists():
            problems.append(f"not downloaded: {lf.name}")
        elif lf.stat().st_size != b.size:
            problems.append(f"size differs: {lf.name} (bucket {b.size:,} vs local {lf.stat().st_size:,} bytes)")
        else:
            local_crc = _local_crc32c(lf) if b.crc32c else None
            if local_crc is not None and local_crc != b.crc32c:
                problems.append(f"checksum differs: {lf.name}")
            else:
                verified.append(lf)
    extra = sorted({p.name for p in local_dir.glob("*.parquet")} - {Path(b.name).name for b in blobs})
    if extra:
        problems.append(f"local folder has parquet files that are not in the bucket (stale dump?): {extra[:5]} "
                        "- the DuckDB table was probably loaded from them too; remove them and re-run `load`")

    raw = _ident(args.raw_table)
    rows_table = rows_files = None
    if not problems:
        try:
            con = _connect(Path(args.duckdb_file), read_only=True)
            try:
                rows_table = _scalar(con, f"SELECT count(*) FROM {raw}")
                files_sql = ", ".join("'" + p.as_posix().replace("'", "''") + "'" for p in verified)
                rows_files = _scalar(con, f"SELECT count(*) FROM read_parquet([{files_sql}])")
            finally:
                con.close()
        except Exception as exc:
            problems.append(f"cannot verify DuckDB table {raw!r} in {args.duckdb_file}: {exc} - run the load stage")
        else:
            if rows_table != rows_files:
                problems.append(f"DuckDB table has {rows_table:,} rows but the downloaded files hold {rows_files:,}")

    if problems:
        raise SystemExit("NOT deleting anything from the bucket:\n  - " + "\n  - ".join(problems[:15]))

    total = sum(b.size for b in blobs)
    log.info("verified: %d files (%s) match the local copies byte-for-byte; DuckDB %s holds %s rows = rows in files",
             len(blobs), _gb(total), raw, f"{rows_table:,}")
    manifest = Path(args.manifest)
    if manifest.exists():
        try:
            src_rows = json.loads(manifest.read_text()).get("source_rows_at_export_start")
            log.info("BigQuery source had %s rows when the export started (the table keeps growing, so the "
                     "dump may hold more)", f"{src_rows:,}")
        except Exception:
            pass

    try:                                                        # does deleting actually stop the billing?
        bucket = gcs.bucket(args.bucket)
        bucket.reload()
        if getattr(bucket, "versioning_enabled", False):
            log.warning("bucket has OBJECT VERSIONING enabled - deleted files remain as non-current versions and keep "
                        "costing until a lifecycle rule removes them")
        sd = getattr(bucket, "soft_delete_policy", None)
        if sd and getattr(sd, "retention_duration_seconds", 0):
            log.warning("bucket has SOFT DELETE (%d days) - deleted files stay billable for that retention period",
                        sd.retention_duration_seconds // 86400)
    except Exception:
        pass

    if args.cleanup_dry_run:
        log.info("DRY RUN - would delete %d files (%s) from gs://%s/%s/:", len(blobs), _gb(total), args.bucket, prefix)
        for b in blobs[:10]:
            log.info("   %s", b.name)
        return

    failed = []
    for b in blobs:
        try:
            b.delete()
        except Exception as exc:
            failed.append(f"{b.name}: {exc}")
    if failed:
        raise SystemExit(f"deleted {len(blobs) - len(failed)}/{len(blobs)} files; failed:\n  - " + "\n  - ".join(failed[:10]))
    log.info("deleted %d parquet files (%s) from gs://%s/%s/", len(blobs), _gb(total), args.bucket, prefix)


# ---------------------------------------------------------------------------------------------
# 5. flatten : DuckDB raw table -> flat table (duckdb / bigquery / both)
# ---------------------------------------------------------------------------------------------
def _duckdb_source(args):
    """Stream the raw table out of DuckDB. Returns (con, base_types, row_iterator)."""
    raw = _ident(args.raw_table)
    con = _connect(Path(args.duckdb_file), read_only=True)
    cols = ", ".join(f'"{c}"' for c in fl.BASE_COLUMNS)
    where = ["TRUE"]
    if args.start_date:
        where.append(f"CAST(start_time AS DATE) >= DATE '{_date(args.start_date)}'")
    if args.end_date:
        where.append(f"CAST(start_time AS DATE) <= DATE '{_date(args.end_date)}'")
    sql = f"SELECT {cols} FROM {raw} WHERE " + " AND ".join(where)
    if args.dedupe_latest:
        sql += (" QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId, modelDisplayName, "
                "modelVersionId ORDER BY start_time DESC, end_time DESC) = 1")
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    log.info("DuckDB query: %s", sql)

    rel = con.sql(sql)                                          # lazy relation -> streams, no full copy in memory
    if hasattr(rel, "to_arrow_reader"):                         # DuckDB >= 1.4
        reader = rel.to_arrow_reader(50_000)
    else:                                                       # older DuckDB
        reader = rel.fetch_arrow_reader(50_000) if hasattr(rel, "fetch_arrow_reader") \
            else con.execute(sql).fetch_record_batch(50_000)
    base_types = {}
    for f in reader.schema:
        t = f.type
        if pa.types.is_large_string(t) or pa.types.is_dictionary(t):
            t = pa.string()
        base_types[f.name] = t

    def rows():
        for batch in reader:
            yield from zip(*(batch.column(i).to_pylist() for i in range(batch.num_columns)))

    return con, base_types, rows()


def _date(s: str) -> str:
    import datetime as dt
    return dt.date.fromisoformat(s).isoformat()                 # validates -> safe to inline in SQL


def stage_flatten(args) -> None:
    if args.sink in ("bigquery", "both") and not args.dest_table:
        raise SystemExit("--dest-table is required for --sink bigquery/both")

    args.output = str(args.flat_parquet)                        # flatten module writes this parquet
    con, base_types, rows = _duckdb_source(args)
    try:
        result = fl.process(base_types, rows, args)
    finally:
        con.close()                                             # release the file before re-opening for write

    if args.sink in ("duckdb", "both"):
        flat = _ident(args.flat_table)
        out = Path(args.flat_parquet).as_posix().replace("'", "''")
        wcon = _connect(Path(args.duckdb_file))
        mp = Path(result["map_parquet"]).as_posix().replace("'", "''")
        wcon.execute(f"CREATE OR REPLACE TABLE {flat} AS SELECT * FROM read_parquet('{out}')")
        wcon.execute(f"CREATE OR REPLACE TABLE {flat}{fl.MAP_SUFFIX} AS SELECT * FROM read_parquet('{mp}')")
        n = _scalar(wcon, f"SELECT count(*) FROM {flat}")
        nc = len(wcon.execute(f"DESCRIBE {flat}").fetchall())
        nm = _scalar(wcon, f"SELECT count(*) FROM {flat}{fl.MAP_SUFFIX}")
        wcon.close()
        log.info("DuckDB table %s created: %s rows x %d columns in %s", flat, f"{n:,}", nc,
                 Path(args.duckdb_file).resolve())
        log.info("DuckDB table %s%s created: %d rows (combination x feature column)", flat, fl.MAP_SUFFIX, nm)

    if args.sink in ("bigquery", "both"):
        fl.load_to_bigquery(args, result["map_parquet"])        # flat table + <dest>_feature_map

    if args.sink == "bigquery" and not args.keep_parquet:
        log.info("local parquet kept at %s (delete it if you do not need it)", args.flat_parquet)
    log.info("done: %s rows", f"{result['rows']:,}")


# ---------------------------------------------------------------------------------------------
# 6. select : one combination -> only the feature columns associated with it
# ---------------------------------------------------------------------------------------------
def _qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(v) -> str:
    return "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"


def stage_select(args) -> None:
    flat = _ident(args.flat_table)
    map_table = flat + fl.MAP_SUFFIX
    filters = {"modelDisplayName": args.model_display_name, "modelVersionId": args.model_version_id,
               "trenchCategory": args.trench_category, "deviceOs": args.device_os}
    given = {k: v for k, v in filters.items() if v is not None}
    if not given:
        raise SystemExit("give at least one of --model-display-name --model-version-id --trench-category --device-os")
    where = " AND ".join(f"{_qi(k)} IS NOT DISTINCT FROM {_lit(v)}" for k, v in given.items())

    con = _connect(Path(args.duckdb_file), read_only=True)
    try:
        flat_cols = [r[0] for r in con.execute(f"DESCRIBE {flat}").fetchall()]
        n_rows = _scalar(con, f"SELECT count(*) FROM {flat} WHERE {where}")
        if n_rows == 0:
            log.error("no rows match %s. Combinations that exist:", given)
            con.sql(f"SELECT feature_set_id, {', '.join(map(_qi, fl.COMBO_COLUMNS))}, count(*) AS n_rows "
                    f"FROM {flat} GROUP BY ALL ORDER BY 2, 3, 4, 5").show(max_rows=60)
            raise SystemExit(1)

        assoc = {r[0] for r in con.execute(
            f"SELECT DISTINCT column_name FROM {map_table} WHERE {where}").fetchall()}
        base_keep = (set(fl.BASE_COLUMNS) - {"calcFeature"}) | {fl.STATUS_COLUMN, fl.ID_COLUMN}
        if args.all_columns:
            cols = flat_cols
        else:
            cols = [c for c in flat_cols if c in base_keep or c in assoc]
        n_feat = sum(c in assoc for c in cols)
        log.info("%s rows match %s | %d associated feature columns selected%s", f"{n_rows:,}", given, len(assoc),
                 " (all other columns kept, holding -1 / Not_Applicable)" if args.all_columns else
                 f" (the other {len(flat_cols) - len(cols)} columns are left out)")
        if not assoc:
            log.warning("this combination has no associated feature columns (every calcFeature was empty)")

        sql = f"SELECT {', '.join(map(_qi, cols))} FROM {flat} WHERE {where}"
        if args.limit:
            sql += f" LIMIT {int(args.limit)}"
        out = args.select_output
        if out:
            fmt = "(FORMAT CSV, HEADER)" if out.lower().endswith(".csv") else "(FORMAT PARQUET, COMPRESSION ZSTD)"
            con.execute(f"COPY ({sql}) TO '{out.replace(chr(39), chr(39) * 2)}' {fmt}")
            log.info("written %s (%d columns, %d of them feature columns)", Path(out).resolve(), len(cols), n_feat)
        else:
            con.sql(sql + ("" if args.limit else " LIMIT 20")).show(max_width=200, max_rows=20)
            log.info("preview only - add --select-output file.parquet|.csv to save the full result")
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["export", "download", "load", "cleanup", "flatten", "select", "all"])
    p.add_argument("--project", default=PROJECT_ID)
    p.add_argument("--source-table", default=SOURCE_TABLE)
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--gcs-prefix", default=GCS_PREFIX)
    p.add_argument("--local-dir", default=str(LOCAL_DUMP_DIR))
    p.add_argument("--workers", type=int, default=8, help="parallel downloads")
    p.add_argument("--duckdb-file", default=str(DUCKDB_FILE))
    p.add_argument("--raw-table", default=RAW_TABLE)
    p.add_argument("--expect-rows", type=int, default=0, help="sanity check for the load stage, e.g. 9850856")
    p.add_argument("--manifest", default="export_manifest.json", help="written by export, read by cleanup")
    # cleanup stage
    p.add_argument("--keep-gcs", action="store_true", help="`all`: do not delete the parquet files from the bucket")
    p.add_argument("--cleanup-dry-run", action="store_true", help="cleanup: verify and list, delete nothing")
    # select stage
    p.add_argument("--model-display-name")
    p.add_argument("--model-version-id")
    p.add_argument("--trench-category")
    p.add_argument("--device-os")
    p.add_argument("--select-output", help="select: write the result to .parquet or .csv (default: preview 20 rows)")
    p.add_argument("--all-columns", action="store_true",
                   help="select: keep every column (non-associated ones hold -1 / Not_Applicable)")
    # flatten stage
    p.add_argument("--sink", choices=["duckdb", "bigquery", "both"], default="duckdb")
    p.add_argument("--flat-table", default=FLAT_TABLE, help="DuckDB table name for the flat table")
    p.add_argument("--flat-parquet", default=str(FLAT_PARQUET), help="intermediate/output parquet")
    p.add_argument("--keep-parquet", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dest-table", default=None,
                   help=f"BigQuery table for --sink bigquery/both, e.g. {DEFAULT_DEST_TABLE}")
    p.add_argument("--write-disposition", default="WRITE_EMPTY",
                   choices=["WRITE_EMPTY", "WRITE_TRUNCATE", "WRITE_APPEND"])
    p.add_argument("--start-date", help="YYYY-MM-DD, filters on CAST(start_time AS DATE)")
    p.add_argument("--end-date", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--limit", type=int)
    p.add_argument("--dedupe-latest", action="store_true")
    p.add_argument("--keep-raw-calcfeature", action="store_true")
    p.add_argument("--fill-null-values", action="store_true")
    p.add_argument("--expand-prediction", action="store_true")
    p.add_argument("--force-numeric", nargs="*", default=[], metavar="COL")
    p.add_argument("--force-categorical", nargs="*", default=[], metavar="COL")
    p.add_argument("--no-inline-feature-list", dest="inline_feature_list", action="store_false",
                   help="do not add the `associated_features` column (feature_set_id + map table remain)")
    p.add_argument("--chunk-rows", type=int, default=fl.DEFAULT_CHUNK_ROWS)
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    stages = {"export": stage_export, "download": stage_download, "load": stage_load,
              "cleanup": stage_cleanup, "flatten": stage_flatten, "select": stage_select}
    if args.stage == "all":
        order = ["export", "download", "load"] + ([] if args.keep_gcs else ["cleanup"]) + ["flatten"]
    else:
        order = [args.stage]
    for name in order:
        log.info("==== stage: %s ====", name)
        stages[name](args)


if __name__ == "__main__":
    main()
