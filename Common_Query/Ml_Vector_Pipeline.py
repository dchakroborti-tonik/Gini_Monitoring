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

Install:  pip install duckdb google-cloud-bigquery google-cloud-storage
Auth   :  gcloud auth application-default login   (or a service account)
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
  safe_cast(`prediction` as numeric) prediction,
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
  safe_cast(`prediction` as numeric) prediction,
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
    workers: int = 8,
) -> list[Path]:
    from google.cloud import storage

    prefix = prefix.strip("/") + "/"
    client = storage.Client(project=project)
    blobs = [b for b in client.list_blobs(bucket, prefix=prefix) if b.name.endswith(".parquet")]
    if not blobs:
        raise FileNotFoundError(f"No parquet files under gs://{bucket}/{prefix}")

    local_dir.mkdir(parents=True, exist_ok=True)

    def _dl(blob) -> Path:
        dest = local_dir / Path(blob.name).name
        blob.download_to_filename(str(dest))
        return dest

    with ThreadPoolExecutor(max_workers=workers) as ex:
        files = list(ex.map(_dl, blobs))
    log.info("Downloaded %d parquet file(s) to %s", len(files), local_dir)
    return sorted(files)


# --------------------------------------------------------------------------- #
# Step 3: DuckDB processing
# --------------------------------------------------------------------------- #
# Numeric text: 12, -3.5, .5, 1e-5  (leading-zero strings like '007' stay String: ids/zip codes)
_NUM_RE = r'^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$'
_LEADING_ZERO_RE = r'^\s*[-+]?0\d'
# ISO-like dates / timestamps: 2024-01-31, 2024/01/31, 2024-01-31T10:20:30.123Z, 2024-01-31 10:20
_DATE_RE = (r'^\s*\d{4}[-/]\d{2}[-/]\d{2}'
            r'([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?\s*(Z|[+-]\d{2}:?\d{2})?)?\s*$')

# CTE chain: normalise calcFeature into valid JSON text (cf_json) or NULL.
_SRC_CTES = r"""
src AS (
  SELECT
    TRY_CAST(customerId AS BIGINT)            AS customerId,
    CAST(digitalLoanAccountId AS VARCHAR)     AS digitalLoanAccountId,
    CAST(modelDisplayName AS VARCHAR)         AS modelDisplayName,
    CAST(modelVersionId AS VARCHAR)           AS modelVersionId,
    CAST(trenchCategory AS VARCHAR)           AS trenchCategory,
    CAST(deviceOs AS VARCHAR)                 AS deviceOs,
    CAST(new_loan_type AS VARCHAR)            AS new_loan_type,
    CAST(application_date AS DATE)            AS application_date,
    CAST(Data_selection AS VARCHAR)           AS Data_selection,
    CAST(CAST(prediction AS DOUBLE) AS VARCHAR) AS prediction,
    trim(CAST(calcFeature AS VARCHAR))        AS cf_raw
  FROM read_parquet('__PATH__')
),
src_json AS (
  SELECT *,
    CASE WHEN json_valid(cf_raw) THEN
           CASE WHEN json_type(cf_raw) = 'VARCHAR'            -- double-encoded JSON string
                THEN CASE WHEN json_valid(json_extract_string(cf_raw, '$'))
                          THEN json_extract_string(cf_raw, '$') END
                ELSE cf_raw END
    END AS cf_json
  FROM src
)
"""

_INSERT_SQL = r"""
INSERT INTO "__TABLE__"
WITH __SRC_CTES__
-- (a) Prediction row
SELECT customerId, digitalLoanAccountId, modelDisplayName, modelVersionId, trenchCategory,
       deviceOs, new_loan_type, application_date, Data_selection,
       'Prediction' AS Features,
       prediction   AS "Values",
       '__DTYPE_FLOAT__' AS Data_type,
       FALSE AS is_delete, TRUE AS is_active, DATE '__RUN_DATE__' AS run_date
FROM src
WHERE prediction IS NOT NULL

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
  FROM (SELECT * FROM src_json WHERE cf_json IS NOT NULL) s,
       json_tree(s.cf_json) jt
  WHERE jt.type NOT IN ('OBJECT', 'ARRAY')   -- leaves only (incl. JSON null)
    AND jt.fullkey <> '$'
) f
"""

_INVALID_COUNT_SQL = r"""
WITH __SRC_CTES__
SELECT count(*) FROM src_json
WHERE cf_raw IS NOT NULL AND cf_raw <> '' AND cf_json IS NULL
"""

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _fill(template: str, path: str, table: str, run_date: str) -> str:
    sql = template.replace("__SRC_CTES__", _SRC_CTES)
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
    }
    for k, v in repl.items():
        sql = sql.replace(k, v)
    return sql


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


def build_duckdb_table(
    parquet_files: Iterable[Path],
    duckdb_path: str = DUCKDB_PATH,
    table_name: str = TABLE_NAME,
    run_date: Optional[date] = None,
    mode: str = "replace",
    memory_limit: Optional[str] = None,
    threads: Optional[int] = None,
) -> str:
    """Flatten parquet files into the vector table inside a local DuckDB file.

    mode="replace": table is rebuilt in a staging table and swapped in only if
                    every file succeeded (a failed run never leaves a half table).
    mode="append" : rows are added to the existing table (created if missing).
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

    con = duckdb.connect(duckdb_path)
    try:
        if memory_limit:
            con.execute(f"SET memory_limit='{memory_limit}'")
        if threads:
            con.execute(f"SET threads={int(threads)}")
        spill = Path(duckdb_path).resolve().parent / ".duckdb_spill"
        con.execute(f"SET temp_directory='{spill}'")

        con.execute(f'DROP TABLE IF EXISTS "{staging}"')
        con.execute(_create_table_sql(staging))

        total_invalid = 0
        for i, f in enumerate(files, 1):
            path = str(f)
            bad = con.execute(_fill(_INVALID_COUNT_SQL, path, staging, run_date_s)).fetchone()[0]
            if bad:
                log.warning("%s: %d row(s) have calcFeature that is not valid JSON "
                            "(Prediction row still loaded, no feature rows)", f.name, bad)
            total_invalid += bad
            con.execute(_fill(_INSERT_SQL, path, staging, run_date_s))
            log.info("[%d/%d] processed %s", i, len(files), f.name)

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
        log.info("Table %s rows=%d | this run by Data_type=%s | invalid calcFeature rows=%d",
                 table_name, n, dist, total_invalid)
        con.execute("CHECKPOINT")
    finally:
        con.close()
        shutil.rmtree(Path(duckdb_path).resolve().parent / ".duckdb_spill", ignore_errors=True)
    return duckdb_path


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
) -> str:
    """BigQuery SQL -> parquet in GCS -> DuckDB table `table_name` in `duckdb_path`.

    skip_export=True re-uses parquet already sitting in GCS (no BigQuery cost),
    handy when only the DuckDB logic changes.
    """
    if not skip_export:
        export_query_to_gcs(sql, project, bucket, gcs_prefix, location,
                            calc_feature_is_json_type)

    tmp_root = Path(local_dir) if local_dir else Path(tempfile.mkdtemp(prefix="ml_vec_"))
    try:
        files = download_parquet_from_gcs(bucket, gcs_prefix, tmp_root, project)
        return build_duckdb_table(files, duckdb_path, table_name, run_date, mode,
                                  memory_limit, threads)
    finally:
        if not keep_local_parquet:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run_pipeline()
    print(f"Done -> {out} (table: {TABLE_NAME})")