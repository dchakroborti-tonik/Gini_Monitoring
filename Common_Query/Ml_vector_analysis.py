"""
Checks + PSI / drift / discrimination metrics on the DuckDB vector table
(dd_ml_models_vector_details).  Everything heavy runs inside DuckDB; only
small result tables come back as pandas DataFrames.

    import ml_vector_analysis as a
    con = a.connect("dd_ml_models_vector_details.duckdb")
    checks = a.sanity_checks(con)                       # dict of DataFrames
    psi = a.psi_numeric(con, "Data_selection = 'Train'", "Data_selection = 'Prod'")
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

TABLE = "dd_ml_models_vector_details"
EPS = 1e-4  # floor for empty bins so ln() is defined


def connect(path: str = "dd_ml_models_vector_details.duckdb", read_only: bool = True):
    return duckdb.connect(path, read_only=read_only)


# --------------------------------------------------------------------------- #
# 1. Sanity checks
# --------------------------------------------------------------------------- #
def sanity_checks(con, table: str = TABLE) -> dict[str, pd.DataFrame]:
    t = f'"{table}"'
    q = {
        "schema": f"DESCRIBE {t}",
        "volume_by_group": f"""
            SELECT Data_selection, modelDisplayName, modelVersionId,
                   count(DISTINCT digitalLoanAccountId) AS loans, count(*) AS rows,
                   min(application_date) AS min_app_date, max(application_date) AS max_app_date,
                   count(*) FILTER (WHERE application_date IS NULL) AS rows_no_app_date
            FROM {t} GROUP BY ALL ORDER BY 1, 2, 3""",
        "features_per_loan": f"""
            SELECT Data_selection, modelVersionId, min(n) AS min_feats, median(n) AS median_feats,
                   max(n) AS max_feats
            FROM (SELECT Data_selection, modelVersionId, digitalLoanAccountId, count(*) AS n
                  FROM {t} GROUP BY ALL) GROUP BY ALL ORDER BY 1, 2""",
        "duplicate_keys": f"""
            SELECT count(*) AS duplicated_keys FROM (
              SELECT 1 FROM {t}
              GROUP BY digitalLoanAccountId, modelVersionId, Data_selection, Features
              HAVING count(*) > 1)""",
        "dtype_summary": f"""
            SELECT Data_type, count(*) AS rows, count(DISTINCT Features) AS features,
                   count(*) FILTER (WHERE "Values" IS NULL) AS null_values
            FROM {t} GROUP BY 1 ORDER BY 2 DESC""",
        "mixed_dtype_features": f"""
            SELECT Features, list(DISTINCT Data_type) AS types, count(*) AS rows
            FROM {t} WHERE "Values" IS NOT NULL AND Features <> 'Prediction'
            GROUP BY Features HAVING count(DISTINCT Data_type) > 1 ORDER BY rows DESC""",
        "prediction_summary": f"""
            SELECT Data_selection, count(*) AS n, min(TRY_CAST("Values" AS DOUBLE)) AS min,
                   quantile_cont(TRY_CAST("Values" AS DOUBLE), 0.5) AS median,
                   max(TRY_CAST("Values" AS DOUBLE)) AS max
            FROM {t} WHERE Features = 'Prediction' GROUP BY 1""",
    }
    return {k: con.execute(v).df() for k, v in q.items()}


# --------------------------------------------------------------------------- #
# 2. Feature profile (null rate, distribution)
# --------------------------------------------------------------------------- #
def feature_profile(con, where: str = "TRUE", table: str = TABLE) -> pd.DataFrame:
    return con.execute(f"""
        WITH x AS (SELECT Features, TRY_CAST("Values" AS DOUBLE) AS v, "Values" AS raw
                   FROM "{table}" WHERE is_active AND NOT is_delete AND ({where}))
        SELECT Features, count(*) AS n,
               round(avg((raw IS NULL)::INT), 4) AS null_rate,
               count(DISTINCT raw) AS distinct_values,
               avg(v) FILTER (WHERE isfinite(v)) AS mean, stddev(v) FILTER (WHERE isfinite(v)) AS std,
               min(v) FILTER (WHERE isfinite(v)) AS min,
               quantile_cont(v, 0.5) FILTER (WHERE isfinite(v)) AS p50,
               max(v) FILTER (WHERE isfinite(v)) AS max
        FROM x GROUP BY Features ORDER BY Features""").df()


# --------------------------------------------------------------------------- #
# 3. PSI
# --------------------------------------------------------------------------- #
# A feature's type = its most common Data_type among NON-null values. (Null leaves are
# tagged 'String', so row-level Data_type alone would split a numeric feature's nulls away.)
_FEATURE_TYPES = """
ft AS (SELECT Features, mode(Data_type) FILTER (WHERE "Values" IS NOT NULL) AS dtype
       FROM "{table}" WHERE is_active AND NOT is_delete GROUP BY Features)"""


def _rating(p: float) -> str:
    return "stable" if p < 0.1 else "moderate shift" if p < 0.25 else "significant shift"


def psi_numeric(con, baseline_where: str, compare_where: str, n_bins: int = 10,
                min_baseline_rows: int = 100, table: str = TABLE) -> pd.DataFrame:
    """PSI for every Float64 feature (and Prediction).

    Bin edges = baseline quantiles; nulls / NaN / Inf / unparseable form their own
    bin, so a change in missing rate is captured.  *_where are SQL predicates on the
    table's columns, e.g. "Data_selection = 'Train'" or
    "application_date BETWEEN '2026-01-01' AND '2026-01-31' AND Data_selection='Prod'".
    """
    qs = [round(i / n_bins, 6) for i in range(1, n_bins)]
    sql = f"""
    WITH {_FEATURE_TYPES.format(table=table)},
    vals AS (
      SELECT t.Features,
             CASE WHEN ({baseline_where}) THEN 'E' WHEN ({compare_where}) THEN 'A' END AS grp,
             TRY_CAST(t."Values" AS DOUBLE) AS v
      FROM "{table}" t JOIN ft ON ft.Features = t.Features
      WHERE t.is_active AND NOT t.is_delete AND ft.dtype = 'Float64'),
    edges AS (
      SELECT Features, list_sort(list_distinct(quantile_cont(v, {qs}))) AS e
      FROM vals WHERE grp = 'E' AND isfinite(v)
      GROUP BY Features HAVING count(*) >= {int(min_baseline_rows)}),
    binned AS (
      SELECT vals.Features, grp,
             CASE WHEN v IS NULL OR NOT isfinite(v) THEN -1
                  ELSE len(list_filter(e, x -> x < v)) END AS bin
      FROM vals JOIN edges USING (Features) WHERE grp IS NOT NULL),
    cnt AS (
      SELECT Features, bin, count(*) FILTER (WHERE grp = 'E') AS e_n,
                            count(*) FILTER (WHERE grp = 'A') AS a_n
      FROM binned GROUP BY ALL),
    tot AS (SELECT Features, sum(e_n) AS e_t, sum(a_n) AS a_t FROM cnt GROUP BY Features),
    pct AS (
      SELECT c.Features, c.e_n, c.a_n, greatest(c.e_n / tot.e_t, {EPS}) AS e_p,
             greatest(c.a_n / tot.a_t, {EPS}) AS a_p
      FROM cnt c JOIN tot USING (Features) WHERE tot.e_t > 0 AND tot.a_t > 0)
    SELECT Features, sum((a_p - e_p) * ln(a_p / e_p)) AS psi,
           sum(e_n)::BIGINT AS baseline_rows, sum(a_n)::BIGINT AS compare_rows
    FROM pct GROUP BY Features ORDER BY psi DESC"""
    df = con.execute(sql).df()
    df["rating"] = df["psi"].map(_rating)
    return df


def psi_categorical(con, baseline_where: str, compare_where: str, max_categories: int = 50,
                    table: str = TABLE) -> pd.DataFrame:
    """PSI for String features with <= max_categories distinct baseline values
    (higher-cardinality text such as ids/free text is skipped).  NULL is its own category;
    categories new in the comparison window get the EPS floor on the baseline side."""
    sql = f"""
    WITH {_FEATURE_TYPES.format(table=table)},
    vals AS (
      SELECT t.Features,
             CASE WHEN ({baseline_where}) THEN 'E' WHEN ({compare_where}) THEN 'A' END AS grp,
             coalesce(t."Values", '__MISSING__') AS cat
      FROM "{table}" t JOIN ft ON ft.Features = t.Features
      WHERE t.is_active AND NOT t.is_delete AND ft.dtype = 'String' AND t.Features <> 'Prediction'),
    ok AS (SELECT Features FROM vals WHERE grp = 'E'
           GROUP BY Features HAVING count(DISTINCT cat) <= {int(max_categories)}),
    cnt AS (
      SELECT Features, cat, count(*) FILTER (WHERE grp = 'E') AS e_n,
                            count(*) FILTER (WHERE grp = 'A') AS a_n
      FROM vals WHERE grp IS NOT NULL AND Features IN (SELECT Features FROM ok) GROUP BY ALL),
    tot AS (SELECT Features, sum(e_n) AS e_t, sum(a_n) AS a_t FROM cnt GROUP BY Features),
    pct AS (
      SELECT c.Features, c.e_n, c.a_n, greatest(c.e_n / tot.e_t, {EPS}) AS e_p,
             greatest(c.a_n / tot.a_t, {EPS}) AS a_p
      FROM cnt c JOIN tot USING (Features) WHERE tot.e_t > 0 AND tot.a_t > 0)
    SELECT Features, sum((a_p - e_p) * ln(a_p / e_p)) AS psi,
           sum(e_n)::BIGINT AS baseline_rows, sum(a_n)::BIGINT AS compare_rows
    FROM pct GROUP BY Features ORDER BY psi DESC"""
    df = con.execute(sql).df()
    df["rating"] = df["psi"].map(_rating)
    return df


def psi_all(con, baseline_where: str, compare_where: str, **kw) -> pd.DataFrame:
    """Numeric + categorical PSI in one table."""
    num = psi_numeric(con, baseline_where, compare_where, **{k: v for k, v in kw.items()
                      if k in ("n_bins", "min_baseline_rows", "table")})
    cat = psi_categorical(con, baseline_where, compare_where, **{k: v for k, v in kw.items()
                          if k in ("max_categories", "table")})
    num["kind"], cat["kind"] = "numeric", "categorical"
    return pd.concat([num, cat], ignore_index=True).sort_values("psi", ascending=False)


def psi_by_month(con, baseline_where: str, compare_extra_where: str = "TRUE",
                 features: list[str] | None = None, **kw) -> pd.DataFrame:
    """PSI of each application month vs the baseline -> wide table (rows=feature, cols=month).
    compare_extra_where narrows the compared population, e.g. "Data_selection = 'Prod'"."""
    months = con.execute(f"""SELECT DISTINCT strftime(date_trunc('month', application_date), '%Y-%m')
                             FROM "{kw.get('table', TABLE)}" WHERE application_date IS NOT NULL
                             AND ({compare_extra_where}) ORDER BY 1""").fetchall()
    out = []
    for (m,) in months:
        cmp_w = (f"({compare_extra_where}) AND "
                 f"strftime(date_trunc('month', application_date), '%Y-%m') = '{m}'")
        r = psi_all(con, baseline_where, cmp_w, **kw)[["Features", "psi"]].assign(month=m)
        out.append(r)
    if not out:
        return pd.DataFrame()
    long = pd.concat(out)
    if features:
        long = long[long["Features"].isin(features)]
    return long.pivot_table(index="Features", columns="month", values="psi").sort_index()


# --------------------------------------------------------------------------- #
# 4. Wide view + score metrics
# --------------------------------------------------------------------------- #
def to_wide(con, features: list[str], where: str = "TRUE", table: str = TABLE) -> pd.DataFrame:
    """One row per (loan, model version, data selection); one column per requested feature.
    Float64 features are cast to DOUBLE, everything else stays text."""
    dtypes = dict(con.execute(f"""
        SELECT Features, mode(Data_type) FILTER (WHERE "Values" IS NOT NULL)
        FROM "{table}" WHERE Features IN ({",".join("'" + f.replace("'", "''") + "'" for f in features)})
        GROUP BY Features""").fetchall())
    cols = []
    for f in features:
        ff = f.replace("'", "''")
        expr = f"""max("Values") FILTER (WHERE Features = '{ff}')"""
        if dtypes.get(f) == "Float64":
            expr = f"TRY_CAST({expr} AS DOUBLE)"
        cols.append(f'{expr} AS "{f.replace(chr(34), "")}"')
    return con.execute(f"""
        SELECT digitalLoanAccountId, customerId, modelVersionId, Data_selection,
               any_value(application_date) AS application_date, {", ".join(cols)}
        FROM "{table}" WHERE is_active AND NOT is_delete AND ({where})
        GROUP BY digitalLoanAccountId, customerId, modelVersionId, Data_selection""").df()


def get_scores(con, where: str = "TRUE", table: str = TABLE) -> pd.DataFrame:
    return to_wide(con, ["Prediction"], where, table)


def auc_gini_ks(y: np.ndarray, score: np.ndarray) -> dict:
    """y: 1 = bad / event, 0 = good.  Rank-based AUC, Gini = 2*AUC-1, KS."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"n": len(y), "bads": n_pos, "auc": np.nan, "gini": np.nan, "ks": np.nan}
    ranks = pd.Series(s).rank(method="average").to_numpy()
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    order = np.argsort(-s, kind="mergesort")
    cum_bad = np.cumsum(y[order]) / n_pos
    cum_good = np.cumsum(1 - y[order]) / n_neg
    return {"n": len(y), "bads": n_pos, "auc": auc, "gini": 2 * auc - 1,
            "ks": float(np.max(np.abs(cum_bad - cum_good)))}


# Example (labels come from wherever you keep outcomes, e.g. BigQuery -> pandas):
#   scores = get_scores(con, "Data_selection = 'Prod'")
#   labels = pd.read_gbq("SELECT digitalLoanAccountId, is_bad FROM ...", project_id="prj-prod-dataplatform")
#   df = scores.merge(labels, on="digitalLoanAccountId")
#   print(auc_gini_ks(df["is_bad"], df["Prediction"]))