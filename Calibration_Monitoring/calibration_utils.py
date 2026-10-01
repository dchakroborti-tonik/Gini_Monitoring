"""
Shared calibration-monitoring math for every per-model notebook/script under
Calibration_Monitoring/. This is the one deliberate deviation from the
Gini_Monitoring convention of duplicating helper functions byte-identically
in every file: the math below has zero per-model variation (unlike SQL and
score-column parsing, which genuinely differ per model and are correctly
kept inline in each notebook), so centralizing it removes a 28-way
maintenance hazard.

Every per-model notebook imports this module via a small sys.path bootstrap
(see the top of any Calibration_* notebook/script) and calls, per bad-rate
target:
    summary_df, band_df = compute_calibration_bands(...)
    gini_df = fetch_gini_for_calibration(...)
then concatenates across the 5 targets and calls write_outputs(...) once.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import brier_score_loss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Same 5 bad-rate definitions Gini already loops over, from the same
# loan_deliquency_data columns. (label_col, maturity_col, bad_rate_type)
BAD_RATE_TARGETS = [
    ("deffpd0", "flg_mature_fpd0", "FPD0"),
    ("deffpd10", "flg_mature_fpd10", "FPD10"),
    ("deffpd30", "flg_mature_fpd30", "FPD30"),
    ("deffspd30", "flg_mature_fspd_30", "FSPD30"),
    ("deffstpd30", "flg_mature_fstpd_30", "FSTPD30"),
]

N_DECILES_DEFAULT = 10
MIN_SAMPLE_SIZE_SUMMARY = 30      # below this -> calibration_status = "Insufficient Data"
MIN_SAMPLE_SIZE_DECILE = 10       # below this -> decile row kept (weights must still sum to 1) but unreliable
MATURITY_RATIO_THRESHOLD = 0.90   # see assign_data_maturity_flag
EPS = 1e-6                        # divide-by-zero guard

# calibration_status thresholds (configurable so risk/pricing can retune
# without touching function bodies)
GREEN_GAP_PP = 1.0
GREEN_OE_LOW, GREEN_OE_HIGH = 0.90, 1.10
AMBER_GAP_PP = 3.0
AMBER_OE_LOW, AMBER_OE_HIGH = 0.75, 1.33

SEGMENT_COLS_CANONICAL = ["model_version", "trench_category", "os_type"]


# ---------------------------------------------------------------------------
# 1. Local query caching (BigQuery cost control)
# ---------------------------------------------------------------------------

def _coerce_bq_extension_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery's pandas client returns DATE/TIME columns using the
    `db-dtypes` extension types (dbdate/dbtime), which DuckDB's Python
    integration (via pyarrow) doesn't recognize. Cast them to plain
    datetime64[ns]/object before handing the frame to duckdb.register()."""
    df = df.copy()
    for col in df.columns:
        dtype_name = str(df[col].dtype)
        if dtype_name in ("dbdate", "date32[day][pyarrow]"):
            df[col] = pd.to_datetime(df[col])
        elif dtype_name == "dbtime":
            df[col] = df[col].astype(str)
    return df


def load_or_pull(
    bq_client: "bigquery.Client",
    sql: str,
    duckdb_path: str | Path,
    table_name: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Run `sql` against BigQuery at most once per `table_name`.

    On the first call for a given (duckdb_path, table_name), executes `sql`
    against BigQuery, caches the result into the local DuckDB file at
    `duckdb_path` as `table_name`, and returns it. Every subsequent call
    with force_refresh=False reads straight from the local DuckDB cache --
    zero additional BigQuery cost. Pass force_refresh=True only for a
    deliberate monitoring-cycle refresh (e.g. once when a new month of data
    needs pulling), never for routine dev/debug re-runs.

    IMPORTANT: use a SEPARATE `duckdb_path` PER MODEL (e.g.
    `calibration_cache_<modelkey>_<tree>.db`), never one shared cache file
    across models. DuckDB takes an exclusive write lock on the whole
    database file, so a shared file would deadlock the moment two models'
    notebooks run concurrently. Per-model files make every model safe to
    run in parallel.
    """
    duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(duckdb_path))
    try:
        table_exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0] > 0

        if table_exists and not force_refresh:
            df = con.execute(f'SELECT * FROM "{table_name}"').df()
            print(f"[load_or_pull] cache hit: '{table_name}' ({len(df)} rows) from {duckdb_path.name} -- no BigQuery cost")
            return df

        print(f"[load_or_pull] {'refreshing' if table_exists else 'pulling'} '{table_name}' from BigQuery ...")
        df = bq_client.query(sql).to_dataframe()
        df = _coerce_bq_extension_dtypes(df)
        con.register("_tmp_df", df)
        con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _tmp_df')
        con.unregister("_tmp_df")
        print(f"[load_or_pull] cached '{table_name}' ({len(df)} rows) into {duckdb_path.name}")
        return df
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 2. Decile binning (mirrors the cascading fallback in
#    Gini_Training_Data_Preparation/psi_training_20260318.ipynb's
#    create_bins_for_features, applied to the PD score instead of a feature)
# ---------------------------------------------------------------------------

def _percentile_edges(scores: np.ndarray, n_bins: int) -> np.ndarray:
    pct_points = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(scores, pct_points))
    return edges


def assign_deciles(scores: pd.Series, n_deciles: int = N_DECILES_DEFAULT) -> tuple[pd.Series, np.ndarray]:
    """Bin `scores` (predicted PD) into up to `n_deciles` bands.

    Cascade: try n_deciles percentile-based edges -> if duplicate edges
    collapse the bin count (common with rounded/point-mass PD scores), retry
    with 5, then 3 bins -> if still degenerate, fall back to equal-width
    bins via np.linspace -> if the whole group is a single constant score,
    return one bin. Outer edges are always forced to +/-inf so pd.cut never
    drops a boundary value. Returns (1-indexed decile labels, bin edges).
    """
    valid = scores.dropna().to_numpy()
    if valid.size == 0:
        return pd.Series(np.nan, index=scores.index), np.array([-np.inf, np.inf])

    if np.nanmin(valid) == np.nanmax(valid):
        edges = np.array([-np.inf, np.inf])
        labels = pd.Series(np.where(scores.isna(), np.nan, 1), index=scores.index)
        return labels, edges

    edges = None
    for n_bins in (n_deciles, 5, 3):
        candidate = _percentile_edges(valid, n_bins)
        if len(candidate) >= n_bins + 1 or len(candidate) >= 3:
            edges = candidate
            break

    if edges is None or len(edges) < 2:
        edges = np.linspace(np.nanmin(valid), np.nanmax(valid), 6)
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([np.nanmin(valid), np.nanmax(valid)])

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binned = pd.cut(scores, bins=edges, labels=False, include_lowest=True, duplicates="drop")
    labels = binned + 1  # 1-indexed, ascending by score
    return labels, edges


# ---------------------------------------------------------------------------
# 3. Core metrics
# ---------------------------------------------------------------------------

def safe_brier_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """sklearn.metrics.brier_score_loss(labels, scores); NaN on any failure
    (empty input, single-class labels sklearn can still score fine, but a
    fully-empty or all-NaN group cannot)."""
    try:
        mask = ~(pd.isna(labels) | pd.isna(scores))
        if mask.sum() == 0:
            return np.nan
        return float(brier_score_loss(np.asarray(labels)[mask], np.asarray(scores)[mask]))
    except Exception:
        return np.nan


def safe_oe_ratio(actual_bad_rate: float, avg_predicted_pd: float) -> float:
    """actual_bad_rate / avg_predicted_pd, NaN-guarded when the denominator
    is ~0 (undefined ratio) rather than raising or returning inf."""
    if avg_predicted_pd is None or pd.isna(avg_predicted_pd) or abs(avg_predicted_pd) < EPS:
        return np.nan
    return actual_bad_rate / avg_predicted_pd


def calibration_status(gap_pp: float, oe_ratio: float, sample_size_ok: bool = True) -> str:
    """Green / Amber / Red / Insufficient Data.

    Green:  |gap_pp| <= 1.0pp  AND  0.90 <= O/E <= 1.10
    Amber:  |gap_pp| <= 3.0pp  AND  0.75 <= O/E <= 1.33   (and not Green)
    Red:    otherwise (including O/E undefined/NaN while bads exist -- a
            severe miss, not a data gap)
    Insufficient Data: sample_size_ok is False, or gap_pp itself is NaN
            because the group was empty
    """
    if not sample_size_ok or pd.isna(gap_pp):
        return "Insufficient Data"

    abs_gap = abs(gap_pp)
    oe_ok_green = (not pd.isna(oe_ratio)) and (GREEN_OE_LOW <= oe_ratio <= GREEN_OE_HIGH)
    oe_ok_amber = (not pd.isna(oe_ratio)) and (AMBER_OE_LOW <= oe_ratio <= AMBER_OE_HIGH)

    if abs_gap <= GREEN_GAP_PP and oe_ok_green:
        return "Green"
    if abs_gap <= AMBER_GAP_PP and oe_ok_amber:
        return "Amber"
    return "Red"


def apply_drift_based_status(
    summary_df: pd.DataFrame,
    n_reference_periods: int = 3,
    reference_period: str | None = None,
) -> pd.DataFrame:
    """Recompute calibration_status around DRIFT from a baseline period,
    not the absolute calibration_gap_pp/oe_ratio.

    WHY: empirically (see beta_stack_model_cash investigation), a model's
    raw `prediction` field is not guaranteed to be a calibrated PD on the
    same scale as the actual bad rate -- it can be a raw/uncalibrated score
    with a large, roughly-CONSTANT offset from the true bad rate (e.g. mean
    score 42% vs actual bad rate 14%). That constant offset is not itself a
    monitoring signal; absolute Green/Amber/Red thresholds would flag every
    single period as Red regardless of whether anything actually changed.
    What matters for monitoring -- and what "calibration shift" means in
    the original brief ("a change over time in the relationship between
    predicted PD and observed bad rate") -- is whether the gap/O-E is
    DRIFTING away from its own established baseline, not its absolute
    level. This is the general, correct framing for every model (it
    degrades gracefully to ~matching the absolute thresholds when a
    model's baseline offset is already near zero), so it's applied
    uniformly rather than special-cased per model.

    For each (model_name, model_version, trench_category, os_type,
    bad_rate_type, segment_type) group, ordered by observation_period:
      - reference_gap_pp = mean(calibration_gap_pp) over the first
        `n_reference_periods` periods (or over `reference_period` alone,
        if given, matched by exact string equality).
      - reference_oe_ratio = mean(oe_ratio) over the same window.
      - calibration_drift_pp = calibration_gap_pp - reference_gap_pp
      - oe_ratio_relative_drift = oe_ratio / reference_oe_ratio, NaN-guarded.
      - calibration_status is recomputed via calibration_status(), fed
        calibration_drift_pp and oe_ratio_relative_drift instead of the
        raw values -- i.e. the SAME Green/Amber/Red cutoffs now apply to
        "how far has this moved from its own baseline" rather than "how
        far from a textbook-perfect 0pp/1.0 O-E".
    Reference-window rows themselves land at ~0pp drift / ~1.0 relative
    O-E, so they correctly show Green (this is the baseline, not yet
    drifted) rather than inheriting whatever the raw offset happened to be.
    """
    df = summary_df.copy()
    group_cols = ["model_name", "model_version", "trench_category", "os_type",
                  "bad_rate_type", "segment_type"]

    df["reference_calibration_gap_pp"] = np.nan
    df["reference_oe_ratio"] = np.nan

    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        gdf = df.loc[idx].sort_values("observation_period")
        if reference_period is not None:
            ref_mask = gdf["observation_period"] == reference_period
            ref_rows = gdf[ref_mask]
            if ref_rows.empty:
                ref_rows = gdf.iloc[: min(n_reference_periods, len(gdf))]
        else:
            ref_rows = gdf.iloc[: min(n_reference_periods, len(gdf))]

        ref_gap = ref_rows["calibration_gap_pp"].mean()
        ref_oe = ref_rows["oe_ratio"].mean()
        df.loc[idx, "reference_calibration_gap_pp"] = ref_gap
        df.loc[idx, "reference_oe_ratio"] = ref_oe

    df["calibration_drift_pp"] = df["calibration_gap_pp"] - df["reference_calibration_gap_pp"]
    df["oe_ratio_relative_drift"] = df.apply(
        lambda r: safe_oe_ratio(r["oe_ratio"], r["reference_oe_ratio"])
        if not pd.isna(r["oe_ratio"]) else np.nan,
        axis=1,
    )

    sample_ok = df["application_count"] >= MIN_SAMPLE_SIZE_SUMMARY
    df["calibration_status"] = [
        calibration_status(drift, oe_drift, ok)
        for drift, oe_drift, ok in zip(df["calibration_drift_pp"], df["oe_ratio_relative_drift"], sample_ok)
    ]
    return df


def expected_calibration_error(band_df: pd.DataFrame, weight_col: str = "population_pct",
                                error_col: str = "abs_calibration_error") -> float:
    """ECE for one (period, model, version, trench, os, bad_rate_type)
    group's decile rows -- population-share-weighted sum of absolute
    calibration error. Identical to band_df['weighted_calibration_error'].sum();
    exposed separately so summary-level ECE and band-level
    weighted_calibration_error are provably the same formula."""
    if band_df.empty:
        return np.nan
    return float((band_df[weight_col] * band_df[error_col]).sum())


# ---------------------------------------------------------------------------
# 4. Segment-combination engine (mirrors calculate_periodic_gini_prod_ver_trench_dimfact,
#    scoped to the 3 dims the calibration schema carries: version/trench/os_type)
# ---------------------------------------------------------------------------

def build_segment_combos(cols: list[str] = SEGMENT_COLS_CANONICAL) -> list[tuple[str, ...]]:
    """Powerset (r=0..len(cols)) over the calibration segment columns.
    r=0 is the "Overall" rollup. 2**3 = 8 combos total -- deliberately not
    Gini's full 9-dim powerset, since calibration_summary/band_details only
    carry model_version/trench_category/os_type."""
    combos = []
    for r in range(len(cols) + 1):
        combos.extend(itertools.combinations(cols, r))
    return combos


def _prep_segment_frame(df: pd.DataFrame, model_version_col: str, trench_col: str,
                         ostype_col: str) -> pd.DataFrame:
    out = df.copy()
    out["model_version"] = out[model_version_col].fillna("Overall").astype(str)
    out["trench_category"] = out[trench_col].fillna("Overall").astype(str)
    out["os_type"] = out[ostype_col].fillna("Overall").astype(str)
    return out


# ---------------------------------------------------------------------------
# 5. Data maturity flag
# ---------------------------------------------------------------------------

def assign_data_maturity_flag(group_mature_count: int, cohort_mature_count: int,
                               threshold: float = MATURITY_RATIO_THRESHOLD) -> int:
    """1 if this bad-rate-target's outcome window has plausibly matured for
    this (period, segment) group, else 0.

    CAVEAT: every reused Gini SQL pull already filters `flg_mature_fpd0 = 1`
    in its outermost WHERE clause, so `dfd` (and therefore `full_cohort_df`
    here) is already restricted to the FPD0-mature population before any
    other target's filter is applied -- there is no way to recover "% of
    the full disbursed cohort matured for FSTPD30" from this pull alone.
    This computes the practical proxy instead:
        maturity_ratio = count(rows mature for THIS target)
                          / count(rows in the FPD0-mature cohort, same period+segment)
    which correctly flags the most recent 1-3 months as immature for
    slow-maturing targets (e.g. FSTPD30) without requiring any change to
    the reused SQL. A true full-cohort denominator would need a second,
    unfiltered query against ml_model_run_details -- left as a documented
    Phase 2 item, not built here.
    """
    if cohort_mature_count in (0, None) or pd.isna(cohort_mature_count):
        return 0
    return int((group_mature_count / cohort_mature_count) >= threshold)


# ---------------------------------------------------------------------------
# 6. Main entry point: compute summary + band tables for one bad-rate target
# ---------------------------------------------------------------------------

def compute_calibration_bands(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    maturity_col: str,
    bad_rate_type: str,
    model_name: str,
    period_col: str = "disbursementdate",
    model_version_col: str = "modelVersionId",
    trench_col: str = "trenchCategory",
    ostype_col: str = "osType",
    account_id_col: str = "digitalLoanAccountId",
    n_deciles: int = N_DECILES_DEFAULT,
    full_cohort_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the calibration summary + PD-decile band tables for ONE
    bad-rate target (e.g. FPD30), across every segment combo in
    build_segment_combos() and every calendar month in `df`.

    `df` must already be filtered to `maturity_col == 1` by the caller (same
    convention as Gini: one BigQuery pull feeds all 5 targets, each
    filtered in pandas). `full_cohort_df` (optional) should be the
    UNFILTERED-by-target pull (i.e. dfd before the per-target maturity
    filter) and is used only for assign_data_maturity_flag's denominator;
    if omitted, data_maturity_flag defaults to 1 for every row (maturity
    check skipped).

    Returns (summary_df, band_df) -- see module docstring / plan for exact
    column lists.
    """
    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=[score_col, label_col, period_col])

    work = _prep_segment_frame(work, model_version_col, trench_col, ostype_col)
    work["observation_period"] = pd.to_datetime(work[period_col]).dt.to_period("M").astype(str)

    cohort = None
    if full_cohort_df is not None:
        cohort = _prep_segment_frame(full_cohort_df.copy(), model_version_col, trench_col, ostype_col)
        cohort["observation_period"] = pd.to_datetime(cohort[period_col]).dt.to_period("M").astype(str)

    summary_rows = []
    band_rows = []

    for combo in build_segment_combos():
        if combo:
            group_cols = ["observation_period", *combo]
        else:
            group_cols = ["observation_period"]

        for period_key, gdf in work.groupby(group_cols, dropna=False):
            if not isinstance(period_key, tuple):
                period_key = (period_key,)
            key_map = dict(zip(group_cols, period_key))
            observation_period = key_map["observation_period"]

            seg = {c: key_map.get(c, "Overall") for c in SEGMENT_COLS_CANONICAL}

            application_count = gdf[account_id_col].nunique()
            if application_count == 0:
                continue  # empty segment/period combo -- no spurious zero row

            bad_count = gdf.loc[gdf[label_col] == 1, account_id_col].nunique()
            avg_predicted_pd = gdf[score_col].mean()
            actual_bad_rate = bad_count / application_count

            calibration_gap = actual_bad_rate - avg_predicted_pd
            calibration_gap_pp = calibration_gap * 100
            oe_ratio = safe_oe_ratio(actual_bad_rate, avg_predicted_pd)
            brier = safe_brier_score(gdf[label_col].to_numpy(), gdf[score_col].to_numpy())

            sample_size_ok = application_count >= MIN_SAMPLE_SIZE_SUMMARY
            status = calibration_status(calibration_gap_pp, oe_ratio, sample_size_ok)

            # data maturity, proxied against the FPD0-mature cohort (see assign_data_maturity_flag)
            if cohort is not None:
                mask = cohort["observation_period"] == observation_period
                for c in SEGMENT_COLS_CANONICAL:
                    if c in combo:
                        mask &= cohort[c] == seg[c]
                cohort_mature_count = cohort.loc[mask, account_id_col].nunique()
                data_maturity_flag = assign_data_maturity_flag(application_count, cohort_mature_count)
            else:
                data_maturity_flag = 1

            # ---- decile pass ----
            decile_labels, edges = assign_deciles(gdf[score_col], n_deciles=n_deciles)
            gdf = gdf.assign(_decile=decile_labels)

            group_band_rows = []
            for decile, ddf in gdf.groupby("_decile", dropna=True):
                d_app_count = ddf[account_id_col].nunique()
                if d_app_count == 0:
                    continue
                d_bad_count = ddf.loc[ddf[label_col] == 1, account_id_col].nunique()
                d_avg_pd = ddf[score_col].mean()
                d_actual_rate = d_bad_count / d_app_count
                d_gap = d_actual_rate - d_avg_pd
                d_abs_error = abs(d_gap)
                d_oe = safe_oe_ratio(d_actual_rate, d_avg_pd)
                d_pop_pct = d_app_count / application_count
                d_weighted_error = d_pop_pct * d_abs_error
                decile_int = int(decile)
                lower = edges[decile_int - 1] if decile_int - 1 < len(edges) else np.nan
                upper = edges[decile_int] if decile_int < len(edges) else np.nan

                group_band_rows.append({
                    "observation_period": observation_period,
                    "model_name": model_name,
                    "model_version": seg["model_version"],
                    "trench_category": seg["trench_category"],
                    "os_type": seg["os_type"],
                    "bad_rate_type": bad_rate_type,
                    "prediction_decile": decile_int,
                    "pd_lower_bound": lower,
                    "pd_upper_bound": upper,
                    "application_count": d_app_count,
                    "bad_count": d_bad_count,
                    "avg_predicted_pd": d_avg_pd,
                    "actual_bad_rate": d_actual_rate,
                    "calibration_gap": d_gap,
                    "abs_calibration_error": d_abs_error,
                    "oe_ratio": d_oe,
                    "population_pct": d_pop_pct,
                    "weighted_calibration_error": d_weighted_error,
                    "data_maturity_flag": data_maturity_flag,
                    "segment_type": "|".join(combo) if combo else "Overall",
                })

            band_rows.extend(group_band_rows)
            group_band_df = pd.DataFrame(group_band_rows)
            ece = expected_calibration_error(group_band_df) if not group_band_df.empty else np.nan

            summary_rows.append({
                "observation_period": observation_period,
                "model_name": model_name,
                "model_version": seg["model_version"],
                "trench_category": seg["trench_category"],
                "os_type": seg["os_type"],
                "bad_rate_type": bad_rate_type,
                "application_count": application_count,
                "bad_count": bad_count,
                "avg_predicted_pd": avg_predicted_pd,
                "actual_bad_rate": actual_bad_rate,
                "calibration_gap_pp": calibration_gap_pp,
                "oe_ratio": oe_ratio,
                "ece": ece,
                "brier_score": brier,
                "gini": np.nan,            # filled in by fetch_gini_for_calibration merge
                "approval_rate": np.nan,   # not derivable from disbursement-filtered SQL -- see module docstring / plan
                "calibration_status": status,
                "segment_type": "|".join(combo) if combo else "Overall",
            })

    summary_df = pd.DataFrame(summary_rows)
    band_df = pd.DataFrame(band_rows)
    return summary_df, band_df


# ---------------------------------------------------------------------------
# 7. Gini read-back (recommended over recomputation -- see plan)
# ---------------------------------------------------------------------------

def fetch_gini_for_calibration(
    bq_client: "bigquery.Client",
    gini_fact_table_id: str,
    model_display_name: str,
    duckdb_path: str | Path,
    table_name: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Read Gini back from the existing fact_<modelkey>_test2/train2 table
    Gini_Monitoring already populates, rather than recomputing it here --
    ONE query per model covering all 5 bad-rate targets (not one per
    target), routed through load_or_pull so repeated notebook re-runs don't
    re-bill BigQuery for this either. Returns columns (observation_period,
    model_version, trench_category, os_type, bad_rate_type, gini); the
    caller filters by bad_rate_type per target and left-merges the rest
    onto compute_calibration_bands' summary_df on (observation_period,
    model_version, trench_category, os_type).

    NOTE: requires knowing that table's actual column names for period type,
    bad-rate label, and model display name filter (these vary slightly
    across Gini scripts -- confirm against the specific fact table's schema
    before wiring this into a given model's notebook; this function assumes
    the common shape: period, bad_rate, Model_display_name, model_version,
    trench_category, ostype, gini_value, start_date).

    IMPORTANT: Gini's segment powerset spans up to 9 dimensions
    (data_selection, model_version, trench_category, loan_type,
    loan_product_type, ostype, apptype, risk_segment, risk_segment_final).
    Every one of those EXCEPT data_selection is NaN-filled to 'Overall' per
    combo by Gini's own update_tables() -- e.g. its "apptype-only" combo
    row also has model_version/trench_category/ostype = 'Overall', same as
    its true "Overall" row -- so all of them must be pinned to 'Overall'
    here too, or this would fan out and match multiple unrelated Gini
    segment_type rows per calibration key. `data_selection` is NOT in
    update_tables()'s fillna list, so it stays NULL (not part of the
    combo) rather than 'Overall'; verified empirically that the NULL row
    and the data_selection='Prod' row carry identical gini_value
    (Test/Train data only ever has one data_selection value), so filtering
    to IS NULL is the correct, non-duplicating choice.

    NOT EVERY Gini fact table has all 9 columns: older/other-lineage
    scripts (confirmed for the Sil trees, whose Gini scripts never pass
    risk_segment_column/risk_segment_final_column and whose fact tables
    don't have those columns at all) only carry a subset. This function
    inspects the actual table schema first and only pins the "Overall"
    filter for columns that exist, so it works across both lineages
    without hardcoding which script variant produced a given table.
    """
    table = bq_client.get_table(gini_fact_table_id)
    available_cols = {f.name for f in table.schema}

    pin_overall_cols = [c for c in
                         ["loan_type", "loan_product_type", "apptype", "risk_segment", "risk_segment_final"]
                         if c in available_cols]
    where_extra = "".join(f"\n          AND {c} = 'Overall'" for c in pin_overall_cols)
    data_selection_clause = "\n          AND data_selection IS NULL" if "data_selection" in available_cols else ""

    query = f"""
        SELECT
            FORMAT_DATE('%Y-%m', start_date) AS observation_period,
            model_version,
            trench_category,
            ostype AS os_type,
            bad_rate AS bad_rate_type,
            gini_value AS gini
        FROM `{gini_fact_table_id}`
        WHERE period = 'Month'
          AND Model_display_name = '{model_display_name}'{data_selection_clause}{where_extra}
    """
    empty = pd.DataFrame(columns=["observation_period", "model_version", "trench_category",
                                   "os_type", "bad_rate_type", "gini"])
    try:
        return load_or_pull(bq_client, query, duckdb_path, table_name, force_refresh=force_refresh)
    except Exception as exc:
        # print (not warnings.warn) -- every calling notebook sets
        # warnings.filterwarnings("ignore") to match Gini's convention,
        # which would otherwise swallow this diagnostic silently.
        print(f"[fetch_gini_for_calibration] FAILED for {model_display_name} ({gini_fact_table_id}): {exc}")
        return empty


# ---------------------------------------------------------------------------
# 8. approval_rate -- documented gap, not silently guessed (see plan §2.7)
# ---------------------------------------------------------------------------

def compute_approval_rate_supplementary(*args, **kwargs):
    """NOT called by default. The SQL reused from Gini filters to
    flagDisbursement = 1, so every pulled row is already a disbursed loan --
    a true approval_rate = disbursed / scored needs a second, unfiltered
    query against ml_model_run_details LEFT JOINed to loan_master_table on
    flagDisbursement. Out of scope for v1; approval_rate ships as NaN.
    """
    raise NotImplementedError(
        "approval_rate requires a supplementary unfiltered query against "
        "ml_model_run_details; not implemented in v1 (see plan)."
    )


# ---------------------------------------------------------------------------
# 9. Output writer (BigQuery + local parquet/csv, one shot per model per run)
# ---------------------------------------------------------------------------

def write_outputs(
    summary_df: pd.DataFrame,
    band_df: pd.DataFrame,
    bq_client: "bigquery.Client",
    summary_table_id: str,
    band_table_id: str,
    local_output_dir: str | Path,
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    """Write both tables to BigQuery (one job each, all 5 bad-rate targets
    already concatenated by the caller so this is a single TRUNCATE per
    table, not Gini's incremental truncate-then-append) and mirror them
    locally as parquet + csv under local_output_dir."""
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)

    job = bq_client.load_table_from_dataframe(summary_df, summary_table_id, job_config=job_config)
    job.result()
    print(f"[write_outputs] wrote {len(summary_df)} rows -> {summary_table_id}")

    job = bq_client.load_table_from_dataframe(band_df, band_table_id, job_config=job_config)
    job.result()
    print(f"[write_outputs] wrote {len(band_df)} rows -> {band_table_id}")

    out_dir = Path(local_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    band_df.to_parquet(out_dir / "band_details.parquet", index=False)
    band_df.to_csv(out_dir / "band_details.csv", index=False)
    print(f"[write_outputs] mirrored local parquet/csv -> {out_dir}")
