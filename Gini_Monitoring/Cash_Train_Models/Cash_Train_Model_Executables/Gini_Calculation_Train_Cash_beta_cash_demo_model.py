# %% [markdown]
# # 🌿 Gini Calculation Cash Train

# %% [markdown]
# This Notebook will calculate gini using the bit faster approach. Also I plan to separate each version of Model separately

# %% [markdown]
# ## 💻 Define Libraries

# %%
# %% [markdown]
# # Jupyter Notebook Loading Header
#
# This is a custom loading header for Jupyter Notebooks in Visual Studio Code.
# It includes common imports and settings to get you started quickly.
# %% [markdown]
## Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.cloud import storage
import os
import tempfile
import time
from datetime import datetime
import uuid
import joblib
import uuid
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
import gcsfs
import duckdb as dd
import pickle
import joblib
from typing import Union
import io
path = r'C:\Users\Dwaipayan\AppData\Roaming\gcloud\application_default_credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
client = bigquery.Client(project='prj-prod-dataplatform')
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
## Configure Settings
# Set options or configurations as needed
pd.set_option('display.max_columns', None)
pd.set_option("Display.max_rows", 100)

# %% [markdown]
# ## 🏗️Functions

# %%
def calculate_gini(scores, labels):
    """
    Calculate Gini coefficient using ROC AUC score.
    Gini = 2 * AUC - 1

    Returns np.nan when:
    - Fewer than 2 observations
    - All labels are the same class (no variation in labels)
    - AUC calculation fails

    IMPORTANT: Labels do NOT need to be 0/1. This function correctly handles
    any binary encoding: 0/1, 1/2, True/False, etc.
    roc_auc_score only requires exactly 2 distinct classes to be present.
    """
    n = len(scores)
    if n < 2:
        return np.nan

    label_sum = np.sum(labels)

    # Handle case where no positive labels exist (all zeros)
    # This prevents division by zero warning
    if label_sum == 0:
        return np.nan

    try:
        auc = roc_auc_score(labels, scores)
        return 2 * auc - 1
    except Exception:
        return np.nan

# %% [markdown]
# ### Gini Optimized

# %%
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import rankdata


def calculate_gini_fast(scores, labels):
    """Rank-based AUC -> Gini. See fast_gini.py for full docstring/rationale."""
    n = scores.shape[0]
    if n < 2:
        return np.nan
    uniq = np.unique(labels)
    if uniq.shape[0] != 2:
        return np.nan
    pos_label = uniq.max()
    is_pos = labels == pos_label
    n_pos = int(is_pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores, method="average")
    sum_ranks_pos = ranks[is_pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return 2.0 * auc - 1.0


def calculate_periodic_gini_prod_ver_trench_dimfact(
    df,
    score_column,
    label_column,
    namecolumn,
    data_selection_column=None,
    model_version_column=None,
    trench_column=None,
    loan_type_column=None,
    loan_product_type_column=None,
    ostype_column=None,
    apptype_column=None,
    risk_segment_column=None,
    risk_segment_final_column=None,
    account_id_column=None,
):
    """
    Drop-in replacement for calculate_periodic_gini_prod_ver_trench_dimfact.

    Same inputs/outputs (fact_table, dimension_table), same segment_type
    naming convention and same set of output columns. Internals differ:

    - Every (segment-column-subset, week/month) grouping is computed with a
      single native pandas groupby instead of a hand-rolled recursive
      filter over the power set of segment values. Pandas's hash-based
      group split does in one pass what the recursive version did with a
      fresh full-dataframe boolean filter per leaf combination.
    - gini_value, distinct_accounts and bad_count are computed together in
      ONE pass over each grouping's row-index buckets (via numpy arrays),
      instead of 3 separate groupby calls + 2 merges per dataset.
    - AUC uses a rank-based (Mann-Whitney) formula instead of
      sklearn.roc_auc_score, which removes per-call validation overhead
      that dominates when the function is invoked over many small groups.
      Verified equivalent to `2*roc_auc_score(labels, scores)-1` to
      floating-point precision (see fast_gini.py for the check).
    """
    required_columns = ["disbursementdate", score_column, label_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")

    optional_columns = {
        "data_selection": data_selection_column,
        "model_version": model_version_column,
        "trench": trench_column,
        "loan_type": loan_type_column,
        "loan_product_type": loan_product_type_column,
        "ostype": ostype_column,
        "apptype": apptype_column,
        "risk_segment": risk_segment_column,
        "risk_segment_final": risk_segment_final_column,
        "account_id": account_id_column,
    }
    for col_name, col in optional_columns.items():
        if col and col not in df.columns:
            raise ValueError(
                f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe"
            )

    df = df.copy()
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df.dropna(subset=[score_column, label_column])

    # precompute both period grains ONCE for the whole frame
    df["week"] = df["disbursementdate"].dt.to_period("W")
    df["month"] = df["disbursementdate"].dt.to_period("M")

    # (display_name_for_segment_type_string, source_column, output_field_name)
    segment_defs = []
    if data_selection_column:
        segment_defs.append(("DataSelection", data_selection_column, "data_selection"))
    if model_version_column:
        segment_defs.append(("ModelVersion", model_version_column, "model_version"))
    if trench_column:
        segment_defs.append(("Trench", trench_column, "trench_category"))
    if loan_type_column:
        segment_defs.append(("LoanType", loan_type_column, "loan_type"))
    if loan_product_type_column:
        segment_defs.append(("ProductType", loan_product_type_column, "loan_product_type"))
    if ostype_column:
        segment_defs.append(("OSType", ostype_column, "ostype"))
    if apptype_column:
        segment_defs.append(("apptype", apptype_column, "apptype"))
    if risk_segment_column:
        segment_defs.append(("risk_segment", risk_segment_column, "risk_segment"))
    if risk_segment_final_column:
        segment_defs.append(("risk_segment_final", risk_segment_final_column, "risk_segment_final"))

    all_output_fields = [
        "data_selection", "model_version", "trench_category", "loan_type",
        "loan_product_type", "ostype", "apptype", "risk_segment", "risk_segment_final",
    ]

    scores_arr = df[score_column].to_numpy()
    labels_arr = df[label_column].to_numpy()
    accounts_arr = df[account_id_column].to_numpy() if account_id_column else None

    # build the list of column-subsets to process: r=0 ("Overall") .. full power set
    combos = [()]
    for r in range(1, len(segment_defs) + 1):
        combos.extend(combinations(segment_defs, r))

    period_specs = [
        ("week", "Week", lambda ts: ts, lambda ts: ts + pd.Timedelta(days=6)),
        ("month", "Month", lambda ts: ts, lambda ts: ts + pd.DateOffset(months=1) - pd.Timedelta(days=1)),
    ]

    rows = []
    for combo in combos:
        combo_source_cols = [col for _, col, _ in combo]
        for period_col, period_label, start_fn, end_fn in period_specs:
            key_cols = combo_source_cols + [period_col]

            gb = df.groupby(key_cols, observed=True, dropna=True, sort=True)
            group_indices = gb.indices  # {key: positional row-index array}

            for key, idx in group_indices.items():
                if not isinstance(key, tuple):
                    key = (key,)
                seg_values = key[:-1]
                period_value = key[-1]

                seg_scores = scores_arr[idx]
                seg_labels = labels_arr[idx]
                gini_value = calculate_gini_fast(seg_scores, seg_labels)

                if account_id_column:
                    seg_accounts = accounts_arr[idx]
                    distinct_accounts = pd.unique(seg_accounts).shape[0]
                    bad_mask = seg_labels == 1
                    bad_count = (
                        pd.unique(seg_accounts[bad_mask]).shape[0] if bad_mask.any() else 0
                    )
                else:
                    distinct_accounts = None
                    bad_count = None

                start_date = start_fn(period_value.to_timestamp())
                end_date = end_fn(period_value.to_timestamp())

                seg_name = (
                    "Overall"
                    if not combo
                    else "_".join(f"{disp}_{val}" for (disp, _, _), val in zip(combo, seg_values))
                )

                row = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "period": period_label,
                    "gini_value": gini_value,
                    "distinct_accounts": distinct_accounts,
                    "bad_count": bad_count,
                    "Model_Name": score_column,
                    "bad_rate": namecolumn,
                    "segment_type": seg_name,
                }
                for field in all_output_fields:
                    row[field] = None
                for (_, _, field), val in zip(combo, seg_values):
                    row[field] = val

                rows.append(row)

    fact_table = pd.DataFrame(rows)
    fact_table = fact_table.sort_values(["segment_type", "start_date"]).reset_index(drop=True)

    dimension_table = (
        fact_table[["Model_Name", "bad_rate", "segment_type"] + all_output_fields]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dimension_table["segment_id"] = range(len(dimension_table))

    fact_table = fact_table.merge(
        dimension_table[["segment_id", "Model_Name", "bad_rate", "segment_type"] + all_output_fields],
        on=["Model_Name", "bad_rate", "segment_type"] + all_output_fields,
        how="left",
    )

    fact_table = fact_table[
        ["segment_id", "start_date", "end_date", "period", "gini_value", "distinct_accounts",
         "bad_count", "Model_Name", "bad_rate", "segment_type"] + all_output_fields
    ]
    dimension_table = dimension_table[
        ["segment_id", "Model_Name", "bad_rate", "segment_type"] + all_output_fields
    ]

    return fact_table, dimension_table

# %% [markdown]
# ### Calculate Periodic Gini Duckdb

# %%
import pandas as pd
import numpy as np
from itertools import combinations
import duckdb


def calculate_periodic_gini_duckdb(
    df,
    score_column,
    label_column,
    namecolumn,
    data_selection_column=None,
    model_version_column=None,
    trench_column=None,
    loan_type_column=None,
    loan_product_type_column=None,
    ostype_column=None,
    apptype_column=None,
    risk_segment_column=None,
    risk_segment_final_column=None,
    account_id_column=None,
    con=None,
):
    """
    Same contract/output as calculate_periodic_gini_prod_ver_trench_dimfact,
    but the per-group AUC (rank-sum / Mann-Whitney) is computed inside DuckDB
    with window functions instead of a Python loop.

    ASSUMPTION: label_column is already 0/1 with 1 = the event you're
    scoring for (matches the normal "bad flag" convention). If your labels
    are some other binary encoding, remap to 0/1 before calling this
    (`(df[label_column] == bad_value).astype(int)`), since SQL doesn't get
    the "treat the larger raw value as positive" inference the Python
    version does for free.

    RANK()-based tie handling: DuckDB's RANK() gives tied scores the same
    (minimum) rank; adding (COUNT(*) OVER (same score) - 1) / 2 converts
    that into the *average* rank scipy/sklearn use for ties, so results
    match calculate_gini_fast / roc_auc_score exactly.
    """
    required_columns = ["disbursementdate", score_column, label_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")

    df = df.copy()
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df.dropna(subset=[score_column, label_column])
    df["week"] = df["disbursementdate"].dt.to_period("W").astype(str)
    df["month"] = df["disbursementdate"].dt.to_period("M").astype(str)

    segment_defs = []
    if data_selection_column:
        segment_defs.append(("DataSelection", data_selection_column, "data_selection"))
    if model_version_column:
        segment_defs.append(("ModelVersion", model_version_column, "model_version"))
    if trench_column:
        segment_defs.append(("Trench", trench_column, "trench_category"))
    if loan_type_column:
        segment_defs.append(("LoanType", loan_type_column, "loan_type"))
    if loan_product_type_column:
        segment_defs.append(("ProductType", loan_product_type_column, "loan_product_type"))
    if ostype_column:
        segment_defs.append(("OSType", ostype_column, "ostype"))
    if apptype_column:
        segment_defs.append(("apptype", apptype_column, "apptype"))
    if risk_segment_column:
        segment_defs.append(("risk_segment", risk_segment_column, "risk_segment"))
    if risk_segment_final_column:
        segment_defs.append(("risk_segment_final", risk_segment_final_column, "risk_segment_final"))

    all_output_fields = [
        "data_selection", "model_version", "trench_category", "loan_type",
        "loan_product_type", "ostype", "apptype", "risk_segment", "risk_segment_final",
    ]

    con = con or duckdb.connect()
    con.register("gini_src", df)

    combos = [()]
    for r in range(1, len(segment_defs) + 1):
        combos.extend(combinations(segment_defs, r))

    period_specs = [
        ("week", "Week", lambda ts: ts, lambda ts: ts + pd.Timedelta(days=6)),
        ("month", "Month", lambda ts: ts, lambda ts: ts + pd.DateOffset(months=1) - pd.Timedelta(days=1)),
    ]

    all_rows = []
    for combo in combos:
        combo_cols = [col for _, col, _ in combo]
        for period_col, period_label, start_fn, end_fn in period_specs:
            part_cols = combo_cols + [period_col]
            part_csv = ", ".join(f'"{c}"' for c in part_cols)

            acct_select = f', "{account_id_column}"' if account_id_column else ""
            acct_agg = (
                f', COUNT(DISTINCT "{account_id_column}") AS distinct_accounts'
                f', COUNT(DISTINCT "{account_id_column}") FILTER (WHERE lbl = 1) AS bad_count'
                if account_id_column else ", NULL AS distinct_accounts, NULL AS bad_count"
            )

            query = f"""
                WITH ranked AS (
                    SELECT
                        {part_csv},
                        "{label_column}" AS lbl
                        {acct_select}
                        , RANK() OVER (PARTITION BY {part_csv} ORDER BY "{score_column}")
                            + (COUNT(*) OVER (PARTITION BY {part_csv}, "{score_column}") - 1) / 2.0
                          AS avg_rank
                    FROM gini_src
                ),
                agg AS (
                    SELECT
                        {part_csv},
                        COUNT(*) AS n,
                        SUM(lbl) AS n_pos,
                        SUM(avg_rank) FILTER (WHERE lbl = 1) AS sum_rank_pos
                        {acct_agg}
                    FROM ranked
                    GROUP BY {part_csv}
                )
                SELECT
                    {part_csv},
                    distinct_accounts, bad_count,
                    CASE WHEN n < 2 OR n_pos = 0 OR n_pos = n THEN NULL
                         ELSE 2.0 * (sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * (n - n_pos)) - 1.0
                    END AS gini_value
                FROM agg
            """
            res = con.sql(query).df()

            for _, r_ in res.iterrows():
                seg_values = tuple(r_[c] for c in combo_cols)
                period_value = pd.Period(r_[period_col], freq="W" if period_col == "week" else "M")
                start_date = start_fn(period_value.to_timestamp())
                end_date = end_fn(period_value.to_timestamp())
                seg_name = "Overall" if not combo else "_".join(
                    f"{disp}_{val}" for (disp, _, _), val in zip(combo, seg_values)
                )
                row = {
                    "start_date": start_date, "end_date": end_date, "period": period_label,
                    "gini_value": r_["gini_value"], "distinct_accounts": r_["distinct_accounts"],
                    "bad_count": r_["bad_count"], "Model_Name": score_column,
                    "bad_rate": namecolumn, "segment_type": seg_name,
                }
                for field in all_output_fields:
                    row[field] = None
                for (_, _, field), val in zip(combo, seg_values):
                    row[field] = val
                all_rows.append(row)

    fact_table = pd.DataFrame(all_rows).sort_values(["segment_type", "start_date"]).reset_index(drop=True)
    dimension_table = (
        fact_table[["Model_Name", "bad_rate", "segment_type"] + all_output_fields]
        .drop_duplicates().reset_index(drop=True)
    )
    dimension_table["segment_id"] = range(len(dimension_table))
    fact_table = fact_table.merge(
        dimension_table[["segment_id", "Model_Name", "bad_rate", "segment_type"] + all_output_fields],
        on=["Model_Name", "bad_rate", "segment_type"] + all_output_fields, how="left",
    )
    fact_table = fact_table[
        ["segment_id", "start_date", "end_date", "period", "gini_value", "distinct_accounts",
         "bad_count", "Model_Name", "bad_rate", "segment_type"] + all_output_fields
    ]
    dimension_table = dimension_table[["segment_id", "Model_Name", "bad_rate", "segment_type"] + all_output_fields]
    return fact_table, dimension_table

# %% [markdown]
# ### Calculate Gini Multiprocessing

# %%
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# ---- shared, read-only, per-worker state -----------------------------------
# Populated in the PARENT process before the pool is created. On Linux
# (fork start method, the default), forked workers inherit this via
# copy-on-write with zero serialization cost -- only the *tasks* (a combo +
# a period name, a few bytes) cross the process boundary, not the data.
_SHARED = {}


def _init_shared(df, score_column, label_column, account_id_column):
    _SHARED["df"] = df
    _SHARED["score_column"] = score_column
    _SHARED["label_column"] = label_column
    _SHARED["account_id_column"] = account_id_column


def calculate_gini_fast(scores, labels):
    n = scores.shape[0]
    if n < 2:
        return np.nan
    uniq = np.unique(labels)
    if uniq.shape[0] != 2:
        return np.nan
    pos_label = uniq.max()
    is_pos = labels == pos_label
    n_pos = int(is_pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores, method="average")
    sum_ranks_pos = ranks[is_pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return 2.0 * auc - 1.0


def _process_task(task):
    """Runs in a worker process. One task = one (segment-combo, period) pair
    -- e.g. "Trench x ModelVersion, weekly" -- which is exactly the unit of
    work that was one iteration of the outer Python loop in the
    single-process version. Each task does a full groupby over the shared
    dataframe, so the per-task work is substantial enough that process
    overhead (task dispatch, result pickling) is negligible next to it.
    """
    combo, period_col, period_label = task
    df = _SHARED["df"]
    score_column = _SHARED["score_column"]
    label_column = _SHARED["label_column"]
    account_id_column = _SHARED["account_id_column"]

    combo_source_cols = [col for _, col, _ in combo]
    key_cols = combo_source_cols + [period_col]

    scores_arr = df[score_column].to_numpy()
    labels_arr = df[label_column].to_numpy()
    accounts_arr = df[account_id_column].to_numpy() if account_id_column else None

    gb = df.groupby(key_cols, observed=True, dropna=True, sort=True)
    rows = []
    for key, idx in gb.indices.items():
        if not isinstance(key, tuple):
            key = (key,)
        seg_values = key[:-1]
        period_value = key[-1]

        seg_scores = scores_arr[idx]
        seg_labels = labels_arr[idx]
        gini_value = calculate_gini_fast(seg_scores, seg_labels)

        if account_id_column:
            seg_accounts = accounts_arr[idx]
            distinct_accounts = pd.unique(seg_accounts).shape[0]
            bad_mask = seg_labels == 1
            bad_count = pd.unique(seg_accounts[bad_mask]).shape[0] if bad_mask.any() else 0
        else:
            distinct_accounts, bad_count = None, None

        ts = period_value.to_timestamp()
        if period_col == "week":
            start_date, end_date = ts, ts + pd.Timedelta(days=6)
        else:
            start_date, end_date = ts, ts + pd.DateOffset(months=1) - pd.Timedelta(days=1)

        seg_name = "Overall" if not combo else "_".join(
            f"{disp}_{val}" for (disp, _, _), val in zip(combo, seg_values)
        )
        row = {
            "start_date": start_date, "end_date": end_date, "period": period_label,
            "gini_value": gini_value, "distinct_accounts": distinct_accounts,
            "bad_count": bad_count, "segment_type": seg_name,
        }
        for (_, _, field), val in zip(combo, seg_values):
            row[field] = val
        rows.append(row)
    return rows


def calculate_periodic_gini_multiproc(
    df, score_column, label_column, namecolumn,
    data_selection_column=None, model_version_column=None, trench_column=None,
    loan_type_column=None, loan_product_type_column=None, ostype_column=None,
    apptype_column=None, risk_segment_column=None, risk_segment_final_column=None,
    account_id_column=None, n_workers=None,
):
    """Same contract as the single-process optimized version. Parallelizes
    across (segment-combo, period) tasks with ProcessPoolExecutor. Best fit
    when you have many segment columns/combos and multiple cores -- each
    core independently owns a full groupby pass, so speedup tracks core
    count fairly closely (modulo the largest single combo, which caps how
    much any one task can be split further)."""
    df = df.copy()
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df.dropna(subset=[score_column, label_column])
    df["week"] = df["disbursementdate"].dt.to_period("W")
    df["month"] = df["disbursementdate"].dt.to_period("M")

    segment_defs = []
    if data_selection_column: segment_defs.append(("DataSelection", data_selection_column, "data_selection"))
    if model_version_column: segment_defs.append(("ModelVersion", model_version_column, "model_version"))
    if trench_column: segment_defs.append(("Trench", trench_column, "trench_category"))
    if loan_type_column: segment_defs.append(("LoanType", loan_type_column, "loan_type"))
    if loan_product_type_column: segment_defs.append(("ProductType", loan_product_type_column, "loan_product_type"))
    if ostype_column: segment_defs.append(("OSType", ostype_column, "ostype"))
    if apptype_column: segment_defs.append(("apptype", apptype_column, "apptype"))
    if risk_segment_column: segment_defs.append(("risk_segment", risk_segment_column, "risk_segment"))
    if risk_segment_final_column: segment_defs.append(("risk_segment_final", risk_segment_final_column, "risk_segment_final"))

    all_output_fields = [
        "data_selection", "model_version", "trench_category", "loan_type",
        "loan_product_type", "ostype", "apptype", "risk_segment", "risk_segment_final",
    ]

    combos = [()]
    for r in range(1, len(segment_defs) + 1):
        combos.extend(combinations(segment_defs, r))

    tasks = []
    for combo in combos:
        tasks.append((combo, "week", "Week"))
        tasks.append((combo, "month", "Month"))

    ctx = mp.get_context("fork")  # required for the zero-copy _SHARED trick; Linux only
    all_rows = []
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
        initializer=_init_shared, initargs=(df, score_column, label_column, account_id_column),
    ) as pool:
        for rows in pool.map(_process_task, tasks):
            all_rows.extend(rows)

    for row in all_rows:
        row["Model_Name"] = score_column
        row["bad_rate"] = namecolumn
        for field in all_output_fields:
            row.setdefault(field, None)

    fact_table = pd.DataFrame(all_rows).sort_values(["segment_type", "start_date"]).reset_index(drop=True)
    dimension_table = (
        fact_table[["Model_Name", "bad_rate", "segment_type"] + all_output_fields]
        .drop_duplicates().reset_index(drop=True)
    )
    dimension_table["segment_id"] = range(len(dimension_table))
    fact_table = fact_table.merge(
        dimension_table[["segment_id", "Model_Name", "bad_rate", "segment_type"] + all_output_fields],
        on=["Model_Name", "bad_rate", "segment_type"] + all_output_fields, how="left",
    )
    fact_table = fact_table[
        ["segment_id", "start_date", "end_date", "period", "gini_value", "distinct_accounts",
         "bad_count", "Model_Name", "bad_rate", "segment_type"] + all_output_fields
    ]
    dimension_table = dimension_table[["segment_id", "Model_Name", "bad_rate", "segment_type"] + all_output_fields]
    return fact_table, dimension_table

# %% [markdown]
# ### update_tables

# %%
def update_tables(
    fact_table: pd.DataFrame,
    dimension_table: pd.DataFrame,
    model_name: str,
    product: str,
) -> tuple:
    """
    Updates fact_table and dimension_table:
    - Sets 'Model_display_name' to the given model_name
    - Replaces NaN values in specified columns with 'Overall'

    Returns:
        Updated fact_table and dimension_table as a tuple
    """
    # Columns where missing values should be replaced
    cols_to_replace = [
        "model_version",
        "trench_category",
        "loan_type",
        "loan_product_type",
        "ostype",
        "apptype",
        "risk_segment", 
        "risk_segment_final",
    ]

    # Update fact_table
    fact_table["Model_display_name"] = model_name
    fact_table["Product_Category"] = product
    fact_table[cols_to_replace] = fact_table[cols_to_replace].fillna("Overall")

    # Update dimension_table
    dimension_table["Model_display_name"] = model_name
    dimension_table["Product_Category"] = product
    dimension_table[cols_to_replace] = dimension_table[cols_to_replace].fillna(
        "Overall"
    )

    return fact_table, dimension_table


# %% [markdown]
# ## 📳 Models

# %%
# ### FPD10
# ### Train
# %%
sq = """
WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      deviceOs osType,
      Data_selection
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-AppScore-Model', 'apps_score_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction beta_cash_app_score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-AppScore-Model'
          THEN 'apps_score_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-AppScore-Model', 'apps_score_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.beta_cash_app_score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffpd10,
      del.flg_mature_fpd10,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.beta_cash_app_score IS NOT NULL
      AND del.flg_mature_fpd10 = 1
      AND upper(loanmaster.new_loan_type) not like '%SIL%'
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;

  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()

df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)
print("Gini Calculation for Beta cash app score FPD10 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
     risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

df_f_fpd10_appscorecash = fact_table.copy()
df_d_fpd10_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fpd10_appscorecash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%

# ### FPD30
# ### Train

# %%
sq = """
WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      deviceOs osType,
      Data_selection
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-AppScore-Model', 'apps_score_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction beta_cash_app_score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-AppScore-Model'
          THEN 'apps_score_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-AppScore-Model', 'apps_score_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.beta_cash_app_score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffpd30,
      del.flg_mature_fpd30,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.beta_cash_app_score IS NOT NULL
      AND del.flg_mature_fpd30 = 1
      AND upper(loanmaster.new_loan_type) not like '%SIL%'
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()



# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)

print("Gini Calculation for Beta cash app score FPD30 started") 
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    risk_segment_column='risk_segment',
    risk_segment_final_column='risk_segment_final',
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

df_f_fpd30_appscorecash = fact_table.copy()
df_d_fpd30_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fpd30_appscorecash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# %%
# ### FSPD30
# ### Train

# %%
sq = """
WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      deviceOs osType,
      Data_selection
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-AppScore-Model', 'apps_score_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction beta_cash_app_score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-AppScore-Model'
          THEN 'apps_score_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-AppScore-Model', 'apps_score_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.beta_cash_app_score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffspd30,
      del.flg_mature_fspd_30,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.beta_cash_app_score IS NOT NULL
      AND del.flg_mature_fspd_30 = 1
      AND upper(loanmaster.new_loan_type) not like '%SIL%'
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()

# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)

print("Gini Calculation for Beta cash app score FSPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    risk_segment_column='risk_segment',
    risk_segment_final_column='risk_segment_final',
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

# %%
df_f_fspd30_appscorecash = fact_table.copy()
df_d_fspd30_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fspd30_appscorecash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%



# ### FSTPD30

# ### Train

# %%
sq = """WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      deviceOs osType,
      Data_selection
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-AppScore-Model', 'apps_score_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction beta_cash_app_score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-AppScore-Model'
          THEN 'apps_score_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-AppScore-Model', 'apps_score_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.beta_cash_app_score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffstpd30,
      del.flg_mature_fstpd_30,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.beta_cash_app_score IS NOT NULL
      AND del.flg_mature_fstpd_30 = 1
      AND upper(loanmaster.new_loan_type) not like '%SIL%'
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;

  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()



# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)

print("Gini Calculation for Beta cash app score FSTPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
       risk_segment_column='risk_segment',
    risk_segment_final_column='risk_segment_final',
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_appscorecash = fact_table.copy()
df_d_fstpd30_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fstpd30_appscorecash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factappscorecash = pd.concat([df_f_fpd0_appscorecash, df_f_fpd10_appscorecash, df_f_fpd30_appscorecash, df_f_fspd30_appscorecash, df_f_fstpd30_appscorecash], ignore_index=True)
dimappscorecash = pd.concat([df_d_fpd0_appscorecash, df_d_fpd10_appscorecash, df_d_fpd30_appscorecash, df_d_fspd30_appscorecash, df_d_fstpd30_appscorecash], ignore_index=True)


# %%


# %% [markdown]
# #### Beta-Cash-Demo-Model

# %%

facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademocash_train2"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betademocash_train2"

# ## Beta-Cash-Demo-Model
# ### FPD0


# ### Train

# %%
sq = """
WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      Data_selection,
      deviceOs osType,
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction Beta_Cash_Demo_Score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-Demo-Model'
          THEN 'beta_demo_model_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType,
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.Beta_Cash_Demo_Score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffpd0,
      del.flg_mature_fpd0,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.Beta_Cash_Demo_Score IS NOT NULL
      AND del.flg_mature_fpd0 = 1
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;

  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()

# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    risk_segment_column='risk_segment',
    risk_segment_final_column='risk_segment_final',
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_betademocash = fact_table.copy()
df_d_fpd0_betademocash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fpd0_betademocash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10

# ### Train

# %%
sq = """
WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      Data_selection,
      deviceOs osType,
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction Beta_Cash_Demo_Score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-Demo-Model'
          THEN 'beta_demo_model_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType,
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.Beta_Cash_Demo_Score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffpd10,
      del.flg_mature_fpd10,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.Beta_Cash_Demo_Score IS NOT NULL
      AND del.flg_mature_fpd10 = 1
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    risk_segment_column='risk_segment',
    risk_segment_final_column='risk_segment_final',
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

df_f_fpd10_betademocash = fact_table.copy()
df_d_fpd10_betademocash = dimension_table.copy()
# %%
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fpd10_betademocash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD30

# ### Train

# %%
sq = """
WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      Data_selection,
      deviceOs osType,
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction Beta_Cash_Demo_Score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-Demo-Model'
          THEN 'beta_demo_model_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType,
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.Beta_Cash_Demo_Score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffpd30,
      del.flg_mature_fpd30,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.Beta_Cash_Demo_Score IS NOT NULL
      AND del.flg_mature_fpd30 = 1
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;

  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()

# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    risk_segment_column='risk_segment',
    risk_segment_final_column='risk_segment_final',
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

df_f_fpd30_betademocash = fact_table.copy()
df_d_fpd30_betademocash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fpd30_betademocash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete



# %%



# ### FSPD30

# ### Train

# %%
sq = """
WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      Data_selection,
      deviceOs osType,
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction Beta_Cash_Demo_Score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-Demo-Model'
          THEN 'beta_demo_model_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType,
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.Beta_Cash_Demo_Score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffspd30,
      del.flg_mature_fspd_30,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.Beta_Cash_Demo_Score IS NOT NULL
      AND del.flg_mature_fspd_30 = 1
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;

  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    risk_segment_column='risk_segment',
    risk_segment_final_column='risk_segment_final',
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

df_f_fspd30_betademocash = fact_table.copy()
df_d_fspd30_betademocash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fspd30_betademocash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%



# ### FSTPD30

# ### Train

# %%
sq = """
WITH
  parsed AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      modelDisplayName,
      modelVersionId,
      start_time,
      end_time,
      prediction,
      trenchCategory,
      REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
      Data_selection,
      deviceOs osType,
    FROM
      prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
    WHERE modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
  ),
  modelname AS (
    SELECT
      customerId,
      digitalLoanAccountId,
      start_time,
      prediction Beta_Cash_Demo_Score,
      CASE
        WHEN modelDisplayName LIKE 'Beta-Cash-Demo-Model'
          THEN 'beta_demo_model_cash'
        ELSE modelDisplayName
        END AS modelDisplayName,
      modelVersionId,
      trenchCategory,
      case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      Data_selection,
      osType,
    FROM parsed
  ),
  deliquency AS (
    SELECT
      loanAccountNumber,
      CASE
        WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1
        ELSE 0
        END deffpd0,
      CASE
        WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1
        ELSE 0
        END deffpd10,
      CASE
        WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1
        ELSE 0
        END deffpd30,
      CASE
        WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1, 2) THEN 1
        ELSE 0
        END deffspd30,
      CASE
        WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1, 2, 3) THEN 1
        ELSE 0
        END deffstpd30,
      CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END flg_mature_fpd0,
      CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END flg_mature_fpd10,
      CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END flg_mature_fpd30,
      CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END flg_mature_fspd_30,
      CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END flg_mature_fstpd_30
    FROM prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data
  ),
  segmentdata AS (
    SELECT
      loan.customerid,
      loan.digitalLoanAccountId,
      trench_category.trenchCategory,
      loan.offer_id,
      CASE
        WHEN COALESCE(trench1_seg.risk_segment) IS NULL
          THEN 'Unsegmented'
        ELSE COALESCE(trench1_seg.risk_segment)
        END AS risk_segment,
      appVersion,
      flagApproval,
      tsa_onboarding_time,
      IF(
        applicationStatus IN ('COMPLETED', 'ACTIVATED', 'APPROVED'),
        'Loan Approved',
        'Loan Not Approved') AS loan_application_status,
      -- if(disbursementDateTime is not null, 'Loan Disbursed', 'Loan Not Approved') loan_application_status
      DATE(decision_date) AS application_date
    FROM
      (
        SELECT DISTINCT
          digitalLoanAccountId,
          customerId,
          applicationStatus,
          disbursementDateTime,
          date(decision_date) decision_date,
          appVersion,
          flagApproval,
          tsa_onboarding_time,
          offer_id
        FROM `risk_credit_mis.loan_master_table`
        WHERE
          date(decision_date) >= date('2025-11-10') AND new_loan_type = 'Quick'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY customerId ORDER BY decision_date desc)=1
      ) loan
    LEFT JOIN
      (
        SELECT
          digitalLoanAccountId,
          CASE
            WHEN trenchCategory = 'Trench 1' THEN 'Trench-1'
            WHEN trenchCategory = 'Trench 2' THEN 'Trench-2'
            WHEN trenchCategory = 'Trench 3' THEN 'Trench-3'
            END AS trenchCategory,
          publish_time
        FROM `audit_balance.ml_model_run_details`
        WHERE
          modelDisplayName IN ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountId ORDER BY publish_time DESC)
          = 1
      ) trench_category
      ON trench_category.digitalLoanAccountId = loan.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT
          cust_id, risk_segment, created_date, created_by, offer_id
        FROM `dl_loans_db_raw.tdbk_loan_offers_trx`
        WHERE offer_type = 'SEGMENTED_ACL'
        -- AND created_by='GCP-API-CALL'
        -- QUALIFY ROW_NUMBER() OVER(PARTITION BY cust_id ORDER BY created_date desc)=1
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
  base AS (
    SELECT DISTINCT
      r.customerId,
      r.digitalLoanAccountId,
      loanmaster.loanAccountNumber,
      r.modelDisplayName,
      r.Beta_Cash_Demo_Score,
      coalesce(
        IF(
          loanmaster.new_loan_type = 'Flex-up',
          loanmaster.startApplyDateTime,
          loanmaster.termsAndConditionsSubmitDateTime),
        CAST(r.start_time AS datetime)) AS appln_submit_datetime,
      date(loanmaster.disbursementDateTime) disbursementdate,
      format_date(
        '%Y-%m',
        coalesce(
          IF(
            loanmaster.new_loan_type = 'Flex-up',
            loanmaster.startApplyDateTime,
            loanmaster.termsAndConditionsSubmitDateTime),
          CAST(r.start_time AS datetime))) AS Application_month,
      Data_selection,
      del.deffstpd30,
      del.flg_mature_fstpd_30,
      loanmaster.new_loan_type,
      modelVersionId,
      r.trenchCategory,
      r.Application_type,
      CASE
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
        WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
        WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1, 2, 3)
          THEN store_tagging
        ELSE 'not applicable'
        END AS loan_product_type,
      coalesce(
        (
          CASE
            WHEN lower(r.osType) LIKE '%andro%' THEN 'android'
            WHEN lower(r.osType) LIKE '%os%' THEN 'ios'
            ELSE lower(r.osType)
            END),
        (
          CASE
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%andro%'
              THEN 'android'
            WHEN
              lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion))
              LIKE '%os%'
              THEN 'ios'
            WHEN lower(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
            ELSE 'ios'
            END)) AS osType,
      coalesce(sd.risk_segment, 'NA') risk_segment,
      coalesce(frs.risk_segment_final, 'NA') risk_segment_final
    FROM modelname r
    LEFT JOIN risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN deliquency del
      ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN
      (
        SELECT DISTINCT
          mer_refferal_code, mer_name mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN worktable_datachampions.TARGET_SPLIT P
          ON P.STORE_NAME = mer_name
        QUALIFY
          row_number()
            OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC)
          = 1
      ) sil_category
      ON loanmaster.purpleKey = sil_category.mer_refferal_code
    LEFT JOIN segmentdata sd
      ON sd.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    LEFT JOIN
      (
        SELECT digitalLoanAccountid, risk_segment_final
        FROM prj-prod-dataplatform.dl_loans_db_raw.tdbk_loan_poi3_response
        WHERE risk_segment_final IS NOT NULL
        QUALIFY
          row_number()
            OVER (PARTITION BY digitalLoanAccountid ORDER BY created_dt DESC)
          = 1
      ) frs
      ON frs.digitalLoanAccountId = loanmaster.digitalLoanAccountId
    WHERE
      loanmaster.flagDisbursement = 1
      AND loanmaster.disbursementDateTime IS NOT NULL
      AND r.Beta_Cash_Demo_Score IS NOT NULL
      AND del.flg_mature_fstpd_30 = 1
  )
SELECT *
FROM base
QUALIFY
  row_number()
    OVER (
      PARTITION BY digitalLoanAccountId, modelVersionId
      ORDER BY appln_submit_datetime
    )
  = 1;

  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    risk_segment_column='risk_segment',
    risk_segment_final_column='risk_segment_final',
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_betademocash = fact_table.copy()
df_d_fstpd30_betademocash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fstpd30_betademocash, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factbetademocash = pd.concat([df_f_fpd0_betademocash, df_f_fpd10_betademocash, df_f_fpd30_betademocash, df_f_fspd30_betademocash, df_f_fstpd30_betademocash], ignore_index=True)
dimbetademocash = pd.concat([df_d_fpd0_betademocash, df_d_fpd10_betademocash, df_d_fpd30_betademocash, df_d_fspd30_betademocash, df_d_fstpd30_betademocash], ignore_index=True)


print("beta_demo_model_cash gini calculation completed")


# %% [markdown]
# # 🪦💀 Graveyard

# %% [markdown]
# ### calculate_periodic_gini_prod_ver_trench_dimfact

# %%
# import pandas as pd
# import numpy as np
# from itertools import combinations
# from datetime import timedelta


# def calculate_gini(scores, labels):
#     """
#     Calculate Gini coefficient using ROC AUC score.
#     Gini = 2 * AUC - 1

#     Returns np.nan when:
#     - Fewer than 2 observations
#     - All labels are the same class (no variation in labels)
#     - AUC calculation fails

#     IMPORTANT: Labels do NOT need to be 0/1. This function correctly handles
#     any binary encoding: 0/1, 1/2, True/False, etc.
#     roc_auc_score only requires exactly 2 distinct classes to be present.
#     """
#     n = len(scores)
#     if n < 2:
#         return np.nan

#     label_sum = np.sum(labels)

#     # Handle case where no positive labels exist (all zeros)
#     # This prevents division by zero warning
#     if label_sum == 0:
#         return np.nan

#     try:
#         auc = roc_auc_score(labels, scores)
#         return 2 * auc - 1
#     except Exception:
#         return np.nan


# def calculate_periodic_gini_prod_ver_trench_dimfact(
#     df,
#     score_column,
#     label_column,
#     namecolumn,
#     data_selection_column=None,
#     model_version_column=None,
#     trench_column=None,
#     loan_type_column=None,
#     loan_product_type_column=None,
#     ostype_column=None,
#     apptype_column=None,
#     risk_segment_column=None,
#     risk_segment_final_column=None,
#     account_id_column=None,
# ):
#     """
#     Calculate periodic Gini coefficients and return Power BI-friendly long format
#     with fact and dimension tables.

#     Returns:
#     - fact_table: Long format with one row per segment per period
#     - dimension_table: Unique segment combinations for filtering

#     Parameters:
#     df: DataFrame with disbursement dates and score/label columns
#     score_column: name of the score column
#     label_column: name of the label column
#     namecolumn: name for the bad rate label
#     data_selection_column: (optional) name of column for data selection (Test/Train)
#     model_version_column: (optional) name of column for model version
#     trench_column: (optional) name of column for trench category
#     loan_type_column: (optional) name of loan type column
#     loan_product_type_column: (optional) name of loan product type column
#     ostype_column: (optional) name of column for OS type
#     account_id_column: (optional) name of column for distinct account IDs
#     """
#     # Input validation
#     required_columns = ["disbursementdate", score_column, label_column]
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError(f"Missing required columns. Need: {required_columns}")

#     optional_columns = {
#         "data_selection": data_selection_column,
#         "model_version": model_version_column,
#         "trench": trench_column,
#         "loan_type": loan_type_column,
#         "loan_product_type": loan_product_type_column,
#         "ostype": ostype_column,
#         "apptype": apptype_column,
#         "risk_segment": risk_segment_column,
#         "risk_segment_final": risk_segment_final_column,
#         "account_id": account_id_column,
#     }

#     for col_name, col in optional_columns.items():
#         if col and col not in df.columns:
#             raise ValueError(
#                 f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe"
#             )

#     # Create a copy to avoid modifying original dataframe
#     df = df.copy()

#     # Ensure date is datetime type
#     df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])

#     # Ensure score and label columns are numeric
#     df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
#     df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

#     # Drop rows with invalid values
#     df = df.dropna(subset=[score_column, label_column])

#     # Define list of datasets to process
#     datasets_to_process = [("Overall", df, {})]

#     # Create list of available segment columns
#     segment_columns = []
#     if data_selection_column:
#         segment_columns.append(("DataSelection", data_selection_column))
#     if model_version_column:
#         segment_columns.append(("ModelVersion", model_version_column))
#     if trench_column:
#         segment_columns.append(("Trench", trench_column))
#     if loan_type_column:
#         segment_columns.append(("LoanType", loan_type_column))
#     if loan_product_type_column:
#         segment_columns.append(("ProductType", loan_product_type_column))
#     if ostype_column:
#         segment_columns.append(("OSType", ostype_column))
#     if apptype_column:
#         segment_columns.append(("apptype", apptype_column))
#     if risk_segment_column:
#         segment_columns.append(("risk_segment", risk_segment_column))
#     if risk_segment_final_column:
#         segment_columns.append(("risk_segment_final", risk_segment_final_column))

#     # Generate all possible combinations of segment columns
#     for r in range(1, len(segment_columns) + 1):
#         for combo in combinations(segment_columns, r):

#             def generate_combinations(
#                 df, segment_columns, index=0, current_filter=None, current_name=""
#             ):
#                 if current_filter is None:
#                     current_filter = {}

#                 if index >= len(segment_columns):
#                     filtered_df = df
#                     for col, val in current_filter.items():
#                         filtered_df = filtered_df[filtered_df[col] == val]

#                     if len(filtered_df) > 0:
#                         yield (
#                             current_name.strip("_"),
#                             filtered_df,
#                             current_filter.copy(),
#                         )
#                     return

#                 seg_name, seg_col = segment_columns[index]
#                 for seg_value in sorted(df[seg_col].dropna().unique()):
#                     new_filter = current_filter.copy()
#                     new_filter[seg_col] = seg_value
#                     new_name = current_name + f"{seg_name}_{seg_value}_"

#                     yield from generate_combinations(
#                         df, segment_columns, index + 1, new_filter, new_name
#                     )

#             for combo_name, combo_df, combo_metadata in generate_combinations(
#                 df, list(combo)
#             ):
#                 datasets_to_process.append((combo_name, combo_df, combo_metadata))

#     all_results = []

#     # Process each dataset
#     for dataset_name, dataset_df, metadata in datasets_to_process:
#         # Calculate weekly Gini
#         dataset_df_copy = dataset_df.copy()
#         dataset_df_copy["week"] = dataset_df_copy["disbursementdate"].dt.to_period("W")
#         weekly_gini = (
#             dataset_df_copy.groupby("week")
#             .apply(
#                 lambda x: (
#                     calculate_gini(x[score_column], x[label_column])
#                     # if len(x) >= 10
#                     # else np.nan
#                 )
#             )
#             .reset_index(name="gini_value")
#         )
#         weekly_gini["period"] = "Week"
#         weekly_gini["start_date"] = weekly_gini["week"].apply(
#             lambda x: x.to_timestamp()
#         )
#         weekly_gini["end_date"] = weekly_gini["start_date"] + timedelta(days=6)

#         # Add distinct account count for weekly
#         if account_id_column:
#             weekly_account_counts = (
#                 dataset_df_copy.groupby("week")[account_id_column]
#                 .nunique()
#                 .reset_index()
#             )
#             weekly_account_counts.columns = ["week", "distinct_accounts"]
#             weekly_gini = weekly_gini.merge(
#                 weekly_account_counts, on="week", how="left"
#             )
#         else:
#             weekly_gini["distinct_accounts"] = None

#         # Add bad count for weekly (distinct accounts where label == 1)
#         if account_id_column:
#             weekly_bad_counts = (
#                 dataset_df_copy[dataset_df_copy[label_column] == 1]
#                 .groupby("week")[account_id_column]
#                 .nunique()
#                 .reset_index()
#             )
#             weekly_bad_counts.columns = ["week", "bad_count"]
#             weekly_gini = weekly_gini.merge(
#                 weekly_bad_counts, on="week", how="left"
#             )
#             weekly_gini["bad_count"] = weekly_gini["bad_count"].fillna(0).astype(int)
#         else:
#             weekly_gini["bad_count"] = None

#         weekly_gini = weekly_gini[
#             ["start_date", "end_date", "gini_value", "period", "distinct_accounts", "bad_count"]
#         ]

#         # Calculate monthly Gini
#         dataset_df_copy = dataset_df.copy()
#         dataset_df_copy["month"] = dataset_df_copy["disbursementdate"].dt.to_period("M")
#         monthly_gini = (
#             dataset_df_copy.groupby("month")
#             .apply(
#                 lambda x: (
#                     calculate_gini(x[score_column], x[label_column])
#                     # if len(x) >= 20
#                     # else np.nan
#                 )
#             )
#             .reset_index(name="gini_value")
#         )
#         monthly_gini["period"] = "Month"
#         monthly_gini["start_date"] = monthly_gini["month"].apply(
#             lambda x: x.to_timestamp()
#         )
#         monthly_gini["end_date"] = (
#             monthly_gini["start_date"] + pd.DateOffset(months=1) - pd.Timedelta(days=1)
#         )

#         # Add distinct account count for monthly
#         if account_id_column:
#             monthly_account_counts = (
#                 dataset_df_copy.groupby("month")[account_id_column]
#                 .nunique()
#                 .reset_index()
#             )
#             monthly_account_counts.columns = ["month", "distinct_accounts"]
#             monthly_gini = monthly_gini.merge(
#                 monthly_account_counts, on="month", how="left"
#             )
#         else:
#             monthly_gini["distinct_accounts"] = None

#         # Add bad count for monthly (distinct accounts where label == 1)
#         if account_id_column:
#             monthly_bad_counts = (
#                 dataset_df_copy[dataset_df_copy[label_column] == 1]
#                 .groupby("month")[account_id_column]
#                 .nunique()
#                 .reset_index()
#             )
#             monthly_bad_counts.columns = ["month", "bad_count"]
#             monthly_gini = monthly_gini.merge(
#                 monthly_bad_counts, on="month", how="left"
#             )
#             monthly_gini["bad_count"] = monthly_gini["bad_count"].fillna(0).astype(int)
#         else:
#             monthly_gini["bad_count"] = None

#         monthly_gini = monthly_gini[
#             ["start_date", "end_date", "gini_value", "period", "distinct_accounts", "bad_count"]
#         ]

#         # Combine results for this dataset
#         gini_results = pd.concat([weekly_gini, monthly_gini], ignore_index=True)
#         gini_results = gini_results.sort_values(by="start_date").reset_index(drop=True)

#         # Add metadata columns
#         gini_results["Model_Name"] = score_column
#         gini_results["bad_rate"] = namecolumn
#         gini_results["segment_type"] = dataset_name

#         # Add individual segment components
#         gini_results["data_selection"] = (
#             metadata.get(data_selection_column, None) if data_selection_column else None
#         )
#         gini_results["model_version"] = (
#             metadata.get(model_version_column, None) if model_version_column else None
#         )
#         gini_results["trench_category"] = (
#             metadata.get(trench_column, None) if trench_column else None
#         )
#         gini_results["loan_type"] = (
#             metadata.get(loan_type_column, None) if loan_type_column else None
#         )
#         gini_results["loan_product_type"] = (
#             metadata.get(loan_product_type_column, None)
#             if loan_product_type_column
#             else None
#         )
#         gini_results["ostype"] = (
#             metadata.get(ostype_column, None) if ostype_column else None
#         )
#         gini_results["apptype"] = (metadata.get(apptype_column, None) if apptype_column else None
#         )
#         gini_results["risk_segment"] = metadata.get(risk_segment_column, None) if risk_segment_column else None
#         gini_results["risk_segment_final"] = metadata.get(risk_segment_final_column, None) if risk_segment_final_column else None   

#         all_results.append(gini_results)

#     # Combine all results
#     fact_table = pd.concat(all_results, ignore_index=True)

#     # Create dimension table (unique segment combinations for filtering)
#     dimension_table = (
#         fact_table[
#             [
#                 "Model_Name",
#                 "bad_rate",
#                 "segment_type",
#                 "data_selection",
#                 "model_version",
#                 "trench_category",
#                 "loan_type",
#                 "loan_product_type",
#                 "ostype",
#                 "apptype",
#                 "risk_segment", 
#                 "risk_segment_final",
#             ]
#         ]
#         .drop_duplicates()
#         .reset_index(drop=True)
#     )
#     dimension_table["segment_id"] = range(len(dimension_table))

#     # Add segment_id to fact table
#     fact_table = fact_table.merge(
#         dimension_table[
#             [
#                 "segment_id",
#                 "Model_Name",
#                 "bad_rate",
#                 "segment_type",
#                 "data_selection",
#                 "model_version",
#                 "trench_category",
#                 "loan_type",
#                 "loan_product_type",
#                 "ostype",
#                 "apptype",
#                 "risk_segment", 
#                 "risk_segment_final",
#             ]
#         ],
#         on=[
#             "Model_Name",
#             "bad_rate",
#             "segment_type",
#             "data_selection",
#             "model_version",
#             "trench_category",
#             "loan_type",
#             "loan_product_type",
#             "ostype",
#             "apptype",
#             "risk_segment", 
#             "risk_segment_final",
#         ],
#         how="left",
#     )

#     # Reorder columns in fact table
#     fact_table = fact_table[
#         [
#             "segment_id",
#             "start_date",
#             "end_date",
#             "period",
#             "gini_value",
#             "distinct_accounts",
#             "bad_count",
#             "Model_Name",
#             "bad_rate",
#             "segment_type",
#             "data_selection",
#             "model_version",
#             "trench_category",
#             "loan_type",
#             "loan_product_type",
#             "ostype",
#             "apptype",
#             "risk_segment", 
#             "risk_segment_final"
#         ]
#     ]

#     # Reorder columns in dimension table
#     dimension_table = dimension_table[
#         [
#             "segment_id",
#             "Model_Name",
#             "bad_rate",
#             "segment_type",
#             "data_selection",
#             "model_version",
#             "trench_category",
#             "loan_type",
#             "loan_product_type",
#             "ostype",
#             "apptype",
#             "risk_segment", 
#             "risk_segment_final"
#         ]
#     ]

#     return fact_table, dimension_table

# %% [markdown]
# # 🔚 End

# %% [markdown]
# 


