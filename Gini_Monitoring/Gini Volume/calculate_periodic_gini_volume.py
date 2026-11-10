import warnings

import numpy as np
import pandas as pd
from itertools import combinations


def calculate_gini_volume_fast(scores, bad_vol, good_vol):
    """
    Volume-weighted Gini via a weighted Mann-Whitney AUC.

    Every account carries two masses instead of one 0/1 label:
        bad_vol  -> its bad volume   (e.g. outstanding principal at the due date if defaulted)
        good_vol -> its good volume  (total_volume - bad_volume)
    AUC = P(a random unit of BAD volume has a higher score than a random unit of
    GOOD volume), ties counted as 0.5. Gini = 2 * AUC - 1.

    With bad_vol = label (0/1) and good_vol = 1 - label this reduces exactly to
    calculate_gini_fast (the count-based version).

    Returns np.nan for fewer than 2 rows, or when bad volume or good volume sums to zero.
    """
    n = scores.shape[0]
    if n < 2:
        return np.nan
    total_bad = bad_vol.sum()
    total_good = good_vol.sum()
    if total_bad <= 0 or total_good <= 0:
        return np.nan

    # collapse to unique score values; tied scores share one bucket
    _, inv = np.unique(scores, return_inverse=True)
    bad_u = np.bincount(inv, weights=bad_vol)
    good_u = np.bincount(inv, weights=good_vol)

    # good volume strictly below each score + half of the good volume tied at it
    good_below = np.cumsum(good_u) - good_u
    auc = (bad_u * (good_below + 0.5 * good_u)).sum() / (total_bad * total_good)
    return 2.0 * auc - 1.0


def calculate_periodic_gini_prod_ver_trench_dimfact_volume(
    df,
    score_column,
    bad_volume_column,
    total_volume_column,
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
    bad_flag_column=None,
):
    """
    Volume-based counterpart of calculate_periodic_gini_prod_ver_trench_dimfact.

    Segmentation logic is unchanged: same power set of segment columns, same
    weekly/monthly grains, same segment_type naming, same fact/dimension
    structure and segment_id assignment. What changes is the Gini itself:

    - bad_volume_column   : per-account bad volume   (e.g. deffpd0_v)
    - total_volume_column : per-account total volume (e.g. flg_mature_fpd0_v)
    - good volume = total_volume - bad_volume (floored at 0)
    - gini_value is the volume-weighted Gini (see calculate_gini_volume_fast)

    Extra fact-table columns: bad_volume and total_volume (sums per segment/period),
    so bad rate by volume = bad_volume / total_volume.

    distinct_accounts / bad_count keep their count-based meaning. If
    bad_flag_column (e.g. 'deffpd0') is given, bad_count = distinct accounts with
    flag == 1 (matches the count-based tables); otherwise bad_count = distinct
    accounts with bad_volume > 0.
    """
    required_columns = ["disbursementdate", score_column, bad_volume_column, total_volume_column]
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
        "bad_flag": bad_flag_column,
    }
    for col_name, col in optional_columns.items():
        if col and col not in df.columns:
            raise ValueError(
                f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe"
            )

    df = df.copy()
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[bad_volume_column] = pd.to_numeric(df[bad_volume_column], errors="coerce")
    df[total_volume_column] = pd.to_numeric(df[total_volume_column], errors="coerce")

    rows_before = len(df)
    df = df.dropna(subset=[score_column, bad_volume_column, total_volume_column])
    n_dropped = rows_before - len(df)
    if n_dropped:
        warnings.warn(
            f"{n_dropped} row(s) dropped for NULL score / bad volume / total volume "
            f"(e.g. a defaulted loan with no outstanding-principal record)."
        )

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
    bad_arr = np.clip(df[bad_volume_column].to_numpy(dtype=float), 0.0, None)
    total_arr = np.clip(df[total_volume_column].to_numpy(dtype=float), 0.0, None)
    n_over = int((bad_arr > total_arr).sum())
    if n_over:
        warnings.warn(
            f"{n_over} row(s) have bad volume > total volume; their good volume is floored at 0."
        )
    good_arr = np.clip(total_arr - bad_arr, 0.0, None)

    accounts_arr = df[account_id_column].to_numpy() if account_id_column else None
    if account_id_column:
        if bad_flag_column:
            bad_acct_arr = pd.to_numeric(df[bad_flag_column], errors="coerce").to_numpy() == 1
        else:
            bad_acct_arr = bad_arr > 0

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
                seg_bad = bad_arr[idx]
                seg_good = good_arr[idx]
                gini_value = calculate_gini_volume_fast(seg_scores, seg_bad, seg_good)

                bad_volume = float(seg_bad.sum())
                total_volume = float(total_arr[idx].sum())

                if account_id_column:
                    seg_accounts = accounts_arr[idx]
                    distinct_accounts = pd.unique(seg_accounts).shape[0]
                    bad_mask = bad_acct_arr[idx]
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
                    "bad_volume": bad_volume,
                    "total_volume": total_volume,
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
         "bad_count", "bad_volume", "total_volume", "Model_Name", "bad_rate", "segment_type"]
        + all_output_fields
    ]
    dimension_table = dimension_table[
        ["segment_id", "Model_Name", "bad_rate", "segment_type"] + all_output_fields
    ]

    return fact_table, dimension_table
