import ast
import json
import pandas as pd
import numpy as np
import re
from typing import Union, Optional
from config_old import Config

def safe_parse(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or value.strip() == "":
        return {}

    # Try JSON first (safer)
    try:
        return json.loads(value)
    except:
        pass

    # Fallback to literal_eval — replace 'nan' with 'None' since nan is not a Python literal
    try:
        import re as _re
        cleaned = _re.sub(r'\bnan\b', 'None', value)
        return ast.literal_eval(cleaned)
    except:
        return {}

def normalize_calc_feature(x):
    if isinstance(x, list):
        result = {}
        for i in x:
            if isinstance(i, dict):
                result.update(i)
        return result
    elif isinstance(x, dict):
        return x
    return {}

def feat_col_exct(df, model_name=None, version=None):
    df = df.reset_index(drop=True)
    df['calcFeature'] = df['calcFeature'].apply(safe_parse)
    df['calcFeature'] = df['calcFeature'].apply(normalize_calc_feature)
    if model_name == 'apps_score_sil_model' and version == 'v2':
        df['appScoreModel'] = df['calcFeature'].apply(
            lambda x: x.get('appScoreModel', {}) if isinstance(x, dict) else {}
        )
        appScore_df = pd.json_normalize(df['appScoreModel'])
        df = pd.concat([df, appScore_df], axis=1)
    if model_name == 'beta_demo_sil_model' and version == 'v2':
        df['demoScoreModel'] = df['calcFeature'].apply(
            lambda x: x.get('demoScoreModel', {}) if isinstance(x, dict) else {}
        )
        demoScore_df = pd.json_normalize(df['demoScoreModel'])
        df = pd.concat([df, demoScore_df], axis=1)
    calcFeature_df = pd.json_normalize(df['calcFeature'])
    df = pd.concat([df.drop(columns=['calcFeature']), calcFeature_df], axis=1)
    return df

def clean_binned_values(series):
    def extract_numeric(val):
        try:
            if pd.isna(val):
                return np.nan

            if isinstance(val, (int, float)):
                return float(val)

            if isinstance(val, str):
                val = val.strip()

                # extract numeric part safely
                match = re.search(r"[-+]?\d*\.?\d+", val)
                if match:
                    return float(match.group())

            return np.nan
        
        except Exception:
            return np.nan  # fail-safe

    return series.apply(extract_numeric)

def is_mostly_numeric(series, threshold=0.9):
    numeric_series = pd.to_numeric(series, errors='coerce')
    non_null_ratio = numeric_series.notna().sum() / len(series)
    return non_null_ratio >= threshold

def calculate_psi(expected, actual, buckets=10, strategy='quantile'):
    psi_results = {}
    # Handle numeric features or categorical features separately
    if pd.api.types.is_numeric_dtype(expected):
        try:
            actual = pd.to_numeric(actual, errors='coerce')
            not_null_expected = expected.dropna()

            if strategy == 'quantile':
                bins = pd.qcut(not_null_expected, q=buckets, duplicates='drop')
            elif strategy == 'uniform':
                bins = pd.cut(not_null_expected, bins=buckets)
            else:
                raise ValueError("strategy must be 'quantile' or 'uniform'")

            bin_edges = bins.cat.categories

            expected_binned = pd.cut(expected, bins=bin_edges).astype(str)
            actual_binned = pd.cut(actual, bins=bin_edges).astype(str)

            # Handle NaNs as separate bin
            expected_binned[expected.isna()] = 'Missing'
            actual_binned[actual.isna()] = 'Missing'

            expected_dist = expected_binned.value_counts(normalize=True)
            actual_dist = actual_binned.value_counts(normalize=True)

        except Exception as e:
            psi_results["error"] = f'Error: {e}'
            return None

    # Handle categorical features
    else:
        expected_dist = expected.fillna('Missing').value_counts(normalize=True)
        actual_dist = actual.fillna('Missing').value_counts(normalize=True)

    # Align all category/bin labels
    all_bins = expected_dist.index.union(actual_dist.index)
    expected_dist = expected_dist.reindex(all_bins, fill_value=0.0001)
    actual_dist = actual_dist.reindex(all_bins, fill_value=0.0001)

    # Compute PSI
    psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
    
    return float(psi)

def calculate_stability_value(
    expected,
    actual,
    buckets=10,
    feature_type=None,   # None, "numeric", or "categorical"
    top_n_categories=None,
    epsilon=1e-6
):
    """
    Calculates PSI/CSI between expected and actual series.

    Returns:
        float
    """

    MISSING_BUCKET = "__MISSING__"
    OTHER_BUCKET = "__OTHER__"
    OTHER_NUMERIC_BUCKET = "__OTHER_NUMERIC_VALUE__"
    NON_MISSING_BUCKET = "__NON_MISSING__"

    expected = pd.Series(expected).copy()
    actual = pd.Series(actual).copy()

    def _clean_numeric(s):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def _fmt_value(x):
        if pd.isna(x):
            return "nan"
        x = float(x)
        if x.is_integer():
            return str(int(x))
        return f"{x:.10g}"

    def _make_numeric_bins(ref_s):
        x = _clean_numeric(ref_s)
        x_non_missing = x.dropna()

        if x_non_missing.empty:
            return {
                "type": "numeric_empty_reference",
                "all_bins": [NON_MISSING_BUCKET, MISSING_BUCKET],
            }

        unique_values = np.sort(x_non_missing.unique())
        n_unique = len(unique_values)

        if n_unique == 2:
            labels = [f"value={_fmt_value(v)}" for v in unique_values]
            return {
                "type": "numeric_binary",
                "values": unique_values,
                "labels": labels,
                "all_bins": [*labels, OTHER_NUMERIC_BUCKET, MISSING_BUCKET],
            }

        if n_unique == 1:
            edges = np.array([-np.inf, np.inf])
            labels = ["bin_00: (-inf, inf]"]
            return {
                "type": "numeric",
                "edges": edges,
                "labels": labels,
                "all_bins": [*labels, MISSING_BUCKET],
            }

        try:
            _, raw_edges = pd.qcut(
                x_non_missing,
                q=buckets,
                retbins=True,
                duplicates="drop"
            )

            raw_edges = np.unique(raw_edges[np.isfinite(raw_edges)])

            if len(raw_edges) <= 1:
                edges = np.array([-np.inf, np.inf])
            else:
                edges = np.r_[-np.inf, raw_edges[1:-1], np.inf]
                edges = np.unique(edges)

        except ValueError:
            edges = np.array([-np.inf, np.inf])

        labels = [
            f"bin_{i:02d}: ({edges[i]}, {edges[i + 1]}]"
            for i in range(len(edges) - 1)
        ]

        return {
            "type": "numeric",
            "edges": edges,
            "labels": labels,
            "all_bins": [*labels, MISSING_BUCKET],
        }

    def _apply_numeric_bins(s, config):
        x = _clean_numeric(s)

        if config["type"] == "numeric_empty_reference":
            return pd.Series(
                np.where(x.isna(), MISSING_BUCKET, NON_MISSING_BUCKET),
                index=s.index,
                dtype="object"
            )

        if config["type"] == "numeric_binary":
            out = pd.Series(OTHER_NUMERIC_BUCKET, index=s.index, dtype="object")

            for value, label in zip(config["values"], config["labels"]):
                out[x == value] = label

            out[x.isna()] = MISSING_BUCKET
            return out.astype(str)

        binned = pd.cut(
            x,
            bins=config["edges"],
            labels=config["labels"],
            include_lowest=True,
            right=True
        )

        out = binned.astype("object")
        out[x.isna()] = MISSING_BUCKET

        return out.astype(str)

    def _make_categorical_bins(ref_s):
        x = ref_s.astype("string")
        x_non_missing = x[x.notna()]

        counts = x_non_missing.value_counts(dropna=True)

        if top_n_categories is None:
            categories = counts.index.astype(str).tolist()
        else:
            categories = counts.head(top_n_categories).index.astype(str).tolist()

        return {
            "type": "categorical",
            "categories": categories,
            "all_bins": [*categories, OTHER_BUCKET, MISSING_BUCKET],
        }

    def _apply_categorical_bins(s, config):
        x = s.astype("string")
        categories = set(config["categories"])

        out = x.astype("object")
        missing_mask = x.isna()

        out = out.where(x.isin(categories), OTHER_BUCKET)
        out[missing_mask] = MISSING_BUCKET

        return out.astype(str)

    if len(expected) == 0:
        raise ValueError("expected series is empty.")

    if len(actual) == 0:
        raise ValueError("actual series is empty.")

    if feature_type is None:
        is_numeric = pd.api.types.is_numeric_dtype(expected)
    else:
        is_numeric = feature_type.lower() in ["numeric", "float", "float64", "int", "int64"]

    if is_numeric:
        config = _make_numeric_bins(expected)
        expected_buckets = _apply_numeric_bins(expected, config)
        actual_buckets = _apply_numeric_bins(actual, config)
    else:
        config = _make_categorical_bins(expected)
        expected_buckets = _apply_categorical_bins(expected, config)
        actual_buckets = _apply_categorical_bins(actual, config)

    expected_count = (
        expected_buckets
        .value_counts(dropna=False)
        .reindex(config["all_bins"], fill_value=0)
        .astype(float)
    )

    actual_count = (
        actual_buckets
        .value_counts(dropna=False)
        .reindex(config["all_bins"], fill_value=0)
        .astype(float)
    )

    if expected_count.sum() == 0:
        raise ValueError("expected bucket count is zero.")

    if actual_count.sum() == 0:
        raise ValueError("actual bucket count is zero.")

    expected_pct = expected_count / expected_count.sum()
    actual_pct = actual_count / actual_count.sum()

    expected_pct_calc = expected_pct.replace(0, epsilon)
    actual_pct_calc = actual_pct.replace(0, epsilon)

    value = np.sum(
        (actual_pct_calc - expected_pct_calc)
        * np.log(actual_pct_calc / expected_pct_calc)
    )

    return float(value)


def calculate_bins_deciles(df, col):
    # Handle NaN values before applying binning
    df = df.dropna(subset=[col])  # Drop rows where the value in the column is NaN
    
    if pd.api.types.is_numeric_dtype(df[col]):
        # For numeric columns, use quantile or uniform binning
        try:
            # Using qcut for quantile-based binning (deciles)
            bins = pd.qcut(df[col], 10, labels=False, duplicates='drop')  # Deciles (10 bins)
            deciles = len(pd.unique(bins))  # Number of deciles used
            
            # For uniform binning
            uniform_bins = pd.cut(df[col], bins=10, labels=False, include_lowest=True)
            bins_used = len(pd.unique(uniform_bins))  # Unique bins used for uniform binning
            return deciles, bins_used
        
        except ValueError as e:
            # print(f"Error in binning column {col}: {e}")
            return None, None
    
    else:
        # For categorical columns, create bins based on value counts
        value_counts = df[col].value_counts(normalize=True)
        bins = value_counts.index.tolist()
        deciles = len(bins)
        bins_used = len(bins)  
        return deciles, bins_used

def get_calcfeature_config(model_config, mode):
    """
    Returns the calculation feature configuration for a given model configuration and mode (train or prod).
    """
    cf = model_config.get('calcfeature_config', {})

    if not cf:
        return {'base_path': '', 'features': []}

    if 'common' in cf:
        return cf['common']

    return cf.get(mode, {'base_path': '', 'features': []})


def build_calcfeature_sql(calcfeature_config, column_name="q1.calcFeature"):
    """
    Builds SQL SELECT expressions and WHERE conditions based on calcfeature_config.

    Supports feature_types like:
        string, int64, float64, numeric, bignumeric, bool / boolean

    Returns:
        select_expr_str, where_conditions_str
    """

    base_path = calcfeature_config.get("base_path", "")
    features = calcfeature_config.get("features", [])
    feature_types = calcfeature_config.get("feature_types", {})

    if not features:
        return "", ""

    type_mapping = {
        "string": "STRING",
        "str": "STRING",

        "int": "INT64",
        "int64": "INT64",
        "integer": "INT64",

        "float": "FLOAT64",
        "float64": "FLOAT64",
        "double": "FLOAT64",

        "numeric": "NUMERIC",
        "bignumeric": "BIGNUMERIC",

        "bool": "BOOL",
        "boolean": "BOOL",
    }

    expressions = []
    where_conditions = []

    for feat in features:
        raw_feature_type = feature_types.get(feat, "string")
        feature_type_key = str(raw_feature_type).strip().lower()

        bigquery_type = type_mapping.get(feature_type_key)

        if bigquery_type is None:
            raise ValueError(
                f"Unsupported feature type '{raw_feature_type}' for feature '{feat}'"
            )

        if base_path:
            json_path = f"{base_path}.{feat}"
        else:
            json_path = f"$.{feat}"

        json_value_expr = f"JSON_VALUE({column_name}, '{json_path}')"

        where_conditions.append(
            f"{json_value_expr} IS NOT NULL "
            f"AND {json_value_expr} != ''"
        )

        if bigquery_type == "STRING":
            expressions.append(
                f"{json_value_expr} AS {feat}"
            )
        else:
            expressions.append(
                f"SAFE_CAST({json_value_expr} AS {bigquery_type}) AS {feat}"
            )

    select_sql = ",\n                ".join(expressions)
    where_sql = " AND ".join(where_conditions)

    return select_sql, where_sql



def build_prod_query(version, where_condition_sql, calcfeature_select, prod_start_date, prod_end_date, model_name=None):
    """
    Builds the production data SQL query against audit_balance.ml_model_run_details.
    calcfeature_where is appended to the WHERE clause if non-empty.
    model_name controls how prediction is parsed.
    """
    # Conditional prediction logic
    if model_name == "apps_score_sil_model":
        prediction_expr = "CAST(JSON_VALUE(q1.prediction, '$.combined_score') AS FLOAT64)"
        query = f"""
        WITH query2_base AS (
            SELECT
                DISTINCT
                r.customerId,
                r.digitalLoanAccountId,
                loanmaster.new_loan_type,
                CASE
                    WHEN loanmaster.loantype = 'BNPL' AND sil_category.store_type = 1 THEN 'Appliance'
                    WHEN loanmaster.loantype = 'BNPL' AND sil_category.store_type = 2 THEN 'Mobile'
                    WHEN loanmaster.loantype = 'BNPL' AND sil_category.store_type = 3 THEN 'Mall'
                    WHEN loanmaster.loantype = 'BNPL' AND sil_category.store_type NOT IN (1,2,3) THEN sil_category.store_tagging
                    ELSE 'not applicable'
                END AS loan_product_type
            FROM
                `prj-prod-dataplatform.audit_balance.ml_model_run_details` r
            LEFT JOIN
                `prj-prod-dataplatform.risk_credit_mis.loan_master_table` loanmaster
                ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
            LEFT JOIN
                (
                    SELECT DISTINCT
                        mer_refferal_code,
                        mer_name,
                        store_type,
                        store_tagging
                    FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
                    LEFT JOIN worktable_datachampions.TARGET_SPLIT P
                        ON P.STORE_NAME = mer_name
                    QUALIFY ROW_NUMBER() OVER(PARTITION BY mer_refferal_code ORDER BY created_dt DESC) = 1
                ) sil_category
                ON loanmaster.purpleKey = sil_category.mer_refferal_code
            WHERE
                r.modelDisplayName IN ({where_condition_sql})
        )
        SELECT
            DISTINCT
            q1.digitalLoanAccountId,
            q1.customerId,
            q1.modelDisplayName,
            q1.modelVersionId,
            CASE
                WHEN q1.trenchCategory IS NULL THEN (
                    CASE
                        WHEN mt.ln_user_type = '1_Repeat Applicant' THEN 'Trench 3'
                        WHEN mt.ln_user_type <> '1_Repeat Applicant'
                             AND DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) > 30 THEN 'Trench 2'
                        ELSE 'Trench1'
                    END
                )
                WHEN q1.trenchCategory = '' THEN (
                    CASE
                        WHEN mt.ln_user_type = '1_Repeat Applicant' THEN 'Trench 3'
                        WHEN mt.ln_user_type <> '1_Repeat Applicant'
                             AND DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) > 30 THEN 'Trench 2'
                        ELSE 'Trench 1'
                    END
                )
                ELSE q1.trenchCategory
            END AS trenchCategory,
            q1.deviceOs,
            {calcfeature_select},
            CAST(q1.calcFeature AS STRING) AS calcFeature,
            {prediction_expr} AS prediction,
            q2.new_loan_type AS loan_type,
            q2.loan_product_type AS product_type,
            DATE(q1.publish_time) AS publish_time
        FROM
            `prj-prod-dataplatform.audit_balance.ml_model_run_details` q1
        LEFT JOIN query2_base q2 ON q1.digitalLoanAccountId = q2.digitalLoanAccountId
        LEFT JOIN `prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart` mt
            ON mt.digitalLoanAccountId = q1.digitalLoanAccountId
        WHERE
            q1.modelDisplayName IN ({where_condition_sql})
            AND q1.modelVersionId = '{version}'
            AND DATE(q1.publish_time) BETWEEN DATE('{prod_start_date}') AND DATE('{prod_end_date}');"""
    else:
        prediction_expr = "CAST(q1.prediction AS FLOAT64)"
        query = f"""
        WITH query2_base AS (
            SELECT
                DISTINCT
                r.customerId,
                r.digitalLoanAccountId,
                loanmaster.new_loan_type,
                CASE
                    WHEN loanmaster.loantype = 'BNPL' AND sil_category.store_type = 1 THEN 'Appliance'
                    WHEN loanmaster.loantype = 'BNPL' AND sil_category.store_type = 2 THEN 'Mobile'
                    WHEN loanmaster.loantype = 'BNPL' AND sil_category.store_type = 3 THEN 'Mall'
                    WHEN loanmaster.loantype = 'BNPL' AND sil_category.store_type NOT IN (1,2,3) THEN sil_category.store_tagging
                    ELSE 'not applicable'
                END AS loan_product_type
            FROM
                `prj-prod-dataplatform.audit_balance.ml_model_run_details` r
            LEFT JOIN
                `prj-prod-dataplatform.risk_credit_mis.loan_master_table` loanmaster
                ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
            LEFT JOIN
                (
                    SELECT DISTINCT
                        mer_refferal_code,
                        mer_name,
                        store_type,
                        store_tagging
                    FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
                    LEFT JOIN worktable_datachampions.TARGET_SPLIT P
                        ON P.STORE_NAME = mer_name
                    QUALIFY ROW_NUMBER() OVER(PARTITION BY mer_refferal_code ORDER BY created_dt DESC) = 1
                ) sil_category
                ON loanmaster.purpleKey = sil_category.mer_refferal_code
            WHERE
                r.modelDisplayName IN ({where_condition_sql})
        )
        SELECT
            DISTINCT
            q1.digitalLoanAccountId,
            q1.customerId,
            q1.modelDisplayName,
            q1.modelVersionId,
            CASE
                WHEN q1.trenchCategory IS NULL THEN (
                    CASE
                        WHEN mt.ln_user_type = '1_Repeat Applicant' THEN 'Trench 3'
                        WHEN mt.ln_user_type <> '1_Repeat Applicant'
                             AND DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) > 30 THEN 'Trench 2'
                        ELSE 'Trench1'
                    END
                )
                WHEN q1.trenchCategory = '' THEN (
                    CASE
                        WHEN mt.ln_user_type = '1_Repeat Applicant' THEN 'Trench 3'
                        WHEN mt.ln_user_type <> '1_Repeat Applicant'
                             AND DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) > 30 THEN 'Trench 2'
                        ELSE 'Trench 1'
                    END
                )
                ELSE q1.trenchCategory
            END AS trenchCategory,
            q1.deviceOs,
            {calcfeature_select},
            {prediction_expr} AS prediction,
            q2.new_loan_type AS loan_type,
            q2.loan_product_type AS product_type,
            DATE(q1.publish_time) AS publish_time
        FROM
            `prj-prod-dataplatform.audit_balance.ml_model_run_details` q1
        LEFT JOIN query2_base q2 ON q1.digitalLoanAccountId = q2.digitalLoanAccountId
        LEFT JOIN `prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart` mt
            ON mt.digitalLoanAccountId = q1.digitalLoanAccountId
        WHERE
            q1.modelDisplayName IN ({where_condition_sql})
            AND q1.modelVersionId = '{version}'
            AND DATE(q1.publish_time) BETWEEN DATE('{prod_start_date}') AND DATE('{prod_end_date}');"""

    return query


def get_prediction_value(x, key=None):
                    if isinstance(x, str):
                        try:
                            x = ast.literal_eval(x)
                        except Exception:
                            return None

                    if isinstance(x, dict):
                        return x.get(key) if key else x

                    return x