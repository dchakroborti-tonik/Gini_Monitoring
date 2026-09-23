# %% [markdown]
# # SIL Train Gini Calculation

# %% [markdown]
# ## Define Library

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
# ## Functions

# %% [markdown]
# ### calculate_periodic_gini_prod_ver_trench_dimfact

# %%
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import timedelta
from scipy.stats import rankdata


def calculate_gini_fast(scores, labels):
    """
    Rank-based (Mann-Whitney) AUC -> Gini. Drop-in replacement for the
    previous roc_auc_score-based calculate_gini.

    Verified equivalent to `2 * roc_auc_score(labels, scores) - 1` to
    floating-point precision, but removes the sklearn dependency and the
    per-call validation overhead that dominates when this is invoked over
    many small weekly/monthly segment groups.

    Returns np.nan when:
    - Fewer than 2 observations
    - Labels do not contain exactly 2 distinct classes
    - Either class has zero members (guards both the all-positive and
      all-negative cases; the original SIL version only guarded the
      all-negative case)

    Labels do NOT need to be 0/1 — any binary encoding works (0/1, 1/2,
    True/False, etc.), matching the original function's contract.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

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
    account_id_column=None,
):
    """
    Calculate periodic Gini coefficients and return Power BI-friendly long format
    with fact and dimension tables.

    Returns:
    - fact_table: Long format with one row per segment per period
    - dimension_table: Unique segment combinations for filtering

    Parameters:
    df: DataFrame with disbursement dates and score/label columns
    score_column: name of the score column
    label_column: name of the label column
    namecolumn: name for the bad rate label
    data_selection_column: (optional) name of column for data selection (Test/Train)
    model_version_column: (optional) name of column for model version
    trench_column: (optional) name of column for trench category
    loan_type_column: (optional) name of loan type column
    loan_product_type_column: (optional) name of loan product type column
    ostype_column: (optional) name of column for OS type
    account_id_column: (optional) name of column for distinct account IDs
    """
    # Input validation
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
        "account_id": account_id_column,
    }

    for col_name, col in optional_columns.items():
        if col and col not in df.columns:
            raise ValueError(
                f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe"
            )

    # Create a copy to avoid modifying original dataframe
    df = df.copy()

    # Ensure date is datetime type
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])

    # Ensure score and label columns are numeric
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    # Drop rows with invalid values
    df = df.dropna(subset=[score_column, label_column])

    # Define list of datasets to process
    datasets_to_process = [("Overall", df, {})]

    # Create list of available segment columns
    segment_columns = []
    if data_selection_column:
        segment_columns.append(("DataSelection", data_selection_column))
    if model_version_column:
        segment_columns.append(("ModelVersion", model_version_column))
    if trench_column:
        segment_columns.append(("Trench", trench_column))
    if loan_type_column:
        segment_columns.append(("LoanType", loan_type_column))
    if loan_product_type_column:
        segment_columns.append(("ProductType", loan_product_type_column))
    if ostype_column:
        segment_columns.append(("OSType", ostype_column))
    if apptype_column:
        segment_columns.append(("apptype", apptype_column))

    # Generate all possible combinations of segment columns
    for r in range(1, len(segment_columns) + 1):
        for combo in combinations(segment_columns, r):

            def generate_combinations(
                df, segment_columns, index=0, current_filter=None, current_name=""
            ):
                if current_filter is None:
                    current_filter = {}

                if index >= len(segment_columns):
                    filtered_df = df
                    for col, val in current_filter.items():
                        filtered_df = filtered_df[filtered_df[col] == val]

                    if len(filtered_df) > 0:
                        yield (
                            current_name.strip("_"),
                            filtered_df,
                            current_filter.copy(),
                        )
                    return

                seg_name, seg_col = segment_columns[index]
                for seg_value in sorted(df[seg_col].dropna().unique()):
                    new_filter = current_filter.copy()
                    new_filter[seg_col] = seg_value
                    new_name = current_name + f"{seg_name}_{seg_value}_"

                    yield from generate_combinations(
                        df, segment_columns, index + 1, new_filter, new_name
                    )

            for combo_name, combo_df, combo_metadata in generate_combinations(
                df, list(combo)
            ):
                datasets_to_process.append((combo_name, combo_df, combo_metadata))

    all_results = []

    # Process each dataset
    for dataset_name, dataset_df, metadata in datasets_to_process:
        # Calculate weekly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["week"] = dataset_df_copy["disbursementdate"].dt.to_period("W")
        weekly_gini = (
            dataset_df_copy.groupby("week")
            .apply(
                lambda x: (
                    calculate_gini_fast(x[score_column], x[label_column])
                    # if len(x) >= 10
                    # else np.nan
                )
            )
            .reset_index(name="gini_value")
        )
        weekly_gini["period"] = "Week"
        weekly_gini["start_date"] = weekly_gini["week"].apply(
            lambda x: x.to_timestamp()
        )
        weekly_gini["end_date"] = weekly_gini["start_date"] + timedelta(days=6)

        # Add distinct account count for weekly
        if account_id_column:
            weekly_account_counts = (
                dataset_df_copy.groupby("week")[account_id_column]
                .nunique()
                .reset_index()
            )
            weekly_account_counts.columns = ["week", "distinct_accounts"]
            weekly_gini = weekly_gini.merge(
                weekly_account_counts, on="week", how="left"
            )
        else:
            weekly_gini["distinct_accounts"] = None

        # Add bad count for weekly (distinct accounts where label == 1)
        if account_id_column:
            weekly_bad_counts = (
                dataset_df_copy[dataset_df_copy[label_column] == 1]
                .groupby("week")[account_id_column]
                .nunique()
                .reset_index()
            )
            weekly_bad_counts.columns = ["week", "bad_count"]
            weekly_gini = weekly_gini.merge(
                weekly_bad_counts, on="week", how="left"
            )
            weekly_gini["bad_count"] = weekly_gini["bad_count"].fillna(0).astype(int)
        else:
            weekly_gini["bad_count"] = None

        weekly_gini = weekly_gini[
            ["start_date", "end_date", "gini_value", "period", "distinct_accounts", "bad_count"]
        ]

        # Calculate monthly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["month"] = dataset_df_copy["disbursementdate"].dt.to_period("M")
        monthly_gini = (
            dataset_df_copy.groupby("month")
            .apply(
                lambda x: (
                    calculate_gini_fast(x[score_column], x[label_column])
                    # if len(x) >= 20
                    # else np.nan
                )
            )
            .reset_index(name="gini_value")
        )
        monthly_gini["period"] = "Month"
        monthly_gini["start_date"] = monthly_gini["month"].apply(
            lambda x: x.to_timestamp()
        )
        monthly_gini["end_date"] = (
            monthly_gini["start_date"] + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        )

        # Add distinct account count for monthly
        if account_id_column:
            monthly_account_counts = (
                dataset_df_copy.groupby("month")[account_id_column]
                .nunique()
                .reset_index()
            )
            monthly_account_counts.columns = ["month", "distinct_accounts"]
            monthly_gini = monthly_gini.merge(
                monthly_account_counts, on="month", how="left"
            )
        else:
            monthly_gini["distinct_accounts"] = None

        # Add bad count for monthly (distinct accounts where label == 1)
        if account_id_column:
            monthly_bad_counts = (
                dataset_df_copy[dataset_df_copy[label_column] == 1]
                .groupby("month")[account_id_column]
                .nunique()
                .reset_index()
            )
            monthly_bad_counts.columns = ["month", "bad_count"]
            monthly_gini = monthly_gini.merge(
                monthly_bad_counts, on="month", how="left"
            )
            monthly_gini["bad_count"] = monthly_gini["bad_count"].fillna(0).astype(int)
        else:
            monthly_gini["bad_count"] = None

        monthly_gini = monthly_gini[
            ["start_date", "end_date", "gini_value", "period", "distinct_accounts", "bad_count"]
        ]

        # Combine results for this dataset
        gini_results = pd.concat([weekly_gini, monthly_gini], ignore_index=True)
        gini_results = gini_results.sort_values(by="start_date").reset_index(drop=True)

        # Add metadata columns
        gini_results["Model_Name"] = score_column
        gini_results["bad_rate"] = namecolumn
        gini_results["segment_type"] = dataset_name

        # Add individual segment components
        gini_results["data_selection"] = (
            metadata.get(data_selection_column, None) if data_selection_column else None
        )
        gini_results["model_version"] = (
            metadata.get(model_version_column, None) if model_version_column else None
        )
        gini_results["trench_category"] = (
            metadata.get(trench_column, None) if trench_column else None
        )
        gini_results["loan_type"] = (
            metadata.get(loan_type_column, None) if loan_type_column else None
        )
        gini_results["loan_product_type"] = (
            metadata.get(loan_product_type_column, None)
            if loan_product_type_column
            else None
        )
        gini_results["ostype"] = (
            metadata.get(ostype_column, None) if ostype_column else None
        )
        gini_results["apptype"] = (
            metadata.get(apptype_column, None) if apptype_column else None
        )

        all_results.append(gini_results)

    # Combine all results
    fact_table = pd.concat(all_results, ignore_index=True)

    # Create dimension table (unique segment combinations for filtering)
    dimension_table = (
        fact_table[
            [
                "Model_Name",
                "bad_rate",
                "segment_type",
                "data_selection",
                "model_version",
                "trench_category",
                "loan_type",
                "loan_product_type",
                "ostype",
                "apptype",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dimension_table["segment_id"] = range(len(dimension_table))

    # Add segment_id to fact table
    fact_table = fact_table.merge(
        dimension_table[
            [
                "segment_id",
                "Model_Name",
                "bad_rate",
                "segment_type",
                "data_selection",
                "model_version",
                "trench_category",
                "loan_type",
                "loan_product_type",
                "ostype",
                "apptype",
            ]
        ],
        on=[
            "Model_Name",
            "bad_rate",
            "segment_type",
            "data_selection",
            "model_version",
            "trench_category",
            "loan_type",
            "loan_product_type",
            "ostype",
            "apptype",
        ],
        how="left",
    )

    # Reorder columns in fact table
    fact_table = fact_table[
        [
            "segment_id",
            "start_date",
            "end_date",
            "period",
            "gini_value",
            "distinct_accounts",
            "bad_count",
            "Model_Name",
            "bad_rate",
            "segment_type",
            "data_selection",
            "model_version",
            "trench_category",
            "loan_type",
            "loan_product_type",
            "ostype",
            "apptype",
        ]
    ]

    # Reorder columns in dimension table
    dimension_table = dimension_table[
        [
            "segment_id",
            "Model_Name",
            "bad_rate",
            "segment_type",
            "data_selection",
            "model_version",
            "trench_category",
            "loan_type",
            "loan_product_type",
            "ostype",
            "apptype",
        ]
    ]

    return fact_table, dimension_table

# %%


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
# ### Models

# %% [markdown]
# #### Beta SIL Demo Score

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademosil_test2"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betademosil_test2"

# ### FPD0

# #### Test

# %%
sq = """
WITH cleaned AS (
    SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
deviceOs osType,
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
  osType,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    Application_type,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# ### Train

# # %%
# sq = """
# WITH cleaned AS (
#     SELECT
#   mmrd.customerId, mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
# REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
# Data_selection,
#    deviceOs osType,
#   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#     left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
#   ),
#   modelname as (
#   select customerId,
#   digitalLoanAccountId,
#   start_time,
#   prediction sil_beta_demo_score,
#   case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
#   modelVersionId, trenchCategory,Data_selection,
#      osType,
#   from cleaned
#   ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
# (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   r.sil_beta_demo_score,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#     deffpd0,
#   flg_mature_fpd0,
#   loanmaster.new_loan_type,
#   modelVersionId,
#     trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#          coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#     from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and r.sil_beta_demo_score is not null
#   and flg_mature_fpd0 = 1
#   )
#   select * from base
#   qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#   ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# # %%
# df2 = dfd.copy()


# # #### Concatenate Test and Train Datasets



# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()



# #### Making sure the score is numerical

# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_betademoscoresil = fact_table.copy()
df_d_fpd0_betademoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fpd0_betademoscoresil, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


### FPD10


# ### Test

# %%
sq = """
WITH cleaned AS (
     SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
    deviceOs osType,
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      osType,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
        Application_type,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# # ### Train

# # %%
# sq = """
# WITH cleaned AS (
#     SELECT
#   mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
# REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
# Data_selection,
#     deviceOs osType,
#   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#     left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
#   ),
#   modelname as (
#   select customerId,
#   digitalLoanAccountId,
#   start_time,
#   prediction sil_beta_demo_score,
#   case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
#   modelVersionId, trenchCategory,Data_selection,
#       osType,
#   from cleaned
#   ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
# (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   r.sil_beta_demo_score,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#     del.deffpd10,
#   del.flg_mature_fpd10,
#   loanmaster.new_loan_type,
#   modelVersionId,
#     trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#     from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and r.sil_beta_demo_score is not null
#   and del.flg_mature_fpd10 = 1
#   )
#   select * from base
#   qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#   ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# # %%
# df2 = dfd.copy()


# # #### Concatenate Test and Train Set



# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()



# #### Making sure the score is numerical

# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)


# #### Calculate Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)


# #### Update fact and dimension table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

df_f_fpd10_betademoscoresil = fact_table.copy()
df_d_fpd10_betademoscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fpd10_betademoscoresil, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD30


# ### Test

# %%
sq = """
WITH cleaned AS (
 SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
   deviceOs osType,

  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
     osType,
    from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
        Application_type,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,

    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# # ### Train

# # %%
# sq = """
# WITH cleaned AS (
#     SELECT
#   mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
# REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
# Data_selection,
#     deviceOs osType,
#   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#     left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
#   ),
#   modelname as (
#   select customerId,
#   digitalLoanAccountId,
#   start_time,
#   prediction sil_beta_demo_score,
#   case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
#   modelVersionId, trenchCategory, Data_selection,
#       osType,
#   from cleaned
#   ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
# (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   r.sil_beta_demo_score,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#     del.deffpd30,
#   del.flg_mature_fpd30,
#   loanmaster.new_loan_type,
#   modelVersionId,
#     trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#     from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and r.sil_beta_demo_score is not null
#   and del.flg_mature_fpd30 = 1
#   )
#   select * from base
#   qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#   ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# # %%
# df2 = dfd.copy()


# # ### Concat


# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()



# #### Making sure the score is numerical

# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)


# #### Calculate the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)


# #### Updating fact and dimension tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

df_f_fpd30_betademoscoresil = fact_table.copy()
df_d_fpd30_betademoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fpd30_betademoscoresil, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSPD30


# ### Test

# %%
sq = """
WITH cleaned AS (
     SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
   deviceOs osType,
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
     osType,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    Application_type,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# # ### Train

# # %%
# sq = """
# WITH cleaned AS (
#     SELECT
#   mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#     case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
# REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
# Data_selection,
#     deviceOs osType,
#   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#     left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
#   ),
#   modelname as (
#   select customerId,
#   digitalLoanAccountId,
#   start_time,
#   prediction sil_beta_demo_score,
#   case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
#   modelVersionId, trenchCategory, Data_selection,
#       osType,
#   from cleaned
#   ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
# (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   r.sil_beta_demo_score,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#     del.deffspd30,
#   del.flg_mature_fspd_30,
#   loanmaster.new_loan_type,
#   modelVersionId,
#     trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#     from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and r.sil_beta_demo_score is not null
#   and del.flg_mature_fspd_30 = 1
#   )
#   select * from base
#   qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#   ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# # %%
# df2 = dfd.copy()


# # #### Concatenate Test and Train Dataset



# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

df_f_fspd30_betademoscoresil = fact_table.copy()
df_d_fspd30_betademoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fspd30_betademoscoresil, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSTPD30


# ### Test

# %%
sq = """
WITH cleaned AS (
SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
    deviceOs osType,
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
      osType,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
        Application_type,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# # ### Train

# # %%
# sq = """
# WITH cleaned AS (
#   SELECT
#   mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
# REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
# Data_selection,
#     deviceOs osType,
#   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#     left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
#   ),
#   modelname as (
#   select customerId,
#   digitalLoanAccountId,
#   start_time,
#   prediction sil_beta_demo_score,
#   case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
#   modelVersionId, trenchCategory, Data_selection,
#       osType,
#   from cleaned
#   ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
# (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   r.sil_beta_demo_score,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#     del.deffstpd30,
#   del.flg_mature_fstpd_30,
#   loanmaster.new_loan_type,
#   modelVersionId,
#     trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#     from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and r.sil_beta_demo_score is not null
#   and del.flg_mature_fstpd_30 = 1
#   )
#   select * from base
#   qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#   ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# # %%
# df2 = dfd.copy()


# # ### Concat


# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_betademoscoresil = fact_table.copy()
df_d_fstpd30_betademoscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(df_f_fstpd30_betademoscoresil, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factbetademosil = pd.concat([df_f_fpd0_betademoscoresil, df_f_fpd10_betademoscoresil, df_f_fpd30_betademoscoresil, df_f_fspd30_betademoscoresil, df_f_fstpd30_betademoscoresil], ignore_index=False)
dimbetademosil = pd.concat([df_d_fpd0_betademoscoresil, df_d_fpd10_betademoscoresil, df_d_fpd30_betademoscoresil, df_d_fspd30_betademoscoresil, df_d_fstpd30_betademoscoresil], ignore_index=False)

print(f"beta_demo_model_sil model Gini Calculation and Upload Completed!")


# %% [markdown]
# # End

# %%


# %%



