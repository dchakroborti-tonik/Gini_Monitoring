# %% [markdown]
# # SIL Train Gini Calculation

# %% [markdown]
# ## Define Library

import io
import os
import pickle
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from typing import Union

import duckdb as dd
import gcsfs
import joblib
import matplotlib.pyplot as plt
import numpy as np
# %%
# %% [markdown]
# # Jupyter Notebook Loading Header
#
# This is a custom loading header for Jupyter Notebooks in Visual Studio Code.
# It includes common imports and settings to get you started quickly.
# %% [markdown]
## Import Libraries
import pandas as pd
import seaborn as sns
from google.cloud import bigquery, storage
from sklearn.metrics import roc_auc_score

path = r"C:\Users\Dwaipayan\AppData\Roaming\gcloud\application_default_credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
client = bigquery.Client(project="prj-prod-dataplatform")
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"


import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
## Configure Settings
# Set options or configurations as needed
pd.set_option("display.max_columns", None)
pd.set_option("Display.max_rows", 100)

# %% [markdown]
# ## Functions

# %% [markdown]
# ### calculate_periodic_gini_prod_ver_trench_dimfact

from datetime import timedelta
from itertools import combinations

import numpy as np
# %%
import pandas as pd


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
                    calculate_gini(x[score_column], x[label_column])
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
            weekly_gini = weekly_gini.merge(weekly_bad_counts, on="week", how="left")
            weekly_gini["bad_count"] = weekly_gini["bad_count"].fillna(0).astype(int)
        else:
            weekly_gini["bad_count"] = None

        weekly_gini = weekly_gini[
            [
                "start_date",
                "end_date",
                "gini_value",
                "period",
                "distinct_accounts",
                "bad_count",
            ]
        ]

        # Calculate monthly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["month"] = dataset_df_copy["disbursementdate"].dt.to_period("M")
        monthly_gini = (
            dataset_df_copy.groupby("month")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
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
            [
                "start_date",
                "end_date",
                "gini_value",
                "period",
                "distinct_accounts",
                "bad_count",
            ]
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
# #### cic_model_sil

# %% [markdown]
# ##### FPD0

# %% [markdown]
# ###### Test

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,                                                                                                                --- Added ostype
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
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
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
  case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Alpha_cic_sil_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the cic sil fpd 0 dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df_concat = dfd.copy()

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)
df_concat["deffpd0"] = pd.to_numeric(df_concat["deffpd0"], errors="coerce")
df_concat["flg_mature_fpd0"] = pd.to_numeric(
    df_concat["flg_mature_fpd0"], errors="coerce"
)

# %%
#### Calculating the Gini

import time

start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",  # Add this
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)


end = time.perf_counter()
print(f"To calculate cic sil fpd 0 Elapsed time: {(end - start)/60:.3f} minutes")


# %%
#### Updating Fact and Dimension Table
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, "cic_model_sil", "SIL"
)

# %% [markdown]
# #### Table names

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_cicsiltest1"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_cicsiltest1"

# %%
df_f_fpd0_cicsil = fact_table.copy()
df_d_fpd0_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd0_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd0_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %% [markdown]
# #### FPD10

# %%
#### Test
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
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
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
  case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the cic sil test fpd10 dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %%
start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",  # Add this
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

end = time.perf_counter()
print(f"To calculate cic sil fpd 10 Elapsed time: {(end - start)/60:.3f} minutes")

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
df_f_fpd10_cicsil = fact_table.copy()
df_d_fpd10_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd10_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd10_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %% [markdown]
# #### FPD30

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
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
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the cic sil fpd30 test dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)
df_concat["deffpd30"] = pd.to_numeric(df_concat["deffpd30"], errors="coerce")
df_concat["flg_mature_fpd30"] = pd.to_numeric(
    df_concat["flg_mature_fpd30"], errors="coerce"
)

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",  # Add this
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

end = time.perf_counter()
print(f"To calculate cic sil fpd 0 Elapsed time: {(end - start)/60:.3f} minutes")

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
df_f_fpd30_cicsil = fact_table.copy()
df_d_fpd30_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd30_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd30_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %% [markdown]
# #### FSPD30

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
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
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the cic sil fspd30 test dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)
df_concat["deffspd30"] = pd.to_numeric(df_concat["deffspd30"], errors="coerce")
df_concat["flg_mature_fspd_30"] = pd.to_numeric(
    df_concat["flg_mature_fspd_30"], errors="coerce"
)

# %%
start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

end = time.perf_counter()
print(f"To calculate cic sil fpd 0 Elapsed time: {(end - start)/60:.3f} minutes")


# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
df_f_fspd30_cicsil = fact_table.copy()
df_d_fspd30_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fspd30_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fspd30_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %% [markdown]
# #### FSTPD30

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
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
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the cic sil fstpd30 test dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)
df_concat["deffstpd30"] = pd.to_numeric(df_concat["deffstpd30"], errors="coerce")
df_concat["flg_mature_fstpd_30"] = pd.to_numeric(
    df_concat["flg_mature_fstpd_30"], errors="coerce"
)


# %%

start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)
end = time.perf_counter()
print(f"To calculate cic sil fstpd30 Elapsed time: {(end - start)/60:.3f} minutes")


# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
df_f_fstpd30_cicsil = fact_table.copy()
df_d_fstpd30_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fstpd30_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fstpd30_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
df_fact_cic_sil = pd.concat(
    [
        df_f_fpd0_cicsil,
        df_f_fpd10_cicsil,
        df_f_fpd30_cicsil,
        df_f_fspd30_cicsil,
        df_f_fstpd30_cicsil,
    ],
    ignore_index=True,
)
df_dim_cic_sil = pd.concat(
    [
        df_d_fpd0_cicsil,
        df_d_fpd10_cicsil,
        df_d_fpd30_cicsil,
        df_d_fspd30_cicsil,
        df_d_fstpd30_cicsil,
    ],
    ignore_index=True,
)

# %% [markdown]
# #### alpha_stack_model_sil

# %%
# ## alpha_stack_model_sil

# ### FPD0
# #### Test

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
     deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
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
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Sil_Alpha_Stack_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(
    f"The shape of the alpha stack sil test fpd0 dataframe downloaded is:\t {dfd.shape}"
)
dfd.head()


# %%
df_concat = dfd.copy()

# #### Making Sure the Score is Numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)

# #### Calculating the Gini

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

end = time.perf_counter()
print(
    f"To calculate alpha stack sil fpd 0 Elapsed time: {(end - start)/60:.3f} minutes"
)
# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_alphastacksil = fact_table.copy()
df_d_fpd0_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd0_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd0_alphastacksil.shape}"
)

facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphastacksil_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphastacksil_test1"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD10


# #### Test

# %%
sq = """
with modelname as
  ( SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
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
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(
    f"The shape of the alpha stack sil test fpd10 dataframe downloaded is:\t {dfd.shape}"
)
dfd.head()


# %%
df_concat = dfd.copy()


# #### Making Sure the Score is Numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

end = time.perf_counter()
print(
    f"To calculate alpha stack sil fpd 10 Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)


df_f_fpd10_alphastacksil = fact_table.copy()
df_d_fpd10_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd10_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd10_alphastacksil.shape}"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD30


# #### Test

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
      deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
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
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(
    f"The shape of the alpha stack sil fpd30 test dataframe downloaded is:\t {dfd.shape}"
)
dfd.head()


# %%
df_concat = dfd.copy()

# #### Making Sure the Score is Numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)
end = time.perf_counter()
print(
    f"To calculate alpha stack sil fpd 30 Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)


df_f_fpd30_alphastacksil = fact_table.copy()
df_d_fpd30_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd30_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd30_alphastacksil.shape}"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSPD30


# #### Test

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
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
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Sil_Alpha_Stack_score is not null
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

# #### Making sure the score column in numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

end = time.perf_counter()
print(
    f"To calculate alpha stack sil fspd 30 Elapsed time: {(end - start)/60:.3f} minutes"
)
# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)


df_f_fspd30_alphastacksil = fact_table.copy()
df_d_fspd30_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fspd30_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fspd30_alphastacksil.shape}"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSTPD30

# #### Test

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
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
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(
    f"The shape of the alpha stack sil fstpd30 test dataframe downloaded is:\t {dfd.shape}"
)
dfd.head()


# %%
df_concat = dfd.copy()


# #### Making sure the score column is numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%

start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",
    account_id_column="digitalLoanAccountId",
)

end = time.perf_counter()
print(
    f"To calculate alpha stack sil fstpd 30 Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating the Fact and Dimension Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_alphastacksil = fact_table.copy()
df_d_fstpd30_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fstpd30_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fstpd30_alphastacksil.shape}"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factallaplhastacksil = pd.concat(
    [
        df_f_fpd0_alphastacksil,
        df_f_fpd10_alphastacksil,
        df_f_fpd30_alphastacksil,
        df_f_fspd30_alphastacksil,
        df_f_fstpd30_alphastacksil,
    ],
    ignore_index=True,
)
dimallaplhastacksil = pd.concat(
    [
        df_d_fpd0_alphastacksil,
        df_d_fpd10_alphastacksil,
        df_d_fpd30_alphastacksil,
        df_d_fspd30_alphastacksil,
        df_d_fstpd30_alphastacksil,
    ],
    ignore_index=True,
)

print(f"alpha_stack_model_sil model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### Beta App Score SIL

# %%
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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_app_score,
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
  and r.sil_beta_app_score is not null
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


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
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
end = time.perf_counter()
print(
    f"To calculate beta app score sil fpd 0 Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_appscoresil = fact_table.copy()
df_d_fpd0_appscoresil = dimension_table.copy()

facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_appscoresil_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_appscoresil_test1"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD10


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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_app_score,
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
  and r.sil_beta_app_score is not null
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


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
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
end = time.perf_counter()
print(
    f"To calculate beta app score sil fpd10 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

df_f_fpd10_appscoresil = fact_table.copy()
df_d_fpd10_appscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD30


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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
     deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_app_score,
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
  and r.sil_beta_app_score is not null
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


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
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
end = time.perf_counter()
print(
    f"To calculate beta app score sil fpd30 test Elapsed time: {(end - start)/60:.3f} minutes"
)


# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

df_f_fpd30_appscoresil = fact_table.copy()
df_d_fpd30_appscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSPD30


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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
     deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_app_score,
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
  and r.sil_beta_app_score is not null
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

# #### Making sure the score column in numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
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

end = time.perf_counter()
print(
    f"To calculate beta app score sil fspd30 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

df_f_fspd30_appscoresil = fact_table.copy()
df_d_fspd30_appscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSTPD30


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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
      deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_app_score,
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
  and r.sil_beta_app_score is not null
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


# #### Making sure the score column is numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()


fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
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
end = time.perf_counter()
print(
    f"To calculate beta app score sil fpd 0 Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating the Fact and Dimension Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_appscoresil = fact_table.copy()
df_d_fstpd30_appscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factappscoresil = pd.concat(
    [
        df_f_fpd0_appscoresil,
        df_f_fpd10_appscoresil,
        df_f_fpd30_appscoresil,
        df_f_fspd30_appscoresil,
        df_f_fstpd30_appscoresil,
    ],
    ignore_index=True,
)
demoappscoresil = pd.concat(
    [
        df_d_fpd0_appscoresil,
        df_d_fpd10_appscoresil,
        df_d_fpd30_appscoresil,
        df_d_fspd30_appscoresil,
        df_d_fstpd30_appscoresil,
    ],
    ignore_index=True,
)


# %% [markdown]
# #### Beta SIL Demo Score

# %%
facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademosil_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betademosil_test1"
)

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
job = client.load_table_from_dataframe(
    df_f_fpd0_betademoscoresil, facttable_id, job_config=job_config
)
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
job = client.load_table_from_dataframe(
    df_f_fpd10_betademoscoresil, facttable_id, job_config=job_config
)
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
job = client.load_table_from_dataframe(
    df_f_fpd30_betademoscoresil, facttable_id, job_config=job_config
)
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
job = client.load_table_from_dataframe(
    df_f_fspd30_betademoscoresil, facttable_id, job_config=job_config
)
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
job = client.load_table_from_dataframe(
    df_f_fstpd30_betademoscoresil, facttable_id, job_config=job_config
)
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

factbetademosil = pd.concat(
    [
        df_f_fpd0_betademoscoresil,
        df_f_fpd10_betademoscoresil,
        df_f_fpd30_betademoscoresil,
        df_f_fspd30_betademoscoresil,
        df_f_fstpd30_betademoscoresil,
    ],
    ignore_index=False,
)
dimbetademosil = pd.concat(
    [
        df_d_fpd0_betademoscoresil,
        df_d_fpd10_betademoscoresil,
        df_d_fpd30_betademoscoresil,
        df_d_fspd30_betademoscoresil,
        df_d_fstpd30_betademoscoresil,
    ],
    ignore_index=False,
)

print(f"beta_demo_model_sil model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### Beta SIL STACK Score Model

# %%
# ##
facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betastacksil_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betastacksil_test1"
)

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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
    ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_stack_score,
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
  and r.sil_beta_stack_score is not null
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


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
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

end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fpd0 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")


# #### Assigning the Table Name

df_f_fpd0_betastackscoresil = fact_table.copy()
df_d_fpd0_betastackscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# #### Creating the table


# ### FPD10


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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
  osType
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
  r.sil_beta_stack_score,
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
  and r.sil_beta_stack_score is not null
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


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
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
end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fpd10 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)


df_f_fpd10_betastackscoresil = fact_table.copy()
df_d_fpd10_betastackscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD30


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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_stack_score,
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
  and r.sil_beta_stack_score is not null
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


#


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%

start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
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

end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fpd30 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)

df_f_fpd30_betastackscoresil = fact_table.copy()
df_d_fpd30_betastackscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# #### Inserting the data into Fact and Dimension tables


# ### FSPD30


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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_stack_score,
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
  and r.sil_beta_stack_score is not null
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


# #### Making sure the score column in numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%

start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
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

end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fspd30 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)


# #### Inserting the Data into Fact and Dimension Table

df_f_fspd30_betastackscoresil = fact_table.copy()
df_d_fspd30_betastackscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FSTPD30


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
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
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
  r.sil_beta_stack_score,
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
  and r.sil_beta_stack_score is not null
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


# #### Making sure the score column is numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
start = time.perf_counter()

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
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

end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fstpd30 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# #### Updating the Fact and Dimension Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")


df_f_fstpd30_betastackscoresil = fact_table.copy()
df_d_fstpd30_betastackscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


factbetademosil = pd.concat(
    [
        df_f_fpd0_betastackscoresil,
        df_f_fpd10_betastackscoresil,
        df_f_fpd30_betastackscoresil,
        df_f_fspd30_betastackscoresil,
        df_f_fstpd30_betastackscoresil,
    ],
    ignore_index=False,
)
dimbetademosil = pd.concat(
    [
        df_d_fpd0_betastackscoresil,
        df_d_fpd10_betastackscoresil,
        df_d_fpd30_betastackscoresil,
        df_d_fspd30_betastackscoresil,
        df_d_fstpd30_betastackscoresil,
    ],
    ignore_index=False,
)

# %%

print(f"beta_stack_model_sil model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### beta_stack_model_sil_credo_score

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_credosil_test1"
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_credosil_test1"
)


# ## FPD0

# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
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
   coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
  case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and flg_mature_fpd0 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# %%
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
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

end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fpd0 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
df_f_fpd0_betastackcredoscoresil = fact_table.copy()
df_d_fpd0_betastackcredoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ## FPD10

# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,

    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
    coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd10 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# %%
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
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
end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fpd10 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd10_betastackcredoscoresil = fact_table.copy()
df_d_fpd10_betastackcredoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ## FPD30

# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
       deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# %%
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
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

end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fpd30 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
df_f_fpd30_betastackcredoscoresil = fact_table.copy()
df_d_fpd30_betastackcredoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ## FSPD30

# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()
# %%
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
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
end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fspd30 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fspd30_betastackcredoscoresil = fact_table.copy()
df_d_fspd30_betastackcredoscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ## FSTPD30


# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df_concat = dfd.copy()


# %%
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
start = time.perf_counter()
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
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

end = time.perf_counter()
print(
    f"To calculate sil_beta_stack_score fstpd30 test Elapsed time: {(end - start)/60:.3f} minutes"
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_betastackcredoscoresil = fact_table.copy()
df_d_fstpd30_betastackcredoscoresil = dimension_table.copy()
# %%

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


factbetacredosil = pd.concat(
    [
        df_f_fpd0_betastackcredoscoresil,
        df_f_fpd10_betastackcredoscoresil,
        df_f_fpd30_betastackcredoscoresil,
        df_f_fspd30_betastackcredoscoresil,
        df_f_fstpd30_betastackcredoscoresil,
    ],
    ignore_index=True,
)
dimbetacredosil = pd.concat(
    [
        df_d_fpd0_betastackcredoscoresil,
        df_d_fpd10_betastackcredoscoresil,
        df_d_fpd30_betastackcredoscoresil,
        df_d_fspd30_betastackcredoscoresil,
        df_d_fstpd30_betastackcredoscoresil,
    ],
    ignore_index=True,
)

print(f"beta_stack_credo_score_sil model Gini Calculation and Upload Completed!")


# %%
sq = """ 
create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.fact_siltest1 as 
select * from 
(select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_cicsiltest1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphastacksil_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_appscoresil_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademosil_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betastacksil_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_credosil_test1
)
"""

client.query(sq).result()


# %%
sq = """ 
create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_siltest1 as 
select * from 
(select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_cicsiltest1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphastacksil_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_appscoresil_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betademosil_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betastacksil_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_credosil_test1
)
"""

client.query(sq).result()

# %%
factalldf = pd.concat(
    [
        df_f_fpd0_cicsil,
        df_f_fpd10_cicsil,
        df_f_fpd30_cicsil,
        df_f_fspd30_cicsil,
        df_f_fstpd30_cicsil,
        #    df_f_fpd0_alphacredosil, df_f_fpd10_alphacredosil, df_f_fpd30_alphacredosil, df_f_fspd30_alphacredosil, df_f_fstpd30_alphacredosil,
        df_f_fpd0_alphastacksil,
        df_f_fpd10_alphastacksil,
        df_f_fpd30_alphastacksil,
        df_f_fspd30_alphastacksil,
        df_f_fstpd30_alphastacksil,
        df_f_fpd0_appscoresil,
        df_f_fpd10_appscoresil,
        df_f_fpd30_appscoresil,
        df_f_fspd30_appscoresil,
        df_f_fstpd30_appscoresil,
        df_f_fpd0_betademoscoresil,
        df_f_fpd10_betademoscoresil,
        df_f_fpd30_betademoscoresil,
        df_f_fspd30_betademoscoresil,
        df_f_fstpd30_betademoscoresil,
        df_f_fpd0_betastackscoresil,
        df_f_fpd10_betastackscoresil,
        df_f_fpd30_betastackscoresil,
        df_f_fspd30_betastackscoresil,
        df_f_fstpd30_betastackscoresil,
        df_f_fpd0_betastackcredoscoresil,
        df_f_fpd10_betastackcredoscoresil,
        df_f_fpd30_betastackcredoscoresil,
        df_f_fspd30_betastackcredoscoresil,
        df_f_fstpd30_betastackcredoscoresil,
    ],
    ignore_index=True,
)

# %%
sq = """
create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.fact_sil_combined1 as 
select * from 
(select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_siltest1
union all 
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_siltrain1
)
;
"""
client.query(sq).result()


# %%
sq = """
create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_sil_combined1 as 
select * from 
(select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_siltest1
union all 
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_siltrain1
)
;
"""
client.query(sq).result()

# %% [markdown]
# # End

# %%


# %%
