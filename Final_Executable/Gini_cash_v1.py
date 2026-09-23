# %% [markdown]
# # CASH Train Gini Calculation

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
    risk_segment_column=None,
    risk_segment_final_column=None,
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
        "risk_segment": risk_segment_column,
        "risk_segment_final": risk_segment_final_column,
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
    if risk_segment_column:
        segment_columns.append(("risk_segment", risk_segment_column))
    if risk_segment_final_column:
        segment_columns.append(("risk_segment_final", risk_segment_final_column))

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
        gini_results["risk_segment"] = (
            metadata.get(risk_segment_column, None) if risk_segment_column else None
        )
        gini_results["risk_segment_final"] = (
            metadata.get(risk_segment_final_column, None)
            if risk_segment_final_column
            else None
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
                "risk_segment",
                "risk_segment_final",
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
                "risk_segment",
                "risk_segment_final",
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
            "risk_segment",
            "risk_segment_final",
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
            "risk_segment",
            "risk_segment_final",
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
            "risk_segment",
            "risk_segment_final",
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
# ### Models

# %% [markdown]
# #### cic_model_cash

# %%
facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphaciccash_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphaciccash_test1"
)

# ### FPD0

# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
          modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd0,
  del.flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, 
  r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
        coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fpd0 = 1
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

# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

print("Gini Calculation for cic score cash for FPD0 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffpd0",
    "FPD0",
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
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_ciccash = fact_table.copy()
df_d_fpd0_ciccash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD10

# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
          modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, 
  r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
        coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
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


# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

print("Gini Calculation for cic score cash for FPD10 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
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
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

df_f_fpd10_ciccash = fact_table.copy()
df_d_fpd10_ciccash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%

# ### FPD30

# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
          modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, 
  r.trenchCategory,
   case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
        coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
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


# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

print("Gini Calculation for cic score cash for FPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffpd30",
    "FPD30",
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
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

df_f_fpd30_ciccash = fact_table.copy()
df_d_fpd30_ciccash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
          modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, 
  r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
        coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
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

# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

print("Gini Calculation for cic score cash for FSPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffspd30",
    "FSPD30",
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
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

df_f_fspd30_ciccash = fact_table.copy()
df_d_fspd30_ciccash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete
# %%


# ### FSTPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
          modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, 
  r.trenchCategory,
   case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
        coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
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


# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")
print("Gini Calculation for cic score cash for FSTPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffstpd30",
    "FSTPD30",
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
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_ciccash = fact_table.copy()
df_d_fstpd30_ciccash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%

factciccash = pd.concat(
    [
        df_f_fpd0_ciccash,
        df_f_fpd10_ciccash,
        df_f_fpd30_ciccash,
        df_f_fspd30_ciccash,
        df_f_fstpd30_ciccash,
    ],
    ignore_index=True,
)
demociccash = pd.concat(
    [
        df_d_fpd0_ciccash,
        df_d_fpd10_ciccash,
        df_d_fpd30_ciccash,
        df_d_fspd30_ciccash,
        df_d_fstpd30_ciccash,
    ],
    ignore_index=True,
)

print(f"cic_model_cash model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### Alpha-Cash-Stack-Model

# %%
facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphastackcash_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphastackcash_test1"
)

# ##

# ### FPD0


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
       osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd0,
  del.flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
            coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fpd0 = 1
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

# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

print("Gini Calculation for Alpha Stack score cash for FPD0 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffpd0",
    "FPD0",
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
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_alphastackcash = fact_table.copy()
df_d_fpd0_alphastackcash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
       osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
            coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
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

# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

print("Gini Calculation for Alpha Stack score cash for FPD10 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
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
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

df_f_fpd10_alphastackcash = fact_table.copy()
df_d_fpd10_alphastackcash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# %%


# ### FPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
       osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
   case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
            coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
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

# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

print("Gini Calculation for Alpha Stack score cash for FPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffpd30",
    "FPD30",
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
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

df_f_fpd30_alphastackcash = fact_table.copy()
df_d_fpd30_alphastackcash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%

# ### FSPD30

# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
       osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
            coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
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

# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

print("Gini Calculation for Alpha Stack score cash for FSPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

df_f_fspd30_alphastackcash = fact_table.copy()
df_d_fspd30_alphastackcash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSTPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
       osType,
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
            coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
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

# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

print("Gini Calculation for Alpha Stack score cash for FSTPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_alphastackcash = fact_table.copy()
df_d_fstpd30_alphastackcash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factalphastackcash = pd.concat(
    [
        df_f_fpd0_alphastackcash,
        df_f_fpd10_alphastackcash,
        df_f_fpd30_alphastackcash,
        df_f_fspd30_alphastackcash,
        df_f_fstpd30_alphastackcash,
    ],
    ignore_index=True,
)
dimalphastackcash = pd.concat(
    [
        df_d_fpd0_alphastackcash,
        df_d_fpd10_alphastackcash,
        df_d_fpd30_alphastackcash,
        df_d_fspd30_alphastackcash,
        df_d_fstpd30_alphastackcash,
    ],
    ignore_index=True,
)

print(f"The alpha_stack_model_cash gini calculation completed")

# %% [markdown]
# ####  Beta-Cash-Stack-Model- Credo Score

# %%
# %%
facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betacredocash_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betacredocash_test1"
)

# %%
# ## Beta-Cash-Stack-Model- Credo Score

# ### FPD0
# ### Test

# %%

sq = """
with modelname as
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
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
         modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
             coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd0 = 1
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

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

print("Gini Calculation for Credo score cash for FPD0 started")
# %%
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
df_f_fpd0_credoscorecash = fact_table.copy()
df_d_fpd0_credoscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10


# ### Test

# %%

sq = """
with modelname as
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
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
         modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
    case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
             coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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


df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")
print("Gini Calculation for Credo score cash for FPD10 started")
# %%
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
df_f_fpd10_credoscorecash = fact_table.copy()
df_d_fpd10_credoscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete
# %%


# ### FPD30

# ### Test

# %%

sq = """
with modelname as
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
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
         modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
    case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
             coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
print("Gini Calculation for Credo score cash for FPD30 started")
# %%
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd30_credoscorecash = fact_table.copy()
df_d_fpd30_credoscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSPD30


# ### Test

# %%

sq = """
with modelname as
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
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
         modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
             coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")
print("Gini Calculation for Credo score cash for FSPD30 started")
# %%
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
df_f_fspd30_credoscorecash = fact_table.copy()
df_d_fspd30_credoscorecash = dimension_table.copy()
# %%

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FSTPD30

# ### Test

# %%

sq = """
with modelname as
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
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
         modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
    case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
             coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")
print("Gini Calculation for Credo score cash for FSTPD30 started")
# %%
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_credoscorecash = fact_table.copy()
df_d_fstpd30_credoscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


factcredoscorecash = pd.concat(
    [
        df_f_fpd0_credoscorecash,
        df_f_fpd10_credoscorecash,
        df_f_fpd30_credoscorecash,
        df_f_fspd30_credoscorecash,
        df_f_fstpd30_credoscorecash,
    ],
    ignore_index=True,
)
dimcredoscorecash = pd.concat(
    [
        df_d_fpd0_credoscorecash,
        df_d_fpd10_credoscorecash,
        df_d_fpd30_credoscorecash,
        df_d_fspd30_credoscorecash,
        df_d_fstpd30_credoscorecash,
    ],
    ignore_index=True,
)


print(f" credo_score_cash gini calculation completed")


# %% [markdown]
# #### Beta-Cash-AppScore-Model

# %%

# %%

facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_appscorecash_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_appscorecash_test1"
)

# ## Beta-Cash-AppScore-Model

# ### FPD0

# ### Test
# %%

sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,  deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
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
base as
  (select  distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd0,
  del.flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
    case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fpd0 = 1
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

# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-AppScore-Model for FPD0 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")


df_f_fpd0_appscorecash = fact_table.copy()
df_d_fpd0_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_appscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10

# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,  deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
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
base as
  (select  distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
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


# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-AppScore-Model for FPD10 started")
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
    apptype_column="Application_type",  # Add this
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
job = client.load_table_from_dataframe(
    df_f_fpd10_appscorecash, facttable_id, job_config=job_config
)
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

# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,  deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
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
base as
  (select  distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
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


# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-AppScore-Model for FPD30 started")
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
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
job = client.load_table_from_dataframe(
    df_f_fpd30_appscorecash, facttable_id, job_config=job_config
)
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

# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,  deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
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
base as
  (select  distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
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

# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-AppScore-Model for FSPD30 started")
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
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
job = client.load_table_from_dataframe(
    df_f_fspd30_appscorecash, facttable_id, job_config=job_config
)
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


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,  deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
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
base as
  (select  distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
    case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
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

# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-AppScore-Model for FSTPD30 started")
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
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
job = client.load_table_from_dataframe(
    df_f_fstpd30_appscorecash, facttable_id, job_config=job_config
)
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

factappscorecash = pd.concat(
    [
        df_f_fpd0_appscorecash,
        df_f_fpd10_appscorecash,
        df_f_fpd30_appscorecash,
        df_f_fspd30_appscorecash,
        df_f_fstpd30_appscorecash,
    ],
    ignore_index=True,
)
dimappscorecash = pd.concat(
    [
        df_d_fpd0_appscorecash,
        df_d_fpd10_appscorecash,
        df_d_fpd30_appscorecash,
        df_d_fspd30_appscorecash,
        df_d_fstpd30_appscorecash,
    ],
    ignore_index=True,
)


# %%

# %% [markdown]
#
#
#
#
#
# #### Beta-Cash-Demo-Model

# %%

facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademocash_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betademocash_test1"
)

# ## Beta-Cash-Demo-Model

# ### FPD0

# ### Test
# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  osType,
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
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
  base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd0,
  del.flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fpd0 = 1
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

# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

print("Gini Calculation for Beta-Cash-Demo-Model for FPD0 started")
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
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
job = client.load_table_from_dataframe(
    df_f_fpd0_betademocash, facttable_id, job_config=job_config
)
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

# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  osType,
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
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
  base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
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


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Demo-Model for FPD10 started")
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
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
job = client.load_table_from_dataframe(
    df_f_fpd10_betademocash, facttable_id, job_config=job_config
)
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

# ### Test
# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  osType,
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
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
  base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
    case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
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


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Demo-Model for FPD30 started")
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
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
job = client.load_table_from_dataframe(
    df_f_fpd30_betademocash, facttable_id, job_config=job_config
)
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


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  osType,
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
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
  base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
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


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Demo-Model for FSPD30 started")
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
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
job = client.load_table_from_dataframe(
    df_f_fspd30_betademocash, facttable_id, job_config=job_config
)
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


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  osType,
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
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
  base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
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


df_concat = dfd.copy()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Demo-Model for FSTPD30 started")
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
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
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
job = client.load_table_from_dataframe(
    df_f_fstpd30_betademocash, facttable_id, job_config=job_config
)
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

factbetademocash = pd.concat(
    [
        df_f_fpd0_betademocash,
        df_f_fpd10_betademocash,
        df_f_fpd30_betademocash,
        df_f_fspd30_betademocash,
        df_f_fstpd30_betademocash,
    ],
    ignore_index=True,
)
dimbetademocash = pd.concat(
    [
        df_d_fpd0_betademocash,
        df_d_fpd10_betademocash,
        df_d_fpd30_betademocash,
        df_d_fspd30_betademocash,
        df_d_fstpd30_betademocash,
    ],
    ignore_index=True,
)


print("beta_demo_model_cash gini calculation completed")


# %% [markdown]
# #### Beta-Cash-Stack-Model

# %%

facttable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betastackcash_test1"
)
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betastackcash_test1"
)

# %%
# ## Beta-Cash-Stack-Model


# ### FPD0


# ### Test

# %%

sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
 deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId, osType
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select
  distinct
  r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd0,
  del.flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fpd0 = 1
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

# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Stack-Model for FPD0 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_betastackcash = fact_table.copy()
df_d_fpd0_betastackcash = dimension_table.copy()


job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
 deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId, osType
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select
  distinct
  r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
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

# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Stack-Model for FPD10 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

df_f_fpd10_betastackcash = fact_table.copy()
df_d_fpd10_betastackcash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
 deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId, osType
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select
  distinct
  r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
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


# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Stack-Model for FPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

df_f_fpd30_betastackcash = fact_table.copy()
df_d_fpd30_betastackcash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# %%

# ### FSPD30

# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
 deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId, osType
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select
  distinct
  r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
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


# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Stack-Model for FSPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

# %%
df_f_fspd30_betastackcash = fact_table.copy()
df_d_fspd30_betastackcash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSTPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
 deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details`
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
,
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId, osType
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
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
         modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
base as
  (select
  distinct
  r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, r.trenchCategory,
  case when r.trenchCategory in ('Trench 1', 'Trench 2') then 'New_Applicant' else 'Repeat_Applicant' end Application_type,
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
            ) as osType,
    coalesce(sd.risk_segment, 'NA') risk_segment,
    coalesce(frs.risk_segment_final, 'NA') risk_segment_final
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
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
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
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


# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)
print("Gini Calculation for Beta-Cash-Stack-Model for FSTPD30 started")
# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    apptype_column="Application_type",  # Add this
    risk_segment_column="risk_segment",
    risk_segment_final_column="risk_segment_final",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_betastackcash = fact_table.copy()
df_d_fstpd30_betastackcash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%

print("beta_stack_model_cash gini calculation completed")


# %%
# factalldf = pd.concat([ df_f_fpd0_ciccash, df_f_fpd10_ciccash, df_f_fpd30_ciccash, df_f_fspd30_ciccash, df_f_fstpd30_ciccash,
#                        df_f_fpd0_alphastackcash, df_f_fpd10_alphastackcash, df_f_fpd30_alphastackcash, df_f_fspd30_alphastackcash, df_f_fstpd30_alphastackcash,
#                        df_f_fpd0_credoscorecash, df_f_fpd10_credoscorecash, df_f_fpd30_credoscorecash, df_f_fspd30_credoscorecash, df_f_fstpd30_credoscorecash,
#                        df_f_fpd0_appscorecash, df_f_fpd10_appscorecash, df_f_fpd30_appscorecash, df_f_fspd30_appscorecash, df_f_fstpd30_appscorecash,
#                        df_f_fpd0_betademocash, df_f_fpd10_betademocash, df_f_fpd30_betademocash, df_f_fspd30_betademocash, df_f_fstpd30_betademocash,
#                        df_f_fpd0_betastackcash, df_f_fpd10_betastackcash, df_f_fpd30_betastackcash, df_f_fspd30_betastackcash, df_f_fstpd30_betastackcash,
#                        ], ignore_index=True)
# dimalldf = pd.concat([df_d_fpd0_ciccash, df_d_fpd10_ciccash, df_d_fpd30_ciccash, df_d_fspd30_ciccash, df_d_fstpd30_ciccash,
#                       df_d_fpd0_alphastackcash, df_d_fpd10_alphastackcash, df_d_fpd30_alphastackcash, df_d_fspd30_alphastackcash, df_d_fstpd30_alphastackcash,
#                       df_d_fpd0_credoscorecash, df_d_fpd10_credoscorecash, df_d_fpd30_credoscorecash, df_d_fspd30_credoscorecash, df_d_fstpd30_credoscorecash,
#                       df_d_fpd0_appscorecash, df_d_fpd10_appscorecash, df_d_fpd30_appscorecash, df_d_fspd30_appscorecash, df_d_fstpd30_appscorecash,
#                       df_d_fpd0_betademocash, df_d_fpd10_betademocash, df_d_fpd30_betademocash, df_d_fspd30_betademocash, df_d_fstpd30_betademocash,
#                       df_d_fpd0_betastackcash, df_d_fpd10_betastackcash, df_d_fpd30_betastackcash, df_d_fspd30_betastackcash, df_d_fstpd30_betastackcash,
#                       ], ignore_index =True)

# print(f"The shape of concatenated Fact and dimension table is :\ {factalldf.shape} & {dimalldf.shape}")

# %%
sq = """ 
create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.fact_cashtest1 as 
select * from 
(select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphaciccash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphastackcash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betacredocash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_appscorecash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademocash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betastackcash_test1
)
"""

client.query(sq).result()


# %%
sq = """ 
create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_cashtest1 as 
select * from 
(select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphaciccash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphastackcash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betacredocash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_appscorecash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betademocash_test1
union all
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betastackcash_test1
)
"""

client.query(sq).result()

# %%
sq = """
create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.fact_cash_combined1 as 
select distinct * from 
(select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_cashtest1
union all 
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_cashtrain1
)
;
"""
client.query(sq).result()


# %%
sq = """
create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_cash_combined1 as 
select distinct * from 
(select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_cashtest1
union all 
select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_cashtrain1
)
;
"""
client.query(sq).result()

# %% [markdown]
# # End

# %%
