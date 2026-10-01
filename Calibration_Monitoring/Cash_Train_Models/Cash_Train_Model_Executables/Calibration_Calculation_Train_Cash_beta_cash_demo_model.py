# %% [markdown]
# # CASH Train Calibration Monitoring — beta_cash_demo

# %% [markdown]
# ## Imports, auth, calibration_utils

# %%
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

path = r'C:\Users\Dwaipayan\AppData\Roaming\gcloud\application_default_credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
client = bigquery.Client(project='prj-prod-dataplatform')
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"

warnings.filterwarnings("ignore")

try:
    _here = Path(__file__).resolve()
except NameError:
    _here = Path.cwd()
CALIBRATION_ROOT = _here.parents[2] if _here.parent.name.endswith("_Executables") else _here.parents[1]
sys.path.insert(0, str(CALIBRATION_ROOT))

from calibration_utils import (
    BAD_RATE_TARGETS,
    load_or_pull,
    compute_calibration_bands,
    fetch_gini_for_calibration,
    apply_drift_based_status,
    write_outputs,
)

# %% [markdown]
# ## Config

# %%
FORCE_REFRESH = False

MODEL_NAME = "beta_demo_model_cash"
MODEL_DISPLAY_NAME = "beta_demo_model_cash"
SCORE_COL = "Beta_Cash_Demo_Score"

DUCKDB_PATH = CALIBRATION_ROOT / "calibration_cache_betademocash_train.db"
RAW_TABLE_NAME = "raw_betademocash_train"
GINI_CACHE_TABLE_NAME = "gini_betademocash_train"

SUMMARY_TABLE_ID = "prj-prod-dataplatform.dap_ds_poweruser_playground.calibration_summary_betademocash_train"
BAND_TABLE_ID = "prj-prod-dataplatform.dap_ds_poweruser_playground.calibration_band_betademocash_train"
GINI_FACT_TABLE_ID = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademocash_train2"
LOCAL_OUTPUT_DIR = CALIBRATION_ROOT / "local_output" / "Cash_Train" / "betademocash"

# %% [markdown]
# ## SQL pull
#
# Copied verbatim from
# `Gini_Monitoring/Cash_Train_Models/Cash_Train_Model_Executables/Gini_Calculation_Train_Cash_beta_cash_demo_model.py`
# (lines 1965-2213, the FPD0 section -- NOTE: this Gini file's first ~1950
# lines are leftover appscore-model code never cleaned up; the actual
# Beta-Cash-Demo-Model section starts at facttable_id =
# fact_betademocash_train2, line 1955, which is what's reused here).

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
      del.deffpd10,
      del.flg_mature_fpd10,
      del.deffpd30,
      del.flg_mature_fpd30,
      del.deffspd30,
      del.flg_mature_fspd_30,
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

dfd = load_or_pull(client, sq, DUCKDB_PATH, RAW_TABLE_NAME, force_refresh=FORCE_REFRESH)
print(f"The shape of the dataframe is:\t {dfd.shape}")
dfd.head()

# %% [markdown]
# ## Score coercion

# %%
dfd[SCORE_COL] = pd.to_numeric(dfd[SCORE_COL], errors="coerce")

# %% [markdown]
# ## Gini read-back (all 5 targets, one cached query)

# %%
gini_all = fetch_gini_for_calibration(
    client, GINI_FACT_TABLE_ID, MODEL_DISPLAY_NAME,
    DUCKDB_PATH, GINI_CACHE_TABLE_NAME, force_refresh=FORCE_REFRESH,
)

# %% [markdown]
# ## Calibration computation across all 5 bad-rate targets

# %%
summary_frames, band_frames = [], []
for label_col, maturity_col, name in BAD_RATE_TARGETS:
    df_target = dfd[dfd[maturity_col] == 1].copy()
    df_target[SCORE_COL] = pd.to_numeric(df_target[SCORE_COL], errors="coerce")

    summary_df, band_df = compute_calibration_bands(
        df_target,
        score_col=SCORE_COL,
        label_col=label_col,
        maturity_col=maturity_col,
        bad_rate_type=name,
        model_name=MODEL_NAME,
        model_version_col="modelVersionId",
        trench_col="trenchCategory",
        ostype_col="osType",
        account_id_col="digitalLoanAccountId",
        full_cohort_df=dfd,
    )

    target_gini = gini_all[gini_all["bad_rate_type"] == name] if not gini_all.empty else gini_all
    if not target_gini.empty:
        summary_df = summary_df.drop(columns=["gini"]).merge(
            target_gini.drop(columns=["bad_rate_type"]),
            how="left",
            on=["observation_period", "model_version", "trench_category", "os_type"],
        )

    summary_frames.append(summary_df)
    band_frames.append(band_df)
    print(f"{MODEL_NAME} calibration for {name}: {len(summary_df)} summary rows, {len(band_df)} band rows")

summary_all = pd.concat(summary_frames, ignore_index=True)
band_all = pd.concat(band_frames, ignore_index=True)

# %% [markdown]
# ## Drift-based calibration_status (see calibration_utils.apply_drift_based_status)

# %%
summary_all = apply_drift_based_status(summary_all)

# %% [markdown]
# ## Write outputs (BigQuery + local parquet/csv)

# %%
write_outputs(summary_all, band_all, client, SUMMARY_TABLE_ID, BAND_TABLE_ID, LOCAL_OUTPUT_DIR)
print(f"{MODEL_NAME} calibration monitoring completed")
