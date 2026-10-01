# %% [markdown]
# # CASH Test Calibration Monitoring — beta_cash_stack_model_transaction_score

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

MODEL_NAME = "Transaction_Score_Cash_Model"
MODEL_DISPLAY_NAME = "Transaction_Score_Cash_Model"
SCORE_COL = "trx_score"

DUCKDB_PATH = CALIBRATION_ROOT / "calibration_cache_betatransactionscorecash_test.db"
RAW_TABLE_NAME = "raw_betatransactionscorecash_test"
GINI_CACHE_TABLE_NAME = "gini_betatransactionscorecash_test"

SUMMARY_TABLE_ID = "prj-prod-dataplatform.dap_ds_poweruser_playground.calibration_summary_betatransactionscorecash_test"
BAND_TABLE_ID = "prj-prod-dataplatform.dap_ds_poweruser_playground.calibration_band_betatransactionscorecash_test"
GINI_FACT_TABLE_ID = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betatransactionscorecash_test2"
LOCAL_OUTPUT_DIR = CALIBRATION_ROOT / "local_output" / "Cash_Test" / "betatransactionscorecash"

# %% [markdown]
# ## SQL pull
#
# Copied verbatim from
# `Gini_Monitoring/Cash_Test_Models/Cash_Test_Models_Executables/Gini_calculation_test_cash_beta_cash_stack_model_transaction_score.py`
# (lines 730-896). trx_score is parsed from the calcFeature JSON blob.

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
      ) trench1_seg
      ON trench1_seg.offer_id = loan.offer_id
  ),
base as
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  coalesce(safe_cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.trx_score')AS FLOAT64), safe_cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.trx_score')AS FLOAT64)) AS trx_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  del.deffpd10,
  del.flg_mature_fpd10,
  del.deffpd30,
  del.flg_mature_fpd30,
  del.deffspd30,
  del.flg_mature_fspd_30,
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
  and del.flg_mature_fpd0 = 1
  )
  select *
  from base
  where trx_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
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
