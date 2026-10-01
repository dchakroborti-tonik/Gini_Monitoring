# %% [markdown]
# # CASH Test Calibration Monitoring — Consolidated
#
# Unions the 8 per-model calibration_summary_*_test / calibration_band_*_test
# tables into calibration_summary_cashtest / calibration_band_cashtest, then
# unions those with the Cash Train consolidated tables into
# calibration_summary_cash_combined / calibration_band_cash_combined --
# mirroring Gini_calculation_test_cash_consolidated.py's exact UNION ALL
# pattern (including the Test-consolidated script being the one that also
# performs the Test+Train combine, same as Gini). Must run AFTER all 8
# per-model Cash Test scripts AND the Cash Train consolidated script have
# already populated their tables.

# %%
import os
from google.cloud import bigquery

path = r'C:\Users\Dwaipayan\AppData\Roaming\gcloud\application_default_credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
client = bigquery.Client(project='prj-prod-dataplatform')
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"

DATASET = "prj-prod-dataplatform.dap_ds_poweruser_playground"

CASH_TEST_MODELKEYS = [
    "alphastackcash", "appscorecash", "betademocash", "betastackcash",
    "betacredocash", "betatransactionscorecash", "betaeventcash", "alphaciccash",
]

# %% [markdown]
# ## Union the 8 Cash Test model tables

# %%
for kind in ("summary", "band"):
    union_sql = " UNION ALL ".join(
        f"SELECT * FROM `{DATASET}.calibration_{kind}_{mk}_test`" for mk in CASH_TEST_MODELKEYS
    )
    sq = f"""
    CREATE OR REPLACE TABLE `{DATASET}.calibration_{kind}_cashtest` AS
    SELECT DISTINCT * FROM ({union_sql})
    """
    client.query(sq).result()
    print(f"created/refreshed calibration_{kind}_cashtest")

# %% [markdown]
# ## Combine Cash Test + Cash Train
#
# Requires Calibration_Calculation_Train_Cash_consolidated.py to have
# already populated calibration_summary_cashtrain / calibration_band_cashtrain.

# %%
for kind in ("summary", "band"):
    sq = f"""
    CREATE OR REPLACE TABLE `{DATASET}.calibration_{kind}_cash_combined` AS
    SELECT DISTINCT * FROM (
      SELECT * FROM `{DATASET}.calibration_{kind}_cashtest`
      UNION ALL
      SELECT * FROM `{DATASET}.calibration_{kind}_cashtrain`
    )
    """
    client.query(sq).result()
    print(f"created/refreshed calibration_{kind}_cash_combined")

print("Cash Test calibration consolidation completed")
