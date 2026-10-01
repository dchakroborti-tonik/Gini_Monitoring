# %% [markdown]
# # CASH Train Calibration Monitoring — Consolidated
#
# Unions the 8 per-model calibration_summary_*_train / calibration_band_*_train
# tables into calibration_summary_cashtrain / calibration_band_cashtrain --
# mirroring Gini_Calculation_Train_Cash_consolidated.py's UNION ALL pattern.
# Must run AFTER all 8 per-model Cash Train scripts have already populated
# their tables, and BEFORE the Cash Test consolidated script (which reads
# these tables back for the Test+Train combine).

# %%
import os
from google.cloud import bigquery

path = r'C:\Users\Dwaipayan\AppData\Roaming\gcloud\application_default_credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
client = bigquery.Client(project='prj-prod-dataplatform')
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"

DATASET = "prj-prod-dataplatform.dap_ds_poweruser_playground"

CASH_TRAIN_MODELKEYS = [
    "alphastackcash", "appscorecash", "betademocash", "betastackcash",
    "betacredocash", "betaeventcash", "alphaciccash", "transaction",
]

# %% [markdown]
# ## Union the 8 Cash Train model tables

# %%
for kind in ("summary", "band"):
    union_sql = " UNION ALL ".join(
        f"SELECT * FROM `{DATASET}.calibration_{kind}_{mk}_train`" for mk in CASH_TRAIN_MODELKEYS
    )
    sq = f"""
    CREATE OR REPLACE TABLE `{DATASET}.calibration_{kind}_cashtrain` AS
    SELECT DISTINCT * FROM ({union_sql})
    """
    client.query(sq).result()
    print(f"created/refreshed calibration_{kind}_cashtrain")

print("Cash Train calibration consolidation completed")
