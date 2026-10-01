# %% [markdown]
# # SIL Train Calibration Monitoring — Consolidated
#
# Unions the 6 per-model calibration_summary_*_train / calibration_band_*_train
# tables into calibration_summary_siltrain / calibration_band_siltrain --
# mirroring Gini_calculation_train_sil_consolidate.py's UNION ALL pattern.
# No Sil Test+Train combine, matching Gini's current asymmetry.

# %%
import os
from google.cloud import bigquery

path = r'C:\Users\Dwaipayan\AppData\Roaming\gcloud\application_default_credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
client = bigquery.Client(project='prj-prod-dataplatform')
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"

DATASET = "prj-prod-dataplatform.dap_ds_poweruser_playground"

SIL_TRAIN_MODELKEYS = [
    "alphastacksil", "appscoresil", "betademosil", "credosil", "cicsil",
]
# NOTE: betastacksil (beta_sil_stack_score_model) was already built as the
# tree's reference model with its own table calibration_summary_betastacksil_train.

SIL_TRAIN_MODELKEYS_FULL = ["betastacksil"] + SIL_TRAIN_MODELKEYS

# %% [markdown]
# ## Union the 6 Sil Train model tables

# %%
for kind in ("summary", "band"):
    union_sql = " UNION ALL ".join(
        f"SELECT * FROM `{DATASET}.calibration_{kind}_{mk}_train`" for mk in SIL_TRAIN_MODELKEYS_FULL
    )
    sq = f"""
    CREATE OR REPLACE TABLE `{DATASET}.calibration_{kind}_siltrain` AS
    SELECT DISTINCT * FROM ({union_sql})
    """
    client.query(sq).result()
    print(f"created/refreshed calibration_{kind}_siltrain")

print("Sil Train calibration consolidation completed")
