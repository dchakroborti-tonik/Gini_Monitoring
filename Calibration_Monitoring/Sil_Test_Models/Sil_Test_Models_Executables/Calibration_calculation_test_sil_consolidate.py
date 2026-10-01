# %% [markdown]
# # SIL Test Calibration Monitoring — Consolidated
#
# Unions the 6 per-model calibration_summary_*_test / calibration_band_*_test
# tables into calibration_summary_siltest / calibration_band_siltest --
# mirroring Gini_calculation_test_sil_consolidate.py's UNION ALL pattern.
# No Sil Test+Train combine, matching Gini's current asymmetry (Sil doesn't
# combine Test+Train either).

# %%
import os
from google.cloud import bigquery

path = r'C:\Users\Dwaipayan\AppData\Roaming\gcloud\application_default_credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
client = bigquery.Client(project='prj-prod-dataplatform')
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"

DATASET = "prj-prod-dataplatform.dap_ds_poweruser_playground"

SIL_TEST_MODELKEYS = [
    "appscoresil", "betademosil", "alphastacksil", "credosil", "cicsil",
]
# NOTE: betastacksil (BetaSILSTACKScoreModel) was already built as the tree's
# reference model with its own table calibration_summary_betastacksil_test.

SIL_TEST_MODELKEYS_FULL = ["betastacksil"] + SIL_TEST_MODELKEYS

# %% [markdown]
# ## Union the 6 Sil Test model tables

# %%
for kind in ("summary", "band"):
    union_sql = " UNION ALL ".join(
        f"SELECT * FROM `{DATASET}.calibration_{kind}_{mk}_test`" for mk in SIL_TEST_MODELKEYS_FULL
    )
    sq = f"""
    CREATE OR REPLACE TABLE `{DATASET}.calibration_{kind}_siltest` AS
    SELECT DISTINCT * FROM ({union_sql})
    """
    client.query(sq).result()
    print(f"created/refreshed calibration_{kind}_siltest")

print("Sil Test calibration consolidation completed")
