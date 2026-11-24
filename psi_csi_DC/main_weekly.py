import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from config_old import Config
from config import Config
from models.cash_all_model_weekly import run_cash_weekly
from models.sil_all_model_weekly import run_sil_weekly


def model_factory(model_name):
    model_mapping = {
        'sil_model': run_sil_weekly,
        'cash_model': run_cash_weekly
    }
    return model_mapping.get(model_name)

def run_models():
    for model_name in Config.MODEL_CONFIG:
        model_func = model_factory(model_name)
        if model_func:
            print(f"Running {model_name} (weekly)...")
            model_func()

if __name__ == "__main__":
    run_models()
