import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.sil_all_model import run_sil_monthly
from models.cash_all_model import run_cash_monthly
from config_old import Config

def model_factory(model_name):
    model_mapping = {
        'sil_model': run_sil_monthly,
        'cash_model': run_cash_monthly
    }
    return model_mapping.get(model_name)

def run_models():
    for model_name in Config.MODEL_CONFIG:
        model_func = model_factory(model_name)
        if model_func:
            print(f"Running {model_name}...")
            model_func()  

if __name__ == "__main__":
    run_models()
