"""
Orchestrates the Calibration Monitoring Train-tree scripts (Cash Train +
Sil Train), structurally identical to execute_all_train.py's
ThreadPoolExecutor pattern. Kept SEPARATE from execute_all_calibration.py
(the Test-tree orchestrator) and from execute_all_train.py (Gini) for the
same reasons documented in execute_all_calibration.py's docstring.

SEQUENCING REQUIREMENT: run this script BEFORE execute_all_calibration.py.
Calibration_calculation_test_cash_consolidated.py's Test+Train combine step
reads calibration_summary_cashtrain / calibration_band_cashtrain, which
this script's consolidated step produces.
"""
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CALIBRATION_ROOT = r"D:\Model_Monitoring_Individual_scripts\Calibration_Monitoring"

PY_FILES = [
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_alpha_cash_stack_model.py",
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_beta_cash_stack_credo_score.py",
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_Cash_beta_cash_appscore_model.py",
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_Cash_beta_cash_demo_model.py",
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_Cash_beta_cash_stack_model.py",
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_Cash_beta_event_model_cash.py",
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_Cash_cic_model_cash.py",
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_Cash_transaction_model.py",
    rf"{CALIBRATION_ROOT}\Sil_Train_Models\Sil_Train_Models_Executables\Calibration_calculation_train_sil_alpha_Stackmodel_sil.py",
    rf"{CALIBRATION_ROOT}\Sil_Train_Models\Sil_Train_Models_Executables\Calibration_calculation_train_sil_beta_app_score_sil.py",
    rf"{CALIBRATION_ROOT}\Sil_Train_Models\Sil_Train_Models_Executables\Calibration_calculation_train_sil_beta_sil_demo_score.py",
    rf"{CALIBRATION_ROOT}\Sil_Train_Models\Sil_Train_Models_Executables\Calibration_calculation_train_sil_beta_sil_stack_score_model.py",
    rf"{CALIBRATION_ROOT}\Sil_Train_Models\Sil_Train_Models_Executables\Calibration_calculation_train_sil_beta_stack_model_sil_credo_score.py",
    rf"{CALIBRATION_ROOT}\Sil_Train_Models\Sil_Train_Models_Executables\Calibration_calculation_train_sil_cic_model_sil.py",
]

CONSOLIDATED_FILES = [
    rf"{CALIBRATION_ROOT}\Cash_Train_Models\Cash_Train_Model_Executables\Calibration_Calculation_Train_Cash_consolidated.py",
    rf"{CALIBRATION_ROOT}\Sil_Train_Models\Sil_Train_Models_Executables\Calibration_calculation_train_sil_consolidate.py",
]

# -----------------------------------
# Shared status tracking
# -----------------------------------
status_lock = threading.Lock()
job_status = {
    file: {"state": "QUEUED", "start": None}
    for file in PY_FILES + CONSOLIDATED_FILES
}


def print_status_table():
    with status_lock:
        print("\n" + "-" * 80)
        print(f"{'File':55} {'State':10} {'Elapsed':>10}")
        for file, info in job_status.items():
            elapsed = f"{time.time() - info['start']:.0f}s" if info["start"] else "-"
            print(f"{file.split(chr(92))[-1][:55]:55} {info['state']:10} {elapsed:>10}")
        print("-" * 80 + "\n")


def status_monitor(stop_event, interval=30):
    while not stop_event.wait(interval):
        print_status_table()


def build_command(file_name):
    return [sys.executable, file_name]


def run_python_file(file_name):
    with status_lock:
        job_status[file_name]["state"] = "RUNNING"
        job_status[file_name]["start"] = time.time()

    print(f"Starting {file_name}")

    result = subprocess.run(
        build_command(file_name),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        with status_lock:
            job_status[file_name]["state"] = "FAILED"
        raise Exception(
            f"""
            Failed: {file_name}

            Error:
            {result.stderr}
            """
        )

    with status_lock:
        job_status[file_name]["state"] = "DONE"

    print(f"Completed {file_name}")


def main():
    failed = False

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=status_monitor, args=(stop_event,), daemon=True)
    monitor_thread.start()

    with ThreadPoolExecutor(max_workers=6) as executor:

        futures = [
            executor.submit(run_python_file, file)
            for file in PY_FILES
        ]

        for future in as_completed(futures):
            try:
                future.result()

            except Exception as e:
                print(e)
                failed = True

    stop_event.set()
    print_status_table()

    # -----------------------------------
    # Run consolidated files in parallel, only if all individual jobs passed
    # -----------------------------------
    if not failed:

        print("All parallel calibration train jobs completed successfully")

        with ThreadPoolExecutor(max_workers=len(CONSOLIDATED_FILES)) as executor:

            consolidated_futures = {
                executor.submit(run_python_file, file): file
                for file in CONSOLIDATED_FILES
            }

            for future in as_completed(consolidated_futures):
                file = consolidated_futures[future]
                try:
                    future.result()
                    print(f"Consolidated script completed successfully: {file}")

                except Exception as e:
                    print(f"Consolidated script failed: {file}")
                    print(e)

        print_status_table()

    else:
        print("Skipping consolidated files because some jobs failed")


if __name__ == "__main__":
    main()
