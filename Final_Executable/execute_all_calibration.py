"""
Orchestrates the Calibration Monitoring Test-tree scripts (Cash Test + Sil
Test), structurally identical to execute_all.py's ThreadPoolExecutor
pattern. Kept as a SEPARATE orchestrator from execute_all.py on purpose:
calibration's Gini read-back depends on the Gini fact tables Final_Executable
/execute_all.py already populates, and a bug in a new calibration script
must never block the existing, trusted Gini run.

SEQUENCING REQUIREMENT: run execute_all_calibration_train.py BEFORE this
script. Calibration_calculation_test_cash_consolidated.py's second step
unions calibration_summary_cashtrain / calibration_band_cashtrain into
calibration_summary_cash_combined / calibration_band_cash_combined, which
requires the Train tree's per-model tables (and its own consolidated
script) to have already run -- same dependency Gini's own Test/Train
consolidated scripts already have.
"""
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CALIBRATION_ROOT = r"D:\Model_Monitoring_Individual_scripts\Calibration_Monitoring"

PY_FILES = [
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_Alpha_cash_stack_test.py",
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_calculation_test_cash_beta_cash_appscore_model.py",
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_calculation_test_cash_beta_cash_demo_model.py",
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_calculation_test_cash_beta_cash_stack_model.py",
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_calculation_test_cash_beta_cash_stack_model_credo_score.py",
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_calculation_test_cash_beta_cash_stack_model_transaction_score.py",
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_calculation_test_cash_beta_events_model_cash.py",
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_calculation_test_cash_cic_model_cash.py",
    rf"{CALIBRATION_ROOT}\Sil_Test_Models\Sil_Test_Models_Executables\Calibration_calculation_test_sil_Beta App Score SIL.py",
    rf"{CALIBRATION_ROOT}\Sil_Test_Models\Sil_Test_Models_Executables\Calibration_calculation_test_sil_BetaSILDemoScore.py",
    rf"{CALIBRATION_ROOT}\Sil_Test_Models\Sil_Test_Models_Executables\Calibration_calculation_test_sil_BetaSILSTACKScoreModel.py",
    rf"{CALIBRATION_ROOT}\Sil_Test_Models\Sil_Test_Models_Executables\Calibration_calculation_test_sil_alpha_stack_model_sil.py",
    rf"{CALIBRATION_ROOT}\Sil_Test_Models\Sil_Test_Models_Executables\Calibration_calculation_test_sil_beta_stack_model_sil_credo_score.py",
    rf"{CALIBRATION_ROOT}\Sil_Test_Models\Sil_Test_Models_Executables\Calibration_calculation_test_sil_cic_model_sil.py",
]

CONSOLIDATED_FILES = [
    rf"{CALIBRATION_ROOT}\Cash_Test_Models\Cash_Test_Models_Executables\Calibration_calculation_test_cash_consolidated.py",
    rf"{CALIBRATION_ROOT}\Sil_Test_Models\Sil_Test_Models_Executables\Calibration_calculation_test_sil_consolidate.py",
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

        print("All parallel calibration jobs completed successfully")

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
