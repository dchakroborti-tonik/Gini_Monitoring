import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------
# List your parallel python files
# -----------------------------------
PY_FILES = [
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Cash_Test_Models\Cash_Test_Models_Executables\Gini_Alpha_cash_stack_test.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Cash_Test_Models\Cash_Test_Models_Executables\Gini_calculation_test_cash_beta_cash_appscore_model.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Cash_Test_Models\Cash_Test_Models_Executables\Gini_calculation_test_cash_beta_cash_demo_model.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Cash_Test_Models\Cash_Test_Models_Executables\Gini_calculation_test_cash_beta_cash_stack_model_credo_score.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Cash_Test_Models\Cash_Test_Models_Executables\Gini_calculation_test_cash_beta_cash_stack_model.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Cash_Test_Models\Cash_Test_Models_Executables\Gini_calculation_test_cash_beta_events_model_cash.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Cash_Test_Models\Cash_Test_Models_Executables\Gini_calculation_test_cash_cic_model_cash.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Sil_Test_Models\Sil_Test_Models_Executables\Gini_calculation_test_sil_alpha_stack_model_sil.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Sil_Test_Models\Sil_Test_Models_Executables\Gini_calculation_test_sil_Beta App Score SIL.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Sil_Test_Models\Sil_Test_Models_Executables\Gini_calculation_test_sil_beta_stack_model_sil_credo_score.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Sil_Test_Models\Sil_Test_Models_Executables\Gini_calculation_test_sil_BetaSILDemoScore.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Sil_Test_Models\Sil_Test_Models_Executables\Gini_calculation_test_sil_BetaSILSTACKScoreModel.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Sil_Test_Models\Sil_Test_Models_Executables\Gini_calculation_test_sil_cic_model_sil.py",
]

CONSOLIDATED_FILES = [
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Cash_Test_Models\Cash_Test_Models_Executables\Gini_calculation_test_cash_consolidated.py",
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring_Individual_scripts\Gini_Monitoring\Sil_Test_Models\Sil_Test_Models_Executables\Gini_calculation_test_sil_consolidate.py",
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


# -----------------------------------
# Function to run individual scripts
# -----------------------------------
def build_command(file_name):
    if file_name.lower().endswith(".ipynb"):
        return [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            file_name,
        ]
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

    return file_name


# -----------------------------------
# Run parallel jobs
# -----------------------------------
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

        print("All parallel jobs completed successfully")

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


# -----------------------------------
# Entry point
# -----------------------------------
if __name__ == "__main__":
    main()