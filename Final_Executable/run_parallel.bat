@echo off
REM ==========================================
REM Parallel Gini Jobs - Anaconda Python
REM ==========================================

REM ---- Base paths ----
SET BASE_DIR=D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring\Model_Monitoring_Consolidated\Final_Executable
SET PYTHON_EXE=C:\ProgramData\anaconda3\python.exe

REM ---- Logs ----
SET LOG_MASTER=%BASE_DIR%\master.log
SET LOG_CASH=%BASE_DIR%\cash_gini.log
SET LOG_SIL=%BASE_DIR%\sil_gini.log

REM ---- GCP credentials ----
SET GOOGLE_APPLICATION_CREDENTIALS=%BASE_DIR%\gcp_key.json

echo ========================================= >> "%LOG_MASTER%"
echo Master job started at %DATE% %TIME% >> "%LOG_MASTER%"

REM ---- Start CASH job (parallel) ----
start "CASH_GINI" cmd /c ^
""%PYTHON_EXE%" "%BASE_DIR%\Gini_cash_v1.py" >> "%LOG_CASH%" 2>&1"

REM ---- Start SIL job (parallel) ----
start "SIL_GINI" cmd /c ^
""%PYTHON_EXE%" "%BASE_DIR%\Gini_sil_v1.py" >> "%LOG_SIL%" 2>&1"

echo CASH and SIL jobs triggered at %TIME% >> "%LOG_MASTER%"