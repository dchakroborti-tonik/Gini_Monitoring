# Model Monitoring Individual Scripts

This repository contains individual Python scripts and Jupyter notebooks for monitoring machine learning models at Tonik, including Gini (discrimination) monitoring, calibration monitoring, PSI (Population Stability Index) analysis, and supporting data preparation utilities.

## 📋 Overview

The codebase supports monitoring for multiple model types (Cash, SIL) across different data splits (Train, Test) and model versions. It includes:

- **Gini Monitoring**: Calculate and track Gini coefficients (ROC AUC-based discrimination metrics) for model predictions
- **Calibration Monitoring**: Assess model calibration across different bad-rate targets (FPD0, FPD10, FPD30, FSPD30, FSTPD30) with banding analysis
- **PSI Analysis**: Population Stability Index calculations for feature drift detection
- **Data Preparation**: Utilities to create and prepare monitoring datasets from BigQuery
- **Vector Analysis**: DuckDB-based analysis of ML vector tables

## 🗂️ Project Structure

### Core Monitoring Modules

| Directory | Description |
|---|---|
| `Gini_Monitoring/` | Gini coefficient calculation scripts for discrimination monitoring |
| `Calibration_Monitoring/` | Calibration analysis scripts with shared utilities for monitoring model calibration |

### Supporting Components

| Directory | Description |
|---|---|
| `Common_Query/` | Shared utilities for data access and analysis (DuckDB, BigQuery, vector analysis) |
| `Final_Executable/` | Orchestrator scripts to run multiple monitoring jobs in parallel |
| `PSI_Data_Preparation/` | Data preparation notebooks for PSI calculations |
| `PSI_Training_Data_Preparation/` | Training data preparation for PSI analysis |
| `Gini_Training_Data_Preparation/` | Training data preparation for Gini monitoring |
| `Table_Creation_For_Each_Model_Version/` | Scripts to create monitoring tables per model version (organized by product type and version) |

### Data Files

| File | Description |
|---|---|
| `dd_ml_models_vector_details.duckdb` | DuckDB database containing ML model vector details |
| `models_meta_data.xlsx`, `models_meta_data_updated.xlsx` | Model metadata reference files |
| `Model_Version_Trench_Category_Tracker.xlsx` | Tracker for model versions and trench categories |

## 🔍 Key Components

### 1. Gini Monitoring (`Gini_Monitoring/`)

Calculates Gini coefficients using ROC AUC (Gini = 2*AUC - 1). Organized by model type and split:

- **Cash_Train_Models/** - Training set Gini calculations for Cash models
- **Cash_Test_Models/** - Test/holdout Gini calculations for Cash models
- **Sil_Train_Models/** - Training set Gini calculations for SIL models
- **Sil_Test_Models/** - Test/holdout Gini calculations for SIL models
- **Old_scripts_before_september232026/** - Historical/archived scripts

Each subfolder contains both Jupyter notebooks (`.ipynb`) with dated versions and Python executables (`Executables/*.py`) for production runs.

### 2. Calibration Monitoring (`Calibration_Monitoring/`)

Monitors model calibration across five bad-rate targets:
- FPD0 (First Payment Default 0)
- FPD10 (First Payment Default 10)
- FPD30 (First Payment Default 30)
- FSPD30 (First Severe Payment Default 30)
- FSTPD30 (First Structurally Troubled Payment Default 30)

**Key files:**
- `calibration_utils.py` - Shared calibration utilities (centralized to avoid duplication across 28+ model scripts). Provides `compute_calibration_bands()`, `fetch_gini_for_calibration()`, `write_outputs()`, and BigQuery/DuckDB caching via `load_or_pull()`.
- Model-specific scripts organized similarly to Gini (Cash/SIL, Train/Test) with both `.ipynb` and executable `.py` files

### 3. Final Executables (`Final_Executable/`)

Orchestration scripts for parallel execution of monitoring jobs:

- `execute_all.py` - Runs Gini test models in parallel with status monitoring and consolidated reporting
- `execute_all_train.py` - Parallel execution for training set Gini calculations
- `execute_all_calibration.py` - Parallel execution for calibration test models
- `execute_all_calibration_train.py` - Parallel execution for calibration training models
- `Gini_cash_v1.py`, `Gini_sil_v1.py` - Version-specific Gini calculation scripts

Uses `ThreadPoolExecutor` for concurrent execution with real-time status tracking.

### 4. Common Query Utilities (`Common_Query/`)

- `Ml_vector_analysis.py` - DuckDB-based analysis toolkit for ML vector tables. Provides functions for sanity checks, feature profiling, PSI calculation (numeric/categorical), binning utilities, and discrimination metrics
- `Ml_Vector_Pipeline.py` - Pipeline utilities for ML vector processing
- `Data_Duckdb_table_Check.ipynb` - Validation notebook for DuckDB tables

### 5. Table Creation (`Table_Creation_For_Each_Model_Version/`)

Creates monitoring tables for specific model versions. Organized by product (Cash/SIL) and version (V1, V1.1, V1.2, etc.). Contains Jupyter notebooks that generate BigQuery tables for each model version.

## 🛠️ Technology Stack

- **Python**: Core processing (pandas, numpy, scikit-learn)
- **BigQuery**: Data source for model predictions and outcomes
- **DuckDB**: Local caching and analytical queries
- **Jupyter Notebooks**: Interactive analysis and development
- **Google Cloud Platform**: Storage (GCS), BigQuery, authentication via service account/key files
- **Matplotlib/Seaborn**: Visualization

## ⚙️ Configuration

Scripts typically use Google Cloud authentication. Common patterns:
```python
path = r"C:\Users\Dwaipayan\AppData\Roaming\gcloud\application_default_credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
client = bigquery.Client(project="prj-prod-dataplatform")
```

## 🚀 Usage

### Running Individual Scripts

Execute any Python script directly:
```bash
python path/to/script.py
```

Or open and run Jupyter notebooks interactively.

### Running Batch Jobs

Use the orchestrator scripts in `Final_Executable/`:
```bash
cd Final_Executable
python execute_all.py  # Runs all Gini test model calculations in parallel
```

The orchestrators provide:
- Parallel execution with configurable concurrency
- Real-time status monitoring (every 30 seconds)
- Separate handling of consolidated vs. individual model scripts
- Error handling and output capture

## 📝 Notes

- The codebase has both `.py` executables (for production/automation) and `.ipynb` notebooks (for exploration and development)
- Many scripts follow a consistent structure with imports, BigQuery setup, utility functions, and main execution logic
- The `calibration_utils.py` centralizes shared math to reduce duplication and maintenance overhead
- Old scripts are preserved in `Old_scripts_before_september232026/` for reference
- Local DuckDB caching is used to minimize BigQuery query costs in calibration workflows