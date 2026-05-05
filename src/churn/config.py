from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_DATA_PATH = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
BASELINE_RESULTS_PATH = MODELS_DIR / "baseline_results.json"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2