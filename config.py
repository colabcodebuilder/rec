"""
Configuration constants for the Healthcare Recommendation System.
"""
import os
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Base directory and data subdirectory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Database
DB_NAME = "healthcare.db"
DB_PATH = os.path.join(DATA_DIR, DB_NAME)

# Patient pool
N_PATIENTS = 100
PATIENT_PREFIX = "Patient_"
PATIENT_ZFILL = 3

# Provider pool
N_PROVIDERS = 30
PROVIDER_PREFIX = "Doctor_"
PROVIDER_ZFILL = 3

# Medical conditions
CONDITIONS = [
    "Diabetes", "Hypertension", "Asthma",
    "Cardiac_Issue", "Kidney_Disease", "Migraine", "Arthritis"
]

# Specialty → Conditions mapping
SPECIALITY_MAP = {
    "Cardiologist": ["Hypertension", "Cardiac_Issue"],
    "Neurologist": ["Migraine"],
    "General_Physician": ["Hypertension", "Diabetes", "Asthma", "Migraine", "Arthritis"],
    "Endocrinologist": ["Diabetes"],
    "Nephrologist": ["Kidney_Disease"],
    "Pulmonologist": ["Asthma"]
}

# Age range for patients
AGE_MIN, AGE_MAX = 18, 80

# Score ranges
RISK_SCORE_RANGE = (1.0, 5.0)
URGENCY_SCORE_RANGE = (1.0, 5.0)
EXPERTISE_SCORE_RANGE = (1.0, 5.0)
SUCCESS_RATE_RANGE = (0.7, 0.99)
AVAILABILITY_SCORE_RANGE = (0.6, 1.0)
LOAD_SCORE_RANGE = (0.1, 0.9)

# KNN parameters
KNN_NEIGHBORS = 10
KNN_METRIC = 'cosine'
KNN_ALGORITHM = 'brute'

# Recommendation count
TOP_K_RECS = 5

# Scoring weights for fallback
SCORE_WEIGHTS = {
    "jaccard": 0.35,
    "risk": 0.20,
    "urgency": 0.15,
    "expertise": 0.15,
    "success": 0.10,
    "availability": 0.05
}

# Evaluation
NDCG_K = 5
GROUND_TRUTH_THRESHOLDS = {
    "excellent": 16,
    "good": 12,
    "fair": 8
}

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Feature names for SHAP
FEATURE_NAMES = [
    "jaccard_similarity",
    "risk_score",
    "urgency_score",
    "expertise_score",
    "success_rate",
    "availability_score",
    "inverse_load"
]

# ----------------------------------------------
# MLflow configuration
# ----------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.path.join(DATA_DIR, 'mlflow.db')}"
)
MLFLOW_BACKEND_STORE_URI = os.environ.get(
    "MLFLOW_BACKEND_STORE_URI",
    f"sqlite:///{os.path.join(DATA_DIR, 'mlflow.db')}"
)
MLFLOW_DEFAULT_ARTIFACT_ROOT = os.environ.get(
    "MLFLOW_DEFAULT_ARTIFACT_ROOT",
    os.path.join(BASE_DIR, "mlruns")
)
MLFLOW_HOST = os.environ.get("MLFLOW_HOST", "127.0.0.1")
MLFLOW_PORT = int(os.environ.get("MLFLOW_PORT", "5000"))
MLFLOW_EXPERIMENT_NAME = "healthcare_recommendation"

# Ensure MLflow artifact directory exists
os.makedirs(MLFLOW_DEFAULT_ARTIFACT_ROOT, exist_ok=True)