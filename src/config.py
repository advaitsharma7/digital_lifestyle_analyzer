from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "smart_synthetic_data.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed_lifestyle_data.csv"
MODELS_DIR = ROOT_DIR / "models"
STRESS_MODEL_PATH = MODELS_DIR / "stress_model.pkl"
PRODUCTIVITY_MODEL_PATH = MODELS_DIR / "productivity_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
PRODUCTIVITY_PREPROCESSOR_PATH = MODELS_DIR / "productivity_preprocessor.pkl"
CLUSTER_BUNDLE_PATH = MODELS_DIR / "cluster_bundle.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
SCATTER_SAMPLE_SIZE = 4000

NUMERIC_FEATURES = [
    "Age",
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
    "App_Usage_Count",
]

CATEGORICAL_FEATURES = ["Gender", "Device_Type"]

MODEL_FEATURES = [
    "Age",
    "Gender",
    "Device_Type",
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
    "App_Usage_Count",
]

PRODUCTIVITY_MODEL_FEATURES = [
    "Age",
    "Gender",
    "Device_Type",
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
]

PRODUCTIVITY_NUMERIC_FEATURES = [
    "Age",
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
]

EXPLAINABILITY_FEATURES = [
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
]

APP_USAGE_INPUT_FEATURES = [
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Weekend_Screen_Time_Hours",
]

CLUSTER_FEATURES = [
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
    "Stress_Level",
    "Work_Productivity_Score",
]

PERCENTILE_FIELDS = [
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
]

CORRELATION_FIELDS = [
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Sleep_Hours",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
    "App_Usage_Count",
    "Stress_Level",
    "Work_Productivity_Score",
]

UI_FIELD_LABELS = {
    "Age": "Age",
    "Gender": "Gender",
    "Device_Type": "Device Type",
    "Daily_Phone_Hours": "Daily Phone Hours",
    "Social_Media_Hours": "Social Media Hours",
    "Sleep_Hours": "Sleep Hours",
    "Caffeine_Intake_Cups": "Caffeine Intake",
    "Weekend_Screen_Time_Hours": "Weekend Screen Time",
    "App_Usage_Count": "App Usage Count",
    "Stress_Level": "Stress Level",
    "Work_Productivity_Score": "Productivity Score",
}

STRESS_SCALE_LABELS = {
    1: "Low",
    2: "Light",
    3: "Moderate",
    4: "High",
    5: "Very High",
}

LIFESTYLE_SCORE_WEIGHTS = {
    "Sleep": 0.30,
    "Screen Time": 0.25,
    "Stress": 0.25,
    "Productivity": 0.20,
}

DEFAULT_REQUIRED_INPUTS = {
    "Daily_Phone_Hours": 5.0,
    "Social_Media_Hours": 2.5,
    "Sleep_Hours": 7.0,
    "Caffeine_Intake_Cups": 2.0,
    "Weekend_Screen_Time_Hours": 7.0,
}

RADAR_AXES = ["Sleep", "Screen Time", "Social Media", "Stress", "Productivity"]
