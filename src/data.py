from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH

import pandas as pd


NUMERIC_COLUMNS = [
    "Age",
    "Daily_Phone_Hours",
    "Social_Media_Hours",
    "Work_Productivity_Score",
    "Sleep_Hours",
    "Stress_Level",
    "App_Usage_Count",
    "Caffeine_Intake_Cups",
    "Weekend_Screen_Time_Hours",
]


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in dataset.columns:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    return dataset


@lru_cache(maxsize=1)
def load_raw_dataset() -> pd.DataFrame:
    return clean_dataset(pd.read_csv(RAW_DATA_PATH))


@lru_cache(maxsize=1)
def load_processed_dataset() -> pd.DataFrame:
    path = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else RAW_DATA_PATH
    return clean_dataset(pd.read_csv(path))


def compute_defaults(dataset: pd.DataFrame) -> dict[str, float | str]:
    return {
        "Age": int(round(dataset["Age"].mean())),
        "Gender": dataset["Gender"].mode(dropna=True).iloc[0],
        "Device_Type": dataset["Device_Type"].mode(dropna=True).iloc[0],
        "Daily_Phone_Hours": round(float(dataset["Daily_Phone_Hours"].mean()), 1),
        "Social_Media_Hours": round(float(dataset["Social_Media_Hours"].mean()), 1),
        "Sleep_Hours": round(float(dataset["Sleep_Hours"].mean()), 1),
        "Caffeine_Intake_Cups": round(float(dataset["Caffeine_Intake_Cups"].mean()), 1),
        "Weekend_Screen_Time_Hours": round(
            float(dataset["Weekend_Screen_Time_Hours"].mean()), 1
        ),
        "App_Usage_Count": int(round(dataset["App_Usage_Count"].mean())),
    }


def compute_options(dataset: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "Gender": sorted(dataset["Gender"].dropna().astype(str).unique().tolist()),
        "Device_Type": sorted(
            dataset["Device_Type"].dropna().astype(str).unique().tolist()
        ),
    }


def dataset_ranges(
    dataset: pd.DataFrame, columns: list[str]
) -> dict[str, dict[str, float]]:
    return {
        column: {
            "min": float(dataset[column].min()),
            "max": float(dataset[column].max()),
            "mean": float(dataset[column].mean()),
        }
        for column in columns
        if column in dataset.columns
    }


def estimate_app_usage_count(
    daily_phone_hours: float,
    social_media_hours: float,
    weekend_screen_time_hours: float,
    coefficients: dict[str, float],
    minimum: float,
    maximum: float,
) -> int:
    estimate = (
        coefficients["intercept"]
        + coefficients["Daily_Phone_Hours"] * daily_phone_hours
        + coefficients["Social_Media_Hours"] * social_media_hours
        + coefficients["Weekend_Screen_Time_Hours"] * weekend_screen_time_hours
    )
    clamped = min(maximum, max(minimum, estimate))
    return int(round(clamped))


def percentile_rank(series: pd.Series, value: float) -> float:
    if len(series) == 0:
        return 0.0
    return float((series <= value).mean() * 100.0)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
