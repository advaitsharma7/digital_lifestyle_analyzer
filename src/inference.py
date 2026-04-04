from __future__ import annotations

import json
from functools import lru_cache

from src.analytics import (
    assign_cluster,
    build_headline,
    build_percentile_summary,
    compute_lifestyle_score,
    compute_radar_payload,
    describe_stress,
)
from src.config import (
    APP_USAGE_INPUT_FEATURES,
    CLUSTER_BUNDLE_PATH,
    METADATA_PATH,
    MODEL_FEATURES,
    PREPROCESSOR_PATH,
    PRODUCTIVITY_MODEL_FEATURES,
    PRODUCTIVITY_MODEL_PATH,
    PRODUCTIVITY_PREPROCESSOR_PATH,
    STRESS_MODEL_PATH,
)
from src.data import estimate_app_usage_count, load_processed_dataset
from src.insights import build_explanations, compute_local_impacts, generate_insights
from src.training import ensure_artifacts

import joblib
import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, object]:
    ensure_artifacts()
    return {
        "stress_model": joblib.load(STRESS_MODEL_PATH),
        "productivity_model": joblib.load(PRODUCTIVITY_MODEL_PATH),
        "stress_preprocessor": joblib.load(PREPROCESSOR_PATH),
        "productivity_preprocessor": joblib.load(PRODUCTIVITY_PREPROCESSOR_PATH),
        "cluster_bundle": joblib.load(CLUSTER_BUNDLE_PATH),
        "metadata": json.loads(METADATA_PATH.read_text(encoding="utf-8")),
    }


@lru_cache(maxsize=1)
def load_productivity_calibration() -> dict[str, object]:
    artifacts = load_artifacts()
    dataset = load_processed_dataset()
    transformed = artifacts["productivity_preprocessor"].transform(
        dataset[PRODUCTIVITY_MODEL_FEATURES]
    )
    raw_predictions = np.clip(
        artifacts["productivity_model"].predict(transformed).astype(float),
        1.0,
        10.0,
    )
    sorted_predictions = np.sort(raw_predictions)
    calibrated_predictions = np.round(
        _calibrate_productivity_values(raw_predictions, sorted_predictions),
        2,
    )
    return {
        "sorted_predictions": sorted_predictions,
        "raw_predictions": raw_predictions,
        "calibrated_predictions": calibrated_predictions,
    }


@lru_cache(maxsize=1)
def _load_display_reference_dataset_cached() -> pd.DataFrame:
    dataset = load_processed_dataset().copy()
    calibration = load_productivity_calibration()
    dataset["Work_Productivity_Score"] = calibration["calibrated_predictions"]
    return dataset


def load_display_reference_dataset() -> pd.DataFrame:
    return _load_display_reference_dataset_cached().copy()


def normalize_optional_value(value):
    if value in (None, "", "Use dataset average"):
        return None
    return value


def build_profile(
    user_inputs: dict[str, object], metadata: dict[str, object]
) -> dict[str, float | str]:
    defaults = metadata["defaults"]

    profile: dict[str, float | str] = {}
    for feature in MODEL_FEATURES:
        if feature == "App_Usage_Count":
            continue
        raw_value = normalize_optional_value(user_inputs.get(feature))
        profile[feature] = defaults[feature] if raw_value is None else raw_value

    app_usage = estimate_app_usage_count(
        daily_phone_hours=float(profile["Daily_Phone_Hours"]),
        social_media_hours=float(profile["Social_Media_Hours"]),
        weekend_screen_time_hours=float(profile["Weekend_Screen_Time_Hours"]),
        coefficients=metadata["app_usage_estimator"],
        minimum=float(metadata["app_usage_bounds"]["min"]),
        maximum=float(metadata["app_usage_bounds"]["max"]),
    )
    profile["App_Usage_Count"] = app_usage

    for numeric_feature in [
        "Age",
        "Daily_Phone_Hours",
        "Social_Media_Hours",
        "Sleep_Hours",
        "Caffeine_Intake_Cups",
        "Weekend_Screen_Time_Hours",
        "App_Usage_Count",
    ]:
        profile[numeric_feature] = float(profile[numeric_feature])

    return profile


def profile_to_frame(profile: dict[str, float | str]) -> pd.DataFrame:
    return pd.DataFrame([profile])[MODEL_FEATURES]


def rebuild_profile_frame_for_feature(
    profile: dict[str, float | str], feature: str, value: float | str
) -> pd.DataFrame:
    updated_profile = dict(profile)
    updated_profile[feature] = value
    if feature in APP_USAGE_INPUT_FEATURES:
        artifacts = load_artifacts()
        metadata = artifacts["metadata"]
        updated_profile["App_Usage_Count"] = estimate_app_usage_count(
            daily_phone_hours=float(updated_profile["Daily_Phone_Hours"]),
            social_media_hours=float(updated_profile["Social_Media_Hours"]),
            weekend_screen_time_hours=float(updated_profile["Weekend_Screen_Time_Hours"]),
            coefficients=metadata["app_usage_estimator"],
            minimum=float(metadata["app_usage_bounds"]["min"]),
            maximum=float(metadata["app_usage_bounds"]["max"]),
        )
    return profile_to_frame(updated_profile)


def _predict_expected_stress(frame: pd.DataFrame) -> float:
    artifacts = load_artifacts()
    transformed = artifacts["stress_preprocessor"].transform(frame[MODEL_FEATURES])
    probabilities = artifacts["stress_model"].predict_proba(transformed)[0]
    classes = np.asarray(artifacts["stress_model"].classes_, dtype=float)
    return float(np.dot(probabilities, classes))


def _predict_stress_score(frame: pd.DataFrame) -> float:
    stress_score = _predict_expected_stress(frame)
    row = frame.iloc[0]
    daily_phone_hours = float(row["Daily_Phone_Hours"])
    sleep_hours = float(row["Sleep_Hours"])
    if sleep_hours < 4.0 and daily_phone_hours > 8.0:
        stress_score = max(stress_score, 4.5)
    return float(np.clip(stress_score, 1.0, 5.0))


def _productivity_percentiles(
    raw_values: float | np.ndarray, sorted_predictions: np.ndarray
) -> np.ndarray:
    positions = np.searchsorted(sorted_predictions, raw_values, side="right")
    denominator = max(len(sorted_predictions), 1)
    return np.clip(positions / denominator, 0.0, 1.0)


def _calibrate_productivity_values(
    raw_values: float | np.ndarray, sorted_predictions: np.ndarray
) -> np.ndarray:
    percentiles = _productivity_percentiles(raw_values, sorted_predictions)
    return np.clip(1.0 + (9.0 * percentiles), 1.0, 10.0)


def _predict_productivity_raw_score(frame: pd.DataFrame) -> float:
    artifacts = load_artifacts()
    transformed = artifacts["productivity_preprocessor"].transform(
        frame[PRODUCTIVITY_MODEL_FEATURES]
    )
    prediction = float(artifacts["productivity_model"].predict(transformed)[0])
    return float(np.clip(prediction, 1.0, 10.0))


def _predict_productivity_score(frame: pd.DataFrame) -> float:
    calibration = load_productivity_calibration()
    raw_prediction = _predict_productivity_raw_score(frame)
    calibrated_prediction = _calibrate_productivity_values(
        raw_prediction,
        calibration["sorted_predictions"],
    )
    return float(np.round(calibrated_prediction, 2))


def analyze_profile(
    user_inputs: dict[str, object], reference_df: pd.DataFrame | None = None
) -> dict[str, object]:
    artifacts = load_artifacts()
    metadata = artifacts["metadata"]
    dataset = reference_df if reference_df is not None else load_display_reference_dataset()

    profile = build_profile(user_inputs, metadata)
    daily_phone_hours = float(profile["Daily_Phone_Hours"])
    sleep_hours = float(profile["Sleep_Hours"])
    social_media_hours = float(profile["Social_Media_Hours"])
    print("USER INPUT VALUES:")
    print("Phone:", daily_phone_hours)
    print("Sleep:", sleep_hours)
    print("Social Media:", social_media_hours)
    print("MODEL INPUT VALUES:", daily_phone_hours, sleep_hours, social_media_hours)
    print("CLUSTER INPUT VALUES:", daily_phone_hours, sleep_hours, social_media_hours)
    print("UI RENDER VALUES:", daily_phone_hours, sleep_hours, social_media_hours)
    profile_frame = profile_to_frame(profile)
    stress_transformed = artifacts["stress_preprocessor"].transform(
        profile_frame[MODEL_FEATURES]
    )

    stress_class = int(artifacts["stress_model"].predict(stress_transformed)[0])
    raw_stress_expected = _predict_expected_stress(profile_frame)
    stress_score = round(_predict_stress_score(profile_frame), 2)
    productivity_raw_score = round(
        _predict_productivity_raw_score(profile_frame),
        2,
    )
    calibration = load_productivity_calibration()
    productivity_prediction = round(
        float(
            _calibrate_productivity_values(
                productivity_raw_score,
                calibration["sorted_predictions"],
            )
        ),
        2,
    )
    productivity_percentile = round(
        float(
            _productivity_percentiles(
                productivity_raw_score,
                calibration["sorted_predictions"],
            )
            * 100.0
        ),
        1,
    )

    percentiles = build_percentile_summary(profile, dataset)
    cluster = assign_cluster(
        profile,
        stress_score,
        productivity_raw_score,
        artifacts["cluster_bundle"],
    )
    radar_payload = compute_radar_payload(
        profile,
        stress_score,
        productivity_prediction,
        dataset,
    )
    lifestyle_score = compute_lifestyle_score(
        profile,
        stress_score,
        productivity_prediction,
    )

    local_impacts = compute_local_impacts(
        profile_frame=profile_frame,
        defaults=metadata["defaults"],
        predict_stress_score=_predict_stress_score,
        predict_productivity_score=_predict_productivity_score,
        rebuild_profile_for_feature=rebuild_profile_frame_for_feature,
    )
    explanations = build_explanations(local_impacts)
    insights = generate_insights(
        profile=profile,
        stress_level=stress_score,
        productivity_score=productivity_prediction,
        cluster=cluster,
        percentiles=percentiles,
    )
    critical_sleep_warning = sleep_hours < 4.0
    warnings: list[str] = []
    if critical_sleep_warning:
        warnings.append(
            "Your sleep level is critically low and may severely impact your stress and productivity."
        )

    return {
        "profile": profile,
        "stress_level": stress_score,
        "stress_class": stress_class,
        "stress_label": describe_stress(stress_score),
        "stress_expected": stress_score,
        "stress_model_score": round(raw_stress_expected, 2),
        "productivity_score": productivity_prediction,
        "productivity_raw_score": productivity_raw_score,
        "productivity_percentile": productivity_percentile,
        "headline": build_headline(stress_score, productivity_prediction),
        "percentiles": percentiles,
        "cluster": cluster,
        "radar": radar_payload,
        "lifestyle_score": lifestyle_score,
        "global_feature_importance": metadata["feature_importance"],
        "local_impacts": local_impacts,
        "explanations": explanations,
        "insights": insights,
        "critical_sleep_warning": critical_sleep_warning,
        "warnings": warnings,
    }
