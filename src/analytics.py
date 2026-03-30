from __future__ import annotations

from src.config import (
    CLUSTER_FEATURES,
    LIFESTYLE_SCORE_WEIGHTS,
    PERCENTILE_FIELDS,
    RADAR_AXES,
    STRESS_SCALE_LABELS,
    UI_FIELD_LABELS,
)
from src.data import percentile_rank

import numpy as np
import pandas as pd


def describe_stress(stress_level: int) -> str:
    return STRESS_SCALE_LABELS.get(int(stress_level), "Moderate")


def describe_productivity(score: float) -> str:
    if score >= 7.0:
        return "highly focused"
    if score >= 5.5:
        return "steady"
    if score >= 4.0:
        return "mixed"
    return "fragile"


def build_headline(stress_level: int, productivity_score: float) -> str:
    stress_phrase = describe_stress(stress_level).lower()
    productivity_phrase = describe_productivity(productivity_score)
    return (
        f"You currently look {stress_phrase} on stress and {productivity_phrase} on "
        f"productivity, which makes this a good moment to test which habits move the "
        f"needle most."
    )


def build_percentile_summary(
    profile: dict[str, float | str], reference_df: pd.DataFrame
) -> dict[str, dict[str, float | str]]:
    summary: dict[str, dict[str, float | str]] = {}
    for field in PERCENTILE_FIELDS:
        percentile = percentile_rank(reference_df[field], float(profile[field]))
        summary[field] = {
            "label": UI_FIELD_LABELS[field],
            "value": float(profile[field]),
            "percentile": percentile,
            "comparison": (
                f"Higher than {percentile:.0f}% of benchmark users"
                if field != "Sleep_Hours"
                else f"Longer sleep than {percentile:.0f}% of benchmark users"
            ),
        }
    return summary


def _screen_balance_value(phone_hours: float, weekend_screen: float) -> float:
    return phone_hours + (weekend_screen * 0.35)


def compute_radar_payload(
    profile: dict[str, float | str],
    stress_level: int,
    productivity_score: float,
    reference_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    reference_values = pd.DataFrame(
        {
            "Sleep": reference_df["Sleep_Hours"],
            "Screen Time": _screen_balance_value(
                reference_df["Daily_Phone_Hours"],
                reference_df["Weekend_Screen_Time_Hours"],
            ),
            "Social Media": reference_df["Social_Media_Hours"],
            "Stress": reference_df["Stress_Level"],
            "Productivity": reference_df["Work_Productivity_Score"],
        }
    )

    user_values = {
        "Sleep": float(profile["Sleep_Hours"]),
        "Screen Time": _screen_balance_value(
            float(profile["Daily_Phone_Hours"]),
            float(profile["Weekend_Screen_Time_Hours"]),
        ),
        "Social Media": float(profile["Social_Media_Hours"]),
        "Stress": float(stress_level),
        "Productivity": float(productivity_score),
    }

    user_normalized: dict[str, float] = {}
    benchmark_normalized: dict[str, float] = {}
    inverse_axes = {"Screen Time", "Social Media", "Stress"}

    for axis in RADAR_AXES:
        series = reference_values[axis]
        min_value = float(series.min())
        max_value = float(series.max())
        span = max(max_value - min_value, 1e-6)

        user_score = (user_values[axis] - min_value) / span
        benchmark_score = (float(series.mean()) - min_value) / span
        if axis in inverse_axes:
            user_score = 1.0 - user_score
            benchmark_score = 1.0 - benchmark_score

        user_normalized[axis] = float(np.clip(user_score, 0.0, 1.0))
        benchmark_normalized[axis] = float(np.clip(benchmark_score, 0.0, 1.0))

    return {
        "user": user_normalized,
        "benchmark": benchmark_normalized,
        "raw_user": user_values,
    }


def compute_lifestyle_score(
    profile: dict[str, float | str], stress_level: int, productivity_score: float
) -> dict[str, object]:
    sleep_hours = float(profile["Sleep_Hours"])
    phone_hours = float(profile["Daily_Phone_Hours"])
    weekend_screen = float(profile["Weekend_Screen_Time_Hours"])

    sleep_score = max(0.0, 100.0 - abs(sleep_hours - 8.0) * 28.0)
    screen_score = max(
        0.0,
        100.0
        - (
            ((phone_hours / 12.0) * 0.65)
            + ((weekend_screen / 15.0) * 0.35)
        )
        * 100.0,
    )
    stress_score = max(0.0, min(100.0, ((5.0 - stress_level) / 4.0) * 100.0))
    productivity_component = max(
        0.0, min(100.0, ((productivity_score - 1.0) / 9.0) * 100.0)
    )

    breakdown = {
        "Sleep": round(sleep_score, 1),
        "Screen Time": round(screen_score, 1),
        "Stress": round(stress_score, 1),
        "Productivity": round(productivity_component, 1),
    }

    total = 0.0
    for component, weight in LIFESTYLE_SCORE_WEIGHTS.items():
        total += breakdown[component] * weight

    return {
        "total": round(total, 1),
        "breakdown": breakdown,
        "weights": LIFESTYLE_SCORE_WEIGHTS,
    }


def assign_cluster(
    profile: dict[str, float | str],
    stress_level: int,
    productivity_score: float,
    cluster_bundle: dict[str, object],
) -> dict[str, object]:
    cluster_vector = pd.DataFrame(
        [
            {
                "Daily_Phone_Hours": float(profile["Daily_Phone_Hours"]),
                "Social_Media_Hours": float(profile["Social_Media_Hours"]),
                "Sleep_Hours": float(profile["Sleep_Hours"]),
                "Caffeine_Intake_Cups": float(profile["Caffeine_Intake_Cups"]),
                "Weekend_Screen_Time_Hours": float(
                    profile["Weekend_Screen_Time_Hours"]
                ),
                "Stress_Level": float(stress_level),
                "Work_Productivity_Score": float(productivity_score),
            }
        ]
    )[CLUSTER_FEATURES]

    scaled = cluster_bundle["scaler"].transform(cluster_vector)
    cluster_id = int(cluster_bundle["model"].predict(scaled)[0])
    return {
        "id": cluster_id,
        "label": cluster_bundle["label_map"][cluster_id],
        "description": cluster_bundle["description_map"][cluster_id],
    }
