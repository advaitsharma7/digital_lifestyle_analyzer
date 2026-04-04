from __future__ import annotations

from src.config import EXPLAINABILITY_FEATURES, UI_FIELD_LABELS

import pandas as pd


FEATURE_PHRASES = {
    "Daily_Phone_Hours": "high phone use",
    "Social_Media_Hours": "heavy social media time",
    "Sleep_Hours": "your current sleep pattern",
    "Caffeine_Intake_Cups": "your caffeine intake",
    "Weekend_Screen_Time_Hours": "high weekend screen time",
}


def compute_local_impacts(
    profile_frame: pd.DataFrame,
    defaults: dict[str, float | str],
    predict_stress_score,
    predict_productivity_score,
    rebuild_profile_for_feature,
) -> dict[str, list[dict[str, float | str]]]:
    baseline_stress = predict_stress_score(profile_frame)
    baseline_productivity = predict_productivity_score(profile_frame)

    impacts: list[dict[str, float | str]] = []
    for feature in EXPLAINABILITY_FEATURES:
        comparison_frame = rebuild_profile_for_feature(
            profile_frame.iloc[0].to_dict(), feature, defaults[feature]
        )
        comparison_stress = predict_stress_score(comparison_frame)
        comparison_productivity = predict_productivity_score(comparison_frame)
        impacts.append(
            {
                "feature": feature,
                "label": UI_FIELD_LABELS[feature],
                "stress_delta": round(baseline_stress - comparison_stress, 3),
                "productivity_delta": round(
                    baseline_productivity - comparison_productivity, 3
                ),
            }
        )

    stress_impacts = sorted(
        impacts, key=lambda item: abs(float(item["stress_delta"])), reverse=True
    )
    productivity_impacts = sorted(
        impacts, key=lambda item: abs(float(item["productivity_delta"])), reverse=True
    )
    return {
        "stress": stress_impacts,
        "productivity": productivity_impacts,
    }


def build_explanations(
    impacts: dict[str, list[dict[str, float | str]]]
) -> dict[str, str]:
    stress_drivers = [
        item for item in impacts["stress"] if float(item["stress_delta"]) > 0.05
    ][:3]
    stress_guards = [
        item for item in impacts["stress"] if float(item["stress_delta"]) < -0.05
    ][:2]

    if stress_drivers:
        driver_text = ", ".join(FEATURE_PHRASES[item["feature"]] for item in stress_drivers)
        stress_text = f"Your stress looks elevated mainly because of {driver_text}."
    else:
        guard_text = ", ".join(
            FEATURE_PHRASES[item["feature"]] for item in stress_guards
        ) or "your current sleep and screen habits"
        stress_text = (
            f"Your current pattern does not show a major stress spike, and {guard_text} "
            "are helping keep it steadier."
        )

    productivity_drags = [
        item
        for item in impacts["productivity"]
        if float(item["productivity_delta"]) < -0.05
    ][:3]
    productivity_boosters = [
        item
        for item in impacts["productivity"]
        if float(item["productivity_delta"]) > 0.05
    ][:2]

    if productivity_drags:
        drag_text = ", ".join(
            FEATURE_PHRASES[item["feature"]] for item in productivity_drags
        )
        productivity_text = (
            f"Your productivity is being pulled down mostly by {drag_text}."
        )
    else:
        boost_text = ", ".join(
            FEATURE_PHRASES[item["feature"]] for item in productivity_boosters
        ) or "your current routine"
        productivity_text = (
            f"{boost_text.capitalize()} is supporting a steadier productivity outlook."
        )

    return {
        "stress": stress_text,
        "productivity": productivity_text,
    }


def generate_insights(
    profile: dict[str, float | str],
    stress_level: float,
    productivity_score: float,
    cluster: dict[str, object],
    percentiles: dict[str, dict[str, float | str]],
) -> list[str]:
    insights: list[str] = []

    sleep_hours = float(profile["Sleep_Hours"])
    social_hours = float(profile["Social_Media_Hours"])
    phone_hours = float(profile["Daily_Phone_Hours"])
    caffeine = float(profile["Caffeine_Intake_Cups"])
    weekend_screen = float(profile["Weekend_Screen_Time_Hours"])

    if sleep_hours < 4.0:
        insights.append(
            "Your sleep level is critically low and should be treated as the first priority before any productivity optimization."
        )
    if sleep_hours < 6.0:
        insights.append(
            "Sleep is your biggest pressure point right now. Pushing it closer to seven hours is likely to reduce stress quickly."
        )
    elif sleep_hours >= 7.3:
        insights.append(
            "Your sleep pattern is giving you a solid base. Keeping it consistent should protect both mood and focus."
        )

    if social_hours > 4.0:
        insights.append(
            "Social media is running high enough to compete with focus time. Trimming even one hour could noticeably improve your balance."
        )
    elif social_hours < 2.0:
        insights.append(
            "Your social media time is relatively contained, which is helping keep digital overload in check."
        )

    if phone_hours > 8.0 or weekend_screen > 10.0:
        insights.append(
            "Your overall screen load is heavy. A lighter evening phone routine could create the fastest visible improvement."
        )

    if caffeine >= 4.0 and sleep_hours < 6.5:
        insights.append(
            "High caffeine on top of shorter sleep often produces a wired-but-tired pattern that keeps stress elevated."
        )

    if stress_level >= 4:
        insights.append(
            "The model sees a high-stress pattern. Focus on recovery habits first instead of trying to force productivity harder."
        )
    elif productivity_score >= 5.5:
        insights.append(
            "You already have a reasonably stable productivity base. Small improvements to sleep or screen time could compound well."
        )

    sleep_percentile = float(percentiles["Sleep_Hours"]["percentile"])
    if sleep_percentile < 35:
        insights.append(
            "Compared with the benchmark dataset, your sleep sits on the shorter side, which reinforces the stress story."
        )

    insights.append(
        f"Your closest behavioral archetype is {cluster['label'].lower()}, which suggests your current habits are clustering into a recognizable lifestyle pattern."
    )

    deduped: list[str] = []
    for insight in insights:
        if insight not in deduped:
            deduped.append(insight)

    fallback_insights = [
        "A one-hour improvement in sleep or a reduction in social media is usually the cleanest first experiment in this profile.",
        "Treat the simulator as your planning tool: test one change at a time so you can see which habit gives the strongest lift.",
        "The most sustainable next step is usually the smallest repeatable one, not the most extreme screen-time cut.",
    ]
    for fallback in fallback_insights:
        if len(deduped) >= 3:
            break
        if fallback not in deduped:
            deduped.append(fallback)

    return deduped[:5]
