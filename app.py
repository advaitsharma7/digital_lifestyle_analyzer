from __future__ import annotations

from src import bootstrap  # noqa: F401

import streamlit as st

from src.charts import (
    build_before_after_chart,
    build_correlation_chart,
    build_feature_impact_chart,
    build_gauge,
    build_global_importance_chart,
    build_lifestyle_breakdown_chart,
    build_percentile_chart,
    build_radar_chart,
)
from src.config import UI_FIELD_LABELS
from src.data import load_processed_dataset
from src.inference import analyze_profile, load_artifacts
from src.training import ensure_artifacts


st.set_page_config(
    page_title="Digital Lifestyle Analyzer",
    page_icon="DA",
    layout="wide",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top right, rgba(215,164,73,0.10), transparent 28%),
              radial-gradient(circle at top left, rgba(26,111,99,0.12), transparent 26%),
              linear-gradient(180deg, #fbf6ee 0%, #f4efe7 100%);
          }
          .main .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 4rem;
          }
          div[data-testid="stVerticalBlock"] > div:has(> div.stTabs) {
            min-width: 0;
          }
          h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            color: #20303a;
          }
          .hero-panel {
            background: linear-gradient(135deg, rgba(255,249,241,0.96), rgba(243,236,225,0.92));
            border: 1px solid rgba(32,48,58,0.08);
            border-radius: 28px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 22px 60px rgba(32,48,58,0.08);
            margin-bottom: 1.25rem;
          }
          .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            color: #8a6a2c;
            font-weight: 700;
            margin-bottom: 0.6rem;
          }
          .hero-panel h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.6rem;
            line-height: 1.05;
          }
          .hero-panel p {
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.6;
            color: #43525a;
          }
          .note-card, .score-card {
            background: rgba(255, 250, 241, 0.92);
            border: 1px solid rgba(32,48,58,0.08);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 32px rgba(32,48,58,0.06);
            margin-bottom: 0.9rem;
          }
          .score-card .big {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 3rem;
            color: #1a6f63;
            line-height: 1;
            margin: 0.2rem 0 0.4rem 0;
          }
          .score-card p, .note-card p {
            margin: 0;
            color: #43525a;
            line-height: 1.55;
          }
          .insight-list {
            display: grid;
            gap: 0.75rem;
          }
          .insight-chip {
            background: rgba(255,250,241,0.88);
            border-left: 4px solid #1a6f63;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: #2c3d45;
            box-shadow: 0 10px 24px rgba(32,48,58,0.05);
          }
          .footer-note {
            margin-top: 1.5rem;
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(32,48,58,0.04);
            color: #4b5860;
            font-size: 0.95rem;
          }
          div[data-testid="stMetric"] {
            background: rgba(255, 250, 241, 0.78);
            border: 1px solid rgba(32,48,58,0.08);
            border-radius: 18px;
            padding: 0.6rem 0.8rem;
          }
          @media (max-width: 900px) {
            .main .block-container {
              padding-top: 1rem;
              padding-left: 1rem;
              padding-right: 1rem;
            }
            .hero-panel {
              padding: 1.2rem 1rem;
              border-radius: 22px;
            }
            .hero-panel h1 {
              font-size: 2rem;
              line-height: 1.1;
            }
            .hero-panel p,
            .score-card p,
            .note-card p {
              font-size: 0.95rem;
            }
            .note-card, .score-card, .footer-note {
              border-radius: 18px;
              padding: 0.95rem;
            }
            .score-card .big {
              font-size: 2.4rem;
            }
            .insight-chip {
              border-radius: 14px;
              padding: 0.8rem 0.9rem;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state(defaults: dict[str, object], reference_df) -> None:
    if "baseline_inputs" not in st.session_state:
        st.session_state.baseline_inputs = {
            "Daily_Phone_Hours": float(defaults["Daily_Phone_Hours"]),
            "Social_Media_Hours": float(defaults["Social_Media_Hours"]),
            "Sleep_Hours": float(defaults["Sleep_Hours"]),
            "Caffeine_Intake_Cups": float(defaults["Caffeine_Intake_Cups"]),
            "Weekend_Screen_Time_Hours": float(defaults["Weekend_Screen_Time_Hours"]),
            "Age": None,
            "Gender": None,
            "Device_Type": None,
        }
        st.session_state.analysis = analyze_profile(
            st.session_state.baseline_inputs, reference_df
        )
        reset_simulation_state(st.session_state.baseline_inputs)

    baseline_inputs = st.session_state.baseline_inputs
    input_defaults = {
        "input_daily_phone_hours": float(baseline_inputs["Daily_Phone_Hours"]),
        "input_social_media_hours": float(baseline_inputs["Social_Media_Hours"]),
        "input_sleep_hours": float(baseline_inputs["Sleep_Hours"]),
        "input_caffeine_intake": float(baseline_inputs["Caffeine_Intake_Cups"]),
        "input_weekend_screen_time": float(baseline_inputs["Weekend_Screen_Time_Hours"]),
        "input_include_age": baseline_inputs.get("Age") is not None,
        "input_age": int(
            round(
                baseline_inputs["Age"]
                if baseline_inputs.get("Age") is not None
                else defaults["Age"]
            )
        ),
        "input_gender": baseline_inputs.get("Gender") or "Use dataset average",
        "input_device_type": baseline_inputs.get("Device_Type")
        or "Use dataset average",
    }
    for key, value in input_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_simulation_state(inputs: dict[str, object]) -> None:
    st.session_state.sim_phone = float(inputs["Daily_Phone_Hours"])
    st.session_state.sim_social = float(inputs["Social_Media_Hours"])
    st.session_state.sim_sleep = float(inputs["Sleep_Hours"])


def stress_gauge_color(stress_level: int) -> str:
    if stress_level >= 4:
        return "#c65d42"
    if stress_level == 3:
        return "#d7a449"
    return "#1a6f63"


def productivity_gauge_color(score: float) -> str:
    if score >= 6.0:
        return "#1a6f63"
    if score >= 4.0:
        return "#d7a449"
    return "#c65d42"


def format_delta(value: float, positive_good: bool = True, digits: int = 1) -> str:
    sign = "+" if value >= 0 else ""
    tone = "better" if (value >= 0 and positive_good) or (value < 0 and not positive_good) else "worse"
    return f"{sign}{value:.{digits}f} ({tone})"


def scenario_summary(
    baseline_result: dict[str, object], scenario_result: dict[str, object]
) -> str:
    stress_shift = int(scenario_result["stress_level"]) - int(baseline_result["stress_level"])
    productivity_shift = float(scenario_result["productivity_score"]) - float(
        baseline_result["productivity_score"]
    )
    score_shift = float(scenario_result["lifestyle_score"]["total"]) - float(
        baseline_result["lifestyle_score"]["total"]
    )

    if stress_shift < 0 and productivity_shift > 0:
        return (
            f"This scenario lowers stress by {abs(stress_shift)} level(s), lifts "
            f"productivity by {productivity_shift:.1f}, and adds {score_shift:.1f} "
            "points to your lifestyle score."
        )
    if stress_shift > 0 and productivity_shift < 0:
        return (
            "This change moves you in the wrong direction on both stress and focus, so it is "
            "probably not the tradeoff you want."
        )
    return (
        f"Compared with your current pattern, this scenario shifts productivity by "
        f"{productivity_shift:.1f} and lifestyle score by {score_shift:.1f}."
    )


def user_point_for_field(result: dict[str, object], field: str) -> float | None:
    if field in result["profile"]:
        return float(result["profile"][field])
    if field == "Stress_Level":
        return float(result["stress_level"])
    if field == "Work_Productivity_Score":
        return float(result["productivity_score"])
    return None


apply_styles()

with st.spinner("Preparing the lifestyle models and benchmark dataset..."):
    ensure_artifacts()
    artifacts = load_artifacts()
    reference_df = load_processed_dataset()

metadata = artifacts["metadata"]
initialize_state(metadata["defaults"], reference_df)

st.markdown(
    f"""
    <div class="hero-panel">
      <div class="eyebrow">Digital Lifestyle Analyzer</div>
      <h1>Read your digital rhythm, then simulate a better one.</h1>
      <p>
        This app benchmarks your habits against {metadata['dataset_summary']['rows']:,} synthetic
        lifestyle profiles, predicts stress and productivity, and shows which changes are most
        likely to improve your balance.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_left, input_right = st.columns(2)

with input_left:
    st.slider(
        "Daily Phone Hours",
        min_value=0.0,
        max_value=12.0,
        step=0.1,
        key="input_daily_phone_hours",
    )
    st.slider(
        "Social Media Hours",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key="input_social_media_hours",
    )
    st.slider(
        "Sleep Hours",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key="input_sleep_hours",
    )

with input_right:
    st.slider(
        "Caffeine Intake (cups)",
        min_value=0.0,
        max_value=8.0,
        step=0.5,
        key="input_caffeine_intake",
    )
    st.slider(
        "Weekend Screen Time",
        min_value=0.0,
        max_value=15.0,
        step=0.1,
        key="input_weekend_screen_time",
    )

with st.expander("Personalize results (optional)", expanded=False):
    st.checkbox("Include age", key="input_include_age")
    st.slider(
        "Age",
        min_value=13,
        max_value=80,
        key="input_age",
        disabled=not st.session_state.input_include_age,
    )
    st.caption("Age is only applied when `Include age` is checked.")

    gender_options = ["Use dataset average"] + metadata["options"]["Gender"]
    if st.session_state.input_gender not in gender_options:
        st.session_state.input_gender = gender_options[0]
    st.selectbox("Gender", gender_options, key="input_gender")

    device_options = ["Use dataset average"] + metadata["options"]["Device_Type"]
    if st.session_state.input_device_type not in device_options:
        st.session_state.input_device_type = device_options[0]
    st.selectbox("Device Type", device_options, key="input_device_type")

submitted = st.button("Analyze My Lifestyle", width="stretch")

if submitted:
    updated_inputs = {
        "Daily_Phone_Hours": st.session_state.input_daily_phone_hours,
        "Social_Media_Hours": st.session_state.input_social_media_hours,
        "Sleep_Hours": st.session_state.input_sleep_hours,
        "Caffeine_Intake_Cups": st.session_state.input_caffeine_intake,
        "Weekend_Screen_Time_Hours": st.session_state.input_weekend_screen_time,
        "Age": st.session_state.input_age if st.session_state.input_include_age else None,
        "Gender": None
        if st.session_state.input_gender == "Use dataset average"
        else st.session_state.input_gender,
        "Device_Type": None
        if st.session_state.input_device_type == "Use dataset average"
        else st.session_state.input_device_type,
    }
    st.session_state.baseline_inputs = updated_inputs
    st.session_state.analysis = analyze_profile(updated_inputs, reference_df)
    reset_simulation_state(updated_inputs)
    st.rerun()

analysis = st.session_state.analysis

st.markdown(
    f"""
    <div class="note-card">
      <div class="eyebrow">Current Read</div>
      <p>{analysis['headline']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_mid, top_right = st.columns([1.05, 1.05, 0.85])

with top_left:
    st.plotly_chart(
        build_gauge(
            title="Predicted Stress Level",
            value=float(analysis["stress_level"]),
            minimum=1,
            maximum=5,
            accent=stress_gauge_color(int(analysis["stress_level"])),
        ),
        width="stretch",
    )

with top_mid:
    st.plotly_chart(
        build_gauge(
            title="Predicted Productivity Score",
            value=float(analysis["productivity_score"]),
            minimum=1,
            maximum=10,
            accent=productivity_gauge_color(float(analysis["productivity_score"])),
        ),
        width="stretch",
    )

with top_right:
    st.markdown(
        f"""
        <div class="score-card">
          <div class="eyebrow">Lifestyle Score</div>
          <div class="big">{analysis['lifestyle_score']['total']:.0f}</div>
          <p>Your weighted balance score blends sleep, screen time, stress, and productivity into one read.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="note-card">
          <div class="eyebrow">Behavioral Archetype</div>
          <p><strong>{analysis['cluster']['label']}</strong><br>{analysis['cluster']['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("## Why This Is Happening")
why_left, why_right = st.columns(2)

with why_left:
    st.markdown(
        f"""
        <div class="note-card">
          <div class="eyebrow">Stress Drivers</div>
          <p>{analysis['explanations']['stress']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stress_tabs = st.tabs(["Your Current Pattern", "RandomForest Model Importance"])
    with stress_tabs[0]:
        st.plotly_chart(
            build_feature_impact_chart(analysis["local_impacts"]["stress"], "stress"),
            width="stretch",
        )
        st.caption(
            "This chart is personalized. It estimates how your current values are nudging the stress prediction away from a typical baseline."
        )
    with stress_tabs[1]:
        st.plotly_chart(
            build_global_importance_chart(
                analysis["global_feature_importance"]["stress"], "stress"
            ),
            width="stretch",
        )
        st.caption(
            "This is the true RandomForest feature-importance view aggregated across the training dataset."
        )

with why_right:
    st.markdown(
        f"""
        <div class="note-card">
          <div class="eyebrow">Productivity Drivers</div>
          <p>{analysis['explanations']['productivity']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    productivity_tabs = st.tabs(
        ["Your Current Pattern", "RandomForest Model Importance"]
    )
    with productivity_tabs[0]:
        st.plotly_chart(
            build_feature_impact_chart(
                analysis["local_impacts"]["productivity"], "productivity"
            ),
            width="stretch",
        )
        st.caption(
            "This chart is personalized. It shows which of your current habits are supporting or dragging on the productivity forecast."
        )
    with productivity_tabs[1]:
        st.plotly_chart(
            build_global_importance_chart(
                analysis["global_feature_importance"]["productivity"],
                "productivity",
            ),
            width="stretch",
        )
        st.caption(
            "This is the true RandomForest feature-importance view aggregated across the training dataset."
        )

st.markdown("## How You Compare")
compare_left, compare_right = st.columns([0.95, 1.05])

with compare_left:
    st.plotly_chart(
        build_percentile_chart(analysis["percentiles"]),
        width="stretch",
    )

with compare_right:
    st.plotly_chart(
        build_radar_chart(analysis["radar"]),
        width="stretch",
    )

st.markdown("## Scenario Lab")
lab_left, lab_right = st.columns([0.85, 1.15])

with lab_left:
    st.markdown(
        """
        <div class="note-card">
          <div class="eyebrow">What-If Controls</div>
          <p>Change sleep, social media, and phone time to see how your projected outcome shifts before you commit to the habit change.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.slider(
        "Scenario Sleep Hours",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key="sim_sleep",
    )
    st.slider(
        "Scenario Social Media Hours",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key="sim_social",
    )
    st.slider(
        "Scenario Daily Phone Hours",
        min_value=0.0,
        max_value=12.0,
        step=0.1,
        key="sim_phone",
    )
    if st.button("Reset Scenario", width="stretch"):
        reset_simulation_state(st.session_state.baseline_inputs)
        st.rerun()

scenario_inputs = dict(st.session_state.baseline_inputs)
scenario_inputs.update(
    {
        "Sleep_Hours": st.session_state.sim_sleep,
        "Social_Media_Hours": st.session_state.sim_social,
        "Daily_Phone_Hours": st.session_state.sim_phone,
    }
)
scenario_result = analyze_profile(scenario_inputs, reference_df)

with lab_right:
    st.markdown(
        f"""
        <div class="note-card">
          <div class="eyebrow">Projected Outcome</div>
          <p>{scenario_summary(analysis, scenario_result)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric(
        "Stress",
        f"{scenario_result['stress_level']}/5",
        delta=format_delta(
            float(scenario_result["stress_level"]) - float(analysis["stress_level"]),
            positive_good=False,
            digits=0,
        ),
    )
    metric_b.metric(
        "Productivity",
        f"{scenario_result['productivity_score']:.1f}/10",
        delta=format_delta(
            float(scenario_result["productivity_score"])
            - float(analysis["productivity_score"]),
            positive_good=True,
        ),
    )
    metric_c.metric(
        "Lifestyle Score",
        f"{scenario_result['lifestyle_score']['total']:.0f}/100",
        delta=format_delta(
            float(scenario_result["lifestyle_score"]["total"])
            - float(analysis["lifestyle_score"]["total"]),
            positive_good=True,
        ),
    )
    st.plotly_chart(
        build_before_after_chart(analysis, scenario_result),
        width="stretch",
    )
    if scenario_result["cluster"]["label"] != analysis["cluster"]["label"]:
        st.info(
            f"This scenario would shift your archetype from {analysis['cluster']['label']} "
            f"to {scenario_result['cluster']['label']}."
        )

st.markdown("## Correlation Explorer")
corr_controls = st.columns([0.5, 0.5, 1.0])
with corr_controls[0]:
    x_field = st.selectbox(
        "X Axis",
        metadata["dataset_summary"]["correlation_fields"],
        index=0,
        format_func=lambda field: UI_FIELD_LABELS.get(field, field),
    )
with corr_controls[1]:
    y_field = st.selectbox(
        "Y Axis",
        metadata["dataset_summary"]["correlation_fields"],
        index=6,
        format_func=lambda field: UI_FIELD_LABELS.get(field, field),
    )

user_point = (
    user_point_for_field(analysis, x_field),
    user_point_for_field(analysis, y_field),
)
if None in user_point:
    user_point = None

st.plotly_chart(
    build_correlation_chart(reference_df, x_field, y_field, user_point),
    width="stretch",
)

st.markdown("## What Should You Do Next?")
insight_left, insight_right = st.columns([0.95, 1.05])

with insight_left:
    st.markdown(
        '<div class="insight-list">'
        + "".join(
            f'<div class="insight-chip">{insight}</div>' for insight in analysis["insights"]
        )
        + "</div>",
        unsafe_allow_html=True,
    )

with insight_right:
    st.plotly_chart(
        build_lifestyle_breakdown_chart(analysis["lifestyle_score"]),
        width="stretch",
    )

st.markdown(
    f"""
    <div class="footer-note">
      Stress model accuracy: <strong>{metadata['metrics']['stress_accuracy']:.2%}</strong>.
      Productivity RMSE: <strong>{metadata['metrics']['productivity_rmse']:.2f}</strong>.
      These outputs are benchmarked against a synthetic lifestyle dataset and are intended for reflection, not diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
