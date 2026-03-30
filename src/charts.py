from __future__ import annotations

from src.config import SCATTER_SAMPLE_SIZE, UI_FIELD_LABELS

import numpy as np
import pandas as pd
import plotly.graph_objects as go


COLORS = {
    "teal": "#1a6f63",
    "ink": "#20303a",
    "sand": "#f4efe7",
    "gold": "#d7a449",
    "coral": "#d16b4c",
    "soft": "#7aa49b",
    "positive": "#227c62",
    "negative": "#c65d42",
    "neutral": "#8b9197",
}


def build_gauge(
    title: str,
    value: float,
    minimum: float,
    maximum: float,
    accent: str,
) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 34, "color": COLORS["ink"]}},
            title={"text": title, "font": {"size": 18, "color": COLORS["ink"]}},
            gauge={
                "axis": {"range": [minimum, maximum], "tickwidth": 1},
                "bar": {"color": accent, "thickness": 0.38},
                "bgcolor": "#fffaf1",
                "borderwidth": 0,
                "steps": [
                    {"range": [minimum, minimum + (maximum - minimum) * 0.33], "color": "#d7efe5"},
                    {"range": [minimum + (maximum - minimum) * 0.33, minimum + (maximum - minimum) * 0.66], "color": "#f7e7bf"},
                    {"range": [minimum + (maximum - minimum) * 0.66, maximum], "color": "#f0d2c7"},
                ],
            },
        )
    )
    figure.update_layout(
        margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
    )
    return figure


def build_percentile_chart(
    percentile_summary: dict[str, dict[str, float | str]]
) -> go.Figure:
    labels = [entry["label"] for entry in percentile_summary.values()]
    values = [float(entry["percentile"]) for entry in percentile_summary.values()]
    hover = [str(entry["comparison"]) for entry in percentile_summary.values()]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=[COLORS["teal"], COLORS["gold"], COLORS["soft"]]),
            text=[f"{value:.0f}th pct" for value in values],
            textposition="outside",
            hovertext=hover,
            hovertemplate="%{y}<br>%{hovertext}<extra></extra>",
        )
    )
    figure.update_layout(
        margin=dict(l=24, r=24, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], title="Percentile"),
        yaxis=dict(autorange="reversed"),
        height=280,
    )
    return figure


def build_radar_chart(radar_payload: dict[str, dict[str, float]]) -> go.Figure:
    axes = list(radar_payload["user"].keys())
    user_values = [radar_payload["user"][axis] for axis in axes]
    benchmark_values = [radar_payload["benchmark"][axis] for axis in axes]

    axes_loop = axes + [axes[0]]
    user_loop = user_values + [user_values[0]]
    benchmark_loop = benchmark_values + [benchmark_values[0]]

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=benchmark_loop,
            theta=axes_loop,
            fill="toself",
            name="Dataset Average",
            line=dict(color=COLORS["neutral"], width=2),
            fillcolor="rgba(139, 145, 151, 0.18)",
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=user_loop,
            theta=axes_loop,
            fill="toself",
            name="Your Profile",
            line=dict(color=COLORS["teal"], width=3),
            fillcolor="rgba(26, 111, 99, 0.24)",
        )
    )
    figure.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], showticklabels=False, ticks=""),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        height=360,
    )
    return figure


def build_before_after_chart(
    baseline_result: dict[str, object], scenario_result: dict[str, object]
) -> go.Figure:
    baseline_scores = {
        "Stress Balance": ((5.0 - float(baseline_result["stress_level"])) / 4.0) * 100.0,
        "Productivity": ((float(baseline_result["productivity_score"]) - 1.0) / 9.0) * 100.0,
        "Lifestyle Score": float(baseline_result["lifestyle_score"]["total"]),
    }
    scenario_scores = {
        "Stress Balance": ((5.0 - float(scenario_result["stress_level"])) / 4.0) * 100.0,
        "Productivity": ((float(scenario_result["productivity_score"]) - 1.0) / 9.0) * 100.0,
        "Lifestyle Score": float(scenario_result["lifestyle_score"]["total"]),
    }

    categories = list(baseline_scores.keys())
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=categories,
            y=[baseline_scores[category] for category in categories],
            name="Current",
            marker_color="#d2b484",
        )
    )
    figure.add_trace(
        go.Bar(
            x=categories,
            y=[scenario_scores[category] for category in categories],
            name="Scenario",
            marker_color=COLORS["teal"],
        )
    )
    figure.update_layout(
        barmode="group",
        margin=dict(l=24, r=24, t=24, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100], title="Better-state score"),
        height=330,
    )
    return figure


def build_feature_impact_chart(
    impacts: list[dict[str, float | str]],
    metric: str,
) -> go.Figure:
    top_items = impacts[:5]
    if metric == "stress":
        values = [float(item["stress_delta"]) for item in top_items]
        positive_meaning = "pushes stress up"
        title = "What is driving stress"
    else:
        values = [float(item["productivity_delta"]) for item in top_items]
        positive_meaning = "supports productivity"
        title = "What is shaping productivity"

    colors = [
        COLORS["negative"] if value > 0 else COLORS["positive"] for value in values
    ]
    if metric == "productivity":
        colors = [
            COLORS["positive"] if value > 0 else COLORS["negative"] for value in values
        ]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=[str(item["label"]) for item in top_items],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Effect: %{x:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=24, r=24, t=56, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=positive_meaning,
        yaxis=dict(autorange="reversed"),
        height=300,
    )
    return figure


def build_global_importance_chart(
    importances: dict[str, float],
    metric: str,
) -> go.Figure:
    ordered_items = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:7]
    labels = [UI_FIELD_LABELS.get(name, name) for name, _ in ordered_items]
    values = [float(value) * 100.0 for _, value in ordered_items]

    if metric == "stress":
        title = "RandomForest Global Importance: Stress"
        accent = COLORS["coral"]
    else:
        title = "RandomForest Global Importance: Productivity"
        accent = COLORS["teal"]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=accent,
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>Importance: %{x:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=24, r=24, t=56, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Share of model importance",
        yaxis=dict(autorange="reversed"),
        height=320,
    )
    return figure


def build_lifestyle_breakdown_chart(score_payload: dict[str, object]) -> go.Figure:
    breakdown = score_payload["breakdown"]
    figure = go.Figure(
        go.Bar(
            x=list(breakdown.keys()),
            y=[float(value) for value in breakdown.values()],
            marker_color=[COLORS["teal"], COLORS["gold"], COLORS["coral"], COLORS["soft"]],
            text=[f"{float(value):.0f}" for value in breakdown.values()],
            textposition="outside",
        )
    )
    figure.update_layout(
        margin=dict(l=24, r=24, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100], title="Sub-score"),
        height=280,
    )
    return figure


def build_correlation_chart(
    reference_df: pd.DataFrame,
    x_field: str,
    y_field: str,
    user_point: tuple[float, float] | None = None,
) -> go.Figure:
    fields = [x_field, y_field]
    sample = reference_df[fields].dropna()
    if len(sample) > SCATTER_SAMPLE_SIZE:
        sample = sample.sample(SCATTER_SAMPLE_SIZE, random_state=42)

    x_values = sample[x_field].to_numpy(dtype=float)
    y_values = sample[y_field].to_numpy(dtype=float)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            name="Benchmark Users",
            marker=dict(
                color="rgba(26, 111, 99, 0.22)",
                size=7,
                line=dict(width=0),
            ),
        )
    )

    if len(sample) > 1 and len(np.unique(x_values)) > 1:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
        line_y = (slope * line_x) + intercept
        figure.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Trend",
                line=dict(color=COLORS["gold"], width=3),
            )
        )

    if user_point is not None:
        figure.add_trace(
            go.Scatter(
                x=[user_point[0]],
                y=[user_point[1]],
                mode="markers",
                name="Your Profile",
                marker=dict(color=COLORS["coral"], size=14, symbol="diamond"),
            )
        )

    figure.update_layout(
        margin=dict(l=24, r=24, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=UI_FIELD_LABELS.get(x_field, x_field),
        yaxis_title=UI_FIELD_LABELS.get(y_field, y_field),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return figure
