"""Predicted yield + interval gauge/band visualization."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_yield_gauge(prediction: dict):
    """
    Render the yield forecast panel with gauge and interval band.

    Parameters
    ----------
    prediction : dict
        PredictResponse data.
    """
    predicted = prediction["predicted_yield"]
    lower = prediction["lower_bound"]
    upper = prediction["upper_bound"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Predicted Yield",
            value=f"{predicted:.1f} bu/acre",
        )
    with col2:
        st.metric(
            label="90% Interval Lower",
            value=f"{lower:.1f} bu/acre",
        )
    with col3:
        st.metric(
            label="90% Interval Upper",
            value=f"{upper:.1f} bu/acre",
        )

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted,
        number={"suffix": " bu/acre", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 80], "ticksuffix": " bu/acre"},
            "bar": {"color": "#6366f1"},
            "steps": [
                {"range": [0, 20], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [20, 40], "color": "rgba(251, 191, 36, 0.15)"},
                {"range": [40, 60], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [60, 80], "color": "rgba(59, 130, 246, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#f43f5e", "width": 3},
                "thickness": 0.8,
                "value": predicted,
            },
        },
        title={"text": f"Crop Year {prediction['year']}", "font": {"size": 16}},
    ))

    # Add interval annotation
    fig.add_annotation(
        x=0.5, y=-0.15,
        text=f"90% Prediction Interval: [{lower:.1f}, {upper:.1f}] bu/acre",
        showarrow=False,
        font={"size": 14, "color": "#94a3b8"},
        xref="paper", yref="paper",
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )

    st.plotly_chart(fig, use_container_width=True)
