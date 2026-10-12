"""Historical yield vs. predicted — line chart with interval bands."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_history_chart(history: dict):
    """
    Render the historical actual vs. predicted yield line chart.

    Parameters
    ----------
    history : dict
        HistoryResponse data with 'records' list.
    """
    records = history.get("records", [])
    if not records:
        st.warning("No historical data available.")
        return

    years = [r["year"] for r in records]
    actuals = [r["actual_yield"] for r in records]
    predicted = [r["predicted_yield"] for r in records]
    lowers = [r.get("lower_bound") for r in records]
    uppers = [r.get("upper_bound") for r in records]

    fig = go.Figure()

    # Confidence interval band (if available)
    valid_bounds = [i for i, (l, u) in enumerate(zip(lowers, uppers)) if l is not None and u is not None]
    if valid_bounds:
        band_years = [years[i] for i in valid_bounds]
        band_lowers = [lowers[i] for i in valid_bounds]
        band_uppers = [uppers[i] for i in valid_bounds]

        fig.add_trace(go.Scatter(
            x=band_years + band_years[::-1],
            y=band_uppers + band_lowers[::-1],
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="90% Prediction Interval",
        ))

    # Actual yield
    fig.add_trace(go.Scatter(
        x=years,
        y=actuals,
        mode="lines+markers",
        name="Actual Yield",
        line=dict(color="#22c55e", width=2.5),
        marker=dict(size=5),
    ))

    # Predicted yield
    fig.add_trace(go.Scatter(
        x=years,
        y=predicted,
        mode="lines+markers",
        name="Predicted Yield",
        line=dict(color="#6366f1", width=2, dash="dash"),
        marker=dict(size=4, symbol="diamond"),
    ))

    # Highlight drift years
    for r in records:
        if r.get("drift_severity") == "HIGH":
            fig.add_vline(
                x=r["year"],
                line=dict(color="rgba(239, 68, 68, 0.3)", width=1, dash="dot"),
            )

    fig.update_layout(
        title="Historical Yield: Actual vs. Predicted",
        xaxis_title="Crop Year",
        yaxis_title="Yield (bu/acre)",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
    )

    st.plotly_chart(fig, use_container_width=True)
