"""Adaptation status panel — current status + event log."""

from __future__ import annotations

import streamlit as st


def render_adaptation_status(prediction: dict):
    """
    Render the adaptation status panel.

    Parameters
    ----------
    prediction : dict
        PredictResponse data (has adaptation_applied, adaptation_strategy).
    """
    adapted = prediction.get("adaptation_applied", False)
    strategy = prediction.get("adaptation_strategy", "none")

    if adapted:
        color = "#3b82f6"
        icon = "🔄"
        status_text = "APPLIED"
        strategy_display = strategy.replace("_", " ").title()
    else:
        color = "#22c55e"
        icon = "✓"
        status_text = "NOT NEEDED"
        strategy_display = "—"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border: 1px solid {color}44;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
        ">
            <div style="font-size: 14px; color: #94a3b8; margin-bottom: 4px;">
                Adaptive Recalibration
            </div>
            <div style="font-size: 28px; font-weight: 700; color: {color};">
                {icon} {status_text}
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">
                Strategy: {strategy_display}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explanation
    if adapted:
        if strategy == "bias_correction":
            st.markdown(
                "📝 **Bias correction** applied: the prediction was shifted based on "
                "recent residual patterns to account for detected distribution drift."
            )
        elif strategy == "windowed_retrain":
            st.markdown(
                "📝 **Windowed retrain** applied: the model was retrained on recent "
                "years only, giving more weight to current climate patterns."
            )
        elif strategy == "ensemble":
            st.markdown(
                "📝 **Ensemble blending** applied: predictions from the full-history "
                "and recent-window models were blended, weighting toward recent data."
            )
    else:
        st.markdown(
            "Current weather conditions are within the model's training distribution. "
            "No recalibration needed."
        )

    # Lead time note
    lead_note = prediction.get("lead_time_note", "")
    if lead_note:
        st.info(f"📅 **Lead time:** {lead_note}")
