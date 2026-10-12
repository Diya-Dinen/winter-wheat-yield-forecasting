"""Reliability card — RELIABLE/UNRELIABLE badge with explanation."""

from __future__ import annotations

import streamlit as st


def render_reliability_card(prediction: dict):
    """
    Render the reliability assessment card.

    Parameters
    ----------
    prediction : dict
        PredictResponse data (has reliability, drift_severity).
    """
    reliability = prediction.get("reliability", "UNRELIABLE")
    is_reliable = reliability == "RELIABLE"

    color = "#22c55e" if is_reliable else "#ef4444"
    icon = "✅" if is_reliable else "⚠️"
    interval_width = prediction.get("upper_bound", 0) - prediction.get("lower_bound", 0)

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
                Prediction Reliability
            </div>
            <div style="font-size: 28px; font-weight: 700; color: {color};">
                {icon} {reliability}
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 8px;">
                Interval width: {interval_width:.1f} bu/acre<br>
                Confidence level: 90%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explanation
    if is_reliable:
        st.markdown(
            "The prediction interval has adequate coverage and weather conditions "
            "are within the model's training distribution."
        )
    else:
        reasons = []
        if prediction.get("drift_severity") == "HIGH":
            reasons.append("Weather conditions show **significant distribution shift** from training data.")
        if interval_width > 20:
            reasons.append("Prediction interval is **unusually wide**, indicating high uncertainty.")
        if prediction.get("weather_source") == "forecast":
            reasons.append("Weather data includes **forecast** values with inherent uncertainty.")

        if reasons:
            st.markdown("**Why unreliable:**")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("Moderate drift or wide interval reduces prediction confidence.")
