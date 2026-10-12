"""Drift status panel — severity badge + contributing variable breakdown."""

from __future__ import annotations

import streamlit as st


# Severity color mapping
_SEVERITY_COLORS = {
    "LOW": ("#22c55e", "🟢"),
    "MODERATE": ("#f59e0b", "🟡"),
    "HIGH": ("#ef4444", "🔴"),
}


def render_drift_panel(prediction: dict, drift_details: dict | None = None):
    """
    Render the drift status panel.

    Parameters
    ----------
    prediction : dict
        PredictResponse data (has drift_severity, drift_score, contributing_variables).
    drift_details : dict, optional
        Full DriftResponse data for detailed breakdown.
    """
    severity = prediction.get("drift_severity", "LOW")
    score = prediction.get("drift_score", 0.0)
    contributors = prediction.get("contributing_variables", [])
    color, emoji = _SEVERITY_COLORS.get(severity, ("#94a3b8", "⚪"))

    # Severity badge
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
                Distribution Shift Status
            </div>
            <div style="font-size: 28px; font-weight: 700; color: {color};">
                {emoji} {severity}
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">
                Composite drift score: {score:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Contributing variables
    if contributors:
        st.markdown("**Contributing Variables:**")
        for var in contributors:
            st.markdown(f"- `{var}`")
    else:
        st.markdown("*No significant variable drift detected.*")

    # Detailed breakdown (if available)
    if drift_details and drift_details.get("univariate_summary"):
        uni = drift_details["univariate_summary"]
        with st.expander("📊 Detailed Drift Breakdown"):
            st.markdown(f"**Features tested:** {uni.get('n_features_tested', '?')}")
            st.markdown(f"**Features drifted:** {uni.get('n_features_drifted', '?')}")
            st.markdown(f"**Max PSI:** {uni.get('max_psi', 0):.4f}")

            drifted = uni.get("drifted_features", [])
            if drifted:
                st.markdown("| Feature | KS p-value | PSI | Severity |")
                st.markdown("|---|---|---|---|")
                for feat in drifted:
                    st.markdown(
                        f"| `{feat['feature']}` | {feat['ks_pvalue']:.4f} | "
                        f"{feat['psi_score']:.4f} | {feat['psi_severity']} |"
                    )

            if drift_details.get("multivariate"):
                multi = drift_details["multivariate"]
                st.markdown("---")
                st.markdown("**Multivariate Analysis:**")
                st.markdown(
                    f"- Mahalanobis distance: **{multi['mahalanobis_distance']:.2f}** "
                    f"({multi['mahalanobis_severity']})"
                )
                st.markdown(
                    f"- PCA reconstruction error: {multi['pca_reconstruction_error']:.4f} "
                    f"(p{multi['pca_reconstruction_percentile']:.0f})"
                )
