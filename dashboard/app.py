"""
Winter Wheat Yield Forecasting Dashboard — Sumner County, KS

Single-page Streamlit application displaying:
  1. Yield forecast + prediction interval (gauge)
  2. Historical actual vs. predicted (line chart)
  3. Drift status (severity badge + variable breakdown)
  4. Reliability flag
  5. Adaptation status

Communicates with the FastAPI backend via httpx.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Winter Wheat Yield Forecast — Sumner Co., KS",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark theme, glassmorphism, Inter font
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 24px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Import components
# ---------------------------------------------------------------------------
from dashboard.components.yield_gauge import render_yield_gauge
from dashboard.components.history_chart import render_history_chart
from dashboard.components.drift_panel import render_drift_panel
from dashboard.components.reliability_card import render_reliability_card
from dashboard.components.adaptation_status import render_adaptation_status
from dashboard.utils import get_health, post_predict, get_drift, get_history

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🌾 Winter Wheat Yield Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Adaptive Temporal OOD Forecasting — Sumner County, Kansas</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # API health check
    health = get_health()
    if health.get("model_loaded"):
        st.success(f"✅ API online — {health['data_years']} years loaded")
    else:
        st.error("❌ API offline or model not loaded")
        st.info("Start the API: `uvicorn api.main:app`\nThen run the training pipeline.")

    st.markdown("---")

    # Year selector
    year = st.number_input(
        "Crop Year",
        min_value=1950,
        max_value=2100,
        value=2023,
        step=1,
        help="Select the crop year to forecast (Oct year-1 through Jul year)",
    )

    use_live = st.checkbox(
        "Use live weather",
        value=False,
        help="Fetch current weather from NOAA NWS (for current/future years)",
    )

    predict_btn = st.button("🔍 Generate Forecast", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown(
        "This dashboard forecasts winter wheat yield using adaptive "
        "machine learning with temporal drift detection and conformal "
        "prediction intervals."
    )
    st.markdown(
        "**Data sources:** USDA NASS, NOAA CDO/NWS  \n"
        "**Models:** Ridge, Random Forest, XGBoost  \n"
        "**Uncertainty:** Split conformal prediction (90%)"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if predict_btn and health.get("model_loaded"):
    with st.spinner("Generating forecast..."):
        try:
            prediction = post_predict(year, use_live)

            # Row 1: Yield gauge (full width)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            render_yield_gauge(prediction)
            st.markdown('</div>', unsafe_allow_html=True)

            # Row 2: Drift + Reliability + Adaptation (3 columns)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                # Try to get detailed drift
                try:
                    drift_details = get_drift(year)
                except Exception:
                    drift_details = None
                render_drift_panel(prediction, drift_details)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                render_reliability_card(prediction)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                render_adaptation_status(prediction)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as exc:
            err = str(exc)
            # Extract the JSON detail message from a 400/422 API error if present
            detail = err
            if "400" in err or "422" in err:
                import re
                m = re.search(r'"detail"\s*:\s*"([^"]+)"', err)
                if m:
                    detail = m.group(1)
                elif "detail" in err:
                    try:
                        import json, httpx
                        # Re-request to get structured error
                    except Exception:
                        pass

            st.error(f"**Forecast failed for {year}**")

            # Show a helpful human-readable explanation
            low = detail.lower()
            if "no stored features" in low or "no historical features" in low:
                st.warning(
                    f"**Year {year} is not in the training data.**  \n"
                    f"The model was trained on years up to 2023.  \n"
                    f"To forecast year {year}, re-run the training pipeline:  \n"
                    f"```\npython -m src.pipeline.train_pipeline --force-refresh\n```"
                )
            elif "future" in low or "in-progress" in low:
                st.warning(
                    f"**Year {year} is a future growing season.**  \n"
                    f"Enable **'Use live weather'** in the sidebar to get a forecast "
                    f"based on current NOAA observations."
                )
            elif "live weather" in low or "nws" in low or "no live weather" in low:
                st.warning(
                    f"**Live weather data unavailable for {year}.**  \n"
                    f"The NOAA NWS API did not return observations for this period.  \n"
                    f"This can happen if:  \n"
                    f"- The growing season has not started yet (before Oct)  \n"
                    f"- The NWS API is temporarily unavailable  \n"
                    f"Try a historical year (≤ 2023) or try again later."
                )
            elif "not fitted" in low or "model" in low:
                st.warning(
                    "**Model not ready.**  \n"
                    "Run the training pipeline first:  \n"
                    "```\npython -m src.pipeline.train_pipeline\n```"
                )
            else:
                st.warning(f"**Detail:** {detail}")

# Row 3: Historical chart (always show if API is up)
if health.get("model_loaded"):
    st.markdown("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    try:
        history = get_history()
        render_history_chart(history)
    except Exception as exc:
        st.warning(f"Could not load history: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

elif not health.get("model_loaded"):
    st.markdown("---")
    st.markdown(
        """
        ### 🚀 Getting Started

        1. **Set up API keys** — Copy `.env.example` to `.env` and add your USDA NASS and NOAA CDO keys
        2. **Install dependencies** — `pip install -e .`
        3. **Run training pipeline** — `python -m src.pipeline.train_pipeline`
        4. **Start API** — `uvicorn api.main:app --reload`
        5. **Launch dashboard** — `streamlit run dashboard/app.py`
        """
    )
