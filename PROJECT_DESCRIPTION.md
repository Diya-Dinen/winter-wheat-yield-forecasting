# 🌾 Adaptive Temporal Out-of-Distribution (OOD) Winter Wheat Yield Forecasting
## End-to-End System Project Description & Technical Documentation

---

## 📌 1. Executive Summary

This project implements an **Adaptive, Uncertainty-Aware Machine Learning System** designed to forecast annual **Winter Wheat Yield (bushels per acre)** for **Sumner County, Kansas**. 

Traditional machine learning models deployed in agricultural environments assume that historical climate distributions will remain constant over time (i.i.d. assumption). However, climate change and extreme weather events (such as droughts, unexpected heatwaves, or unseasonal summer rain) introduce significant **Temporal Out-of-Distribution (OOD)** shifts. When unhandled, static ML models suffer severe performance degradation and produce overconfident, inaccurate forecasts.

To overcome this, this project combines **XGBoost Regression**, **MAPIE Split Conformal Prediction** (to deliver 90% statistically guaranteed prediction intervals), **Multi-Method Statistical Drift Detection** (KS-Test, PSI, Mahalanobis Distance, PCA Reconstruction Error), and **Dynamic Adaptation Strategies** (Bias Correction for moderate drift and Windowed Retraining for severe drift).

---

## 🏗️ 2. High-Level Architecture Overview

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         1. DATA INGESTION                              │
 │  USDA NASS QuickStats API (Yields) ───► Daily Weather & Yield Cache   │
 │  NOAA CDO / NWS APIs (Weather)    ───► 77 Years (1948–2025) Data      │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     2. FEATURE ENGINEERING                             │
 │  Transform Daily Weather into 40 Growth-Stage Segmented Features       │
 │  (Fall Planting, Winter Dormancy, Spring Greenup, Summer Harvest)      │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    3. DRIFT DETECTION ENGINE                           │
 │  Univariate: Kolmogorov-Smirnov (KS) Test, Population Stability Index  │
 │  Multivariate: Mahalanobis Distance, PCA Reconstruction Error          │
 │  Classification: LOW 🟢  |  MODERATE 🟡  |  HIGH 🔴 Risk Level         │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                  4. MODEL & ADAPTATION PIPELINE                        │
 │  • LOW DRIFT      ──► Standard XGBoost + MAPIE Conformal Intervals     │
 │  • MODERATE DRIFT ──► Bias Correction (Shift by 5-yr rolling error)    │
 │  • HIGH DRIFT     ──► Windowed Retrain (Train on recent 20-40 years)   │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    5. SERVING & PRESENTATION                           │
 │  FastAPI REST Service (/predict, /drift, /history, /adapt)             │
 │  Streamlit Glassmorphism Web Dashboard & Interactive Analytics        │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 3. Problem Statement & Business Context

### Context
Sumner County, Kansas, is known as the "Wheat Capital of the World". Forecasting crop yields months before harvest is essential for regional grain elevators, insurance underwriters, commodity traders, and agricultural policy makers.

### The Challenge
- **Decadal Climate Shift**: Temperature and precipitation distributions from 1950–1980 differ significantly from modern weather patterns (2010–2025).
- **Silent Model Failure**: Standard supervised models evaluate new data against legacy distributions and fail to signal uncertainty when climate parameters breach historical bounds.
- **Actionable Risk Quantification**: Farmers and traders need not just a point prediction (e.g., "35.2 bu/acre"), but a calibrated range (e.g., "90% chance yield will be between 29.1 and 41.3 bu/acre").

---

## 🌐 4. Data Ingestion & Datasets

The project automatically retrieves and caches data spanning **77 crop years (1948–2025)**:

1. **USDA NASS QuickStats API**:
   - **Target Variable**: Winter Wheat Yield measured in `BU / ACRE`.
   - Fetches historical county-level crop production records for Sumner County, KS (FIPS 20191).
   - Implementation: [`src/data/fetch_nass.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/data/fetch_nass.py)

2. **NOAA NCEI Climate Data Online (CDO) & NWS APIs**:
   - Station: Caldwell, KS (USC00141233).
   - Daily Metrics: Maximum Temperature (`TMAX`), Minimum Temperature (`TMIN`), Precipitation (`PRCP`), Snowfall (`SNOW`).
   - Handles missing data imputation and unit normalization (°C, mm).
   - Implementation: [`src/data/fetch_noaa.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/data/fetch_noaa.py)

---

## ⚙️ 5. Agronomic Feature Engineering

Rather than feeding raw daily temperatures into a black-box model, daily weather observations are aggregated into **40 agronomically relevant features** grouped by winter wheat crop development stages:

- **Growth Stages**:
  1. **Fall (Oct – Dec)**: Germination, emergence, and initial tillering.
  2. **Winter (Jan – Feb)**: Dormancy, cold hardiness, and winterkill protection.
  3. **Spring (Mar – May)**: Greenup, jointing, heading, and flowering.
  4. **Summer (Jun – Jul)**: Grain filling and harvest phase.

- **Engineered Metrics per Stage**:
  - Growing Degree Days (GDD, base temperature 0°C and 4.4°C).
  - Cumulative Precipitation & Dry Spell Duration (max consecutive days with PRCP < 1mm).
  - Extreme Heat Days ($T_{max} > 32^\circ C$ or $35^\circ C$).
  - Freeze Risk / Frost Days ($T_{min} < 0^\circ C$ during critical flowering/spring phases).
  - Diurnal Temperature Range (DTR).

Implementation: [`src/data/feature_engineering.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/data/feature_engineering.py)

---

## 🤖 6. Machine Learning Model & Conformal Prediction

### Baseline Model: XGBoost Regressor
- Hyperparameters tuned for time-series tabular data (`n_estimators=300`, `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`).
- Predicts expected point yield $\hat{y}$ given the 40 seasonal weather features.
- Implementation: [`src/models/baseline.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/models/baseline.py)

### Conformal Prediction: MAPIE (Model-Agnostic Prediction Interval Estimator)
- Uses **Split Conformal Regression** to produce prediction bounds $[\hat{y}_{lower}, \hat{y}_{upper}]$.
- Provides finite-sample coverage guarantees ($1 - \alpha = 90\%$).
- Adaptively expands interval widths when weather features diverge from training distributions.
- Implementation: [`src/models/conformal.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/models/conformal.py)

---

## 📊 7. Drift Detection Framework

To flag Temporal OOD conditions, the system evaluates incoming seasonal weather features against reference baseline historical distributions using a ensemble drift detector:

### 1. Univariate Statistical Tests
- **Kolmogorov-Smirnov (KS) Test**: Evaluates whether individual feature distributions differ significantly ($p < 0.05$).
- **Population Stability Index (PSI)**: Measures distribution shifts across binned quantiles ($\text{PSI} > 0.25$ indicates major drift).
- Implementation: [`src/drift/univariate.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/drift/univariate.py)

### 2. Multivariate Geometric Tests
- **Mahalanobis Distance**: Accounts for covariance across all 40 weather features to detect multidimensional extreme anomalies.
- **PCA Reconstruction Error**: Projects features onto principal components and measures residual reconstruction error for out-of-subspace anomalies.
- Implementation: [`src/drift/multivariate.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/drift/multivariate.py)

### 3. Risk Level Classification
- 🟢 **LOW**: $0 - 2$ drifting features. No adaptation required.
- 🟡 **MODERATE**: $3 - 5$ drifting features or elevated Mahalanobis distance. Trigger **Bias Correction**.
- 🔴 **HIGH**: $> 5$ drifting features or extreme multivariate anomaly score. Trigger **Windowed Retraining**.

Implementation: [`src/drift/detector.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/drift/detector.py)

---

## 🔄 8. Dynamic Adaptation Strategies

When drift is detected for a crop year, the system dynamically alters its inference pipeline:

1. **Bias Correction Strategy (Moderate Drift)**:
   - Computes rolling residuals from the preceding $k=5$ crop years.
   - Adjusts point prediction by adding the systematic bias vector:
     $$\hat{y}_{adapted} = \hat{y}_{base} + \frac{1}{k} \sum_{i=1}^k (y_{t-i} - \hat{y}_{t-i})$$
   - Implementation: [`src/adaptation/bias_correction.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/adaptation/bias_correction.py)

2. **Windowed Retraining Strategy (High Drift)**:
   - Truncates distant legacy training data (e.g., 1950s–1980s) that may no longer reflect current climate regimes.
   - Retrains XGBoost and recalibrates MAPIE using only the recent window (e.g., past 25–35 years).
   - Implementation: [`src/adaptation/windowed_retrain.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/adaptation/windowed_retrain.py)

3. **Adaptation Controller Strategy Router**:
   - Encapsulates strategy decision making and fallback logic.
   - Implementation: [`src/adaptation/strategy.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/src/adaptation/strategy.py)

---

## 🌐 9. API Backend Microservice (FastAPI)

The backend provides high-performance RESTful endpoints built with FastAPI and Pydantic schemas:

- `POST /predict`: Generates yield forecast, conformal interval, drift status, and applied adaptation strategy for a given year.
- `GET /drift/{year}`: Detailed breakdown of weather variables driving drift (p-values, PSI scores, Mahalanobis distances).
- `GET /history`: Returns historical ground truth yield vs. predicted yield and conformal intervals for all 77 years.
- `GET /reliability/{year}`: Calculates conformal interval coverage and reliability statistics.
- `POST /adapt`: Forces execution of a specific adaptation strategy (for testing/experimentation).
- `GET /health`: System diagnostic health check.

Implementation: [`api/main.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/api/main.py) & [`api/routes/`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/api/routes)

---

## 🖥️ 10. Interactive Streamlit UI Dashboard

The user frontend is a glassmorphic dashboard built with Streamlit and Plotly:

- **Yield Forecast Header**: Large radial gauge showing predicted yield (bu/acre) alongside 90% confidence boundaries.
- **Drift Indicator Badge**: Visual color-coded status badge (LOW, MODERATE, HIGH) with detailed expandable weather attribute chips.
- **Adaptation Log**: Explains why a specific strategy (e.g. Windowed Retrain) was triggered and its impact on forecast error.
- **77-Year Historical Chart**: Interactive Plotly time-series chart showing actual yield, point predictions, confidence intervals, and drift indicators across decades.

Implementation: [`dashboard/app.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/dashboard/app.py)

---

## 🧪 11. Experimental Evaluation & Backtesting Results

The system includes a complete backtesting framework ([`experiments/run_all.py`](file:///c:/Users/DiyaDinen/Downloads/agr1/agr1/experiments/run_all.py)) that executes walk-forward temporal evaluation across historical years.

### Key Finding (Case Study: Year 2025)
- **Extreme Weather Profile**: Summer precipitation reached 398 mm (average is 213 mm), while heat days dropped significantly.
- **Drift Detected**: HIGH risk level triggered due to extreme summer precipitation anomaly.
- **Static Baseline Prediction**: $30.6\text{ bu/acre}$
- **Adaptive Prediction (Windowed Retrain)**: **$35.8\text{ bu/acre}$**
- **Ground Truth Yield**: $46.3\text{ bu/acre}$
- **Outcome**: Adaptation reduced yield prediction error by **33%** compared to the static model.

---

## 📁 12. Project Directory Structure

```
agr1/
├── api/                        # FastAPI Web Backend
│   ├── main.py                 # FastAPI application entrypoint
│   ├── dependencies.py         # Shared API dependencies & singletons
│   ├── schemas.py              # Pydantic request/response schemas
│   └── routes/                 # Route controllers (predict, drift, history, adapt)
├── config/                     # Configuration parameters & settings
├── dashboard/                  # Streamlit Frontend Web App
│   ├── app.py                  # Main Streamlit dashboard
│   ├── utils.py                # Dashboard API client & formatting helpers
│   └── components/             # Reusable UI widgets & custom HTML/CSS
├── data/                       # Data storage directory
│   ├── raw/                    # Raw weather and yield downloads
│   ├── processed/              # Feature matrices & Parquet files
│   └── processed/artifacts/    # Trained models, scalers & conformal predictors
├── docker/                     # Dockerfiles & container configs
├── experiments/                # Research scripts & backtesting framework
│   ├── run_all.py              # Complete benchmark pipeline script
│   └── results/                # Benchmarking JSON/CSV evaluation logs
├── src/                        # Core Python Machine Learning Package
│   ├── data/                   # NOAA/NASS fetching & agronomic feature engineering
│   ├── models/                 # XGBoost training, validation & MAPIE conformal wrapper
│   ├── drift/                  # KS-test, PSI, Mahalanobis & PCA drift detection algorithms
│   ├── adaptation/             # Bias correction & windowed retraining logic
│   └── pipeline/               # End-to-end execution pipelines (train & inference)
├── pyproject.toml              # Build settings & project dependencies
└── README.md                   # Quickstart guide & environment setup instructions
```

---

## 🚀 13. Running & Deploying the System

### Local Quickstart
```bash
# 1. Activate Environment
.\venv\Scripts\Activate.ps1

# 2. Run FastAPI Backend (Port 8000)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# 3. Run Streamlit Dashboard (Port 8501)
streamlit run dashboard/app.py --server.port 8501
```

### Docker Deployment
```bash
# Build and launch services using Docker Compose
docker-compose up --build
```

---

## 🔮 14. Future Scope & Roadmap

1. **Spatial Expansion**: Extend the single-county Caldwell/Sumner framework to grid-level spatial grids across the entire US Wheat Belt.
2. **Satellite & Remote Sensing**: Integrate Sentinel-2 NDVI/EVI vegetation indices and SMAP soil moisture data.
3. **Deep Learning Backbones**: Incorporate Temporal Fusion Transformers (TFT) or Temporal Convolutional Networks (TCN) alongside XGBoost.
4. **Automated Continuous Re-calibration**: Implement automated retraining triggers in cloud MLOps environments (e.g. AWS SageMaker Pipelines / GCP Vertex AI).
