# 🌾 Winter Wheat Yield Forecasting — Sumner County, KS

---

## Quick Start & Running the Project

Follow these step-by-step instructions to set up, train, and run the project services.

### 1. Environment Setup

```bash
# Clone or navigate into the project directory
cd path/to/agr1

# Create a virtual environment
python -m venv venv

# Activate the virtual environment:
# - On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# - On Windows (CMD):
.\venv\Scripts\activate.bat
# - On Linux / macOS:
source venv/bin/activate

# Install the project and dependencies
pip install -e ".[dev]"
```

---

### 2. Model Training (Optional if using pre-trained artifacts)

Pre-trained artifacts (`model_xgboost.pkl`, `conformal_predictor.pkl`, `training_data.parquet`) are included in `data/processed/artifacts/`. If you want to retrain the pipeline from scratch (downloads weather data from NOAA):

```bash
python -m src.pipeline.train_pipeline
```

To force refresh all raw weather and yield datasets:
```bash
python -m src.pipeline.train_pipeline --force-refresh
```

---

### 3. Running the Application Services

The application consists of a **FastAPI backend API** and a **Streamlit frontend dashboard**.

#### Option A: Running in Two Terminal Windows

**Terminal 1 — Start FastAPI Backend:**
```bash
.\venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*API Swagger Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*  
*Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)*

**Terminal 2 — Start Streamlit Dashboard:**
```bash
.\venv\Scripts\Activate.ps1
streamlit run dashboard/app.py --server.port 8501
```
*Dashboard Interface: [http://127.0.0.1:8501](http://127.0.0.1:8501)*

#### Option B: Direct Python Execution (No activation required)

```bash
# Start API Backend
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Start Dashboard (in another terminal)
.\venv\Scripts\streamlit.exe run dashboard/app.py --server.port 8501
```

---

## Retrain with latest data

```bash
python -m src.pipeline.train_pipeline --force-refresh
```

---

## Run experiments (static vs drift-aware comparison)

```bash
python experiments/run_all.py
```

Results saved to `experiments/results/`

---

## How the Model Works

### 1. Train
The model learns from **77 years** of historical data (1948–2025).  
Each year: 40 weather features → yield (bu/acre).

### 2. New year arrives
Fetch that year's weather from NOAA.  
Compute 40 features: temperature, precipitation, frost days, heat days, dry spells — per growth stage (fall/winter/spring/summer).

### 3. Detect drift
Compare this year's weather against all training years.  
→ Is this year unusual compared to what the model was trained on?

```
LOW      — weather is normal → predict directly
MODERATE — weather is somewhat different → apply bias correction
HIGH     — weather is very different → retrain on recent years only
```

### 4. Predict
XGBoost produces a point prediction.  
MAPIE wraps it with a **90% confidence interval** — not just one number.

### 5. Adapt automatically

**MODERATE drift → Bias Correction**  
Look at the last 5 years of prediction errors.  
Shift the prediction by that average error.

**HIGH drift → Windowed Retrain**  
Discard old 1950s–1980s data.  
Retrain the model on only the most recent 20–40 years.  
Recent climate is more relevant than old decades.

### 6. Real example — Year 2025

| | Value |
|---|---|
| Summer rain | 398 mm (avg = 213 mm) — double normal |
| Heat days | 8 (avg = 19) — much cooler than usual |
| Drift detected | HIGH |
| Before adaptation | 30.6 bu/acre |
| After windowed retrain | **35.8 bu/acre** |
| Actual yield (NASS) | 46.3 bu/acre |
| Error reduced by | 33% |

---

## Verify adaptation is working

```bash
# Predict year 2025 — should show HIGH drift + windowed_retrain applied
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2025, "use_live_weather": false}'

# See which weather variables caused drift
curl http://localhost:8000/drift/2025

# Check all 77 years at once
curl http://localhost:8000/history

# Force adaptation on any year
curl -X POST http://localhost:8000/adapt \
  -H "Content-Type: application/json" \
  -d '{"year": 2023, "force": true}'
```

---

# 📚 Final Project Documentation

## 1. Abstract
This project presents an **Adaptive Temporal Out-of-Distribution (OOD) Winter Wheat Yield Forecasting** system for Sumner County, Kansas. It uses historical weather data and machine learning (XGBoost) to predict crop yields. Crucially, it incorporates **Drift Detection** and **Conformal Prediction** to identify unusual weather patterns and dynamically adapt predictions, ensuring reliable forecasts even in anomalous climate conditions.

## 2. Introduction
Accurate crop yield forecasting is critical for food security and agricultural planning. However, climate change has introduced significant volatility into weather patterns, causing traditional, static machine learning models to degrade over time. This project introduces a dynamic approach that monitors the weather for distribution shifts and recalibrates the model automatically.

## 3. Problem Statement
The goal is to predict winter wheat yield (bu/acre) for Sumner County, KS using daily weather data, while explicitly detecting when a new crop year presents out-of-distribution (OOD) weather compared to the training data, and adapting the prediction to maintain accuracy.

## 4. Objectives
- Build an end-to-end ML pipeline for crop yield prediction.
- Engineer agronomically meaningful features based on wheat growth stages.
- Quantify prediction uncertainty using Conformal Prediction.
- Detect temporal drift using univariate and multivariate statistical tests.
- Adapt predictions dynamically using Bias Correction and Windowed Retraining.

## 5. Existing System
Traditional agricultural ML systems train a static model on historical data. When deployed, they assume the new data follows the same distribution as the training data (i.i.d. assumption). When extreme weather events occur (droughts, heatwaves), these models fail silently, providing overconfident and inaccurate predictions.

## 6. Literature Review
- **Crop Yield Prediction**: Extensive research utilizes Random Forest and XGBoost for yield forecasting based on climate variables.
- **Conformal Prediction**: A framework for yielding prediction intervals with finite-sample coverage guarantees, increasingly applied in critical forecasting domains.
- **Drift Detection**: Techniques like the Kolmogorov-Smirnov (KS) test, Population Stability Index (PSI), and Mahalanobis distance are standard for identifying data shifts in production ML systems.

## 7. Research Gap
Most existing crop yield models lack robust mechanisms to handle temporal OOD data. Furthermore, they rarely provide statistically guaranteed confidence intervals. This project bridges that gap by combining yield forecasting with automated drift detection and conformal prediction.

## 8. Proposed System
The **Adaptive Temporal OOD Forecasting System** combines:
1. A strong baseline regressor (XGBoost).
2. A Conformal Predictor (MAPIE) for 90% confidence intervals.
3. A Drift Detector combining KS, PSI, Mahalanobis distance, and PCA reconstruction error.
4. An Adaptation Strategy that applies bias correction for moderate drift and windowed retraining for high drift.

## 9. Methodology
1. **Data Collection**: Fetch 77 years of yield and weather data via APIs.
2. **Feature Engineering**: Compute 40 features (temperature extremes, GDD, dry spells) segmented by growth stage (fall, winter, spring, summer).
3. **Training**: Train XGBoost on historical data.
4. **Calibration**: Calibrate MAPIE on a holdout set.
5. **Inference**: For a new year, fetch live weather, compute features, assess drift against the training distribution, and output an adapted prediction with an uncertainty interval.

## 10. Dataset
- **USDA NASS QuickStats API**: Historical winter wheat yield (bu/acre) for Sumner County, KS.
- **NOAA CDO / NWS APIs**: Daily weather observations (TMAX, TMIN, PRCP, SNOW) from the Caldwell, KS station (1948–Present).

## 11. Architecture
- **Frontend**: Streamlit Dashboard for interactive visualization.
- **Backend API**: FastAPI serving inference, drift, and history endpoints.
- **ML Pipeline**: Modular Python package (src/) handling data, modeling, drift detection, and adaptation.

## 12. Technology Stack
- **Languages**: Python 3.9+
- **Machine Learning**: Scikit-Learn, XGBoost, MAPIE
- **Data Processing**: Pandas, NumPy, PyArrow
- **Web / API**: FastAPI, Uvicorn, Streamlit, Plotly
- **Infrastructure**: Docker, Docker Compose

## 13. Implementation
The system is implemented as a modular Python package. The training pipeline (	rain_pipeline.py) caches raw data, engineers features, trains the model, and saves artifacts as pickles/parquet files. The inference pipeline (inference_pipeline.py) loads these artifacts, processes live weather, detects drift, and returns an adapted forecast.

## 14. AI Model
- **Baseline Model**: XGBoost Regressor (300 estimators, max depth 5, learning rate 0.05).
- **Uncertainty Wrapper**: Split Conformal Regressor (MAPIE) targeting a 90% confidence level.

## 15. API
The FastAPI backend exposes the following endpoints:
- POST /predict: Generates a forecast for a specific year.
- GET /drift/{year}: Returns detailed drift analysis.
- GET /history: Retrieves predictions and actuals for all historical years.
- GET /reliability/{year}: Assesses prediction reliability.
- POST /adapt: Forces a specific adaptation strategy for testing.

## 16. Database
The system uses lightweight local file storage:
- **Raw Data**: CSV files (data/raw/).
- **Processed Data**: Versioned Parquet files (data/processed/).
- **Model Artifacts**: Pickle files (data/processed/artifacts/).

## 17. Dashboard
The Streamlit dashboard provides a rich, glassmorphism-styled UI featuring:
- A prominent yield forecast gauge with confidence intervals.
- A drift status panel highlighting anomalous weather variables.
- A reliability card and adaptation status indicator.
- An interactive historical chart comparing actual vs. predicted yields over 70+ years.

## 18. Testing
The system was evaluated via chronological backtesting. Models were trained on expanding windows of historical data, and tested on subsequent years to evaluate point accuracy, conformal interval coverage, and adaptation effectiveness.

## 19. Experiments
Experiments were conducted to compare the static baseline model against the drift-aware adaptive model. The framework allows running backtests across all historical years to measure how adaptation strategies (bias correction, windowed retrain) impact the Mean Absolute Error (MAE).

## 20. Results
The adaptive model significantly reduces error during highly anomalous years. For example, in years with extreme summer precipitation or unusual temperature profiles (triggering a "HIGH" drift flag), applying the windowed retrain strategy successfully shifted predictions closer to the actual yield, reducing error by up to 30% compared to the static baseline.

## 21. Comparison
- **Static XGBoost**: Fast, but struggles during unprecedented climate events.
- **Adaptive XGBoost (Proposed)**: Slightly higher inference latency (during retraining), but much more robust to distribution shifts.

## 22. Limitations
- **Data Dependency**: Highly reliant on the continuity and quality of a single weather station (Caldwell, KS).
- **Spatial Scope**: Currently limited to a single county; spatial generalization requires further development.
- **Window Size**: The optimal window size for retraining depends heavily on the specific nature of the climate drift.

## 23. Future Scope
- Expand the model to cover multiple counties or entire states.
- Incorporate satellite imagery (e.g., NDVI) and soil moisture data.
- Implement deep learning approaches (e.g., LSTMs, Transformers) for the baseline forecaster.
- Deploy the system to a cloud provider (AWS/GCP) for public access.

## 24. Conclusion
The Adaptive Temporal OOD Winter Wheat Yield Forecasting system successfully demonstrates that integrating drift detection and automatic adaptation into agricultural ML pipelines can significantly improve robustness. By recognizing when it is operating outside its training distribution, the system avoids catastrophic failures and provides reliable, uncertainty-aware forecasts.

## 25. References
- USDA National Agricultural Statistics Service (NASS) QuickStats API.
- NOAA National Centers for Environmental Information (NCEI) Climate Data Online (CDO).
- MAPIE - Model Agnostic Prediction Interval Estimator Documentation.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
