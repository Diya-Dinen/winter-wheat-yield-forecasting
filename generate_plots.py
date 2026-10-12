import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

base_dir = r"c:\Users\DiyaDinen\Downloads\agr1\agr1"
results_csv = os.path.join(base_dir, "experiments", "results", "drift_aware_results.csv")
artifacts_dir = os.path.join(base_dir, "data", "processed", "artifacts")

df = pd.read_csv(results_csv)
df_xgb = df[df["model"] == "xgboost"].sort_values("test_year")

# 3. ACTUAL VS PREDICTED CROP YIELD
plt.figure(figsize=(10, 5))
plt.plot(df_xgb["test_year"], df_xgb["y_true"], label="Actual Yield", marker='o', color='#1f77b4')
plt.plot(df_xgb["test_year"], df_xgb["y_pred_adapted"], label="Predicted Yield (Adaptive)", marker='s', color='#ff7f0e')
plt.xlabel("Year")
plt.ylabel("Crop Yield (bu/acre)")
plt.title("Actual vs Predicted Winter Wheat Yield")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fig_actual_vs_predicted.png", dpi=300)
plt.close()

# 4. CLIMATE / FEATURE DRIFT
plt.figure(figsize=(10, 5))
plt.bar(df_xgb["test_year"], df_xgb["drift_score"], color='lightcoral', edgecolor='black')
plt.axhline(y=0.25, color='r', linestyle='--', label='Drift Threshold (PSI > 0.25)')
plt.xlabel("Year")
plt.ylabel("Overall Drift Score")
plt.title("Detected Climate/Feature Drift Over Time")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fig_climate_drift.png", dpi=300)
plt.close()

# 5. PREDICTION INTERVAL / UNCERTAINTY
plt.figure(figsize=(10, 5))
plt.fill_between(df_xgb["test_year"], df_xgb["lower_bound"], df_xgb["upper_bound"], color='#ff7f0e', alpha=0.3, label="90% Prediction Interval")
plt.plot(df_xgb["test_year"], df_xgb["y_true"], label="Actual Yield", marker='o', color='#1f77b4')
plt.plot(df_xgb["test_year"], df_xgb["y_pred_adapted"], label="Predicted Yield", linestyle='--', color='#d62728')
plt.xlabel("Year")
plt.ylabel("Crop Yield (bu/acre)")
plt.title("Yield Prediction with Uncertainty Intervals (MAPIE)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fig_prediction_uncertainty.png", dpi=300)
plt.close()

# 6. FEATURE IMPORTANCE
try:
    with open(os.path.join(artifacts_dir, "model_xgboost.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(artifacts_dir, "feature_cols.txt"), "r") as f:
        features = f.read().splitlines()
    
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimator"): # MAPIE wrapping
        if hasattr(model.estimator, "feature_importances_"):
             importances = model.estimator.feature_importances_
        elif hasattr(model.estimator, "estimator_") and hasattr(model.estimator.estimator_, "feature_importances_"):
             importances = model.estimator.estimator_.feature_importances_
    elif hasattr(model, "_model") and hasattr(model._model, "feature_importances_"):
        importances = model._model.feature_importances_
        
    if importances is not None:
        indices = np.argsort(importances)[-15:]
        top_features = [features[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_features, top_importances, color='teal')
        plt.xlabel("Relative Importance")
        plt.title("Top 15 Feature Importances (XGBoost)")
        plt.tight_layout()
        plt.savefig("fig_feature_importance.png", dpi=300)
        plt.close()
    else:
        print("Could not extract feature importances from model.")
except Exception as e:
    print(f"Error generating feature importance: {e}")

# 1. & 2. Draw diagrams using matplotlib 
import matplotlib.patches as patches

# 1. Architecture
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

def draw_box(ax, x, y, width, height, text, facecolor='#e6f2ff'):
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                 edgecolor='black', facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, weight='bold')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8))

draw_box(ax, 0.5, 4.5, 2.5, 0.8, "NOAA/NASS APIs\n(Data Ingestion)", "#d9ead3")
draw_box(ax, 4.0, 4.5, 2.5, 0.8, "Feature Engineering\n(40 Growth-Stage Vars)", "#cfe2f3")
draw_box(ax, 7.5, 4.5, 3.0, 0.8, "Drift Detection Engine\n(KS-Test, PSI, Mahalanobis)", "#fff2cc")

draw_box(ax, 0.5, 2.0, 3.0, 0.8, "XGBoost Regressor\n(Base Yield Model)", "#fce5cd")
draw_box(ax, 4.5, 2.0, 3.0, 0.8, "Adaptation Strategies\n(Bias Correction / Retrain)", "#f4cccc")
draw_box(ax, 8.5, 2.0, 2.5, 0.8, "MAPIE Conformal\nPrediction Intervals", "#ead1dc")

draw_box(ax, 4.5, -0.5, 3.0, 0.8, "FastAPI & Streamlit\n(Dashboard / Output)", "#d0e0e3")

draw_arrow(ax, 1.75, 4.5, 1.75, 2.8)
draw_arrow(ax, 3.0, 4.9, 4.0, 4.9)
draw_arrow(ax, 6.5, 4.9, 7.5, 4.9)
draw_arrow(ax, 9.0, 4.5, 9.0, 2.8)
draw_arrow(ax, 9.0, 4.5, 6.0, 2.8)
draw_arrow(ax, 3.5, 2.4, 4.5, 2.4)
draw_arrow(ax, 7.5, 2.4, 8.5, 2.4)
draw_arrow(ax, 6.0, 2.0, 6.0, 0.3)

plt.title("Overall System Architecture", fontsize=14, weight='bold')
plt.savefig("fig_system_architecture.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Workflow
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')
steps = [
    "New Environmental & Weather Data",
    "Preprocessing & Missing Data Imputation",
    "Agronomic Feature Engineering (40 variables)",
    "XGBoost Yield Prediction (Point Estimate)",
    "Temporal Drift Detection (PSI, Mahalanobis)",
    "Adaptive Strategy (e.g. Windowed Retraining)",
    "MAPIE Uncertainty Estimation (90% Interval)",
    "Final Forecast Output & Dashboard"
]
colors = ['#d9ead3', '#cfe2f3', '#cfe2f3', '#fce5cd', '#fff2cc', '#f4cccc', '#ead1dc', '#d0e0e3']

for i, (step, color) in enumerate(zip(steps, colors)):
    y_pos = 8 - i*1.2
    draw_box(ax, 3, y_pos, 4, 0.8, step, color)
    if i < len(steps) - 1:
        draw_arrow(ax, 5, y_pos, 5, y_pos - 0.4)

plt.title("Crop Yield Forecasting Workflow", fontsize=14, weight='bold')
plt.savefig("fig_forecasting_workflow.png", dpi=300, bbox_inches='tight')
plt.close()
