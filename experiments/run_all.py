"""
Full experiment suite: annual rolling-origin evaluation of static vs. drift-aware models.

Protocol (leakage-free):
  * For every test year t with at least CV_MIN_TRAIN_YEARS earlier labeled years,
    train ONLY on years < t and predict t (one-step-ahead, expanding window).
  * Static and drift-aware/adapted variants use the SAME training rows at each origin.
  * Interval half-widths come from the model's own PAST out-of-sample residuals
    (rolling split-conformal), so no training data is spent on calibration.
  * Drift reference = years < t; severity from a calibrated empirical p-value.
  * Every model is compared with naive baselines; R2 is reported as computed
    (it can be negative) together with skill scores vs. those baselines.

Outputs (experiments/results/): static_results.csv, drift_aware_results.csv,
summary_comparison.csv, adaptation_log.json, run_metadata.json
"""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    CONFORMAL_ALPHA,
    CONFORMAL_MIN_RESIDUALS,
    CONFORMAL_WINDOW,
    CV_MIN_TRAIN_YEARS,
    EXPERIMENTS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
)
from src.adaptation.strategy import adapt_prediction, save_adaptation_log
from src.drift.detector import detect_drift
from src.models.baseline import ALL_MODELS, create_model, naive_forecasts
from src.models.conformal import ConformalResult, conformal_halfwidth, make_interval
from src.pipeline.train_pipeline import _drop_incomplete_weather_years

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = "yield_bu_per_acre"
NAIVE_NAMES = [
    "naive_expanding_mean", "naive_rolling5_mean", "naive_rolling10_mean",
    "naive_last_year", "naive_linear_trend",
]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mape": float(np.mean(np.abs(err) / np.abs(y_true)) * 100),
        "bias": float(np.mean(err)),
    }


def _skill(y_true, y_pred, y_ref) -> float:
    """MSE skill score vs a reference forecast: 1 - MSE_model / MSE_ref (>0 = better)."""
    mse = np.mean((y_true - y_pred) ** 2)
    mse_ref = np.mean((y_true - y_ref) ** 2)
    return float(1.0 - mse / mse_ref) if mse_ref > 0 else float("nan")


def _block_bootstrap_skill_ci(y_true, y_pred, y_ref, block=3, n_boot=2000):
    """95% moving-block-bootstrap CI for the MSE skill score."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    if n < 2 * block:
        return (float("nan"), float("nan"))
    e_m = (y_true - y_pred) ** 2
    e_r = (y_true - y_ref) ** 2
    n_blocks = math.ceil(n / block)
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        vals.append(1.0 - e_m[idx].mean() / e_r[idx].mean())
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiments():
    features_path = PROCESSED_DIR / "artifacts" / "training_data.parquet"
    if not features_path.exists():
        logger.error(
            "Training data not found at %s. Run: python -m src.pipeline.train_pipeline",
            features_path,
        )
        return

    df = pd.read_parquet(features_path)
    with open(PROCESSED_DIR / "artifacts" / "feature_cols.txt") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Idempotent safety net: exclude crop-years with incomplete weather even if the
    # training table was built before this check existed.
    df = _drop_incomplete_weather_years(df)
    df = df.sort_values("year").reset_index(drop=True)
    years = df["year"].astype(int).tolist()

    test_years = [t for i, t in enumerate(years) if i >= CV_MIN_TRAIN_YEARS]
    logger.info("=" * 70)
    logger.info("EXPERIMENT SUITE (annual rolling origin)")
    logger.info("Labeled years: %d (%d-%d), %d features", len(years), years[0], years[-1], len(feature_cols))
    logger.info("Test years: %d (%d-%d), min train years %d",
                len(test_years), test_years[0], test_years[-1], CV_MIN_TRAIN_YEARS)
    logger.info("=" * 70)

    static_rows, drift_rows = [], []

    # ---- naive baselines (history only) ----
    for t in test_years:
        past = df[df["year"] < t]
        fc = naive_forecasts(past["year"].values, past[TARGET].values, t)
        y_t = float(df.loc[df["year"] == t, TARGET].iloc[0])
        for name, pred in fc.items():
            static_rows.append({
                "model": name, "mode": "naive", "test_year": t, "train_size": len(past),
                "y_true": y_t, "y_pred": pred, "mae": abs(y_t - pred),
                "mape": abs(y_t - pred) / abs(y_t) * 100,
            })

    # ---- ML models ----
    for model_name in ALL_MODELS:
        logger.info("--- %s ---", model_name)
        cols = feature_cols + ["year"] if model_name == "detrended_ridge" else feature_cols
        hist_orig: list[float] = []    # signed residuals of the unadapted forecasts
        hist_adapt: list[float] = []   # signed residuals of the final (adapted) forecasts

        for t in test_years:
            train = df[df["year"] < t]
            test = df[df["year"] == t]
            X_tr, y_tr = train[cols].values, train[TARGET].values
            X_te, y_t = test[cols].values, float(test[TARGET].iloc[0])

            model = create_model(model_name)
            model.fit(X_tr, y_tr)
            y_pred = float(model.predict(X_te)[0])

            # Interval half-width from PAST out-of-sample residuals only
            enough = len(hist_orig) >= CONFORMAL_MIN_RESIDUALS
            hw = conformal_halfwidth(np.array(hist_orig), CONFORMAL_ALPHA, CONFORMAL_WINDOW) if enough else float("nan")
            cr = make_interval(y_pred, hw if enough else 0.0, CONFORMAL_ALPHA)

            # Drift (reference = years < t; year column excluded)
            drift = detect_drift(train[feature_cols].values, test[feature_cols].values, feature_cols)

            # Adaptation (uses the same training rows; bias uses past OOS residuals)
            adapted_cr, event = adapt_prediction(
                model=model, conformal_result=cr, drift_result=drift, df=df,
                feature_cols=cols, target_col=TARGET, test_year=int(t), X_test=X_te,
                recent_residuals=np.array(hist_orig), model_name=model_name,
            )
            y_adapted = float(adapted_cr.point_prediction)

            # Interval for the FINAL forecast is recalibrated from the final forecast's own past residuals
            enough_a = len(hist_adapt) >= CONFORMAL_MIN_RESIDUALS
            hw_a = conformal_halfwidth(np.array(hist_adapt), CONFORMAL_ALPHA, CONFORMAL_WINDOW) if enough_a else float("nan")
            final = make_interval(y_adapted, hw_a, CONFORMAL_ALPHA) if enough_a else None

            hist_orig.append(y_t - y_pred)
            hist_adapt.append(y_t - y_adapted)

            static_rows.append({
                "model": model_name, "mode": "static", "test_year": t, "train_size": len(y_tr),
                "y_true": y_t, "y_pred": y_pred, "mae": abs(y_t - y_pred),
                "mape": abs(y_t - y_pred) / abs(y_t) * 100,
            })
            drift_rows.append({
                "model": model_name, "mode": "drift_aware", "test_year": t, "train_size": len(y_tr),
                "y_true": y_t, "y_pred_original": y_pred, "y_pred_adapted": y_adapted,
                "mae_original": abs(y_t - y_pred), "mae_adapted": abs(y_t - y_adapted),
                "lower_bound": final.lower_bound if final else np.nan,
                "upper_bound": final.upper_bound if final else np.nan,
                "interval_width": final.interval_width if final else np.nan,
                "covered": bool(final.lower_bound <= y_t <= final.upper_bound) if final else np.nan,
                "drift_severity": drift.severity, "drift_score": drift.score,
                "drift_pvalue": drift.multivariate.mahalanobis_pvalue if drift.multivariate else np.nan,
                "adaptation_applied": event.adaptation_applied,
                "adaptation_strategy": event.strategy_used,
            })

    static_df = pd.DataFrame(static_rows)
    drift_df = pd.DataFrame(drift_rows)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    static_df.to_csv(EXPERIMENTS_DIR / "static_results.csv", index=False)
    drift_df.to_csv(EXPERIMENTS_DIR / "drift_aware_results.csv", index=False)

    # ---- summary ----
    piv = static_df.pivot(index="test_year", columns="model", values="y_pred")
    y_true = static_df[static_df["model"] == "naive_expanding_mean"].sort_values("test_year")["y_true"].values
    ref_mean = piv["naive_expanding_mean"].values
    ref_r5 = piv["naive_rolling5_mean"].values

    summary = []
    for name in NAIVE_NAMES + list(ALL_MODELS):
        yp = piv[name].values
        m = _metrics(y_true, yp)
        lo, hi = _block_bootstrap_skill_ci(y_true, yp, ref_mean)
        summary.append({
            "model": name, "mode": "naive" if name.startswith("naive") else "static",
            "n_splits": len(yp), **{k: round(v, 3) for k, v in m.items()},
            "skill_vs_expanding_mean": round(_skill(y_true, yp, ref_mean), 3),
            "skill_vs_expanding_mean_ci95": f"[{lo:.2f}, {hi:.2f}]",
            "skill_vs_rolling5": round(_skill(y_true, yp, ref_r5), 3),
        })

    for name in ALL_MODELS:
        d = drift_df[drift_df["model"] == name].sort_values("test_year")
        yt = d["y_true"].values
        m = _metrics(yt, d["y_pred_adapted"].values)
        cov = d.dropna(subset=["covered"])
        k, n = int(cov["covered"].astype(bool).sum()), len(cov)
        wlo, whi = _wilson(k, n)
        summary.append({
            "model": name, "mode": "drift_aware", "n_splits": len(d),
            **{k_: round(v, 3) for k_, v in m.items()},
            "skill_vs_expanding_mean": round(_skill(yt, d["y_pred_adapted"].values, ref_mean), 3),
            "skill_vs_rolling5": round(_skill(yt, d["y_pred_adapted"].values, ref_r5), 3),
            "coverage_pct": round(100 * k / n, 1) if n else np.nan,
            "coverage_ci95": f"[{100*wlo:.0f}, {100*whi:.0f}]",
            "n_with_interval": n,
            "avg_interval_width": round(float(cov["interval_width"].mean()), 2) if n else np.nan,
            "adaptation_rate_pct": round(float(d["adaptation_applied"].mean() * 100), 1),
            "n_drift_HIGH": int((d["drift_severity"] == "HIGH").sum()),
            "n_drift_MODERATE": int((d["drift_severity"] == "MODERATE").sum()),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(EXPERIMENTS_DIR / "summary_comparison.csv", index=False)
    logger.info("\n%s", summary_df.to_string(index=False))

    save_adaptation_log()

    # ---- metadata ----
    import sklearn, scipy, xgboost
    meta = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "xgboost": xgboost.__version__,
        "random_state": RANDOM_STATE, "n_labeled_years": len(years),
        "labeled_year_range": [years[0], years[-1]], "test_years": [test_years[0], test_years[-1]],
        "n_test_years": len(test_years), "min_train_years": CV_MIN_TRAIN_YEARS,
    }
    with open(EXPERIMENTS_DIR / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    best_naive = min(summary_df[summary_df["mode"] == "naive"]["rmse"])
    best_ml = summary_df[summary_df["mode"] == "static"].sort_values("rmse").iloc[0]
    logger.info("Best naive RMSE: %.2f | best static ML: %s RMSE %.2f, R2 %.3f",
                best_naive, best_ml["model"], best_ml["rmse"], best_ml["r2"])
    logger.info("Experiment suite complete. Results in %s", EXPERIMENTS_DIR)


if __name__ == "__main__":
    run_experiments()
