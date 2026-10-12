"""
End-to-end training pipeline.

Orchestrates: data fetch → preprocess → feature engineering → model training →
drift detection → conformal calibration → adaptation setup.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    CONFORMAL_ALPHA,
    CONFORMAL_WINDOW,
    EXPECTED_SEASON_DAYS,
    INTERIM_DIR,
    MIN_SEASON_COVERAGE,
    PROCESSED_DIR,
)
from src.data.feature_engineering import engineer_features
from src.data.preprocess import build_modeling_table, load_latest_table
from src.models.baseline import create_model
from src.models.conformal import ConformalPredictor

logger = logging.getLogger(__name__)


def run_training_pipeline(
    model_name: str = "xgboost",
    force_refresh: bool = False,
    cal_fraction: float = 0.2,
) -> dict:
    """
    Run the full training pipeline.

    Parameters
    ----------
    model_name : str
        Which baseline model to train ('ridge', 'random_forest', 'xgboost').
    force_refresh : bool
        Force re-download of raw data.
    cal_fraction : float
        Fraction of training data to hold out for conformal calibration.

    Returns
    -------
    dict with keys:
        - 'model': trained BaseYieldModel
        - 'conformal': calibrated ConformalPredictor
        - 'feature_cols': list of feature column names
        - 'df': full merged+engineered DataFrame
        - 'target_col': target column name
    """
    target_col = "yield_bu_per_acre"

    # --- 1. Build modeling table ---
    logger.info("=" * 60)
    logger.info("Step 1: Building modeling table")
    logger.info("=" * 60)
    modeling_df = build_modeling_table(force_refresh=force_refresh)

    # --- 2. Feature engineering ---
    logger.info("=" * 60)
    logger.info("Step 2: Engineering features")
    logger.info("=" * 60)
    features_df = engineer_features()

    # --- 3. Merge features with yield ---
    df = pd.merge(
        modeling_df[["year", target_col]],
        features_df,
        on="year",
        how="inner",
    )

    # Drop crop-years whose weather record is incomplete (e.g. 1948, 1953, 1954 have
    # months of missing observations; their seasonal sums/means are not comparable).
    df = _drop_incomplete_weather_years(df)

    # Drop rows with too many NaN features
    feature_cols = [c for c in features_df.columns if c != "year"]
    df = df.dropna(subset=feature_cols, thresh=int(len(feature_cols) * 0.7))
    df = df.reset_index(drop=True)

    # Fill remaining NaN with column median
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    logger.info("Final dataset: %d years, %d features", len(df), len(feature_cols))

    # --- 4. Sort chronologically ---
    df = df.sort_values("year").reset_index(drop=True)

    # --- 5. Train model + calibrate conformal ---
    logger.info("=" * 60)
    logger.info("Step 3: Training %s + conformal calibration", model_name)
    logger.info("=" * 60)

    X = df[feature_cols].values
    y = df[target_col].values

    # Final model uses ALL labeled years (no calibration hold-out taken from training).
    # Interval width comes from past out-of-sample one-step-ahead residuals.
    model = create_model(model_name)
    conformal = ConformalPredictor(model, alpha=CONFORMAL_ALPHA)
    conformal.fit(X, y)
    conformal.set_residuals(_rolling_origin_residuals(model_name, X, y))
    logger.info(
        "Conformal: %d out-of-sample residuals, half-width=%.2f bu/acre",
        conformal.n_residuals, conformal.halfwidth,
    )

    # --- 6. Save artifacts ---
    artifacts_dir = PROCESSED_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Save full dataset
    df.to_parquet(artifacts_dir / "training_data.parquet", index=False)

    # Save model
    with open(artifacts_dir / f"model_{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save conformal predictor
    with open(artifacts_dir / "conformal_predictor.pkl", "wb") as f:
        pickle.dump(conformal, f)

    # Save feature columns
    with open(artifacts_dir / "feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))

    logger.info("Training pipeline complete — artifacts saved to %s", artifacts_dir)

    return {
        "model": model,
        "conformal": conformal,
        "feature_cols": feature_cols,
        "df": df,
        "target_col": target_col,
    }


def _drop_incomplete_weather_years(df: pd.DataFrame) -> pd.DataFrame:
    """Remove crop-years with < MIN_SEASON_COVERAGE of the expected Oct-Jul days."""
    path = INTERIM_DIR / "daily_weather_with_crop_year.csv"
    if not path.exists():
        return df
    daily = pd.read_csv(path, usecols=["date", "year"])
    counts = daily.groupby("year").size()
    bad = sorted(int(y) for y, n in counts.items()
                 if n < MIN_SEASON_COVERAGE * EXPECTED_SEASON_DAYS)
    bad = [y for y in bad if y in set(df["year"].astype(int))]
    if bad:
        logger.warning("Excluding crop-years with incomplete weather: %s", bad)
    return df[~df["year"].isin(bad)].reset_index(drop=True)


def _rolling_origin_residuals(model_name: str, X: np.ndarray, y: np.ndarray,
                              n_last: int = CONFORMAL_WINDOW) -> np.ndarray:
    """Signed residuals (actual - predicted) of one-step-ahead forecasts for the last
    ``n_last`` years, each from a model trained only on earlier rows."""
    n = len(y)
    start = max(10, n - n_last)
    out = []
    for i in range(start, n):
        m = create_model(model_name)
        m.fit(X[:i], y[:i])
        out.append(float(y[i] - m.predict(X[i:i + 1])[0]))
    return np.array(out)


def load_trained_artifacts() -> dict:
    """Load previously saved training artifacts."""
    artifacts_dir = PROCESSED_DIR / "artifacts"

    df = pd.read_parquet(artifacts_dir / "training_data.parquet")

    with open(artifacts_dir / "feature_cols.txt") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Find the saved model (prefer xgboost)
    model_path = None
    for name in ["xgboost", "random_forest", "ridge"]:
        p = artifacts_dir / f"model_{name}.pkl"
        if p.exists():
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(f"No saved model found in {artifacts_dir}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(artifacts_dir / "conformal_predictor.pkl", "rb") as f:
        conformal = pickle.load(f)

    return {
        "model": model,
        "conformal": conformal,
        "feature_cols": feature_cols,
        "df": df,
        "target_col": "yield_bu_per_acre",
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run the wheat yield training pipeline")
    parser.add_argument(
        "--model", default="xgboost",
        choices=["xgboost", "random_forest", "ridge"],
        help="Which model to train (default: xgboost)",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download raw data even if cached",
    )
    args = parser.parse_args()

    artifacts = run_training_pipeline(
        model_name=args.model,
        force_refresh=args.force_refresh,
    )
    df = artifacts["df"]
    print(f"\n✓ Training complete.")
    print(f"  Dataset : {len(df)} crop-years ({int(df['year'].min())}–{int(df['year'].max())})")
    print(f"  Features: {len(artifacts['feature_cols'])}")
    print(f"  Model   : {args.model}")
    print(f"  Artifacts saved to: {PROCESSED_DIR / 'artifacts'}")
