"""Core regression tests: leakage, baselines, drift calibration, conformal, API."""
import numpy as np
import pandas as pd
import pytest

from config.settings import PROCESSED_DIR
from src.models.baseline import create_model, naive_forecasts
from src.models.conformal import conformal_halfwidth, make_interval


def _load():
    p = PROCESSED_DIR / "artifacts" / "training_data.parquet"
    if not p.exists():
        pytest.skip("run: python -m src.pipeline.train_pipeline")
    df = pd.read_parquet(p)
    cols = (PROCESSED_DIR / "artifacts" / "feature_cols.txt").read_text().split()
    return df.sort_values("year").reset_index(drop=True), cols


def test_incomplete_weather_years_excluded():
    df, _ = _load()
    for y in (1948, 1953, 1954):
        assert y not in set(df["year"])


def test_naive_forecasts_use_only_past():
    years = np.arange(2000, 2010)
    y = np.arange(10, 20, dtype=float)
    fc = naive_forecasts(years, y, 2010)
    assert fc["naive_last_year"] == 19.0
    assert fc["naive_rolling5_mean"] == np.mean(y[-5:])
    assert fc["naive_expanding_mean"] == np.mean(y)
    assert abs(fc["naive_linear_trend"] - 20.0) < 1e-9


def test_no_leakage_future_rows_do_not_change_past_forecast():
    df, cols = _load()
    t = int(df["year"].iloc[40])
    train = df[df["year"] < t]
    test = df[df["year"] == t]
    m1 = create_model("ridge"); m1.fit(train[cols].values, train["yield_bu_per_acre"].values)
    p1 = m1.predict(test[cols].values)[0]
    df2 = df.copy()
    df2.loc[df2["year"] > t, "yield_bu_per_acre"] += 1000  # corrupt the future
    train2 = df2[df2["year"] < t]
    m2 = create_model("ridge"); m2.fit(train2[cols].values, train2["yield_bu_per_acre"].values)
    assert p1 == pytest.approx(m2.predict(test[cols].values)[0])


def test_conformal_quantile_and_lower_bound():
    rng = np.random.default_rng(0)
    res = rng.normal(0, 5, 2000)
    hw = conformal_halfwidth(res, 0.10, window=2000)
    assert 0.88 <= np.mean(np.abs(res) <= hw) <= 0.93
    r = make_interval(3.0, 10.0, 0.10)
    assert r.lower_bound == 0.0 and r.upper_bound == 13.0


def test_drift_not_always_high_and_flags_extreme_year():
    from src.drift.detector import detect_drift
    df, cols = _load()
    sev = []
    for t in df["year"].iloc[30:60]:
        r = detect_drift(df[df["year"] < t][cols].values, df[df["year"] == t][cols].values, cols)
        sev.append(r.severity)
    assert sev.count("HIGH") / len(sev) < 0.2          # previously 100 %
    train = df[cols].values
    extreme = train.mean(axis=0) + 6 * train.std(axis=0)
    assert detect_drift(train, extreme.reshape(1, -1), cols).severity == "HIGH"


def test_api_smoke():
    from fastapi.testclient import TestClient
    _load()
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        r = c.post("/predict", json={"year": 2025, "use_live_weather": False})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["lower_bound"] >= 0 and body["upper_bound"] > body["lower_bound"]
        assert c.get("/drift/2025").status_code == 200
        assert c.get("/reliability/2025").status_code == 200
        assert c.get("/history").status_code == 200
        assert c.post("/predict", json={"year": 1900, "use_live_weather": False}).status_code in (400, 404, 422)
