"""
apply_fixes.py - one-time patch for the wheat-yield project.
Run from the project root (folder containing pyproject.toml):   python apply_fixes.py
Backs up every file it edits to backups_before_fix/ and aborts without changes
if any expected code block is not found (so it can never half-apply).
"""
import pathlib, re, shutil, sys

ROOT = pathlib.Path(".").resolve()
BACKUP = ROOT / "backups_before_fix"
if BACKUP.exists():
    sys.exit("backups_before_fix/ already exists - patch seems already applied. Aborting.")

EDITS = {}  # path -> list of (old, new)
def edit(path, old, new):
    EDITS.setdefault(path, []).append((old, new))

# ------------------------------------------------------------------ config/settings.py
edit("config/settings.py", "CV_STEP_YEARS = 5               # Step size between test-year cutoffs",
     "CV_STEP_YEARS = 1               # Annual rolling-origin evaluation (one test year per cutoff)")
edit("config/settings.py", "CONFORMAL_ALPHA = 0.10          # 90% prediction interval (1 - alpha)",
'''CONFORMAL_ALPHA = 0.10          # 90% prediction interval (1 - alpha)
CONFORMAL_WINDOW = 20           # Most recent out-of-sample residuals used for calibration
CONFORMAL_MIN_RESIDUALS = 10    # Fewer residuals than this -> interval flagged as unreliable
MIN_SEASON_COVERAGE = 0.95      # Crop-years with less weather coverage are excluded
EXPECTED_SEASON_DAYS = 304      # Oct 1 - Jul 31 (non-leap)
MIN_DRIFT_REF_YEARS = 15        # Minimum reference years for drift detection''')
edit("config/settings.py", "KS_ALPHA = 0.05                 # p-value threshold for KS test",
'''KS_ALPHA = 0.05                 # p-value threshold for KS test
DRIFT_P_HIGH = 0.025            # Empirical p-value <= this -> HIGH drift
DRIFT_P_MODERATE = 0.10         # Empirical p-value <= this -> MODERATE drift
DRIFT_Z_FEATURE = 3.0           # |z| above this flags an individual feature
MIN_BATCH_FOR_PSI_KS = 8        # KS/PSI only meaningful with at least this many test rows''')

# ------------------------------------------------------------------ src/models/baseline.py
edit("src/models/baseline.py", "from sklearn.linear_model import Ridge\n",
     "from sklearn.linear_model import Ridge, RidgeCV\n")
edit("src/models/baseline.py",
"# ---------------------------------------------------------------------------\n# Factory",
'''# ---------------------------------------------------------------------------
# Detrended Ridge (trend + weather-driven residual)
# ---------------------------------------------------------------------------

class DetrendedRidgeModel(BaseYieldModel):
    """
    Linear time trend + RidgeCV on the trend residuals.

    IMPORTANT: the LAST column of X must be the crop year.  Callers use
    ``feature_cols + ["year"]`` for this model.  Yields have a technology trend
    (corr(year, yield) ~ 0.6 here); modelling it explicitly lets the weather
    features explain only the remaining year-to-year variation.
    """

    name = "detrended_ridge"

    def __init__(self, alphas: tuple = (10.0, 100.0, 1000.0)):
        self.alphas = tuple(alphas)
        self._coef = (0.0, 0.0)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=self.alphas)),
        ])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        years = X[:, -1]
        self._coef = tuple(np.polyfit(years, y, 1))
        residual = y - np.polyval(self._coef, years)
        self._pipeline.fit(X[:, :-1], residual)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        trend = np.polyval(self._coef, X[:, -1])
        return trend + self._pipeline.predict(X[:, :-1])

    def get_params(self) -> dict[str, Any]:
        return {"model": self.name, "alphas": self.alphas}


# ---------------------------------------------------------------------------
# Naive baselines (computed from the yield history only)
# ---------------------------------------------------------------------------

def naive_forecasts(years: np.ndarray, y: np.ndarray, test_year: int) -> dict[str, float]:
    """
    Naive one-step-ahead forecasts for ``test_year`` using ONLY past yields.

    Every model must be compared against these; a model that cannot beat
    them has no demonstrated skill.
    """
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    return {
        "naive_expanding_mean": float(np.mean(y)),
        "naive_rolling5_mean": float(np.mean(y[-5:])),
        "naive_rolling10_mean": float(np.mean(y[-10:])),
        "naive_last_year": float(y[-1]),
        "naive_linear_trend": float(np.polyval(np.polyfit(years, y, 1), test_year)),
    }


# ---------------------------------------------------------------------------
# Factory''')
edit("src/models/baseline.py", '    "xgboost": XGBoostModel,\n}',
     '    "xgboost": XGBoostModel,\n    "detrended_ridge": DetrendedRidgeModel,\n}')

# ------------------------------------------------------------------ src/models/validation.py
edit("src/models/validation.py", '    for name in ALL_MODELS:\n        logger.info("=" * 60)',
     '    for name in ALL_MODELS:\n        if name == "detrended_ridge":\n            continue  # needs a \'year\' column; evaluated in experiments/run_all.py\n        logger.info("=" * 60)')

# ------------------------------------------------------------------ src/drift/univariate.py
edit("src/drift/univariate.py", "from config.settings import KS_ALPHA, PSI_HIGH, PSI_MODERATE",
     "from config.settings import (\n    DRIFT_Z_FEATURE,\n    KS_ALPHA,\n    MIN_BATCH_FOR_PSI_KS,\n    PSI_HIGH,\n    PSI_MODERATE,\n)")
edit("src/drift/univariate.py", '''    results = []
    for i in range(n_features):
        train_col = X_train[:, i]
        test_col = X_test[:, i]
''', '''    # KS and PSI compare DISTRIBUTIONS and are meaningless for a single test year
    # (PSI from one observation is ~8-10 for ANY value).  For small test samples use a
    # per-feature z-score against the training distribution instead.  The result fields
    # are reused: ks_statistic = |z|, ks_pvalue = two-sided normal p-value of z,
    # ks_drifted = |z| > DRIFT_Z_FEATURE, PSI is not computed (0.0 / LOW).
    if X_test.shape[0] < MIN_BATCH_FOR_PSI_KS:
        from scipy.stats import norm
        results = []
        for i in range(n_features):
            tr = X_train[:, i]
            tr = tr[~np.isnan(tr)]
            te = X_test[:, i]
            te = te[~np.isnan(te)]
            if len(tr) < 3 or len(te) == 0 or np.std(tr, ddof=1) == 0:
                z, p = 0.0, 1.0
            else:
                z = abs(float((te.mean() - tr.mean()) / np.std(tr, ddof=1)))
                p = float(2 * (1 - norm.cdf(z)))
            results.append(UnivariateDriftResult(
                feature=feature_names[i], ks_statistic=z, ks_pvalue=p,
                ks_drifted=z > DRIFT_Z_FEATURE, psi_score=0.0, psi_severity="LOW",
            ))
        return results

    results = []
    for i in range(n_features):
        train_col = X_train[:, i]
        test_col = X_test[:, i]
''')

# ------------------------------------------------------------------ src/drift/multivariate.py
edit("src/drift/multivariate.py", '''from config.settings import (
    MAHALANOBIS_HIGH,
    MAHALANOBIS_MODERATE,
    PCA_VARIANCE_RETAINED,
)''', '''from sklearn.covariance import LedoitWolf

from config.settings import (
    DRIFT_P_HIGH,
    DRIFT_P_MODERATE,
    MIN_DRIFT_REF_YEARS,
    PCA_VARIANCE_RETAINED,
)''')
edit("src/drift/multivariate.py", '''    mmd_score: float | None  # None if not enough test samples
''', '''    mmd_score: float | None  # None if not enough test samples
    mahalanobis_pvalue: float | None = None  # empirical p-value vs leave-one-out null
''')
edit("src/drift/multivariate.py", '''    maha_dist = _mahalanobis_distance(X_train, X_test)
    maha_sev = _mahalanobis_severity(maha_dist)
''', '''    maha_dist, maha_p = _calibrated_mahalanobis(X_train, X_test)
    maha_sev = _pvalue_severity(maha_p)
''')
edit("src/drift/multivariate.py", '''        mmd_score=mmd,
    )
''', '''        mmd_score=mmd,
        mahalanobis_pvalue=maha_p,
    )
''')
edit("src/drift/multivariate.py", '''def _mahalanobis_severity(dist: float) -> str:
    """Map Mahalanobis distance to severity label."""
    if dist >= MAHALANOBIS_HIGH:
        return "HIGH"
    elif dist >= MAHALANOBIS_MODERATE:
        return "MODERATE"
    else:
        return "LOW"
''', '''def _shrunk_mahalanobis(ref: np.ndarray, point: np.ndarray) -> float:
    """Mahalanobis distance with a Ledoit-Wolf shrunk covariance on standardized data."""
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0) + 1e-9
    z_ref = (ref - mu) / sd
    z = (point - mu) / sd
    prec = LedoitWolf().fit(z_ref).precision_
    return float(np.sqrt(max(z @ prec @ z, 0.0)))


def _calibrated_mahalanobis(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[float, float | None]:
    """
    Shrinkage Mahalanobis distance of the test centroid, plus an EMPIRICAL p-value.

    The raw distance has no fixed scale with 40 features and ~25-75 samples (the old
    fixed thresholds 3/5 were always exceeded).  Instead, the distance is compared with
    a leave-one-year-out null: each training year's distance to the rest of the
    training years.  p = (1 + #{null >= d}) / (n + 1).  Returns p=None if there are
    fewer than MIN_DRIFT_REF_YEARS reference years (cannot calibrate).
    """
    train_means = np.nanmean(X_train, axis=0)
    X_tr = np.where(np.isnan(X_train), train_means, X_train)
    X_te = np.where(np.isnan(X_test), train_means, X_test)

    dist = _shrunk_mahalanobis(X_tr, X_te.mean(axis=0))
    n = X_tr.shape[0]
    if n < MIN_DRIFT_REF_YEARS:
        return dist, None

    null = np.array([
        _shrunk_mahalanobis(np.delete(X_tr, i, axis=0), X_tr[i]) for i in range(n)
    ])
    p_value = float((1 + np.sum(null >= dist)) / (n + 1))
    return dist, p_value


def _pvalue_severity(p_value: float | None) -> str:
    """Map an empirical drift p-value to a severity label."""
    if p_value is None:
        return "LOW"  # not enough reference years to judge
    if p_value <= DRIFT_P_HIGH:
        return "HIGH"
    if p_value <= DRIFT_P_MODERATE:
        return "MODERATE"
    return "LOW"
''')

# ------------------------------------------------------------------ src/drift/detector.py  (replace function to EOF)
EDITS.setdefault("src/drift/detector.py", []).append(("__REPLACE_FROM__def _combine_scores(", '''def _combine_scores(
    uni_results: list[UnivariateDriftResult],
    multi_result: MultivariateDriftResult,
) -> tuple[str, float, list[str]]:
    """
    Severity comes from the CALIBRATED multivariate p-value (empirical, leave-one-year-out).

    score = 1 - p_value (0..1; higher = more unusual).  Severity: p <= 0.025 HIGH,
    p <= 0.10 MODERATE, otherwise LOW.  The previous composite (PSI from one point,
    uncalibrated Mahalanobis, saturated PCA percentile) was always ~0.75 = HIGH.
    Contributing variables are the features with the largest |z| above the threshold.
    """
    p_value = multi_result.mahalanobis_pvalue
    severity = multi_result.mahalanobis_severity
    score = 0.0 if p_value is None else float(np.clip(1.0 - p_value, 0.0, 1.0))

    flagged = [r for r in uni_results if r.ks_drifted]
    flagged.sort(key=lambda r: r.ks_statistic, reverse=True)
    contributors = [r.feature for r in flagged][:5]

    return severity, score, contributors
'''))

# ------------------------------------------------------------------ src/adaptation/windowed_retrain.py (replace block between markers)
EDITS.setdefault("src/adaptation/windowed_retrain.py", []).append((
    "__REPLACE_BETWEEN__    if window_sizes is None:\n        window_sizes = RETRAIN_WINDOW_SIZES|||    # Final retrain with best window size", '''    if window_sizes is None:
        window_sizes = RETRAIN_WINDOW_SIZES

    # Inner validation over the last N_INNER past years (past-only), not a single year.
    # A single-year choice is mostly noise.
    N_INNER = 8
    past_years = sorted(y for y in df["year"].unique() if y < test_year)
    inner_years = past_years[-N_INNER:]

    best_size = window_sizes[0]
    best_mae = float("inf")
    best_model = None

    for ws in window_sizes:
        errors = []
        for inner_year in inner_years:
            test_row = df[df["year"] == inner_year]
            if len(test_row) == 0:
                continue
            inner_model = windowed_retrain(
                model, df, feature_cols, target_col,
                test_year=int(inner_year), window_size=ws,
            )
            y_true = test_row[target_col].values[0]
            y_pred = inner_model.predict(test_row[feature_cols].values)[0]
            errors.append(abs(y_true - y_pred))
        if not errors:
            continue
        mae = float(np.mean(errors))
        logger.debug("  Window %d -> inner MAE=%.2f over %d years", ws, mae, len(errors))
        if mae < best_mae:
            best_mae = mae
            best_size = ws

'''))

# ------------------------------------------------------------------ src/adaptation/strategy.py
S = "src/adaptation/strategy.py"
edit(S, '''    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
''', '''    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
''')
edit(S, '''            "year": self.year,
            "drift_severity"''', '''            "model": self.model,
            "year": self.year,
            "drift_severity"''')
edit(S, "round(self.adapted_prediction, 2) if self.adapted_prediction else None",
        "round(self.adapted_prediction, 2) if self.adapted_prediction is not None else None")
edit(S, '''    X_test: np.ndarray,
    force: bool = False,
) -> tuple[ConformalResult, AdaptationEvent]:''', '''    X_test: np.ndarray,
    force: bool = False,
    recent_residuals: np.ndarray | None = None,
    model_name: str = "",
) -> tuple[ConformalResult, AdaptationEvent]:''')
edit(S, '''        strategy, result = _try_bias_correction(
            model, original, df, feature_cols, target_col, test_year
        )
    else:  # HIGH''', '''        strategy, result = _try_bias_correction(
            model, original, df, feature_cols, target_col, test_year, recent_residuals
        )
    else:  # HIGH''')
edit(S, '''            strategy, result = _try_bias_correction(
                model, original, df, feature_cols, target_col, test_year
            )''', '''            strategy, result = _try_bias_correction(
                model, original, df, feature_cols, target_col, test_year, recent_residuals
            )''')
edit(S, "        _adaptation_log.append(event)\n        return original, event",
        "        event.model = model_name\n        _adaptation_log.append(event)\n        return original, event")
edit(S, "    _adaptation_log.append(event)\n\n    if adapted:",
        "    event.model = model_name\n    _adaptation_log.append(event)\n\n    if adapted:")
EDITS[S].append(("__REPLACE_BETWEEN__def _try_bias_correction(|||def _try_windowed_retrain(", '''def _try_bias_correction(
    model: BaseYieldModel,
    original: ConformalResult,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_year: int,
    recent_residuals: np.ndarray | None = None,
) -> tuple[str, ConformalResult | None]:
    """
    Bias correction from recent OUT-OF-SAMPLE residuals (actual - predicted).

    The model is trained on all years < test_year, so re-predicting those years would
    give in-sample residuals (~0, meaningless).  Callers must pass the past
    out-of-sample residuals; without them no correction is applied.  The mean is
    shrunk toward 0 (n / (n + 2)) because a handful of residuals is a noisy estimate.
    """
    if recent_residuals is None or len(recent_residuals) == 0:
        return "bias_correction", None

    res = np.asarray(recent_residuals, dtype=float)[-BIAS_CORRECTION_WINDOW:]
    bias = float(np.mean(res)) * len(res) / (len(res) + 2.0)

    corrected_point, corrected_lower, corrected_upper = apply_bias_correction(
        original.point_prediction,
        original.lower_bound,
        original.upper_bound,
        bias,
    )

    return "bias_correction", ConformalResult(
        point_prediction=corrected_point,
        lower_bound=max(0.0, corrected_lower),
        upper_bound=corrected_upper,
        interval_width=corrected_upper - max(0.0, corrected_lower),
        alpha=original.alpha,
    )


'''))

# ------------------------------------------------------------------ src/pipeline/train_pipeline.py
T = "src/pipeline/train_pipeline.py"
edit(T, "from config.settings import PROCESSED_DIR\n", '''from config.settings import (
    CONFORMAL_ALPHA,
    CONFORMAL_WINDOW,
    EXPECTED_SEASON_DAYS,
    INTERIM_DIR,
    MIN_SEASON_COVERAGE,
    PROCESSED_DIR,
)
''')
edit(T, "    # Drop rows with too many NaN features\n", '''    # Drop crop-years whose weather record is incomplete (e.g. 1948, 1953, 1954 have
    # months of missing observations; their seasonal sums/means are not comparable).
    df = _drop_incomplete_weather_years(df)

    # Drop rows with too many NaN features
''')
edit(T, '''    model = create_model(model_name)
    conformal = ConformalPredictor(model)
    conformal.fit_and_calibrate(X, y, cal_fraction=cal_fraction)
''', '''    # Final model uses ALL labeled years (no calibration hold-out taken from training).
    # Interval width comes from past out-of-sample one-step-ahead residuals.
    model = create_model(model_name)
    conformal = ConformalPredictor(model, alpha=CONFORMAL_ALPHA)
    conformal.fit(X, y)
    conformal.set_residuals(_rolling_origin_residuals(model_name, X, y))
    logger.info(
        "Conformal: %d out-of-sample residuals, half-width=%.2f bu/acre",
        conformal.n_residuals, conformal.halfwidth,
    )
''')
edit(T, "def load_trained_artifacts() -> dict:", '''def _drop_incomplete_weather_years(df: pd.DataFrame) -> pd.DataFrame:
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


def load_trained_artifacts() -> dict:''')

# ------------------------------------------------------------------ src/pipeline/inference_pipeline.py
I = "src/pipeline/inference_pipeline.py"
edit(I, '''    X_train = train_df[feature_cols].values
    drift_result = detect_drift(X_train, X_test, feature_names=feature_cols)
''', '''    # Reference distribution = years BEFORE the requested year (never include the test year)
    ref = train_df[train_df["year"] < year]
    if len(ref) < 15:
        ref = train_df
    X_train = ref[feature_cols].values
    drift_result = detect_drift(X_train, X_test, feature_names=feature_cols)
''')
edit(I, '''        X_test=X_test,
        force=force_adapt,
    )''', '''        X_test=X_test,
        force=force_adapt,
        recent_residuals=conformal.signed_residuals,
        model_name=getattr(model, "name", ""),
    )''')

# ------------------------------------------------------------------ api/routes (blocking work must not run in async handlers)
for f, old, new in [
    ("api/routes/adapt.py", "async def trigger_adaptation(", "def trigger_adaptation("),
    ("api/routes/drift.py", "async def get_drift(", "def get_drift("),
    ("api/routes/history.py", "async def get_history(", "def get_history("),
    ("api/routes/predict.py", "async def predict(", "def predict("),
    ("api/routes/reliability.py", "async def get_reliability(", "def get_reliability("),
]:
    edit(f, old, new)

# ------------------------------------------------------------------ pyproject.toml
edit("pyproject.toml", '    "mapie>=0.6",\n', "")
edit("pyproject.toml", 'requires-python = ">=3.9"', 'requires-python = ">=3.11"')

# ================================================================== apply (all-or-nothing)
def apply(text, old, new, path):
    if old.startswith("__REPLACE_FROM__"):
        marker = old[len("__REPLACE_FROM__"):]
        i = text.find(marker)
        if i < 0: raise SystemExit(f"[{path}] block not found: {marker[:60]}")
        return text[:i] + new
    if old.startswith("__REPLACE_BETWEEN__"):
        a, b = old[len("__REPLACE_BETWEEN__"):].split("|||")
        i, j = text.find(a), text.find(b)
        if i < 0 or j < 0 or j < i: raise SystemExit(f"[{path}] block not found: {a[:60]}")
        return text[:i] + new + text[j:]
    if text.count(old) < 1: raise SystemExit(f"[{path}] block not found: {old[:60]!r}")
    return text.replace(old, new, 1)

new_texts = {}
for path, edits in EDITS.items():
    p = ROOT / path
    if not p.exists(): raise SystemExit(f"Missing file: {path} (run from project root)")
    t = p.read_text(encoding="utf-8")
    for old, new in edits:
        t = apply(t, old, new, path)
    new_texts[path] = t

for path, t in new_texts.items():
    dst = BACKUP / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / path, dst)
    (ROOT / path).write_text(t, encoding="utf-8", newline="\n")
print(f"Patched {len(new_texts)} files. Originals saved in {BACKUP.name}/")
print("NEXT: replace src/models/conformal.py and experiments/run_all.py, add tests/, requirements.txt")
