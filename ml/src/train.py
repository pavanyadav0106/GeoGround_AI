"""
train.py
========
GeoGround AI — Phase 3: ML Model Training

Trains and evaluates groundwater depth regression models:
  - Baseline (mean predictor)
  - Random Forest
  - XGBoost  ← primary model
  - LightGBM ← optional comparison

Validation strategy:
  TEMPORAL split — train on observations before 2017, validate on 2017+
  This simulates "predict future conditions at monitored wells" rather than
  a simple random split, which would leak temporal patterns.

  For spatial validation (leave-one-well-out), see --spatial-cv flag.

Outputs:
  - models/best_model.ubj            ← XGBoost binary format
  - models/preprocessor.pkl          ← sklearn ColumnTransformer
  - models/feature_importance.csv
  - models/eval_report.json
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# Windows consoles can still default to a legacy code page.  Training emits a
# human-readable report, so force UTF-8 where the interpreter supports it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR    = Path(__file__).resolve().parent
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PROC_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed"
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Feature Definitions ─────────────────────────────────────────────────────

# Numerical features used in training
NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "obs_year",
    "obs_month",
    "elevation_m",
    "rainfall_mm",
    "temp_c",
    "humidity_pct",
    "n_nearby_wells",
    "nearest_well_km",
    "avg_nearby_depth_m",
    "min_nearby_depth_m",
    "max_nearby_depth_m",
    "std_nearby_depth_m",
    "mean_depth_m",
    "std_depth_m",
    "min_depth_m",
    "max_depth_m",
    "trend_slope_m_yr",
    # Soil (null for Hyderabad urban — imputed with median)
    "clay_pct",
    "sand_pct",
    "silt_pct",
    "soil_ph",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "obs_season",       # Pre-Monsoon / Kharif / Post-Monsoon / Rabi
    "lulc_label",       # Built_Up / Cropland / Water_Body / etc.
]

# Binary flags
BINARY_FEATURES = [
    "lulc_is_built_up",
    "lulc_is_cropland",
    "lulc_is_water",
]

TARGET = "depth_m"

# Temporal split year — train on data BEFORE this year
TRAIN_BEFORE_YEAR = 2017


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_training_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded training data: %d rows × %d cols from %s", len(df), len(df.columns), path.name)

    # Ensure obs_date is datetime
    if "obs_date" in df.columns:
        df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")

    # Keep only rows with a valid target
    n_before = len(df)
    df = df[df[TARGET].notna()].copy()
    log.info("Rows with valid %s: %d / %d", TARGET, len(df), n_before)

    return df


def temporal_split(df: pd.DataFrame, split_year: int = TRAIN_BEFORE_YEAR):
    """
    Split into train/test by observation year.

    Train: obs_year < split_year
    Test:  obs_year >= split_year

    This is the recommended split for time-series groundwater data to avoid
    temporal leakage (future data bleeding into past predictions).
    """
    train = df[df["obs_year"] < split_year].copy()
    test  = df[df["obs_year"] >= split_year].copy()

    log.info(
        "Temporal split (year < %d): train=%d rows, test=%d rows",
        split_year, len(train), len(test)
    )
    if len(test) == 0:
        log.warning(
            "Test set is empty (all data before %d). Using last 20%% for testing.",
            split_year
        )
        n_test = max(1, int(len(df) * 0.2))
        df = df.sort_values("obs_year")
        train = df.iloc[:-n_test].copy()
        test  = df.iloc[-n_test:].copy()

    return train, test


def leave_one_well_out_cv(df: pd.DataFrame, model_factory, preprocessor):
    """
    Spatial cross-validation: leave one well out at a time.
    Returns average MAE across all wells.
    """
    wells = df["well_name"].unique()
    maes  = []

    for well in wells:
        train = df[df["well_name"] != well]
        test  = df[df["well_name"] == well]

        if len(train) < 10 or len(test) < 3:
            continue

        X_train = preprocessor.fit_transform(train)
        X_test  = preprocessor.transform(test)
        y_train = train[TARGET].values
        y_test  = test[TARGET].values

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))

    avg_mae = np.mean(maes)
    log.info("Leave-one-well-out CV: MAE = %.3f m (n_folds=%d)", avg_mae, len(maes))
    return avg_mae, maes


# ─── Preprocessing Pipeline ─────────────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that handles:
    - Numerical features: impute median → standard scale
    - Categorical features: impute 'missing' → ordinal encode
    - Binary features: impute 0 → pass through
    """
    # Only include features that actually exist in the DataFrame
    num_feats  = [f for f in NUMERIC_FEATURES     if f in df.columns]
    cat_feats  = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    bin_feats  = [f for f in BINARY_FEATURES       if f in df.columns]

    log.info("Features — numeric:%d  categorical:%d  binary:%d",
             len(num_feats), len(cat_feats), len(bin_feats))

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    transformers = []
    if num_feats:
        transformers.append(("num", numeric_pipeline, num_feats))
    if cat_feats:
        transformers.append(("cat", categorical_pipeline, cat_feats))
    if bin_feats:
        transformers.append(("bin", binary_pipeline, bin_feats))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Store feature names for importance mapping later
    preprocessor._feature_names_in = num_feats + cat_feats + bin_feats
    return preprocessor, num_feats + cat_feats + bin_feats


# ─── Metrics ─────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)

    print(f"  {name:20s}  MAE={mae:.3f}m  RMSE={rmse:.3f}m  R²={r2:.4f}")
    return {"model": name, "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


# ─── Training ────────────────────────────────────────────────────────────────

def train_all(
    df: pd.DataFrame,
    split_year: int = TRAIN_BEFORE_YEAR,
    spatial_cv: bool = False,
) -> dict:
    """Train all models and return the best one."""
    import xgboost as xgb
    import lightgbm as lgb

    train_df, test_df = temporal_split(df, split_year)

    preprocessor, all_features = build_preprocessor(df)

    X_train_raw = train_df[all_features] if all_features else train_df
    X_test_raw  = test_df[all_features]  if all_features else test_df
    y_train     = train_df[TARGET].values
    y_test      = test_df[TARGET].values

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    log.info(
        "X_train: %s, X_test: %s, y_train mean=%.2f, y_test mean=%.2f",
        X_train.shape, X_test.shape, y_train.mean(), y_test.mean()
    )

    results = []
    trained_models = {}

    print("\n" + "─" * 60)
    print(f"  EVALUATION (temporal split — test year >= {split_year})")
    print("─" * 60)

    # ── Baseline ──────────────────────────────────────────────
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    preds_base = baseline.predict(X_test)
    results.append(evaluate("Baseline (mean)", y_test, preds_base))

    # ── Random Forest ─────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, preds_rf))
    trained_models["RandomForest"] = rf

    # ── XGBoost ───────────────────────────────────────────────
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=30,
        eval_metric="rmse",
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    preds_xgb = xgb_model.predict(X_test)
    results.append(evaluate("XGBoost", y_test, preds_xgb))
    trained_models["XGBoost"] = xgb_model

    # ── LightGBM ──────────────────────────────────────────────
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    preds_lgb = lgb_model.predict(X_test)
    results.append(evaluate("LightGBM", y_test, preds_lgb))
    trained_models["LightGBM"] = lgb_model

    # ── Pick best ─────────────────────────────────────────────
    best_result = min(results[1:], key=lambda r: r["mae"])  # exclude baseline
    best_name   = best_result["model"]
    best_model  = trained_models[best_name]
    print(f"\n  BEST MODEL: {best_name}  (MAE={best_result['mae']:.3f} m)")

    # ── Feature Importance ────────────────────────────────────
    feature_names = all_features
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        imp_df = (
            pd.DataFrame({"feature": feature_names[:len(imp)], "importance": imp})
            .sort_values("importance", ascending=False)
        )
        imp_path = MODELS_DIR / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        log.info("Feature importance saved: %s", imp_path)

        print("\n  TOP 10 FEATURES:")
        for _, row in imp_df.head(10).iterrows():
            bar = "█" * int(row["importance"] * 40)
            print(f"  {row['feature']:35s} {row['importance']:.4f}  {bar}")

    # ── Save model & preprocessor ─────────────────────────────
    if best_name == "XGBoost":
        model_path = MODELS_DIR / "best_model.ubj"
        best_model.save_model(str(model_path))
        log.info("Saved XGBoost model: %s", model_path)
    else:
        model_path = MODELS_DIR / "best_model.pkl"
        joblib.dump(best_model, model_path)
        log.info("Saved model: %s", model_path)

    prep_path = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, prep_path)
    log.info("Saved preprocessor: %s", prep_path)

    # Save metadata
    meta = {
        "best_model":      best_name,
        "model_path":      str(model_path),
        "preprocessor":    str(prep_path),
        "features":        feature_names,
        "target":          TARGET,
        "split_year":      split_year,
        "train_rows":      int(len(train_df)),
        "test_rows":       int(len(test_df)),
        "results":         results,
        "best_result":     best_result,
    }
    eval_path = MODELS_DIR / "eval_report.json"
    eval_path.write_text(json.dumps(meta, indent=2))
    log.info("Evaluation report saved: %s", eval_path)

    # ── Optional spatial CV ───────────────────────────────────
    if spatial_cv and "well_name" in df.columns:
        print("\n  Running leave-one-well-out spatial CV on XGBoost...")
        preprocessor_cv, _ = build_preprocessor(df)
        def xgb_factory():
            return xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=42, verbosity=0, n_jobs=-1
            )
        avg_mae, _ = leave_one_well_out_cv(df, xgb_factory, preprocessor_cv)
        meta["spatial_cv_mae"] = round(avg_mae, 4)
        eval_path.write_text(json.dumps(meta, indent=2))
        print(f"  Spatial CV MAE: {avg_mae:.3f} m")

    return meta


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoGround AI — ML Training")
    parser.add_argument(
        "--data", default=str(PROC_DIR / "training_data_hyderabad.csv"),
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--split-year", type=int, default=TRAIN_BEFORE_YEAR,
        help=f"Temporal split year (train < year, test >= year). Default: {TRAIN_BEFORE_YEAR}"
    )
    parser.add_argument(
        "--spatial-cv", action="store_true",
        help="Run leave-one-well-out spatial cross-validation (slow)"
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Training data not found: %s", data_path)
        log.error("Run: python scripts/build_training_data.py first")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  GEOGROUND AI — PHASE 3: ML TRAINING")
    print("═" * 60)

    df   = load_training_data(data_path)
    meta = train_all(df, split_year=args.split_year, spatial_cv=args.spatial_cv)

    print("\n" + "═" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Best model : {meta['best_model']}")
    print(f"  Test MAE   : {meta['best_result']['mae']} m")
    print(f"  Test RMSE  : {meta['best_result']['rmse']} m")
    print(f"  Test R²    : {meta['best_result']['r2']}")
    print(f"  Model file : {meta['model_path']}")
    print("═" * 60 + "\n")
    print("  Next: python scripts/build_training_data.py  (full Telangana)")
    print("  Then: python scripts/train_model.py --data data/processed/training_data_telangana.csv")


if __name__ == "__main__":
    main()
