"""
predict.py
==========
GeoGround AI — Phase 4: Inference Pipeline

Loads the trained XGBoost model and preprocessor, then produces a
structured prediction for a given (lat, lon) query point.

This module is consumed by the FastAPI ML service.

Usage (standalone):
    python src/predict.py --lat 17.385 --lon 78.486
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SRC_DIR    = Path(__file__).resolve().parent
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PROC_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed"

sys.path.insert(0, str(SRC_DIR))

from utils import (
    haversine_km,
    classify_groundwater_condition,
    compute_groundwater_trend,
    compute_confidence_score,
    season_from_month,
)
from feature_engineering import (
    fetch_nasa_power,
    fetch_soilgrids,
    fetch_worldcover_point,
    fetch_elevation_batch,
)

log = logging.getLogger(__name__)


# ─── Model Loading ───────────────────────────────────────────────────────────

_model_cache       = None
_preprocessor_cache = None
_eval_meta_cache    = None
_wells_cache        = None
_obs_cache          = None


def load_model(models_dir: Path = MODELS_DIR):
    """Load the trained model (cached after first call)."""
    global _model_cache, _preprocessor_cache, _eval_meta_cache

    if _model_cache is not None:
        return _model_cache, _preprocessor_cache, _eval_meta_cache

    import joblib

    # Find model file (prefer XGBoost .ubj, fall back to .pkl)
    model_path = models_dir / "best_model.ubj"
    if model_path.exists():
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
    else:
        model_path = models_dir / "best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found in {models_dir}. "
                "Run: python scripts/train_model.py"
            )
        model = joblib.load(model_path)

    preprocessor = joblib.load(models_dir / "preprocessor.pkl")

    meta_path = models_dir / "eval_report.json"
    eval_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    _model_cache        = model
    _preprocessor_cache = preprocessor
    _eval_meta_cache    = eval_meta

    log.info("Loaded model: %s", model_path.name)
    return model, preprocessor, eval_meta


def load_wells_and_observations(proc_dir: Path = PROC_DIR):
    """Load monitoring well data for nearby-well lookups (cached)."""
    global _wells_cache, _obs_cache

    if _wells_cache is not None:
        return _wells_cache, _obs_cache

    # Prefer Telangana (larger coverage) but fall back to Hyderabad
    wells_path = proc_dir / "wells_telangana.csv"
    obs_path   = proc_dir / "observations_telangana.csv"

    if not wells_path.exists():
        wells_path = proc_dir / "wells_hyderabad.csv"
        obs_path   = proc_dir / "observations_hyderabad.csv"

    if not wells_path.exists():
        log.warning("No wells file found — nearby-well features will be unavailable")
        return pd.DataFrame(), pd.DataFrame()

    wells = pd.read_csv(wells_path, low_memory=False)
    obs   = pd.read_csv(obs_path, low_memory=False, parse_dates=["obs_date"])

    _wells_cache = wells
    _obs_cache   = obs

    log.info("Loaded %d wells, %d observations", len(wells), len(obs))
    return wells, obs


# ─── Feature Collection for Query Point ─────────────────────────────────────

def collect_features_for_query(
    lat: float,
    lon: float,
    radius_km: float = 15.0,
    query_date: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Collect all features for a query (lat, lon) point.

    Returns a flat dict ready to be passed to the ML preprocessor.
    """
    import datetime as dt

    if query_date is None:
        query_date = pd.Timestamp.now()

    obs_year  = query_date.year
    obs_month = query_date.month
    obs_season = season_from_month(obs_month)

    features = {
        "latitude":   lat,
        "longitude":  lon,
        "obs_year":   obs_year,
        "obs_month":  obs_month,
        "obs_season": obs_season,
    }

    # ── Elevation ────────────────────────────────────────────
    try:
        elevs = fetch_elevation_batch([(lat, lon)])
        features["elevation_m"] = elevs[0] if elevs else None
    except Exception as e:
        log.warning("Elevation fetch failed: %s", e)
        features["elevation_m"] = None

    # ── NASA POWER weather ───────────────────────────────────
    try:
        climate = fetch_nasa_power(lat, lon, well_name=f"query_{lat:.4f}_{lon:.4f}")
        if climate:
            key = f"{obs_year:04d}-{obs_month:02d}"
            monthly = climate.get(key, {})
            features["rainfall_mm"]  = monthly.get("rainfall_mm")
            features["temp_c"]       = monthly.get("temp_c")
            features["humidity_pct"] = monthly.get("humidity_pct")
        else:
            features.update({"rainfall_mm": None, "temp_c": None, "humidity_pct": None})
    except Exception as e:
        log.warning("NASA POWER fetch failed: %s", e)
        features.update({"rainfall_mm": None, "temp_c": None, "humidity_pct": None})

    # ── Soil ─────────────────────────────────────────────────
    try:
        soil = fetch_soilgrids(lat, lon) or {}
        features["clay_pct"]  = soil.get("clay_pct")
        features["sand_pct"]  = soil.get("sand_pct")
        features["silt_pct"]  = soil.get("silt_pct")
        features["soil_ph"]   = soil.get("soil_ph")
    except Exception as e:
        log.warning("SoilGrids fetch failed: %s", e)
        features.update({"clay_pct": None, "sand_pct": None,
                         "silt_pct": None, "soil_ph": None})

    # ── LULC ─────────────────────────────────────────────────
    try:
        lulc = fetch_worldcover_point(lat, lon) or {}
        features["lulc_label"]      = lulc.get("lulc_label")
        features["lulc_code"]       = lulc.get("lulc_code")
        features["lulc_is_built_up"] = 1 if lulc.get("lulc_code") == 50 else 0
        features["lulc_is_cropland"] = 1 if lulc.get("lulc_code") == 40 else 0
        features["lulc_is_water"]    = 1 if lulc.get("lulc_code") == 80 else 0
    except Exception as e:
        log.warning("WorldCover fetch failed: %s", e)
        features.update({"lulc_label": None, "lulc_code": None,
                         "lulc_is_built_up": 0, "lulc_is_cropland": 0, "lulc_is_water": 0})

    # ── Nearby Wells ──────────────────────────────────────────
    wells, obs = load_wells_and_observations()
    nearby_wells_info = []

    if not wells.empty:
        valid_wells = wells.dropna(subset=["latitude", "longitude"])
        valid_wells = valid_wells.copy()
        valid_wells["distance_km"] = valid_wells.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
        )
        nearby = valid_wells[valid_wells["distance_km"] <= radius_km].copy()
        nearby = nearby.sort_values("distance_km")

        if len(nearby) > 0:
            # Get recent depth readings (last 2 years)
            cutoff_year = obs_year - 2
            recent_obs = obs[
                (obs["well_name"].isin(nearby["well_name"])) &
                (obs["obs_date"].dt.year >= cutoff_year)
            ] if not obs.empty else pd.DataFrame()

            # Per-nearby-well info for the response
            for _, w in nearby.head(10).iterrows():
                w_recent = recent_obs[recent_obs["well_name"] == w["well_name"]] if not recent_obs.empty else pd.DataFrame()
                recent_depth = float(w_recent["depth_m"].mean()) if len(w_recent) > 0 else None
                nearby_wells_info.append({
                    "well_name":    w["well_name"],
                    "distance_km":  round(float(w["distance_km"]), 2),
                    "recent_depth_m": round(recent_depth, 2) if recent_depth is not None else None,
                    "latitude":     float(w["latitude"]),
                    "longitude":    float(w["longitude"]),
                })

            # Aggregate features
            all_recent_depths = recent_obs["depth_m"].dropna().tolist() if not recent_obs.empty else []

            features["n_nearby_wells"]     = len(nearby)
            features["nearest_well_km"]    = round(float(nearby["distance_km"].min()), 3)
            features["avg_nearby_depth_m"] = round(float(np.mean(all_recent_depths)), 3) if all_recent_depths else None
            features["min_nearby_depth_m"] = round(float(np.min(all_recent_depths)), 3) if all_recent_depths else None
            features["max_nearby_depth_m"] = round(float(np.max(all_recent_depths)), 3) if all_recent_depths else None
            features["std_nearby_depth_m"] = round(float(np.std(all_recent_depths)), 3) if all_recent_depths else None

            # Trend from nearby wells
            if len(all_recent_depths) >= 3 and not recent_obs.empty:
                trend_label, trend_slope = compute_groundwater_trend(
                    recent_obs["depth_m"], recent_obs["obs_date"]
                )
            else:
                trend_label, trend_slope = "Stable", 0.0

            features["trend_slope_m_yr"] = trend_slope
        else:
            features.update({
                "n_nearby_wells": 0, "nearest_well_km": None,
                "avg_nearby_depth_m": None, "min_nearby_depth_m": None,
                "max_nearby_depth_m": None, "std_nearby_depth_m": None,
                "trend_slope_m_yr": 0.0,
            })
            trend_label = "Stable"
    else:
        features.update({
            "n_nearby_wells": 0, "nearest_well_km": None,
            "avg_nearby_depth_m": None, "min_nearby_depth_m": None,
            "max_nearby_depth_m": None, "std_nearby_depth_m": None,
            "trend_slope_m_yr": 0.0,
        })
        trend_label = "Stable"

    return features, trend_label, nearby_wells_info


# ─── Main Prediction Function ─────────────────────────────────────────────────

def predict(
    lat: float,
    lon: float,
    radius_km: float = 15.0,
    query_date: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Produce a structured groundwater condition prediction for (lat, lon).

    Returns:
        dict with keys:
            latitude, longitude,
            estimated_depth_m, condition, trend, confidence,
            nearby_wells, weather, feature_snapshot, disclaimer
    """
    model, preprocessor, eval_meta = load_model()

    # Collect features
    features, trend_label, nearby_wells_info = collect_features_for_query(
        lat, lon, radius_km=radius_km, query_date=query_date
    )

    # Build single-row DataFrame matching preprocessor expectations
    feature_df = pd.DataFrame([features])

    # Preprocess
    X = preprocessor.transform(feature_df)

    # Predict
    predicted_depth = float(model.predict(X)[0])
    predicted_depth = max(0.0, round(predicted_depth, 2))

    # Classify
    condition   = classify_groundwater_condition(predicted_depth)

    # Confidence
    n_wells     = features.get("n_nearby_wells", 0) or 0
    nearest_km  = features.get("nearest_well_km") or radius_km
    confidence, confidence_explanation = compute_confidence_score(
        n_wells, nearest_km, max_radius_km=radius_km
    )

    return {
        "latitude":           lat,
        "longitude":          lon,
        "estimated_depth_m":  predicted_depth,
        "condition":          condition,
        "trend":              trend_label,
        "confidence":         confidence,
        "confidence_note":    confidence_explanation,
        "nearby_wells":       nearby_wells_info,
        "weather": {
            "rainfall_mm":  features.get("rainfall_mm"),
            "temp_c":       features.get("temp_c"),
            "humidity_pct": features.get("humidity_pct"),
        },
        "soil": {
            "clay_pct":  features.get("clay_pct"),
            "sand_pct":  features.get("sand_pct"),
            "silt_pct":  features.get("silt_pct"),
            "soil_ph":   features.get("soil_ph"),
        },
        "elevation_m":        features.get("elevation_m"),
        "lulc_label":         features.get("lulc_label"),
        "model_info": {
            "model_type":   eval_meta.get("best_model", "Unknown"),
            "train_mae_m":  eval_meta.get("best_result", {}).get("mae"),
            "train_r2":     eval_meta.get("best_result", {}).get("r2"),
            "n_wells_used": int(n_wells),
        },
        "disclaimer": (
            "This is an estimate based on nearby monitoring wells and environmental data. "
            "It does not replace a professional hydrogeological survey. "
            "Depth thresholds (Excellent/Good/Moderate/Poor) are project-defined "
            "interpretation categories, not official CGWB standards."
        ),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GeoGround AI — Predict groundwater depth")
    parser.add_argument("--lat",  type=float, required=True,  help="Latitude")
    parser.add_argument("--lon",  type=float, required=True,  help="Longitude")
    parser.add_argument("--radius", type=float, default=15.0, help="Search radius km")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = predict(args.lat, args.lon, radius_km=args.radius)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
