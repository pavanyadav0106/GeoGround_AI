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
    compute_groundwater_health_index,
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

    meta_path = models_dir / "eval_report.json"
    eval_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    target_path = eval_meta.get("model_path")
    if target_path and Path(target_path).exists():
        model_path = Path(target_path)
    elif (models_dir / "best_model.pkl").exists():
        model_path = models_dir / "best_model.pkl"
    elif (models_dir / "best_model.ubj").exists():
        model_path = models_dir / "best_model.ubj"
    else:
        raise FileNotFoundError(
            f"No trained model found in {models_dir}. "
            "Run: python scripts/train_model.py"
        )

    if model_path.suffix == ".ubj":
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
    else:
        model = joblib.load(model_path)

    preprocessor = joblib.load(models_dir / "preprocessor.pkl")

    _model_cache        = model
    _preprocessor_cache = preprocessor
    _eval_meta_cache    = eval_meta

    log.info("Loaded model: %s (%s)", model_path.name, eval_meta.get("best_model", "Custom"))
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
            monthly = climate.get(key)
            # If current month/year is beyond NASA archive (e.g. 2024-2026),
            # retrieve the most recent recorded year for the same calendar month
            if not monthly or monthly.get("rainfall_mm") is None:
                month_suffix = f"-{obs_month:02d}"
                same_month_entries = [
                    v for k, v in sorted(climate.items())
                    if k.endswith(month_suffix) and v.get("rainfall_mm") is not None
                ]
                if same_month_entries:
                    monthly = same_month_entries[-1]
                else:
                    rain_vals = [v.get("rainfall_mm") for v in climate.values() if v.get("rainfall_mm") is not None]
                    temp_vals = [v.get("temp_c") for v in climate.values() if v.get("temp_c") is not None]
                    hum_vals  = [v.get("humidity_pct") for v in climate.values() if v.get("humidity_pct") is not None]
                    monthly = {
                        "rainfall_mm": round(float(np.mean(rain_vals)), 2) if rain_vals else None,
                        "temp_c": round(float(np.mean(temp_vals)), 2) if temp_vals else None,
                        "humidity_pct": round(float(np.mean(hum_vals)), 2) if hum_vals else None,
                    }

            if monthly:
                features["rainfall_mm"]  = monthly.get("rainfall_mm")
                features["temp_c"]       = monthly.get("temp_c")
                features["humidity_pct"] = monthly.get("humidity_pct")
            else:
                features.update({"rainfall_mm": None, "temp_c": None, "humidity_pct": None})
        else:
            features.update({"rainfall_mm": None, "temp_c": None, "humidity_pct": None})
    except Exception as e:
        log.warning("NASA POWER fetch failed: %s", e)
        features.update({"rainfall_mm": None, "temp_c": None, "humidity_pct": None})

    # ── Soil (SoilGrids with Regional Geological Fallback) ───
    try:
        soil = fetch_soilgrids(lat, lon) or {}
        clay = soil.get("clay_pct")
        sand = soil.get("sand_pct")
        silt = soil.get("silt_pct")
        ph   = soil.get("soil_ph")

        # If urban paved centroid or API masked, use regional Telangana geological soil profile
        if clay is None or sand is None:
            # Telangana Soil Classification:
            # Northern/Godavari basin (Adilabad/Nizamabad): Black Cotton Soil (Vertisol)
            # Central/Southern (Hyderabad/Rangareddy/Medak/Nalgonda): Red Sandy Loam (Alfisol)
            if lat > 18.5:
                clay, sand, silt, ph = 42.0, 28.0, 30.0, 7.6  # Black Cotton / Vertisol
            elif lon > 79.5:
                clay, sand, silt, ph = 32.0, 43.0, 25.0, 7.2  # Mixed Loam / Alluvial
            else:
                clay, sand, silt, ph = 24.0, 56.0, 20.0, 6.8  # Red Sandy Loam (Alfisol)

        features["clay_pct"] = clay
        features["sand_pct"] = sand
        features["silt_pct"] = silt
        features["soil_ph"]  = ph
    except Exception as e:
        log.warning("Soil profile resolution failed: %s", e)
        features.update({"clay_pct": 26.0, "sand_pct": 52.0,
                         "silt_pct": 22.0, "soil_ph": 6.8})

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
            # Look for valid observations (filtering out 0.0 non-physical sensor nulls)
            nearby_names = set(nearby["well_name"])
            matched_obs = obs[obs["well_name"].isin(nearby_names)].copy() if not obs.empty else pd.DataFrame()
            
            if not matched_obs.empty:
                # Filter out 0.0 / non-physical sensor zeros if positive measurements exist
                positive_obs = matched_obs[matched_obs["depth_m"] > 0.1]
                if not positive_obs.empty:
                    matched_obs = positive_obs

                # Prioritize the most recent observations per well (latest 5 readings)
                recent_obs = matched_obs.sort_values("obs_date").groupby("well_name").tail(5)
            else:
                recent_obs = pd.DataFrame()

            # Per-nearby-well info and distance-weighted aggregation
            well_depth_weights = []
            for _, w in nearby.iterrows():
                w_recent = recent_obs[recent_obs["well_name"] == w["well_name"]] if not recent_obs.empty else pd.DataFrame()
                if w_recent.empty and not matched_obs.empty:
                    w_recent = matched_obs[matched_obs["well_name"] == w["well_name"]].sort_values("obs_date").tail(3)
                
                # Compute average of latest valid readings for this specific well
                valid_w_depths = w_recent["depth_m"].dropna() if not w_recent.empty else pd.Series()
                if len(valid_w_depths) > 0:
                    recent_depth = float(valid_w_depths.mean())
                elif "mean_depth_m" in w and pd.notna(w["mean_depth_m"]) and w["mean_depth_m"] > 0.1:
                    recent_depth = float(w["mean_depth_m"])
                else:
                    recent_depth = None

                dist = float(w["distance_km"])
                if recent_depth is not None and recent_depth > 0:
                    # Inverse Distance Weighting: closer wells have exponentially higher influence
                    weight = 1.0 / ((dist + 0.15) ** 2)
                    well_depth_weights.append((recent_depth, weight))

                if len(nearby_wells_info) < 10:
                    nearby_wells_info.append({
                        "well_name":    w["well_name"],
                        "distance_km":  round(dist, 2),
                        "recent_depth_m": round(recent_depth, 2) if recent_depth is not None else None,
                        "latitude":     float(w["latitude"]),
                        "longitude":    float(w["longitude"]),
                    })

            # Distance-weighted depth estimation & robust statistics
            if well_depth_weights:
                depths_arr  = np.array([d for d, _ in well_depth_weights])
                weights_arr = np.array([wt for _, wt in well_depth_weights])
                idw_avg_d   = round(float(np.sum(depths_arr * weights_arr) / np.sum(weights_arr)), 3)
                min_d       = round(float(np.min(depths_arr)), 3)
                max_d       = round(float(np.max(depths_arr)), 3)
                std_d       = round(float(np.std(depths_arr)), 3)
            else:
                all_recent_depths = recent_obs["depth_m"].dropna().tolist() if not recent_obs.empty else []
                idw_avg_d = round(float(np.mean(all_recent_depths)), 3) if all_recent_depths else None
                min_d     = round(float(np.min(all_recent_depths)), 3) if all_recent_depths else None
                max_d     = round(float(np.max(all_recent_depths)), 3) if all_recent_depths else None
                std_d     = round(float(np.std(all_recent_depths)), 3) if all_recent_depths else None

            features["n_nearby_wells"]     = len(nearby)
            features["nearest_well_km"]    = round(float(nearby["distance_km"].min()), 3)
            features["avg_nearby_depth_m"] = idw_avg_d
            features["min_nearby_depth_m"] = min_d
            features["max_nearby_depth_m"] = max_d
            features["std_nearby_depth_m"] = std_d
            features["mean_depth_m"]       = idw_avg_d
            features["min_depth_m"]        = min_d
            features["max_depth_m"]        = max_d
            features["std_depth_m"]        = std_d

            # Trend from nearby wells
            if not recent_obs.empty and len(recent_obs["depth_m"].dropna()) >= 3:
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
                "mean_depth_m": None, "min_depth_m": None, "max_depth_m": None, "std_depth_m": None,
                "trend_slope_m_yr": 0.0,
            })
            trend_label = "Stable"
    else:
        features.update({
            "n_nearby_wells": 0, "nearest_well_km": None,
            "avg_nearby_depth_m": None, "min_nearby_depth_m": None,
            "max_nearby_depth_m": None, "std_nearby_depth_m": None,
            "mean_depth_m": None, "min_depth_m": None, "max_depth_m": None, "std_depth_m": None,
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

    # Confidence based on Wells + Rainfall + Soil + Land Type (LULC)
    n_wells     = features.get("n_nearby_wells", 0) or 0
    nearest_km  = features.get("nearest_well_km") or radius_km
    rainfall_mm = features.get("rainfall_mm")
    soil_data   = {
        "clay_pct": features.get("clay_pct"),
        "sand_pct": features.get("sand_pct"),
        "silt_pct": features.get("silt_pct"),
        "soil_ph":  features.get("soil_ph"),
    }
    lulc_data   = {
        "lulc_label": features.get("lulc_label"),
        "lulc_code":  features.get("lulc_code"),
    }
    std_depth_m = features.get("std_nearby_depth_m")

    confidence, confidence_explanation = compute_confidence_score(
        n_wells=n_wells,
        nearest_distance_km=nearest_km,
        max_radius_km=radius_km,
        rainfall_mm=rainfall_mm,
        soil_data=soil_data,
        lulc_data=lulc_data,
        std_depth_m=std_depth_m,
    )

    # Predict (only for points within calibrated regional range)
    if confidence > 0 and n_wells > 0:
        predicted_depth = float(model.predict(X)[0])
        predicted_depth = max(0.0, round(predicted_depth, 2))
    else:
        predicted_depth = None

    # Classify condition based on depth and confidence
    condition   = classify_groundwater_condition(
        predicted_depth, confidence_pct=confidence, n_wells=n_wells
    )

    # Compute Groundwater Health & Availability Index (0 - 100%)
    health_score, health_status, health_desc = compute_groundwater_health_index(
        depth_m=predicted_depth,
        trend_slope_m_yr=features.get("trend_slope_m_yr", 0.0),
        confidence_pct=confidence,
        n_wells=n_wells,
    )

    if confidence <= 0:
        trend_label = "Unmonitored"

    return {
        "latitude":                 lat,
        "longitude":                lon,
        "estimated_depth_m":        predicted_depth,
        "groundwater_health_score": health_score,
        "health_status":            health_status,
        "health_description":       health_desc,
        "condition":                condition,
        "trend":                    trend_label,
        "confidence":               confidence,
        "confidence_note":          confidence_explanation,
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
