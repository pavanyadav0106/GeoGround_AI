"""
utils.py
========
GeoGround AI — Shared utility functions for the ML pipeline.
"""

import math
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance in kilometres between two points on Earth.
    Uses the Haversine formula.

    Args:
        lat1, lon1: First point (decimal degrees, WGS84)
        lat2, lon2: Second point (decimal degrees, WGS84)

    Returns:
        Distance in kilometres.
    """
    R = 6371.0  # Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def find_nearby_wells(
    query_lat: float,
    query_lon: float,
    wells_df: pd.DataFrame,
    radius_km: float = 15.0,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    exclude_well_id: Optional[str] = None,
    well_id_col: str = "well_id",
) -> pd.DataFrame:
    """
    Return wells within `radius_km` of the query point, sorted by distance.

    This function is used BOTH for:
      1. Prediction — finding nearby wells for a user query location.
      2. Feature engineering — finding wells near a TRAINING well
         (where the training well itself MUST be excluded via exclude_well_id
         to prevent data leakage).

    Args:
        query_lat, query_lon: Query coordinates.
        wells_df: DataFrame containing well locations.
        radius_km: Search radius.
        lat_col, lon_col: Column names for coordinates.
        exclude_well_id: ID of the well to exclude (mandatory when building
                         training features to prevent data leakage).
        well_id_col: Column name for well identifier.

    Returns:
        Filtered DataFrame with 'distance_km' column added, sorted ascending.
    """
    if wells_df.empty:
        return wells_df

    df = wells_df.dropna(subset=[lat_col, lon_col]).copy()

    # Exclude the target well itself — CRITICAL for data leakage prevention
    if exclude_well_id is not None and well_id_col in df.columns:
        df = df[df[well_id_col] != exclude_well_id]

    if df.empty:
        return df

    df["distance_km"] = df.apply(
        lambda row: haversine_km(query_lat, query_lon, row[lat_col], row[lon_col]),
        axis=1,
    )

    nearby = df[df["distance_km"] <= radius_km].copy()
    nearby = nearby.sort_values("distance_km").reset_index(drop=True)

    return nearby


def compute_groundwater_trend(
    observations: pd.Series,
    dates: pd.Series,
) -> Tuple[str, float]:
    """
    Compute groundwater level trend from a time-series of depth observations.

    Trend is determined by linear regression slope over the observation period.

    Classification thresholds (project-defined):
        slope > +0.3 m/year  → "Decreasing" (depth increasing = water table falling)
        slope < -0.3 m/year  → "Increasing" (depth decreasing = water table rising)
        else                 → "Stable"

    Note: depth_m is measured as depth BELOW ground level.
    An increasing depth number means the water table is FALLING.
    We display trend in terms of groundwater AVAILABILITY, so:
      - slope > 0 (depth growing) → Decreasing availability
      - slope < 0 (depth shrinking) → Increasing (recovering)

    Args:
        observations: Series of depth_m values.
        dates: Corresponding pandas datetime Series.

    Returns:
        Tuple of (trend_label: str, slope_m_per_year: float)
    """
    df = pd.DataFrame({"depth": observations, "date": dates}).dropna()

    if len(df) < 3:
        return "Stable", 0.0

    df = df.sort_values("date")
    # Convert date to numeric (decimal years since epoch)
    df["t"] = (df["date"] - pd.Timestamp("1970-01-01")).dt.days / 365.25

    x = df["t"].values
    y = df["depth"].values

    if len(x) < 2 or np.std(x) == 0:
        return "Stable", 0.0

    slope = float(np.polyfit(x, y, 1)[0])  # m per year

    STABLE_THRESHOLD = 0.3  # m/year

    if slope > STABLE_THRESHOLD:
        label = "Decreasing"    # water table falling
    elif slope < -STABLE_THRESHOLD:
        label = "Increasing"    # water table recovering
    else:
        label = "Stable"

    return label, round(slope, 3)


def compute_groundwater_health_index(
    depth_m: Optional[float],
    trend_slope_m_yr: float = 0.0,
    confidence_pct: int = 100,
    n_wells: int = 1,
) -> Tuple[Optional[int], str, str]:
    """
    Compute the Groundwater Health & Availability Index (0 - 100%)
    based on depth below ground level and aquifer depletion trend.

    Scoring Scale:
      • 80 - 100%: Safe / Abundant Water Table (0 - 8m BGL)
      • 60 - 79% : Good / Stable Availability (8 - 15m BGL)
      • 35 - 59% : Semi-Critical / Moderate Stress (15 - 22m BGL)
      • 15 - 34% : Critical / Over-Exploited Red Zone (22 - 32m BGL)
      • 0 - 14%  : Severe Depletion / Deep Aquifer Crisis (> 32m BGL)

    Returns:
        Tuple of (health_score_pct, health_status_label, health_description)
    """
    if n_wells == 0 or confidence_pct <= 0 or depth_m is None or pd.isna(depth_m):
        return None, "Unmonitored", "Location is outside calibrated monitoring coverage."

    # Base score: maps 0m -> 100%, 35m+ -> 0%
    base_score = max(0.0, 100.0 - (depth_m / 35.0 * 100.0))

    # Trend adjustment: falling water table reduces health index; rising improves it
    if trend_slope_m_yr > 0.3:
        trend_penalty = min(15.0, trend_slope_m_yr * 10.0)
        base_score = max(0.0, base_score - trend_penalty)
    elif trend_slope_m_yr < -0.3:
        trend_bonus = min(10.0, abs(trend_slope_m_yr) * 8.0)
        base_score = min(100.0, base_score + trend_bonus)

    health_score = int(round(np.clip(base_score, 0, 100)))

    if health_score >= 80:
        status = "Safe & Abundant"
        desc = f"Water table is shallow ({depth_m:.1f}m BGL) with healthy recharge potential."
    elif health_score >= 60:
        status = "Good / Sustainable"
        desc = f"Moderate water level ({depth_m:.1f}m BGL) under sustainable utilization."
    elif health_score >= 35:
        status = "Semi-Critical / Stressed"
        desc = f"Significant drawdown ({depth_m:.1f}m BGL). Conservation recommended."
    elif health_score >= 15:
        status = "Critical / Over-Exploited"
        desc = f"Severe groundwater depletion ({depth_m:.1f}m BGL). Active over-extraction red zone."
    else:
        status = "Extreme Crisis"
        desc = f"Critical aquifer failure ({depth_m:.1f}m BGL). Water table severely exhausted."

    return health_score, status, desc


def classify_groundwater_condition(
    depth_m: float,
    confidence_pct: int = 100,
    n_wells: int = 1,
) -> str:
    """
    Classify estimated groundwater depth into hydrogeological category.
    """
    if n_wells == 0 or confidence_pct <= 0:
        return "Uncertain (No Data)"
    if pd.isna(depth_m) or depth_m < 0:
        return "Unknown"
    if depth_m <= 8:
        return "Excellent"
    if depth_m <= 15:
        return "Good"
    if depth_m <= 22:
        return "Moderate"
    if depth_m <= 30:
        return "Critical"
    return "Severe Depletion"


def compute_confidence_score(
    n_wells: int,
    nearest_distance_km: float,
    max_radius_km: float = 15.0,
    rainfall_mm: Optional[float] = None,
    soil_data: Optional[dict] = None,
    lulc_data: Optional[dict] = None,
    std_depth_m: Optional[float] = None,
) -> Tuple[int, str]:
    """
    Compute a multi-source confidence score for a groundwater prediction based on:
      1. Observation Wells (40%): Spatial proximity, monitoring density & local water table consistency.
      2. Weather / Rainfall (20%): NASA POWER rainfall & climate telemetry availability & plausibility.
      3. Soil Hydrogeology (20%): ISRIC SoilGrids texture balance (clay, sand, silt) & soil pH.
      4. Land Cover Type (20%): ESA WorldCover land use infiltration characteristics.

    Returns:
        Tuple of (confidence_pct: int, explanation: str)
    """
    if n_wells == 0:
        return 0, f"No monitoring wells found within {max_radius_km:.0f} km radius. Location is outside the calibrated monitoring network."

    # 1. Observation Well Score (Weight: 40%)
    # - Proximity (0 to 1): closer nearest well gives higher confidence
    proximity_score = max(0.0, 1.0 - (nearest_distance_km / max_radius_km))
    # - Density (0 to 1): scales up to 8 wells
    density_score = min(n_wells / 8.0, 1.0)
    # - Local Aquifer Consistency (0 to 1): lower variance among neighboring wells = more predictable aquifer
    if std_depth_m is not None and not np.isnan(std_depth_m) and n_wells >= 2:
        if std_depth_m <= 2.5:
            consistency_score = 1.0
        elif std_depth_m <= 6.0:
            consistency_score = 0.85
        elif std_depth_m <= 12.0:
            consistency_score = 0.70
        else:
            consistency_score = 0.55  # High local drawdown variance
    else:
        consistency_score = 0.80

    wells_score = 0.45 * proximity_score + 0.35 * density_score + 0.20 * consistency_score

    # 2. Weather & Rainfall Score (Weight: 20%)
    if rainfall_mm is not None and not np.isnan(rainfall_mm):
        if 0.0 <= rainfall_mm <= 600.0:
            weather_score = 0.95
        else:
            weather_score = 0.70
        weather_note = f"Rainfall: {rainfall_mm:.1f} mm"
    else:
        weather_score = 0.30
        weather_note = "Rainfall telemetry unavailable"

    # 3. Soil Composition & Texture Score (Weight: 20%)
    soil = soil_data or {}
    clay = soil.get("clay_pct")
    sand = soil.get("sand_pct")
    silt = soil.get("silt_pct")
    ph   = soil.get("soil_ph")

    has_soil_texture = all(v is not None and not np.isnan(v) for v in (clay, sand, silt))
    if has_soil_texture:
        # Check realistic texture sum
        tex_sum = float(clay) + float(sand) + float(silt)
        if 85.0 <= tex_sum <= 115.0:
            soil_score = 0.95
        else:
            soil_score = 0.80
        soil_note = f"Soil: Clay {clay:.0f}%, Sand {sand:.0f}%, Silt {silt:.0f}%"
    elif any(v is not None and not np.isnan(v) for v in (clay, sand, silt, ph)):
        soil_score = 0.60
        soil_note = "Partial soil telemetry"
    else:
        soil_score = 0.25
        soil_note = "Soil data unavailable"

    # 4. Land Cover Type (LULC) Score (Weight: 20%)
    lulc = lulc_data or {}
    lulc_label = lulc.get("lulc_label") or lulc.get("label") or "Unknown"
    lulc_code  = lulc.get("lulc_code") or lulc.get("code")

    # Infiltration / Recharge predictability based on ESA WorldCover
    if lulc_code == 40 or "crop" in str(lulc_label).lower():
        lulc_score = 0.95  # Cropland / Agricultural recharge
        lulc_desc = f"LULC: {lulc_label} (High Infiltration)"
    elif lulc_code in (10, 20, 30) or any(k in str(lulc_label).lower() for k in ("tree", "shrub", "grass")):
        lulc_score = 0.90  # Natural vegetation / Forest / Grassland
        lulc_desc = f"LULC: {lulc_label} (Natural Recharge)"
    elif lulc_code == 80 or "water" in str(lulc_label).lower():
        lulc_score = 0.85  # Open water surface
        lulc_desc = f"LULC: {lulc_label} (Surface Water Proximity)"
    elif lulc_code == 50 or "built" in str(lulc_label).lower() or "urban" in str(lulc_label).lower():
        lulc_score = 0.70  # Urban / Impermeable concrete surfaces limit direct infiltration
        lulc_desc = f"LULC: {lulc_label} (Impervious Surface)"
    elif lulc_code == 60 or "bare" in str(lulc_label).lower() or "barren" in str(lulc_label).lower():
        lulc_score = 0.80
        lulc_desc = f"LULC: {lulc_label} (Barren / Rocky)"
    else:
        lulc_score = 0.35
        lulc_desc = f"LULC: {lulc_label}"

    # Combined Multi-Factor Confidence
    raw_confidence = (
        0.40 * wells_score +
        0.20 * weather_score +
        0.20 * soil_score +
        0.20 * lulc_score
    )
    confidence_pct = int(round(np.clip(raw_confidence * 100, 10, 99)))

    explanation = (
        f"Multi-Factor Confidence: {confidence_pct}% | "
        f"{n_wells} nearby wells (closest {nearest_distance_km:.1f} km) • "
        f"{weather_note} • {soil_note} • {lulc_desc}"
    )

    return confidence_pct, explanation


def season_from_month(month: int) -> str:
    """Return Indian hydrological season name for a calendar month (1–12)."""
    if month in (3, 4, 5):
        return "Pre-Monsoon"
    if month in (6, 7, 8, 9):
        return "Kharif"
    if month in (10, 11, 12):
        return "Post-Monsoon"
    return "Rabi"
