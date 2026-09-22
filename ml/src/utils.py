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


def classify_groundwater_condition(
    depth_m: float,
    confidence_pct: int = 100,
    n_wells: int = 1,
) -> str:
    """
    Classify estimated groundwater depth into a condition category.

    These thresholds are project-defined interpretation categories.
    They are NOT official CGWB government standards.

    If no physical monitoring wells exist within the search radius (or confidence is 0%),
    returns 'Uncertain (No Data)' so that uncalibrated out-of-region predictions
    are not falsely marked as 'Excellent'.

    | Depth (m BGL) | Condition               |
    |---------------|-------------------------|
    | (0 wells)     | Uncertain (No Data)     |
    | 0 – 10        | Excellent               |
    | 10 – 20       | Good                    |
    | 20 – 30       | Moderate                |
    | > 30          | Poor                    |

    Args:
        depth_m: Estimated groundwater depth in metres below ground level.
        confidence_pct: Calculated confidence percentage.
        n_wells: Number of nearby physical monitoring wells.

    Returns:
        Condition label as string.
    """
    if n_wells == 0 or confidence_pct <= 0:
        return "Uncertain (No Data)"
    if pd.isna(depth_m) or depth_m < 0:
        return "Unknown"
    if depth_m <= 10:
        return "Excellent"
    if depth_m <= 20:
        return "Good"
    if depth_m <= 30:
        return "Moderate"
    return "Poor"


def compute_confidence_score(
    n_wells: int,
    avg_distance_km: float,
    max_radius_km: float = 15.0,
) -> Tuple[int, str]:
    """
    Compute a confidence score for a prediction based on nearby well count
    and average distance.

    Formula (project-defined — NOT a statistically calibrated probability):
        well_score     = min(n_wells / 10, 1.0)          # maxes at 10 wells
        distance_score = 1 - (avg_distance_km / max_radius_km)
        raw_confidence = 0.6 * well_score + 0.4 * distance_score

    Args:
        n_wells: Number of monitoring wells found within radius.
        avg_distance_km: Average distance of those wells from the query point.
        max_radius_km: Search radius used (for normalisation).

    Returns:
        Tuple of (confidence_pct: int, explanation: str)
    """
    if n_wells == 0:
        return 0, f"No monitoring wells found within {max_radius_km:.0f} km radius. Location is outside the calibrated monitoring network."

    well_score     = min(n_wells / 10.0, 1.0)
    distance_score = max(0.0, 1.0 - (avg_distance_km / max_radius_km))

    raw = 0.6 * well_score + 0.4 * distance_score
    confidence_pct = int(round(raw * 100))

    explanation = (
        f"Based on {n_wells} nearby monitoring well{'s' if n_wells != 1 else ''} "
        f"(avg. distance: {avg_distance_km:.1f} km). "
        f"More nearby wells = higher confidence."
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
