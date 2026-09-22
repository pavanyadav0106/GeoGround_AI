"""
preprocessing.py
================
GeoGround AI — Phase 1.1: Groundwater Data Preprocessing

Converts the raw NWDP observation CSV into two clean artefacts:
  1. wells.csv       — unique physical monitoring wells with metadata
  2. observations.csv — cleaned time-series of depth readings per well

Key operations:
  - Identify unique physical wells by Station code + Lat/Lon
  - Parse observation dates robustly
  - Derive season, year, month features
  - Compute per-well observation counts and time ranges
  - Save training-ready processed files
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from utils import season_from_month

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# ─── Column map for the NWDP Telangana GW Level CSV ─────────────────────────
RAW_COL_MAP = {
    "Station":                                        "well_name",
    "Agency":                                         "agency",
    "State":                                          "state",
    "District":                                       "district",
    "Tehsil":                                         "tehsil",
    "Block":                                          "block",
    "Village":                                        "village",
    "Latitude":                                       "latitude",
    "Longitude":                                      "longitude",
    "RL_MSL":                                         "elevation_msl",
    "Data Acquisition Time":                          "obs_date_raw",
    "Groundwater Level Quarterly Manual (meter)":     "depth_m",
    "Basin":                                          "basin",
}


# ─── Load & Rename ───────────────────────────────────────────────────────────

def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load all NWDP CSVs from raw directory and apply column renaming."""
    csv_files = sorted(raw_dir.glob("nwdp_gwl_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No NWDP CSV files found in {raw_dir}")

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="cp1252", low_memory=False)

        if len(df) == 0:
            log.warning("Empty file: %s — skipping", f.name)
            continue

        df["_source"] = f.name
        frames.append(df)
        log.info("Loaded %s  (%d rows)", f.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined: %d rows", len(combined))
    return combined


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the column map. Keep only columns we care about."""
    df = df.rename(columns=RAW_COL_MAP)
    keep = list(RAW_COL_MAP.values()) + ["_source"]
    existing_keep = [c for c in keep if c in df.columns]
    return df[existing_keep].copy()


# ─── Cleaning ────────────────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse obs_date_raw into obs_date (datetime), plus year/month/season."""
    df["obs_date"] = pd.to_datetime(
        df["obs_date_raw"], dayfirst=True, errors="coerce"
    )
    df["obs_year"]   = df["obs_date"].dt.year.astype("Int16")
    df["obs_month"]  = df["obs_date"].dt.month.astype("Int8")
    df["obs_season"] = df["obs_month"].apply(
        lambda m: season_from_month(int(m)) if pd.notna(m) else None
    )
    null_dates = df["obs_date"].isna().sum()
    if null_dates:
        log.warning("%d rows have unparseable obs_date — will be dropped", null_dates)
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns; remove physically implausible depth values."""
    df["depth_m"]   = pd.to_numeric(df["depth_m"],   errors="coerce")
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Groundwater depth: 0–200 m BGL is the plausible range for this region
    n_before = len(df)
    invalid_depth = (df["depth_m"] < 0) | (df["depth_m"] > 200)
    df.loc[invalid_depth, "depth_m"] = np.nan
    n_after = df["depth_m"].notna().sum()
    log.info("Valid depth_m rows: %d / %d", n_after, n_before)

    # India bounding box
    outside_india = ~(
        df["latitude"].between(8, 37) &
        df["longitude"].between(68, 98)
    )
    df.loc[outside_india & df["latitude"].notna(), ["latitude", "longitude"]] = np.nan
    return df


def normalise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and title-case string administrative fields."""
    for col in ["district", "tehsil", "block", "village", "agency", "state", "basin"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace({"-": None, "Nan": None, "None": None})
    return df


# ─── Well Table ──────────────────────────────────────────────────────────────

def build_well_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to unique physical monitoring wells.

    A unique well is defined by its Station code (well_name).
    We take the mode/first of administrative fields and the
    mean of lat/lon (should be identical per station — verify with std).

    Returns a DataFrame with one row per unique well.
    """
    agg = (
        df.groupby("well_name")
        .agg(
            latitude       = ("latitude",  "mean"),
            longitude      = ("longitude", "mean"),
            lat_std        = ("latitude",  "std"),
            lon_std        = ("longitude", "std"),
            district       = ("district",  "first"),
            tehsil         = ("tehsil",    "first"),
            block          = ("block",     "first"),
            village        = ("village",   "first"),
            agency         = ("agency",    "first"),
            state          = ("state",     "first"),
            basin          = ("basin",     "first"),
            n_observations = ("depth_m",   "count"),
            first_obs      = ("obs_date",  "min"),
            last_obs       = ("obs_date",  "max"),
            mean_depth_m   = ("depth_m",   "mean"),
            std_depth_m    = ("depth_m",   "std"),
            min_depth_m    = ("depth_m",   "min"),
            max_depth_m    = ("depth_m",   "max"),
        )
        .reset_index()
    )

    # Warn if any well has inconsistent coordinates across rows
    drifters = agg[(agg["lat_std"] > 0.001) | (agg["lon_std"] > 0.001)]
    if len(drifters):
        log.warning(
            "%d wells have inconsistent coordinates (std > 0.001°) — "
            "using mean. Wells: %s",
            len(drifters), list(drifters["well_name"])
        )

    agg = agg.drop(columns=["lat_std", "lon_std"])
    agg["years_coverage"] = (
        (agg["last_obs"] - agg["first_obs"]).dt.days / 365.25
    ).round(1)

    log.info(
        "Well table: %d unique wells | mean obs per well: %.1f | "
        "total valid obs: %d",
        len(agg),
        agg["n_observations"].mean(),
        agg["n_observations"].sum(),
    )
    return agg


# ─── Clean Observations ──────────────────────────────────────────────────────

def build_observation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the clean observation table — one row per (well, date) reading.
    Drop rows with null obs_date or null depth_m.
    """
    obs = df[df["depth_m"].notna() & df["obs_date"].notna()].copy()
    obs = obs.sort_values(["well_name", "obs_date"]).reset_index(drop=True)

    # Keep only essential columns; environmental features will be joined later
    keep_cols = [
        "well_name", "district", "tehsil", "agency",
        "latitude", "longitude",
        "obs_date", "obs_year", "obs_month", "obs_season",
        "depth_m",
    ]
    obs = obs[[c for c in keep_cols if c in obs.columns]]

    log.info(
        "Observation table: %d rows | %d unique wells | date range %s → %s",
        len(obs),
        obs["well_name"].nunique(),
        obs["obs_date"].min().date(),
        obs["obs_date"].max().date(),
    )
    return obs


# ─── Main ────────────────────────────────────────────────────────────────────

def run_preprocessing(
    raw_dir: Path = RAW_DIR,
    proc_dir: Path = PROC_DIR,
    district_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    Args:
        raw_dir: Directory containing NWDP CSVs.
        proc_dir: Output directory for processed files.
        district_filter: If set, filter to this district (e.g. "Hyderabad").
                         If None, use all Telangana data.

    Returns:
        (wells_df, observations_df)
    """
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Load
    raw = load_raw(raw_dir)

    # Rename
    df = rename_columns(raw)

    # Clean
    df = parse_dates(df)
    df = clean_numeric(df)
    df = normalise_strings(df)

    # Optional district filter
    if district_filter:
        variants = [district_filter.title(), district_filter.upper()]
        mask = df["district"].isin(variants)
        if mask.sum() == 0:
            mask = df["district"].str.contains(district_filter, case=False, na=False)
        df = df[mask].copy()
        log.info(
            "District filter '%s': %d rows, %d unique wells",
            district_filter, len(df), df["well_name"].nunique()
        )

    # Build tables
    wells_df = build_well_table(df)
    obs_df   = build_observation_table(df)

    # Save
    suffix = f"_{district_filter.lower()}" if district_filter else "_telangana"
    wells_path = proc_dir / f"wells{suffix}.csv"
    obs_path   = proc_dir / f"observations{suffix}.csv"

    wells_df.to_csv(wells_path, index=False)
    obs_df.to_csv(obs_path, index=False)

    log.info("Saved: %s", wells_path)
    log.info("Saved: %s", obs_path)

    return wells_df, obs_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GeoGround AI — Preprocessing")
    parser.add_argument("--district", default=None,
                        help="Filter to specific district (default: all Telangana)")
    parser.add_argument("--all-telangana", action="store_true",
                        help="Process full Telangana dataset")
    args = parser.parse_args()

    district = None if args.all_telangana else (args.district or None)
    wells, obs = run_preprocessing(district_filter=district)

    print(f"\nWells saved:        {len(wells)} unique monitoring wells")
    print(f"Observations saved: {len(obs)} clean readings")
    print(f"Date range:         {obs['obs_date'].min().date()} → {obs['obs_date'].max().date()}")
