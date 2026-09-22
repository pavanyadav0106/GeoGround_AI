"""
data_loader.py
==============
GeoGround AI — Phase 0: NWDP Groundwater Data Loader & Validator

Loads the CGWB/NWDP "Ground Water Level (Manual – Quarterly)" CSV
for Telangana and performs validation checks required for the
Phase 0 GO/NO-GO feasibility decision.

Usage:
    python src/data_loader.py --input data/raw/ --district "Hyderabad"
    python src/data_loader.py --validate          # Run full Phase 0 report
"""

import argparse
import os
import sys
import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

RAW_DATA_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Minimum thresholds for the GO/NO-GO decision
MIN_WELLS_WITH_COORDS = 15
MIN_YEARS_COVERAGE    = 5
MIN_TOTAL_ROWS        = 200
MAX_MISSING_RATE      = 0.30   # 30%

# Known column name variants in NWDP exports
# The portal has changed column headings across different download years.
COLUMN_ALIASES = {
    "well_id":       ["station_code", "wid", "id", "well_id", "station_id",
                      "nhns_code", "slno", "SlNo"],
    "well_name":     ["station_name", "well_name", "name", "obs_well_name",
                      "Station", "station"],  # NWDP Telangana uses 'Station'
    "district":      ["district", "district_name", "dist_name", "District"],
    "mandal":        ["mandal", "block", "block_name", "taluk",
                      "Tehsil", "tehsil"],    # NWDP uses 'Tehsil' for sub-district
    "village":       ["village", "village_name", "Village"],
    "latitude":      ["latitude", "lat", "y", "Latitude"],
    "longitude":     ["longitude", "lon", "long", "x", "Longitude"],
    "obs_date":      ["observation_date", "obs_date", "date", "year_month",
                      "data_date", "measure_date",
                      "Data Acquisition Time",  # NWDP Telangana exact column name
                      "data_acquisition_time"],
    "depth_m":       ["water_level_m", "depth_m", "gwl_m", "water_level",
                      "depth_below_gl", "wl_below_gl", "gwl", "gwlevel",
                      "Groundwater Level Quarterly Manual (meter)",  # NWDP Telangana exact
                      "groundwater_level_quarterly_manual__meter_"],
    "well_type":     ["well_type", "type_of_well", "source_type"],
    # Extra useful columns in this dataset
    "agency":        ["Agency", "agency"],
    "state":         ["State", "state"],
    "basin":         ["Basin", "basin"],
}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _normalise_col(name: str) -> str:
    """Lowercase + strip + remove special chars for fuzzy column matching."""
    import re
    return re.sub(r"[^a-z0-9_]", "_", name.strip().lower())


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap raw NWDP column names to canonical internal names using COLUMN_ALIASES.
    Unrecognised columns are kept as-is with a warning.
    """
    norm_to_raw = {_normalise_col(c): c for c in df.columns}
    rename_map = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalised_alias = _normalise_col(alias)
            if normalised_alias in norm_to_raw:
                raw_col = norm_to_raw[normalised_alias]
                if raw_col != canonical:
                    rename_map[raw_col] = canonical
                break  # first match wins

    if rename_map:
        log.info("Column remap: %s", rename_map)
        df = df.rename(columns=rename_map)

    # Report unmapped important columns
    missing = [c for c in COLUMN_ALIASES if c not in df.columns]
    if missing:
        log.warning("Missing canonical columns after remap: %s", missing)

    return df


def load_csv_files(directory: Path, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load all CSV files matching `pattern` in `directory` into a single DataFrame.
    Handles encoding issues gracefully.
    """
    files = sorted(directory.glob(pattern))
    if not files:
        log.error("No CSV files found in %s matching '%s'", directory, pattern)
        log.error("Please download the NWDP Telangana GW Level CSV and place it in: %s", directory)
        log.error("Download URL: https://nwdp.nwic.gov.in/")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="cp1252", low_memory=False)

        df["_source_file"] = f.name
        dfs.append(df)
        log.info("Loaded: %s  (%d rows, %d cols)", f.name, len(df), len(df.columns))

    combined = pd.concat(dfs, ignore_index=True)
    log.info("Combined: %d total rows, %d columns", len(combined), len(combined.columns))
    return combined


def clean_groundwater_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply canonical column mapping and basic cleaning to raw NWDP data.
    """
    if df.empty:
        return df

    df = _map_columns(df)

    # ── Date Parsing ───────────────────────────────────────────
    if "obs_date" in df.columns:
        df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce", dayfirst=True)
        df["obs_year"]   = df["obs_date"].dt.year
        df["obs_month"]  = df["obs_date"].dt.month
        df["obs_season"] = df["obs_month"].map(_month_to_season)
    else:
        log.warning("'obs_date' column not found — date-based features unavailable")

    # ── Depth Cleaning ─────────────────────────────────────────
    if "depth_m" in df.columns:
        df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
        # Remove physically implausible values
        n_before = len(df)
        df = df[(df["depth_m"].isna()) | (df["depth_m"].between(0.0, 200.0))]
        removed = n_before - len(df)
        if removed:
            log.warning("Removed %d rows with implausible depth_m (outside 0–200 m)", removed)

    # ── Coordinate Cleaning ────────────────────────────────────
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Hyderabad bounding box sanity check (approx)
    # Lat: 17.2°–17.6°N  Lon: 78.2°–78.7°E
    if "latitude" in df.columns and "longitude" in df.columns:
        india_mask = (
            df["latitude"].between(8.0, 37.0) &
            df["longitude"].between(68.0, 98.0)
        )
        invalid_coords = (~india_mask) & df["latitude"].notna()
        if invalid_coords.any():
            log.warning(
                "%d rows have coordinates outside India bounding box — flagged",
                invalid_coords.sum()
            )
            df.loc[invalid_coords, ["latitude", "longitude"]] = np.nan

    # ── District Normalisation ─────────────────────────────────
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip().str.title()

    return df


def _month_to_season(month: Optional[int]) -> Optional[str]:
    """Map calendar month to Indian hydrological season."""
    if pd.isna(month):
        return None
    month = int(month)
    if month in (1, 2):
        return "Rabi"           # Post-Rabi / winter
    elif month in (3, 4, 5):
        return "Pre-Monsoon"    # Hottest; water table at annual minimum
    elif month in (6, 7, 8, 9):
        return "Kharif"         # South-West monsoon
    else:  # 10, 11, 12
        return "Post-Monsoon"   # Water table recovery


def filter_by_district(df: pd.DataFrame, district: str) -> pd.DataFrame:
    """Filter observations to a single district (case-insensitive, stripped)."""
    if "district" not in df.columns:
        log.error("Cannot filter by district: column missing after remap.")
        return df

    query_variants = [district.strip().title(), district.strip().upper(), district.strip().lower()]
    mask = df["district"].isin(query_variants)

    # Fuzzy fallback: substring match
    if mask.sum() == 0:
        log.warning(
            "Exact district match for '%s' returned 0 rows. Trying substring…", district
        )
        mask = df["district"].str.contains(district, case=False, na=False)

    result = df[mask].copy()
    log.info(
        "District filter '%s': %d rows from %d unique wells",
        district, len(result),
        result["well_id"].nunique() if "well_id" in result.columns else "?",
    )
    return result


# ─── Feasibility Report ─────────────────────────────────────────────────────

def run_feasibility_report(df: pd.DataFrame, district: str = "Hyderabad") -> dict:
    """
    Compute all Phase 0 feasibility metrics and return a structured report dict.
    Prints a human-readable summary.
    """
    report = {}

    print("\n" + "═" * 65)
    print(f"  GEOGROUND AI — PHASE 0 FEASIBILITY REPORT")
    print(f"  District: {district}")
    print("═" * 65)

    if df.empty:
        print("  ❌  No data loaded. Cannot continue feasibility analysis.")
        print("  Action: Download NWDP CSV and place in ml/data/raw/")
        return {"go": False, "reason": "No data loaded"}

    # ── 1. Well count ──────────────────────────────────────────
    total_rows = len(df)
    well_col = "well_id" if "well_id" in df.columns else None

    if well_col:
        total_wells = df[well_col].nunique()
    else:
        total_wells = None
        log.warning("well_id column not found; well count unavailable")

    print(f"\n  📊 DATASET OVERVIEW")
    print(f"     Total rows (all observations) : {total_rows:,}")
    print(f"     Unique monitoring wells        : {total_wells or 'UNKNOWN'}")

    report["total_rows"]  = total_rows
    report["total_wells"] = total_wells

    # ── 2. Lat/Lon availability ────────────────────────────────
    has_lat = "latitude"  in df.columns
    has_lon = "longitude" in df.columns

    if has_lat and has_lon:
        wells_with_coords = df.dropna(subset=["latitude", "longitude"])
        n_wells_with_coords = (
            wells_with_coords[well_col].nunique()
            if well_col else len(wells_with_coords)
        )
        coord_rate = n_wells_with_coords / total_wells * 100 if total_wells else 0
        print(f"\n  📍 COORDINATE AVAILABILITY")
        print(f"     Wells with valid lat/lon  : {n_wells_with_coords}")
        print(f"     Coverage rate             : {coord_rate:.1f}%")
    else:
        n_wells_with_coords = 0
        coord_rate = 0
        print(f"\n  📍 COORDINATE AVAILABILITY")
        print(f"     ⚠️  latitude/longitude columns not found in data!")
        print(f"     Fallback: Mandal centroid geocoding will be required.")

    report["wells_with_coords"] = n_wells_with_coords
    report["coord_rate_pct"]    = round(coord_rate, 1)

    # ── 3. Temporal coverage ───────────────────────────────────
    print(f"\n  📅 TEMPORAL COVERAGE")
    if "obs_date" in df.columns and df["obs_date"].notna().any():
        min_date = df["obs_date"].min()
        max_date = df["obs_date"].max()
        years    = df["obs_year"].dropna().unique() if "obs_year" in df.columns else []
        n_years  = len(years)
        seasons  = df["obs_season"].dropna().unique() if "obs_season" in df.columns else []

        print(f"     Date range    : {min_date.date()} → {max_date.date()}")
        print(f"     Unique years  : {n_years}  ({sorted(years)[:5]}{'...' if n_years > 5 else ''})")
        print(f"     Seasons found : {list(seasons)}")
    else:
        n_years  = 0
        print(f"     ⚠️  obs_date column missing or all-null")

    report["years_coverage"] = n_years

    # ── 4. Depth (target) missing values ──────────────────────
    print(f"\n  🎯 TARGET VARIABLE  (depth_m — depth to water table)")
    if "depth_m" in df.columns:
        n_null_depth   = df["depth_m"].isna().sum()
        missing_rate   = n_null_depth / total_rows
        usable_rows    = df["depth_m"].notna().sum()
        depth_min      = df["depth_m"].min()
        depth_max      = df["depth_m"].max()
        depth_mean     = df["depth_m"].mean()

        print(f"     Usable rows (depth_m not null) : {usable_rows:,}")
        print(f"     Missing rate                   : {missing_rate*100:.1f}%")
        print(f"     depth_m range                  : {depth_min:.1f} – {depth_max:.1f} m")
        print(f"     depth_m mean                   : {depth_mean:.2f} m")
    else:
        usable_rows  = 0
        missing_rate = 1.0
        print(f"     ❌  depth_m column not found!")

    report["usable_rows"]   = int(usable_rows)
    report["missing_rate"]  = round(missing_rate, 3)

    # ── 5. GO / NO-GO Decision ────────────────────────────────
    print(f"\n  {'═'*55}")
    print(f"  ✅  GO / ❌ NO-GO DECISION")
    print(f"  {'─'*55}")

    checks = {
        "Wells with lat/lon ≥ 15"      : n_wells_with_coords >= MIN_WELLS_WITH_COORDS,
        "Years of coverage ≥ 5"        : n_years >= MIN_YEARS_COVERAGE,
        "Usable rows ≥ 200"            : usable_rows >= MIN_TOTAL_ROWS,
        "Missing depth_m < 30%"        : missing_rate < MAX_MISSING_RATE,
    }

    all_pass = True
    for check, passed in checks.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {check}")
        if not passed:
            all_pass = False

    decision = "GO ✅" if all_pass else "NO-GO ❌"
    print(f"\n  PHASE 0 DECISION: {decision}")

    if not all_pass:
        print(f"\n  RECOMMENDATION:")
        if n_wells_with_coords < MIN_WELLS_WITH_COORDS:
            print(f"  → Expand study area to include Ranga Reddy, Medchal-Malkajgiri")
            print(f"    (Greater Hyderabad Metropolitan Region) to increase well count")
        if n_years < MIN_YEARS_COVERAGE:
            print(f"  → Data temporal coverage is insufficient.")
            print(f"    Try widening to all Telangana or check for older NWDP exports.")
        if usable_rows < MIN_TOTAL_ROWS:
            print(f"  → Total usable rows too low for reliable ML training.")
            print(f"    Expand to state-level Telangana dataset as training corpus.")

    print("═" * 65 + "\n")

    report["checks"]   = checks
    report["go"]       = all_pass
    report["decision"] = decision

    return report


# ─── CLI Entry Point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoGround AI — Phase 0 Data Loader")
    parser.add_argument(
        "--input", default=str(RAW_DATA_DIR),
        help="Directory containing NWDP CSV files (default: ml/data/raw/)",
    )
    parser.add_argument(
        "--district", default="Hyderabad",
        help="District to validate (default: Hyderabad)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run full Phase 0 feasibility report",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save cleaned DataFrame to data/processed/gw_observations.csv",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────
    log.info("Loading CSV files from: %s", input_dir)
    raw_df = load_csv_files(input_dir)

    if raw_df.empty:
        log.error("No data loaded. Place NWDP CSV(s) in %s", input_dir)
        sys.exit(1)

    # ── Clean ─────────────────────────────────────────────────
    log.info("Cleaning data…")
    clean_df = clean_groundwater_data(raw_df)

    # ── Filter ────────────────────────────────────────────────
    district_df = filter_by_district(clean_df, args.district)

    # ── Validate / Report ─────────────────────────────────────
    if args.validate:
        report = run_feasibility_report(district_df, district=args.district)
        report_path = PROC_DATA_DIR / f"phase0_report_{args.district.lower()}.txt"
        PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Also run report on full Telangana data for reference
        print("  — Full Telangana dataset (for reference if Hyderabad NO-GO) —")
        run_feasibility_report(clean_df, district="Telangana (All Districts)")

    # ── Save cleaned data ─────────────────────────────────────
    if args.save and not district_df.empty:
        out_path = PROC_DATA_DIR / f"gw_observations_{args.district.lower()}.csv"
        PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        district_df.to_csv(out_path, index=False)
        log.info("Saved cleaned data: %s  (%d rows)", out_path, len(district_df))

    return district_df


if __name__ == "__main__":
    main()
