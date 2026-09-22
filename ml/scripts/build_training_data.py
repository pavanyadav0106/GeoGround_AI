"""
build_training_data.py
======================
GeoGround AI — Phase 1 Orchestrator: Build the Unified Training Dataset

Runs the complete data pipeline:
  Phase 1.1  preprocess groundwater observations → wells.csv + observations.csv
  Phase 1.2  NASA POWER weather (rainfall, temp, humidity)
  Phase 1.3  SoilGrids soil properties (clay, silt, sand, pH, SOC)
  Phase 1.4  SRTM elevation
  Phase 1.5  ESA WorldCover LULC
  Phase 1.6  Nearby-well spatial aggregation + trend features
  → Output:  training_data.csv  (the unified ML training table)

Usage:
    # Full Telangana dataset (recommended for training — more data)
    python scripts/build_training_data.py

    # Hyderabad only (faster for testing the pipeline)
    python scripts/build_training_data.py --district Hyderabad

    # Skip slow API fetches (use cached results only)
    python scripts/build_training_data.py --cache-only

    # Skip nearby-well computation (very slow for 100k+ rows)
    python scripts/build_training_data.py --no-nearby
"""

import argparse
import logging
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import run_preprocessing, PROC_DIR, RAW_DIR
from feature_engineering import build_feature_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="GeoGround AI — Build unified training dataset"
    )
    parser.add_argument(
        "--district", default=None,
        help="Filter to specific district. Default: all Telangana (recommended)"
    )
    parser.add_argument(
        "--radius-km", type=float, default=15.0,
        help="Search radius for nearby-well aggregation (default: 15 km)"
    )
    parser.add_argument(
        "--no-nasa",    action="store_true", help="Skip NASA POWER fetching"
    )
    parser.add_argument(
        "--no-soil",    action="store_true", help="Skip SoilGrids fetching"
    )
    parser.add_argument(
        "--no-elev",    action="store_true", help="Skip elevation fetching"
    )
    parser.add_argument(
        "--no-lulc",    action="store_true", help="Skip WorldCover LULC fetching"
    )
    parser.add_argument(
        "--no-nearby",  action="store_true", help="Skip nearby-well aggregation"
    )
    parser.add_argument(
        "--no-trend",   action="store_true", help="Skip trend feature computation"
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Only use cached API responses; do not make new API calls"
    )
    args = parser.parse_args()

    t0 = time.time()

    print("\n" + "═" * 65)
    print("  GEOGROUND AI — PHASE 1: BUILD TRAINING DATA")
    print(f"  District: {args.district or 'All Telangana'}")
    print(f"  Radius:   {args.radius_km} km")
    print("═" * 65 + "\n")

    # ── Phase 1.1: Preprocess GW data ─────────────────────────
    print("  [1.1] Preprocessing groundwater observations...")
    wells_df, obs_df = run_preprocessing(
        raw_dir=RAW_DIR,
        proc_dir=PROC_DIR,
        district_filter=args.district,
    )
    print(f"        Wells:        {len(wells_df):,}")
    print(f"        Observations: {len(obs_df):,}")
    print()

    # ── Phase 1.2–1.6: Environmental features ─────────────────
    print("  [1.2-1.6] Collecting environmental features...")
    print(f"            NASA POWER : {'SKIP' if args.no_nasa else 'YES'}")
    print(f"            SoilGrids  : {'SKIP' if args.no_soil else 'YES'}")
    print(f"            Elevation  : {'SKIP' if args.no_elev else 'YES'}")
    print(f"            WorldCover : {'SKIP' if args.no_lulc else 'YES'}")
    print(f"            Nearby     : {'SKIP' if args.no_nearby else 'YES'}")
    print(f"            Trend      : {'SKIP' if args.no_trend else 'YES'}")
    print(f"            Cache-only : {'YES' if args.cache_only else 'NO'}")
    print()

    training_df = build_feature_table(
        obs_df       = obs_df,
        wells_df     = wells_df,
        radius_km    = args.radius_km,
        fetch_nasa   = not args.no_nasa,
        fetch_soil   = not args.no_soil,
        fetch_elev   = not args.no_elev,
        fetch_lulc   = not args.no_lulc,
        fetch_nearby = not args.no_nearby,
        fetch_trend  = not args.no_trend,
    )

    # ── Save training data ─────────────────────────────────────
    suffix = f"_{args.district.lower()}" if args.district else "_telangana"
    out_path = PROC_DIR / f"training_data{suffix}.csv"
    training_df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print("\n" + "═" * 65)
    print(f"  PHASE 1 COMPLETE in {elapsed:.1f}s")
    print(f"  Training data: {out_path}")
    print(f"  Rows:    {len(training_df):,}")
    print(f"  Columns: {len(training_df.columns)}")
    print()
    print("  Feature columns:")
    feature_cols = [c for c in training_df.columns if c not in
                    ("well_name", "district", "tehsil", "agency", "state",
                     "basin", "obs_date_raw", "block", "village", "_source")]
    for col in feature_cols:
        null_pct = training_df[col].isna().mean() * 100
        print(f"    {col:35s}  null={null_pct:.1f}%")
    print("═" * 65 + "\n")

    print("  Next: python scripts/train_model.py")
    return training_df


if __name__ == "__main__":
    main()
