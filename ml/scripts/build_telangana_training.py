"""
build_telangana_training.py
============================
GeoGround AI — Optimised Phase 1 runner for full Telangana dataset.

Differences from build_training_data.py:
  • Reads the already-preprocessed wells_telangana.csv / observations_telangana.csv
    directly (skips the slow re-preprocessing from raw CSVs).
  • Defaults to --no-nearby for the large 154 k-row dataset (nearby-well aggregation
    is O(n_obs × n_wells) and takes hours; use --with-nearby only on a subset).
  • All API calls are cached; re-runs are fast.

Usage:
    # Recommended (fast, skips nearby-well aggregation):
    python scripts/build_telangana_training.py

    # Full pipeline including nearby-well features (very slow, ~2-4 hours):
    python scripts/build_telangana_training.py --with-nearby

    # Skip specific API fetches:
    python scripts/build_telangana_training.py --no-nasa --no-soil --no-elev --no-lulc
"""

import argparse
import logging
import sys
import time
from pathlib import Path

SRC_DIR  = Path(__file__).resolve().parents[1] / "src"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
sys.path.insert(0, str(SRC_DIR))

from feature_engineering import build_feature_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="GeoGround AI — Build Telangana training dataset (optimised)"
    )
    parser.add_argument(
        "--radius-km", type=float, default=15.0,
        help="Search radius for nearby-well aggregation (default: 15 km)"
    )
    parser.add_argument("--no-nasa",    action="store_true", help="Skip NASA POWER fetching")
    parser.add_argument("--no-soil",    action="store_true", help="Skip SoilGrids fetching")
    parser.add_argument("--no-elev",    action="store_true", help="Skip elevation fetching")
    parser.add_argument("--no-lulc",    action="store_true", help="Skip WorldCover LULC")
    parser.add_argument("--no-trend",   action="store_true", help="Skip trend features")
    parser.add_argument(
        "--with-nearby", action="store_true",
        help="Include nearby-well aggregation (VERY SLOW for 154k rows — hours)"
    )
    args = parser.parse_args()

    t0 = time.time()

    print("\n" + "=" * 70)
    print("  GEOGROUND AI -- BUILD TELANGANA TRAINING DATA (optimised)")
    print(f"  Radius:     {args.radius_km} km")
    print(f"  NASA POWER: {'SKIP' if args.no_nasa  else 'YES (cached)'}")
    print(f"  SoilGrids:  {'SKIP' if args.no_soil  else 'YES (cached)'}")
    print(f"  Elevation:  {'SKIP' if args.no_elev  else 'YES (cached)'}")
    print(f"  WorldCover: {'SKIP' if args.no_lulc  else 'YES (cached)'}")
    print(f"  Trend:      {'SKIP' if args.no_trend else 'YES'}")
    print(f"  Nearby:     {'YES (slow!)' if args.with_nearby else 'SKIP (use --with-nearby to enable)'}")
    print("=" * 70 + "\n")

    # ── Load already-preprocessed Telangana data ──────────────────────────
    wells_path = PROC_DIR / "wells_telangana.csv"
    obs_path   = PROC_DIR / "observations_telangana.csv"

    if not wells_path.exists() or not obs_path.exists():
        log.error("Preprocessed Telangana files not found.")
        log.error("Expected:")
        log.error("  %s", wells_path)
        log.error("  %s", obs_path)
        log.error("Run:  python scripts/build_training_data.py")
        sys.exit(1)

    log.info("Loading preprocessed Telangana data...")
    wells_df = pd.read_csv(wells_path, low_memory=False)
    obs_df   = pd.read_csv(obs_path,   low_memory=False, parse_dates=["obs_date"])

    # Restore Int dtypes after CSV load
    for col in ("obs_year", "obs_month"):
        if col in obs_df.columns:
            obs_df[col] = pd.to_numeric(obs_df[col], errors="coerce")

    print(f"  Wells loaded:        {len(wells_df):,}")
    print(f"  Observations loaded: {len(obs_df):,}")
    print()

    # ── Build feature table ───────────────────────────────────────────────
    training_df = build_feature_table(
        obs_df       = obs_df,
        wells_df     = wells_df,
        radius_km    = args.radius_km,
        fetch_nasa   = not args.no_nasa,
        fetch_soil   = not args.no_soil,
        fetch_elev   = not args.no_elev,
        fetch_lulc   = not args.no_lulc,
        fetch_nearby = args.with_nearby,
        fetch_trend  = not args.no_trend,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = PROC_DIR / "training_data_telangana.csv"
    training_df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  COMPLETE in {elapsed:.1f}s")
    print(f"  Output:  {out_path}")
    print(f"  Rows:    {len(training_df):,}")
    print(f"  Columns: {len(training_df.columns)}")
    print()
    print("  Feature null rates:")
    feature_cols = [c for c in training_df.columns if c not in
                    ("well_name", "district", "tehsil", "agency", "state",
                     "basin", "obs_date", "obs_date_raw", "block", "village", "_source")]
    for col in feature_cols:
        null_pct = training_df[col].isna().mean() * 100
        bar = "#" * int((100 - null_pct) / 5)
        print(f"    {col:35s}  null={null_pct:5.1f}%  {bar}")
    print("=" * 70 + "\n")
    print("  Next: python scripts/train_model.py --data data/processed/training_data_telangana.csv")

    return training_df


if __name__ == "__main__":
    main()
