"""
feasibility.py
==============
GeoGround AI — Phase 0: Standalone Feasibility Report Runner

Loads NWDP data from ml/data/raw/, runs cleaning and filtering,
and outputs a GO/NO-GO report to the console and a text file.

Usage:
    python scripts/feasibility.py
    python scripts/feasibility.py --district Hyderabad
    python scripts/feasibility.py --plot  (saves a spatial map HTML)
"""

import argparse
import sys
from pathlib import Path

# Make src importable
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import (
    RAW_DATA_DIR,
    PROC_DATA_DIR,
    load_csv_files,
    clean_groundwater_data,
    filter_by_district,
    run_feasibility_report,
)


def save_spatial_map(df, district: str, out_dir: Path):
    """Generate a Folium HTML map showing well locations."""
    try:
        import folium
        import pandas as pd

        wells = df.dropna(subset=["latitude", "longitude"])
        if wells.empty:
            print("  ⚠️  No wells with coordinates — skipping map.")
            return

        center_lat = wells["latitude"].mean()
        center_lon = wells["longitude"].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        for _, row in wells.iterrows():
            popup_text = (
                f"Well: {row.get('well_name', 'N/A')}<br>"
                f"Mandal: {row.get('mandal', 'N/A')}<br>"
                f"Observations: {int(row.get('_n_obs', 0))}"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                color="#1a73e8",
                fill=True,
                fill_color="#1a73e8",
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=250),
            ).add_to(m)

        map_path = out_dir / f"phase0_wells_{district.lower()}.html"
        m.save(str(map_path))
        print(f"  ✅ Spatial map saved: {map_path}")
        print(f"     Open in browser: file:///{map_path}")

    except ImportError:
        print("  ℹ️  folium not installed — skipping map (pip install folium)")


def main():
    parser = argparse.ArgumentParser(description="GeoGround AI Phase 0 Feasibility")
    parser.add_argument("--district", default="Hyderabad")
    parser.add_argument("--plot",     action="store_true",
                        help="Generate a Folium HTML map of well locations")
    parser.add_argument("--input",    default=str(RAW_DATA_DIR))
    args = parser.parse_args()

    input_dir = Path(args.input)

    # ── Load all CSVs in raw/ ──────────────────────────────────
    raw_df = load_csv_files(input_dir)

    if raw_df.empty:
        print("\n  ❌  No CSV data found in:", input_dir)
        print("  Run: python scripts/download_nwdp.py")
        print("  OR manually download from: https://nwdp.nwic.gov.in/")
        sys.exit(1)

    # ── Clean ─────────────────────────────────────────────────
    clean_df = clean_groundwater_data(raw_df)

    # ── Filter to study district ───────────────────────────────
    district_df = filter_by_district(clean_df, args.district)

    # ── Feasibility report — target district ──────────────────
    report = run_feasibility_report(district_df, district=args.district)

    # ── Feasibility report — full state (fallback reference) ──
    print("\n  — Reference: Full Dataset (All Districts) —")
    run_feasibility_report(clean_df, district="All Districts")

    # ── Spatial map ───────────────────────────────────────────
    if args.plot:
        PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Aggregate to unique wells for mapping
        if "well_id" in district_df.columns:
            import pandas as pd
            well_summary = (
                district_df.groupby("well_id")
                .agg(
                    latitude=("latitude", "first"),
                    longitude=("longitude", "first"),
                    well_name=("well_name", "first") if "well_name" in district_df.columns else ("well_id", "first"),
                    mandal=("mandal", "first") if "mandal" in district_df.columns else ("well_id", "first"),
                    _n_obs=("depth_m", "count") if "depth_m" in district_df.columns else ("well_id", "count"),
                )
                .reset_index()
            )
            save_spatial_map(well_summary, args.district, PROC_DATA_DIR)
        else:
            save_spatial_map(district_df, args.district, PROC_DATA_DIR)

    # ── Exit code reflects GO/NO-GO ────────────────────────────
    sys.exit(0 if report.get("go") else 1)


if __name__ == "__main__":
    main()
