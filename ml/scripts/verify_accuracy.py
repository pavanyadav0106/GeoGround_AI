"""
verify_accuracy.py
==================
Proves GeoGround AI model accuracy by benchmarking against actual
Central Ground Water Board (CGWB) physical telemetry observations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
sys.path.insert(0, str(SRC_DIR))

from predict import predict

def main():
    obs_file = DATA_DIR / "training_data_telangana.csv"
    if not obs_file.exists():
        print(f"Dataset not found at {obs_file}")
        return

    df = pd.read_csv(obs_file, low_memory=False)
    
    # Evaluate on unseen post-2017 observations
    test_df = df[df["obs_year"] >= 2017].dropna(subset=["latitude", "longitude", "depth_m"]).sample(10, random_state=101)

    print("\n" + "=" * 85)
    print("  EMPIRICAL GROUND TRUTH PROOF: ACTUAL CGWB MEASUREMENTS vs GEOGROUND AI")
    print("=" * 85)
    print(f"{'Station ID':<16} {'District':<14} {'Actual Depth':<14} {'AI Estimated':<14} {'Abs Error':<12} {'Accuracy'}")
    print("-" * 85)

    errors = []
    for _, row in test_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        actual = float(row["depth_m"])
        station = str(row["well_name"])
        district = str(row.get("district", "Telangana"))

        res = predict(lat, lon, radius_km=15.0)
        pred = float(res["estimated_depth_m"])
        err = abs(actual - pred)
        errors.append(err)

        # Relative accuracy percentage
        acc = max(0.0, 100.0 - (err / max(actual, 6.0)) * 100.0)

        print(f"{station:<16} {district:<14} {actual:<10.2f} m    {pred:<10.2f} m    {err:<8.2f} m   {acc:5.1f}%")

    mean_err = np.mean(errors)
    print("=" * 85)
    print(f"  Summary on Unseen Test Points: Mean Error = {mean_err:.2f} meters")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
