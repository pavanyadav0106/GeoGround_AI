"""
train_model.py
==============
GeoGround AI — Phase 3 runner: invoke the ML training pipeline.

Usage:
    # Train on Hyderabad data (fast, 24 wells)
    python scripts/train_model.py

    # Train on full Telangana data (recommended for better generalisation)
    python scripts/train_model.py --data data/processed/training_data_telangana.csv

    # Use a different temporal split year
    python scripts/train_model.py --split-year 2016

    # Run spatial cross-validation too
    python scripts/train_model.py --spatial-cv
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from train import main

if __name__ == "__main__":
    main()
