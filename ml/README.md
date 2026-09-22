# GeoGround AI — ML Service

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Phase 0: Groundwater Data Feasibility

### Step 1 — Download NWDP data

```bash
python scripts/download_nwdp.py --state Telangana
```

If automated download fails, follow the manual instructions printed to console.

Manual download URL: https://nwdp.nwic.gov.in/

Search for: **"Ground Water Level Manual Quarterly Telangana"**

Save the CSV(s) to: `ml/data/raw/`

### Step 2 — Run feasibility analysis

```bash
# Basic report
python scripts/feasibility.py --district Hyderabad

# With spatial map (opens HTML in browser)
python scripts/feasibility.py --district Hyderabad --plot
```

Or via the data loader directly:

```bash
python src/data_loader.py --validate --district Hyderabad
```

### GO/NO-GO Thresholds

| Metric | Minimum |
|--------|---------|
| Wells with lat/lon | ≥ 15 |
| Years of coverage | ≥ 5 |
| Usable rows | ≥ 200 |
| Missing depth_m | < 30% |

---

## Directory Structure

```
ml/
├── data/
│   ├── raw/          ← Place downloaded NWDP CSVs here
│   ├── processed/    ← Cleaned output files
│   └── external/     ← Environmental data (weather, soil, LULC)
├── notebooks/        ← Jupyter exploration notebooks
├── scripts/
│   ├── download_nwdp.py    ← NWDP data downloader
│   └── feasibility.py      ← Phase 0 GO/NO-GO reporter
├── src/
│   ├── data_loader.py      ← CSV parser + validator
│   ├── utils.py            ← Shared utilities (haversine, trend, etc.)
│   ├── preprocessing.py    ← Feature preprocessing (Phase 2)
│   ├── feature_engineering.py  ← Feature builder (Phase 2)
│   ├── train.py            ← Model training (Phase 3)
│   ├── evaluate.py         ← Evaluation metrics (Phase 3)
│   └── predict.py          ← Inference pipeline (Phase 4)
├── models/           ← Saved trained models (.pkl / .ubj)
└── requirements.txt
```

---

## Data Sources

| Data Type | Source | Access |
|-----------|--------|--------|
| Groundwater | CGWB via NWDP | CSV download (free) |
| Weather | NASA POWER API | REST API (free, no key) |
| Soil | SoilGrids (ISRIC) | COG via rasterio (free) |
| Elevation | NASA SRTM | COG via rasterio (free) |
| Land Use | ESA WorldCover | AWS S3 COG (free) |

---

## Important Notes

1. **Do not fabricate groundwater data.** All training data must come from real CGWB/NWDP observations.
2. **Data leakage prevention**: When computing nearby-well aggregation features for a training well, that well itself is always excluded from its own neighborhood calculation.
3. **Confidence scores** are explainable heuristics (well count + distance), NOT statistically calibrated probabilities.
4. **Groundwater condition thresholds** (Excellent/Good/Moderate/Poor) are project-defined interpretation categories, NOT official CGWB standards.
