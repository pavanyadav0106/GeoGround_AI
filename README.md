# GeoGround AI

> AI-Based Groundwater Level Estimation Using GPS Location and Multi-Source Geospatial Data

**Study Area**: Hyderabad district, Telangana, India (expandable to additional districts)

**Status**: 🔄 Phase 4 — FastAPI ML Service In Progress

---

## Important Disclaimer

GeoGround AI **estimates** groundwater conditions at a GPS location by combining historical groundwater observations from nearby monitoring wells with environmental and geospatial data using machine learning. It does **not** directly measure groundwater or replace physical hydrogeological surveys.

---

## Architecture

```
React Frontend (Leaflet.js)
        ↓
NestJS Backend (API gateway, orchestration, PostgreSQL)
        ↓
FastAPI ML Service
        ↓
XGBoost Model (trained on CGWB/NWDP + environmental data)
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Docker + Docker Compose (optional, for containerised setup)

---

## Quick Start

### 1. Clone and setup environment variables

```bash
git clone <repo>
cd geoground-ai
cp .env.example .env
# Edit .env with your values
```

### 2. Phase 0 — Validate groundwater data (DO THIS FIRST)

```bash
cd ml
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Try automated NWDP download
python scripts/download_nwdp.py

# Run feasibility analysis
python scripts/feasibility.py --district Hyderabad --plot
```

See [ml/README.md](ml/README.md) for detailed instructions.

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Groundwater Data Feasibility | ✅ Done |
| 1 | Data Pipeline (NWDP & CGWB) | ✅ Done |
| 2 | Feature Engineering (NASA, Soil, SRTM, LULC) | ✅ Done |
| 3 | ML Training (XGBoost Engine, MAE=3.04m, R²=0.51) | ✅ Done |
| 4 | FastAPI ML Microservice (:8000) | ✅ Done |
| 5 | Backend Gateway & Reverse Geocoding (:3001) | ✅ Done |
| 6 | React + Leaflet Dark Dashboard (:5173) | ✅ Done |
| 7 | Full-Stack Integration & Launcher | ✅ Done |

---

## Data Sources

| Data | Source | License |
|------|--------|---------|
| Groundwater observations | CGWB via NWDP / India-WRIS | Government of India (Open) |
| Weather (rainfall, temp, humidity) | NASA POWER API | NASA (Public) |
| Soil properties | SoilGrids — ISRIC | CC BY 4.0 |
| Elevation | NASA SRTM via OpenTopoData | NASA (Public) |
| Land use / Land cover | ESA WorldCover | CC BY 4.0 |

---

## Project Structure

```
geoground-ai/
├── ml/                    <- Python ML pipeline (Phase 0-4)
│   ├── data/raw/          <- NWDP CSVs (downloaded)
│   ├── data/processed/    <- Cleaned + enriched CSVs
│   ├── data/external/     <- Cached API responses
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── utils.py
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── scripts/
│   │   ├── download_nwdp.py
│   │   ├── feasibility.py
│   │   ├── build_training_data.py
│   │   ├── build_telangana_training.py
│   │   └── train_model.py
│   └── models/            <- Saved trained models (.pkl / .ubj)
│
├── ml_service/            <- FastAPI ML service (Phase 4)
│   ├── main.py            <- API endpoints
│   └── README.md
│
├── backend/               <- NestJS application (Phase 5)
├── frontend/              <- React + Leaflet dashboard (Phase 6)
├── database/              <- PostgreSQL migrations (Phase 5)
├── docs/                  <- Architecture & API documentation
│
├── run_service.py         <- Convenience launcher for ML service
├── .env.example
├── docker-compose.yml     (Phase 7)
└── README.md
```

---

## Limitations

- Predictions are estimates based on nearby monitoring wells — not direct measurements
- Confidence degrades in areas with sparse monitoring coverage
- Hyderabad district has limited CGWB monitoring wells due to its urban, hard-rock aquifer setting
- The system cannot account for individual private borewell extractions
- Results should not replace professional hydrogeological assessments
