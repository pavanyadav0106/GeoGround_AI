# GeoGround AI — System Architecture

GeoGround AI is an AI-powered groundwater level estimation platform designed for Telangana, India (expandable pan-India). It combines historical monitoring data from Central Ground Water Board (CGWB) / National Water Data Portal (NWDP) with multi-source environmental and geospatial features (NASA POWER, ISRIC SoilGrids, NASA SRTM, ESA WorldCover) to predict depth-to-water table at any GPS location.

---

## 🏗️ High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 React Frontend (Vite)                       │
│    Interactive Leaflet Map • GPS Pinpoint • Depth Gauge     │
│       Confidence Radar • Well Explorer • Analytics          │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST / JSON
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 NestJS Backend Gateway                      │
│   • Request Orchestration & Input Validation (class-validator)
│   • PostgreSQL + PostGIS (Spatial caching & prediction logs)│
│   • Reverse Geocoding (Nominatim / OSM)                     │
│   • District Aggregations & Health Summaries                │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST / Internal HTTP
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI ML Inference Service                │
│   • Real-time Environmental Feature Gathering (Cached)      │
│   • XGBoost + LightGBM Groundwater Regression Engine        │
│   • Spatial KD-Tree Nearby Well Analysis & Temporal Trends  │
│   • Explainable Heuristic Confidence Scoring               │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│     Trained ML Models        │    │    External Geo Data APIs    │
│ • best_model.ubj (XGBoost)   │    │ • NASA POWER (Weather)       │
│ • preprocessor.pkl           │    │ • ISRIC SoilGrids (Soil)     │
│ • wells_telangana.csv        │    │ • OpenTopoData / SRTM (Elev) │
│ • observations_telangana.csv │    │ • ESA WorldCover (LULC)      │
└──────────────────────────────┘    └──────────────────────────────┘
```

---

## 📦 Component Overview

### 1. ML Core & Service (`ml/` & `ml_service/`)
- **Language**: Python 3.10+
- **Framework**: FastAPI, XGBoost, Scikit-learn, LightGBM, Pandas, NumPy
- **Responsibilities**:
  - Offline training on 154,504 historical observations across 1,025 wells (16.2 years).
  - Online feature extraction for requested `(lat, lon)`.
  - Groundwater depth regression and classification into condition categories:
    - *Excellent* (< 5m BGL)
    - *Good* (5–10m BGL)
    - *Moderate* (10–20m BGL)
    - *Critical/Poor* (> 20m BGL)
  - Distance & density-based confidence calculation.

### 2. Backend Gateway (`backend/`)
- **Framework**: NestJS (TypeScript)
- **Database**: PostgreSQL 14+ with Prisma or TypeORM
- **Responsibilities**:
  - Central API gateway for web and mobile clients.
  - Caches frequent location queries to prevent duplicate external API calls.
  - Logs user queries, regional statistics, and seasonal reports.
  - Reverse geocoding for human-readable address resolution (district, mandal, village).

### 3. Frontend Dashboard (`frontend/`)
- **Framework**: React + Vite (Vanilla CSS design system)
- **Map Library**: Leaflet.js / React-Leaflet with custom map styling
- **Key Features**:
  - Interactive map with click-to-predict and GPS location detector.
  - Real-time gauge showing estimated groundwater depth (m BGL).
  - Nearby monitoring well clusters with distance, historical trends, and depth charts.
  - Environmental breakdown (rainfall, temperature, soil composition, elevation, land use).
  - Confidence breakdown with transparent explanations.

---

## 🔐 Security & Deployment

- Environment variables stored securely in `.env` (template in `.env.example`).
- Rate limiting on external geocoding and NASA POWER APIs with local disk-caching.
- CORS policy configured across NestJS and FastAPI.
- Containerized deployment ready via Docker Compose.
