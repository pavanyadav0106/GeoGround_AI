# GeoGround AI — API Specification

This document defines the complete API contracts for the GeoGround AI platform across the **ML Inference Service (FastAPI)** and the **Backend Gateway (NestJS)**.

---

## ⚡ 1. FastAPI ML Service (`http://localhost:8000`)

### `GET /health`
Liveness probe.
- **Response**:
  ```json
  {
    "status": "ok",
    "service": "geoground-ai-ml"
  }
  ```

### `GET /model/info`
Returns metadata about the active ML model and validation metrics.
- **Response**:
  ```json
  {
    "status": "loaded",
    "model_type": "XGBoost",
    "train_rows": 120400,
    "test_rows": 34104,
    "split_year": 2017,
    "features": ["latitude", "longitude", "obs_year", "obs_month", "elevation_m", "..."],
    "target": "depth_m",
    "test_metrics": {
      "model": "XGBoost",
      "mae": 1.42,
      "rmse": 2.15,
      "r2": 0.841
    }
  }
  ```

### `POST /predict`
Predicts groundwater depth and condition for a GPS location.
- **Request Body**:
  ```json
  {
    "latitude": 17.3850,
    "longitude": 78.4867,
    "radius_km": 15.0,
    "query_date": "2024-06-15"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "latitude": 17.385,
    "longitude": 78.4867,
    "estimated_depth_m": 8.45,
    "condition": "Good",
    "trend": "Stable",
    "confidence": 0.82,
    "confidence_note": "High confidence: 8 monitoring wells located within 15 km (nearest: 2.1 km).",
    "nearby_wells": [
      {
        "well_name": "CGWB_HYD_01",
        "distance_km": 2.14,
        "recent_depth_m": 7.9,
        "latitude": 17.391,
        "longitude": 78.472
      }
    ],
    "weather": {
      "rainfall_mm": 112.4,
      "temp_c": 31.2,
      "humidity_pct": 65.0
    },
    "soil": {
      "clay_pct": 28.5,
      "sand_pct": 45.2,
      "silt_pct": 26.3,
      "soil_ph": 6.8
    },
    "elevation_m": 505.0,
    "lulc_label": "Built_Up",
    "model_info": {
      "model_type": "XGBoost",
      "train_mae_m": 1.42,
      "train_r2": 0.841,
      "n_wells_used": 8
    },
    "disclaimer": "This is an AI estimation based on nearby CGWB monitoring wells and environmental features. It does not replace physical hydrogeological borehole testing."
  }
  ```

### `GET /wells/nearby`
Retrieves physical CGWB monitoring wells within a radius.
- **Query Parameters**:
  - `lat`: float (required)
  - `lon`: float (required)
  - `radius_km`: float (default 15.0)
  - `limit`: int (default 20)

---

## 🌐 2. NestJS Backend Gateway (`http://localhost:3001`)

### `POST /api/v1/groundwater/estimate`
Main user-facing endpoint. Reverse geocodes the location and proxies to FastAPI ML Service with caching.
- **Request Body**:
  ```json
  {
    "latitude": 17.3850,
    "longitude": 78.4867,
    "radiusKm": 15
  }
  ```
- **Response**:
  ```json
  {
    "queryId": "uuid-v4",
    "location": {
      "latitude": 17.3850,
      "longitude": 78.4867,
      "displayName": "Charminar, Hyderabad, Telangana, India",
      "district": "Hyderabad",
      "state": "Telangana"
    },
    "prediction": {
      "estimatedDepthMeters": 8.45,
      "waterLevelStatus": "Good",
      "historicalTrend": "Stable",
      "confidenceScore": 0.82,
      "confidenceExplanation": "..."
    },
    "environmentalSnapshot": {
      "rainfallMm": 112.4,
      "temperatureCelsius": 31.2,
      "relativeHumidityPct": 65.0,
      "elevationMeters": 505.0,
      "landCover": "Built_Up",
      "soil": {
        "clay": 28.5,
        "sand": 45.2,
        "silt": 26.3,
        "ph": 6.8
      }
    },
    "nearbyMonitoringWells": [ ... ],
    "timestamp": "2026-08-29T16:20:00.000Z"
  }
  ```

### `GET /api/v1/districts/summary`
District-wide groundwater health summaries and monitoring density for the map overview.
