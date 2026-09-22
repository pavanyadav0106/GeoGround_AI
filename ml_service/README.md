# GeoGround AI — ML Service

FastAPI service that serves the trained XGBoost groundwater model.

## Setup

```bash
# Use the same venv as the ml/ pipeline (already has all deps)
cd ml
venv\Scripts\activate        # Windows

# OR create a dedicated venv for the service only:
# python -m venv venv_service
# venv_service\Scripts\activate
# pip install -r ../ml_service/requirements.txt
# pip install -r requirements.txt   (for the ML deps)
```

## Run

```bash
# From the project root
python ml_service/main.py

# Or with uvicorn directly:
cd ml_service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Service starts on **http://localhost:8000**

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/model/info` | Model metadata and test metrics |
| POST | `/predict` | Predict at a single (lat, lon) |
| POST | `/predict/batch` | Batch predictions (up to 50 points) |
| GET | `/wells/nearby` | List monitoring wells near a point |
| GET | `/docs` | Swagger UI (interactive API docs) |
| GET | `/redoc` | ReDoc API docs |

## Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 17.385, "longitude": 78.486, "radius_km": 15}'
```

Response:
```json
{
  "latitude": 17.385,
  "longitude": 78.486,
  "estimated_depth_m": 12.34,
  "condition": "Good",
  "trend": "Stable",
  "confidence": 0.72,
  "confidence_note": "...",
  "nearby_wells": [...],
  "weather": {"rainfall_mm": 45.2, "temp_c": 28.1, "humidity_pct": 68.4},
  "soil": {"clay_pct": 28.1, "sand_pct": 41.3, "silt_pct": 30.6, "soil_ph": 6.8},
  "elevation_m": 536.0,
  "lulc_label": "Built_Up",
  "model_info": {"model_type": "XGBoost", "train_mae_m": 1.23, "train_r2": 0.87},
  "disclaimer": "..."
}
```

## Example: Nearby Wells

```bash
curl "http://localhost:8000/wells/nearby?lat=17.385&lon=78.486&radius_km=20"
```

## Prerequisites

The model must be trained before starting the service:

```bash
cd ml
python scripts/train_model.py --data data/processed/training_data_telangana.csv
```
