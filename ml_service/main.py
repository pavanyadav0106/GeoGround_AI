"""
main.py
=======
GeoGround AI — Phase 4: FastAPI ML Service

Exposes the trained groundwater prediction model as a REST API.

Endpoints:
  GET  /health               — liveness probe
  GET  /model/info           — model metadata
  POST /predict              — predict groundwater depth at (lat, lon)
  POST /predict/batch        — batch predictions (up to 50 points)
  GET  /wells/nearby         — list monitoring wells near a point
  GET  /docs                 — OpenAPI docs (Swagger UI)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add ML src to path
ML_ROOT = Path(__file__).resolve().parents[1] / "ml"
SRC_DIR = ML_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="GeoGround AI — ML Service",
    description=(
        "AI-based groundwater level estimation for GPS locations in Telangana, India. "
        "Uses XGBoost trained on CGWB/NWDP monitoring data enriched with NASA POWER "
        "weather, SoilGrids soil properties, SRTM elevation, and ESA WorldCover LULC."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # NestJS backend + dev frontend
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=8.0, le=37.0,  description="Latitude (India bounds)")
    longitude: float = Field(..., ge=68.0, le=98.0, description="Longitude (India bounds)")
    radius_km: float = Field(15.0, ge=1.0, le=100.0, description="Search radius for nearby wells (km)")
    query_date: Optional[str] = Field(
        None,
        description="ISO date string YYYY-MM-DD. Defaults to today.",
        examples=["2024-06-15"],
    )

    @field_validator("query_date", mode="before")
    @classmethod
    def validate_date(cls, v):
        if v is None:
            return v
        import pandas as pd
        try:
            pd.Timestamp(v)
        except Exception:
            raise ValueError(f"Invalid date format: {v!r}. Use YYYY-MM-DD.")
        return v


class BatchPredictRequest(BaseModel):
    points: list[PredictRequest] = Field(
        ..., min_length=1, max_length=50,
        description="List of prediction requests (max 50)"
    )


class NearbyWellsResponse(BaseModel):
    latitude: float
    longitude: float
    radius_km: float
    n_wells: int
    wells: list[dict]


# ─── Startup: warm up model ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load the model and well data so first request is fast."""
    try:
        from predict import load_model, load_wells_and_observations
        load_model()
        load_wells_and_observations()
        log.info("Model and well data loaded successfully on startup.")
    except FileNotFoundError as e:
        log.warning(
            "Model not found on startup (will fail on first /predict call): %s", e
        )
    except Exception as e:
        log.error("Startup error: %s", e)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    """Liveness probe — returns 200 OK if service is running."""
    return {"status": "ok", "service": "geoground-ai-ml"}


@app.get("/model/info", tags=["Meta"])
def model_info():
    """Return metadata about the loaded model."""
    try:
        from predict import load_model
        _, _, meta = load_model()
        return {
            "status":       "loaded",
            "model_type":   meta.get("best_model"),
            "train_rows":   meta.get("train_rows"),
            "test_rows":    meta.get("test_rows"),
            "split_year":   meta.get("split_year"),
            "features":     meta.get("features", []),
            "target":       meta.get("target"),
            "test_metrics": meta.get("best_result", {}),
            "all_results":  meta.get("results", []),
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {e}. Run: python scripts/train_model.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", tags=["Prediction"])
def predict_endpoint(req: PredictRequest):
    """
    Predict groundwater depth at a GPS location.

    Returns estimated depth (m below ground level), condition classification,
    trend, confidence score, nearby wells, and environmental feature snapshot.
    """
    try:
        import pandas as pd
        from predict import predict

        query_date = pd.Timestamp(req.query_date) if req.query_date else None
        result = predict(
            lat=req.latitude,
            lon=req.longitude,
            radius_km=req.radius_km,
            query_date=query_date,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {e}. Run: python scripts/train_model.py"
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Prediction error for (%.4f, %.4f)", req.latitude, req.longitude)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchPredictRequest):
    """
    Batch prediction for up to 50 points.

    Each point is predicted independently. Failed points are returned
    with an 'error' field instead of 'estimated_depth_m'.
    """
    import pandas as pd
    from predict import predict

    results = []
    for point in req.points:
        try:
            query_date = pd.Timestamp(point.query_date) if point.query_date else None
            result = predict(
                lat=point.latitude,
                lon=point.longitude,
                radius_km=point.radius_km,
                query_date=query_date,
            )
            results.append(result)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            log.warning(
                "Batch prediction error for (%.4f, %.4f): %s",
                point.latitude, point.longitude, e
            )
            results.append({
                "latitude":  point.latitude,
                "longitude": point.longitude,
                "error":     str(e),
            })

    return {"n_points": len(results), "predictions": results}


@app.get("/wells/nearby", tags=["Wells"])
def nearby_wells(
    lat: float = Query(..., ge=8.0,  le=37.0,  description="Latitude"),
    lon: float = Query(..., ge=68.0, le=98.0,  description="Longitude"),
    radius_km: float = Query(15.0, ge=1.0, le=100.0, description="Search radius (km)"),
    limit: int = Query(20, ge=1, le=200, description="Max wells to return"),
):
    """
    List monitoring wells within radius_km of the given point.
    Returns well names, distances, and recent average depths.
    """
    try:
        import pandas as pd
        from predict import load_wells_and_observations
        from utils import haversine_km

        wells, obs = load_wells_and_observations()

        if wells.empty:
            return NearbyWellsResponse(
                latitude=lat, longitude=lon, radius_km=radius_km,
                n_wells=0, wells=[]
            )

        valid = wells.dropna(subset=["latitude", "longitude"]).copy()
        valid["distance_km"] = valid.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
        )
        nearby = valid[valid["distance_km"] <= radius_km].sort_values("distance_km").head(limit)

        # Attach recent mean depth from obs
        cutoff_year = pd.Timestamp.now().year - 3
        recent_obs = obs[obs["obs_date"].dt.year >= cutoff_year] if not obs.empty else pd.DataFrame()

        well_list = []
        for _, w in nearby.iterrows():
            w_obs = recent_obs[recent_obs["well_name"] == w["well_name"]] if not recent_obs.empty else pd.DataFrame()
            recent_depth = float(w_obs["depth_m"].mean()) if len(w_obs) > 0 else None
            well_list.append({
                "well_name":        w["well_name"],
                "distance_km":      round(float(w["distance_km"]), 3),
                "latitude":         float(w["latitude"]),
                "longitude":        float(w["longitude"]),
                "district":         w.get("district"),
                "n_observations":   int(w.get("n_observations", 0)),
                "recent_depth_m":   round(recent_depth, 2) if recent_depth is not None else None,
                "years_coverage":   w.get("years_coverage"),
            })

        return {
            "latitude":  lat,
            "longitude": lon,
            "radius_km": radius_km,
            "n_wells":   len(well_list),
            "wells":     well_list,
        }

    except Exception as e:
        log.exception("nearby_wells error")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("ML_SERVICE_PORT", "8000")),
        reload=True,
        log_level="info",
    )
