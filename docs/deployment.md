# GeoGround AI — Cloud Deployment Guide (Render & Railway)

This guide walks you through deploying GeoGround AI to **Render** or **Railway**.

---

## Architecture Overview

```
[User Browser]
       │
       ▼
[React Frontend] (Static Site / CDN)
       │
       ▼  (REST API calls)
[Backend Gateway (Node.js/Express)]
       │
       ▼  (Internal HTTP)
[ML Service (Python/FastAPI)]
```

---

## Option 1: Deploy on Render (Recommended — Free Tier)

Render supports Infrastructure-as-Code via the provided [`render.yaml`](../render.yaml) blueprint.

### Step 1: Push Project to GitHub
Make sure your latest code and model weights are pushed:
```bash
git add .
git commit -m "chore: prepare cloud deployment configs and model artifacts"
git push origin main
```

### Step 2: Create Blueprint on Render
1. Go to [dashboard.render.com](https://dashboard.render.com) and sign in.
2. Click **New +** (top right) and select **Blueprint**.
3. Connect your GitHub repository (`pavanyadav0106/GeoGround_AI`).
4. Render will automatically read [`render.yaml`](../render.yaml) and discover all 3 services:
   - `geoground-ml-service` (Python Web Service)
   - `geoground-backend` (Node Web Service)
   - `geoground-frontend` (Static Site)
5. Click **Apply**.

### Step 3: Configure Frontend API URL
1. Once `geoground-backend` finishes deploying, copy its URL (e.g. `https://geoground-backend.onrender.com`).
2. Go to your `geoground-frontend` service -> **Environment** tab.
3. Add or update the variable:
   - **Key**: `VITE_API_BASE`
   - **Value**: `https://geoground-backend.onrender.com/api/v1`
4. Click **Save Changes** (Render will re-deploy the frontend with the live backend URL).

---

## Option 2: Deploy on Railway

Railway allows you to deploy multi-container microservices with zero configuration.

### Step 1: Push to GitHub
Ensure all code and model files are committed and pushed to your GitHub repo.

### Step 2: Deploy Services on Railway
1. Go to [railway.app](https://railway.app) and click **New Project** -> **Deploy from GitHub repo**.
2. Select your repository.
3. Railway allows you to add multiple services from the same repo:
   - **Service 1: ML Microservice**
     - Settings -> Root Directory: `/`
     - Dockerfile Path: `ml_service/Dockerfile`
     - Generate Domain (e.g. `geoground-ml.up.railway.app`)
   - **Service 2: Backend Gateway**
     - Click **+ New Service** -> **GitHub Repo** -> select the same repo.
     - Settings -> Root Directory: `backend`
     - Dockerfile Path: `backend/Dockerfile`
     - Variables: `ML_SERVICE_URL` = `https://geoground-ml.up.railway.app`
     - Generate Domain (e.g. `geoground-backend.up.railway.app`)
   - **Service 3: React Frontend**
     - Click **+ New Service** -> **GitHub Repo** -> select the same repo.
     - Settings -> Root Directory: `frontend`
     - Dockerfile Path: `frontend/Dockerfile`
     - Variables: `VITE_API_BASE` = `https://geoground-backend.up.railway.app/api/v1`
     - Generate Domain.

---

## Option 3: Single-Command Docker Compose (VPS / Self-Hosted)

If you have a Linux VPS (DigitalOcean, Hetzner, AWS EC2, or Azure VM):

```bash
# 1. Clone repository
git clone https://github.com/pavanyadav0106/GeoGround_AI.git
cd GeoGround_AI

# 2. Start full-stack with Docker Compose
docker compose up -d --build

# 3. Check status
docker compose ps
```
The app will be live at `http://<your-server-ip>`.
