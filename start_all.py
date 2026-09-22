"""
start_all.py
============
GeoGround AI — Full-Stack Orchestration Runner
Starts:
  1. FastAPI ML Microservice (Port 8000)
  2. Express/Node Backend Gateway (Port 3001)
  3. React + Vite Frontend Dashboard (Port 5173)

Usage:
  python start_all.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ML_DIR = ROOT / "ml"
ML_SERVICE_DIR = ROOT / "ml_service"
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

ML_PYTHON = ML_DIR / "venv" / "Scripts" / "python.exe"
if not ML_PYTHON.exists():
    ML_PYTHON = ML_DIR / "venv" / "bin" / "python"
if not ML_PYTHON.exists():
    ML_PYTHON = Path(sys.executable)

print("=" * 65)
print("  🚀 STARTING GEOGROUND AI FULL-STACK SUITE")
print("=" * 65)
print(f"  • ML Service:     http://localhost:8000/docs")
print(f"  • Backend Gateway: http://localhost:3001/api/v1/health")
print(f"  • React Frontend:  http://localhost:5173")
print("=" * 65)

procs = []

try:
    # 1. Start FastAPI ML Service
    print("\n[1/3] Launching FastAPI ML Inference Service on :8000...")
    p_ml = subprocess.Popen(
        [str(ML_PYTHON), str(ML_SERVICE_DIR / "main.py")],
        cwd=str(ROOT),
    )
    procs.append(p_ml)
    time.sleep(2)

    # 2. Start Backend Gateway
    print("[2/3] Launching Backend Gateway on :3001...")
    p_backend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(BACKEND_DIR),
        shell=True,
    )
    procs.append(p_backend)
    time.sleep(2)

    # 3. Start Frontend Dashboard
    print("[3/3] Launching React + Vite Frontend on :5173...")
    p_frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(FRONTEND_DIR),
        shell=True,
    )
    procs.append(p_frontend)

    print("\n✅ All services running! Press Ctrl+C to terminate all services.\n")
    
    # Keep main process alive
    for p in procs:
        p.wait()

except KeyboardInterrupt:
    print("\n🛑 Shutting down GeoGround AI services...")
    for p in procs:
        p.terminate()
    print("👋 Done.")
