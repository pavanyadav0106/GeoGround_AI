"""
run_service.py
==============
GeoGround AI — convenience launcher for the FastAPI ML service.
Run from the project root:  python run_service.py
"""

import subprocess
import sys
from pathlib import Path

ML_VENV = Path(__file__).parent / "ml" / "venv" / "Scripts" / "python.exe"
if not ML_VENV.exists():
    # Unix path
    ML_VENV = Path(__file__).parent / "ml" / "venv" / "bin" / "python"

SERVICE = Path(__file__).parent / "ml_service" / "main.py"

print("Starting GeoGround AI ML Service...")
print(f"  Python:  {ML_VENV}")
print(f"  Service: {SERVICE}")
print(f"  URL:     http://localhost:8000")
print(f"  Docs:    http://localhost:8000/docs")
print()

subprocess.run([str(ML_VENV), str(SERVICE)], check=True)
