FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions
COPY ml/requirements.txt /app/ml_requirements.txt
COPY ml_service/requirements.txt /app/service_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/ml_requirements.txt
RUN pip install --no-cache-dir -r /app/service_requirements.txt

# Copy application code
COPY ml/src /app/ml/src
COPY ml_service /app/ml_service

EXPOSE 8000

CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
