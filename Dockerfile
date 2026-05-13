# ---- Stage 1: Builder ----
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Final Image ----
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    MLFLOW_TRACKING_URI=sqlite:///app/data/mlflow.db \
    MLFLOW_PORT=5000

WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local /usr/local

# Copy your application code (including api.py)
COPY config.py .
COPY database.py .
COPY data_generator.py .
COPY recommendation_engine.py .
COPY evaluation.py .
COPY explainability.py .
COPY mlflow_tracking.py .
COPY api.py .
# If you still need main.py for other tasks, copy it too
COPY main.py .

# Run as root to avoid volume permission headaches (dev‑friendly)
EXPOSE 8000 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server (the same as `python api.py`)
CMD ["python", "api.py"]
