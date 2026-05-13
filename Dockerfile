# ===========================================
# Healthcare Recommendation System Dockerfile
# Multi-stage build with MLflow support
# (Reliable copy of full Python installation)
# ===========================================

# ---- Stage 1: Builder ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install packages globally
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Final Image ----
FROM python:3.10-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    MLFLOW_TRACKING_URI=sqlite:///app/data/mlflow.db \
    MLFLOW_PORT=5000

WORKDIR /app

# Create non-root user and data directory
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Copy the entire Python installation from builder (includes site-packages, bin, libs)
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY config.py .
COPY database.py .
COPY data_generator.py .
COPY recommendation_engine.py .
COPY evaluation.py .
COPY explainability.py .
COPY mlflow_tracking.py .
COPY api.py .
COPY main.py .

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE ${APP_PORT}
EXPOSE ${MLFLOW_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${APP_PORT}/health')" || exit 1

# Default command
CMD ["python", "main.py", "api"]
