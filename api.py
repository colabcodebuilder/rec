"""
FastAPI application for the Healthcare Recommendation Engine.
"""
import sys
import logging
import socket
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
import uvicorn

import config
from database import get_connection, fetch_patient, fetch_all_providers, seed_database
from recommendation_engine import build_features, fallback_score
from explainability import build_explanation_summary
from mlflow_tracking import MLflowManager

# Logging
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("health_api")

# Initialize MLflow manager
mlflow_mgr = MLflowManager()

app = FastAPI(
    title="Healthcare Recommendation Engine",
    description="Explainable provider recommendations for patients with MLflow tracking",
    version="1.0.0"
)


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "recommend": "/recommend/{patient_id}",
            "experiments": "/mlflow/experiments",
            "best_model": "/mlflow/best-model",
            "model_info": "/mlflow/models/{model_name}"
        }
    }


@app.get("/recommend/{patient_id}")
def recommend(
    patient_id: str,
    explain: bool = Query(False, description="Include SHAP-based explanations"),
    log_to_mlflow: bool = Query(False, description="Log this recommendation to MLflow")
):
    """
    Generate top-5 provider recommendations for a given patient.
    Optionally include explainability details and log to MLflow.
    """
    conn = None
    try:
        conn = get_connection()
        patient = fetch_patient(conn, patient_id)

        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

        providers = fetch_all_providers(conn)

        results = []
        for provider in providers:
            features = build_features(patient, provider)
            score = fallback_score(features)

            item = {
                "provider_id": provider["provider_id"],
                "speciality": provider.get("speciality", "General Physician"),
                "score": round(float(score), 4)
            }

            if explain:
                p_tags = set(
                    t.strip().lower()
                    for t in str(patient.get("condition_tags", "")).split(",")
                    if t.strip()
                )
                d_tags = set(
                    t.strip().lower()
                    for t in str(provider.get("conditions_treated", "")).split(",")
                    if t.strip()
                )
                common = p_tags & d_tags
                item["explanation"] = build_explanation_summary(
                    features, common, provider.get("speciality", "Unknown")
                )

            results.append(item)

        ranked = sorted(results, key=lambda x: x["score"], reverse=True)[:config.TOP_K_RECS]

        # Log to MLflow if requested
        if log_to_mlflow:
            mlflow_mgr.log_recommendation_explanation(patient_id, ranked)

        logger.info("Generated %d recommendations for %s", len(ranked), patient_id)

        return {
            "patient_id": patient_id,
            "patient_conditions": list(
                set(t.strip().lower() for t in patient.get("condition_tags", "").split(",") if t.strip())
            ),
            "recommendations": ranked,
            "mlflow_logged": log_to_mlflow
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing recommendation: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# ==========================================
# MLflow Management Endpoints
# ==========================================

@app.get("/mlflow/experiments")
def list_experiments():
    """
    List all MLflow experiments.
    """
    try:
        experiments = mlflow_mgr.list_experiments()
        return {"experiments": experiments, "count": len(experiments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/best-model")
def get_best_model(
    metric: str = Query("test_rmse", description="Metric to optimize (test_rmse, test_r2)"),
    ascending: bool = Query(True, description="True=minimize, False=maximize")
):
    """
    Get the best model run based on a metric.
    """
    try:
        best_run = mlflow_mgr.get_best_run(metric=metric, ascending=ascending)
        if best_run:
            return {
                "best_run": {
                    "run_id": best_run.get("run_id"),
                    "metrics": {
                        k: v for k, v in best_run.items()
                        if k.startswith("metrics.")
                    },
                    "params": {
                        k: v for k, v in best_run.items()
                        if k.startswith("params.")
                    }
                }
            }
        return {"message": "No runs found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/models/{model_name}")
def get_model_info(
    model_name: str = "healthcare_scoring_model",
    stage: str = Query("Production", description="Model stage (Production, Staging, Archived)")
):
    """
    Get information about a registered model.
    """
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])

        if versions:
            version_info = []
            for v in versions:
                run = client.get_run(v.run_id)
                version_info.append({
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "metrics": run.data.metrics,
                    "params": run.data.params
                })
            return {"model_name": model_name, "versions": version_info}
        return {"message": f"No model '{model_name}' found in stage '{stage}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """
    Detailed health check including MLflow status.
    """
    mlflow_status = "healthy"
    try:
        mlflow_mgr.list_experiments()
    except Exception:
        mlflow_status = "unhealthy"

    return {
        "status": "healthy",
        "mlflow": mlflow_status,
        "mlflow_uri": mlflow_mgr.experiment.artifact_location if mlflow_mgr.experiment else None
    }


def start_server() -> None:
    """
    Start the FastAPI server directly using uvicorn.
    """
    # Get local IP for display
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "=" * 60)
    print("🚀 Healthcare Recommendation API")
    print("=" * 60)
    print(f"📍 Local:   http://localhost:{config.API_PORT}")
    print(f"📍 Network: http://{local_ip}:{config.API_PORT}")
    print(f"📖 API Docs: http://localhost:{config.API_PORT}/docs")
    print(f"💡 Example: http://localhost:{config.API_PORT}/recommend/Patient_001?explain=true")
    print(f"📊 MLflow UI: http://localhost:5000")
    print(f"🔍 MLflow API: http://localhost:{config.API_PORT}/mlflow/experiments")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    seed_database()
    start_server()