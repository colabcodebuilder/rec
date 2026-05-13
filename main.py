"""
Main entry point for the Healthcare Recommendation System.
"""
import argparse
import logging
import warnings
import numpy as np

import config
from database import seed_database
from recommendation_engine import RecommendationEngine
from evaluation import evaluate_recommendations, plot_evaluation
from explainability import RecommendationExplainer
from mlflow_tracking import MLflowManager, train_and_log_model

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_interactive():
    """Run interactive recommendation CLI."""
    engine = RecommendationEngine()
    logger.info("Healthcare Recommendation System (KNN Collaborative Filtering)")
    logger.info("-" * 60)

    while True:
        user_input = input("\nEnter Patient ID (e.g., Patient_010) or 'quit': ").strip()
        if user_input.lower() == "quit":
            break

        results = engine.get_recommendations(user_input)
        if results:
            p_conds = engine.get_patient_conditions(user_input)
            print(f"\nResults for {user_input} (Diagnoses: {', '.join(sorted(p_conds))})")
            print("-" * 85)
            for res in results:
                print(
                    f"Provider: {res['provider_id']} | "
                    f"{res['speciality']:<18} | "
                    f"Treats: {res['matched_conditions']} | "
                    f"Score: {res['score']:.4f}"
                )
        else:
            print("Patient not found or no matching specialists available.")


def run_evaluation():
    """Run evaluation metrics and generate plots."""
    engine = RecommendationEngine()
    eval_results = evaluate_recommendations(engine)

    # Log to MLflow
    mlflow_mgr = MLflowManager()
    mlflow_mgr.log_dataset_info(
        n_patients=len(engine.patients_df),
        n_providers=len(engine.providers_df),
        n_conditions=len(config.CONDITIONS)
    )
    mlflow_mgr.log_evaluation_metrics({
        "mean_ndcg@5": eval_results["mean_ndcg"]
    })

    plot_evaluation(eval_results)


def run_explainability_demo(log_to_mlflow: bool = True):
    """Demonstrate SHAP explainability with mock data."""
    # Train a model and log to MLflow
    X = np.random.rand(200, len(config.FEATURE_NAMES))
    weights = np.array([0.5, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1])
    y = X @ weights + np.random.randn(200) * 0.05

    model, run_id, test_rmse = train_and_log_model(
        X=X,
        y=y,
        feature_names=config.FEATURE_NAMES,
        params={
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1
        }
    )
    logger.info(f"Model trained and logged to MLflow. Run ID: {run_id}")

    # Create explainer with trained model
    explainer = RecommendationExplainer(model=model)

    mock_patient = {
        "patient_id": "P-100",
        "condition_tags": "migraine",
        "risk_score": 4.5,
        "urgency_score": 3.0
    }
    mock_provider = {
        "provider_id": "D-022",
        "conditions_treated": "migraine",
        "expertise_score": 4.8,
        "success_rate": 0.9,
        "availability_score": 0.7,
        "load_score": 0.2
    }

    explanation = explainer.explain(mock_patient, mock_provider, debug=True)
    print("\n--- Feature Impact Table ---")
    print(explanation.to_string(index=False))
    explainer.plot_summary(explanation)

    # Log explanation to MLflow
    if log_to_mlflow:
        mlflow_mgr = MLflowManager()
        mlflow_mgr.log_recommendation_explanation(
            "P-100",
            [{"provider_id": "D-022", "score": 0.85, "speciality": "Neurologist"}],
            shap_explanation=explanation
        )
        logger.info("Explanation logged to MLflow")


def run_mlflow_ui():
    """Launch MLflow UI server."""
    import subprocess
    import os

    logger.info("Starting MLflow UI on http://localhost:5000")
    subprocess.run([
        "mlflow", "ui",
        "--backend-store-uri", f"sqlite:///{os.path.join('data', 'mlflow.db')}",
        "--host", "0.0.0.0",
        "--port", "5000"
    ])


def run_train_model():
    """Train and register a model in MLflow."""
    X = np.random.rand(500, len(config.FEATURE_NAMES))
    weights = np.array([0.5, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1])
    y = X @ weights + np.random.randn(500) * 0.05

    model, run_id, test_rmse = train_and_log_model(
        X=X,
        y=y,
        feature_names=config.FEATURE_NAMES,
        params={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9
        }
    )

    # Register model to Staging
    mlflow_mgr = MLflowManager()
    mlflow_mgr.register_model("healthcare_scoring_model")

    logger.info(f"Training complete. Run ID: {run_id}, Test RMSE: {test_rmse:.4f}")
    logger.info("Model registered to Staging stage")


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare Recommendation System with MLflow"
    )
    parser.add_argument(
        "mode",
        choices=[
            "seed", "interactive", "evaluate", "explain",
            "api", "mlflow-ui", "train", "register"
        ],
        help="Operation mode"
    )

    args = parser.parse_args()

    if args.mode == "seed":
        seed_database()

    elif args.mode == "interactive":
        run_interactive()

    elif args.mode == "evaluate":
        run_evaluation()

    elif args.mode == "explain":
        run_explainability_demo()

    elif args.mode == "api":
        seed_database()
        from api import start_server
        start_server()

    elif args.mode == "mlflow-ui":
        run_mlflow_ui()

    elif args.mode == "train":
        run_train_model()

    elif args.mode == "register":
        mlflow_mgr = MLflowManager()
        mlflow_mgr.register_model()
        logger.info("Latest model")