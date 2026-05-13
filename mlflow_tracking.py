"""
MLflow experiment tracking and model registry for Healthcare Recommendation System.
"""
import os
import logging
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# MLflow configuration – use values from config for consistency
# ------------------------------------------------------------------
MLFLOW_TRACKING_URI = config.MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME = config.MLFLOW_EXPERIMENT_NAME

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class MLflowManager:
    """
    Manages MLflow experiment tracking, model logging, and registry operations.
    """

    def __init__(self):
        """Initialize MLflow manager and ensure experiment exists."""
        self.experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                tags={"project": "healthcare-recsys", "version": "1.0.0"}
            )
        else:
            self.experiment_id = self.experiment.experiment_id

    def log_dataset_info(
        self,
        n_patients: int,
        n_providers: int,
        n_conditions: int,
        dataset_hash: Optional[str] = None
    ) -> None:
        """
        Log dataset metadata as MLflow parameters.
        """
        mlflow.log_params({
            "n_patients": n_patients,
            "n_providers": n_providers,
            "n_conditions": n_conditions,
            "dataset_timestamp": datetime.now().isoformat()
        })
        if dataset_hash:
            mlflow.log_param("dataset_hash", dataset_hash)

    def log_model_training(
        self,
        model: GradientBoostingRegressor,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: Dict[str, Any],
        feature_names: List[str]
    ) -> Tuple[str, float]:
        """
        Log model training run with metrics, parameters, and model artifact.

        Args:
            model: Trained scikit-learn model
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            params: Model hyperparameters
            feature_names: Names of features

        Returns:
            Tuple of (run_id, test_score)
        """
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("feature_names", feature_names)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # Log metrics
            mlflow.log_metrics({
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2
            })

            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                for name, importance in importance_dict.items():
                    mlflow.log_metric(f"feature_importance_{name}", importance)

            # Log model
            signature = mlflow.models.infer_signature(
                X_train[:5],
                model.predict(X_train[:5])
            )
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_train[:3],
                registered_model_name="healthcare_scoring_model"
            )

            # Log feature names as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write('\n'.join(feature_names))
                mlflow.log_artifact(f.name, "features")

            logger.info(
                f"MLflow run completed: {run.info.run_id} | "
                f"Test RMSE: {test_rmse:.4f} | Test R²: {test_r2:.4f}"
            )

            return run.info.run_id, test_rmse

    def log_evaluation_metrics(self, eval_metrics: Dict[str, float]) -> None:
        """
        Log evaluation metrics (NDCG, etc.) to the current run.

        Args:
            eval_metrics: Dictionary of evaluation metric name -> value
        """
        with mlflow.start_run(experiment_id=self.experiment_id):
            mlflow.log_metrics(eval_metrics)

    def log_recommendation_explanation(
        self,
        patient_id: str,
        recommendations: List[Dict[str, Any]],
        shap_explanation: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Log individual recommendation explanations as artifacts.

        Args:
            patient_id: Patient identifier
            recommendations: List of recommendation dictionaries
            shap_explanation: SHAP explanation DataFrame if available
        """
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"explanation_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            mlflow.log_param("patient_id", patient_id)
            mlflow.log_param("n_recommendations", len(recommendations))

            # Log recommendations as artifact
            recs_df = pd.DataFrame(recommendations)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                recs_df.to_csv(f.name, index=False)
                mlflow.log_artifact(f.name, "recommendations")

            # Log SHAP explanation
            if shap_explanation is not None:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    shap_explanation.to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, "explanations")

            # Log top scores
            if recommendations:
                top_scores = {f"top_{i+1}_score": rec["score"] for i, rec in enumerate(recommendations[:5])}
                mlflow.log_metrics(top_scores)

    def register_model(self, model_name: str = "healthcare_scoring_model") -> None:
        """
        Register the latest model version to a stage.

        Args:
            model_name: Name of the registered model
        """
        client = mlflow.tracking.MlflowClient()

        # Get latest version
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            latest_version = latest_versions[0]
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Staging"
            )
            logger.info(f"Model {model_name} v{latest_version.version} promoted to Staging")

    def load_model(self, model_name: str = "healthcare_scoring_model", stage: str = "Production") -> Any:
        """
        Load a registered model from MLflow registry.

        Args:
            model_name: Name of the registered model
            stage: Model stage (Production, Staging, Archived)

        Returns:
            Loaded model object
        """
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model: {model_name} (stage: {stage})")
            return model
        except Exception as e:
            logger.warning(f"Could not load model {model_name}/{stage}: {e}")
            return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.

        Returns:
            List of experiment dictionaries
        """
        experiments = mlflow.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            }
            for exp in experiments
        ]

    def get_best_run(self, metric: str = "test_rmse", ascending: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to optimize
            ascending: True for minimize (RMSE), False for maximize (R²)

        Returns:
            Best run information dictionary or None
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        if not runs.empty:
            return runs.iloc[0].to_dict()
        return None

    # ----------------------------------------------------------------
    #  N E W   M E T H O D   –  search_runs
    # ----------------------------------------------------------------
    def search_runs(self, max_results: int = 10) -> pd.DataFrame:
        """
        Search and return all runs for this experiment.

        Args:
            max_results: Maximum number of runs to return.

        Returns:
            DataFrame of runs.
        """
        return mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=max_results
        )


def train_and_log_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    params: Optional[Dict[str, Any]] = None
) -> Tuple[GradientBoostingRegressor, str, float]:
    """
    Train a GradientBoostingRegressor and log to MLflow.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: Feature names
        params: Model hyperparameters (uses defaults if None)

    Returns:
        Tuple of (trained_model, run_id, test_rmse)
    """
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": config.RANDOM_SEED
        }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED
    )

    # Train model
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    # Log to MLflow
    mlflow_mgr = MLflowManager()
    run_id, test_rmse = mlflow_mgr.log_model_training(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params=params,
        feature_names=feature_names
    )

    return model, run_id, test_rmse