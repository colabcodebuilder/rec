"""
End-to-end training pipeline for the Healthcare Recommendation System.
Handles synthetic data generation, model training, evaluation,
MLflow logging, and optional ONNX export with hyperparameter sweeps.
"""

import logging
import sys
import os
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_generator import (
    generate_patients_df,
    generate_providers_df,
    compute_jaccard_similarity,
    tag_string_to_set
)
from recommendation_engine import build_features, fallback_score
from mlflow_tracking import MLflowManager, train_and_log_model
from onnx_export import ONNXExporter, ONNXPredictor

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("training_pipeline")

# ------------------------------------------------------------------
# Helper: generate feature matrix and target for training
# ------------------------------------------------------------------
def generate_training_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a synthetic dataset of (patient, provider) pairs
    and computes the target score using fallback_score.
    """
    patients_df = generate_patients_df()
    providers_df = generate_providers_df()

    features_list = []
    targets = []

    for _ in range(n_samples):
        p_row = patients_df.sample(1).iloc[0].to_dict()
        d_row = providers_df.sample(1).iloc[0].to_dict()
        feat = build_features(p_row, d_row)
        score = fallback_score(feat)
        features_list.append(feat)
        targets.append(score)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y

# ------------------------------------------------------------------
# Pipeline class
# ------------------------------------------------------------------
class TrainingPipeline:
    """
    Encapsulates the full training workflow.
    """

    def __init__(self, experiment_name: str = config.MLFLOW_EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        self.mlflow_mgr = MLflowManager()
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None
        self.metrics = {}

    def load_data(self, n_samples: int = 5000, test_size: float = 0.2):
        logger.info(f"Generating synthetic training data ({n_samples} samples)...")
        X, y = generate_training_data(n_samples)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_SEED
        )
        logger.info(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        return self

    def train(self, params: Optional[Dict[str, Any]] = None):
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if params is None:
            params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "random_state": config.RANDOM_SEED
            }

        logger.info("Training GradientBoostingRegressor...")
        self.model = GradientBoostingRegressor(**params)
        self.model.fit(self.X_train, self.y_train)
        logger.info("Training completed.")
        return self

    def evaluate(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        self.metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(self.y_train, y_pred_train))),
            "test_rmse": float(np.sqrt(mean_squared_error(self.y_test, y_pred_test))),
            "train_r2": float(r2_score(self.y_train, y_pred_train)),
            "test_r2": float(r2_score(self.y_test, y_pred_test))
        }
        logger.info(f"Metrics: {self.metrics}")
        return self.metrics

    def log_to_mlflow(self, params: Dict[str, Any]):
        if self.model is None:
            raise ValueError("No model to log.")
        # Use the existing train_and_log_model which does splitting again,
        # but we already split. We'll use MLflowManager directly.
        run_id, test_rmse = self.mlflow_mgr.log_model_training(
            model=self.model,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            params=params,
            feature_names=config.FEATURE_NAMES
        )
        logger.info(f"Model logged to MLflow. Run ID: {run_id}")
        return run_id

    def export_onnx(self, onnx_path: Optional[str] = None) -> str:
        if self.model is None:
            raise ValueError("No model to export.")
        exporter = ONNXExporter()
        # We can't use export_best_model because it expects MLflow, so we convert directly.
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, len(config.FEATURE_NAMES)]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type, target_opset=12)

        if onnx_path is None:
            os.makedirs("onnx_models", exist_ok=True)
            onnx_path = os.path.join("onnx_models", "healthcare_scoring_pipeline.onnx")
        import onnx
        onnx.save_model(onnx_model, onnx_path)
        logger.info(f"ONNX model exported to {onnx_path}")
        return onnx_path

    def run(self,
            n_samples: int = 5000,
            test_size: float = 0.2,
            params: Optional[Dict[str, Any]] = None,
            export_onnx: bool = True):
        self.load_data(n_samples, test_size)
        self.train(params)
        self.evaluate()
        if params is None:
            params = self.model.get_params()
        self.log_to_mlflow(params)
        if export_onnx:
            self.export_onnx()
        logger.info("Pipeline finished successfully.")
        return self.model, self.metrics


# ------------------------------------------------------------------
# Hyperparameter tuning with GridSearchCV
# ------------------------------------------------------------------
def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray,
                          param_grid: Optional[Dict] = None) -> Tuple[GradientBoostingRegressor, Dict]:
    """
    Perform grid search to find best hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 0.9]
        }
    model = GradientBoostingRegressor(random_state=config.RANDOM_SEED)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error',
                        verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV score (RMSE): {-grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Healthcare Recommendation Training Pipeline")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of synthetic training samples")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--no_onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning first")
    args = parser.parse_args()

    # 1. Generate data
    X, y = generate_training_data(args.n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=config.RANDOM_SEED
    )

    # 2. Optional tuning
    if args.tune:
        logger.info("Starting hyperparameter tuning...")
        model, best_params = hyperparameter_tuning(X_train, y_train)
    else:
        best_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "random_state": config.RANDOM_SEED
        }
        model = GradientBoostingRegressor(**best_params)
        model.fit(X_train, y_train)

    # 3. Evaluate
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    logger.info(f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")

    # 4. Log to MLflow
    mlflow_mgr = MLflowManager()
    run_id, _ = mlflow_mgr.log_model_training(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params=best_params,
        feature_names=config.FEATURE_NAMES
    )
    logger.info(f"Run logged to MLflow: {run_id}")

    # 5. Export ONNX
    if not args.no_onnx:
        exporter = ONNXExporter()
        # Convert directly
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, len(config.FEATURE_NAMES)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
        os.makedirs("onnx_models", exist_ok=True)
        onnx_path = os.path.join("onnx_models", "healthcare_pipeline_model.onnx")
        import onnx
        onnx.save_model(onnx_model, onnx_path)
        logger.info(f"ONNX model saved to {onnx_path}")

        # Verify
        predictor = ONNXPredictor(onnx_path)
        sample = X_test[:1].astype(np.float32)
        pred = predictor.predict(sample)
        logger.info(f"ONNX test prediction: {pred[0][0]:.4f}")

    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    main()