"""
ONNX model export and optimization for healthcare recommendation system.
"""
import os
import logging
import tempfile
from typing import Optional, Dict, Any, Tuple
import numpy as np

import mlflow
import mlflow.pyfunc
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxconverter_common import FloatType
from sklearn.ensemble import GradientBoostingRegressor

import config

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Export MLflow models to ONNX format for optimized inference.
    Supports fallback to creating a model from scratch if no MLflow runs exist.
    """

    def __init__(self, output_dir: str = "onnx_models"):
        """
        Initialize ONNX exporter.

        Args:
            output_dir: Directory to save ONNX models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"ONNX exporter initialized. Output directory: {output_dir}")

    def export_best_model(
        self,
        model_name: str = "healthcare_scoring_model",
        stage: str = "Staging",
        metric: str = "test_rmse",
        optimize: bool = True,
        create_fallback: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Find the best MLflow model and export to ONNX.
        Falls back to creating a model from scratch if no MLflow runs exist.

        Args:
            model_name: MLflow registered model name
            stage: Model stage to search
            metric: Metric to determine best model
            optimize: Apply ONNX optimization
            create_fallback: Create a model from scratch if no MLflow runs found

        Returns:
            Tuple of (onnx_path, model_info)
        """
        logger.info(f"Finding best model: {model_name} (stage: {stage})")
        sklearn_model = None
        run_id = None
        metric_value = None

        # Try to get model from MLflow
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)

            if experiment is not None:
                # Search for best run
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=[f"metrics.{metric} ASC"],
                    max_results=1
                )

                if not runs.empty:
                    best_run = runs.iloc[0]
                    run_id = best_run["run_id"]
                    metric_value = float(best_run[f"metrics.{metric}"])
                    logger.info(f"Best run found: {run_id} | {metric}: {metric_value:.4f}")

                    # Load the scikit-learn model
                    model_uri = f"runs:/{run_id}/model"
                    logger.info(f"Loading model from: {model_uri}")
                    mlflow_model = mlflow.pyfunc.load_model(model_uri)
                    sklearn_model = self._extract_sklearn_model(mlflow_model)
                    logger.info(f"Extracted sklearn model: {type(sklearn_model).__name__}")
                else:
                    logger.warning("No runs found in MLflow experiment")
            else:
                logger.warning(f"Experiment '{config.MLFLOW_EXPERIMENT_NAME}' not found in MLflow")
        except Exception as e:
            logger.warning(f"Could not load model from MLflow: {e}")

        # Fallback: Create a model from scratch
        if sklearn_model is None:
            if create_fallback:
                logger.info("Creating fallback model from scratch...")
                sklearn_model, run_id, metric_value = self._create_fallback_model()
            else:
                error_msg = "No MLflow model found and fallback creation is disabled"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Convert to ONNX
        onnx_path = self._convert_to_onnx(
            sklearn_model,
            model_name,
            run_id or "fallback",
            optimize
        )

        # Verify ONNX model
        self._verify_onnx_model(onnx_path)

        # Create model info
        model_info = {
            "run_id": run_id or "fallback",
            "model_name": model_name,
            "stage": stage,
            "metric": metric,
            "metric_value": metric_value or 0.0,
            "onnx_path": onnx_path,
            "input_features": config.FEATURE_NAMES,
            "input_shape": [1, len(config.FEATURE_NAMES)],
            "is_fallback": sklearn_model is not None and run_id == "fallback"
        }

        # Save model info
        info_path = onnx_path.replace(".onnx", "_info.json")
        import json
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2, default=str)
        logger.info(f"Model info saved to: {info_path}")

        logger.info(f"ONNX model exported successfully to: {onnx_path}")
        logger.info(f"  - Run ID: {run_id or 'fallback'}")
        logger.info(f"  - Metric: {metric} = {model_info['metric_value']:.4f}")
        logger.info(f"  - Features: {config.FEATURE_NAMES}")
        logger.info(f"  - Fallback: {model_info['is_fallback']}")

        return onnx_path, model_info

    def _create_fallback_model(self) -> Tuple[Any, str, float]:
        """
        Create a GradientBoostingRegressor model from scratch for ONNX export.

        Returns:
            Tuple of (model, run_id, metric_value)
        """
        logger.info("Generating synthetic training data...")
        X = np.random.rand(500, len(config.FEATURE_NAMES))
        weights = np.array([0.5, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1])
        y = X @ weights + np.random.randn(500) * 0.05

        logger.info("Training GradientBoostingRegressor...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=config.RANDOM_SEED
        )
        model.fit(X, y)

        # Calculate training metric
        from sklearn.metrics import mean_squared_error
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        logger.info(f"Fallback model trained. Train RMSE: {rmse:.4f}")

        return model, "fallback", rmse

    def _extract_sklearn_model(self, mlflow_model) -> Any:
        """
        Extract the underlying sklearn model from MLflow pyfunc model.
        """
        # Try to get the sklearn model directly
        if hasattr(mlflow_model, '_model_impl'):
            model_impl = mlflow_model._model_impl
            if hasattr(model_impl, 'python_model'):
                logger.debug("Extracted model via _model_impl.python_model")
                return model_impl.python_model
            logger.debug("Extracted model via _model_impl")
            return model_impl

        # Try unwrapping
        if hasattr(mlflow_model, 'unwrap_python_model'):
            logger.debug("Extracted model via unwrap_python_model")
            return mlflow_model.unwrap_python_model()

        error_msg = "Cannot extract sklearn model from MLflow pyfunc model"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _convert_to_onnx(
        self,
        sklearn_model,
        model_name: str,
        run_id: str,
        optimize: bool = True
    ) -> str:
        """
        Convert sklearn model to ONNX format.

        Args:
            sklearn_model: Trained sklearn model
            model_name: Name for the ONNX model
            run_id: MLflow run ID or fallback identifier
            optimize: Apply optimization

        Returns:
            Path to saved ONNX model
        """
        # Define input type
        initial_type = [(
            "float_input",
            FloatTensorType([None, len(config.FEATURE_NAMES)])
        )]

        # Convert to ONNX
        logger.info("Converting sklearn model to ONNX format...")
        logger.debug(f"Input shape: [None, {len(config.FEATURE_NAMES)}]")
        logger.debug(f"Target opset: 12")

        onnx_model = convert_sklearn(
            sklearn_model,
            initial_types=initial_type,
            target_opset=12
        )
        logger.info("ONNX conversion completed successfully")

        # Optimize if requested
        if optimize:
            logger.info("Optimizing ONNX model...")
            try:
                onnx_model = self._optimize_onnx(onnx_model)
                logger.info("ONNX optimization completed")
            except Exception as e:
                logger.warning(f"ONNX optimization failed: {e}, using unoptimized model")

        # Save model
        model_filename = f"{model_name}_{run_id[:8]}.onnx"
        onnx_path = os.path.join(self.output_dir, model_filename)

        onnx.save_model(onnx_model, onnx_path)
        logger.info(f"ONNX model saved: {onnx_path}")

        # Get model size
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.2f} MB")

        return onnx_path

    def _optimize_onnx(self, onnx_model) -> Any:
        """
        Optimize ONNX model for inference.
        """
        try:
            from onnx import optimizer
            passes = [
                "eliminate_deadend",
                "eliminate_identity",
                "eliminate_nop_dropout",
                "eliminate_nop_pad",
                "eliminate_nop_transpose",
                "eliminate_unused_initializer",
                "extract_constant_to_initializer",
                "fuse_add_bias_into_conv",
                "fuse_bn_into_conv",
                "fuse_consecutive_concats",
                "fuse_consecutive_log_softmax",
                "fuse_consecutive_reduce_unsqueeze",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_matmul_add_bias_into_gemm",
                "fuse_pad_into_conv",
                "fuse_transpose_into_gemm"
            ]

            logger.debug(f"Applying {len(passes)} optimization passes")
            optimized_model = optimizer.optimize(onnx_model, passes)
            logger.info("ONNX optimization passes applied successfully")
            return optimized_model
        except ImportError:
            logger.warning("ONNX optimizer not available, skipping optimization")
            return onnx_model
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
            return onnx_model

    def _verify_onnx_model(self, onnx_path: str) -> None:
        """
        Verify ONNX model with a test inference.

        Args:
            onnx_path: Path to ONNX model
        """
        # Check model validity
        logger.info("Verifying ONNX model...")
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure verified successfully")
        except Exception as e:
            logger.warning(f"ONNX model check failed (non-critical): {e}")

        # Test inference
        try:
            session = ort.InferenceSession(onnx_path)
            test_input = np.random.randn(1, len(config.FEATURE_NAMES)).astype(np.float32)
            outputs = session.run(None, {"float_input": test_input})

            logger.info("Test inference successful")
            logger.info(f"  - Output shape: {outputs[0].shape}")
            logger.info(f"  - Output value: {outputs[0][0]:.4f}")
        except Exception as e:
            logger.error(f"Test inference failed: {e}")
            raise

    def export_with_quantization(
        self,
        model_name: str = "healthcare_scoring_model",
        stage: str = "Staging"
    ) -> Tuple[str, str]:
        """
        Export both full and quantized ONNX models.

        Args:
            model_name: MLflow model name
            stage: Model stage

        Returns:
            Tuple of (full_model_path, quantized_model_path)
        """
        # Export full model
        logger.info("Exporting full precision model...")
        full_path, model_info = self.export_best_model(
            model_name=model_name,
            stage=stage,
            optimize=True
        )

        # Create quantized version
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = full_path.replace(".onnx", "_quantized.onnx")
            logger.info("Creating quantized ONNX model (INT8)...")

            quantize_dynamic(
                model_input=full_path,
                model_output=quantized_path,
                weight_type=QuantType.QUInt8
            )

            # Log sizes
            full_size = os.path.getsize(full_path) / (1024 * 1024)
            quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
            reduction = (1 - quant_size / full_size) * 100

            logger.info("Quantization complete:")
            logger.info(f"  - Full model: {full_size:.2f} MB")
            logger.info(f"  - Quantized model: {quant_size:.2f} MB")
            logger.info(f"  - Size reduction: {reduction:.1f}%")

            return full_path, quantized_path

        except ImportError:
            logger.warning("ONNX Runtime quantization not available, returning full model only")
            return full_path, full_path


class ONNXPredictor:
    """
    Fast inference using ONNX Runtime.
    """

    def __init__(self, onnx_path: str):
        """
        Initialize ONNX predictor.

        Args:
            onnx_path: Path to ONNX model file
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.onnx_path = onnx_path
        logger.info(f"Initializing ONNX predictor from: {onnx_path}")

        # Set session options for better performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(onnx_path, sess_options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get model metadata
        try:
            onnx_model = onnx.load(onnx_path)
            self.metadata = {
                "producer": onnx_model.producer_name,
                "version": onnx_model.producer_version,
                "opset": onnx_model.opset_import[0].version if onnx_model.opset_import else "unknown"
            }
        except Exception:
            self.metadata = {
                "producer": "unknown",
                "version": "unknown",
                "opset": "unknown"
            }

        logger.info("ONNX predictor initialized successfully")
        logger.info(f"  - Input: {self.input_name}")
        logger.info(f"  - Output: {self.output_name}")
        logger.info(f"  - Opset: {self.metadata['opset']}")
        logger.info(f"  - Producer: {self.metadata['producer']} v{self.metadata['version']}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on input features.

        Args:
            features: Input features array

        Returns:
            Prediction array
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features = features.astype(np.float32)
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: features}
        )
        return outputs[0]

    def predict_batch(self, features_list: list) -> np.ndarray:
        """
        Run batch inference.

        Args:
            features_list: List of feature arrays

        Returns:
            Batch predictions
        """
        features = np.array(features_list, dtype=np.float32)
        logger.debug(f"Running batch prediction on {len(features_list)} samples")
        return self.predict(features)

    def benchmark(self, n_iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark ONNX inference speed.

        Args:
            n_iterations: Number of inference iterations

        Returns:
            Dictionary with timing metrics
        """
        import time

        logger.info(f"Running benchmark with {n_iterations} iterations...")

        test_input = np.random.randn(1, len(config.FEATURE_NAMES)).astype(np.float32)

        # Warmup
        logger.debug("Warming up (10 iterations)...")
        for _ in range(10):
            self.predict(test_input)

        # Benchmark
        logger.debug(f"Running benchmark ({n_iterations} iterations)...")
        start = time.time()
        for _ in range(n_iterations):
            self.predict(test_input)
        total_time = time.time() - start

        results = {
            "n_iterations": n_iterations,
            "total_time_seconds": round(total_time, 4),
            "avg_time_ms": round((total_time / n_iterations) * 1000, 4),
            "inferences_per_second": round(n_iterations / total_time, 2)
        }

        logger.info("Benchmark complete:")
        logger.info(f"  - Total time: {total_time:.3f}s")
        logger.info(f"  - Avg latency: {results['avg_time_ms']:.3f} ms")
        logger.info(f"  - Throughput: {results['inferences_per_second']:.0f} inf/s")

        return results


def list_available_onnx_models(output_dir: str = "onnx_models") -> list:
    """
    List all available ONNX models in the output directory.

    Args:
        output_dir: Directory to search for ONNX models

    Returns:
        List of model filenames
    """
    if not os.path.exists(output_dir):
        return []
    return sorted([f for f in os.listdir(output_dir) if f.endswith('.onnx')])


def main():
    """Export best model to ONNX format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("ONNX Model Export Pipeline")
    logger.info("=" * 60)

    # Check existing models
    existing_models = list_available_onnx_models()
    if existing_models:
        logger.info(f"Found {len(existing_models)} existing ONNX model(s):")
        for model in existing_models:
            model_path = os.path.join("onnx_models", model)
            size_kb = os.path.getsize(model_path) / 1024
            logger.info(f"  - {model} ({size_kb:.1f} KB)")

    exporter = ONNXExporter()

    try:
        # Export best model (with fallback)
        onnx_path, model_info = exporter.export_best_model(
            model_name="healthcare_scoring_model",
            stage="Staging",
            metric="test_rmse",
            create_fallback=True  # Create fallback if no MLflow runs
        )

        logger.info("=" * 60)
        logger.info("ONNX Export Summary")
        logger.info("=" * 60)
        logger.info(f"Model path: {onnx_path}")
        logger.info(f"Run ID: {model_info['run_id']}")
        logger.info(f"Metric: {model_info['metric']} = {model_info['metric_value']:.4f}")
        logger.info(f"Features: {model_info['input_features']}")
        logger.info(f"Is fallback: {model_info['is_fallback']}")

        # Test inference
        predictor = ONNXPredictor(onnx_path)
        test_input = np.random.randn(1, len(config.FEATURE_NAMES)).astype(np.float32)
        result = predictor.predict(test_input)
        logger.info(f"Test inference result: {result[0]:.4f}")

        # Benchmark
        bench = predictor.benchmark(1000)
        logger.info(f"Benchmark: {bench['inferences_per_second']:.0f} inferences/sec")
        logger.info(f"Average latency: {bench['avg_time_ms']:.3f} ms")

        logger.info("ONNX export completed successfully!")

    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()