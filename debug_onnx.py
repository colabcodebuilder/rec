"""
Debug script to test ONNX model creation and loading.
"""
import os
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

def test_onnx_step_by_step():
    """Test each step of ONNX model creation and loading."""
    
    # Step 1: Check imports
    logger.info("=" * 50)
    logger.info("Step 1: Checking imports...")
    try:
        import onnx
        logger.info(f"✅ onnx: {onnx.__version__}")
    except ImportError as e:
        logger.error(f"❌ onnx not installed: {e}")
        return False
    
    try:
        import onnxruntime as ort
        logger.info(f"✅ onnxruntime: {ort.__version__}")
    except ImportError as e:
        logger.error(f"❌ onnxruntime not installed: {e}")
        return False
    
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        logger.info("✅ skl2onnx installed")
    except ImportError as e:
        logger.error(f"❌ skl2onnx not installed: {e}")
        return False
    
    # Step 2: Create a simple model
    logger.info("\n" + "=" * 50)
    logger.info("Step 2: Creating test sklearn model...")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = np.random.rand(100, len(config.FEATURE_NAMES))
        weights = np.array([0.5, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1])
        y = X @ weights + np.random.randn(100) * 0.05
        
        model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        logger.info(f"✅ Model trained: {type(model).__name__}")
    except Exception as e:
        logger.error(f"❌ Failed to create model: {e}")
        return False
    
    # Step 3: Convert to ONNX
    logger.info("\n" + "=" * 50)
    logger.info("Step 3: Converting to ONNX...")
    try:
        initial_type = [(
            "float_input",
            FloatTensorType([None, len(config.FEATURE_NAMES)])
        )]
        
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12
        )
        logger.info("✅ Conversion successful")
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        return False
    
    # Step 4: Save ONNX model
    logger.info("\n" + "=" * 50)
    logger.info("Step 4: Saving ONNX model...")
    os.makedirs("onnx_models", exist_ok=True)
    onnx_path = os.path.join("onnx_models", "test_debug.onnx")
    
    try:
        onnx.save_model(onnx_model, onnx_path)
        file_size = os.path.getsize(onnx_path) / 1024
        logger.info(f"✅ Model saved: {onnx_path} ({file_size:.1f} KB)")
    except Exception as e:
        logger.error(f"❌ Failed to save: {e}")
        return False
    
    # Step 5: Load and verify
    logger.info("\n" + "=" * 50)
    logger.info("Step 5: Loading ONNX model...")
    try:
        # Check file exists
        if not os.path.exists(onnx_path):
            logger.error(f"❌ File not found: {onnx_path}")
            return False
        
        # Load with onnxruntime
        session = ort.InferenceSession(onnx_path)
        logger.info(f"✅ Session created")
        logger.info(f"   Input name: {session.get_inputs()[0].name}")
        logger.info(f"   Input shape: {session.get_inputs()[0].shape}")
        logger.info(f"   Output name: {session.get_outputs()[0].name}")
    except Exception as e:
        logger.error(f"❌ Failed to load: {e}")
        return False
    
    # Step 6: Test inference
    logger.info("\n" + "=" * 50)
    logger.info("Step 6: Testing inference...")
    try:
        test_input = np.random.randn(1, len(config.FEATURE_NAMES)).astype(np.float32)
        outputs = session.run(None, {"float_input": test_input})
        
        # FIXED: Extract the scalar value properly
        output_value = float(outputs[0].flatten()[0])
        
        logger.info("✅ Inference successful")
        logger.info(f"   Output shape: {outputs[0].shape}")
        logger.info(f"   Output value: {output_value:.4f}")
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        return False
    
    # Step 7: Test ONNXPredictor class
    logger.info("\n" + "=" * 50)
    logger.info("Step 7: Testing ONNXPredictor class...")
    try:
        from onnx_export import ONNXPredictor
        
        predictor = ONNXPredictor(onnx_path)
        result = predictor.predict(test_input)
        
        # FIXED: Extract scalar value
        result_value = float(result.flatten()[0])
        
        logger.info("✅ ONNXPredictor works!")
        logger.info(f"   Result: {result_value:.4f}")
    except Exception as e:
        logger.error(f"❌ ONNXPredictor failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 50)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("=" * 50)
    return True


if __name__ == "__main__":
    success = test_onnx_step_by_step()
    
    if success:
        print("\n✅ ONNX is working correctly!")
        print("The model at onnx_models/test_debug.onnx can be loaded.")
        print("\nNow run: streamlit run streamlit_app.py")
    else:
        print("\n❌ ONNX setup has issues. Check the errors above.")