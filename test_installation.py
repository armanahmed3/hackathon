"""
Test script to verify that all required packages are installed correctly
"""

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import fastapi
        print("✓ FastAPI imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FastAPI: {e}")
        return False
    
    try:
        import uvicorn
        print("✓ Uvicorn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Uvicorn: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Scikit-learn: {e}")
        return False
    
    try:
        import tensorflow
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import PIL
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pillow: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import joblib
        print("✓ Joblib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Joblib: {e}")
        return False
    
    try:
        import json
        print("✓ JSON imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import JSON: {e}")
        return False
    
    return True

def test_tensorflow_gpu():
    """Test if TensorFlow can access GPU (if available)"""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check if GPU is available
        if tf.config.list_physical_devices('GPU'):
            print("✓ GPU is available for TensorFlow")
        else:
            print("⚠ GPU is not available for TensorFlow (using CPU)")
            
        return True
    except Exception as e:
        print(f"✗ Error testing TensorFlow: {e}")
        return False

def test_sklearn_version():
    """Test scikit-learn version"""
    try:
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
        return True
    except Exception as e:
        print(f"✗ Error testing scikit-learn: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Teachable Machine Installation")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n✗ Some imports failed. Please check your installation.")
        return False
    
    print()
    
    # Test TensorFlow
    if not test_tensorflow_gpu():
        print("\n✗ TensorFlow test failed.")
        return False
    
    print()
    
    # Test scikit-learn
    if not test_sklearn_version():
        print("\n✗ Scikit-learn test failed.")
        return False
    
    print("\n" + "=" * 40)
    print("✓ All tests passed! Installation is ready.")
    return True

if __name__ == "__main__":
    main()