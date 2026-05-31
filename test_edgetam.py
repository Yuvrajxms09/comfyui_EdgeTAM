#!/usr/bin/env python3
"""
Test script for EdgeTAM ComfyUI nodes
Verifies that EdgeTAM is properly installed and can be imported.
"""

import sys
import os
import traceback
from pathlib import Path


def _prefer_local_edgetam_sam2():
    repo_path = Path(__file__).resolve().parent / "EdgeTAM"
    if repo_path.exists():
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

        loaded = sys.modules.get("sam2")
        loaded_file = os.path.abspath(getattr(loaded, "__file__", "") or "") if loaded else ""
        if loaded and loaded_file and repo_path_str not in loaded_file:
            print(f"Removing preloaded sam2 from {loaded_file}")
            for name in [key for key in list(sys.modules.keys()) if key == "sam2" or key.startswith("sam2.")]:
                del sys.modules[name]

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError:
        print("✗ Pillow not found")
        return False
    
    return True

def test_edgetam():
    """Test EdgeTAM imports."""
    print("\nTesting EdgeTAM...")
    
    try:
        _prefer_local_edgetam_sam2()
        import sam2
        print("✓ SAM2 package found")
        print(f"  sam2 resolved to: {getattr(sam2, '__file__', '<unknown>')}")
    except ImportError:
        print("✗ SAM2 package not found")
        print("  Please install EdgeTAM:")
        print("  git clone https://github.com/facebookresearch/EdgeTAM.git")
        print("  cd EdgeTAM && python -m pip install --break-system-packages -e .")
        return False
    
    try:
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        print("✓ EdgeTAM build functions")
    except ImportError as e:
        print(f"✗ EdgeTAM build functions: {e}")
        return False
    
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✓ EdgeTAM image predictor")
    except ImportError as e:
        print(f"✗ EdgeTAM image predictor: {e}")
        return False
    
    return True

def test_node_imports():
    """Test ComfyUI node imports."""
    print("\nTesting node imports...")
    
    try:
        from edgetam_utils import get_model_path, get_config_path
        print("✓ EdgeTAM utilities")
    except ImportError as e:
        print(f"✗ EdgeTAM utilities: {e}")
        return False
    
    try:
        from edgetam_nodes import EdgeTAMVideoTracker, EdgeTAMSelectedPersonBridge, InteractiveMaskEditor
        print("✓ EdgeTAM nodes")
    except ImportError as e:
        print(f"✗ EdgeTAM nodes: {e}")
        return False
    
    return True

def test_model_download():
    """Test model download functionality."""
    print("\nTesting model download...")
    
    try:
        from edgetam_utils import get_model_path
        model_path = get_model_path()
        
        if os.path.exists(model_path):
            print(f"✓ Model found at: {model_path}")
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"  Size: {file_size:.1f} MB")
        else:
            print(f"✗ Model not found at: {model_path}")
            print("  Model will be downloaded on first use")
        expected_space_model = Path(__file__).resolve().parent / "models" / "edgetam.pt"
        print(f"  Expected model location: {expected_space_model}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model download test failed: {e}")
        return False

def test_config():
    """Test configuration file."""
    print("\nTesting configuration...")
    
    try:
        config_path = Path(__file__).resolve().parent / "EdgeTAM" / "sam2" / "configs" / "edgetam.yaml"
        
        if os.path.exists(config_path):
            print(f"✓ Config found at: {config_path}")
        else:
            print(f"✗ Config not found at: {config_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_device():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from edgetam_utils import get_device
        device = get_device()
        print(f"✓ Detected device: {device}")
        
        import torch
        if device == "cuda" and torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name()}")
        elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  Apple Silicon MPS available")
        elif device == "cpu":
            print("  Using CPU")
        
        return True
        
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("EdgeTAM ComfyUI Node Test Suite")
    print("=" * 40)
    print(f"Python interpreter: {sys.executable}")
    print(f"Repo root: {Path(__file__).resolve().parent}")
    
    tests = [
        test_imports,
        test_edgetam,
        test_node_imports,
        test_model_download,
        test_config,
        test_device,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! EdgeTAM is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
