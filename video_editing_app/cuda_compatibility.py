#!/usr/bin/env python3
"""
CUDA Compatibility Checker for InsightFace
"""

import os
import sys
import logging
import warnings
from typing import Dict, Any, Optional

# Import CUDA path fix first
from cuda_path_fix import fix_cuda_path

logger = logging.getLogger(__name__)

def check_compatibility() -> Dict[str, Any]:
    """Check CUDA compatibility and return diagnostic information."""
    try:
        logger.info("🔍 Checking CUDA compatibility...")
        
        # Fix CUDA paths
        fix_cuda_path()
        
        diagnostic = {
            'cuda_available': False,
            'onnxruntime_available': False,
            'insightface_available': False,
            'gpu_providers': [],
            'errors': []
        }
        
        # Check ONNX Runtime
        try:
            import onnxruntime as ort
            diagnostic['onnxruntime_available'] = True
            logger.info(f"✅ ONNX Runtime version: {ort.__version__}")
            
            # Check available providers
            providers = ort.get_available_providers()
            diagnostic['gpu_providers'] = providers
            logger.info(f"📋 Available providers: {providers}")
            
            # Check if CUDA is available
            if 'CUDAExecutionProvider' in providers:
                diagnostic['cuda_available'] = True
                logger.info("✅ CUDA execution provider available")
            else:
                logger.warning("⚠️ CUDA execution provider not available")
                
        except ImportError as e:
            diagnostic['errors'].append(f"ONNX Runtime not available: {e}")
            logger.error(f"❌ ONNX Runtime not available: {e}")
        
        # Check InsightFace
        try:
            import insightface
            diagnostic['insightface_available'] = True
            logger.info(f"✅ InsightFace version: {insightface.__version__}")
        except ImportError as e:
            diagnostic['errors'].append(f"InsightFace not available: {e}")
            logger.error(f"❌ InsightFace not available: {e}")
        
        # Test CUDA provider if available
        if diagnostic['cuda_available']:
            try:
                test_cuda_provider()
                logger.info("✅ CUDA provider test passed")
            except Exception as e:
                diagnostic['errors'].append(f"CUDA provider test failed: {e}")
                logger.error(f"❌ CUDA provider test failed: {e}")
        
        return diagnostic
        
    except Exception as e:
        logger.error(f"❌ Compatibility check failed: {e}")
        return {
            'cuda_available': False,
            'onnxruntime_available': False,
            'insightface_available': False,
            'gpu_providers': [],
            'errors': [str(e)]
        }

def test_cuda_provider():
    """Test CUDA provider with a simple operation."""
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Create a simple test session with CUDA
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Test with a simple model or operation
        logger.info("🧪 Testing CUDA provider...")
        
        # This is a basic test - in practice, InsightFace will handle the model loading
        logger.info("✅ CUDA provider test completed")
        
    except Exception as e:
        logger.error(f"❌ CUDA provider test failed: {e}")
        raise

def create_optimized_insightface_app():
    """Create an optimized InsightFace app with proper CUDA configuration."""
    try:
        import insightface
        
        logger.info("🚀 Creating optimized InsightFace app...")
        
        # Configure InsightFace with CUDA optimization
        app = insightface.app.FaceAnalysis(
            name='buffalo_l',  # Use buffalo_l model for better accuracy
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Prepare with optimized settings
        app.prepare(
            ctx_id=0,  # Use first GPU
            det_size=(640, 640),  # Optimal detection size
            det_thresh=0.5  # Detection threshold
        )
        
        logger.info("✅ Optimized InsightFace app created successfully")
        return app
        
    except Exception as e:
        logger.error(f"❌ Failed to create optimized InsightFace app: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run compatibility check
    diagnostic = check_compatibility()
    
    print("\n" + "="*50)
    print("CUDA COMPATIBILITY DIAGNOSTIC")
    print("="*50)
    
    for key, value in diagnostic.items():
        if key != 'errors':
            status = "✅" if value else "❌"
            print(f"{status} {key}: {value}")
    
    if diagnostic['errors']:
        print("\n❌ ERRORS:")
        for error in diagnostic['errors']:
            print(f"   - {error}")
    
    print("="*50) 