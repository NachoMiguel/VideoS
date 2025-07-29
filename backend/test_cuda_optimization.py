#!/usr/bin/env python3
"""
CUDA Optimization Test Script for InsightFace
Tests CUDA detection, GPU memory, and InsightFace performance with CUDA.
"""

import logging
import time
import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cuda_detection():
    """Test CUDA detection capabilities."""
    logger.info("=" * 60)
    logger.info("TESTING CUDA DETECTION")
    logger.info("=" * 60)
    
    try:
        from core.cuda_utils import get_cuda_status, cuda_detector
        
        # Get CUDA status
        cuda_status = get_cuda_status()
        
        logger.info(f"CUDA Available: {cuda_status['cuda_available']}")
        logger.info(f"Available Providers: {cuda_status['available_providers']}")
        
        if cuda_status['gpu_info']:
            logger.info("GPU Information:")
            for device_id, info in cuda_status['gpu_info'].items():
                logger.info(f"  GPU {device_id}: {info['name']}")
                logger.info(f"    Memory: {info['free_memory'] / 1024**3:.1f}GB free / {info['total_memory'] / 1024**3:.1f}GB total")
                logger.info(f"    Utilization: {info['memory_utilization']:.1f}%")
        
        return cuda_status['cuda_available']
        
    except Exception as e:
        logger.error(f"CUDA detection failed: {e}")
        return False

def test_insightface_initialization():
    """Test InsightFace initialization with CUDA optimization."""
    logger.info("=" * 60)
    logger.info("TESTING INSIGHTFACE INITIALIZATION")
    logger.info("=" * 60)
    
    try:
        from core.cuda_utils import cuda_optimizer
        import insightface
        
        # Test optimized initialization
        logger.info("Testing CUDA-optimized InsightFace initialization...")
        start_time = time.time()
        
        app = cuda_optimizer.create_optimized_insightface_app(
            model_name="buffalo_l",
            device_id=0
        )
        
        init_time = time.time() - start_time
        logger.info(f"✅ InsightFace initialized successfully in {init_time:.2f}s")
        
        # Test basic functionality
        logger.info("Testing basic face detection...")
        
        # Create a test image (simple colored rectangle)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(test_image, (100, 100), (300, 300), (255, 255, 255), -1)
        
        start_time = time.time()
        faces = app.get(test_image)
        detection_time = time.time() - start_time
        
        logger.info(f"✅ Face detection test completed in {detection_time:.3f}s")
        logger.info(f"   Detected faces: {len(faces)}")
        
        return True
        
    except Exception as e:
        logger.error(f"InsightFace initialization failed: {e}")
        return False

def test_performance_comparison():
    """Compare CPU vs CUDA performance."""
    logger.info("=" * 60)
    logger.info("TESTING PERFORMANCE COMPARISON")
    logger.info("=" * 60)
    
    try:
        import insightface
        
        # Create test image
        test_image = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        cv2.rectangle(test_image, (200, 150), (600, 450), (255, 255, 255), -1)
        cv2.rectangle(test_image, (700, 200), (1000, 400), (200, 200, 200), -1)
        
        # Test CPU-only initialization
        logger.info("Testing CPU-only initialization...")
        start_time = time.time()
        
        cpu_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        cpu_app.prepare(ctx_id=0, det_size=(640, 640))
        
        cpu_init_time = time.time() - start_time
        logger.info(f"CPU initialization time: {cpu_init_time:.2f}s")
        
        # Test CPU detection
        start_time = time.time()
        for _ in range(5):  # Multiple runs for averaging
            cpu_faces = cpu_app.get(test_image)
        cpu_detection_time = (time.time() - start_time) / 5
        logger.info(f"CPU detection time (avg): {cpu_detection_time:.3f}s")
        
        # Test CUDA initialization (if available)
        try:
            from core.cuda_utils import cuda_optimizer
            
            logger.info("Testing CUDA initialization...")
            start_time = time.time()
            
            cuda_app = cuda_optimizer.create_optimized_insightface_app(
                model_name="buffalo_l",
                device_id=0
            )
            
            cuda_init_time = time.time() - start_time
            logger.info(f"CUDA initialization time: {cuda_init_time:.2f}s")
            
            # Test CUDA detection
            start_time = time.time()
            for _ in range(5):  # Multiple runs for averaging
                cuda_faces = cuda_app.get(test_image)
            cuda_detection_time = (time.time() - start_time) / 5
            logger.info(f"CUDA detection time (avg): {cuda_detection_time:.3f}s")
            
            # Calculate speedup
            if cpu_detection_time > 0:
                speedup = cpu_detection_time / cuda_detection_time
                logger.info(f"CUDA speedup: {speedup:.2f}x faster")
            
        except Exception as e:
            logger.warning(f"CUDA performance test failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance comparison failed: {e}")
        return False

def test_memory_optimization():
    """Test GPU memory optimization."""
    logger.info("=" * 60)
    logger.info("TESTING MEMORY OPTIMIZATION")
    logger.info("=" * 60)
    
    try:
        from core.cuda_utils import cuda_detector, cuda_optimizer
        
        # Get memory info
        if cuda_detector.gpu_info:
            for device_id, info in cuda_detector.gpu_info.items():
                logger.info(f"GPU {device_id} memory status:")
                logger.info(f"  Free: {info['free_memory'] / 1024**3:.1f}GB")
                logger.info(f"  Used: {info['used_memory'] / 1024**3:.1f}GB")
                logger.info(f"  Total: {info['total_memory'] / 1024**3:.1f}GB")
                
                # Test memory availability
                has_2gb = cuda_detector.check_gpu_memory_availability(device_id, 2.0)
                has_4gb = cuda_detector.check_gpu_memory_availability(device_id, 4.0)
                
                logger.info(f"  Has 2GB available: {has_2gb}")
                logger.info(f"  Has 4GB available: {has_4gb}")
                
                # Test optimal batch size
                optimal_batch = cuda_detector.get_optimal_batch_size(device_id, 10)
                logger.info(f"  Optimal batch size: {optimal_batch}")
        
        # Test optimized configuration
        config = cuda_optimizer.get_optimized_config(device_id=0)
        logger.info(f"Optimized configuration:")
        logger.info(f"  Batch size: {config.get('batch_size', 'N/A')}")
        logger.info(f"  Max workers: {config.get('max_workers', 'N/A')}")
        logger.info(f"  GPU memory fraction: {config.get('gpu_memory_fraction', 'N/A')}")
        
        if 'cuda_provider_options' in config:
            logger.info(f"  CUDA provider options: {config['cuda_provider_options']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Memory optimization test failed: {e}")
        return False

def main():
    """Run all CUDA optimization tests."""
    logger.info("Starting CUDA Optimization Tests for InsightFace")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: CUDA Detection
    results['cuda_detection'] = test_cuda_detection()
    
    # Test 2: InsightFace Initialization
    results['insightface_init'] = test_insightface_initialization()
    
    # Test 3: Performance Comparison
    results['performance'] = test_performance_comparison()
    
    # Test 4: Memory Optimization
    results['memory_optimization'] = test_memory_optimization()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - CUDA optimization is working!")
    else:
        logger.info("⚠️  Some tests failed - check the logs above for details")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 