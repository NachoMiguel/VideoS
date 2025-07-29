#!/usr/bin/env python3
"""
CUDA Path Fix - Must be imported BEFORE any ONNX Runtime imports
"""

import os
import sys
import site
import logging

logger = logging.getLogger(__name__)

def fix_cuda_path():
    """Add NVIDIA library paths to system PATH before any imports."""
    try:
        # Get site-packages directory
        site_packages = site.getsitepackages()[0]
        
        # NVIDIA library paths
        nvidia_cudnn_path = os.path.join(site_packages, "nvidia", "cudnn", "bin")
        nvidia_cublas_path = os.path.join(site_packages, "nvidia", "cublas", "bin")
        
        # Add to PATH if they exist and aren't already there
        paths_added = []
        
        if os.path.exists(nvidia_cudnn_path):
            if nvidia_cudnn_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = nvidia_cudnn_path + os.pathsep + os.environ.get('PATH', '')
                paths_added.append(nvidia_cudnn_path)
                logger.info(f"Added cuDNN path: {nvidia_cudnn_path}")
        
        if os.path.exists(nvidia_cublas_path):
            if nvidia_cublas_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = nvidia_cublas_path + os.pathsep + os.environ.get('PATH', '')
                paths_added.append(nvidia_cublas_path)
                logger.info(f"Added cuBLAS path: {nvidia_cublas_path}")
        
        if paths_added:
            logger.info(f"✅ CUDA paths added to environment: {paths_added}")
            return True
        else:
            logger.info("✅ CUDA paths already in environment")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to fix CUDA path: {e}")
        return False

# Auto-execute when imported
if __name__ != "__main__":
    fix_cuda_path() 