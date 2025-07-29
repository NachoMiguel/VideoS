#!/usr/bin/env python3
"""
Simple CUDA/CUDNN Status Checker
"""

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    print(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
    else:
        print("❌ CUDA is not available")
        
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")
except Exception as e:
    print(f"❌ Error checking CUDA: {e}") 