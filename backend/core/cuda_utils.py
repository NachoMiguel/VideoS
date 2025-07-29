"""
CUDA Utilities for InsightFace Optimization
Provides CUDA detection, GPU memory management, and provider optimization.
"""

import logging
import os
from typing import List, Dict, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import nvidia.ml.py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("NVML not available - GPU memory info limited")


class CUDADetector:
    """Detects and manages CUDA capabilities for InsightFace optimization."""
    
    def __init__(self):
        self._cuda_available = False
        self._gpu_info = {}
        self._available_providers = []
        self._detect_cuda_capabilities()
    
    def _detect_cuda_capabilities(self):
        """Detect CUDA capabilities and available providers."""
        try:
            # Check ONNX Runtime providers
            if ONNX_AVAILABLE:
                self._available_providers = ort.get_available_providers()
                self._cuda_available = "CUDAExecutionProvider" in self._available_providers
                logger.info(f"Available ONNX providers: {self._available_providers}")
            
            # Check PyTorch CUDA
            if TORCH_AVAILABLE:
                torch_cuda_available = torch.cuda.is_available()
                if torch_cuda_available:
                    gpu_count = torch.cuda.device_count()
                    logger.info(f"PyTorch CUDA available: {torch_cuda_available}, GPU count: {gpu_count}")
                    
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory
                        logger.info(f"GPU {i}: {gpu_name}, Memory: {gpu_memory / 1024**3:.1f}GB")
            
            # Get detailed GPU info via NVML
            if NVML_AVAILABLE:
                self._get_nvml_gpu_info()
            
            logger.info(f"CUDA detection complete - Available: {self._cuda_available}")
            
        except Exception as e:
            logger.error(f"Error detecting CUDA capabilities: {e}")
            self._cuda_available = False
    
    def _get_nvml_gpu_info(self):
        """Get detailed GPU information using NVML."""
        try:
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                self._gpu_info[i] = {
                    'name': name,
                    'total_memory': memory_info.total,
                    'free_memory': memory_info.free,
                    'used_memory': memory_info.used,
                    'memory_utilization': (memory_info.used / memory_info.total) * 100
                }
                
                logger.info(f"GPU {i}: {name}")
                logger.info(f"  Memory: {memory_info.used / 1024**3:.1f}GB used / {memory_info.total / 1024**3:.1f}GB total")
                logger.info(f"  Utilization: {self._gpu_info[i]['memory_utilization']:.1f}%")
            
            nvml.nvmlShutdown()
            
        except Exception as e:
            logger.warning(f"Could not get NVML GPU info: {e}")
    
    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available
    
    @property
    def available_providers(self) -> List[str]:
        """Get available ONNX providers."""
        return self._available_providers.copy()
    
    @property
    def gpu_info(self) -> Dict:
        """Get GPU information."""
        return self._gpu_info.copy()
    
    def get_optimal_providers(self, prefer_cuda: bool = True) -> List[str]:
        """Get optimal provider list for InsightFace."""
        if not ONNX_AVAILABLE:
            return ["CPUExecutionProvider"]
        
        if prefer_cuda and self._cuda_available:
            # CUDA with CPU fallback
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            # CPU only
            return ["CPUExecutionProvider"]
    
    def get_cuda_provider_options(self, device_id: int = 0, memory_fraction: float = 0.5) -> Dict:
        """Get CUDA provider options for optimal performance."""
        options = {
            'device_id': device_id,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB default
            'cudnn_conv_use_max_workspace': '1',
            'do_copy_in_default_stream': '1',
        }
        
        # Adjust memory limit based on available GPU memory
        if device_id in self._gpu_info:
            total_memory = self._gpu_info[device_id]['total_memory']
            memory_limit = int(total_memory * memory_fraction)
            options['gpu_mem_limit'] = memory_limit
            logger.info(f"Set GPU memory limit to {memory_limit / 1024**3:.1f}GB for device {device_id}")
        
        return options
    
    def check_gpu_memory_availability(self, device_id: int = 0, required_memory_gb: float = 1.0) -> bool:
        """Check if enough GPU memory is available."""
        if device_id not in self._gpu_info:
            return False
        
        free_memory_gb = self._gpu_info[device_id]['free_memory'] / (1024**3)
        return free_memory_gb >= required_memory_gb
    
    def get_optimal_batch_size(self, device_id: int = 0, base_batch_size: int = 10) -> int:
        """Calculate optimal batch size based on GPU memory."""
        if device_id not in self._gpu_info:
            return base_batch_size
        
        free_memory_gb = self._gpu_info[device_id]['free_memory'] / (1024**3)
        
        # Rough estimation: 1GB per batch of 10 faces
        if free_memory_gb >= 4.0:
            return base_batch_size * 2  # 20
        elif free_memory_gb >= 2.0:
            return base_batch_size  # 10
        else:
            return max(1, base_batch_size // 2)  # 5


class InsightFaceCUDAOptimizer:
    """Optimizes InsightFace configuration for CUDA usage."""
    
    def __init__(self, cuda_detector: CUDADetector = None):
        self.cuda_detector = cuda_detector or CUDADetector()
        self.logger = logging.getLogger(__name__)
    
    def get_optimized_config(self, device_id: int = 0) -> Dict:
        """Get optimized configuration for InsightFace."""
        config = {
            'providers': self.cuda_detector.get_optimal_providers(),
            'det_size': (640, 640),  # Optimal for accuracy
            'det_thresh': 0.5,
            'rec_thresh': 0.6,
            'face_align': True,
            'gpu_memory_fraction': 0.5,
            'batch_size': self.cuda_detector.get_optimal_batch_size(device_id),
            'parallel_processing': True,
            'max_workers': min(4, os.cpu_count() or 1)
        }
        
        # Add CUDA-specific optimizations
        if self.cuda_detector.cuda_available:
            config['cuda_provider_options'] = self.cuda_detector.get_cuda_provider_options(
                device_id, config['gpu_memory_fraction']
            )
            
            # Adjust batch size based on memory
            if self.cuda_detector.check_gpu_memory_availability(device_id, 2.0):
                config['batch_size'] = min(config['batch_size'] * 2, 20)
            
            self.logger.info(f"CUDA optimization applied - Batch size: {config['batch_size']}")
        else:
            self.logger.info("CUDA not available - using CPU optimization")
        
        return config
    
    def create_insightface_app(self, model_name: str = "buffalo_l", device_id: int = 0):
        """Create optimized InsightFace app instance."""
        try:
            import insightface
            
            config = self.get_optimized_config(device_id)
            
            # Create provider list with options
            providers = []
            for provider in config['providers']:
                if provider == "CUDAExecutionProvider" and 'cuda_provider_options' in config:
                    providers.append((provider, config['cuda_provider_options']))
                else:
                    providers.append(provider)
            
            self.logger.info(f"Creating InsightFace app with providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
            
            app = insightface.app.FaceAnalysis(
                name=model_name,
                providers=providers
            )
            
            # Prepare with optimized settings
            app.prepare(
                ctx_id=device_id,
                det_size=config['det_size']
            )
            
            self.logger.info("InsightFace app created successfully with CUDA optimization")
            return app
            
        except Exception as e:
            self.logger.error(f"Failed to create optimized InsightFace app: {e}")
            raise


# Global instances
cuda_detector = CUDADetector()
cuda_optimizer = InsightFaceCUDAOptimizer(cuda_detector)


def get_cuda_status() -> Dict:
    """Get comprehensive CUDA status information."""
    return {
        'cuda_available': cuda_detector.cuda_available,
        'available_providers': cuda_detector.available_providers,
        'gpu_info': cuda_detector.gpu_info,
        'optimal_providers': cuda_detector.get_optimal_providers(),
        'memory_available': {
            device_id: info['free_memory'] / (1024**3) 
            for device_id, info in cuda_detector.gpu_info.items()
        }
    }


def optimize_insightface_config(device_id: int = 0) -> Dict:
    """Get optimized configuration for InsightFace."""
    return cuda_optimizer.get_optimized_config(device_id)


def create_optimized_insightface_app(model_name: str = "buffalo_l", device_id: int = 0):
    """Create optimized InsightFace app instance."""
    return cuda_optimizer.create_insightface_app(model_name, device_id) 