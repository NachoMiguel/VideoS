# CUDA Optimization Guide for InsightFace

## Overview

This guide explains how to optimize InsightFace face detection and recognition using CUDA acceleration. The system automatically detects CUDA capabilities and optimizes performance based on your GPU configuration.

## 🚀 Quick Start

### 1. Prerequisites

- **CUDA Toolkit**: Version 11.8 or later
- **NVIDIA GPU**: Compute capability 6.0 or higher
- **Python**: 3.8+ with pip
- **Windows**: Latest NVIDIA drivers

### 2. Installation

```bash
# Install CUDA-optimized dependencies
pip install -r requirements.txt

# Test CUDA optimization
python test_cuda_optimization.py
```

### 3. Verify Installation

Run the diagnostic script to verify everything is working:

```bash
cd backend
python test_cuda_optimization.py
```

## 🔧 Configuration

### Automatic Optimization

The system automatically optimizes InsightFace based on your GPU:

- **GPU Memory**: Automatically adjusts batch sizes and memory limits
- **Provider Selection**: Prioritizes CUDA with CPU fallback
- **Performance Tuning**: Optimizes for your specific hardware

### Manual Configuration

You can override automatic settings in `core/config.py`:

```python
# InsightFace Settings
insightface_model_name: str = "buffalo_l"  # Model size
insightface_providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
insightface_det_size: tuple = (640, 640)  # Detection resolution
insightface_gpu_memory_fraction: float = 0.5  # GPU memory usage
```

## 📊 Performance Benefits

### Speed Improvements

- **Face Detection**: 2-5x faster with CUDA
- **Face Recognition**: 3-8x faster with CUDA
- **Batch Processing**: Up to 10x faster for multiple faces

### Memory Optimization

- **Automatic Memory Management**: Adjusts based on available GPU memory
- **Batch Size Optimization**: Dynamically scales based on memory
- **Memory Monitoring**: Real-time GPU memory tracking

## 🛠️ Technical Details

### CUDA Detection

The system automatically detects:

- **ONNX Runtime Providers**: Available execution providers
- **GPU Information**: Memory, utilization, and capabilities
- **CUDA Compatibility**: Version and driver compatibility

### Provider Hierarchy

1. **CUDAExecutionProvider**: Primary GPU acceleration
2. **CPUExecutionProvider**: Fallback for CPU processing

### Memory Management

```python
# Automatic memory optimization
gpu_memory_fraction = 0.5  # Use 50% of available GPU memory
batch_size = optimal_batch_size  # Calculated based on memory
memory_limit = total_memory * gpu_memory_fraction
```

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Not Detected

**Symptoms**: `CUDA Available: False`

**Solutions**:
```bash
# Check CUDA installation
nvcc --version

# Check NVIDIA drivers
nvidia-smi

# Reinstall onnxruntime-gpu
pip uninstall onnxruntime
pip install onnxruntime-gpu==1.16.3
```

#### 2. Out of Memory Errors

**Symptoms**: `CUDA out of memory`

**Solutions**:
```python
# Reduce memory usage
insightface_gpu_memory_fraction = 0.3  # Use 30% instead of 50%

# Reduce batch size
face_detection_batch_size = 5  # Smaller batches
```

#### 3. Slow Performance

**Symptoms**: No speedup with CUDA

**Solutions**:
```python
# Check provider configuration
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Verify GPU utilization
nvidia-smi  # Should show GPU usage during processing
```

### Diagnostic Commands

```bash
# Test CUDA detection
python -c "from core.cuda_utils import get_cuda_status; print(get_cuda_status())"

# Test InsightFace initialization
python -c "from core.cuda_utils import create_optimized_insightface_app; app = create_optimized_insightface_app()"

# Monitor GPU usage
nvidia-smi -l 1  # Update every second
```

## 📈 Performance Monitoring

### GPU Metrics

The system tracks:

- **Memory Usage**: Free/used/total memory
- **Utilization**: GPU compute utilization
- **Temperature**: GPU temperature monitoring
- **Power**: Power consumption (if available)

### Performance Logging

```python
# Enable detailed logging
logging.getLogger('core.cuda_utils').setLevel(logging.DEBUG)

# Monitor performance
from core.cuda_utils import get_cuda_status
status = get_cuda_status()
print(f"GPU Memory: {status['memory_available']}")
```

## 🔄 Integration

### Face Detection Classes

All face detection classes automatically use CUDA optimization:

- `backend/video/face_detection.py`
- `ai_shared_lib/face_detection.py`
- `video_editing_app/services/face_detection.py`

### Automatic Fallback

If CUDA fails, the system automatically falls back to:

1. **CPU Processing**: Full functionality with slower performance
2. **OpenCV Fallback**: Basic face detection if InsightFace fails
3. **Error Handling**: Graceful degradation with logging

## 🎯 Best Practices

### 1. Memory Management

- Monitor GPU memory usage during processing
- Adjust `gpu_memory_fraction` based on your workload
- Use batch processing for multiple videos

### 2. Performance Tuning

- Use `buffalo_l` model for best accuracy
- Adjust `det_size` based on your video resolution
- Enable parallel processing for multiple videos

### 3. Monitoring

- Run `nvidia-smi` during processing
- Check logs for CUDA optimization status
- Monitor memory usage patterns

## 📋 System Requirements

### Minimum Requirements

- **GPU**: NVIDIA GTX 1060 or equivalent
- **Memory**: 4GB GPU memory
- **CUDA**: Version 11.8+
- **Drivers**: Latest NVIDIA drivers

### Recommended Requirements

- **GPU**: NVIDIA RTX 3070 or better
- **Memory**: 8GB+ GPU memory
- **CUDA**: Version 12.0+
- **Storage**: SSD for faster model loading

## 🔗 Related Files

- `core/cuda_utils.py`: CUDA detection and optimization
- `test_cuda_optimization.py`: Diagnostic and testing script
- `core/config.py`: Configuration settings
- `requirements.txt`: Dependencies with GPU support

## 📞 Support

If you encounter issues:

1. Run the diagnostic script: `python test_cuda_optimization.py`
2. Check GPU status: `nvidia-smi`
3. Verify CUDA installation: `nvcc --version`
4. Review logs for detailed error messages

## 🚀 Advanced Configuration

### Custom Provider Options

```python
# Advanced CUDA configuration
cuda_options = {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
    'cudnn_conv_use_max_workspace': '1',
    'do_copy_in_default_stream': '1',
}
```

### Multi-GPU Support

```python
# Use specific GPU
app = cuda_optimizer.create_optimized_insightface_app(
    model_name="buffalo_l",
    device_id=1  # Use GPU 1
)
```

### Memory Optimization

```python
# Dynamic memory allocation
config = cuda_optimizer.get_optimized_config(device_id=0)
batch_size = config['batch_size']  # Automatically optimized
```

---

**Note**: This optimization is automatically applied to all InsightFace usage in the system. No code changes are required - the system will automatically detect and use CUDA when available. 