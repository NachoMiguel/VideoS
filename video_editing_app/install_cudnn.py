#!/usr/bin/env python3
"""
CUDNN Installation Script via Pip
Based on NVIDIA documentation: https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ Success: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def check_prerequisites():
    """Check and upgrade pip and wheel"""
    print("🔍 Checking prerequisites...")
    
    # Check pip version
    success, output = run_command("python -m pip --version", "Checking pip version")
    if not success:
        return False
    
    # Upgrade pip and wheel
    success, output = run_command("python -m pip install --upgrade pip wheel", "Upgrading pip and wheel")
    if not success:
        print("⚠️ Warning: Could not upgrade pip/wheel, continuing anyway...")
    
    return True

def detect_cuda_version():
    """Detect CUDA version from system"""
    print("\n🔍 Detecting CUDA version...")
    
    # Try to detect CUDA version from nvidia-smi
    success, output = run_command("nvidia-smi", "Checking NVIDIA driver and CUDA")
    if success:
        print("✅ NVIDIA driver detected")
        # Look for CUDA version in output
        if "CUDA Version: 12." in output:
            return "12"
        elif "CUDA Version: 11." in output:
            return "11"
    
    # Try to detect from CUDA installation
    cuda_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\CUDA",
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✅ Found CUDA installation at: {path}")
            # Look for version directories
            for item in os.listdir(path):
                if item.startswith("v12."):
                    return "12"
                elif item.startswith("v11."):
                    return "11"
    
    print("⚠️ Could not detect CUDA version, defaulting to CUDA 12")
    return "12"

def install_cudnn(cuda_version):
    """Install CUDNN for the specified CUDA version"""
    print(f"\n🚀 Installing CUDNN for CUDA {cuda_version}...")
    
    if cuda_version == "12":
        package = "nvidia-cudnn-cu12"
    elif cuda_version == "11":
        package = "nvidia-cudnn-cu11"
    else:
        print(f"❌ Unsupported CUDA version: {cuda_version}")
        return False
    
    success, output = run_command(f"python -m pip install {package}", f"Installing {package}")
    return success

def verify_installation():
    """Verify CUDNN installation"""
    print("\n🔍 Verifying CUDNN installation...")
    
    # Try to import torch and check CUDNN
    verify_script = """
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
    print("✅ CUDNN installation successful!")
else:
    print("❌ CUDA is not available")
"""
    
    with open("verify_cudnn.py", "w") as f:
        f.write(verify_script)
    
    success, output = run_command("python verify_cudnn.py", "Verifying CUDNN installation")
    
    # Clean up
    if os.path.exists("verify_cudnn.py"):
        os.remove("verify_cudnn.py")
    
    return success

def main():
    """Main installation process"""
    print("🚀 CUDNN Installation via Pip")
    print("=" * 50)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed")
        return False
    
    # Step 2: Detect CUDA version
    cuda_version = detect_cuda_version()
    print(f"📋 Detected CUDA version: {cuda_version}")
    
    # Step 3: Install CUDNN
    if not install_cudnn(cuda_version):
        print("❌ CUDNN installation failed")
        return False
    
    # Step 4: Verify installation
    if not verify_installation():
        print("❌ CUDNN verification failed")
        return False
    
    print("\n🎉 CUDNN installation completed successfully!")
    print("You can now test your video editing app with CUDA acceleration.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 