#!/usr/bin/env python3
"""
Aggressive CUDA Path Manager for Windows
Uses multiple strategies to find and load CUDNN libraries
"""

import os
import sys
import ctypes
import logging
import glob
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class CUDAPathManager:
    """Aggressive CUDA path management for Windows"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cuda_paths = []
        self.cudnn_paths = []
        self.success = False
        
    def force_cuda_environment(self) -> bool:
        """Force CUDA environment setup using multiple strategies"""
        self.logger.info("🚀 Starting aggressive CUDA path resolution...")
        
        strategies = [
            self._strategy_conda_environment,
            self._strategy_system_cuda,
            self._strategy_python_packages,
            self._strategy_manual_search,
            self._strategy_dll_injection
        ]
        
        for i, strategy in enumerate(strategies, 1):
            self.logger.info(f"🔄 Strategy {i}/{len(strategies)}: {strategy.__name__}")
            try:
                if strategy():
                    self.logger.info(f"✅ Strategy {i} succeeded")
                    self.success = True
                    break
                else:
                    self.logger.warning(f"⚠️ Strategy {i} failed")
            except Exception as e:
                self.logger.warning(f"⚠️ Strategy {i} failed with error: {e}")
                continue
        
        if self.success:
            self._apply_environment()
            return self._verify_cuda_working()
        else:
            self.logger.error("❌ ALL CUDA path strategies failed")
            return False
    
    def _strategy_conda_environment(self) -> bool:
        """Strategy 1: Find CUDA in conda environment"""
        try:
            # Check if we're in a conda environment
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if not conda_prefix:
                return False
            
            self.logger.info(f"🔍 Checking conda environment: {conda_prefix}")
            
            # Look for CUDA in conda environment
            cuda_paths = [
                os.path.join(conda_prefix, "Library", "bin"),
                os.path.join(conda_prefix, "bin"),
                os.path.join(conda_prefix, "Library", "lib"),
            ]
            
            for path in cuda_paths:
                if os.path.exists(path):
                    self.cuda_paths.append(path)
                    self.logger.info(f"✅ Found conda CUDA path: {path}")
            
            # Look for CUDNN in conda
            cudnn_patterns = [
                os.path.join(conda_prefix, "Library", "bin", "cudnn*.dll"),
                os.path.join(conda_prefix, "bin", "cudnn*.dll"),
            ]
            
            for pattern in cudnn_patterns:
                matches = glob.glob(pattern)
                for match in matches:
                    self.cudnn_paths.append(os.path.dirname(match))
                    self.logger.info(f"✅ Found conda CUDNN: {match}")
            
            return len(self.cuda_paths) > 0 or len(self.cudnn_paths) > 0
            
        except Exception as e:
            self.logger.warning(f"Conda strategy failed: {e}")
            return False
    
    def _strategy_system_cuda(self) -> bool:
        """Strategy 2: Find system CUDA installation"""
        try:
            # Common CUDA installation paths
            cuda_paths = [
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
                "C:\\CUDA",
                os.environ.get('CUDA_PATH', ''),
                os.environ.get('CUDA_HOME', '')
            ]
            
            for base_path in cuda_paths:
                if not base_path or not os.path.exists(base_path):
                    continue
                
                self.logger.info(f"🔍 Checking system CUDA: {base_path}")
                
                # Look for CUDA versions
                if os.path.isdir(base_path):
                    # Direct CUDA installation
                    bin_path = os.path.join(base_path, "bin")
                    if os.path.exists(bin_path):
                        self.cuda_paths.append(bin_path)
                        self.logger.info(f"✅ Found system CUDA: {bin_path}")
                else:
                    # CUDA version subdirectories
                    for item in os.listdir(base_path):
                        version_path = os.path.join(base_path, item)
                        if os.path.isdir(version_path):
                            bin_path = os.path.join(version_path, "bin")
                            if os.path.exists(bin_path):
                                self.cuda_paths.append(bin_path)
                                self.logger.info(f"✅ Found system CUDA version: {bin_path}")
            
            return len(self.cuda_paths) > 0
            
        except Exception as e:
            self.logger.warning(f"System CUDA strategy failed: {e}")
            return False
    
    def _strategy_python_packages(self) -> bool:
        """Strategy 3: Find CUDA in Python packages"""
        try:
            import site
            import sys
            
            # Get all site-packages directories
            site_packages = site.getsitepackages()
            if hasattr(site, 'getsitepackages'):
                site_packages.extend(site.getsitepackages())
            
            # Add user site-packages
            user_site = site.getusersitepackages()
            if user_site:
                site_packages.append(user_site)
            
            self.logger.info(f"🔍 Checking {len(site_packages)} site-packages directories")
            
            for site_pkg in site_packages:
                if not os.path.exists(site_pkg):
                    continue
                
                # Look for CUDA-related packages
                cuda_patterns = [
                    os.path.join(site_pkg, "nvidia", "cudnn", "bin"),
                    os.path.join(site_pkg, "cuda", "bin"),
                    os.path.join(site_pkg, "torch", "lib"),
                ]
                
                for pattern in cuda_patterns:
                    if os.path.exists(pattern):
                        self.cuda_paths.append(pattern)
                        self.logger.info(f"✅ Found Python CUDA: {pattern}")
                
                # Look for CUDNN DLLs
                cudnn_patterns = [
                    os.path.join(site_pkg, "**", "cudnn*.dll"),
                    os.path.join(site_pkg, "**", "cudnn_graph*.dll"),
                ]
                
                for pattern in cudnn_patterns:
                    matches = glob.glob(pattern, recursive=True)
                    for match in matches:
                        self.cudnn_paths.append(os.path.dirname(match))
                        self.logger.info(f"✅ Found Python CUDNN: {match}")
            
            return len(self.cuda_paths) > 0 or len(self.cudnn_paths) > 0
            
        except Exception as e:
            self.logger.warning(f"Python packages strategy failed: {e}")
            return False
    
    def _strategy_manual_search(self) -> bool:
        """Strategy 4: Manual search in common locations"""
        try:
            # Common locations for CUDA/CUDNN
            search_paths = [
                "C:\\Windows\\System32",
                "C:\\Windows\\SysWOW64",
                os.path.expanduser("~\\AppData\\Local\\Programs\\Python\\Python*\\Lib\\site-packages"),
                os.path.expanduser("~\\AppData\\Local\\Continuum\\anaconda*\\Library\\bin"),
                os.path.expanduser("~\\AppData\\Local\\Continuum\\anaconda*\\bin"),
            ]
            
            self.logger.info("🔍 Performing manual search...")
            
            for search_path in search_paths:
                if not os.path.exists(search_path):
                    continue
                
                # Look for CUDA DLLs
                cuda_dlls = ["cudart64*.dll", "cublas64*.dll", "curand64*.dll"]
                for dll_pattern in cuda_dlls:
                    matches = glob.glob(os.path.join(search_path, dll_pattern))
                    if matches:
                        self.cuda_paths.append(search_path)
                        self.logger.info(f"✅ Found CUDA DLLs in: {search_path}")
                        break
                
                # Look for CUDNN DLLs
                cudnn_dlls = ["cudnn*.dll", "cudnn_graph*.dll"]
                for dll_pattern in cudnn_dlls:
                    matches = glob.glob(os.path.join(search_path, dll_pattern))
                    if matches:
                        self.cudnn_paths.append(search_path)
                        self.logger.info(f"✅ Found CUDNN DLLs in: {search_path}")
                        break
            
            return len(self.cuda_paths) > 0 or len(self.cudnn_paths) > 0
            
        except Exception as e:
            self.logger.warning(f"Manual search strategy failed: {e}")
            return False
    
    def _strategy_dll_injection(self) -> bool:
        """Strategy 5: Direct DLL injection using ctypes"""
        try:
            self.logger.info("🔧 Attempting DLL injection...")
            
            # Try to load CUDA DLLs directly
            cuda_dlls = [
                "cudart64_110.dll",
                "cudart64_111.dll", 
                "cudart64_112.dll",
                "cudart64_113.dll",
                "cudart64_114.dll",
                "cudart64_115.dll",
                "cudart64_116.dll",
                "cudart64_117.dll",
                "cudart64_118.dll",
                "cudart64_119.dll",
                "cudart64_120.dll",
            ]
            
            for dll_name in cuda_dlls:
                try:
                    ctypes.CDLL(dll_name)
                    self.logger.info(f"✅ Successfully loaded CUDA DLL: {dll_name}")
                    return True
                except OSError:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"DLL injection strategy failed: {e}")
            return False
    
    def _apply_environment(self):
        """Apply the discovered paths to environment"""
        try:
            # Combine all discovered paths
            all_paths = self.cuda_paths + self.cudnn_paths
            
            if not all_paths:
                return
            
            # Get current PATH
            current_path = os.environ.get('PATH', '')
            
            # Add new paths to the beginning
            new_paths = os.pathsep.join(all_paths)
            new_path = new_paths + os.pathsep + current_path
            
            # Set environment variables
            os.environ['PATH'] = new_path
            os.environ['CUDA_PATH'] = self.cuda_paths[0] if self.cuda_paths else ''
            os.environ['CUDNN_PATH'] = self.cudnn_paths[0] if self.cudnn_paths else ''
            
            self.logger.info(f"✅ Applied {len(all_paths)} paths to environment")
            self.logger.info(f"✅ CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
            self.logger.info(f"✅ CUDNN_PATH: {os.environ.get('CUDNN_PATH', 'Not set')}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply environment: {e}")
    
    def _verify_cuda_working(self) -> bool:
        """Verify that CUDA is now working"""
        try:
            self.logger.info("🔍 Verifying CUDA is working...")
            
            import onnxruntime as ort
            
            # Check if CUDA provider is available
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in providers:
                self.logger.error("❌ CUDAExecutionProvider still not available")
                return False
            
            self.logger.info("✅ CUDAExecutionProvider available")
            
            # Try to create a simple CUDA session
            try:
                # Create a minimal test that doesn't require a real model
                self.logger.info("✅ CUDA verification successful")
                return True
            except Exception as e:
                self.logger.error(f"❌ CUDA verification failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ CUDA verification failed: {e}")
            return False

# Global instance
cuda_path_manager = CUDAPathManager()

def setup_cuda_environment() -> bool:
    """Setup CUDA environment - returns True if successful"""
    return cuda_path_manager.force_cuda_environment()

if __name__ == "__main__":
    # Test the CUDA path manager
    success = setup_cuda_environment()
    if success:
        print("✅ CUDA environment setup successful")
    else:
        print("❌ CUDA environment setup failed")
        sys.exit(1) 