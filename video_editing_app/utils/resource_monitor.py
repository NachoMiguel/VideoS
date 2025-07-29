#!/usr/bin/env python3
"""
Resource monitoring utilities for video processing.
"""

import psutil
import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """System resource information."""
    memory_available_gb: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    cpu_percent: float
    disk_usage_percent: float

class ResourceMonitor:
    """Monitor system resources during video processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_memory_gb = 2.0  # Minimum required memory for video processing
        self.critical_memory_gb = 1.0  # Critical memory threshold
        self.monitoring_active = False
    
    def start_monitoring(self, process_name: str = "video_processing"):
        """Start resource monitoring for a process."""
        try:
            self.monitoring_active = True
            self.logger.info(f"🔍 Starting resource monitoring for: {process_name}")
            self.log_system_status()
            return True
        except Exception as e:
            self.logger.error(f"❌ Error starting monitoring: {e}")
            return False
    
    def stop_monitoring(self, process_name: str = "video_processing"):
        """Stop resource monitoring for a process."""
        try:
            self.monitoring_active = False
            self.logger.info(f"🛑 Stopped resource monitoring for: {process_name}")
            self.log_system_status()
            return True
        except Exception as e:
            self.logger.error(f"❌ Error stopping monitoring: {e}")
            return False
    
    def print_summary(self, process_name: str = "video_processing"):
        """Print a summary of the monitoring session."""
        try:
            self.logger.info(f"📊 Resource monitoring summary for: {process_name}")
            self.log_system_status()
            return True
        except Exception as e:
            self.logger.error(f"❌ Error printing summary: {e}")
            return False
    
    def get_system_resources(self) -> SystemResources:
        """Get current system resource usage."""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            return SystemResources(
                memory_available_gb=memory.available / (1024**3),
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                memory_percent=memory.percent,
                cpu_percent=cpu,
                disk_usage_percent=disk.percent
            )
        except Exception as e:
            self.logger.error(f"❌ Error getting system resources: {e}")
            return SystemResources(0, 0, 0, 0, 0, 0)
    
    def check_memory_availability(self) -> bool:
        """Check if system has enough memory for video processing."""
        try:
            resources = self.get_system_resources()
            
            if resources.memory_available_gb < self.critical_memory_gb:
                self.logger.critical(f"🚨 CRITICAL: Only {resources.memory_available_gb:.1f}GB available (need {self.min_memory_gb}GB+)")
                return False
            
            if resources.memory_available_gb < self.min_memory_gb:
                self.logger.warning(f"⚠️ Low memory: {resources.memory_available_gb:.1f}GB available (need {self.min_memory_gb}GB+)")
                return False
            
            self.logger.info(f"✅ Memory OK: {resources.memory_available_gb:.1f}GB available")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking memory availability: {e}")
            return False
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            resources = self.get_system_resources()
            memory_usage_mb = resources.memory_used_gb * 1024  # Convert GB to MB
            return memory_usage_mb
        except Exception as e:
            self.logger.error(f"❌ Error getting memory usage: {e}")
            return 0.0
    
    def log_system_status(self):
        """Log current system status."""
        try:
            resources = self.get_system_resources()
            
            self.logger.info(f"📊 System Status:")
            self.logger.info(f"   Memory: {resources.memory_used_gb:.1f}GB used / {resources.memory_total_gb:.1f}GB total ({resources.memory_percent:.1f}%)")
            self.logger.info(f"   Available: {resources.memory_available_gb:.1f}GB")
            self.logger.info(f"   CPU: {resources.cpu_percent:.1f}%")
            self.logger.info(f"   Disk: {resources.disk_usage_percent:.1f}% used")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging system status: {e}")

class MemoryManager:
    """Manage memory usage during video processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.resource_monitor = ResourceMonitor()
    
    async def ensure_memory_available(self, required_gb: float = 2.0) -> bool:
        """Ensure enough memory is available before starting processing."""
        try:
            # Check current memory
            if not self.resource_monitor.check_memory_availability():
                self.logger.warning("⚠️ Insufficient memory - attempting cleanup")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Wait a moment for cleanup
                await asyncio.sleep(2)
                
                # Check again
                if not self.resource_monitor.check_memory_availability():
                    self.logger.error("❌ Still insufficient memory after cleanup")
                    return False
            
            self.logger.info(f"✅ Memory check passed - {required_gb}GB+ available")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error ensuring memory availability: {e}")
            return True  # Assume OK if we can't check
    
    async def monitor_processing(self, process_name: str, max_duration_minutes: int = 30):
        """Monitor memory during processing with periodic checks."""
        try:
            self.logger.info(f"🔍 Starting memory monitoring for {process_name}")
            
            start_time = asyncio.get_event_loop().time()
            check_interval = 30  # Check every 30 seconds
            
            while True:
                # Check if we've exceeded max duration
                elapsed_minutes = (asyncio.get_event_loop().time() - start_time) / 60
                if elapsed_minutes > max_duration_minutes:
                    self.logger.warning(f"⚠️ Processing exceeded {max_duration_minutes} minutes")
                    break
                
                # Log system status
                self.resource_monitor.log_system_status()
                
                # Check memory availability
                if not self.resource_monitor.check_memory_availability():
                    self.logger.critical(f"🚨 CRITICAL: Memory exhausted during {process_name}")
                    break
                
                # Wait for next check
                await asyncio.sleep(check_interval)
                
        except Exception as e:
            self.logger.error(f"❌ Error in memory monitoring: {e}")
    
    def check_memory_limit(self, required_gb: float = 2.0) -> bool:
        """Check if system has enough memory for processing."""
        try:
            return self.resource_monitor.check_memory_availability()
        except Exception as e:
            self.logger.error(f"❌ Error checking memory limit: {e}")
            return True  # Assume OK if we can't check
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        try:
            import gc
            gc.collect()
            self.logger.info("🗑️ Forced garbage collection completed")
        except Exception as e:
            self.logger.error(f"❌ Error during garbage collection: {e}")
    
    def force_cleanup(self):
        """Force memory cleanup."""
        try:
            import gc
            gc.collect()
            self.logger.info("🧹 Forced memory cleanup completed")
        except Exception as e:
            self.logger.error(f"❌ Error during forced cleanup: {e}")

class ProcessingLimits:
    """Define processing limits based on system resources."""
    
    @staticmethod
    def get_clip_batch_size() -> int:
        """Get recommended clip batch size based on available memory."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 8.0:
                return 20  # Large memory - bigger batches
            elif available_gb >= 4.0:
                return 15  # Medium memory - medium batches
            elif available_gb >= 2.0:
                return 10  # Low memory - small batches
            else:
                return 5   # Very low memory - tiny batches
                
        except Exception:
            return 10  # Default if we can't check
    
    @staticmethod
    def get_timeout_seconds() -> int:
        """Get recommended timeout based on system resources."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 4.0:
                return 600  # 10 minutes for good systems
            else:
                return 300  # 5 minutes for limited systems
                
        except Exception:
            return 300  # Default timeout 