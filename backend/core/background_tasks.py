"""
Background Task Manager
Handles periodic cleanup tasks and performance monitoring.
"""

import asyncio
import time
from typing import Dict, List
from datetime import datetime, timedelta
from core.config import settings
from core.logger import logger
from core.session import manager as session_manager
from core.performance_monitor import monitor as performance_monitor
from pathlib import Path

class BackgroundTaskManager:
    """Manages background tasks for cleanup and monitoring."""
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        self.shutdown_event = asyncio.Event()
        
    async def start(self):
        """Start all background tasks."""
        if self.running:
            logger.warning("Background tasks already running")
            return
        
        self.running = True
        logger.info("Starting background tasks...")
        
        # Start session cleanup task
        self.tasks['session_cleanup'] = asyncio.create_task(
            self._session_cleanup_loop()
        )
        
        # Start performance monitoring if enabled
        if settings.performance_monitoring_enabled:
            self.tasks['performance_monitor'] = asyncio.create_task(
                self._performance_monitor_loop()
            )
            
        # Start system health monitoring
        self.tasks['health_monitor'] = asyncio.create_task(
            self._health_monitor_loop()
        )
        
        logger.info(f"Started {len(self.tasks)} background tasks")
    
    async def stop(self):
        """Stop all background tasks gracefully."""
        if not self.running:
            return
            
        logger.info("Stopping background tasks...")
        self.running = False
        self.shutdown_event.set()
        
        # Cancel all tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelled task: {task_name}")
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        self.tasks.clear()
        logger.info("Background tasks stopped")
    
    async def _session_cleanup_loop(self):
        """Periodically clean up expired sessions."""
        while self.running:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(settings.cleanup_interval_minutes * 60)
            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _performance_monitor_loop(self):
        """Periodically collect and report performance metrics."""
        while self.running:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(settings.performance_report_interval_minutes * 60)
            except asyncio.CancelledError:
                logger.info("Performance monitor task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _health_monitor_loop(self):
        """Periodically check system health."""
        while self.running:
            try:
                await self._check_system_health()
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                logger.info("Health monitor task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions and their resources."""
        try:
            start_time = time.time()
            
            # Get current session count
            initial_count = len(session_manager.sessions)
            
            # Clean up expired sessions
            await session_manager.cleanup_expired_sessions()
            
            # Get final count
            final_count = len(session_manager.sessions)
            cleaned_count = initial_count - final_count
            
            duration = time.time() - start_time
            
            if cleaned_count > 0:
                logger.info(f"Session cleanup completed: {cleaned_count} expired sessions removed in {duration:.2f}s")
            else:
                logger.debug(f"Session cleanup completed: no expired sessions found ({final_count} active)")
                
        except Exception as e:
            logger.error(f"Failed to clean up expired sessions: {e}")
    
    async def _collect_performance_metrics(self):
        """Collect and optionally save performance metrics."""
        try:
            # Get performance summary
            summary = performance_monitor.get_performance_summary()
            
            # Log summary
            if summary.get('status') != 'no_data':
                total_ops = summary['summary']['total_operations']
                success_rate = summary['summary']['success_rate']
                avg_duration = summary['summary']['avg_duration']
                
                logger.info(f"Performance Summary: {total_ops} operations, "
                           f"{success_rate:.2%} success rate, "
                           f"{avg_duration:.2f}s avg duration")
            
            # Save detailed report if configured
            if settings.performance_report_interval_minutes > 0:
                report_path = await performance_monitor.save_performance_report()
                logger.debug(f"Performance report saved: {report_path}")
                
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    async def _check_system_health(self):
        """Check system health and log warnings."""
        try:
            health = performance_monitor.check_system_health()
            
            if health['status'] == 'critical':
                logger.warning(f"System health CRITICAL: {health['critical']}")
                for rec in health['recommendations']:
                    logger.warning(f"Recommendation: {rec}")
            elif health['status'] == 'warning':
                logger.warning(f"System health WARNING: {health['warnings']}")
            else:
                logger.debug(f"System health OK: memory={health['memory_percent']:.1f}%, "
                           f"cpu={health['cpu_percent']:.1f}%, "
                           f"disk={health['disk_percent']:.1f}%")
                
        except Exception as e:
            logger.error(f"Failed to check system health: {e}")
    
    def get_task_status(self) -> Dict[str, Dict]:
        """Get status of all background tasks."""
        status = {}
        
        for task_name, task in self.tasks.items():
            if task.done():
                if task.cancelled():
                    task_status = "cancelled"
                elif task.exception():
                    task_status = f"error: {task.exception()}"
                else:
                    task_status = "completed"
            else:
                task_status = "running"
            
            status[task_name] = {
                'status': task_status,
                'running': not task.done()
            }
        
        return status

# Global background task manager instance
manager = BackgroundTaskManager()

# Startup and shutdown handlers
async def startup_background_tasks():
    """Start background tasks on application startup."""
    # Create required directories
    await _create_required_directories()
    
    # Start background task manager
    await manager.start()

async def _create_required_directories():
    """Create all required directories for the application."""
    directories = [
        settings.upload_dir,
        settings.output_dir,
        settings.temp_dir,
        settings.cache_dir,
        settings.test_scripts_dir,
        settings.test_audio_dir,
        settings.test_characters_dir,
        f"{settings.temp_dir}/performance_reports",
        "logs"
    ]
    
    for dir_path in directories:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            # Don't fail startup for directory creation issues
            pass

async def shutdown_background_tasks():
    """Stop background tasks on application shutdown."""
    await manager.stop() 