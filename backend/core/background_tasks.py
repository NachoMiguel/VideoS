"""
Background Task Manager
Handles essential cleanup tasks only.
"""

import asyncio
import time
from typing import Dict
from core.config import settings
from core.logger import logger
from core.session import manager as session_manager
from pathlib import Path

class BackgroundTaskManager:
    """Manages essential background tasks for cleanup."""
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
    async def start(self):
        """Start essential background tasks."""
        if self.running:
            logger.warning("Background tasks already running")
            return
        
        self.running = True
        logger.info("Starting essential background tasks...")
        
        # Only start session cleanup task
        self.tasks['session_cleanup'] = asyncio.create_task(
            self._session_cleanup_loop()
        )
        
        logger.info(f"Started {len(self.tasks)} background tasks")
    
    async def stop(self):
        """Stop all background tasks gracefully."""
        if not self.running:
            return
            
        logger.info("Stopping background tasks...")
        self.running = False
        
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