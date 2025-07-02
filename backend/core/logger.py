import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
import json
from .config import settings

class UnicodeSafeLogger:
    """Logger that handles Unicode characters safely across different platforms."""
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler with safe encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter with safe encoding
        console_formatter = SafeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with UTF-8 encoding
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_formatter = SafeFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not create file handler: {e}")
    
    def _safe_message(self, message: str) -> str:
        """Ensure message is safe for logging."""
        try:
            # Try to encode/decode to catch any problematic characters
            return str(message).encode('utf-8', errors='replace').decode('utf-8')
        except Exception:
            # If all else fails, use ASCII-safe representation
            return repr(message)
    
    def info(self, message: str):
        self.logger.info(self._safe_message(message))
    
    def error(self, message: str):
        self.logger.error(self._safe_message(message))
    
    def warning(self, message: str):
        self.logger.warning(self._safe_message(message))
    
    def debug(self, message: str):
        self.logger.debug(self._safe_message(message))
    
    def critical(self, message: str):
        self.logger.critical(self._safe_message(message))
    
    def log_api_usage(self, service: str, action: str, credits_used: int = 0):
        """Log API usage for credit tracking."""
        usage_data = {
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "action": action,
            "credits_used": credits_used
        }
        self.info(f"API_USAGE: {json.dumps(usage_data)}")
    
    def log_processing_phase(self, session_id: str, phase: str, progress: int, message: str = ""):
        """Log processing phase for WebSocket updates."""
        phase_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "phase": phase,
            "progress": progress,
            "message": message
        }
        self.info(f"PROCESSING_PHASE: {json.dumps(phase_data)}")
    
    def log_error_with_context(self, error: Exception, context: dict):
        """Log error with additional context for debugging."""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        self.error(f"ERROR_WITH_CONTEXT: {json.dumps(error_data, default=str)}")

class SafeFormatter(logging.Formatter):
    """Formatter that handles Unicode characters safely."""
    
    def format(self, record):
        try:
            # Ensure all record attributes are safe
            if hasattr(record, 'getMessage'):
                record.msg = self._safe_string(record.getMessage())
            else:
                record.msg = self._safe_string(str(record.msg))
            
            return super().format(record)
        except Exception:
            # Fallback to basic formatting
            return f"{record.levelname}: {self._safe_string(str(record.msg))}"
    
    def _safe_string(self, s: str) -> str:
        """Convert string to safe representation."""
        try:
            return str(s).encode('utf-8', errors='replace').decode('utf-8')
        except Exception:
            return repr(s)

def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = settings.base_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Application log file
    app_log = log_dir / "app.log"
    file_handler = logging.handlers.RotatingFileHandler(
        app_log,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Error log file
    error_log = log_dir / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log,
        maxBytes=10485760,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # API usage log
    api_logger = logging.getLogger('api')
    api_log = log_dir / "api_usage.log"
    api_handler = logging.handlers.RotatingFileHandler(
        api_log,
        maxBytes=10485760,
        backupCount=5
    )
    api_handler.setFormatter(formatter)
    api_logger.addHandler(api_handler)
    api_logger.propagate = False  # Don't propagate to root logger

    # Set specific loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

def log_api_usage(api_name: str, credits_used: int = 0, error: str = None):
    """Log API usage with credit tracking."""
    logger = logging.getLogger('api')
    timestamp = datetime.now().isoformat()
    
    if error:
        message = f"{timestamp} - {api_name} - ERROR: {error}"
        logger.error(message)
    else:
        message = f"{timestamp} - {api_name} - Credits used: {credits_used}"
        logger.info(message)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)

# Global logger instances
logger = UnicodeSafeLogger("ai_video_slicer", "logs/app.log")
api_logger = UnicodeSafeLogger("api_usage", "logs/api_usage.log")
processing_logger = UnicodeSafeLogger("processing", "logs/processing.log")
error_logger = UnicodeSafeLogger("errors", "logs/errors.log", level=logging.ERROR) 