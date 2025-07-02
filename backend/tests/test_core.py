#!/usr/bin/env python3
"""
Unit tests for core functionality.
"""
import unittest
import asyncio
import os
import sys
from pathlib import Path
import tempfile
import json

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from core.config import settings
from core.logger import logger
from core.exceptions import *

class TestCoreConfig(unittest.TestCase):
    """Test core configuration functionality."""
    
    def test_settings_initialization(self):
        """Test that settings are properly initialized."""
        self.assertIsNotNone(settings)
        self.assertIsInstance(settings.PROJECT_NAME, str)
        self.assertIsInstance(settings.max_file_size, int)
        self.assertTrue(settings.max_file_size > 0)
    
    def test_file_size_limits(self):
        """Test file size limit configurations."""
        self.assertEqual(settings.max_video_file_size_bytes, 419430400)  # 400MB
        self.assertEqual(settings.max_videos_per_session, 3)
        self.assertEqual(settings.max_video_duration_seconds, 2400)  # 40 minutes
    
    def test_parallel_processing_settings(self):
        """Test parallel processing configurations."""
        self.assertIsInstance(settings.parallel_processing, bool)
        self.assertIsInstance(settings.max_workers, int)
        self.assertTrue(settings.max_workers > 0)

class TestCoreExceptions(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_base_exception(self):
        """Test base exception functionality."""
        error = AIVideoSlicerException("Test error", "TEST_ERROR")
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.error_code, "TEST_ERROR")
    
    def test_specific_exceptions(self):
        """Test specific exception types."""
        exceptions_to_test = [
            (ConfigurationError, "Config error"),
            (VideoProcessingError, "Video error"),
            (FaceDetectionError, "Face error"),
            (AudioGenerationError, "Audio error"),
            (ImageSearchError, "Image error"),
            (ValidationError, "Validation error"),
        ]
        
        for exception_class, message in exceptions_to_test:
            with self.subTest(exception=exception_class.__name__):
                error = exception_class(message)
                self.assertIsInstance(error, AIVideoSlicerException)
                self.assertEqual(str(error), message)

class TestCoreLogger(unittest.TestCase):
    """Test logging functionality."""
    
    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        self.assertIsNotNone(logger)
    
    def test_logging_levels(self):
        """Test different logging levels."""
        # Test that logger can handle different levels without errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        # If we get here without exceptions, logging is working

if __name__ == "__main__":
    unittest.main() 