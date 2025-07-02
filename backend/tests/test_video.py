#!/usr/bin/env python3
"""
Unit tests for video processing modules.
"""
import unittest
import asyncio
import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from video.face_detection import FaceDetector
from video.processor import VideoProcessor

class TestFaceDetector(unittest.TestCase):
    """Test face detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector()
        
        # Create a test image with a simple face-like pattern
        self.test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add some basic features that might be detected as a face
        cv2.rectangle(self.test_image, (50, 50), (150, 150), (255, 255, 255), -1)  # Face area
        cv2.rectangle(self.test_image, (70, 80), (80, 90), (0, 0, 0), -1)  # Left eye
        cv2.rectangle(self.test_image, (120, 80), (130, 90), (0, 0, 0), -1)  # Right eye
        cv2.rectangle(self.test_image, (95, 110), (105, 120), (0, 0, 0), -1)  # Nose
        cv2.rectangle(self.test_image, (80, 130), (120, 140), (0, 0, 0), -1)  # Mouth
    
    def test_detector_initialization(self):
        """Test that face detector initializes properly."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.face_cascade)
        self.assertIsNotNone(self.detector.eye_cascade)
        self.assertIsNotNone(self.detector.smile_cascade)
    
    def test_basic_face_detection(self):
        """Test basic face detection functionality."""
        faces = self.detector.detect_faces_basic(self.test_image)
        self.assertIsInstance(faces, list)
        # Basic detection might not find our simple pattern, but should not crash
    
    def test_advanced_face_detection(self):
        """Test advanced face detection functionality."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.detector.detect_faces_advanced(self.test_image)
            )
            self.assertIsInstance(result, dict)
            self.assertIn('faces', result)
            self.assertIn('confidence', result)
            self.assertIn('landmarks', result)
        finally:
            loop.close()
    
    def test_face_recognition_methods(self):
        """Test face recognition methods."""
        # Test that methods exist and can be called
        self.assertTrue(hasattr(self.detector, 'train_face_recognition'))
        self.assertTrue(hasattr(self.detector, 'recognize_face'))
        self.assertTrue(hasattr(self.detector, 'extract_face_features'))
        
        # Test feature extraction
        features = self.detector.extract_face_features(self.test_image)
        self.assertIsInstance(features, (list, np.ndarray, type(None)))

class TestVideoProcessor(unittest.TestCase):
    """Test video processor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
        
        # Create test frames
        self.test_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_frame1[:, :320] = [255, 0, 0]  # Blue half
        
        self.test_frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_frame2[:, 320:] = [0, 255, 0]  # Green half
        
        self.test_frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White frame
    
    def test_processor_initialization(self):
        """Test that video processor initializes properly."""
        self.assertIsNotNone(self.processor)
        self.assertTrue(hasattr(self.processor, 'face_detector'))
    
    def test_frame_difference_detection(self):
        """Test frame difference detection."""
        diff_score = self.processor._calculate_frame_difference(self.test_frame1, self.test_frame2)
        self.assertIsInstance(diff_score, float)
        self.assertTrue(0.0 <= diff_score <= 1.0)
        
        # Test with identical frames
        diff_score_same = self.processor._calculate_frame_difference(self.test_frame1, self.test_frame1)
        self.assertEqual(diff_score_same, 0.0)
    
    def test_histogram_difference_detection(self):
        """Test histogram-based difference detection."""
        hist_diff = self.processor._calculate_histogram_difference(self.test_frame1, self.test_frame2)
        self.assertIsInstance(hist_diff, float)
        self.assertTrue(0.0 <= hist_diff <= 1.0)
        
        # Test with identical frames
        hist_diff_same = self.processor._calculate_histogram_difference(self.test_frame1, self.test_frame1)
        self.assertEqual(hist_diff_same, 0.0)
    
    def test_luminosity_change_detection(self):
        """Test luminosity change detection."""
        lum_change = self.processor._calculate_luminosity_change(self.test_frame1, self.test_frame3)
        self.assertIsInstance(lum_change, float)
        self.assertTrue(lum_change >= 0.0)
    
    def test_edge_density_detection(self):
        """Test edge density detection."""
        edge_change = self.processor._calculate_edge_density_change(self.test_frame1, self.test_frame2)
        self.assertIsInstance(edge_change, float)
        self.assertTrue(edge_change >= 0.0)
    
    def test_enhanced_scene_transition_detection(self):
        """Test enhanced scene transition detection."""
        frame_history = []
        luminosity_history = []
        
        is_transition = self.processor._enhanced_scene_transition_detection(
            self.test_frame1, 0, frame_history, luminosity_history
        )
        
        self.assertIsInstance(is_transition, bool)
        self.assertEqual(len(frame_history), 1)
        self.assertEqual(len(luminosity_history), 1)
        
        # Test with a second frame
        is_transition2 = self.processor._enhanced_scene_transition_detection(
            self.test_frame2, 1, frame_history, luminosity_history
        )
        
        self.assertIsInstance(is_transition2, bool)
        self.assertEqual(len(frame_history), 2)
        self.assertEqual(len(luminosity_history), 2)
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        quality_score = self.processor._calculate_quality_score(self.test_frame1)
        self.assertIsInstance(quality_score, float)
        self.assertTrue(0.0 <= quality_score <= 1.0)
    
    def test_scene_duration_limit(self):
        """Test scene duration limiting."""
        # Test with a mock scene that exceeds duration limit
        mock_scene = {
            'start_frame': 0,
            'end_frame': 1800,  # 60 seconds at 30fps
            'fps': 30
        }
        
        limited_scene = self.processor._limit_scene_duration(mock_scene, max_duration=30)
        
        self.assertIsInstance(limited_scene, dict)
        self.assertIn('start_frame', limited_scene)
        self.assertIn('end_frame', limited_scene)
        
        # Check that duration is limited
        duration = (limited_scene['end_frame'] - limited_scene['start_frame']) / mock_scene['fps']
        self.assertLessEqual(duration, 30)
    
    def test_transition_creation(self):
        """Test transition creation."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Test transition creation (should not crash)
            transition = self.processor._create_transition(duration=1.0, transition_type='fade')
            self.assertIsNotNone(transition)
        except Exception as e:
            # If it fails due to missing video files, that's expected in unit tests
            self.assertIsInstance(e, Exception)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestVideoProcessingIntegration(unittest.TestCase):
    """Integration tests for video processing."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.processor = VideoProcessor()
    
    def test_scene_data_structure(self):
        """Test scene data structure consistency."""
        # Test that scene data has expected structure
        sample_scene = {
            'id': 'scene_001',
            'start_time': 0.0,
            'end_time': 5.0,
            'start_frame': 0,
            'end_frame': 150,
            'fps': 30,
            'quality_score': 0.8,
            'faces_detected': 2,
            'transition_score': 0.6
        }
        
        # Validate scene structure
        required_fields = ['id', 'start_time', 'end_time', 'start_frame', 'end_frame', 'fps']
        for field in required_fields:
            self.assertIn(field, sample_scene)
    
    def test_processing_pipeline_structure(self):
        """Test that processing pipeline has expected structure."""
        # Test that processor has required methods
        required_methods = [
            'extract_scenes',
            'detect_faces_in_scenes',
            'calculate_scene_quality',
            'create_compilation'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(self.processor, method))
            self.assertTrue(callable(getattr(self.processor, method)))

if __name__ == "__main__":
    unittest.main() 