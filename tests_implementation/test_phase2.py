import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the parent directory to the path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from video.face_detection import FaceDetector
from services.video_processor import VideoProcessor
from api.routes import analyze_video_scenes, _identify_characters

class TestCharacterIdentificationIntegration:
    """Test the character identification integration in frame analysis."""
    
    @pytest.fixture
    def detector(self):
        """Create a FaceDetector instance with trained characters."""
        with patch('video.face_detection.insightface'):
            detector = FaceDetector()
            # Mock trained characters
            detector.known_faces = {
                'Jean Claude Van Damme': [np.random.rand(512).astype(np.float32)],
                'Steven Seagal': [np.random.rand(512).astype(np.float32)]
            }
            return detector
    
    @patch('video.face_detection.cv2')
    def test_analyze_frame_with_character_identification(self, mock_cv2, detector):
        """Test that frame analysis includes character identification."""
        # Mock frame and face detection
        mock_frame = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_faces = [{
            'bbox': [10, 10, 50, 50],
            'confidence': 0.95,
            'embedding': np.random.rand(512).astype(np.float32),
            'quality_score': 0.85
        }]
        
        # Mock the _detect_faces_insightface method
        detector._detect_faces_insightface = Mock(return_value=mock_faces)
        
        # Mock identify_character to return a match
        detector.identify_character = Mock(return_value=('Jean Claude Van Damme', 0.87))
        
        result = asyncio.run(detector._analyze_frame(mock_frame, 5.0))
        
        assert result is not None
        assert result['timestamp'] == 5.0
        assert result['character_count'] == 1
        assert result['total_faces'] == 1
        assert len(result['faces']) == 1
        
        face = result['faces'][0]
        assert face['character'] == 'Jean Claude Van Damme'
        assert face['character_confidence'] == 0.87
        assert face['character_identified'] is True
    
    @patch('video.face_detection.cv2')
    def test_analyze_frame_no_character_match(self, mock_cv2, detector):
        """Test frame analysis when no character matches are found."""
        mock_frame = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_faces = [{
            'bbox': [10, 10, 50, 50],
            'confidence': 0.95,
            'embedding': np.random.rand(512).astype(np.float32),
            'quality_score': 0.85
        }]
        
        detector._detect_faces_insightface = Mock(return_value=mock_faces)
        detector.identify_character = Mock(return_value=None)  # No match
        
        result = asyncio.run(detector._analyze_frame(mock_frame, 5.0))
        
        assert result is not None
        assert result['character_count'] == 0
        assert result['total_faces'] == 1
        
        face = result['faces'][0]
        assert face['character'] is None
        assert face['character_confidence'] == 0.0
        assert face['character_identified'] is False
    
    @patch('video.face_detection.cv2')
    def test_analyze_frame_no_trained_characters(self, mock_cv2, detector):
        """Test frame analysis when no characters are trained."""
        detector.known_faces = {}  # No trained characters
        
        mock_frame = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_faces = [{
            'bbox': [10, 10, 50, 50],
            'confidence': 0.95,
            'embedding': np.random.rand(512).astype(np.float32),
            'quality_score': 0.85
        }]
        
        detector._detect_faces_insightface = Mock(return_value=mock_faces)
        
        result = asyncio.run(detector._analyze_frame(mock_frame, 5.0))
        
        assert result is not None
        assert result['character_count'] == 0
        
        face = result['faces'][0]
        assert face['character'] is None
        assert face['character_confidence'] == 0.0
        assert face['character_identified'] is False


class TestTrainingPipelineIntegration:
    """Test the training pipeline integration in video processing."""
    
    @pytest.fixture
    def video_processor(self):
        """Create a VideoProcessor instance for testing."""
        with patch('services.video_processor.main_face_detector') as mock_detector:
            mock_config = Mock()
            mock_config.max_workers = 2
            mock_session_manager = Mock()
            
            processor = VideoProcessor(mock_config, mock_session_manager)
            processor.face_detector = mock_detector
            return processor
    
    def test_video_processor_uses_correct_face_detector(self, video_processor):
        """Test that VideoProcessor uses the main InsightFace detector."""
        # Should use the main detector, not create a new one
        assert hasattr(video_processor, 'face_detector')
        assert video_processor.face_detector is not None
    
    @patch('services.image_search.image_search_service')
    def test_train_face_recognition_with_error_handling(self, mock_image_service, video_processor):
        """Test the training wrapper method."""
        characters = ['Jean Claude Van Damme', 'Steven Seagal']
        
        # Mock the face detector's training method
        video_processor.face_detector.train_face_recognition_async = AsyncMock(return_value=True)
        
        result = asyncio.run(video_processor._train_face_recognition_with_error_handling(characters, 'test_session'))
        
        assert result is True
        video_processor.face_detector.train_face_recognition_async.assert_called_once_with(characters)


class TestSceneCharacterLinking:
    """Test scene-character linking functionality."""
    
    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector with trained characters."""
        detector = Mock()
        detector.identify_character = Mock(return_value=('Jean Claude Van Damme', 0.85))
        return detector
    
    def test_identify_characters_with_pre_identified_faces(self, mock_detector):
        """Test character identification when faces are already identified."""
        faces = [{
            'bbox': [10, 10, 50, 50],
            'character': 'Jean Claude Van Damme',
            'character_confidence': 0.90,
            'character_identified': True
        }]
        
        with patch('api.routes.detector', mock_detector):
            result = asyncio.run(_identify_characters(faces))
        
        assert len(result) == 1
        assert result[0]['name'] == 'Jean Claude Van Damme'
        assert result[0]['confidence'] == 0.90
        # Should not call identify_character since face is already identified
        mock_detector.identify_character.assert_not_called()
    
    def test_identify_characters_with_embeddings(self, mock_detector):
        """Test character identification with face embeddings."""
        faces = [{
            'bbox': [10, 10, 50, 50],
            'embedding': np.random.rand(512).astype(np.float32),
            'character_identified': False
        }]
        
        with patch('api.routes.detector', mock_detector):
            result = asyncio.run(_identify_characters(faces))
        
        assert len(result) == 1
        assert result[0]['name'] == 'Jean Claude Van Damme'
        assert result[0]['confidence'] == 0.85
        mock_detector.identify_character.assert_called_once()
    
    def test_identify_characters_no_match(self, mock_detector):
        """Test character identification when no match is found."""
        faces = [{
            'bbox': [10, 10, 50, 50],
            'embedding': np.random.rand(512).astype(np.float32),
            'character_identified': False
        }]
        
        mock_detector.identify_character = Mock(return_value=None)  # No match
        
        with patch('api.routes.detector', mock_detector):
            result = asyncio.run(_identify_characters(faces))
        
        assert len(result) == 1
        assert result[0]['name'] == 'unknown'
        assert result[0]['confidence'] == 0.0
    
    @patch('api.routes.processor')
    def test_analyze_video_scenes_with_character_summary(self, mock_processor):
        """Test scene analysis includes character summary."""
        mock_scenes = [{
            'timestamp': 10.0,
            'duration': 5.0,
            'faces': [{
                'bbox': [10, 10, 50, 50],
                'character': 'Jean Claude Van Damme',
                'character_confidence': 0.85,
                'character_identified': True
            }]
        }]
        
        mock_processor.process_videos_parallel = AsyncMock(return_value=mock_scenes)
        
        with patch('api.routes._identify_characters', return_value=[
            {'character_id': 'Jean Claude Van Damme', 'name': 'Jean Claude Van Damme', 'confidence': 0.85}
        ]):
            result = asyncio.run(analyze_video_scenes(['test_video.mp4']))
        
        assert len(result) == 1
        scene = result[0]
        
        assert 'character_summary' in scene
        assert scene['character_summary']['total_characters'] == 1
        assert scene['character_summary']['identified_characters'] == 1
        assert 'Jean Claude Van Damme' in scene['character_summary']['character_names']
        assert scene['character_summary']['avg_confidence'] == 0.85


def test_phase2_integration_full_pipeline():
    """Test the complete Phase 2 integration."""
    with patch('video.face_detection.insightface'):
        detector = FaceDetector()
        
        # Test training pipeline
        assert hasattr(detector, 'train_face_recognition_async')
        assert callable(detector.train_face_recognition_async)
        
        # Test character identification
        assert hasattr(detector, 'identify_character')
        assert callable(detector.identify_character)
        
        # Test scene analysis integration
        assert hasattr(detector, 'get_trained_characters')
        assert callable(detector.get_trained_characters)
        
        # Test that all components are connected
        trained_chars = detector.get_trained_characters()
        assert isinstance(trained_chars, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 