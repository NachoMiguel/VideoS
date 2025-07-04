import pytest
import numpy as np
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from video.character_persistence import CharacterPersistence
from video.face_detection import FaceDetector

class TestCharacterPersistence:
    """Test the character persistence layer."""
    
    @pytest.fixture
    def temp_persistence(self):
        """Create a temporary persistence instance."""
        temp_dir = tempfile.mkdtemp()
        persistence = CharacterPersistence(temp_dir)
        yield persistence
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_save_and_load_character_training(self, temp_persistence):
        """Test saving and loading character training data."""
        character_name = "Test Character"
        embeddings = [np.random.rand(512).astype(np.float32) for _ in range(3)]
        metadata = {'test_field': 'test_value'}
        
        # Save character training
        success = temp_persistence.save_character_training(character_name, embeddings, metadata)
        assert success is True
        
        # Load character training
        loaded_embeddings = temp_persistence.load_character_training(character_name)
        assert loaded_embeddings is not None
        assert len(loaded_embeddings) == 3
        
        # Verify embeddings are the same
        for orig, loaded in zip(embeddings, loaded_embeddings):
            assert np.array_equal(orig, loaded)
    
    def test_is_character_cached(self, temp_persistence):
        """Test checking if character is cached."""
        character_name = "Test Character"
        
        # Should not be cached initially
        assert temp_persistence.is_character_cached(character_name) is False
        
        # Save character
        embeddings = [np.random.rand(512).astype(np.float32)]
        temp_persistence.save_character_training(character_name, embeddings)
        
        # Should be cached now
        assert temp_persistence.is_character_cached(character_name) is True
    
    def test_get_cached_characters(self, temp_persistence):
        """Test getting list of cached characters."""
        # Initially empty
        assert temp_persistence.get_cached_characters() == []
        
        # Add some characters
        characters = ["Character 1", "Character 2", "Character 3"]
        for char in characters:
            embeddings = [np.random.rand(512).astype(np.float32)]
            temp_persistence.save_character_training(char, embeddings)
        
        cached = temp_persistence.get_cached_characters()
        assert len(cached) == 3
        assert all(char in cached for char in characters)
    
    def test_clear_character_cache(self, temp_persistence):
        """Test clearing specific character cache."""
        character_name = "Test Character"
        embeddings = [np.random.rand(512).astype(np.float32)]
        
        # Save and verify
        temp_persistence.save_character_training(character_name, embeddings)
        assert temp_persistence.is_character_cached(character_name) is True
        
        # Clear and verify
        success = temp_persistence.clear_character_cache(character_name)
        assert success is True
        assert temp_persistence.is_character_cached(character_name) is False
    
    def test_clear_all_cache(self, temp_persistence):
        """Test clearing all cache."""
        # Add multiple characters
        for i in range(3):
            character_name = f"Character {i}"
            embeddings = [np.random.rand(512).astype(np.float32)]
            temp_persistence.save_character_training(character_name, embeddings)
        
        # Verify they exist
        assert len(temp_persistence.get_cached_characters()) == 3
        
        # Clear all
        success = temp_persistence.clear_all_cache()
        assert success is True
        assert len(temp_persistence.get_cached_characters()) == 0
    
    def test_get_cache_stats(self, temp_persistence):
        """Test getting cache statistics."""
        # Add some characters
        for i in range(2):
            character_name = f"Character {i}"
            embeddings = [np.random.rand(512).astype(np.float32) for _ in range(3)]
            temp_persistence.save_character_training(character_name, embeddings)
        
        stats = temp_persistence.get_cache_stats()
        
        assert 'total_characters' in stats
        assert 'valid_characters' in stats
        assert 'cache_size_mb' in stats
        assert stats['total_characters'] == 2
        assert stats['valid_characters'] == 2
        assert stats['cache_size_mb'] > 0


class TestPerformanceOptimizations:
    """Test performance optimizations in face detection."""
    
    @pytest.fixture
    def detector_with_settings(self):
        """Create detector with performance settings."""
        with patch('video.face_detection.insightface'):
            detector = FaceDetector()
            # Mock trained characters
            detector.known_faces = {
                'High Confidence Character': [np.random.rand(512).astype(np.float32)],
                'Low Confidence Character': [np.random.rand(512).astype(np.float32)]
            }
            return detector
    
    @patch('video.face_detection.settings')
    def test_frame_analysis_skips_no_faces(self, mock_settings, detector_with_settings):
        """Test that frames with no faces are skipped when setting enabled."""
        mock_settings.skip_frames_with_no_faces = True
        mock_settings.enable_face_quality_filter = False
        
        detector_with_settings._detect_faces_insightface = Mock(return_value=[])
        
        result = asyncio.run(detector_with_settings._analyze_frame(
            np.random.rand(480, 640, 3).astype(np.uint8), 5.0
        ))
        
        assert result is None  # Should skip
    
    @patch('video.face_detection.settings')
    def test_frame_analysis_returns_empty_when_skip_disabled(self, mock_settings, detector_with_settings):
        """Test that frames with no faces return empty result when skip disabled."""
        mock_settings.skip_frames_with_no_faces = False
        mock_settings.enable_face_quality_filter = False
        
        detector_with_settings._detect_faces_insightface = Mock(return_value=[])
        
        result = asyncio.run(detector_with_settings._analyze_frame(
            np.random.rand(480, 640, 3).astype(np.uint8), 5.0
        ))
        
        assert result is not None
        assert result['total_faces'] == 0
        assert result['character_count'] == 0
    
    @patch('video.face_detection.settings')
    def test_character_confidence_threshold_filtering(self, mock_settings, detector_with_settings):
        """Test that low confidence character matches are filtered out."""
        mock_settings.character_confidence_threshold = 0.8
        mock_settings.enable_face_quality_filter = False
        mock_settings.max_faces_per_frame = 10
        mock_settings.batch_character_identification = False
        
        # Mock face detection
        mock_faces = [{
            'bbox': [10, 10, 50, 50],
            'confidence': 0.95,
            'embedding': np.random.rand(512).astype(np.float32),
            'quality_score': 0.85
        }]
        detector_with_settings._detect_faces_insightface = Mock(return_value=mock_faces)
        
        # Mock low confidence character identification
        detector_with_settings.identify_character = Mock(return_value=('Low Confidence Character', 0.7))
        
        result = asyncio.run(detector_with_settings._analyze_frame(
            np.random.rand(480, 640, 3).astype(np.uint8), 5.0
        ))
        
        # Should have detected face but not identified character due to low confidence
        assert result['total_faces'] == 1
        assert result['character_count'] == 0
        assert result['faces'][0]['character_identified'] is False
    
    @patch('video.face_detection.settings')
    def test_batch_character_identification(self, mock_settings, detector_with_settings):
        """Test batch character identification."""
        mock_settings.character_confidence_threshold = 0.75
        mock_settings.enable_face_quality_filter = False
        mock_settings.max_faces_per_frame = 10
        mock_settings.batch_character_identification = True
        
        # Mock multiple faces
        mock_faces = [
            {
                'bbox': [10, 10, 50, 50],
                'confidence': 0.95,
                'embedding': np.random.rand(512).astype(np.float32),
                'quality_score': 0.85
            },
            {
                'bbox': [60, 60, 50, 50],
                'confidence': 0.90,
                'embedding': np.random.rand(512).astype(np.float32),
                'quality_score': 0.80
            }
        ]
        detector_with_settings._detect_faces_insightface = Mock(return_value=mock_faces)
        
        # Mock batch identification
        detector_with_settings._batch_identify_characters = AsyncMock(return_value=2)
        
        result = asyncio.run(detector_with_settings._analyze_frame(
            np.random.rand(480, 640, 3).astype(np.uint8), 5.0
        ))
        
        assert result['total_faces'] == 2
        assert result['character_count'] == 2
        detector_with_settings._batch_identify_characters.assert_called_once()
    
    def test_batch_identify_characters_implementation(self, detector_with_settings):
        """Test the batch character identification implementation."""
        faces = [
            {
                'bbox': [10, 10, 50, 50],
                'embedding': np.random.rand(512).astype(np.float32)
            },
            {
                'bbox': [60, 60, 50, 50],
                'embedding': np.random.rand(512).astype(np.float32)
            }
        ]
        
        # Mock settings
        with patch('video.face_detection.settings') as mock_settings:
            mock_settings.character_confidence_threshold = 0.5
            
            # Mock high similarity for first face
            detector_with_settings.compare_faces = Mock(side_effect=[0.85, 0.3, 0.2, 0.75])
            
            result = asyncio.run(detector_with_settings._batch_identify_characters(faces, 5.0))
            
            # Should identify at least one character with high confidence
            assert result >= 0
            assert len(faces) == 2


class TestFaceDetectorPersistenceIntegration:
    """Test FaceDetector integration with persistence layer."""
    
    @pytest.fixture
    def detector_with_temp_persistence(self):
        """Create detector with temporary persistence."""
        temp_dir = tempfile.mkdtemp()
        
        with patch('video.face_detection.character_persistence') as mock_persistence:
            # Setup mock persistence
            persistence_instance = CharacterPersistence(temp_dir)
            mock_persistence.load_character_training = persistence_instance.load_character_training
            mock_persistence.save_character_training = persistence_instance.save_character_training
            mock_persistence.get_cached_characters = persistence_instance.get_cached_characters
            mock_persistence.clear_all_cache = persistence_instance.clear_all_cache
            
            with patch('video.face_detection.insightface'):
                detector = FaceDetector()
                yield detector, persistence_instance
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_detector_loads_cached_characters_on_startup(self, detector_with_temp_persistence):
        """Test that detector loads cached characters on startup."""
        detector, persistence = detector_with_temp_persistence
        
        # Pre-save some character data
        character_name = "Cached Character"
        embeddings = [np.random.rand(512).astype(np.float32)]
        persistence.save_character_training(character_name, embeddings)
        
        # Reload detector (simulate startup)
        detector._load_cached_characters()
        
        # Should have loaded the character
        assert character_name in detector.known_faces
        assert len(detector.known_faces[character_name]) == 1
    
    def test_train_character_faces_uses_cache(self, detector_with_temp_persistence):
        """Test that training uses cached data when available."""
        detector, persistence = detector_with_temp_persistence
        
        # Pre-cache a character
        character_name = "Cached Character"
        cached_embeddings = [np.random.rand(512).astype(np.float32) for _ in range(3)]
        persistence.save_character_training(character_name, cached_embeddings)
        
        # Mock image detection (shouldn't be called for cached character)
        detector._detect_faces_insightface = Mock()
        
        # Train characters
        character_images = {character_name: ['image1.jpg', 'image2.jpg']}
        result = asyncio.run(detector.train_character_faces(character_images))
        
        # Should use cached data
        assert character_name in result
        assert len(result[character_name]) == 3
        # Face detection shouldn't be called for cached character
        detector._detect_faces_insightface.assert_not_called()
    
    def test_clear_character_training_clears_cache(self, detector_with_temp_persistence):
        """Test that clearing training also clears cache."""
        detector, persistence = detector_with_temp_persistence
        
        # Add some training data
        character_name = "Test Character"
        embeddings = [np.random.rand(512).astype(np.float32)]
        persistence.save_character_training(character_name, embeddings)
        detector.known_faces[character_name] = embeddings
        
        # Verify data exists
        assert character_name in detector.known_faces
        assert persistence.is_character_cached(character_name) is True
        
        # Clear training
        detector.clear_character_training()
        
        # Should clear both runtime and cached data
        assert len(detector.known_faces) == 0
        assert persistence.is_character_cached(character_name) is False


def test_phase3_integration_complete():
    """Test complete Phase 3 integration."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test persistence layer
        persistence = CharacterPersistence(temp_dir)
        assert hasattr(persistence, 'save_character_training')
        assert hasattr(persistence, 'load_character_training')
        assert hasattr(persistence, 'get_cache_stats')
        
        # Test face detector with persistence
        with patch('video.face_detection.insightface'):
            detector = FaceDetector()
            assert hasattr(detector, '_load_cached_characters')
            assert hasattr(detector, '_batch_identify_characters')
            assert hasattr(detector, 'get_cache_statistics')
        
        # Test settings exist
        from video.face_detection import settings
        assert hasattr(settings, 'character_confidence_threshold')
        assert hasattr(settings, 'batch_character_identification')
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 