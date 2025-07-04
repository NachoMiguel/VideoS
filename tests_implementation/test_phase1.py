import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the parent directory to the path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from video.face_detection import FaceDetector
from services.image_search import ImageSearchService

class TestFaceDetectorAPIMethodsPhase1:
    """Test the missing API methods implementation."""
    
    @pytest.fixture
    def detector(self):
        """Create a FaceDetector instance for testing."""
        with patch('video.face_detection.insightface'):
            detector = FaceDetector()
            # Mock known faces for testing
            detector.known_faces = {
                'test_character': [np.random.rand(512).astype(np.float32)]
            }
            return detector
    
    def test_get_face_embedding_from_existing_data(self, detector):
        """Test extracting embedding from face data that already has embedding."""
        test_embedding = np.random.rand(512).astype(np.float32)
        face_data = {'embedding': test_embedding}
        
        result = asyncio.run(detector.get_face_embedding(face_data))
        
        assert np.array_equal(result, test_embedding)
    
    def test_get_face_embedding_invalid_data(self, detector):
        """Test error handling when face data is invalid."""
        invalid_face_data = {'bbox': [10, 10, 50, 50]}  # No frame or embedding
        
        with pytest.raises(ValueError, match="Cannot extract embedding"):
            asyncio.run(detector.get_face_embedding(invalid_face_data))
    
    def test_compare_embeddings(self, detector):
        """Test comparing two embeddings."""
        emb1 = np.random.rand(512).astype(np.float32)
        emb2 = np.random.rand(512).astype(np.float32)
        
        result = asyncio.run(detector.compare_embeddings(emb1, emb2))
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    @patch('services.image_search.image_search_service')
    def test_train_face_recognition_async_success(self, mock_image_service, detector):
        """Test successful async training."""
        # Mock image search service
        mock_image_service.search_character_images = AsyncMock(return_value={
            'test_character': ['/path/to/image1.jpg', '/path/to/image2.jpg']
        })
        
        # Mock train_character_faces method
        detector.train_character_faces = AsyncMock(return_value={
            'test_character': [np.random.rand(512).astype(np.float32)]
        })
        
        result = asyncio.run(detector.train_face_recognition_async(['test_character']))
        
        assert result is True
        mock_image_service.search_character_images.assert_called_once_with(['test_character'])
        detector.train_character_faces.assert_called_once()
    
    @patch('services.image_search.image_search_service')
    def test_train_face_recognition_async_no_images(self, mock_image_service, detector):
        """Test async training with no images found."""
        mock_image_service.search_character_images = AsyncMock(return_value={})
        
        result = asyncio.run(detector.train_face_recognition_async(['test_character']))
        
        assert result is False
    
    def test_get_trained_characters(self, detector):
        """Test getting list of trained characters."""
        result = detector.get_trained_characters()
        
        assert isinstance(result, list)
        assert 'test_character' in result
    
    def test_clear_character_training(self, detector):
        """Test clearing character training data."""
        assert len(detector.known_faces) > 0
        
        detector.clear_character_training()
        
        assert len(detector.known_faces) == 0
    
    def test_get_character_training_status(self, detector):
        """Test getting character training status."""
        result = detector.get_character_training_status()
        
        assert isinstance(result, dict)
        assert 'test_character' in result
        assert result['test_character']['trained'] is True
        assert result['test_character']['embedding_count'] == 1


class TestImageSearchServicePhase1:
    """Test the image search service implementation."""
    
    @pytest.fixture
    def image_service(self):
        """Create an ImageSearchService instance for testing."""
        return ImageSearchService()
    
    def test_is_valid_image_url_valid(self, image_service):
        """Test URL validation with valid image URLs."""
        valid_urls = [
            'https://example.com/image.jpg',
            'https://example.com/photo.png',
            'https://example.com/pic.jpeg',
            'https://example.com/avatar.webp'
        ]
        
        for url in valid_urls:
            assert image_service._is_valid_image_url(url) is True
    
    def test_is_valid_image_url_invalid(self, image_service):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            'not_a_url',
            'https://example.com/document.pdf',
            'https://example.com/video.mp4',
            ''
        ]
        
        for url in invalid_urls:
            assert image_service._is_valid_image_url(url) is False
    
    @patch('aiohttp.ClientSession.get')
    def test_search_google_images_success(self, mock_get, image_service):
        """Test successful Google image search."""
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'items': [
                {'link': 'https://example.com/image1.jpg'},
                {'link': 'https://example.com/image2.png'}
            ]
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Mock API keys
        image_service.google_api_key = 'test_key'
        image_service.google_engine_id = 'test_engine'
        image_service.session = Mock()
        image_service.session.get = mock_get
        
        result = asyncio.run(image_service._search_google_images('test_character'))
        
        assert isinstance(result, list)
        assert len(result) <= 5  # Should limit to 5 images
    
    def test_search_local_images_known_character(self, image_service):
        """Test local image search for known characters."""
        result = asyncio.run(image_service._search_local_images('jean claude van damme'))
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('http' in url for url in result)
    
    def test_search_local_images_unknown_character(self, image_service):
        """Test local image search for unknown characters."""
        result = asyncio.run(image_service._search_local_images('unknown_character'))
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    @patch('aiohttp.ClientSession.head')
    def test_validate_image_urls(self, mock_head, image_service):
        """Test URL validation."""
        # Mock successful validation
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_head.return_value.__aenter__.return_value = mock_response
        
        image_service.session = Mock()
        image_service.session.head = mock_head
        
        test_urls = ['https://example.com/image1.jpg', 'https://example.com/image2.png']
        result = asyncio.run(image_service._validate_image_urls(test_urls))
        
        assert isinstance(result, list)
        assert len(result) == 2


def test_phase1_integration():
    """Test integration between FaceDetector and ImageSearchService."""
    with patch('video.face_detection.insightface'):
        detector = FaceDetector()
        
        # Test that the methods exist and are callable
        assert hasattr(detector, 'get_face_embedding')
        assert hasattr(detector, 'compare_embeddings')
        assert hasattr(detector, 'train_face_recognition_async')
        assert hasattr(detector, 'get_trained_characters')
        assert hasattr(detector, 'clear_character_training')
        assert hasattr(detector, 'get_character_training_status')
        
        # Test ImageSearchService integration
        image_service = ImageSearchService()
        assert hasattr(image_service, '_search_google_images')
        assert hasattr(image_service, '_search_local_images')
        assert hasattr(image_service, '_validate_image_urls')


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 