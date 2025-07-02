#!/usr/bin/env python3
"""
Unit tests for service modules.
"""
import unittest
import asyncio
import os
import sys
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, AsyncMock

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from services.openai import openai_service
from services.elevenlabs import elevenlabs_service
from services.image_search import image_search_service

class TestOpenAIService(unittest.TestCase):
    """Test OpenAI service functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_script = """
        Jean Claude Vandamme was a martial artist.
        Steven Seagal practiced aikido.
        They both starred in action movies.
        """
    
    def test_extract_characters_fallback(self):
        """Test fallback character extraction."""
        characters = openai_service._extract_characters_fallback(self.sample_script)
        self.assertIsInstance(characters, list)
        self.assertTrue(len(characters) >= 0)
    
    def test_create_basic_segments(self):
        """Test basic script segmentation."""
        segments = openai_service._create_basic_segments(self.sample_script)
        self.assertIsInstance(segments, list)
        self.assertTrue(len(segments) > 0)
        
        for segment in segments:
            self.assertIn('text', segment)
            self.assertIn('characters', segment)
            self.assertIn('actions', segment)
            self.assertIn('tone', segment)
            self.assertIn('duration_estimate', segment)

class TestElevenLabsService(unittest.TestCase):
    """Test ElevenLabs service functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = "This is a test script for audio generation."
    
    def test_get_next_api_key(self):
        """Test API key rotation."""
        # Mock API keys
        elevenlabs_service.api_keys = ["key1", "key2", "key3"]
        
        key1 = elevenlabs_service._get_next_api_key()
        key2 = elevenlabs_service._get_next_api_key()
        key3 = elevenlabs_service._get_next_api_key()
        key4 = elevenlabs_service._get_next_api_key()  # Should rotate back
        
        self.assertIn(key1, elevenlabs_service.api_keys)
        self.assertIn(key2, elevenlabs_service.api_keys)
        self.assertIn(key3, elevenlabs_service.api_keys)
        self.assertEqual(key4, key1)  # Should rotate back to first
    
    def test_select_voice_for_segment(self):
        """Test voice selection logic."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Test character-specific voice selection
            voice_jcvd = loop.run_until_complete(
                elevenlabs_service._select_voice_for_segment("Jean Claude Vandamme speaks")
            )
            voice_seagal = loop.run_until_complete(
                elevenlabs_service._select_voice_for_segment("Steven Seagal responds")
            )
            voice_action = loop.run_until_complete(
                elevenlabs_service._select_voice_for_segment("Epic fight scene with combat")
            )
            voice_narrator = loop.run_until_complete(
                elevenlabs_service._select_voice_for_segment("The story begins")
            )
            
            self.assertIsInstance(voice_jcvd, str)
            self.assertIsInstance(voice_seagal, str)
            self.assertIsInstance(voice_action, str)
            self.assertIsInstance(voice_narrator, str)
        finally:
            loop.close()
    
    def test_get_usage_stats(self):
        """Test usage statistics."""
        stats = elevenlabs_service.get_usage_stats()
        
        self.assertIn('usage_tracker', stats)
        self.assertIn('current_key_index', stats)
        self.assertIn('total_characters', stats)
        self.assertIn('available_keys', stats)
        
        self.assertIsInstance(stats['usage_tracker'], dict)
        self.assertIsInstance(stats['current_key_index'], int)
        self.assertIsInstance(stats['total_characters'], int)
        self.assertIsInstance(stats['available_keys'], int)

class TestImageSearchService(unittest.TestCase):
    """Test Image Search service functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_characters = ["Jean Claude Vandamme", "Steven Seagal"]
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        test_cases = [
            ("Jean Claude Vandamme", "Jean_Claude_Vandamme"),
            ("Steven/Seagal", "Steven_Seagal"),
            ("Test<>Character", "Test__Character"),
            ("Very Long Character Name That Exceeds Fifty Characters Limit", "Very_Long_Character_Name_That_Exceeds_Fifty_Ch"),
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input=input_name):
                result = image_search_service._sanitize_filename(input_name)
                self.assertEqual(result, expected)
    
    def test_get_known_character_images(self):
        """Test known character image retrieval."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Test known characters
            jcvd_images = loop.run_until_complete(
                image_search_service._get_known_character_images("Jean Claude Vandamme")
            )
            seagal_images = loop.run_until_complete(
                image_search_service._get_known_character_images("Steven Seagal")
            )
            unknown_images = loop.run_until_complete(
                image_search_service._get_known_character_images("Unknown Character")
            )
            
            self.assertIsInstance(jcvd_images, list)
            self.assertIsInstance(seagal_images, list)
            self.assertIsInstance(unknown_images, list)
            
            self.assertTrue(len(jcvd_images) > 0)
            self.assertTrue(len(seagal_images) > 0)
            self.assertEqual(len(unknown_images), 0)
        finally:
            loop.close()
    
    def test_generate_placeholder_images(self):
        """Test placeholder image generation."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            placeholders = loop.run_until_complete(
                image_search_service._generate_placeholder_images("Test Character")
            )
            
            self.assertIsInstance(placeholders, list)
            self.assertTrue(len(placeholders) > 0)
            
            # Check that URLs are properly formatted
            for url in placeholders:
                self.assertTrue(url.startswith("https://via.placeholder.com/"))
                self.assertIn("text=", url)
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main() 