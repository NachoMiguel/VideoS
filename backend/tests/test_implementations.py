#!/usr/bin/env python3
"""
Test script to verify the new implementations work correctly.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from services.openai import openai_service
from services.elevenlabs import elevenlabs_service
from services.image_search import image_search_service
from video.face_detection import FaceDetector
from video.processor import VideoProcessor
from core.config import settings
from core.logger import logger

async def test_character_extraction():
    """Test character extraction from script."""
    print("Testing character extraction...")
    
    sample_script = """
    Welcome to this amazing video! Today we're going to explore the incredible world of action movies.
    
    Jean Claude Vandamme was known for his incredible martial arts skills and flexibility. 
    His movies were always full of intense fight scenes.
    
    Steven Seagal brought a different style to action movies with his aikido background. 
    His calm demeanor contrasted with explosive action sequences.
    
    Both of these action stars defined a generation of martial arts cinema.
    """
    
    try:
        characters = await openai_service.extract_characters_from_script(sample_script)
        print(f"✅ Character extraction successful: {characters}")
        return True
    except Exception as e:
        print(f"❌ Character extraction failed: {str(e)}")
        return False

async def test_image_search():
    """Test character image search."""
    print("Testing image search...")
    
    characters = ["Jean Claude Vandamme", "Steven Seagal"]
    
    try:
        character_images = await image_search_service.search_character_images(characters)
        print(f"✅ Image search successful: Found images for {len(character_images)} characters")
        for char, images in character_images.items():
            print(f"  - {char}: {len(images)} images")
        return True
    except Exception as e:
        print(f"❌ Image search failed: {str(e)}")
        return False

async def test_face_detection():
    """Test face detection system."""
    print("Testing face detection...")
    
    try:
        detector = FaceDetector()
        print("✅ Face detector initialized successfully")
        
        # Test voice availability
        voices = await detector.get_available_voices() if hasattr(detector, 'get_available_voices') else []
        print(f"✅ Face detection system ready")
        return True
    except Exception as e:
        print(f"❌ Face detection failed: {str(e)}")
        return False

async def test_audio_generation():
    """Test audio generation."""
    print("Testing audio generation...")
    
    sample_text = "This is a test of the audio generation system."
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Test audio generation
        result = await elevenlabs_service.generate_audio_from_script(sample_text, output_path)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Audio generation successful: {output_path} ({file_size} bytes)")
            os.unlink(output_path)  # Clean up
            return True
        else:
            print(f"❌ Audio generation failed: Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Audio generation failed: {str(e)}")
        return False

async def test_video_processor():
    """Test video processor enhancements."""
    print("Testing video processor...")
    
    try:
        processor = VideoProcessor()
        print("✅ Video processor initialized successfully")
        
        # Test scene transition detection methods
        import numpy as np
        import cv2
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:, :320] = [255, 0, 0]  # Blue half
        test_frame[:, 320:] = [0, 255, 0]  # Green half
        
        # Test enhanced scene detection components
        frame_history = []
        luminosity_history = []
        
        result = processor._enhanced_scene_transition_detection(
            test_frame, 0, frame_history, luminosity_history
        )
        
        print(f"✅ Scene transition detection working: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Video processor test failed: {str(e)}")
        return False

async def test_script_analysis():
    """Test script analysis functionality."""
    print("Testing script analysis...")
    
    sample_script = """
    Jean Claude Vandamme enters the scene with a powerful kick.
    Steven Seagal responds with his signature aikido move.
    The two action heroes engage in an epic battle.
    """
    
    try:
        segments = await openai_service.analyze_script_segments(sample_script)
        print(f"✅ Script analysis successful: {len(segments)} segments")
        for i, segment in enumerate(segments):
            print(f"  Segment {i+1}: {segment.get('text', '')[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Script analysis failed: {str(e)}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting implementation tests...\n")
    
    tests = [
        ("Character Extraction", test_character_extraction),
        ("Image Search", test_image_search),
        ("Face Detection", test_face_detection),
        ("Audio Generation", test_audio_generation),
        ("Video Processor", test_video_processor),
        ("Script Analysis", test_script_analysis),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All implementations are working correctly!")
    else:
        print(f"⚠️  {total - passed} implementations need attention")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main()) 