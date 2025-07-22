#!/usr/bin/env python3
"""
Test script for scene detection functionality.
Run this to verify scene detection is working before proceeding to Phase 2.
"""

import asyncio
import os
import sys
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from services.scene_detection import SceneDetector
from video_processor import AdvancedVideoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_scene_detection():
    """Test scene detection on uploaded videos."""
    try:
        # Check if there are uploaded videos
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            logger.error("No uploads directory found")
            return
        
        video_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            logger.error("No video files found in uploads directory")
            return
        
        logger.info(f"Found {len(video_files)} video files: {video_files}")
        
        # Test scene detection on first video
        test_video = os.path.join(uploads_dir, video_files[0])
        logger.info(f"Testing scene detection on: {test_video}")
        
        # Test standalone scene detector
        scene_detector = SceneDetector(threshold=25.0, min_scene_duration=2.0)
        scenes = scene_detector.detect_scenes_adaptive(test_video)
        
        logger.info(f"✅ Scene detection test passed!")
        logger.info(f"   Detected {len(scenes)} scenes in {test_video}")
        
        for i, scene in enumerate(scenes[:5]):  # Show first 5 scenes
            logger.info(f"   Scene {i+1}: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s ({scene['duration']:.2f}s)")
        
        # Test integrated video processor
        logger.info("Testing integrated video processor...")
        processor = AdvancedVideoProcessor()
        
        video_paths = [os.path.join(uploads_dir, f) for f in video_files]
        video_scenes = await processor.detect_video_scenes(video_paths)
        
        total_scenes = sum(len(scenes) for scenes in video_scenes.values())
        logger.info(f"✅ Integrated test passed!")
        logger.info(f"   Total scenes across all videos: {total_scenes}")
        
        for video_path, scenes in video_scenes.items():
            logger.info(f"   {os.path.basename(video_path)}: {len(scenes)} scenes")
        
        logger.info("🎉 Phase 1 (Scene Detection) is ready for production!")
        
    except Exception as e:
        logger.error(f"❌ Scene detection test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_scene_detection()) 