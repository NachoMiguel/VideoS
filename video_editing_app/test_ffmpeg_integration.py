#!/usr/bin/env python3
"""
Test script for FFmpeg video processing (primary method).
This script tests the FFmpeg video processor to ensure it works correctly.
"""

import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.ffmpeg_processor import FFmpegVideoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ffmpeg_processor():
    """Test the FFmpeg video processor (primary method)."""
    try:
        logger.info("Testing FFmpeg video processor (primary method)...")
        
        # Initialize processor
        processor = FFmpegVideoProcessor()
        
        # Test data - mock scenes
        test_scenes = [
            {
                'video_path': 'uploads/video_0_Jean Claude Van Damme FINALLY Breaks Silence On Steven Seagal, And It\'s Terrible - Good Old Days (1080p, h264).mp4',
                'start_time': 0.0,
                'end_time': 10.0,
                'duration': 10.0
            },
            {
                'video_path': 'uploads/video_0_Jean Claude Van Damme FINALLY Breaks Silence On Steven Seagal, And It\'s Terrible - Good Old Days (1080p, h264).mp4',
                'start_time': 10.0,
                'end_time': 20.0,
                'duration': 10.0
            },
            {
                'video_path': 'uploads/video_1_Jean-Claude Van Damme FINALLY Breaks Silence On Steven Seagal, And It\'s Worse Than You Think.mp4',
                'start_time': 0.0,
                'end_time': 10.0,
                'duration': 10.0
            }
        ]
        
        # Check if test files exist
        valid_scenes = []
        for scene in test_scenes:
            if os.path.exists(scene['video_path']):
                valid_scenes.append(scene)
                logger.info(f"✅ Found test video: {scene['video_path']}")
            else:
                logger.warning(f"⚠️ Test video not found: {scene['video_path']}")
        
        if not valid_scenes:
            logger.error("No valid test videos found. Please ensure test videos are in the uploads directory.")
            return False
        
        # Test audio file
        test_audio = 'uploads/audio.mp3'  # Adjust path as needed
        if not os.path.exists(test_audio):
            logger.warning(f"Test audio not found: {test_audio}")
            test_audio = None
        
        # Test output path
        output_path = 'test_output_ffmpeg_primary.mp4'
        
        logger.info(f"Testing FFmpeg primary method with {len(valid_scenes)} valid scenes")
        
        # Test FFmpeg assembly (primary method)
        try:
            result = await processor.assemble_video_ffmpeg(valid_scenes, test_audio, output_path)
            logger.info(f"✅ FFmpeg primary test successful: {result}")
            
            # Check if output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Output file created: {output_path} ({file_size} bytes)")
                return True
            else:
                logger.error("❌ Output file was not created")
                return False
                
        except Exception as e:
            logger.error(f"❌ FFmpeg primary test failed: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Test setup failed: {str(e)}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting FFmpeg primary method test...")
    
    success = await test_ffmpeg_processor()
    
    if success:
        logger.info("🎉 FFmpeg primary method test PASSED!")
    else:
        logger.error("❌ FFmpeg primary method test FAILED!")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1) 