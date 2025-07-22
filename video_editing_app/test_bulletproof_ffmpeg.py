#!/usr/bin/env python3
"""
Test script for bulletproof FFmpeg video processor.
Tests batch processing, error handling, and file verification.
"""

import asyncio
import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from services.ffmpeg_processor import FFmpegVideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class BulletproofFFmpegTester:
    """Test suite for bulletproof FFmpeg processor."""
    
    def __init__(self):
        self.test_dir = tempfile.mkdtemp(prefix="ffmpeg_test_")
        self.logger = logger
        
    async def run_all_tests(self):
        """Run all tests and report results."""
        self.logger.info("🚀 Starting bulletproof FFmpeg tests...")
        
        tests = [
            ("Test 1: Small batch processing", self.test_small_batch),
            ("Test 2: Large batch processing", self.test_large_batch),
            ("Test 3: File verification", self.test_file_verification),
            ("Test 4: Error handling", self.test_error_handling),
            ("Test 5: Timeout handling", self.test_timeout_handling),
            ("Test 6: Cleanup verification", self.test_cleanup),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Running: {test_name}")
                self.logger.info(f"{'='*60}")
                
                await test_func()
                results.append((test_name, "✅ PASSED"))
                self.logger.info(f"✅ {test_name} PASSED")
                
            except Exception as e:
                results.append((test_name, f"❌ FAILED: {str(e)}"))
                self.logger.error(f"❌ {test_name} FAILED: {str(e)}")
        
        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        
        passed = sum(1 for _, result in results if "PASSED" in result)
        total = len(results)
        
        for test_name, result in results:
            self.logger.info(f"{test_name}: {result}")
        
        self.logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 ALL TESTS PASSED - Bulletproof FFmpeg is ready!")
        else:
            self.logger.error("💥 Some tests failed - Review implementation")
        
        # Cleanup
        self.cleanup()
    
    async def test_small_batch(self):
        """Test processing with small number of clips (should use standard concatenation)."""
        self.logger.info("Testing small batch processing (< 50 clips)...")
        
        # Create test processor with small batch size
        processor = FFmpegVideoProcessor(max_clips_per_batch=50, timeout_seconds=30)
        
        # Create mock scenes (small batch)
        scenes = self._create_mock_scenes(25)  # 25 clips
        
        # Create mock audio file
        audio_path = self._create_mock_audio()
        output_path = os.path.join(self.test_dir, "small_batch_output.mp4")
        
        # Test assembly
        result = await processor.assemble_video_ffmpeg(scenes, audio_path, output_path)
        
        # Verify result
        if not os.path.exists(result):
            raise Exception("Output file not created")
        
        file_size = os.path.getsize(result)
        if file_size < 1024:
            raise Exception(f"Output file too small: {file_size} bytes")
        
        self.logger.info(f"✅ Small batch test passed - Output: {result} ({file_size} bytes)")
    
    async def test_large_batch(self):
        """Test processing with large number of clips (should use batch processing)."""
        self.logger.info("Testing large batch processing (> 50 clips)...")
        
        # Create test processor with small batch size to trigger batching
        processor = FFmpegVideoProcessor(max_clips_per_batch=10, timeout_seconds=60)
        
        # Create mock scenes (large batch)
        scenes = self._create_mock_scenes(75)  # 75 clips
        
        # Create mock audio file
        audio_path = self._create_mock_audio()
        output_path = os.path.join(self.test_dir, "large_batch_output.mp4")
        
        # Test assembly
        result = await processor.assemble_video_ffmpeg(scenes, audio_path, output_path)
        
        # Verify result
        if not os.path.exists(result):
            raise Exception("Output file not created")
        
        file_size = os.path.getsize(result)
        if file_size < 1024:
            raise Exception(f"Output file too small: {file_size} bytes")
        
        self.logger.info(f"✅ Large batch test passed - Output: {result} ({file_size} bytes)")
    
    async def test_file_verification(self):
        """Test file verification functionality."""
        self.logger.info("Testing file verification...")
        
        processor = FFmpegVideoProcessor()
        
        # Test with non-existent file
        result = processor._verify_video_file("/nonexistent/file.mp4")
        if result:
            raise Exception("File verification should fail for non-existent file")
        
        # Test with empty file
        empty_file = os.path.join(self.test_dir, "empty.mp4")
        with open(empty_file, 'w') as f:
            pass  # Create empty file
        
        result = processor._verify_video_file(empty_file)
        if result:
            raise Exception("File verification should fail for empty file")
        
        # Test with small file
        small_file = os.path.join(self.test_dir, "small.mp4")
        with open(small_file, 'w') as f:
            f.write("x" * 100)  # 100 bytes
        
        result = processor._verify_video_file(small_file)
        if result:
            raise Exception("File verification should fail for small file")
        
        self.logger.info("✅ File verification test passed")
    
    async def test_error_handling(self):
        """Test error handling with invalid inputs."""
        self.logger.info("Testing error handling...")
        
        processor = FFmpegVideoProcessor()
        
        # Test with empty scenes list
        try:
            await processor.assemble_video_ffmpeg([], "audio.mp3", "output.mp4")
            raise Exception("Should have failed with empty scenes")
        except Exception as e:
            if "No valid clips could be created" not in str(e):
                raise Exception(f"Unexpected error: {str(e)}")
        
        # Test with invalid video paths
        invalid_scenes = [
            {"video_path": "/nonexistent/video.mp4", "start_time": 0, "end_time": 10}
        ]
        
        try:
            await processor.assemble_video_ffmpeg(invalid_scenes, "audio.mp3", "output.mp4")
            raise Exception("Should have failed with invalid video paths")
        except Exception as e:
            if "No valid clips could be created" not in str(e):
                raise Exception(f"Unexpected error: {str(e)}")
        
        self.logger.info("✅ Error handling test passed")
    
    async def test_timeout_handling(self):
        """Test timeout handling."""
        self.logger.info("Testing timeout handling...")
        
        # Create processor with very short timeout
        processor = FFmpegVideoProcessor(timeout_seconds=1)
        
        # Create a large number of scenes to potentially trigger timeout
        scenes = self._create_mock_scenes(100)
        audio_path = self._create_mock_audio()
        output_path = os.path.join(self.test_dir, "timeout_test.mp4")
        
        try:
            await processor.assemble_video_ffmpeg(scenes, audio_path, output_path)
            self.logger.info("✅ Timeout test passed (no timeout occurred)")
        except Exception as e:
            if "timeout" in str(e).lower():
                self.logger.info("✅ Timeout test passed (timeout handled correctly)")
            else:
                raise Exception(f"Unexpected error during timeout test: {str(e)}")
    
    async def test_cleanup(self):
        """Test cleanup functionality."""
        self.logger.info("Testing cleanup functionality...")
        
        processor = FFmpegVideoProcessor()
        
        # Create some temporary files
        temp_files = []
        for i in range(5):
            temp_file = os.path.join(processor.temp_dir, f"test_file_{i}.txt")
            with open(temp_file, 'w') as f:
                f.write(f"test content {i}")
            temp_files.append(temp_file)
        
        # Test cleanup
        await processor._cleanup_temp_files(temp_files)
        
        # Verify files are cleaned up
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                raise Exception(f"File not cleaned up: {temp_file}")
        
        self.logger.info("✅ Cleanup test passed")
    
    def _create_mock_scenes(self, count: int) -> list:
        """Create mock scenes for testing."""
        scenes = []
        
        # Use a real video file if available, otherwise create mock data
        test_video = self._find_test_video()
        
        for i in range(count):
            scene = {
                "video_path": test_video,
                "start_time": i * 2.0,  # 2 second intervals
                "end_time": (i + 1) * 2.0,
                "duration": 2.0
            }
            scenes.append(scene)
        
        return scenes
    
    def _find_test_video(self) -> str:
        """Find a test video file or create a mock one."""
        # Look for test videos in common locations
        possible_paths = [
            "uploads/video_0_Jean Claude Van Damme FINALLY Breaks Silence On Steven Seagal, And It's Terrible - Good Old Days (1080p, h264).mp4",
            "uploads/video_1_Jean-Claude Van Damme FINALLY Breaks Silence On Steven Seagal, And It's Worse Than You Think.mp4",
            "test_data/test_video.mp4"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no real video found, create a mock path (this will fail gracefully)
        self.logger.warning("⚠️ No test video found, using mock path (test will fail gracefully)")
        return "mock_video.mp4"
    
    def _create_mock_audio(self) -> str:
        """Create a mock audio file for testing."""
        audio_path = os.path.join(self.test_dir, "mock_audio.mp3")
        
        # Look for real audio file first
        real_audio_paths = [
            "uploads/audio_audiofinal.MP3",
            "test_data/test_audio.mp3"
        ]
        
        for path in real_audio_paths:
            if os.path.exists(path):
                return path
        
        # Create mock audio file
        with open(audio_path, 'w') as f:
            f.write("mock audio content")
        
        return audio_path
    
    def cleanup(self):
        """Clean up test files."""
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                self.logger.info(f"🧹 Cleaned up test directory: {self.test_dir}")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup error: {str(e)}")

async def main():
    """Main test runner."""
    tester = BulletproofFFmpegTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 