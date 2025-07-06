#!/usr/bin/env python3
"""
Integration Validation Test
Tests all the fixes made to ensure end-to-end functionality works.
"""

import asyncio
import json
import tempfile
import os
from pathlib import Path
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationValidator:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.session = None
        self.session_id = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_scene_data_transmission(self):
        """Test that scene data is properly transmitted via WebSocket."""
        logger.info("🔍 Testing Scene Data Transmission...")
        
        # Create a small test video file
        test_file = await self._create_test_video()
        
        try:
            # Step 1: Upload video
            logger.info("  📤 Uploading test video...")
            upload_response = await self._upload_video(test_file)
            assert upload_response['status'] == 'success', f"Upload failed: {upload_response}"
            
            self.session_id = upload_response['session_id']
            logger.info(f"  ✅ Video uploaded, session_id: {self.session_id}")
            
            # Step 2: Start processing
            logger.info("  🔄 Starting video processing...")
            process_response = await self._start_processing()
            assert process_response['status'] == 'processing_started', f"Processing failed: {process_response}"
            logger.info("  ✅ Processing started successfully")
            
            # Step 3: Monitor WebSocket for scene data
            logger.info("  👂 Monitoring WebSocket for scene data...")
            scenes_received = await self._monitor_websocket_for_scenes()
            assert scenes_received, "No scenes received via WebSocket"
            logger.info("  ✅ Scene data received successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Scene data transmission test failed: {str(e)}")
            return False
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    async def test_download_endpoint(self):
        """Test that download endpoint works with session_manager."""
        logger.info("🔍 Testing Download Endpoint...")
        
        if not self.session_id:
            logger.error("  ❌ No session_id available for download test")
            return False
            
        try:
            # Wait for processing to complete
            await self._wait_for_processing_completion()
            
            # Test download endpoint
            logger.info("  📥 Testing download endpoint...")
            download_response = await self._test_download()
            assert download_response, "Download endpoint failed"
            logger.info("  ✅ Download endpoint working")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Download endpoint test failed: {str(e)}")
            return False
    
    async def test_script_modification(self):
        """Test that script modification works with session_manager."""
        logger.info("🔍 Testing Script Modification...")
        
        if not self.session_id:
            logger.error("  ❌ No session_id available for script modification test")
            return False
            
        try:
            # Test script modification
            logger.info("  ✏️ Testing script modification...")
            modification_response = await self._test_script_modification()
            assert modification_response['success'], f"Script modification failed: {modification_response}"
            logger.info("  ✅ Script modification working")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Script modification test failed: {str(e)}")
            return False
    
    async def _create_test_video(self):
        """Create a small test video file using ffmpeg."""
        import subprocess
        
        # Create a temporary video file (1 second, 320x240)
        test_file = tempfile.mktemp(suffix='.mp4')
        
        try:
            # Use ffmpeg to create a test video
            cmd = [
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=30',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30', test_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return test_file
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg not available, create a dummy file
            logger.warning("  ⚠️ ffmpeg not available, creating dummy video file")
            with open(test_file, 'wb') as f:
                f.write(b'dummy video content')
            return test_file
    
    async def _upload_video(self, file_path):
        """Upload video to the backend."""
        with open(file_path, 'rb') as f:
            form_data = aiohttp.FormData()
            form_data.add_field('file', f, filename='test_video.mp4', content_type='video/mp4')
            
            async with self.session.post(f"{self.base_url}/api/video/upload", data=form_data) as response:
                return await response.json()
    
    async def _start_processing(self):
        """Start video processing."""
        async with self.session.post(f"{self.base_url}/api/video/process/{self.session_id}") as response:
            return await response.json()
    
    async def _monitor_websocket_for_scenes(self):
        """Monitor WebSocket for scene data."""
        import websockets
        
        try:
            uri = f"ws://localhost:8000/api/video/ws/{self.session_id}"
            async with websockets.connect(uri) as websocket:
                # Wait for messages for up to 30 seconds
                timeout = 30
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        # Check if we received scene data
                        if data.get('type') == 'scenes':
                            logger.info(f"  📊 Received {len(data.get('scenes', []))} scenes")
                            return True
                        elif data.get('type') == 'completion':
                            # Also check completion message for scenes
                            result = data.get('result', {})
                            if result.get('scenes'):
                                logger.info(f"  📊 Received {len(result['scenes'])} scenes in completion")
                                return True
                        
                    except asyncio.TimeoutError:
                        continue
                        
                return False
                
        except Exception as e:
            logger.error(f"  ❌ WebSocket monitoring failed: {str(e)}")
            return False
    
    async def _wait_for_processing_completion(self):
        """Wait for processing to complete."""
        # Wait up to 60 seconds for processing completion
        for _ in range(60):
            async with self.session.get(f"{self.base_url}/api/video/status/{self.session_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'completed':
                        return True
                    elif data.get('status') == 'error':
                        raise Exception(f"Processing failed: {data.get('error')}")
            await asyncio.sleep(1)
        
        raise Exception("Processing timeout")
    
    async def _test_download(self):
        """Test the download endpoint."""
        async with self.session.get(f"{self.base_url}/api/v1/download/{self.session_id}") as response:
            return response.status == 200
    
    async def _test_script_modification(self):
        """Test script modification endpoint."""
        data = {
            "session_id": self.session_id,
            "action": "shorten",
            "selected_text": "This is a test script that needs to be shortened.",
            "context_before": "Previous context.",
            "context_after": "Following context."
        }
        
        async with self.session.post(f"{self.base_url}/api/v1/modify-script", json=data) as response:
            return await response.json()

async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting Integration Validation Tests...")
    
    async with IntegrationValidator() as validator:
        tests = [
            ("Scene Data Transmission", validator.test_scene_data_transmission),
            ("Download Endpoint", validator.test_download_endpoint),
            ("Script Modification", validator.test_script_modification),
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = await test_func()
                results.append((test_name, result))
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                results.append((test_name, False))
                logger.error(f"{test_name}: ❌ FAILED - {str(e)}")
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info(f"{'='*50}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
            return True
        else:
            logger.error("❌ Some integration tests failed")
            return False

if __name__ == "__main__":
    asyncio.run(main()) 