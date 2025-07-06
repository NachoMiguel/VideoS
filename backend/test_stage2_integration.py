#!/usr/bin/env python3
"""
Stage 2 Integration Test - API Contract Validation
Tests the critical fixes made to ensure frontend-backend integration works.
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Add backend to Python path
sys.path.append(str(Path(__file__).parent))

class Stage2IntegrationTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log a test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append((test_name, success, message))
        print(f"{status} {test_name}: {message}")
    
    async def test_video_upload_endpoint(self):
        """Test /api/video/upload endpoint exists and accepts file uploads."""
        try:
            # Test if endpoint exists (should get 422 for missing file)
            async with self.session.post(f"{self.base_url}/api/video/upload") as response:
                # We expect 422 (validation error) not 404 (missing endpoint)
                if response.status == 422:
                    self.log_test("Video Upload Endpoint", True, "Endpoint exists and validates")
                    return True
                else:
                    self.log_test("Video Upload Endpoint", False, f"Unexpected status: {response.status}")
                    return False
        except Exception as e:
            self.log_test("Video Upload Endpoint", False, f"Error: {str(e)}")
            return False
    
    async def test_websocket_endpoint(self):
        """Test WebSocket endpoint at /api/video/ws/{session_id}."""
        try:
            # Test WebSocket connection
            session_id = "test-session"
            ws_url = f"{self.base_url.replace('http', 'ws')}/api/video/ws/{session_id}"
            
            async with self.session.ws_connect(ws_url) as ws:
                # Send ping
                await ws.send_str("ping")
                
                # Receive pong
                response = await asyncio.wait_for(ws.receive(), timeout=5.0)
                
                if response.type == aiohttp.WSMsgType.TEXT and response.data == "pong":
                    self.log_test("WebSocket Endpoint", True, "WebSocket ping/pong working")
                    return True
                else:
                    self.log_test("WebSocket Endpoint", False, f"Unexpected response: {response.data}")
                    return False
                    
        except Exception as e:
            self.log_test("WebSocket Endpoint", False, f"Error: {str(e)}")
            return False
    
    async def test_extract_scene_endpoint(self):
        """Test /api/video/extract-scene/{session_id} accepts JSON body."""
        try:
            session_id = "test-session"
            test_data = {"start_time": 0.0, "end_time": 10.0}
            
            async with self.session.post(
                f"{self.base_url}/api/video/extract-scene/{session_id}",
                json=test_data
            ) as response:
                # We expect 404 (session not found) not 422 (validation error)
                if response.status == 404:
                    self.log_test("Extract Scene Endpoint", True, "Endpoint accepts JSON body")
                    return True
                else:
                    self.log_test("Extract Scene Endpoint", False, f"Unexpected status: {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test("Extract Scene Endpoint", False, f"Error: {str(e)}")
            return False
    
    async def test_script_editor_endpoint(self):
        """Test /api/v1/modify-script endpoint exists."""
        try:
            # Test if endpoint exists (should get 422 for missing data)
            async with self.session.post(f"{self.base_url}/api/v1/modify-script") as response:
                # We expect 422 (validation error) not 404 (missing endpoint)
                if response.status == 422:
                    self.log_test("Script Editor Endpoint", True, "Endpoint exists and validates")
                    return True
                else:
                    self.log_test("Script Editor Endpoint", False, f"Unexpected status: {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test("Script Editor Endpoint", False, f"Error: {str(e)}")
            return False
    
    async def test_download_endpoint(self):
        """Test /api/v1/download/{session_id} endpoint exists."""
        try:
            session_id = "test-session"
            async with self.session.get(f"{self.base_url}/api/v1/download/{session_id}") as response:
                # We expect 404 (session not found) not 405 (method not allowed)
                if response.status == 404:
                    self.log_test("Download Endpoint", True, "Endpoint exists")
                    return True
                else:
                    self.log_test("Download Endpoint", False, f"Unexpected status: {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test("Download Endpoint", False, f"Error: {str(e)}")
            return False
    
    async def test_processing_endpoint(self):
        """Test /api/video/process/{session_id} endpoint exists."""
        try:
            session_id = "test-session"
            async with self.session.post(f"{self.base_url}/api/video/process/{session_id}") as response:
                # We expect 404 (session not found) not 405 (method not allowed)
                if response.status == 404:
                    self.log_test("Processing Endpoint", True, "Endpoint exists")
                    return True
                else:
                    self.log_test("Processing Endpoint", False, f"Unexpected status: {response.status}")
                    return False
                    
        except Exception as e:
            self.log_test("Processing Endpoint", False, f"Error: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all Stage 2 integration tests."""
        print("🔍 Running Stage 2 Integration Tests...")
        print("=" * 60)
        
        # Test all endpoints
        tests = [
            self.test_video_upload_endpoint(),
            self.test_websocket_endpoint(),
            self.test_extract_scene_endpoint(),
            self.test_script_editor_endpoint(),
            self.test_download_endpoint(),
            self.test_processing_endpoint()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Calculate success rate
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, success, _ in self.test_results if success)
        
        print("=" * 60)
        print(f"Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 SUCCESS: All Stage 2 integration tests passed!")
            print("✅ API Contract fixes working correctly")
            return True
        else:
            print("⚠️  ATTENTION: Some tests failed")
            print("❌ API Contract needs additional fixes")
            return False

async def main():
    """Run the Stage 2 integration test."""
    try:
        async with Stage2IntegrationTester() as tester:
            success = await tester.run_all_tests()
            
            if success:
                print("\n✅ STAGE 2 INTEGRATION - COMPLETE")
                return 0
            else:
                print("\n❌ STAGE 2 INTEGRATION - NEEDS FIXES")
                return 1
                
    except Exception as e:
        print(f"Integration test failed with error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 