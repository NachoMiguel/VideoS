#!/usr/bin/env python3
"""
API Contract Validation Test
Tests that all frontend API calls have corresponding backend endpoints.
"""

import asyncio
import aiohttp
import sys
from pathlib import Path

# Add backend to Python path
sys.path.append(str(Path(__file__).parent))

class APIContractTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_endpoint_exists(self, method: str, endpoint: str) -> bool:
        """Test if an endpoint exists (returns non-404)."""
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.request(method, url) as response:
                # We expect various responses, but NOT 404 (endpoint missing)
                if response.status == 404:
                    return False
                return True
        except Exception as e:
            print(f"Error testing {method} {endpoint}: {str(e)}")
            return False
    
    async def test_all_contracts(self):
        """Test all API contracts from frontend."""
        
        # Frontend API calls identified from analysis
        contracts = [
            ("POST", "/api/video/upload"),
            ("POST", "/api/video/process"),
            ("POST", "/api/v1/modify-script"),
            ("GET", "/api/v1/download/test-session-id"),
            ("POST", "/api/v1/extract-transcript"),
            ("POST", "/api/video/extract-scene/test-session-id"),
        ]
        
        print("🔍 Testing API Contract Alignment...")
        print("=" * 50)
        
        total_tests = len(contracts)
        passed_tests = 0
        
        for method, endpoint in contracts:
            exists = await self.test_endpoint_exists(method, endpoint)
            status = "✅ PASS" if exists else "❌ FAIL"
            print(f"{status} {method:>6} {endpoint}")
            
            if exists:
                passed_tests += 1
        
        print("=" * 50)
        print(f"Results: {passed_tests}/{total_tests} endpoints available")
        
        if passed_tests == total_tests:
            print("🎉 SUCCESS: All API contracts aligned!")
            return True
        else:
            print("⚠️  ATTENTION: Some endpoints missing - check main.py router inclusion")
            return False

async def main():
    """Run the API contract validation test."""
    try:
        async with APIContractTester() as tester:
            success = await tester.test_all_contracts()
            
            if success:
                print("\n✅ API Contract Stage 1 - COMPLETE")
                return 0
            else:
                print("\n❌ API Contract Stage 1 - NEEDS FIXES")
                return 1
                
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 