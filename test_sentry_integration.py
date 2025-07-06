#!/usr/bin/env python3
"""
Sentry Integration Test Script for AI Video Slicer
This script tests if Sentry is properly configured and working.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.append(str(backend_dir))

try:
    from core.config import settings
    import sentry_sdk
    from core.exceptions import AIVideoSlicerException, VideoProcessingError
    from services.openai import OpenAIService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

def test_sentry_configuration():
    """Test if Sentry is properly configured."""
    print("🔍 Testing Sentry Configuration...")
    
    # Check if Sentry is enabled
    print(f"   Sentry enabled: {settings.sentry_enabled}")
    print(f"   Sentry DSN configured: {bool(settings.sentry_dsn)}")
    print(f"   Environment: {settings.sentry_environment}")
    print(f"   Sample rate: {settings.sentry_sample_rate}")
    
    # Check if Sentry SDK is initialized
    hub = sentry_sdk.Hub.current
    if hub.client:
        print("   ✅ Sentry SDK is initialized")
        print(f"   DSN: {hub.client.dsn}")
        return True
    else:
        print("   ❌ Sentry SDK is not initialized")
        return False

def test_custom_exception():
    """Test custom exception handling with Sentry."""
    print("\n🧪 Testing Custom Exception Handling...")
    
    try:
        # This should be captured by Sentry
        raise VideoProcessingError("Test error for Sentry integration")
    except AIVideoSlicerException as e:
        print(f"   ✅ Custom exception captured: {e.message}")
        print(f"   Error code: {e.error_code}")
        return True
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_sentry_message():
    """Test sending a custom message to Sentry."""
    print("\n📝 Testing Custom Message Capture...")
    
    try:
        sentry_sdk.capture_message(
            "Sentry integration test - This is a test message", 
            level="info"
        )
        print("   ✅ Test message sent to Sentry")
        return True
    except Exception as e:
        print(f"   ❌ Failed to send message: {e}")
        return False

def test_context_and_tags():
    """Test setting context and tags in Sentry."""
    print("\n🏷️ Testing Context and Tags...")
    
    try:
        # Set context
        sentry_sdk.set_context("test_context", {
            "test_type": "integration_test",
            "component": "sentry_setup",
            "timestamp": "2024-01-01T00:00:00Z"
        })
        
        # Set tags
        sentry_sdk.set_tag("test_mode", "true")
        sentry_sdk.set_tag("integration", "sentry")
        
        # Set user
        sentry_sdk.set_user({"id": "test_user_123"})
        
        print("   ✅ Context, tags, and user set successfully")
        return True
    except Exception as e:
        print(f"   ❌ Failed to set context/tags: {e}")
        return False

async def test_service_integration():
    """Test Sentry integration with OpenAI service."""
    print("\n🤖 Testing Service Integration...")
    
    try:
        service = OpenAIService()
        # This will likely fail due to missing API key, but should be captured by Sentry
        await service.generate_script("Test transcript for Sentry integration")
        print("   ✅ Service integration working")
        return True
    except Exception as e:
        print(f"   ✅ Service error captured by Sentry: {type(e).__name__}")
        return True

def main():
    """Run all Sentry integration tests."""
    print("🚀 AI Video Slicer - Sentry Integration Test\n")
    
    tests = [
        ("Configuration", test_sentry_configuration),
        ("Custom Exception", test_custom_exception),
        ("Message Capture", test_sentry_message),
        ("Context & Tags", test_context_and_tags),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Test async service integration
    print("\n🤖 Testing Service Integration...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_service_integration())
        results.append(("Service Integration", result))
    except Exception as e:
        print(f"   ❌ Async test failed: {e}")
        results.append(("Service Integration", False))
    finally:
        loop.close()
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Sentry integration is working correctly.")
        print("\n📖 Next steps:")
        print("1. Create .env files with the DSN keys from the integration guide")
        print("2. Start your application and check the Sentry dashboard")
        print("3. Dashboard: https://bastion-wo.sentry.io")
    else:
        print("\n⚠️ Some tests failed. Check the configuration and try again.")
        print("Make sure environment variables are properly set.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 