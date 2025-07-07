#!/usr/bin/env python3
"""
ElevenLabs Configuration Diagnostic Tool
----------------------------------------
This script validates and reports on ElevenLabs API configuration.
Run this to diagnose configuration issues.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_environment():
    """Load environment variables from .env file."""
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
    else:
        print(f"❌ .env file not found at: {env_path}")
        return False
    return True

def check_elevenlabs_keys():
    """Check ElevenLabs API key configuration."""
    print("\n🔍 ElevenLabs API Key Analysis:")
    print("-" * 50)
    
    keys_found = []
    
    # Check numbered keys (1-4)
    for i in range(1, 5):
        key = os.getenv(f"ELEVENLABS_API_KEY_{i}")
        if key:
            # Show partial key for security
            masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            keys_found.append(i)
            print(f"✅ ELEVENLABS_API_KEY_{i}: {masked_key}")
        else:
            print(f"❌ ELEVENLABS_API_KEY_{i}: Not set")
    
    # Check fallback single key
    single_key = os.getenv("ELEVENLABS_API_KEY")
    if single_key:
        masked_key = f"{single_key[:8]}...{single_key[-4:]}" if len(single_key) > 12 else "***"
        print(f"✅ ELEVENLABS_API_KEY (fallback): {masked_key}")
    else:
        print(f"❌ ELEVENLABS_API_KEY (fallback): Not set")
    
    return keys_found, single_key

def check_other_config():
    """Check other related configuration."""
    print("\n🔧 Other Configuration:")
    print("-" * 50)
    
    configs = [
        ("ELEVENLABS_VOICE_ID", "nPczCjzI2devNBz1zQrb"),
        ("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"),
        ("ELEVENLABS_TIMEOUT", "30"),
        ("ELEVENLABS_RETRY_ATTEMPTS", "3"),
        ("MAX_CREDITS_PER_ACCOUNT", "10000"),
        ("CREDIT_WARNING_THRESHOLD", "0.8"),
    ]
    
    for key, default in configs:
        value = os.getenv(key, default)
        status = "✅" if os.getenv(key) else "⚠️ (using default)"
        print(f"{status} {key}: {value}")

def test_config_import():
    """Test importing the configuration."""
    print("\n🧪 Configuration Import Test:")
    print("-" * 50)
    
    try:
        from core.config import settings
        print("✅ Successfully imported settings")
        
        # Test ElevenLabs config
        config_valid, config_message = settings.validate_elevenlabs_config()
        if config_valid:
            print(f"✅ Configuration validation: {config_message}")
        else:
            print(f"❌ Configuration validation failed: {config_message}")
            
        # Show detailed status
        status = settings.elevenlabs_config_status
        print(f"\n📊 Configuration Status:")
        for key, value in status.items():
            icon = "✅" if value else "❌"
            print(f"  {icon} {key}: {value}")
            
    except Exception as e:
        print(f"❌ Failed to import configuration: {e}")
        return False
    
    return True

def test_service_initialization():
    """Test ElevenLabs service initialization."""
    print("\n🚀 Service Initialization Test:")
    print("-" * 50)
    
    try:
        from services.elevenlabs import ElevenLabsService
        service = ElevenLabsService()
        print("✅ ElevenLabs service initialized successfully")
        
        # Get service status
        status = service.get_configuration_status()
        print(f"\n📊 Service Status:")
        print(f"  ✅ API Keys: {status['total_api_keys']}")
        print(f"  ✅ Parallel Processing: {status['parallel_enabled']}")
        print(f"  ✅ Max Concurrent: {status['max_concurrent']}")
        print(f"  ✅ Voice ID: {status['voice_id']}")
        print(f"  ✅ Model ID: {status['model_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False

def provide_recommendations(keys_found, single_key):
    """Provide configuration recommendations."""
    print("\n💡 Recommendations:")
    print("-" * 50)
    
    total_keys = len(keys_found) + (1 if single_key else 0)
    
    if total_keys == 0:
        print("🚨 CRITICAL: No ElevenLabs API keys configured!")
        print("   Action: Add at least ELEVENLABS_API_KEY_1 to your .env file")
        print("   Example: ELEVENLABS_API_KEY_1=your_api_key_here")
    elif total_keys == 1:
        print("⚠️  PERFORMANCE: Only 1 API key configured")
        print("   Action: Add more keys (ELEVENLABS_API_KEY_2, etc.) for better throughput")
        print("   Benefit: Parallel processing and credit rotation")
    elif total_keys < 4:
        print(f"✅ GOOD: {total_keys} API keys configured")
        print(f"   Optional: Add {4 - total_keys} more keys for optimal performance")
    else:
        print("🎯 EXCELLENT: Optimal configuration with multiple API keys")
        print("   Status: Ready for high-throughput processing")

def main():
    """Main diagnostic function."""
    print("🔍 ElevenLabs Configuration Diagnostic")
    print("=" * 60)
    
    # Load environment
    if not load_environment():
        sys.exit(1)
    
    # Check API keys
    keys_found, single_key = check_elevenlabs_keys()
    
    # Check other config
    check_other_config()
    
    # Test configuration import
    if not test_config_import():
        sys.exit(1)
    
    # Test service initialization
    if not test_service_initialization():
        sys.exit(1)
    
    # Provide recommendations
    provide_recommendations(keys_found, single_key)
    
    print("\n✅ Diagnostic complete!")

if __name__ == "__main__":
    main() 