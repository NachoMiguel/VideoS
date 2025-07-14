#!/usr/bin/env python3
"""
YouTube Service Diagnostic Tool
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_basic_yt_dlp():
    """Test basic yt-dlp functionality."""
    print("\n🔍 Testing basic yt-dlp...")
    try:
        import yt_dlp
        
        # Test with Rick Roll
        video_url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            print(f"✅ Video title: {info.get('title', 'Unknown')}")
            print(f"✅ Duration: {info.get('duration', 'Unknown')} seconds")
            
            # Check captions
            subtitles = info.get('subtitles', {})
            auto_captions = info.get('automatic_captions', {})
            
            print(f"✅ Manual subtitles: {list(subtitles.keys())}")
            print(f"✅ Auto captions: {list(auto_captions.keys())}")
            
            has_captions = bool(subtitles or auto_captions)
            print(f"✅ Has captions: {has_captions}")
            
            return True
            
    except Exception as e:
        print(f"❌ yt-dlp test failed: {e}")
        return False

async def test_youtube_transcript_api():
    """Test youtube-transcript-api."""
    print("\n🔍 Testing youtube-transcript-api...")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Test with Rick Roll
        video_id = "dQw4w9WgXcQ"
        
        # Try to get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print(f"✅ Transcript segments: {len(transcript)}")
        print(f"✅ First segment: {transcript[0] if transcript else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ youtube-transcript-api test failed: {e}")
        return False

async def test_youtube_service():
    """Test our YouTube service."""
    print("\n🔍 Testing YouTube service...")
    try:
        from services.youtube import YouTubeService
        
        service = YouTubeService()
        video_id = "dQw4w9WgXcQ"
        
        # Test diagnosis
        diagnosis = await service.diagnose_transcript_issue(video_id)
        print(f"✅ Diagnosis complete:")
        print(f"   Video exists: {diagnosis['video_exists']}")
        print(f"   Has captions: {diagnosis['has_captions']}")
        print(f"   Available languages: {diagnosis['available_languages']}")
        print(f"   API errors: {diagnosis['api_errors']}")
        
        return True
        
    except Exception as e:
        print(f"❌ YouTube service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main diagnostic."""
    print("🔍 YouTube Service Diagnostic")
    print("=" * 50)
    
    # Test basic components
    basic_ok = await test_basic_yt_dlp()
    api_ok = await test_youtube_transcript_api()
    service_ok = await test_youtube_service()
    
    print(f"\n📊 RESULTS:")
    print(f"   yt-dlp: {'✅' if basic_ok else '❌'}")
    print(f"   youtube-transcript-api: {'✅' if api_ok else '❌'}")
    print(f"   YouTube service: {'✅' if service_ok else '❌'}")
    
    if not (basic_ok and api_ok and service_ok):
        print("\n🚨 CRITICAL: YouTube extraction is broken!")
        return False
    else:
        print("\n✅ All components working!")
        return True

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)