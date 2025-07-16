import asyncio
import logging
from pathlib import Path
from services.elevenlabs import get_elevenlabs_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_pipeline():
    """Test the actual production pipeline with your real script file."""
    
    # Read your actual script file
    script_file_path = Path("../script_tests/generated_script_20250715_140819_okksWNdNaTk.txt")
    
    if not script_file_path.exists():
        print(f"❌ Script file not found: {script_file_path}")
        return
    
    print("�� TESTING REAL PRODUCTION PIPELINE")
    print("=" * 80)
    print(f"📁 Script file: {script_file_path}")
    
    try:
        # Read the actual script
        with open(script_file_path, 'r', encoding='utf-8') as f:
            real_script = f.read()
        
        print(f"📝 Script length: {len(real_script)} characters")
        print(f"🔍 ASR mistakes to fix: 'JeanClaude Jean-Claude Van Damme', 'Steven Steven Seagal'")
        print("=" * 80)
        
        # Initialize real ElevenLabs service
        elevenlabs_service = get_elevenlabs_service()
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        output_path = str(output_dir / "real_script_audio.mp3")
        
        print("🔄 Starting REAL audio generation...")
        print("   This will use the actual production pipeline:")
        print("   1. TTS preprocessing (name correction, cleanup)")
        print("   2. Script segmentation")
        print("   3. Audio generation per segment")
        print("   4. Audio combination")
        print("=" * 80)
        
        # Run the ACTUAL pipeline with your real script
        final_audio_path = await elevenlabs_service.generate_audio_from_script(
            script_content=real_script,
            output_path=output_path
        )
        
        print("=" * 80)
        print("🎯 REAL PIPELINE RESULTS:")
        print("=" * 80)
        print(f"✅ Audio generated successfully!")
        print(f"✅ Output file: {final_audio_path}")
        print(f"✅ File exists: {Path(final_audio_path).exists()}")
        
        if Path(final_audio_path).exists():
            file_size = Path(final_audio_path).stat().st_size
            print(f"✅ File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        print("\n�� REAL PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("🎵 You can now play the generated audio file!")
        print(f"📁 Audio file location: {final_audio_path}")
        
    except Exception as e:
        print(f"❌ REAL PIPELINE TEST FAILED: {str(e)}")
        logger.error(f"Real pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_pipeline()) 