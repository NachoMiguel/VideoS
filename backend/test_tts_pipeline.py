#!/usr/bin/env python3
"""
TTS Pipeline Testing Script
Tests the complete TTS preprocessing pipeline with the real script.
"""

import asyncio
import logging
from services.elevenlabs import get_elevenlabs_service
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tts_pipeline():
    """Test the complete TTS pipeline with your existing script."""
    
    # Your existing script with ASR mistakes
    test_script = """
    In the world of action cinema, a rivalry simmered quietly for decades. It involved two larger-than-life figures: JeanClaude Jean-Claude Van Damme, known for his grace and power, and Steven Steven Seagal, the enigmatic martial artist with a penchant for wry comments. Their silent war was a fascinating dance, marked by Steven Seagal's subtle jabs and Jean-Claude Van Damme's dignified silence. It all began with Steven Seagal's casual interviews. He dropped comments that seemed innocent at first but carried a sting. "Jean-Claude Van Damme, he's a dancer, isn't he? " Steven Seagal would say with a sly smile. To the untrained ear, it sounded like admiration. But to those who knew, it was a calculated move to paint Jean-Claude Van Damme as soft, mere ballet in a world of gritty action. Jean-Claude Van Damme chose not to respond. For years, he watched the narrative form around him. He was the silent hero, the one who let his actions speak louder than words. His choice was deliberate, a strategic silence that maintained his dignity and mystique. The tension was palpable, a quiet storm brewing beneath the surface. Years went by, and fans speculated. Why didn't Jean-Claude Van Damme strike back? Was it respect, or perhaps something deeper? The silence was intriguing, drawing people in, like moths to a flame. It wasn't weakness - it was strength, a testament to his character. The longer he remained quiet, the more the world wanted to hear his voice. And then, the moment arrived. JeanClaude Jean-Claude Van Damme finally broke his silence on Steven Steven Seagal, a revelation that sent ripples through the industry. "Calling him just a dancer was a calculated move, " he said, his voice steady but charged with emotion. The words were simple, yet they carried the weight of years. They exposed the truth, unraveling the facade Steven Seagal had built. For Jean-Claude Van Damme, it was never about the insults. It was about choosing his battles, knowing when to speak and when to let silence do the talking. His decision to finally address the rivalry wasn't an act of aggression; it was a reclaiming of his narrative. It was as if he had been waiting for the perfect moment, a master strategist in the game of public perception. The industry watched, fascinated by this rare glimpse behind the curtain. Jean-Claude Van Damme wasn't just a dancer, and he wasn't just an action star. He was a man of depth, someone who understood the power of words and the greater power of silence. His fans celebrated his triumph, not over Steven Seagal, but over the need to conform to expectations. As the dust settled, one thing became clear: the rivalry had been a silent war, but it was never about hatred or malice. It was about respect and the unspoken rules of their world. Jean-Claude Van Damme's silence had been his greatest weapon, a shield against the noise. Steven Seagal's comments, once sharp and pointed, seemed to lose their edge. In the end, it wasn't about who was the better fighter or who had the last word. It was about understanding one's place in the narrative and choosing to write one's own story. Jean-Claude Van Damme's decision to speak was a moment of clarity, a declaration that he was more than the sum of others' opinions. The rivalry between JeanClaude Jean-Claude Van Damme and Steven Steven Seagal was a tale of contrasts - a clash between the outspoken and the reserved, the calculated and the spontaneous. It was a story that captivated audiences, not with explosive confrontations, but with the tension of the unspoken. Jean-Claude Van Damme's silence had been a mystery, a question mark hanging in the air. When he finally spoke, it was not just a revelation; it was a resolution. The silent war had ended, not with a bang, but with a whisper. And in that whisper, the world heard the true voice of JeanClaude Jean-Claude Van Damme. The mystery was solved, but the allure of his silence would linger on, a reminder of the power of words - and the power of choosing when to use them.
    """
    
    print("🎯 TESTING TTS PIPELINE")
    print("=" * 80)
    print(f"📝 Original script length: {len(test_script)} characters")
    print(f"🔍 ASR mistakes to fix: 'JeanClaude Jean-Claude Van Damme', 'Steven Steven Seagal'")
    print("=" * 80)
    
    try:
        # Initialize ElevenLabs service
        elevenlabs_service = get_elevenlabs_service()
        
        # Test TTS preprocessing only (no audio generation)
        print("🔄 Testing TTS preprocessing...")
        cleaned_script = await elevenlabs_service._preprocess_script_for_tts(test_script)
        
        print("=" * 80)
        print(" RESULTS:")
        print("=" * 80)
        print(f"✅ Original length: {len(test_script)} characters")
        print(f"✅ Cleaned length: {len(cleaned_script)} characters")
        
        # Check for corrections
        original_mistakes = [
            "JeanClaude Jean-Claude Van Damme",
            "Steven Steven Seagal"
        ]
        
        corrections_found = []
        for mistake in original_mistakes:
            if mistake not in cleaned_script:
                corrections_found.append(f"✅ Fixed: '{mistake}'")
            else:
                corrections_found.append(f"❌ Still present: '{mistake}'")
        
        print("\n🔧 CORRECTIONS:")
        for correction in corrections_found:
            print(f"   {correction}")
        
        # Show sample of cleaned script
        print("\n📝 SAMPLE OF CLEANED SCRIPT:")
        print("-" * 40)
        print(cleaned_script[:500] + "..." if len(cleaned_script) > 500 else cleaned_script)
        print("-" * 40)
        
        print("\n🎯 TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_tts_pipeline()) 