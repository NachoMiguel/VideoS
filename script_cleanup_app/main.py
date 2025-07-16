#!/usr/bin/env python3
"""
Script Cleanup Application with Video Context Support
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Add backend services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from script_processor import ScriptProcessor

async def main():
    """Main entry point for script cleanup."""
    
    print("🧹 SCRIPT CLEANUP APPLICATION")
    print("=" * 50)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("❌ Usage: python main.py <input_script_path> [video_id]")
        print("Examples:")
        print("  python main.py input/script.txt okksWNdNaTk")
        print("  python main.py input/script.txt")
        return
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Extract video context if video_id provided
    video_context = None
    if len(sys.argv) >= 3:
        video_id = sys.argv[2]
        try:
            from services.youtube import YouTubeService
            youtube_service = YouTubeService()
            print(f" Extracting context from video ID: {video_id}")
            video_context = await youtube_service.extract_video_context(video_id)
            print(f"✅ Video context extracted successfully")
        except Exception as e:
            print(f"❌ Failed to extract video context: {str(e)}")
            print(f"⚠️ Continuing without video context...")
    else:
        print(f"ℹ️ No video ID provided - will use script extraction fallback")
    
    # Initialize processor
    processor = ScriptProcessor()
    
    # Process the script
    print(f" Processing: {input_path}")
    print(f"📊 File size: {input_path.stat().st_size} bytes")
    
    try:
        # Read input script
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_script = f.read()
        
        print(f" Raw script length: {len(raw_script)} characters")
        
        # Process script with context
        cleaned_script = await processor.process_script(raw_script, video_context)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = input_path.stem
        output_path = Path("output") / f"cleaned_{original_name}_{timestamp}.txt"
        output_path.parent.mkdir(exist_ok=True)
        
        # Write the cleaned script with proper line breaks
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(cleaned_script)
        
        print(f"\n✅ CLEANUP COMPLETE")
        print(f" Output saved to: {output_path}")
        print(f"📊 Cleaned script length: {len(cleaned_script)} characters")
        
        # Show before/after comparison
        processor.show_comparison(raw_script, cleaned_script)
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    asyncio.run(main()) 