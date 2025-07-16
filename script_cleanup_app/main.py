#!/usr/bin/env python3
"""
Script Cleanup Application
Standalone app to apply all cleanup processes to a manually created script.
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
    
    # Get input script path
    if len(sys.argv) != 2:
        print("❌ Usage: python main.py <input_script_path>")
        print("Example: python main.py input/my_script.txt")
        print("Example: python main.py ../script_tests/generated_script_20250715_140819_okksWNdNaTk.txt")
        return
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Initialize processor
    processor = ScriptProcessor()
    
    # Process the script
    print(f"�� Processing: {input_path}")
    print(f"📊 File size: {input_path.stat().st_size} bytes")
    
    try:
        # Read input script
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_script = f.read()
        
        print(f"�� Raw script length: {len(raw_script)} characters")
        
        # Extract video context if video_id is provided
        video_context = None
        if len(sys.argv) >= 3:
            video_id = sys.argv[2]
            try:
                from services.youtube import YouTubeService
                youtube_service = YouTubeService()
                video_context = await youtube_service.extract_video_context(video_id)
                print(f"✅ Extracted video context for entity variations")
            except Exception as e:
                print(f"Failed to extract video context: {str(e)}")
        
        # Process script with video context
        cleaned_script = await processor.process_script(raw_script, video_context)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = input_path.stem
        output_path = Path("output") / f"cleaned_{original_name}_{timestamp}.txt"
        output_path.parent.mkdir(exist_ok=True)
        
        # Save cleaned script
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_script)
        
        print(f"\n✅ CLEANUP COMPLETE")
        print(f"�� Output saved to: {output_path}")
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