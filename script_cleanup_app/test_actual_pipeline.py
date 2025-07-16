#!/usr/bin/env python3
"""Test the actual script cleanup pipeline"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from script_processor import ScriptProcessor

async def test_actual_pipeline():
    """Test the actual pipeline with a real script."""
    
    # Read the actual script
    with open('input/generated_script_20250715_140819_okksWNdNaTk.txt', 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    print(f"Original script length: {len(script_content)}")
    print(f"Original preview: {script_content[:200]}...")
    
    # Create processor
    processor = ScriptProcessor()
    
    # Test ONLY the paragraph formatting method
    print("\n" + "="*50)
    print("TESTING PARAGRAPH FORMATTING ONLY")
    print("="*50)
    
    formatted_script = processor._format_paragraphs(script_content)
    
    print(f"Formatted script length: {len(formatted_script)}")
    print(f"Formatted preview: {formatted_script[:300]}...")
    
    # Check for line breaks
    if '\n\n' in formatted_script:
        print("✅ Line breaks found in formatted script")
        paragraph_count = formatted_script.count('\n\n') + 1
        print(f"Number of paragraphs: {paragraph_count}")
    else:
        print("❌ NO LINE BREAKS found in formatted script")
    
    # Write to test file
    with open('test_actual_output.txt', 'w', encoding='utf-8') as f:
        f.write(formatted_script)
    
    print("Written to test_actual_output.txt")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_actual_pipeline()) 