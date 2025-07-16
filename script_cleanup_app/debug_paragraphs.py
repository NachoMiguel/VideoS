#!/usr/bin/env python3
"""Debug paragraph formatting"""

def debug_paragraph_formatting():
    # Test the exact logic from the script processor
    text = "This is sentence one. This is sentence two. This is sentence three. The scene was set. This is sentence four. This is sentence five."
    
    sentences = text.split('. ')
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        
        if "the scene was set" in sentence.lower():
            if current_paragraph:
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
    
    if current_paragraph:
        paragraphs.append('. '.join(current_paragraph) + '.')
    
    print("PARAGRAPHS CREATED:")
    for i, p in enumerate(paragraphs):
        print(f"Paragraph {i+1}: '{p}'")
    
    print(f"\nJOINING WITH '\\n\\n':")
    result = '\n\n'.join(paragraphs)
    
    print(f"RESULT: '{result}'")
    print(f"LENGTH: {len(result)}")
    print(f"REPR: {repr(result)}")
    
    # Check if \n\n is actually there
    if '\n\n' in result:
        print("✅ \\n\\n found in result")
    else:
        print("❌ \\n\\n NOT found in result")
    
    # Write to file and check
    with open('debug_output.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    # Read back and check
    with open('debug_output.txt', 'r', encoding='utf-8') as f:
        read_back = f.read()
    
    print(f"\nREAD BACK: '{read_back}'")
    print(f"READ BACK REPR: {repr(read_back)}")

if __name__ == "__main__":
    debug_paragraph_formatting() 