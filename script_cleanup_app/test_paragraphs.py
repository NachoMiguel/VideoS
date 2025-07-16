#!/usr/bin/env python3
"""Test paragraph formatting directly"""

def test_paragraph_formatting():
    # Sample text
    text = "This is sentence one. This is sentence two. This is sentence three. The scene was set. This is sentence four. This is sentence five."
    
    # Simple paragraph formatting
    sentences = text.split('. ')
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        
        # Break on "The scene was set"
        if "the scene was set" in sentence.lower():
            if current_paragraph:
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
    
    # Add remaining
    if current_paragraph:
        paragraphs.append('. '.join(current_paragraph) + '.')
    
    # Join with double line breaks
    result = '\n\n'.join(paragraphs)
    
    print("ORIGINAL:")
    print(text)
    print("\n" + "="*50 + "\n")
    print("FORMATTED:")
    print(result)
    print("\n" + "="*50 + "\n")
    print("LENGTH:", len(result))
    print("PARAGRAPHS:", len(paragraphs))
    print("CONTAINS \\n\\n:", "\\n\\n" in result)
    
    # Write to file
    with open('test_output.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    print("Written to test_output.txt")

if __name__ == "__main__":
    test_paragraph_formatting() 