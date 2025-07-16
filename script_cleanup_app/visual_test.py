#!/usr/bin/env python3
"""Visual test for paragraph formatting"""

def visual_test():
    # Create a test with obvious visual breaks
    text = "This is paragraph one. It has multiple sentences. The scene was set. This is paragraph two. It also has sentences."
    
    # Format with paragraphs
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
    
    # Join with double line breaks
    result = '\n\n'.join(paragraphs)
    
    print("VISUAL TEST:")
    print("="*50)
    print(result)
    print("="*50)
    
    # Write to file
    with open('visual_test.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    print("Written to visual_test.txt")
    print("Open this file in Notepad++ or VS Code to see the line breaks!")

if __name__ == "__main__":
    visual_test() 