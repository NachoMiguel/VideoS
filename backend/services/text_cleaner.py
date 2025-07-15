#!/usr/bin/env python3
"""
Senior Engineer Solution: Comprehensive Text Cleaning for Voice Generation
"""

import re
import unicodedata

class VoiceOptimizedTextCleaner:
    """Clean text specifically for high-quality voice generation."""
    
    def __init__(self):
        self.encoding_fixes = {
            # Common UTF-8 corruption patterns
            'â': '—',           # Em dash
            'â€™': "'",         # Right single quotation
            'â€œ': '"',         # Left double quotation  
            'â€': '"',          # Right double quotation
            'â€¦': '...',       # Horizontal ellipsis
            'â€"': '—',         # Em dash
            'â€"': '–',         # En dash
            'Ã¡': 'á',          # Latin small letter a with acute
            'Ã©': 'é',          # Latin small letter e with acute
            'Ã­': 'í',          # Latin small letter i with acute
            'Ã³': 'ó',          # Latin small letter o with acute
            'Ãº': 'ú',          # Latin small letter u with acute
            'Ã±': 'ñ',          # Latin small letter n with tilde
        }
        
        self.voice_unfriendly_chars = {
            # Replace with voice-friendly alternatives
            '—': ' - ',          # Em dash to dash with spaces
            '–': ' - ',          # En dash to dash with spaces  
            '…': '...',          # Ellipsis to three periods
            '"': '"',            # Curly quotes to straight
            '"': '"',
            ''': "'",
            ''': "'",
            '•': '',             # Remove bullet points
            '◦': '',
            '▪': '',
            '§': 'section',      # Section symbol
            '©': 'copyright',    # Copyright symbol
            '®': 'registered',   # Registered trademark
            '™': 'trademark',    # Trademark
            '°': ' degrees',     # Degree symbol
            '%': ' percent',     # Better pronunciation
        }

    def clean_for_voice(self, text: str) -> str:
        """Comprehensive cleaning for voice generation."""
        
        # Step 1: Fix encoding corruption
        cleaned = self._fix_encoding_corruption(text)
        
        # Step 2: Normalize Unicode  
        cleaned = self._normalize_unicode(cleaned)
        
        # Step 3: Replace voice-unfriendly characters
        cleaned = self._replace_voice_unfriendly_chars(cleaned)
        
        # Step 4: Clean whitespace and formatting
        cleaned = self._clean_formatting(cleaned)
        
        # Step 5: Validate result
        self._validate_voice_readiness(cleaned)
        
        return cleaned

    def format_script_for_display(self, script: str) -> str:
        """Format script for clean display - removes metadata and improves paragraph structure."""
        
        # Step 1: Remove metadata headers if present
        cleaned = self._remove_metadata_headers(script)
        
        # Step 2: Remove any section headings
        cleaned = self._remove_section_headings(cleaned)
        
        # Step 3: Improve paragraph structure
        cleaned = self._improve_paragraph_structure(cleaned)
        
        # Step 4: Clean up formatting
        cleaned = self._clean_script_formatting(cleaned)
        
        # Step 5: Apply voice cleaning
        cleaned = self.clean_for_voice(cleaned)
        
        return cleaned

    def _remove_metadata_headers(self, script: str) -> str:
        """Remove metadata headers and keep only the actual script content."""
        
        # Remove common metadata patterns
        patterns_to_remove = [
            r'=== GENERATED SCRIPT ===.*?=== SCRIPT CONTENT ===',
            r'Session ID:.*?\n',
            r'Video ID:.*?\n', 
            r'YouTube URL:.*?\n',
            r'Generated:.*?\n',
            r'Script Length:.*?\n',
            r'Prompt Used:.*?\n',
            r'=== END SCRIPT ===',
            r'=== SCRIPT CONTENT ===',
        ]
        
        cleaned = script
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        return cleaned.strip()

    def _improve_paragraph_structure(self, script: str) -> str:
        """Improve paragraph structure for better readability."""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', script)
        
        # Group sentences into logical paragraphs
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Start new paragraph for certain keywords (section breaks)
            section_indicators = [
                'introduction', 'conclusion', 'background', 'however', 
                'meanwhile', 'later', 'finally', 'additionally', 'furthermore',
                'moreover', 'nevertheless', 'conversely', 'subsequently'
            ]
            
            sentence_lower = sentence.lower()
            should_start_new_paragraph = any(
                indicator in sentence_lower for indicator in section_indicators
            )
            
            # Also start new paragraph if current one is getting long (>3 sentences)
            if should_start_new_paragraph or len(current_paragraph) >= 3:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            current_paragraph.append(sentence)
        
        # Add final paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with double line breaks
        return '\n\n'.join(paragraphs)

    def _clean_script_formatting(self, script: str) -> str:
        """Clean up script-specific formatting issues."""
        
        # Remove excessive whitespace
        script = re.sub(r'\s+', ' ', script)
        
        # Fix spacing around punctuation
        script = re.sub(r'\s+([,.!?;:])', r'\1', script)
        script = re.sub(r'([,.!?;:])\s*', r'\1 ', script)
        
        # Remove multiple consecutive punctuation
        script = re.sub(r'[.]{4,}', '...', script)
        script = re.sub(r'[!]{2,}', '!', script)
        script = re.sub(r'[?]{2,}', '?', script)
        
        # Clean up quotes and parentheses
        script = re.sub(r'"+', '"', script)
        script = re.sub(r"'+", "'", script)
        script = re.sub(r'\(\s+', '(', script)
        script = re.sub(r'\s+\)', ')', script)
        
        # Remove any remaining special formatting characters
        script = re.sub(r'[*#$%^&+=]', '', script)
        
        return script.strip()

    def _remove_section_headings(self, script: str) -> str:
        """Remove any section headings or chapter titles that might be in the script."""
        
        # Remove common section heading patterns
        heading_patterns = [
            r'^\s*(introduction|conclusion|background|overview|summary|chapter|section|part)\s*[:\-]?\s*$',
            r'^\s*[A-Z][A-Z\s]+[:\-]?\s*$',  # ALL CAPS headings
            r'^\s*\*\*[^*]+\*\*\s*$',  # **Heading** format
            r'^\s*#[^#]+\s*$',  # #Heading format
            r'^\s*[0-9]+\.\s*[A-Z][^.]*$',  # 1. Heading format
        ]
        
        lines = script.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if this line matches any heading pattern
            is_heading = any(re.match(pattern, line, re.IGNORECASE) for pattern in heading_patterns)
            
            if not is_heading:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _fix_encoding_corruption(self, text: str) -> str:
        """Fix common UTF-8 encoding corruption."""
        for corrupted, correct in self.encoding_fixes.items():
            text = text.replace(corrupted, correct)
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to prevent TTS issues."""
        # Decompose and recompose to fix any Unicode issues
        text = unicodedata.normalize('NFKC', text)
        return text

    def _replace_voice_unfriendly_chars(self, text: str) -> str:
        """Replace characters that cause TTS problems."""
        for unfriendly, friendly in self.voice_unfriendly_chars.items():
            text = text.replace(unfriendly, friendly)
        return text

    def _clean_formatting(self, text: str) -> str:
        """Clean up formatting for better voice flow."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[.]{4,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Clean up quotes
        text = re.sub(r'"+', '"', text)
        text = re.sub(r"'+", "'", text)
        
        return text.strip()

    def _validate_voice_readiness(self, text: str) -> None:
        """Validate text is ready for voice generation."""
        problematic_patterns = [
            (r'[^\x00-\x7F\u00C0-\u017F\u0100-\u024F]', 'Non-Latin characters detected'),
            (r'â', 'Encoding corruption still present'),
            (r'[^\w\s.,!?;:\'"()&\-\[\]]', 'Special characters that may break TTS'),
        ]
        
        for pattern, warning in problematic_patterns:
            if re.search(pattern, text):
                print(f"⚠️ WARNING: {warning}")
                matches = re.findall(pattern, text)
                print(f"   Found: {set(matches)}")

# Global instance
text_cleaner = VoiceOptimizedTextCleaner() 