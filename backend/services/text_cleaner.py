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