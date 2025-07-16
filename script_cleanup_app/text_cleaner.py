#!/usr/bin/env python3
"""Minimal text cleaner for script cleanup app"""

import re
import unicodedata

class VoiceOptimizedTextCleaner:
    def __init__(self):
        self.voice_unfriendly_chars = {
            '—': ' - ', '–': ' - ', '…': '...', '"': '"', '"': '"',
            ''': "'", ''': "'", '•': '', '◦': '', '▪': '',
            '§': 'section', '©': 'copyright', '®': 'registered',
            '™': 'trademark', '°': ' degrees', '%': ' percent',
        }

    def clean_for_voice(self, text: str) -> str:
        # Fix encoding corruption
        text = self._fix_encoding_corruption(text)
        # Replace voice-unfriendly characters
        text = self._replace_voice_unfriendly_chars(text)
        # Clean formatting
        text = self._clean_formatting(text)
        return text

    def _fix_encoding_corruption(self, text: str) -> str:
        encoding_fixes = {
            'â': '—', 'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€¦': '...',
            'â€"': '—', 'â€"': '–', 'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í',
            'Ã³': 'ó', 'Ãº': 'ú', 'Ã±': 'ñ'
        }
        for mistake, correction in encoding_fixes.items():
            text = text.replace(mistake, correction)
        return text

    def _replace_voice_unfriendly_chars(self, text: str) -> str:
        for char, replacement in self.voice_unfriendly_chars.items():
            text = text.replace(char, replacement)
        return text

    def _clean_formatting(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip() 