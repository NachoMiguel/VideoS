#!/usr/bin/env python3
"""
Script Processor - Core cleanup logic
"""

import logging
import re
from typing import Dict, List, Optional
import sys
import os

# Add backend to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Now import the services
from services.text_cleaner import VoiceOptimizedTextCleaner
from services.entity_variation_manager import entity_variation_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ScriptProcessor:
    """Core script processing and cleanup."""
    
    def __init__(self):
        self.text_cleaner = VoiceOptimizedTextCleaner()
        self.original_script = ""
        self.cleaned_script = ""
        
    async def process_script(self, script_content: str) -> str:
        """Process script through all cleanup steps."""
        
        self.original_script = script_content
        logger.info("🎯 Starting script cleanup pipeline")
        
        # Step 1: Text Cleaner
        logger.info("Step 1: Text Cleaner")
        step1_script = self.text_cleaner.clean_for_voice(script_content)
        logger.info(f"   Length: {len(script_content)} → {len(step1_script)} characters")
        
        # Step 2: Extract entities FIRST (before corrections)
        logger.info("Step 2: Entity Extraction")
        entities = self._extract_entities_from_script(step1_script)
        logger.info(f"   Found entities: {entities}")
        
        # Step 3: Name Corrections (fix ASR mistakes)
        logger.info("Step 3: Name Corrections")
        step3_script = await self._apply_name_corrections(step1_script)
        logger.info(f"   Length: {len(step1_script)} → {len(step3_script)} characters")
        
        # Step 4: Entity Variations (apply natural speech patterns)
        logger.info("Step 4: Entity Variations")
        step4_script = await self._apply_entity_variations(step3_script, entities)
        logger.info(f"   Length: {len(step3_script)} → {len(step4_script)} characters")
        
        # Step 5: TTS Optimization
        logger.info("Step 5: TTS Optimization")
        step5_script = await self._optimize_for_tts(step4_script)
        logger.info(f"   Length: {len(step4_script)} → {len(step5_script)} characters")
        
        # Step 6: Final Validation
        logger.info("Step 6: Final Validation")
        final_script = await self._validate_for_tts(step5_script)
        logger.info(f"   Length: {len(step5_script)} → {len(final_script)} characters")
        
        self.cleaned_script = final_script
        logger.info("✅ Script cleanup completed")
        
        return final_script
    
    async def _apply_name_corrections(self, script: str) -> str:
        """Apply basic name corrections."""
        corrections = {
            "JeanClaude Jean-Claude Van Damme": "Jean-Claude Van Damme",
            "Steven Steven Seagal": "Steven Seagal",
            "JeanClaude Van Damme": "Jean-Claude Van Damme",
            "Vanam": "Jean-Claude Van Damme",
            "Seagull": "Steven Seagal",
        }
        
        corrected_script = script
        for mistake, correction in corrections.items():
            if mistake in corrected_script:
                corrected_script = corrected_script.replace(mistake, correction)
                logger.info(f"   Applied correction: '{mistake}' → '{correction}'")
        
        return corrected_script
    
    async def _apply_entity_variations(self, script: str, entities: List[str]) -> str:
        """Apply entity variations for natural speech."""
        if entities:
            logger.info(f"   Applying variations for entities: {entities}")
            
            # Reset and register entities
            entity_variation_manager.reset_mentions()
            entity_variation_manager.register_entities(entities)
            
            # Apply variations
            varied_script = entity_variation_manager.apply_variations_to_text(script)
            
            # Log what changed
            stats = entity_variation_manager.get_statistics()
            logger.info(f"   Entity variations applied: {stats['total_mentions']} mentions")
            
            return varied_script
        else:
            logger.info("   No entities found for variation")
            return script
    
    def _extract_entities_from_script(self, script: str) -> List[str]:
        """Extract ONLY proper entity names from script content."""
        import re
        
        # Clean script first (remove punctuation, normalize)
        clean_script = re.sub(r'[^\w\s\-]', ' ', script)
        clean_script = re.sub(r'\s+', ' ', clean_script)
        
        # STRICT patterns for proper names only
        name_patterns = [
            # Full names with hyphens: Jean-Claude Van Damme
            r'\b[A-Z][a-z]+(?:\-[A-Z][a-z]+)*\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            # Standard two-part names: Steven Seagal
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        ]
        
        entities = []
        for pattern in name_patterns:
            matches = re.findall(pattern, clean_script)
            entities.extend(matches)
        
        # STRICT filtering
        common_words = {
            'The', 'This', 'That', 'With', 'And', 'But', 'For', 'You', 'All', 'New', 'Now', 'Old',
            'Would', 'Could', 'Should', 'Will', 'Can', 'May', 'Might', 'Must', 'Yet', 'Still',
            'Others', 'Some', 'Many', 'Few', 'Each', 'Every', 'Any', 'All', 'Both', 'Either'
        }
        
        filtered_entities = []
        for entity in entities:
            # Skip if contains common words
            if any(word in entity for word in common_words):
                continue
            # Skip if too short
            if len(entity) < 8:
                continue
            # Skip if contains duplicate words
            words = entity.split()
            if len(words) != len(set(words)):
                continue
            # Skip duplicates
            if entity not in filtered_entities:
                filtered_entities.append(entity)
        
        logger.info(f" Raw entities found: {entities}")
        logger.info(f" Filtered entities: {filtered_entities}")
        
        return filtered_entities[:5]  # Limit to top 5
    
    async def _optimize_for_tts(self, script: str) -> str:
        """Optimize script for TTS."""
        optimizations = {
            "#": "", "@": "", "\\": "", "|": "", "^": "", "~": "", "=": "",
            "<": "", ">": "", "$": "", "€": "", "£": "", "¥": "", "¢": "",
            "Mr.": "Mister", "Mrs.": "Missus", "Dr.": "Doctor", 
            "Prof.": "Professor", "vs.": "versus", "etc.": "and so on",
            "i.e.": "that is", "e.g.": "for example",
        }
        
        optimized_script = script
        for pattern, replacement in optimizations.items():
            if pattern in optimized_script:
                optimized_script = optimized_script.replace(pattern, replacement)
                logger.info(f"   Applied optimization: '{pattern}' → '{replacement}'")
        
        # Convert numbers to words
        import re
        number_patterns = [
            (r'\b(\d+)\b', lambda m: self._number_to_words(int(m.group(1)))),
        ]
        
        for pattern, replacement_func in number_patterns:
            optimized_script = re.sub(pattern, replacement_func, optimized_script)
        
        return optimized_script
    
    def _number_to_words(self, num: int) -> str:
        """Convert numbers to words."""
        if num == 0:
            return "zero"
        elif num < 20:
            words = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
            return words[num]
        elif num < 100:
            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            if num % 10 == 0:
                return tens[num // 10]
            else:
                return f"{tens[num // 10]} {self._number_to_words(num % 10)}"
        elif num < 1000:
            if num % 100 == 0:
                return f"{self._number_to_words(num // 100)} hundred"
            else:
                return f"{self._number_to_words(num // 100)} hundred {self._number_to_words(num % 100)}"
        else:
            return str(num)
    
    async def _validate_for_tts(self, script: str) -> str:
        """Final validation and cleanup."""
        # Remove multiple spaces
        script = re.sub(r'\s+', ' ', script)
        
        # Remove leading/trailing whitespace
        script = script.strip()
        
        # Ensure script is not empty
        if not script:
            raise ValueError("Script is empty after processing")
        
        logger.info(f"   Final validation completed")
        return script
    
    def show_comparison(self, original: str, cleaned: str):
        """Show before/after comparison."""
        print("\n📊 BEFORE/AFTER COMPARISON:")
        print("=" * 50)
        
        # Show sample differences
        original_lines = original.split('\n')[:3]
        cleaned_lines = cleaned.split('\n')[:3]
        
        print("ORIGINAL (first 3 lines):")
        for line in original_lines:
            print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
        
        print("\nCLEANED (first 3 lines):")
        for line in cleaned_lines:
            print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
        
        print(f"\n📊 Length change: {len(original)} → {len(cleaned)} characters")
        
        # Show specific improvements
        print("\n🔧 IMPROVEMENTS APPLIED:")
        print("=" * 30)
        
        # Check for repetitive names
        if "Jean-Claude Van Damme" in original and "Jean-Claude Van Damme" in cleaned:
            original_count = original.count("Jean-Claude Van Damme")
            cleaned_count = cleaned.count("Jean-Claude Van Damme")
            if original_count > cleaned_count:
                print(f"   ✅ Reduced repetitive names: {original_count} → {cleaned_count}")
        
        # Check for abbreviations
        if "Mr." in original and "Mister" in cleaned:
            print("   ✅ Converted abbreviations (Mr. → Mister)")
        
        # Check for numbers
        import re
        numbers_in_original = len(re.findall(r'\b\d+\b', original))
        if numbers_in_original > 0:
            print(f"   ✅ Converted {numbers_in_original} numbers to words") 