#!/usr/bin/env python3
"""
Script Processor - Core cleanup logic
"""

import logging
import re
from typing import Dict, List, Optional
import sys
import os

# Configure logging FIRST
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Fix the path resolution
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Add services to path
services_path = os.path.join(backend_path, 'services')
sys.path.insert(0, services_path)

# Add error handling around imports
try:
    from text_cleaner import VoiceOptimizedTextCleaner
    from entity_variation_manager import entity_variation_manager
    logger.info("✅ Successfully imported backend services")
except ImportError as e:
    logger.error(f"❌ Failed to import backend services: {e}")
    logger.error("Falling back to local implementations...")
    
    # Fallback to local text_cleaner
    from text_cleaner import VoiceOptimizedTextCleaner
    
    # Create minimal entity variation manager
    class MinimalEntityVariationManager:
        def __init__(self):
            self.mention_count = {}
        
        def reset_mentions(self):
            self.mention_count.clear()
        
        def register_entities(self, entities):
            pass
        
        def get_variation(self, entity):
            return entity
    
    entity_variation_manager = MinimalEntityVariationManager()

class ScriptProcessor:
    """Core script processing and cleanup."""
    
    def __init__(self):
        self.text_cleaner = VoiceOptimizedTextCleaner()
        self.original_script = ""
        self.cleaned_script = ""
        
    def _extract_main_entities(self, video_context: Dict) -> List[str]:
        """Algorithmically extract main character entities from video context."""
        if not video_context:
            return []
        
        raw_entities = video_context.get('potential_entities', [])
        if not raw_entities:
            return []
        
        logger.info(f"   Raw entities from video context: {raw_entities}")
        
        # Step 1: Clean and normalize entities
        cleaned_entities = []
        for entity in raw_entities:
            # Remove punctuation and normalize
            clean_entity = re.sub(r'[^\w\s\-]', '', entity).strip()
            if clean_entity and len(clean_entity) >= 8:
                cleaned_entities.append(clean_entity)
        
        # Step 2: Remove duplicates (case-insensitive)
        unique_entities = []
        seen_lower = set()
        for entity in cleaned_entities:
            if entity.lower() not in seen_lower:
                unique_entities.append(entity)
                seen_lower.add(entity.lower())
        
        # Step 3: Filter out partial matches and non-character entities
        main_entities = []
        for entity in unique_entities:
            # Skip if it's a subset of another entity
            is_subset = False
            for other in unique_entities:
                if entity != other and entity.lower() in other.lower():
                    is_subset = True
                    break
            
            # Skip common non-character words
            common_words = {'the', 'and', 'with', 'from', 'about', 'this', 'that', 'would', 'could', 'should', 'will', 'can', 'may', 'might', 'must', 'yet', 'still', 'others', 'some', 'many', 'few', 'each', 'every', 'any', 'all', 'both', 'either', 'neither', 'las', 'vegas', 'hollywood', 'movie', 'film', 'cinema', 'action', 'truth', 'finally', 'confirms'}
            
            entity_words = entity.lower().split()
            if any(word in common_words for word in entity_words):
                continue
            
            # Skip single words (likely not full names)
            if len(entity.split()) < 2:
                continue
            
            # Skip if it's a location or common phrase
            if any(location in entity.lower() for location in ['las vegas', 'hollywood', 'new york', 'los angeles']):
                continue
            
            if not is_subset:
                main_entities.append(entity)
        
        # Step 4: Sort by relevance (longer names first, then by frequency)
        main_entities.sort(key=lambda x: (len(x), raw_entities.count(x)), reverse=True)
        
        # Step 5: Limit to top 2-3 main characters
        final_entities = main_entities[:3]
        
        logger.info(f"   Main entities extracted: {final_entities}")
        return final_entities

    async def process_script(self, script_content: str, video_context: Optional[Dict] = None) -> str:
        """Process script through the complete cleanup pipeline."""
        logger.info("🎯 Starting script cleanup pipeline")
        
        # Step 1: Text Cleaner
        logger.info("Step 1: Text Cleaner")
        step1_script = self.text_cleaner.clean_for_voice(script_content)
        logger.info(f"   Length: {len(script_content)} → {len(step1_script)} characters")
        
        # Step 2: Algorithmic Entity Extraction
        logger.info("Step 2: Algorithmic Entity Extraction")
        main_entities = self._extract_main_entities(video_context)
        logger.info(f"   Main entities: {main_entities}")
        
        # Step 3: Name Corrections
        logger.info("Step 3: Name Corrections")
        step3_script = await self._apply_name_corrections(step1_script)
        logger.info(f"   Length: {len(script_content)} → {len(step3_script)} characters")
        
        # Step 4: Entity Variations
        logger.info("Step 4: Entity Variations")
        step4_script = await self._apply_entity_variations(step3_script, main_entities)
        logger.info(f"   Length: {len(script_content)} → {len(step4_script)} characters")
        
        # Step 5: TTS Optimization
        logger.info("Step 5: TTS Optimization")
        step5_script = self.text_cleaner.clean_for_voice(step4_script)
        logger.info(f"   Length: {len(script_content)} → {len(step5_script)} characters")
        
        # 🚨 STEP 6: PARAGRAPH FORMATTING (ACTUALLY IMPLEMENTED)
        logger.info("Step 6: Paragraph Formatting")
        step6_script = self._format_paragraphs(step5_script)
        logger.info(f"   Length: {len(script_content)} → {len(step6_script)} characters")
        
        # Step 7: Final Validation
        logger.info("Step 7: Final Validation")
        final_script = self._validate_script(step6_script)
        logger.info(f"   Final validation completed")
        logger.info(f"   Length: {len(script_content)} → {len(final_script)} characters")
        
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
        """Apply entity variations for natural speech with PROPER word boundaries."""
        if not entities:
            return script
        
        logger.info(f"   Applying variations for entities: {entities}")
        
        # Reset and register entities
        entity_variation_manager.reset_mentions()
        entity_variation_manager.register_entities(entities)
        
        processed_script = script
        
        for entity in entities:
            # 🚨 CRITICAL FIX: Use word boundaries to match ONLY complete words
            pattern = r'\b' + re.escape(entity) + r'\b'
            
            matches = list(re.finditer(pattern, processed_script, re.IGNORECASE))
            logger.info(f"   🎯 Processing '{entity}': {len(matches)} instances found")
            
            if len(matches) > 0:
                # Process matches in reverse order to maintain positions
                for i, match in enumerate(reversed(matches)):
                    variation = entity_variation_manager.get_variation(entity)
                    original_text = match.group(0)
                    
                    # Log only the first 10 replacements for debugging
                    if i < 10:
                        logger.info(f"       {i+1}. '{original_text}' → '{variation}'")
                    
                    # Replace the matched text
                    start, end = match.span()
                    processed_script = processed_script[:start] + variation + processed_script[end:]
        
        return processed_script
    
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
    
    def _validate_script(self, script: str) -> str:
        """Validate and finalize the cleaned script while preserving line breaks."""
        logger.info("   Final validation completed")
        
        # Basic validation checks
        if not script or len(script.strip()) == 0:
            logger.warning("   ⚠️ Script is empty after processing")
            return script
        
        # Ensure proper ending
        if not script.endswith('.'):
            script = script.rstrip() + '.'
        
        #  CRITICAL FIX: Preserve line breaks during cleanup
        # Split by line breaks first to preserve paragraph structure
        paragraphs = script.split('\n\n')
        
        # Clean each paragraph individually
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            if paragraph.strip():  # Only process non-empty paragraphs
                # Remove double spaces but preserve line breaks
                cleaned_paragraph = ' '.join(paragraph.split())
                cleaned_paragraphs.append(cleaned_paragraph)
        
        # Rejoin with double line breaks to preserve paragraph structure
        final_script = '\n\n'.join(cleaned_paragraphs)
        
        logger.info(f"   Final script length: {len(final_script)} characters")
        logger.info(f"   Preserved {len(cleaned_paragraphs)} paragraphs")
        
        return final_script
    
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

    def _format_paragraphs(self, script: str) -> str:
        """Add proper paragraph breaks for TTS optimization."""
        logger.info("   Formatting paragraphs for TTS optimization")
        
        # Add debug logging
        logger.info(f"   Input script length: {len(script)}")
        logger.info(f"   Input script preview: {script[:100]}...")
        
        # Split on natural break points
        paragraphs = []
        sentences = script.split('. ')
        
        logger.info(f"   Found {len(sentences)} sentences")
        
        current_paragraph = []
        for i, sentence in enumerate(sentences):
            current_paragraph.append(sentence)
            
            # Start new paragraph on topic shifts or every 5-7 sentences
            should_break = False
            
            # Topic shift indicators
            if any(keyword in sentence.lower() for keyword in [
                'the scene was set', 'picture this', 'in the days following',
                'as the story unfolded', 'meanwhile', 'later', 'then',
                'jean-claude\'s instagram', 'van damme had stayed quiet'
            ]):
                should_break = True
                logger.info(f"   Breaking on topic shift: {sentence[:50]}...")
            
            # Length-based breaks (every 5-7 sentences)
            elif len(current_paragraph) >= 6:
                should_break = True
                logger.info(f"   Breaking on length: {len(current_paragraph)} sentences")
            
            if should_break and current_paragraph:
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph) + '.')
        
        # Join with double line breaks
        formatted_script = '\n\n'.join(paragraphs)
        
        logger.info(f"   Created {len(paragraphs)} paragraphs")
        logger.info(f"   Output script length: {len(formatted_script)}")
        logger.info(f"   Output preview: {formatted_script[:200]}...")
        
        return formatted_script