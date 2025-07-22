#!/usr/bin/env python3
"""
Entity Variation Manager for natural speech patterns
"""

import re
import logging
from typing import Dict, List, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class EntityVariationManager:
    def __init__(self):
        self.entity_variations: Dict[str, List[str]] = {}
        self.mention_count: Dict[str, int] = defaultdict(int)
        self.first_name_conflicts: Set[str] = set()

    def register_entities(self, entities: List[str]) -> None:
        """Register entities and generate their variations."""
        logger.info(f"🎯 Registering {len(entities)} entities for variation management")
        
        # Step 1: Analyze entities for conflicts
        self._analyze_entity_conflicts(entities)
        
        # Step 2: Generate variations for each entity
        for entity in entities:
            variations = self._generate_entity_variations(entity)
            self.entity_variations[entity] = variations
            logger.info(f"📝 Entity '{entity}' → variations: {variations}")
        
        logger.info(f"✅ Registered {len(self.entity_variations)} entities with variations")

    def _analyze_entity_conflicts(self, entities: List[str]) -> None:
        """Analyze entities for first name conflicts."""
        first_names = defaultdict(list)
        
        for entity in entities:
            parts = entity.split()
            if len(parts) >= 2:
                # Handle hyphenated first names
                if '-' in parts[0]:
                    first_name = parts[0].lower()  # "jean-claude"
                else:
                    first_name = parts[0].lower()  # "steven"
                
                first_names[first_name].append(entity)
        
        # Identify conflicts (multiple entities with same first name)
        for first_name, entity_list in first_names.items():
            if len(entity_list) > 1:
                self.first_name_conflicts.add(first_name)
                logger.warning(f"⚠️ First name conflict detected: '{first_name}' in {entity_list}")

    def _generate_entity_variations(self, entity: str) -> List[str]:
        """Generate proper name variations with conflict resolution."""
        if len(entity.split()) == 1:
            return [entity]  # Single word - no variations
        
        parts = entity.split()
        variations = [entity]  # Full name first
        
        if len(parts) >= 2:
            # Handle hyphenated first names
            if '-' in parts[0]:
                first_name = parts[0]  # "Jean-Claude"
                last_name = " ".join(parts[1:])  # "Van Damme"
            else:
                first_name = parts[0]  # "Steven"
                last_name = " ".join(parts[1:])  # "Seagal"
            
            # Check for first name conflicts
            first_name_lower = first_name.lower()
            if first_name_lower in self.first_name_conflicts:
                # Conflict detected - only use last name
                logger.info(f"🎯 Conflict resolution: Using only last name '{last_name}' for '{entity}'")
                variations.append(last_name)
            else:
                # No conflict - can use both first name and last name
                variations.extend([first_name, last_name])
        
        return variations

    def apply_variations_to_text(self, text: str) -> str:
        for entity in self.entity_variations.keys():
            pattern = re.escape(entity)
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in reversed(matches):
                start, end = match.span()
                original = text[start:end]
                variation = self.select_variation(entity)
                
                if original.isupper():
                    replacement = variation.upper()
                elif original.istitle():
                    replacement = variation.title()
                else:
                    replacement = variation
                
                text = text[:start] + replacement + text[end:]
        
        return text

    def select_variation(self, entity: str) -> str:
        if entity not in self.entity_variations:
            return entity
        
        variations = self.entity_variations[entity]
        mention_count = self.mention_count[entity]
        
        if mention_count == 0:
            selected = variations[0]  # Full name for first mention
        else:
            if len(variations) > 1:
                variation_index = (mention_count % (len(variations) - 1)) + 1
                selected = variations[variation_index]
            else:
                selected = variations[0]
        
        self.mention_count[entity] += 1
        return selected

    def reset_mentions(self) -> None:
        self.mention_count.clear()

    def get_statistics(self) -> Dict:
        return {
            "total_entities": len(self.entity_variations),
            "total_mentions": sum(self.mention_count.values()),
            "entity_mentions": dict(self.mention_count)
        }

    def get_variation(self, entity: str) -> str:
        """Get the next variation for an entity based on mention count."""
        if entity not in self.entity_variations:
            return entity
        
        variations = self.entity_variations[entity]
        mention_count = self.mention_count[entity]
        
        # First mention gets full name, subsequent mentions get variations
        if mention_count == 0:
            # First mention - use full name
            variation = variations[0]  # Full name
        else:
            # Subsequent mentions - cycle through variations (skip full name)
            variation_index = (mention_count % (len(variations) - 1)) + 1
            variation = variations[variation_index]
        
        # Increment mention count
        self.mention_count[entity] += 1
        
        logger.debug(f"   Entity '{entity}' mention #{mention_count + 1} → '{variation}'")
        return variation

# Global instance
entity_variation_manager = EntityVariationManager() 