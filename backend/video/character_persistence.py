import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import json
from datetime import datetime, timedelta

from core.logger import logger

class CharacterPersistence:
    """Handles persistent storage and retrieval of trained character data."""
    
    def __init__(self, cache_dir: str = "cache/characters"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.characters_file = self.cache_dir / "characters.json"
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_ttl_days = 30  # Character cache expires after 30 days
        self.max_cache_size_mb = 500  # Maximum cache size in MB
        
    def save_character_training(self, character_name: str, embeddings: List[np.ndarray], 
                              metadata: Dict[str, Any] = None) -> bool:
        """Save trained character embeddings and metadata."""
        try:
            # Create character hash for consistent file naming
            char_hash = self._generate_character_hash(character_name)
            
            # Save embeddings
            embeddings_file = self.embeddings_dir / f"{char_hash}.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Update character registry
            character_data = {
                'name': character_name,
                'hash': char_hash,
                'embedding_count': len(embeddings),
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self._update_character_registry(character_name, character_data)
            
            logger.info(f"Saved {len(embeddings)} embeddings for character: {character_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save character training for {character_name}: {str(e)}")
            return False
    
    def load_character_training(self, character_name: str) -> Optional[List[np.ndarray]]:
        """Load trained character embeddings."""
        try:
            registry = self._load_character_registry()
            
            if character_name not in registry:
                logger.debug(f"Character {character_name} not found in registry")
                return None
            
            char_data = registry[character_name]
            
            # Check if cache is expired
            if self._is_cache_expired(char_data):
                logger.debug(f"Cache expired for character: {character_name}")
                self._remove_character_cache(character_name)
                return None
            
            # Load embeddings
            char_hash = char_data['hash']
            embeddings_file = self.embeddings_dir / f"{char_hash}.pkl"
            
            if not embeddings_file.exists():
                logger.warning(f"Embeddings file missing for character: {character_name}")
                return None
            
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Update last accessed time
            char_data['last_accessed'] = datetime.now().isoformat()
            self._update_character_registry(character_name, char_data)
            
            logger.info(f"Loaded {len(embeddings)} embeddings for character: {character_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load character training for {character_name}: {str(e)}")
            return None
    
    def get_cached_characters(self) -> List[str]:
        """Get list of characters with cached training data."""
        try:
            registry = self._load_character_registry()
            
            # Filter out expired entries
            valid_characters = []
            for char_name, char_data in registry.items():
                if not self._is_cache_expired(char_data):
                    valid_characters.append(char_name)
            
            return valid_characters
            
        except Exception as e:
            logger.error(f"Failed to get cached characters: {str(e)}")
            return []
    
    def is_character_cached(self, character_name: str) -> bool:
        """Check if character training data is cached and valid."""
        try:
            registry = self._load_character_registry()
            
            if character_name not in registry:
                return False
            
            char_data = registry[character_name]
            
            # Check if cache is expired
            if self._is_cache_expired(char_data):
                return False
            
            # Check if embeddings file exists
            char_hash = char_data['hash']
            embeddings_file = self.embeddings_dir / f"{char_hash}.pkl"
            
            return embeddings_file.exists()
            
        except Exception as e:
            logger.error(f"Failed to check character cache for {character_name}: {str(e)}")
            return False
    
    def clear_character_cache(self, character_name: str) -> bool:
        """Clear cached data for a specific character."""
        try:
            return self._remove_character_cache(character_name)
        except Exception as e:
            logger.error(f"Failed to clear cache for {character_name}: {str(e)}")
            return False
    
    def clear_all_cache(self) -> bool:
        """Clear all cached character data."""
        try:
            # Remove all embedding files
            for file in self.embeddings_dir.glob("*.pkl"):
                file.unlink()
            
            # Clear registry
            if self.characters_file.exists():
                self.characters_file.unlink()
            
            logger.info("Cleared all character cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear all cache: {str(e)}")
            return False
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries and return count of cleaned entries."""
        try:
            registry = self._load_character_registry()
            cleaned_count = 0
            
            for char_name, char_data in list(registry.items()):
                if self._is_cache_expired(char_data):
                    if self._remove_character_cache(char_name):
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            registry = self._load_character_registry()
            
            # Calculate cache size
            total_size = 0
            for file in self.embeddings_dir.glob("*.pkl"):
                total_size += file.stat().st_size
            
            total_size_mb = total_size / (1024 * 1024)
            
            # Count valid and expired entries
            valid_count = 0
            expired_count = 0
            
            for char_data in registry.values():
                if self._is_cache_expired(char_data):
                    expired_count += 1
                else:
                    valid_count += 1
            
            return {
                'total_characters': len(registry),
                'valid_characters': valid_count,
                'expired_characters': expired_count,
                'cache_size_mb': round(total_size_mb, 2),
                'max_cache_size_mb': self.max_cache_size_mb,
                'cache_utilization': round((total_size_mb / self.max_cache_size_mb) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    def _generate_character_hash(self, character_name: str) -> str:
        """Generate consistent hash for character name."""
        return hashlib.md5(character_name.lower().encode()).hexdigest()
    
    def _load_character_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load character registry from file."""
        try:
            if not self.characters_file.exists():
                return {}
            
            with open(self.characters_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load character registry: {str(e)}")
            return {}
    
    def _save_character_registry(self, registry: Dict[str, Dict[str, Any]]) -> bool:
        """Save character registry to file."""
        try:
            with open(self.characters_file, 'w') as f:
                json.dump(registry, f, indent=2)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save character registry: {str(e)}")
            return False
    
    def _update_character_registry(self, character_name: str, character_data: Dict[str, Any]) -> bool:
        """Update character registry with new data."""
        try:
            registry = self._load_character_registry()
            registry[character_name] = character_data
            return self._save_character_registry(registry)
            
        except Exception as e:
            logger.error(f"Failed to update character registry: {str(e)}")
            return False
    
    def _is_cache_expired(self, char_data: Dict[str, Any]) -> bool:
        """Check if character cache is expired."""
        try:
            created_at = datetime.fromisoformat(char_data['created_at'])
            expiry_date = created_at + timedelta(days=self.cache_ttl_days)
            return datetime.now() > expiry_date
            
        except Exception as e:
            logger.error(f"Failed to check cache expiry: {str(e)}")
            return True  # Assume expired if we can't check
    
    def _remove_character_cache(self, character_name: str) -> bool:
        """Remove character from cache."""
        try:
            registry = self._load_character_registry()
            
            if character_name not in registry:
                return True
            
            char_data = registry[character_name]
            char_hash = char_data['hash']
            
            # Remove embeddings file
            embeddings_file = self.embeddings_dir / f"{char_hash}.pkl"
            if embeddings_file.exists():
                embeddings_file.unlink()
            
            # Remove from registry
            del registry[character_name]
            self._save_character_registry(registry)
            
            logger.info(f"Removed character cache for: {character_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove character cache for {character_name}: {str(e)}")
            return False

# Global persistence instance
character_persistence = CharacterPersistence() 