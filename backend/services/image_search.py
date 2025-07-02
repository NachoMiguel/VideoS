import aiohttp
import asyncio
import os
import hashlib
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import aiofiles
from pathlib import Path

from core.config import settings
from core.logger import logger
from core.exceptions import ImageSearchError

class ImageSearchService:
    def __init__(self):
        self.session = None
        self.image_cache_dir = Path(settings.cache_dir) / "character_images"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def search_character_images(self, characters: List[str]) -> Dict[str, List[str]]:
        """Search for character images and download them."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            character_images = {}
            
            for character in characters:
                try:
                    logger.info(f"Searching images for character: {character}")
                    image_urls = await self._search_images_for_character(character)
                    
                    if image_urls:
                        # Download and cache images
                        cached_paths = await self._download_and_cache_images(character, image_urls)
                        character_images[character] = cached_paths
                        logger.info(f"Found {len(cached_paths)} images for {character}")
                    else:
                        logger.warning(f"No images found for character: {character}")
                        character_images[character] = []
                        
                except Exception as e:
                    logger.error(f"Error searching images for {character}: {str(e)}")
                    character_images[character] = []
                    
                # Rate limiting
                await asyncio.sleep(1)
            
            return character_images
            
        except Exception as e:
            logger.error(f"Character image search error: {str(e)}")
            raise ImageSearchError(f"Image search failed: {str(e)}")
        finally:
            if self.session:
                await self.session.close()
                self.session = None
    
    async def _search_images_for_character(self, character_name: str) -> List[str]:
        """Search for images of a specific character."""
        try:
            # Use a combination of search methods
            image_urls = []
            
            # Method 1: Use known character database (if available)
            known_images = await self._get_known_character_images(character_name)
            if known_images:
                image_urls.extend(known_images)
            
            # Method 2: Generate placeholder/stock images if no real images found
            if not image_urls:
                placeholder_images = await self._generate_placeholder_images(character_name)
                image_urls.extend(placeholder_images)
            
            return image_urls[:5]  # Limit to 5 images per character
            
        except Exception as e:
            logger.error(f"Error searching images for {character_name}: {str(e)}")
            return []
    
    async def _get_known_character_images(self, character_name: str) -> List[str]:
        """Get images for known characters from predefined sources."""
        try:
            # Known character mappings (for demo purposes)
            known_characters = {
                "Jean Claude Vandamme": [
                    "https://via.placeholder.com/300x400/0066cc/ffffff?text=JCVD",
                    "https://via.placeholder.com/300x400/cc6600/ffffff?text=Van+Damme"
                ],
                "Steven Seagal": [
                    "https://via.placeholder.com/300x400/006600/ffffff?text=Seagal",
                    "https://via.placeholder.com/300x400/660066/ffffff?text=Steven+S"
                ]
            }
            
            # Check for exact match or partial match
            for known_char, urls in known_characters.items():
                if character_name.lower() in known_char.lower() or known_char.lower() in character_name.lower():
                    return urls
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting known character images: {str(e)}")
            return []
    
    async def _generate_placeholder_images(self, character_name: str) -> List[str]:
        """Generate placeholder images for unknown characters."""
        try:
            # Create placeholder URLs with character names
            placeholders = []
            name_parts = character_name.split()
            
            # Generate different placeholder styles
            colors = ["0066cc", "cc6600", "006600", "660066", "cc0066"]
            
            for i, color in enumerate(colors[:3]):  # Max 3 placeholders
                if len(name_parts) >= 2:
                    initials = f"{name_parts[0][0]}{name_parts[1][0]}"
                else:
                    initials = character_name[:2].upper()
                
                url = f"https://via.placeholder.com/300x400/{color}/ffffff?text={initials}"
                placeholders.append(url)
            
            return placeholders
            
        except Exception as e:
            logger.error(f"Error generating placeholder images: {str(e)}")
            return []
    
    async def _download_and_cache_images(self, character_name: str, image_urls: List[str]) -> List[str]:
        """Download images and cache them locally."""
        try:
            cached_paths = []
            character_dir = self.image_cache_dir / self._sanitize_filename(character_name)
            character_dir.mkdir(exist_ok=True)
            
            for i, url in enumerate(image_urls):
                try:
                    # Generate filename
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filename = f"{character_name}_{i}_{url_hash}.jpg"
                    file_path = character_dir / filename
                    
                    # Check if already cached
                    if file_path.exists():
                        cached_paths.append(str(file_path))
                        continue
                    
                    # Download image
                    if self.session:
                        async with self.session.get(url, timeout=10) as response:
                            if response.status == 200:
                                content = await response.read()
                                
                                # Save to cache
                                async with aiofiles.open(file_path, 'wb') as f:
                                    await f.write(content)
                                
                                cached_paths.append(str(file_path))
                                logger.debug(f"Cached image: {file_path}")
                            else:
                                logger.warning(f"Failed to download image {url}: {response.status}")
                    
                except Exception as e:
                    logger.error(f"Error downloading image {url}: {str(e)}")
                    continue
            
            return cached_paths
            
        except Exception as e:
            logger.error(f"Error caching images for {character_name}: {str(e)}")
            return []
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove extra spaces and limit length
        sanitized = '_'.join(sanitized.split())[:50]
        return sanitized
    
    async def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cached images."""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            for file_path in self.image_cache_dir.rglob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        logger.debug(f"Removed old cached image: {file_path}")
            
            logger.info(f"Cache cleanup completed")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")

# Global service instance
image_search_service = ImageSearchService() 