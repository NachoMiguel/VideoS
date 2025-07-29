import aiohttp
import asyncio
import os
import hashlib
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import aiofiles
from pathlib import Path
import logging
import json
import sys

# Add yt root to path for ai_shared_lib imports
yt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, yt_root)

from ai_shared_lib.config import settings

logger = logging.getLogger(__name__)

class ImageSearchService:
    def __init__(self):
        self.session = None
        self.image_cache_dir = Path(settings.CACHE_DIR) / "character_images"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.google_api_key = settings.google_custom_search_api_key
        self.google_engine_id = settings.google_custom_search_engine_id
        
    async def search_character_images(self, characters: List[str]) -> Dict[str, List[str]]:
        """Search for character images and download them."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            character_images = {}
            
            for character in characters:
                try:
                    logger.info(f"Searching images for character: {character}")
                    
                    # Try Google Custom Search first
                    image_urls = await self._search_google_images(character)
                    
                    # If not enough images, try local database
                    if len(image_urls) < 3:
                        local_images = await self._search_local_images(character)
                        image_urls.extend(local_images)
                    
                    # Validate and filter images
                    if image_urls:
                        validated_urls = await self._validate_image_urls(image_urls)
                        if validated_urls:
                            # Download and cache images
                            cached_paths = await self._download_and_cache_images(character, validated_urls)
                            character_images[character] = cached_paths
                            logger.info(f"Found {len(cached_paths)} images for {character}")
                        else:
                            logger.warning(f"No valid images found for character: {character}")
                            character_images[character] = []
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
            return {}
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def _search_google_images(self, character_name: str) -> List[str]:
        """Search for images using Google Custom Search API."""
        try:
            if not self.google_api_key or not self.google_engine_id:
                logger.warning("Google Custom Search API not configured")
                return []
            
            # Prepare search query
            query = f"{character_name} actor headshot portrait"
            url = "https://www.googleapis.com/customsearch/v1"
            
            params = {
                'key': self.google_api_key,
                'cx': self.google_engine_id,
                'q': query,
                'searchType': 'image',
                'num': 10,
                'imgType': 'face',
                'imgSize': 'medium',
                'safe': 'active'
            }
            
            if self.session:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        image_urls = []
                        for item in data.get('items', []):
                            img_url = item.get('link')
                            if img_url and self._is_valid_image_url(img_url):
                                image_urls.append(img_url)
                        
                        logger.info(f"Google Search found {len(image_urls)} images for {character_name}")
                        return image_urls[:5]  # Limit to 5 images
                    else:
                        logger.error(f"Google API error: {response.status} - {await response.text()}")
                        return []
            
            return []
            
        except Exception as e:
            logger.error(f"Google image search error for {character_name}: {str(e)}")
            return []

    async def _search_local_images(self, character_name: str) -> List[str]:
        """Search local cached images for known characters."""
        try:
            logger.info(f"Searching cached images for character: {character_name}")
            
            # First, try to find cached images in the character_images directory
            cached_images = await self._get_cached_character_images(character_name)
            if cached_images:
                logger.info(f"Found {len(cached_images)} cached images for {character_name}")
                return cached_images
            
            # Fallback to hardcoded local database for known characters
            local_database = {
                "jean claude van damme": [
                    "https://m.media-amazon.com/images/M/MV5BMjI4NDI1MjY5OF5BMl5BanBnXkFtZTcwNTgzNzY3MQ@@._V1_.jpg",
                    "https://m.media-amazon.com/images/M/MV5BMTQ2NzgxMTAxNV5BMl5BanBnXkFtZTcwMzg1NjIyMQ@@._V1_.jpg"
                ],
                "steven seagal": [
                    "https://m.media-amazon.com/images/M/MV5BMTk4MDI2NDEzNl5BMl5BanBnXkFtZTcwMTI3NjcyMQ@@._V1_.jpg",
                    "https://m.media-amazon.com/images/M/MV5BMTgzNjA1MjY1NV5BMl5BanBnXkFtZTcwMjU3NDEyMQ@@._V1_.jpg"
                ]
            }
            
            # Normalize character name for lookup
            normalized_name = character_name.lower().strip()
            
            # Try exact match first
            if normalized_name in local_database:
                return local_database[normalized_name]
            
            # Try partial match
            for known_char, urls in local_database.items():
                if known_char in normalized_name or normalized_name in known_char:
                    logger.info(f"Found partial match for {character_name}: {known_char}")
                    return urls
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching local images for {character_name}: {str(e)}")
            return []

    async def _get_cached_character_images(self, character_name: str) -> List[str]:
        """Get cached character images from the character_images directory."""
        try:
            if not self.image_cache_dir.exists():
                logger.warning(f"Character images cache directory does not exist: {self.image_cache_dir}")
                return []
            
            # Try different variations of the character name
            possible_names = [
                character_name,
                character_name.lower(),
                character_name.title(),
                character_name.replace(" ", ""),
                character_name.replace(" ", "_"),
                character_name.replace(" ", ".")
            ]
            
            cached_images = []
            
            for name_variant in possible_names:
                character_dir = self.image_cache_dir / self._sanitize_filename(name_variant)
                if character_dir.exists() and character_dir.is_dir():
                    logger.info(f"Found character directory: {character_dir}")
                    
                    # Get all image files in the directory
                    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
                    for file_path in character_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                            cached_images.append(str(file_path))
                    
                    if cached_images:
                        logger.info(f"Found {len(cached_images)} cached images for {character_name} in {character_dir}")
                        break
            
            return cached_images
            
        except Exception as e:
            logger.error(f"Error getting cached images for {character_name}: {str(e)}")
            return []

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL."""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
            path_lower = parsed.path.lower()
            
            # Check if URL ends with valid extension or has image indicators
            has_extension = any(path_lower.endswith(ext) for ext in valid_extensions)
            has_image_indicator = 'image' in url.lower() or 'photo' in url.lower()
            
            return has_extension or has_image_indicator
            
        except Exception:
            return False

    async def _validate_image_urls(self, urls: List[str]) -> List[str]:
        """Validate that URLs return actual images."""
        validated = []
        
        for url in urls[:10]:  # Limit validation to prevent excessive requests
            try:
                if self.session:
                    async with self.session.head(url, timeout=5) as response:
                        if response.status == 200:
                            content_type = response.headers.get('content-type', '').lower()
                            if content_type.startswith('image/'):
                                validated.append(url)
                            else:
                                logger.debug(f"URL {url} is not an image (content-type: {content_type})")
                        else:
                            logger.debug(f"URL {url} returned status {response.status}")
            except Exception as e:
                logger.debug(f"Failed to validate URL {url}: {str(e)}")
                continue
        
        return validated

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
        """Sanitize filename for safe file system usage."""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        return sanitized[:50] 