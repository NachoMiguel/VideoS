import aiohttp
import asyncio
import logging
from typing import List, Dict, Optional
import os
from config import settings

logger = logging.getLogger(__name__)

class ImageSearchService:
    def __init__(self):
        self.api_key = settings.google_custom_search_api_key
        self.engine_id = settings.google_custom_search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search_images(self, query: str, max_results: int = 10) -> List[str]:
        """Search for images using Google Custom Search API."""
        try:
            if not self.api_key or not self.engine_id:
                logger.warning("Google API credentials not configured, using mock results")
                return self._get_mock_images(query, max_results)
            
            params = {
                'key': self.api_key,
                'cx': self.engine_id,
                'q': query,
                'searchType': 'image',
                'num': min(max_results, 10)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        images = []
                        for item in data.get('items', []):
                            if 'link' in item:
                                images.append(item['link'])
                        return images[:max_results]
                    else:
                        logger.error(f"Google API error: {response.status}")
                        return self._get_mock_images(query, max_results)
                        
        except Exception as e:
            logger.error(f"Image search failed: {str(e)}")
            return self._get_mock_images(query, max_results)
    
    def _get_mock_images(self, query: str, max_results: int) -> List[str]:
        """Return mock image URLs for testing."""
        mock_images = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg"
        ]
        return mock_images[:max_results]
    
    async def search_character_images(self, characters: List[str]) -> Dict[str, List[str]]:
        """Search for images of multiple characters."""
        results = {}
        for character in characters:
            images = await self.search_images(f"{character} face", max_results=5)
            results[character] = images
        return results 