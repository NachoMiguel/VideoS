from typing import Optional, Dict, Any
import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import logging
import re
from datetime import datetime
from core.config import settings
from core.exceptions import VideoNotFoundError, TranscriptNotFoundError
import requests

logger = logging.getLogger(__name__)

class YouTubeService:
    def __init__(self):
        self.api_key = settings.youtube_api_key
        self._youtube_client = None  # Lazy initialization
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }

    @property
    def youtube(self):
        """Lazy initialization of YouTube API client - only when needed and if API key is available."""
        if self._youtube_client is None:
            if not self.api_key:
                raise ValueError("YouTube API key not configured - cannot use YouTube Data API features")
            
            try:
                self._youtube_client = build('youtube', 'v3', developerKey=self.api_key)
                logger.info("✅ YouTube Data API client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize YouTube Data API client: {str(e)}")
                raise
        
        return self._youtube_client

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
            r'(?:embed\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError("Invalid YouTube URL format")

    async def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get video metadata using YouTube Data API - requires API key."""
        if not self.api_key:
            logger.warning("⚠️ YouTube API key not configured - returning minimal video info")
            return {
                'title': f"Video {video_id}",
                'description': "API key required for full video information",
                'duration': "Unknown",
                'view_count': "Unknown",
                'published_at': "Unknown"
            }
        
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            )
            response = request.execute()

            if not response['items']:
                raise VideoNotFoundError(f"Video {video_id} not found")

            video = response['items'][0]
            return {
                'title': video['snippet']['title'],
                'description': video['snippet']['description'],
                'duration': video['contentDetails']['duration'],
                'view_count': video['statistics']['viewCount'],
                'published_at': video['snippet']['publishedAt']
            }
        except HttpError as e:
            logger.error(f"YouTube API error: {str(e)}")
            raise

    async def get_transcript(self, video_id: str, language: str = 'en') -> list[dict]:
        """Get video transcript - works WITHOUT YouTube API key."""
        
        logger.info(f"🎯 Starting transcript extraction for video {video_id}")
        methods_tried = []
        
        # Method 1: Try direct extraction first (fastest)
        try:
            logger.info(f"🌐 Method 1: Direct extraction")
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            logger.info(f"✅ Direct method successful: {len(transcript)} transcript entries")
            return transcript
            
        except Exception as e:
            full_error = str(e)
            methods_tried.append(f"Direct: {full_error}")
            logger.warning(f"⚠️ Direct method failed: {full_error}")

        # Method 2: Enhanced headers with user agent spoofing
        try:
            logger.info(f"🎭 Method 2: Enhanced headers")
            
            import requests
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'max-age=0'
            })
            
            # Temporarily patch requests to use our session
            original_session = requests.Session
            requests.Session = lambda: session
            
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
                logger.info(f"✅ Enhanced headers successful: {len(transcript)} transcript entries")
                return transcript
            finally:
                requests.Session = original_session
            
        except Exception as e:
            full_error = str(e)
            methods_tried.append(f"Enhanced headers: {full_error}")
            logger.warning(f"⚠️ Enhanced headers failed: {full_error}")

        # Method 3: Try with multiple language fallbacks
        try:
            logger.info(f"🌍 Method 3: Multiple language fallback")
            
            # Try common language codes if requested language fails
            language_fallbacks = [language, 'en', 'en-US', 'en-GB'] 
            unique_langs = list(dict.fromkeys(language_fallbacks))  # Remove duplicates, preserve order
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=unique_langs)
            logger.info(f"✅ Language fallback successful: {len(transcript)} transcript entries")
            return transcript
            
        except Exception as e:
            full_error = str(e)
            methods_tried.append(f"Language fallback: {full_error}")
            logger.warning(f"⚠️ Language fallback failed: {full_error}")

        # Method 4: Try Tor proxy if available (optional)
        try:
            logger.info(f"🔥 Method 4: Tor proxy attempt")
            proxies = {
                "https": "socks5://127.0.0.1:9150",
                "http": "socks5://127.0.0.1:9150",
            }
            
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=[language],
                proxies=proxies
            )
            
            logger.info(f"✅ Tor proxy successful: {len(transcript)} transcript entries")
            return transcript
            
        except Exception as e:
            full_error = str(e)
            methods_tried.append(f"Tor proxy: {full_error}")
            logger.warning(f"⚠️ Tor proxy failed: {full_error}")

        # All methods failed
        logger.error("🚨 ALL TRANSCRIPT EXTRACTION METHODS FAILED:")
        for i, method_error in enumerate(methods_tried, 1):
            logger.error(f"   Method {i}: {method_error}")
        
        final_error = f"All transcript extraction methods failed for video {video_id}"
        raise TranscriptNotFoundError(final_error)

    def validate_video(self, video_id: str) -> bool:
        """Validate video length and availability - requires API key."""
        if not self.api_key:
            logger.warning("⚠️ YouTube API key not configured - cannot validate video, assuming valid")
            return True
        
        try:
            info = self.youtube.videos().list(
                part="contentDetails,status",
                id=video_id
            ).execute()

            if not info['items']:
                return False

            video = info['items'][0]
            
            # Check if video is available
            if video['status']['privacyStatus'] != 'public':
                return False

            # Parse duration and check if it's within limits
            duration = video['contentDetails']['duration']
            duration_seconds = self._parse_duration(duration)
            
            return duration_seconds <= settings.max_video_duration_seconds

        except Exception as e:
            logger.error(f"Video validation error: {str(e)}")
            return False

    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds."""
        match = re.match(
            r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?',
            duration
        )
        if not match:
            return 0
            
        days, hours, minutes, seconds = match.groups()
        
        total_seconds = 0
        if days:
            total_seconds += int(days) * 86400
        if hours:
            total_seconds += int(hours) * 3600
        if minutes:
            total_seconds += int(minutes) * 60
        if seconds:
            total_seconds += int(seconds)
            
        return total_seconds

    async def get_video_segments(self, video_id: str, segment_length: int = 30) -> list[dict]:
        """Get video segments based on transcript timing."""
        transcript = await self.get_transcript(video_id)
        segments = []
        current_segment = {
            'start': 0,
            'text': [],
            'duration': 0
        }

        for entry in transcript:
            if current_segment['duration'] + entry['duration'] > segment_length:
                # Finalize current segment
                segments.append({
                    'start': current_segment['start'],
                    'text': ' '.join(current_segment['text']),
                    'duration': current_segment['duration']
                })
                # Start new segment
                current_segment = {
                    'start': entry['start'],
                    'text': [entry['text']],
                    'duration': entry['duration']
                }
            else:
                current_segment['text'].append(entry['text'])
                current_segment['duration'] += entry['duration']

        # Add final segment
        if current_segment['text']:
            segments.append({
                'start': current_segment['start'],
                'text': ' '.join(current_segment['text']),
                'duration': current_segment['duration']
            })

        return segments