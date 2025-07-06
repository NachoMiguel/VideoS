from typing import Optional, Dict, Any
import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import re
from datetime import datetime
from core.config import settings
from core.exceptions import VideoNotFoundError, TranscriptNotFoundError

logger = logging.getLogger(__name__)

class YouTubeService:
    def __init__(self):
        self.api_key = settings.youtube_api_key
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }

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
        """Get video metadata using YouTube Data API."""
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
        """Get video transcript using yt-dlp."""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # First try to get automatic captions
                info = ydl.extract_info(
                    f'https://www.youtube.com/watch?v={video_id}',
                    download=False
                )
                
                if 'automatic_captions' in info and language in info['automatic_captions']:
                    captions = info['automatic_captions'][language]
                elif 'subtitles' in info and language in info['subtitles']:
                    captions = info['subtitles'][language]
                else:
                    raise TranscriptNotFoundError(
                        f"No transcript found for video {video_id} in language {language}"
                    )

                # Process captions into a consistent format
                transcript = []
                for caption in captions:
                    if 'text' in caption:
                        transcript.append({
                            'text': caption['text'],
                            'start': caption.get('start', 0),
                            'duration': caption.get('duration', 0)
                        })

                return transcript

        except Exception as e:
            logger.error(f"Transcript extraction error for video {video_id}: {str(e)}")
            raise TranscriptNotFoundError(str(e))

    def validate_video(self, video_id: str) -> bool:
        """Validate video length and availability."""
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