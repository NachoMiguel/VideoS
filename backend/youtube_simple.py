#!/usr/bin/env python3
"""Simplified YouTube service using basic yt-dlp approach"""

import yt_dlp
import re
from typing import List, Dict, Any
from core.logger import logger
from core.exceptions import TranscriptNotFoundError

class SimpleYouTubeService:
    """Simplified YouTube service that just extracts transcripts reliably."""
    
    def __init__(self):
        self.ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US'],
            'subtitlesformat': 'json3',
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
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
    
    async def get_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """Get transcript using basic yt-dlp approach."""
        video_url = f"https://youtube.com/watch?v={video_id}"
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                # Try to get subtitles
                subtitles = info.get('subtitles', {})
                auto_captions = info.get('automatic_captions', {})
                
                # Prefer manual subtitles over auto-generated
                caption_data = None
                if 'en' in subtitles:
                    caption_data = subtitles['en']
                elif 'en-US' in subtitles:
                    caption_data = subtitles['en-US']
                elif 'en' in auto_captions:
                    caption_data = auto_captions['en']
                elif 'en-US' in auto_captions:
                    caption_data = auto_captions['en-US']
                
                if not caption_data:
                    raise TranscriptNotFoundError("No captions available")
                
                # Find JSON3 format
                json3_url = None
                for fmt in caption_data:
                    if fmt.get('ext') == 'json3':
                        json3_url = fmt['url']
                        break
                
                if not json3_url:
                    raise TranscriptNotFoundError("No JSON3 subtitles available")
                
                # Download and parse subtitles
                import requests
                import json
                
                response = requests.get(json3_url)
                subtitle_data = response.json()
                
                # Parse to transcript format
                transcript = []
                events = subtitle_data.get('events', [])
                
                for event in events:
                    if 'segs' in event and event.get('tStartMs') is not None:
                        text_segments = []
                        for seg in event['segs']:
                            if 'utf8' in seg:
                                text_segments.append(seg['utf8'])
                        
                        if text_segments:
                            transcript.append({
                                'text': ''.join(text_segments).strip(),
                                'start': event['tStartMs'] / 1000.0,
                                'duration': event.get('dDurationMs', 0) / 1000.0
                            })
                
                logger.info(f"✅ Extracted {len(transcript)} transcript segments")
                return transcript
                
        except Exception as e:
            logger.error(f"Transcript extraction failed: {str(e)}")
            raise TranscriptNotFoundError(f"Failed to extract transcript: {str(e)}") 