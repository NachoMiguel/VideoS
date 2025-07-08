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
import asyncio
import tempfile
import json
import os

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

    async def diagnose_transcript_issue(self, video_id: str) -> dict:
        """Comprehensive diagnosis of transcript extraction issues."""
        diagnosis = {
            'video_id': video_id,
            'video_exists': False,
            'has_captions': False,
            'available_languages': [],
            'api_errors': [],
            'recommendations': []
        }
        
        try:
            # Check if video exists and has captions
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # Try to list available transcripts first
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            diagnosis['video_exists'] = True
            diagnosis['has_captions'] = True
            
            # Get available languages
            for transcript in transcript_list:
                diagnosis['available_languages'].append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
                
            logger.info(f"📊 Video {video_id} has {len(diagnosis['available_languages'])} caption tracks")
            
        except Exception as e:
            error_msg = str(e).lower()
            diagnosis['api_errors'].append(str(e))
            
            if 'no transcript found' in error_msg:
                diagnosis['video_exists'] = True
                diagnosis['has_captions'] = False
                diagnosis['recommendations'].append('Video exists but has no captions available')
            elif 'video unavailable' in error_msg:
                diagnosis['video_exists'] = False
                diagnosis['recommendations'].append('Video is private, deleted, or geo-blocked')
            elif 'no element found' in error_msg:
                diagnosis['recommendations'].extend([
                    'YouTube API structure changed - update youtube-transcript-api',
                    'Try using different user agent or proxy',
                    'YouTube may be blocking automated requests'
                ])
            else:
                diagnosis['recommendations'].append('Unknown error - check network connectivity and dependencies')
        
        return diagnosis

    async def get_transcript_yt_dlp_native(self, video_id: str, language: str = 'en') -> list[dict]:
        """Extract transcript using yt-dlp's native subtitle capabilities."""
        import yt_dlp
        import tempfile
        import json
        import os
        
        video_url = f"https://youtube.com/watch?v={video_id}"
        
        # Configure yt-dlp for subtitle extraction
        ydl_opts = {
            'writesubtitles': True,           # Extract manual subtitles
            'writeautomaticsub': True,       # Extract auto-generated captions
            'subtitleslangs': [language, 'en', 'en-US'],  # Language preference
            'subtitlesformat': 'json3',      # Get structured data with timestamps
            'skip_download': True,           # Don't download video
            'quiet': True,
            'no_warnings': True,
            'outtmpl': tempfile.gettempdir() + '/%(id)s.%(ext)s',
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info and subtitles
                info = ydl.extract_info(video_url, download=False)
                
                # Try to download subtitles
                ydl.download([video_url])
                
                # Find the downloaded subtitle file
                subtitle_file = None
                temp_dir = tempfile.gettempdir()
                
                # Look for subtitle files
                for file in os.listdir(temp_dir):
                    if file.startswith(video_id) and file.endswith('.json3'):
                        subtitle_file = os.path.join(temp_dir, file)
                        break
                
                if not subtitle_file or not os.path.exists(subtitle_file):
                    raise TranscriptNotFoundError(f"No subtitles found for {video_id}")
                
                # Parse JSON3 subtitle format
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    subtitle_data = json.load(f)
                
                # Convert to standard transcript format
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
                                'start': event['tStartMs'] / 1000.0,  # Convert ms to seconds
                                'duration': event.get('dDurationMs', 0) / 1000.0
                            })
                
                # Cleanup temp file
                os.unlink(subtitle_file)
                
                logger.info(f"✅ yt-dlp extracted {len(transcript)} transcript segments")
            return transcript
            
        except Exception as e:
            logger.error(f"yt-dlp transcript extraction failed: {str(e)}")
            raise TranscriptNotFoundError(f"Could not extract transcript: {str(e)}")

    async def get_transcript_updated_free(self, video_id: str, language: str = 'en') -> list[dict]:
        """Updated youtube-transcript-api with modern techniques."""
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.formatters import JSONFormatter
        import requests
        import time
        import random
        
        # Modern browser headers rotation
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # Create session with realistic headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        # Add random delay to appear more human
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Monkey patch the session
        original_session = requests.Session
        requests.Session = lambda: session
        
        try:
            # Try multiple language codes
            language_options = [language]
            if language != 'en':
                language_options.extend(['en', 'en-US', 'en-GB'])
            
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=language_options
            )
            
            logger.info(f"✅ Updated youtube-transcript-api: {len(transcript)} segments")
            return transcript
            
        except Exception as e:
            logger.error(f"Updated youtube-transcript-api failed: {str(e)}")
            raise TranscriptNotFoundError(f"Transcript extraction failed: {str(e)}")
        finally:
            requests.Session = original_session
            
    async def get_transcript_hybrid_free(self, video_id: str, language: str = 'en') -> list[dict]:
        """Hybrid approach: yt-dlp validation + youtube-transcript-api extraction."""
        import yt_dlp
        
        # Step 1: Use yt-dlp to validate video and check caption availability
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        video_url = f"https://youtube.com/watch?v={video_id}"
        video_info = None
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                video_info = ydl.extract_info(video_url, download=False)
                
                # Check if any captions are available
                has_captions = bool(
                    video_info.get('subtitles') or 
                    video_info.get('automatic_captions')
                )
                
                if not has_captions:
                    raise TranscriptNotFoundError("No captions available according to yt-dlp")
                    
                logger.info("✅ yt-dlp confirms captions are available")
                
            except Exception as e:
                logger.warning(f"yt-dlp validation failed: {e}")
                # Continue anyway - might still work with youtube-transcript-api
        
        # Step 2: Try yt-dlp native extraction first
        try:
            return await self.get_transcript_yt_dlp_native(video_id, language)
        except Exception as e:
            logger.warning(f"yt-dlp native extraction failed: {e}")
        
        # Step 3: Fallback to updated youtube-transcript-api
        try:
            return await self.get_transcript_updated_free(video_id, language)
        except Exception as e:
            logger.error(f"All free methods failed: {e}")
            raise TranscriptNotFoundError(f"No free transcript available for {video_id}")

    async def extract_video_context(self, video_id: str) -> dict:
        """Extract comprehensive video context for transcript correction."""
        
        video_url = f"https://youtube.com/watch?v={video_id}"
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(video_url, download=False)
                
                # Extract key context information
                context = {
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'tags': info.get('tags', []),
                    'channel': info.get('uploader', ''),
                    'categories': info.get('categories', []),
                    
                    # Extract potential names/entities from title and description
                    'potential_entities': self._extract_entities_from_text(
                        f"{info.get('title', '')} {info.get('description', '')}"
                    ),
                    
                    # Determine video topic/genre
                    'topic_hints': self._analyze_video_topic(info),
                    
                    # Check subtitle availability types
                    'has_manual_subs': bool(info.get('subtitles')),
                    'has_auto_subs': bool(info.get('automatic_captions')),
                    'available_sub_languages': list(info.get('subtitles', {}).keys()),
                }
                
                logger.info(f"📊 Video context: {len(context['potential_entities'])} entities found")
                logger.info(f"🎯 Detected topics: {context['topic_hints']['primary_topics']}")
                return context
            
            except Exception as e:
                logger.error(f"Context extraction failed: {e}")
                return {}

    def _extract_entities_from_text(self, text: str) -> list[str]:
        """Extract potential person names and entities from text."""
        import re
        
        entities = []
        
        # Pattern 1: Capitalized words (potential names)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Pattern 2: Common name patterns
        name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Middle Last
            r'\b[A-Z]\.\s*[A-Z][a-z]+\b',  # J. Smith
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # Remove duplicates and common words
        common_words = {'The', 'This', 'That', 'With', 'And', 'But', 'For', 'You', 'All', 'New', 'Now', 'Old', 'How', 'What', 'Why', 'When', 'Where'}
        entities = [e for e in set(entities) if e not in common_words and len(e) > 2]
        
        return entities

    def _analyze_video_topic(self, video_info: dict) -> dict:
        """Analyze video to determine topic/genre for context (GENERAL, not martial arts specific)."""
        title = video_info.get('title', '').lower()
        description = video_info.get('description', '').lower()
        tags = [tag.lower() for tag in video_info.get('tags', [])]
        
        content = f"{title} {description} {' '.join(tags)}"
        
        # GENERAL topic detection (removed martial arts bias)
        topics = {
            'entertainment': ['movie', 'film', 'cinema', 'actor', 'actress', 'hollywood', 'celebrity', 'star', 'famous'],
            'music': ['song', 'music', 'album', 'band', 'singer', 'concert', 'guitar', 'piano', 'artist'],
            'sports': ['sport', 'game', 'match', 'player', 'team', 'championship', 'football', 'basketball', 'athlete'],
            'politics': ['politics', 'government', 'president', 'election', 'policy', 'congress', 'senator'],
            'business': ['business', 'entrepreneur', 'ceo', 'company', 'billionaire', 'success', 'money'],
            'comedy': ['comedy', 'comedian', 'funny', 'joke', 'standup', 'humor', 'laugh'],
            'news': ['news', 'interview', 'breaking', 'report', 'journalist', 'media'],
        }
        
        detected_topics = []
        for topic, keywords in topics.items():
            if any(keyword in content for keyword in keywords):
                detected_topics.append(topic)
        
        return {
            'primary_topics': detected_topics,
            'is_celebrity_content': any(word in content for word in ['celebrity', 'star', 'famous', 'actor', 'actress', 'singer', 'artist']),
            'content_type': 'interview' if 'interview' in content else 'entertainment' if detected_topics else 'general'
        }

    async def get_prioritized_transcript(self, video_id: str, language: str = 'en') -> tuple[list[dict], str]:
        """Get transcript prioritizing manual subtitles over auto-generated."""
        
        video_url = f"https://youtube.com/watch?v={video_id}"
        
        # Try manual subtitles first
        ydl_opts_manual = {
            'writesubtitles': True,           # ONLY manual subtitles
            'writeautomaticsub': False,      # NO auto-generated
            'subtitleslangs': [language, 'en', 'en-US'],
            'subtitlesformat': 'json3',
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
            'outtmpl': tempfile.gettempdir() + '/manual_%(id)s.%(ext)s',
        }
        
        transcript = None
        transcript_type = None
        
        # ATTEMPT 1: Manual subtitles (highest quality)
        try:
            with yt_dlp.YoutubeDL(ydl_opts_manual) as ydl:
                ydl.download([video_url])
                
                subtitle_file = self._find_subtitle_file(video_id, tempfile.gettempdir(), 'manual_')
                if subtitle_file:
                    transcript = self._parse_subtitle_file(subtitle_file)
                    transcript_type = "manual"
                    logger.info(f"✅ Using MANUAL subtitles: {len(transcript)} segments")
                    os.unlink(subtitle_file)
                    
        except Exception as e:
            logger.warning(f"Manual subtitles failed: {e}")
        
        # ATTEMPT 2: Auto-generated if manual failed
        if not transcript:
            ydl_opts_auto = {
                'writesubtitles': False,         # NO manual
                'writeautomaticsub': True,       # ONLY auto-generated
                'subtitleslangs': [language, 'en', 'en-US'],
                'subtitlesformat': 'json3',
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
                'outtmpl': tempfile.gettempdir() + '/auto_%(id)s.%(ext)s',
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts_auto) as ydl:
                    ydl.download([video_url])
                    
                    subtitle_file = self._find_subtitle_file(video_id, tempfile.gettempdir(), 'auto_')
                    if subtitle_file:
                        transcript = self._parse_subtitle_file(subtitle_file)
                        transcript_type = "auto-generated"
                        logger.warning(f"⚠️ Using AUTO-GENERATED captions: {len(transcript)} segments")
                        os.unlink(subtitle_file)
                        
            except Exception as e:
                logger.error(f"Auto-generated captions failed: {e}")
                raise TranscriptNotFoundError("No captions available")
        
        return transcript, transcript_type

    def _find_subtitle_file(self, video_id: str, temp_dir: str, prefix: str = '') -> str:
        """Find downloaded subtitle file."""
        for file in os.listdir(temp_dir):
            if file.startswith(prefix + video_id) and file.endswith('.json3'):
                return os.path.join(temp_dir, file)
        return None

    def _parse_subtitle_file(self, subtitle_file: str) -> list[dict]:
        """Parse subtitle file to transcript format."""
        
        with open(subtitle_file, 'r', encoding='utf-8') as f:
            subtitle_data = json.load(f)
        
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
        
        return transcript

    def _build_correction_dictionary(self, context: dict) -> dict:
        """Build DYNAMIC correction dictionary based on video context."""
        corrections = {}
        
        # DYNAMIC APPROACH: Use entities found in video metadata
        entities = context.get('potential_entities', [])
        
        for entity in entities:
            # Generate phonetic variations for ANY name found in title/description
            phonetic_variations = self._generate_phonetic_variations(entity)
            for variation in phonetic_variations:
                if variation.lower() != entity.lower():  # Don't correct to itself
                    corrections[variation.lower()] = entity
                    
            # Generate common ASR mistakes for any celebrity name
            asr_variations = self._generate_asr_mistakes(entity)
            for variation in asr_variations:
                corrections[variation.lower()] = entity
        
        # UNIVERSAL celebrity name patterns (not domain-specific)
        corrections.update(self._get_universal_name_corrections())
        
        logger.info(f"📚 Built DYNAMIC correction dictionary with {len(corrections)} entries for entities: {entities}")
        return corrections

    def _generate_phonetic_variations(self, name: str) -> list[str]:
        """Generate common phonetic variations of a name."""
        variations = [name]
        
        name_lower = name.lower()
        
        # Common mispronunciations and phonetic rules
        phonetic_rules = [
            ('ph', 'f'), ('gh', 'f'), ('tion', 'shun'), ('sion', 'shun'),
            ('ch', 'k'), ('ck', 'k'), ('qu', 'kw'), ('x', 'ks'),
            ('c', 'k'), ('s', 'z'), ('th', 'f'), ('th', 'd'),
            ('ea', 'e'), ('ou', 'u'), ('igh', 'i'),
        ]
        
        for old, new in phonetic_rules:
            if old in name_lower:
                variations.append(name_lower.replace(old, new))
        
        # Remove duplicates
        return list(set(variations))

    def _generate_asr_mistakes(self, name: str) -> list[str]:
        """Generate common ASR (Automatic Speech Recognition) mistakes for any name."""
        variations = []
        name_parts = name.split()
        
        # For each part of the name, generate common mistakes
        for part in name_parts:
            part_lower = part.lower()
            
            # Common ASR substitutions that happen to ANY celebrity name
            asr_rules = [
                # Sound-alike endings
                ('al', 'ull'),    # Seagal → Seagull
                ('er', 'ar'),     # Miller → Millar  
                ('on', 'an'),     # Johnson → Johnsan
                ('en', 'an'),     # Steven → Stevan
                ('le', 'el'),     # Castle → Castel
                
                # Common vowel confusions
                ('i', 'e'),       # Smith → Smeth
                ('a', 'o'),       # Brad → Brod
                ('o', 'a'),       # Tom → Tam
                ('u', 'o'),       # Cruz → Croz
                
                # Common consonant confusions  
                ('f', 'ph'),      # Stephen → Stefen
                ('k', 'c'),       # Mark → Marc
                ('s', 'z'),       # Rose → Roze
                ('th', 'f'),      # Smith → Smif
                ('ch', 'sh'),     # Rich → Rish
            ]
            
            # Apply rules to generate variations
            for old, new in asr_rules:
                if old in part_lower:
                    variation = part_lower.replace(old, new)
                    variations.append(variation)
        
        # Also generate variations for the full name
        full_name_variations = []
        name_lower = name.lower()
        
        # Space removal (common in ASR)
        if ' ' in name_lower:
            full_name_variations.append(name_lower.replace(' ', ''))
            full_name_variations.append(name_lower.replace(' ', '-'))
        
        # Hyphen confusion
        if '-' in name_lower:
            full_name_variations.append(name_lower.replace('-', ' '))
            full_name_variations.append(name_lower.replace('-', ''))
        
        return variations + full_name_variations

    def _get_universal_name_corrections(self) -> dict:
        """Universal corrections that apply to common name patterns (not domain-specific)."""
        return {
            # Common title corrections
            'mister': 'Mr.',
            'doctor': 'Dr.',
            'professor': 'Professor',
            
            # Common name component fixes
            'junior': 'Jr.',
            'senior': 'Sr.',
            'the third': 'III',
            'the second': 'II',
            
            # Common ASR mistakes for any name
            'mac': 'Mc',     # McDonald → MacDonald
            'o\'': 'O\'',    # o'connor → O'Connor
            'de ': 'De ',    # de niro → De Niro
            'van ': 'Van ',  # van dyke → Van Dyke
            'la ': 'La ',    # la beouf → La Beouf
        }

    async def correct_transcript_with_context(self, transcript: list[dict], context: dict, transcript_type: str) -> list[dict]:
        """Apply intelligent corrections based on video context."""
        
        if not transcript:
            return transcript
        
        logger.info(f"🔧 Starting intelligent transcript correction ({transcript_type})")
        
        # Build correction dictionary from context
        corrections = self._build_correction_dictionary(context)
        
        if corrections:
            logger.info(f"📚 Built correction dictionary with {len(corrections)} entries")
        
        # Apply corrections
        corrected_transcript = []
        corrections_count = 0
        
        for segment in transcript:
            original_text = segment['text']
            corrected_text = self._apply_corrections(original_text, corrections)
            
            if corrected_text != original_text:
                corrections_count += 1
                logger.info(f"🔧 CORRECTED: '{original_text}' → '{corrected_text}'")
            
            corrected_segment = segment.copy()
            corrected_segment['text'] = corrected_text
            corrected_segment['original_text'] = original_text
            corrected_segment['transcript_type'] = transcript_type
            corrected_transcript.append(corrected_segment)
        
        logger.info(f"✅ Applied {corrections_count} corrections to transcript")
        return corrected_transcript

    def _apply_corrections(self, text: str, corrections: dict) -> str:
        """Apply corrections to text while preserving context."""
        import re
        
        corrected_text = text
        
        # Apply word-level corrections (case-insensitive)
        for wrong, correct in corrections.items():
            # Use word boundary regex to avoid partial matches
            pattern = r'\b' + re.escape(wrong) + r'\b'
            corrected_text = re.sub(pattern, correct, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text

    async def calculate_quality_score(self, transcript: list[dict], context: dict) -> float:
        """Calculate transcript quality score (0-1)."""
        if not transcript:
            return 0.0
        
        score_factors = []
        
        # Factor 1: Transcript type (manual > auto)
        manual_segments = sum(1 for t in transcript if t.get('transcript_type') == 'manual')
        if manual_segments > 0:
            score_factors.append(0.9)  # High score for manual subtitles
            logger.info("✅ Quality boost: Manual subtitles detected")
        else:
            score_factors.append(0.5)  # Lower score for auto-generated
            logger.info("⚠️ Quality concern: Auto-generated captions")
        
        # Factor 2: Entity recognition accuracy
        entities_found = len(context.get('potential_entities', []))
        if entities_found > 0:
            # Check how many entities appear correctly in transcript
            full_text = ' '.join(t['text'] for t in transcript)
            entities_in_transcript = sum(
                1 for entity in context['potential_entities'] 
                if entity.lower() in full_text.lower()
            )
            entity_accuracy = entities_in_transcript / entities_found
            score_factors.append(entity_accuracy)
            logger.info(f"📊 Entity accuracy: {entities_in_transcript}/{entities_found} ({entity_accuracy:.2f})")
        
        # Factor 3: Text quality indicators
        avg_segment_length = sum(len(t['text']) for t in transcript) / len(transcript)
        if avg_segment_length > 15:  # Reasonable segment length
            score_factors.append(0.8)
        elif avg_segment_length > 8:
            score_factors.append(0.6)
        else:
            score_factors.append(0.3)
        
        # Factor 4: Corrections applied (more corrections = lower initial quality)
        corrections_applied = sum(1 for t in transcript if t.get('original_text') and t['original_text'] != t['text'])
        if corrections_applied == 0:
            score_factors.append(0.9)  # No corrections needed
        elif corrections_applied / len(transcript) < 0.1:
            score_factors.append(0.7)  # Few corrections
        else:
            score_factors.append(0.5)  # Many corrections needed
        
        # Calculate weighted average
        final_score = sum(score_factors) / len(score_factors) if score_factors else 0.5
        return min(max(final_score, 0.0), 1.0)  # Clamp between 0 and 1

    async def get_transcript(self, video_id: str, language: str = 'en') -> list[dict]:
        """Enhanced transcript extraction with intelligent correction."""
        
        logger.info(f"🎯 Starting ENHANCED transcript extraction for video {video_id}")
        
        try:
            # Step 1: Extract video context
            logger.info("📊 Extracting video context...")
            context = await self.extract_video_context(video_id)
            
            # Step 2: Get prioritized transcript (manual > auto)
            logger.info("🎯 Getting prioritized transcript...")
            transcript, transcript_type = await self.get_prioritized_transcript(video_id, language)
            
            # Step 3: Apply intelligent corrections
            logger.info("🔧 Applying intelligent corrections...")
            corrected_transcript = await self.correct_transcript_with_context(
                transcript, context, transcript_type
            )
            
            # Step 4: Calculate quality score
            quality_score = await self.calculate_quality_score(corrected_transcript, context)
            
            # Step 5: Log results
            logger.info(f"🏆 ENHANCED TRANSCRIPT EXTRACTION COMPLETE:")
            logger.info(f"   📊 Segments: {len(corrected_transcript)}")
            logger.info(f"   🎯 Type: {transcript_type}")
            logger.info(f"   📈 Quality Score: {quality_score:.2f}/1.0")
            logger.info(f"   🔧 Corrections Applied: {sum(1 for t in corrected_transcript if t.get('original_text'))}")
            logger.info(f"   🎬 Video Context: {', '.join(context.get('topic_hints', {}).get('primary_topics', ['general']))}")
            
            return corrected_transcript
            
        except Exception as e:
            logger.error(f"Enhanced transcript extraction failed: {e}")
            
            # Fallback to basic method
            logger.info("🔄 Falling back to basic transcript extraction")
            try:
                return await self.get_transcript_yt_dlp_native(video_id, language)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise TranscriptNotFoundError(f"All transcript extraction methods failed for {video_id}")

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