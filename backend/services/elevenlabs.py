from typing import Optional, BinaryIO, Dict, List, Any
import requests
import logging
import json
import asyncio
from pathlib import Path
import aiohttp
import httpx
import time
from core.config import settings
from core.exceptions import VoiceNotFoundError, ElevenLabsError, AudioGenerationError, CreditExhaustionError
from core.parallel import parallel_processor, parallel_task
from core.credit_manager import credit_manager, ServiceType
from core.parallel_error_handler import OperationType
import os
import aiofiles

from .youtube import YouTubeService
from .text_cleaner import text_cleaner
from .entity_variation_manager import entity_variation_manager

logger = logging.getLogger(__name__)

class ElevenLabsService:
    """Enhanced ElevenLabs service with parallel processing and multi-account rotation."""
    
    def __init__(self):
        # Validate configuration with detailed status
        config_valid, config_message = settings.validate_elevenlabs_config()
        
        if not config_valid:
            logger.error(f"ElevenLabs configuration error: {config_message}")
            raise ElevenLabsError(f"ElevenLabs API keys not configured: {config_message}")
        
        # Log configuration status
        logger.info(f"ElevenLabs initialization: {config_message}")
        config_status = settings.elevenlabs_config_status
        logger.info(f"Configuration details: {config_status}")
        
        self.api_keys = settings.elevenlabs_api_keys
        self.current_key_index = 0
        self.voice_id = settings.elevenlabs_voice_id
        self.model_id = settings.elevenlabs_model_id
        self.timeout = settings.elevenlabs_timeout
        self.retry_attempts = settings.elevenlabs_retry_attempts
        
        # Credit tracking per account
        self.credit_usage = {i: 0 for i in range(len(self.api_keys))}
        self.max_credits_per_account = settings.max_credits_per_account
        self.warning_threshold = settings.credit_warning_threshold
        
        # Parallel processing settings
        self.parallel_enabled = settings.parallel_audio_generation
        self.max_concurrent = min(len(self.api_keys), settings.concurrent_api_calls)
        
        self.base_url = "https://api.elevenlabs.io/v1"
        self.voice_cache = {}
        self.usage_tracker = {key: 0 for key in self.api_keys if key}
        
        # Performance warnings
        if len(self.api_keys) == 1:
            logger.warning("Running with single ElevenLabs API key - throughput will be limited")
        elif len(self.api_keys) < 4:
            logger.info(f"Running with {len(self.api_keys)} ElevenLabs API keys - consider adding more for optimal performance")
        
        # 🎯 NEW: Track exhausted accounts
        self.exhausted_accounts = set()
        
        logger.info(f"ElevenLabs service initialized with {len(self.api_keys)} accounts, parallel processing: {self.parallel_enabled}")
        
        # Initialize TTS cleanup components
        self.youtube_service = YouTubeService()
    
    def _get_next_api_key(self) -> tuple[str, int]:
        """Get the next available API key with automatic rotation using credit manager."""
        try:
            account = credit_manager.get_available_account(ServiceType.ELEVENLABS)
            # Find the index of this account in our local list
            for i, key in enumerate(self.api_keys):
                if key == account.api_key:
                    return account.api_key, i
            
            # Fallback to old method if not found
            logger.warning("Account not found in local list, using fallback method")
            return self._get_next_api_key_fallback()
            
        except CreditExhaustionError as e:
            logger.error(f"ElevenLabs credit exhaustion: {str(e)}")
            raise e
        except Exception as e:
            logger.warning(f"Error getting account from credit manager: {str(e)}, using fallback")
            return self._get_next_api_key_fallback()
    
    def _get_next_api_key_fallback(self) -> tuple[str, int]:
        """Fallback method for API key rotation with better account selection."""
        attempts = 0
        while attempts < len(self.api_keys):
            key_index = (self.current_key_index + attempts) % len(self.api_keys)
            current_usage = self.credit_usage[key_index]
            
            # 🎯 NEW: Check if account has been marked as exhausted
            if hasattr(self, 'exhausted_accounts') and key_index in self.exhausted_accounts:
                attempts += 1
                continue
            
            if current_usage < self.max_credits_per_account * self.warning_threshold:
                self.current_key_index = (key_index + 1) % len(self.api_keys)
                return self.api_keys[key_index], key_index
            
            attempts += 1
        
        # All accounts near limit, use the one with least usage
        best_key_index = min(self.credit_usage.keys(), key=lambda k: self.credit_usage[k])
        logger.warning(f"All accounts near credit limit, using account {best_key_index}")
        return self.api_keys[best_key_index], best_key_index
    
    def _update_credit_usage(self, key_index: int, characters_used: int):
        """Update credit usage for an account."""
        # Rough estimate: 1000 characters ≈ 1 credit
        credits_used = max(1, characters_used // 1000)
        self.credit_usage[key_index] += credits_used
        
        usage_percentage = (self.credit_usage[key_index] / self.max_credits_per_account) * 100
        logger.info(f"Account {key_index} usage: {usage_percentage:.1f}% ({self.credit_usage[key_index]}/{self.max_credits_per_account} credits)")
        
        if usage_percentage > self.warning_threshold * 100:
            logger.warning(f"Account {key_index} approaching credit limit: {usage_percentage:.1f}%")
    
    @parallel_task('api_call')
    async def generate_audio_segment(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        key_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate audio for a single text segment."""
        try:
            # Get API key
            if key_index is not None and key_index < len(self.api_keys):
                api_key = self.api_keys[key_index]
                used_key_index = key_index
            else:
                api_key, used_key_index = self._get_next_api_key()
            
            # Prepare request
            voice_id = voice_id or self.voice_id
            model_id = model_id or self.model_id
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            
            data = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            # Make API request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=data, headers=headers)
                
                if response.status_code == 200:
                    # Update credit usage
                    self._update_credit_usage(used_key_index, len(text))
                    
                    return {
                        "success": True,
                        "audio_data": response.content,
                        "text": text,
                        "key_index": used_key_index,
                        "characters": len(text)
                    }
                else:
                    error_msg = f"ElevenLabs API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "text": text,
                        "key_index": used_key_index
                    }
        
        except Exception as e:
            logger.error(f"Audio generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": text
            }
    
    async def generate_audio_parallel(
        self,
        text_segments: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Generate audio for multiple text segments in parallel."""
        if not self.parallel_enabled or len(text_segments) <= 1:
            return await self._generate_audio_sequential(text_segments, progress_callback)
        
        try:
            logger.info(f"Starting parallel audio generation for {len(text_segments)} segments")
            
            # Prepare parallel tasks with load balancing
            api_calls = []
            for i, text in enumerate(text_segments):
                # Distribute segments across available API keys
                key_index = i % len(self.api_keys)
                
                api_calls.append({
                    'func': self.generate_audio_segment,
                    'args': [text],
                    'kwargs': {'key_index': key_index},
                    'segment_id': f'audio_segment_{i}'
                })
            
            # Execute in parallel with progress tracking
            async def audio_progress_callback(message, progress):
                if progress_callback:
                    await progress_callback(f"Generating audio: {message}", progress)
            
            results = await parallel_processor.parallel_api_calls(api_calls, audio_progress_callback)
            
            # Process results
            audio_results = []
            successful = 0
            
            for i, result in enumerate(results):
                if result.success and result.result:
                    audio_results.append(result.result)
                    if result.result.get('success', False):
                        successful += 1
                else:
                    logger.error(f"Audio generation failed for segment {i}: {result.error}")
                    # Add placeholder result
                    audio_results.append({
                        "success": False,
                        "error": str(result.error),
                        "text": text_segments[i] if i < len(text_segments) else "",
                        "segment_index": i
                    })
            
            logger.info(f"Parallel audio generation completed: {successful}/{len(text_segments)} successful")
            return audio_results
            
        except Exception as e:
            logger.error(f"Parallel audio generation error: {str(e)}")
            # Fallback to sequential processing
            return await self._generate_audio_sequential(text_segments, progress_callback)
    
    async def _generate_audio_sequential(
        self,
        text_segments: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Generate audio sequentially (fallback method)."""
        results = []
        total_segments = len(text_segments)
        
        for i, text in enumerate(text_segments):
            try:
                result = await self.generate_audio_segment(text)
                results.append(result)
                
                if progress_callback:
                    progress = ((i + 1) / total_segments) * 100
                    await progress_callback(f"Generated audio {i+1}/{total_segments}", progress)
                    
            except Exception as e:
                logger.error(f"Sequential audio generation error for segment {i}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "text": text,
                    "segment_index": i
                })
        
        return results
    
    async def save_audio_files(
        self,
        audio_results: List[Dict[str, Any]],
        session_id: str,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """Save generated audio files to disk."""
        if output_dir is None:
            output_dir = Path(settings.temp_dir) / session_id / "audio"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        for i, result in enumerate(audio_results):
            if result.get('success', False) and result.get('audio_data'):
                try:
                    filename = f"segment_{i:03d}.mp3"
                    file_path = output_dir / filename
                    
                    with open(file_path, 'wb') as f:
                        f.write(result['audio_data'])
                    
                    saved_files.append(str(file_path))
                    logger.debug(f"Saved audio segment: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error saving audio segment {i}: {str(e)}")
                    saved_files.append(None)
            else:
                logger.warning(f"Skipping failed audio segment {i}")
                saved_files.append(None)
        
        successful_files = [f for f in saved_files if f is not None]
        logger.info(f"Saved {len(successful_files)}/{len(audio_results)} audio files")
        
        return saved_files
    
    async def combine_audio_segments(
        self,
        audio_files: List[str],
        output_path: str,
        crossfade_duration: float = 0.1
    ) -> str:
        """Combine multiple audio segments into a single file."""
        try:
            from moviepy.editor import AudioFileClip, concatenate_audioclips
            
            # Filter out None values (failed segments)
            valid_files = [f for f in audio_files if f is not None and Path(f).exists()]
            
            if not valid_files:
                raise AudioGenerationError("No valid audio files to combine")
            
            # Load audio clips
            clips = []
            for file_path in valid_files:
                try:
                    clip = AudioFileClip(file_path)
                    clips.append(clip)
                except Exception as e:
                    logger.warning(f"Failed to load audio file {file_path}: {str(e)}")
            
            if not clips:
                raise AudioGenerationError("No audio clips could be loaded")
            
            # Combine with crossfade
            if len(clips) == 1:
                final_audio = clips[0]
            else:
                final_audio = concatenate_audioclips(clips, method="compose")
            
            # Save combined audio
            final_audio.write_audiofile(output_path, logger=None)
            
            # Cleanup individual clips
            for clip in clips:
                clip.close()
            
            logger.info(f"Combined audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error combining audio segments: {str(e)}")
            raise AudioGenerationError(f"Failed to combine audio: {str(e)}")
    
    async def get_voice_list(self) -> List[Dict[str, Any]]:
        """Get available voices from ElevenLabs."""
        try:
            api_key, _ = self._get_next_api_key()
            
            url = "https://api.elevenlabs.io/v1/voices"
            headers = {"xi-api-key": api_key}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('voices', [])
                else:
                    logger.error(f"Failed to get voice list: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting voice list: {str(e)}")
            return []
    
    def get_credit_status(self) -> Dict[str, Any]:
        """Get current credit usage status across all accounts."""
        total_used = sum(self.credit_usage.values())
        total_available = len(self.api_keys) * self.max_credits_per_account
        
        account_status = []
        for i, usage in self.credit_usage.items():
            percentage = (usage / self.max_credits_per_account) * 100
            account_status.append({
                "account_index": i,
                "credits_used": usage,
                "credits_total": self.max_credits_per_account,
                "usage_percentage": percentage,
                "status": "warning" if percentage > self.warning_threshold * 100 else "ok"
            })
        
        return {
            "total_credits_used": total_used,
            "total_credits_available": total_available,
            "overall_usage_percentage": (total_used / total_available) * 100,
            "accounts": account_status,
            "parallel_enabled": self.parallel_enabled,
            "max_concurrent": self.max_concurrent
        }

    async def get_voice_id(self, character_name: str) -> str:
        """Get or create a voice ID for a character."""
        if character_name in self.voice_cache:
            return self.voice_cache[character_name]

        try:
            # First check if we have a saved voice
            voices = await self._get_voices()
            for voice in voices:
                if voice['name'].lower() == character_name.lower():
                    self.voice_cache[character_name] = voice['voice_id']
                    return voice['voice_id']

            # If not found, create a new voice
            voice_id = await self._create_voice(character_name)
            self.voice_cache[character_name] = voice_id
            return voice_id

        except Exception as e:
            logger.error(f"Failed to get/create voice for {character_name}: {str(e)}")
            raise VoiceNotFoundError(f"Could not get voice for {character_name}")

    async def generate_speech(
        self,
        text: str,
        character_name: str,
        output_path: Path,
        stability: float = 0.5,
        similarity_boost: float = 0.75
    ) -> Path:
        """Generate speech for text using a character's voice."""
        try:
            voice_id = await self.get_voice_id(character_name)
            
            async with self.semaphore:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.base_url}/text-to-speech/{voice_id}"
                    
                    data = {
                        "text": text,
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {
                            "stability": stability,
                            "similarity_boost": similarity_boost
                        }
                    }
                    
                    async with session.post(
                        url,
                        headers=self.headers,
                        json=data
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise AudioGenerationError(
                                f"TTS generation failed: {error_text}"
                            )
                        
                        # Save the audio file
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, 'wb') as f:
                            f.write(await response.read())

            return output_path

        except Exception as e:
            logger.error(f"Speech generation error: {str(e)}")
            raise AudioGenerationError(str(e))

    async def _get_voices(self) -> list:
        """Get list of available voices."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/voices",
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        raise VoiceNotFoundError("Failed to get voices list")
                    return await response.json()

        except Exception as e:
            logger.error(f"Failed to get voices: {str(e)}")
            raise VoiceNotFoundError("Could not retrieve voices")

    async def _create_voice(self, name: str) -> str:
        """Create a new voice for a character."""
        try:
            # For test mode, use predefined voices
            if settings.test_mode:
                test_voices = {
                    "jean claude vandamme": "voice_1",
                    "steven seagal": "voice_2"
                }
                return test_voices.get(name.lower(), "default_voice")

            url = f"{self.base_url}/voices/add"
            
            # In production, we would add voice samples here
            # For now, we'll use a default voice
            data = {
                "name": name,
                "description": f"AI-generated voice for {name}",
                "labels": {"type": "character_voice"}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        raise VoiceNotFoundError(f"Failed to create voice for {name}")
                    result = await response.json()
                    return result['voice_id']

        except Exception as e:
            logger.error(f"Voice creation error: {str(e)}")
            raise VoiceNotFoundError(f"Could not create voice for {name}")

    async def check_credit_balance(self) -> int:
        """Check remaining character credits."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/user/subscription",
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        raise Exception("Failed to get subscription info")
                    data = await response.json()
                    return data.get('character_count', 0)

        except Exception as e:
            logger.error(f"Failed to check credit balance: {str(e)}")
            return 0

    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice to free up credits."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/voices/{voice_id}",
                    headers=self.headers
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Failed to delete voice {voice_id}: {str(e)}")
            return False

    def _chunk_text(self, text: str, max_chars: int = 2500) -> list[str]:
        """Split text into chunks for TTS processing."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_chars:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    async def generate_audio_from_script(self, script_content: str, output_path: str, video_context: Optional[Dict] = None) -> str:
        """Generate audio from script content using ElevenLabs TTS with preprocessing."""
        try:
            logger.info("Starting audio generation from script with TTS preprocessing")
            
            # 🎯 NEW: TTS Pipeline Preprocessing with video context
            cleaned_script = await self._preprocess_script_for_tts(script_content, video_context)
            
            # Parse script into segments
            segments = await self._parse_script_segments(cleaned_script)
            
            # 🎯 FIX: Create temp directory before generating audio
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            logger.info(f"✅ Created temp directory: {temp_dir.absolute()}")
            
            # Generate audio for each segment
            audio_files = []
            for i, segment in enumerate(segments):
                try:
                    # 🎯 FIX: Use absolute path and ensure directory exists
                    segment_file = temp_dir / f"segment_{i}.mp3"
                    segment_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    await self._generate_segment_audio(segment, str(segment_file))
                    audio_files.append(str(segment_file))
                    
                    logger.info(f"Generated audio for segment {i+1}/{len(segments)}")
                    
                except Exception as e:
                    logger.error(f"Error generating audio for segment {i}: {str(e)}")
                    continue
            
            if not audio_files:
                raise AudioGenerationError("No audio segments were generated successfully")
            
            # Combine audio files
            final_audio = await self._combine_audio_files(audio_files, output_path)
            
            # Cleanup temporary files
            for temp_file in audio_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except:
                    pass
            
            logger.info(f"Audio generation completed: {final_audio}")
            return final_audio
            
        except Exception as e:
            logger.error(f"Audio generation error: {str(e)}")
            raise AudioGenerationError(f"Failed to generate audio: {str(e)}")

    async def _preprocess_script_for_tts(self, script_content: str, video_context: Optional[Dict] = None) -> str:
        """Preprocess script content for optimal TTS generation."""
        try:
            logger.info("Starting TTS script preprocessing")
            
            # Step 1: TTS Character Cleanup
            logger.info("Step 1: TTS Character Cleanup")
            cleaned_script = text_cleaner.clean_for_voice(script_content)
            
            # Step 2: Name Correction with Entity Variations
            logger.info("Step 2: Name Correction and Entity Variations")
            corrected_script = await self._apply_name_corrections(cleaned_script, video_context)
            
            # Step 3: TTS Optimization
            logger.info("Step 3: TTS Optimization")
            optimized_script = await self._optimize_for_tts(corrected_script)
            
            # Step 4: Final Validation
            logger.info("Step 4: Final Validation")
            final_script = await self._validate_for_tts(optimized_script)
            
            logger.info(f"TTS preprocessing completed. Original: {len(script_content)} chars, Final: {len(final_script)} chars")
            return final_script
            
        except Exception as e:
            logger.error(f"TTS preprocessing error: {str(e)}")
            # Fallback to original script if preprocessing fails
            return script_content

    async def _apply_name_corrections(self, script: str, video_context: Optional[Dict] = None) -> str:
        """
        Apply name corrections for TTS preprocessing.
        
        This method focuses ONLY on:
        1. Basic ASR mistake corrections (fallback)
        2. Entity variation management for natural speech
        
        Entity extraction and dynamic corrections are already handled in the script generation phase.
        """
        try:
            logger.info("🎯 Starting TTS name correction and variation application")
            
            # Step 1: Apply basic ASR corrections (fallback only)
            corrected_script = await self._apply_basic_corrections(script)
            
            # Step 2: Apply entity variations for natural speech (NEW - solves repetitive names)
            corrected_script = await self._apply_entity_variations(corrected_script, video_context)
            
            return corrected_script
            
        except Exception as e:
            logger.error(f"TTS name correction error: {str(e)}")
            return script

    async def _apply_basic_corrections(self, script: str) -> str:
        """Apply basic ASR mistake corrections (fallback only)."""
        # Common ASR mistake patterns - minimal fallback
        corrections = {
            "JeanClaude Jean-Claude Van Damme": "Jean-Claude Van Damme",
            "Steven Steven Seagal": "Steven Seagal",
            "JeanClaude Van Damme": "Jean-Claude Van Damme",  # Missing hyphen
            "Vanam": "Jean-Claude Van Damme",  # Common ASR mistake
            "Seagull": "Steven Seagal",  # Common ASR mistake
        }
        
        corrected_script = script
        for mistake, correction in corrections.items():
            if mistake in corrected_script:
                corrected_script = corrected_script.replace(mistake, correction)
                logger.info(f"Applied fallback correction: '{mistake}' → '{correction}'")
        
        return corrected_script

    async def _apply_entity_variations(self, script: str, video_context: Optional[Dict] = None) -> str:
        """
        Apply entity variations for natural speech patterns.
        
        This is the NEW feature that solves the repetitive name problem.
        It takes the already-corrected script and applies natural variations.
        """
        try:
            # Extract entities from video context (if available)
            entities = []
            if video_context:
                entities = video_context.get('potential_entities', [])
                logger.info(f"🎯 Found {len(entities)} entities for variation management: {entities}")
            
            # If no entities from context, try to extract from script content
            if not entities:
                entities = self._extract_entities_from_script(script)
                logger.info(f"🎯 Extracted {len(entities)} entities from script: {entities}")
            
            # Apply entity variations if entities found
            if entities:
                # Reset mention counts for new script
                entity_variation_manager.reset_mentions()
                
                # Register entities and generate variations
                entity_variation_manager.register_entities(entities)
                
                # Apply natural variations to script
                varied_script = entity_variation_manager.apply_variations_to_text(script)
                
                # Log statistics
                stats = entity_variation_manager.get_statistics()
                logger.info(f"📊 Entity variation applied: {stats}")
                
                return varied_script
            else:
                logger.warning("⚠️ No entities found for variation management")
                return script
            
        except Exception as e:
            logger.error(f"Entity variation error: {str(e)}")
            return script

    def _extract_entities_from_script(self, script: str) -> List[str]:
        """
        Extract entities from script content for variation management.
        
        This is a simple extraction for TTS preprocessing only.
        The main entity extraction happens during script generation.
        """
        import re
        
        # Simple pattern to find potential entity names in script
        # Look for capitalized word sequences that appear multiple times
        words = script.split()
        potential_entities = []
        
        # Find capitalized word sequences
        for i in range(len(words) - 1):
            if (words[i][0].isupper() and 
                words[i+1][0].isupper() and 
                len(words[i]) > 2 and 
                len(words[i+1]) > 2):
                
                entity = f"{words[i]} {words[i+1]}"
                if entity not in potential_entities:
                    potential_entities.append(entity)
        
        # Filter out common words and short entities
        common_words = {'The', 'This', 'That', 'With', 'And', 'But', 'For', 'You', 'All', 'New', 'Now', 'Old'}
        filtered_entities = []
        
        for entity in potential_entities:
            if (len(entity) >= 8 and 
                not any(word in entity for word in common_words)):
                filtered_entities.append(entity)
        
        return filtered_entities[:5]  # Limit to top 5 entities

    async def _optimize_for_tts(self, script: str) -> str:
        """Optimize script for better TTS quality."""
        try:
            # Replace problematic characters and patterns
            optimizations = {
                # Remove or replace problematic symbols
                "#": "",
                "@": "",
                "\\": "",
                "|": "",
                "^": "",
                "~": "",
                "=": "",
                "<": "",
                ">": "",
                "$": "",
                "€": "",
                "£": "",
                "¥": "",
                "¢": "",
                "₹": "",
                "₽": "",
                "₩": "",
                "₪": "",
                "₦": "",
                "₨": "",
                "₫": "",
                "₭": "",
                "₮": "",
                "₯": "",
                "₰": "",
                "₱": "",
                "₲": "",
                "₳": "",
                "₴": "",
                "₵": "",
                "₶": "",
                "₷": "",
                "₸": "",
                "₹": "",
                "₺": "",
                "₻": "",
                "₼": "",
                "₽": "",
                "₾": "",
                "₿": "",
                
                # Fix common abbreviations
                "Mr.": "Mister",
                "Mrs.": "Missus", 
                "Dr.": "Doctor",
                "Prof.": "Professor",
                "vs.": "versus",
                "etc.": "and so on",
                "i.e.": "that is",
                "e.g.": "for example",
            }
            
            optimized_script = script
            for pattern, replacement in optimizations.items():
                if pattern in optimized_script:
                    optimized_script = optimized_script.replace(pattern, replacement)
            
            # Fix number formatting for TTS
            import re
            # Convert numbers to words for better pronunciation
            number_patterns = [
                (r'\b(\d+)\b', lambda m: self._number_to_words(int(m.group(1)))),
            ]
            
            for pattern, replacement_func in number_patterns:
                optimized_script = re.sub(pattern, replacement_func, optimized_script)
            
            return optimized_script
            
        except Exception as e:
            logger.error(f"TTS optimization error: {str(e)}")
            return script

    def _number_to_words(self, num: int) -> str:
        """Convert numbers to words for better TTS pronunciation."""
        try:
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
                    return f"{tens[num // 10]} {words[num % 10]}"
            elif num < 1000:
                if num % 100 == 0:
                    return f"{words[num // 100]} hundred"
                else:
                    return f"{words[num // 100]} hundred {self._number_to_words(num % 100)}"
            else:
                return str(num)  # Fallback for large numbers
        except:
            return str(num)

    async def _validate_for_tts(self, script: str) -> str:
        """Final validation and cleanup for TTS."""
        try:
            # Remove any remaining problematic characters
            import re
            
            # Remove multiple spaces
            script = re.sub(r'\s+', ' ', script)
            
            # Remove leading/trailing whitespace
            script = script.strip()
            
            # Ensure script is not empty
            if not script:
                raise AudioGenerationError("Script is empty after TTS preprocessing")
            
            # Log validation results
            logger.info(f"TTS validation completed. Final length: {len(script)} characters")
            
            return script
            
        except Exception as e:
            logger.error(f"TTS validation error: {str(e)}")
            return script
    
    async def _parse_script_segments(self, script_content: str) -> List[Dict[str, Any]]:
        """Parse script into manageable segments for TTS."""
        try:
            # Split script into sentences/paragraphs
            import re
            sentences = re.split(r'[.!?]+', script_content)
            
            segments = []
            current_segment = ""
            max_segment_length = 500  # Characters per segment
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed the limit
                if len(current_segment + sentence) > max_segment_length and current_segment:
                    segments.append({
                        "text": current_segment.strip(),
                        "voice": await self._select_voice_for_segment(current_segment),
                        "settings": {
                            "stability": 0.75,
                            "similarity_boost": 0.75,
                            "style": 0.0,
                            "use_speaker_boost": True
                        }
                    })
                    current_segment = sentence
                else:
                    current_segment += " " + sentence if current_segment else sentence
            
            # Add final segment
            if current_segment.strip():
                segments.append({
                    "text": current_segment.strip(),
                    "voice": await self._select_voice_for_segment(current_segment),
                    "settings": {
                        "stability": 0.75,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True
                    }
                })
            
            logger.info(f"Parsed script into {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error parsing script segments: {str(e)}")
            return [{"text": script_content[:500], "voice": "default", "settings": {}}]
    
    async def _select_voice_for_segment(self, text: str) -> str:
        """Select appropriate voice for text segment."""
        try:
            # 🎯 FIX: Use ONLY the configured voice ID from settings
            # No more multiple voices - just ONE voice as configured
            configured_voice_id = self.voice_id
            
            logger.info(f"🎤 Using configured voice ID: {configured_voice_id}")
            return configured_voice_id
                
        except Exception as e:
            logger.error(f"Error selecting voice: {str(e)}")
            # Fallback to configured voice
            return self.voice_id
    
    async def _generate_segment_audio(self, segment: Dict[str, Any], output_file: str) -> None:
        """Generate audio for a single segment with forced account rotation."""
        # 🎯 FIX: Try each account explicitly
        for account_index in range(len(self.api_keys)):
            try:
                api_key = self.api_keys[account_index]
                
                # 🎯 NEW: Skip exhausted accounts
                if hasattr(self, 'exhausted_accounts') and account_index in self.exhausted_accounts:
                    logger.info(f"⏭️ Skipping exhausted Account {account_index + 1}")
                    continue
                
                # 🎯 NEW: Log which account is being used
                logger.info(f"🎤 Using ElevenLabs Account {account_index + 1} for segment audio generation")
                logger.info(f" Account {account_index + 1} API Key: {api_key[:10]}...{api_key[-4:]}")
                
                voice_id = segment["voice"]
                text = segment["text"]
                settings = segment.get("settings", {})
                
                headers = {
                    "xi-api-key": api_key,
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": settings.get("stability", 0.75),
                        "similarity_boost": settings.get("similarity_boost", 0.75),
                        "style": settings.get("style", 0.0),
                        "use_speaker_boost": settings.get("use_speaker_boost", True)
                    }
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.base_url}/text-to-speech/{voice_id}",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        # FIX: Ensure output directory exists
                        output_path = Path(output_file)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save audio file
                        async with aiofiles.open(output_file, 'wb') as f:
                            await f.write(response.content)
                        
                        # Track usage
                        self.usage_tracker[api_key] += len(text)
                        
                        logger.info(f"✅ Account {account_index + 1} - Saved segment audio: {output_file}")
                        return  # Success - exit loop
                        
                    elif response.status_code == 429:
                        # Rate limit hit, mark account as exhausted
                        logger.warning(f"⚠️ Account {account_index + 1} rate limit hit, marking as exhausted")
                        if not hasattr(self, 'exhausted_accounts'):
                            self.exhausted_accounts = set()
                        self.exhausted_accounts.add(account_index)
                        continue  # Try next account
                        
                    elif response.status_code == 401:
                        # Authentication failed, mark account as exhausted
                        logger.error(f"❌ Account {account_index + 1} authentication failed (401), marking as exhausted")
                        if not hasattr(self, 'exhausted_accounts'):
                            self.exhausted_accounts = set()
                        self.exhausted_accounts.add(account_index)
                        continue  # Try next account
                        
                    else:
                        # Other error, try next account
                        logger.error(f"❌ Account {account_index + 1} API error: {response.status_code}")
                        continue  # Try next account
                
            except Exception as e:
                logger.error(f"Error generating segment audio with Account {account_index + 1}: {str(e)}")
                continue  # Try next account
        
        # All accounts failed
        raise AudioGenerationError(f"All {len(self.api_keys)} accounts failed for segment audio generation")
    
    async def _combine_audio_files(self, audio_files: List[str], output_path: str) -> str:
        """Combine multiple audio files into one."""
        try:
            # Use pydub to combine audio files
            from pydub import AudioSegment
            
            combined = AudioSegment.empty()
            
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    segment = AudioSegment.from_mp3(audio_file)
                    combined += segment
                    # Add small pause between segments
                    combined += AudioSegment.silent(duration=500)  # 0.5 second pause
            
            # Export combined audio
            combined.export(output_path, format="mp3")
            
            logger.info(f"Combined {len(audio_files)} audio files into {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error combining audio files: {str(e)}")
            # Fallback: just use the first audio file
            if audio_files and os.path.exists(audio_files[0]):
                import shutil
                shutil.copy2(audio_files[0], output_path)
                return output_path
            raise AudioGenerationError(f"Audio combination failed: {str(e)}")
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices from ElevenLabs."""
        try:
            api_key = self._get_next_api_key()
            headers = {"xi-api-key": api_key}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/voices",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    voices = data.get("voices", [])
                    
                    # Cache voices for future use
                    self.voice_cache = {voice["voice_id"]: voice for voice in voices}
                    
                    return voices
                else:
                    logger.error(f"Failed to get voices: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}")
            return []
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for API keys."""
        return {
            "usage_tracker": self.usage_tracker,
            "current_key_index": self.current_key_index,
            "total_characters": sum(self.usage_tracker.values()),
            "available_keys": len([key for key in self.api_keys if key])
        }

    def get_configuration_status(self) -> dict:
        """Get current configuration status for monitoring/debugging."""
        return {
            "service_initialized": True,
            "total_api_keys": len(self.api_keys),
            "parallel_enabled": self.parallel_enabled,
            "max_concurrent": self.max_concurrent,
            "voice_id": self.voice_id,
            "model_id": self.model_id,
            "timeout": self.timeout,
            "credit_usage": self.credit_usage,
            "config_status": settings.elevenlabs_config_status
        }

# Global service instance placeholder
_elevenlabs_service: Optional[ElevenLabsService] = None

def get_elevenlabs_service() -> ElevenLabsService:
    """Get or create the ElevenLabs service instance with lazy initialization."""
    global _elevenlabs_service
    if _elevenlabs_service is None:
        _elevenlabs_service = ElevenLabsService()
    return _elevenlabs_service

# Legacy compatibility - this will now use lazy initialization
@property
def elevenlabs_service() -> ElevenLabsService:
    """Legacy property for backward compatibility."""
    return get_elevenlabs_service()

# Create a module-level property that acts like the old global instance
import sys
sys.modules[__name__].elevenlabs_service = property(lambda self: get_elevenlabs_service()) 