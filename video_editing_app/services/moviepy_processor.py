#!/usr/bin/env python3
"""
MoviePy-based video processor as fallback for FFmpeg failures.
"""

import os
import logging
import tempfile
import asyncio
import time
import psutil
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip
    from moviepy.video.fx import resize
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MoviePyVideoProcessor:
    """MoviePy-based video processor with comprehensive error handling and chunked processing."""
    
    def __init__(self, max_clips_per_batch: int = 10, timeout_seconds: int = 300):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp(prefix="moviepy_processor_")
        self.max_clips_per_batch = max_clips_per_batch  # REDUCED from 50 to 10
        self.timeout_seconds = timeout_seconds
        
        if not MOVIEPY_AVAILABLE:
            self.logger.warning("MoviePy is not available - some features may be limited")
            # Don't raise error, just log warning
        
        self.logger.info(f"MoviePy processor initialized with temp dir: {self.temp_dir}")
        self.logger.info(f"Chunked processing: max {max_clips_per_batch} clips per batch")
    
    def _check_memory_availability(self) -> bool:
        """Check if system has enough memory for video processing."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Require at least 2GB available for video processing
            if available_gb < 2.0:
                self.logger.warning(f"⚠️ Low memory available: {available_gb:.1f}GB (need 2GB+)")
                return False
            
            self.logger.info(f"✅ Memory available: {available_gb:.1f}GB")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check memory: {e}")
            return True  # Assume OK if we can't check
    
    async def assemble_video_moviepy(self, scenes: List[Dict[str, Any]], audio_path: str, output_path: str) -> str:
        """
        Video assembly using MoviePy with CHUNKED processing for large numbers of clips.
        
        Args:
            scenes: List of scene dictionaries with video_path, start_time, end_time
            audio_path: Path to audio file
            output_path: Output video path
            
        Returns:
            Path to the assembled video
            
        Raises:
            Exception: If any step fails with detailed error information
        """
        temp_files_to_cleanup = []
        
        try:
            self.logger.info(f"🚀 Starting MoviePy video assembly with {len(scenes)} scenes")
            
            # Check memory availability before starting
            if not self._check_memory_availability():
                raise Exception("❌ Insufficient memory for video processing")
            
            # Step 1: Create individual clip files using MoviePy
            clip_files = await self._create_individual_clips(scenes)
            temp_files_to_cleanup.extend(clip_files)
            
            if not clip_files:
                raise Exception("❌ No valid clips could be created - assembly cannot proceed")
            
            self.logger.info(f"✅ Created {len(clip_files)} individual clip files")
            
            # Step 2: Concatenate clips using CHUNKED processing
            final_video = await self._concatenate_clips_chunked(clip_files, output_path)
            temp_files_to_cleanup.append(final_video)
            
            # Step 3: Verify video file exists and is valid
            if not self._verify_video_file(final_video):
                raise Exception(f"❌ Video concatenation failed - output file is invalid: {final_video}")
            
            self.logger.info(f"✅ Video concatenation verified: {final_video}")
            
            # Step 4: Add audio to final video (only if video exists)
            final_video_with_audio = await self._add_audio_moviepy(final_video, audio_path, output_path)
            
            # Step 5: Final verification
            if not self._verify_video_file(final_video_with_audio):
                raise Exception(f"❌ Final video file is invalid: {final_video_with_audio}")
            
            self.logger.info(f"🎉 MoviePy video assembly completed: {final_video_with_audio}")
            return final_video_with_audio
            
        except Exception as e:
            self.logger.error(f"💥 MoviePy video assembly failed: {str(e)}")
            raise
        finally:
            # Always cleanup temporary files
            await self._cleanup_temp_files(temp_files_to_cleanup)
    
    async def assemble_silent_video_moviepy(self, scenes: List[Dict[str, Any]], output_path: str) -> str:
        """
        Silent video assembly using MoviePy with CHUNKED processing.
        No audio merging to avoid issues.
        
        Args:
            scenes: List of scene dictionaries with video_path, start_time, end_time
            output_path: Output video path (silent)
            
        Returns:
            Path to the assembled silent video
            
        Raises:
            Exception: If any step fails with detailed error information
        """
        temp_files_to_cleanup = []
        
        try:
            self.logger.info(f"🚀 Starting MoviePy SILENT video assembly with {len(scenes)} scenes")
            
            # Check memory availability before starting
            if not self._check_memory_availability():
                raise Exception("❌ Insufficient memory for video processing")
            
            # Step 1: Create individual silent clip files using MoviePy
            clip_files = await self._create_individual_silent_clips(scenes)
            temp_files_to_cleanup.extend(clip_files)
            
            if not clip_files:
                raise Exception("❌ No valid clips could be created - assembly cannot proceed")
            
            self.logger.info(f"✅ Created {len(clip_files)} individual silent clip files")
            
            # Step 2: Concatenate clips using CHUNKED processing (no audio)
            final_video = await self._concatenate_silent_clips_chunked(clip_files, output_path)
            temp_files_to_cleanup.append(final_video)
            
            # Step 3: Verify video file exists and is valid
            if not self._verify_video_file(final_video):
                raise Exception(f"❌ Silent video concatenation failed - output file is invalid: {final_video}")
            
            self.logger.info(f"✅ Silent video concatenation verified: {final_video}")
            
            # Step 4: Final verification (no audio step needed)
            if not self._verify_video_file(final_video):
                raise Exception(f"❌ Final silent video file is invalid: {final_video}")
            
            self.logger.info(f"🎉 MoviePy SILENT video assembly completed: {final_video}")
            return final_video
            
        except Exception as e:
            self.logger.error(f"💥 MoviePy silent video assembly failed: {str(e)}")
            raise
        finally:
            # Always cleanup temporary files
            await self._cleanup_temp_files(temp_files_to_cleanup)

    async def _concatenate_clips_chunked(self, clip_files: List[str], output_path: str) -> str:
        """
        Concatenate clips using MoviePy with CHUNKED processing to avoid memory exhaustion.
        """
        try:
            self.logger.info(f"📦 Concatenating {len(clip_files)} clips with CHUNKED MoviePy processing")
            
            if len(clip_files) <= self.max_clips_per_batch:
                # Small number of clips - process all at once
                self.logger.info(f"📦 Processing all {len(clip_files)} clips in single batch")
                return await self._concatenate_clips_moviepy(clip_files, output_path)
            
            # Large number of clips - use chunked processing
            self.logger.info(f"📦 Using chunked processing: {len(clip_files)} clips in batches of {self.max_clips_per_batch}")
            
            # Process clips in chunks
            chunked_videos = []
            for i in range(0, len(clip_files), self.max_clips_per_batch):
                chunk = clip_files[i:i + self.max_clips_per_batch]
                chunk_num = i // self.max_clips_per_batch + 1
                total_chunks = (len(clip_files) + self.max_clips_per_batch - 1) // self.max_clips_per_batch
                
                self.logger.info(f"📦 Processing chunk {chunk_num}/{total_chunks}: {len(chunk)} clips")
                
                # Create temporary output for this chunk
                chunk_output = os.path.join(self.temp_dir, f"chunk_{chunk_num:03d}.mp4")
                
                # Process this chunk
                chunk_video = await self._concatenate_clips_moviepy(chunk, chunk_output)
                chunked_videos.append(chunk_video)
                
                # Force garbage collection after each chunk
                import gc
                gc.collect()
                
                self.logger.info(f"✅ Chunk {chunk_num}/{total_chunks} completed: {chunk_video}")
            
            # Now concatenate all chunked videos
            if len(chunked_videos) == 1:
                # Only one chunk - just rename it
                os.rename(chunked_videos[0], output_path)
                return output_path
            else:
                # Multiple chunks - concatenate them
                self.logger.info(f"📦 Concatenating {len(chunked_videos)} chunks into final video")
                return await self._concatenate_clips_moviepy(chunked_videos, output_path)
            
        except Exception as e:
            raise Exception(f"Chunked MoviePy concatenation failed: {str(e)}")
    
    async def _create_individual_clips(self, scenes: List[Dict[str, Any]]) -> List[str]:
        """Create individual clip files from scenes using MoviePy with comprehensive validation."""
        clip_files = []
        failed_clips = 0
        
        for i, scene in enumerate(scenes):
            try:
                video_path = scene.get('video_path')
                start_time = scene.get('start_time', 0)
                end_time = scene.get('end_time', 0)
                
                # Validate scene data
                if not video_path or not os.path.exists(video_path):
                    self.logger.error(f"❌ Invalid video path for scene {i+1}: {video_path}")
                    failed_clips += 1
                    continue
                
                if end_time <= start_time:
                    self.logger.error(f"❌ Invalid time range for scene {i+1}: {start_time}s to {end_time}s")
                    failed_clips += 1
                    continue
                
                # Create output clip file path
                clip_filename = f"clip_{i+1:04d}.mp4"
                clip_path = os.path.join(self.temp_dir, clip_filename)
                
                self.logger.info(f"🎬 Creating clip {i+1}/{len(scenes)}: {os.path.basename(video_path)} ({start_time:.1f}s to {end_time:.1f}s)")
                
                # Use MoviePy to extract the clip
                await self._extract_clip_moviepy(video_path, start_time, end_time, clip_path)
                
                # Verify clip was created successfully
                if self._verify_video_file(clip_path):
                    clip_files.append(clip_path)
                    self.logger.debug(f"✅ Clip {i+1} created successfully: {os.path.basename(clip_path)}")
                else:
                    self.logger.error(f"❌ Clip {i+1} creation failed - file is invalid: {clip_path}")
                    failed_clips += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error creating clip {i+1}: {str(e)}")
                failed_clips += 1
                continue
        
        self.logger.info(f"📊 Clip creation summary: {len(clip_files)} successful, {failed_clips} failed")
        
        if failed_clips > len(scenes) * 0.5:  # More than 50% failed
            self.logger.warning(f"⚠️ High failure rate: {failed_clips}/{len(scenes)} clips failed")
        
        return clip_files
    
    async def _extract_clip_moviepy(self, video_path: str, start_time: float, end_time: float, output_path: str) -> None:
        """Extract a clip using MoviePy with enhanced resource management."""
        try:
            loop = asyncio.get_event_loop()
            
            def extract_clip():
                video = None
                clip = None
                try:
                    video = VideoFileClip(video_path)
                    clip = video.subclip(start_time, end_time)
                    
                    clip.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True,
                        verbose=False,
                        logger=None,
                        preset='fast',  # Faster encoding
                        threads=2  # Limit threads to reduce memory usage
                    )
                    
                finally:
                    # CRITICAL: Clean up resources to prevent memory leaks
                    if clip:
                        try:
                            clip.close()
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error closing clip: {str(e)}")
                    
                    if video:
                        try:
                            video.close()
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error closing video: {str(e)}")
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
            
            await asyncio.wait_for(
                loop.run_in_executor(None, extract_clip),
                timeout=self.timeout_seconds
            )
            
        except asyncio.TimeoutError:
            raise Exception(f"MoviePy clip extraction timed out after {self.timeout_seconds}s")
        except Exception as e:
            raise Exception(f"MoviePy clip extraction failed: {str(e)}")
    
    async def _concatenate_clips_moviepy(self, clip_files: List[str], output_path: str) -> str:
        """Concatenate clips using MoviePy with enhanced resource management."""
        try:
            self.logger.info(f"📦 Concatenating {len(clip_files)} clips with MoviePy")
            
            loop = asyncio.get_event_loop()
            
            def concatenate_clips():
                clips = []
                try:
                    # Load all clips with proper resource management
                    for i, clip_file in enumerate(clip_files):
                        try:
                            clip = VideoFileClip(clip_file)
                            clips.append(clip)
                            self.logger.debug(f"📦 Loaded clip {i+1}/{len(clip_files)}: {os.path.basename(clip_file)}")
                        except Exception as e:
                            self.logger.error(f"❌ Failed to load clip {i+1}: {str(e)}")
                            continue
                    
                    if not clips:
                        raise Exception("No valid clips to concatenate")
                    
                    # Concatenate clips
                    final_clip = concatenate_videoclips(clips)
                    
                    # Write final video with optimized settings
                    final_clip.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True,
                        verbose=False,
                        logger=None,
                        preset='fast',  # Faster encoding
                        threads=2  # Limit threads to reduce memory usage
                    )
                    
                    return output_path
                    
                finally:
                    # CRITICAL: Clean up all clips to prevent memory leaks
                    for clip in clips:
                        try:
                            clip.close()
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error closing clip: {str(e)}")
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
            
            await asyncio.wait_for(
                loop.run_in_executor(None, concatenate_clips),
                timeout=self.timeout_seconds * 2  # Longer timeout for concatenation
            )
            
            return output_path
            
        except asyncio.TimeoutError:
            raise Exception(f"MoviePy concatenation timed out after {self.timeout_seconds * 2}s")
        except Exception as e:
            raise Exception(f"MoviePy concatenation failed: {str(e)}")
    
    async def _add_audio_moviepy(self, video_path: str, audio_path: str, final_output_path: str) -> str:
        """Add audio to video using MoviePy with enhanced resource management."""
        try:
            if not os.path.exists(audio_path):
                self.logger.warning(f"⚠️ Audio file not found: {audio_path}")
                return video_path
            
            self.logger.info(f"🎵 Adding audio to video using MoviePy")
            
            loop = asyncio.get_event_loop()
            
            def add_audio():
                video = None
                audio = None
                final_video = None
                try:
                    video = VideoFileClip(video_path)
                    audio = AudioFileClip(audio_path)
                    
                    # Resize audio to match video duration
                    if audio.duration > video.duration:
                        audio = audio.subclip(0, video.duration)
                    
                    final_video = video.set_audio(audio)
                    final_video.write_videofile(
                        final_output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True,
                        verbose=False,
                        logger=None,
                        preset='fast',  # Faster encoding
                        threads=2  # Limit threads to reduce memory usage
                    )
                    
                finally:
                    # CRITICAL: Clean up all resources to prevent memory leaks
                    if final_video:
                        try:
                            final_video.close()
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error closing final video: {str(e)}")
                    
                    if audio:
                        try:
                            audio.close()
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error closing audio: {str(e)}")
                    
                    if video:
                        try:
                            video.close()
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error closing video: {str(e)}")
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
            
            await asyncio.wait_for(
                loop.run_in_executor(None, add_audio),
                timeout=self.timeout_seconds
            )
            
            return final_output_path
            
        except asyncio.TimeoutError:
            self.logger.warning(f"⚠️ Audio addition timed out - returning video without audio")
            return video_path
        except Exception as e:
            self.logger.warning(f"⚠️ Could not add audio: {str(e)}")
            return video_path
    
    def _verify_video_file(self, file_path: str) -> bool:
        """Comprehensive verification that a video file exists and is valid."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"❌ File does not exist: {file_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self.logger.error(f"❌ File is empty: {file_path}")
                return False
            
            if file_size < 1024:  # Less than 1KB
                self.logger.error(f"❌ File is too small ({file_size} bytes): {file_path}")
                return False
            
            # Check file extension
            if not file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.logger.warning(f"⚠️ Unexpected file extension: {file_path}")
            
            self.logger.debug(f"✅ File verification passed: {os.path.basename(file_path)} ({file_size} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ File verification failed for {file_path}: {str(e)}")
            return False
    
    async def _cleanup_temp_files(self, files_to_cleanup: List[str]) -> None:
        """Clean up temporary files with enhanced error handling for Windows file locking."""
        try:
            cleaned_count = 0
            failed_count = 0
            
            for file_path in files_to_cleanup:
                if file_path and os.path.exists(file_path):
                    try:
                        # Force garbage collection before file deletion
                        import gc
                        gc.collect()
                        
                        # Try to remove file with retry logic for Windows
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                os.remove(file_path)
                                cleaned_count += 1
                                self.logger.debug(f"🧹 Cleaned up: {os.path.basename(file_path)}")
                                break
                            except PermissionError as pe:
                                if attempt < max_retries - 1:
                                    self.logger.debug(f"⏳ File locked, retrying in 1s: {os.path.basename(file_path)}")
                                    await asyncio.sleep(1)
                                else:
                                    raise pe
                    except Exception as e:
                        failed_count += 1
                        self.logger.warning(f"⚠️ Failed to clean up {os.path.basename(file_path)}: {str(e)}")
            
            # Clean up temp directory if empty
            if os.path.exists(self.temp_dir):
                try:
                    remaining_files = os.listdir(self.temp_dir)
                    if not remaining_files:
                        os.rmdir(self.temp_dir)
                        self.logger.debug(f"🧹 Cleaned up temp directory: {self.temp_dir}")
                    else:
                        self.logger.debug(f"📁 Temp directory not empty ({len(remaining_files)} files remaining): {self.temp_dir}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to clean up temp directory: {str(e)}")
            
            if cleaned_count > 0:
                self.logger.info(f"🧹 Cleaned up {cleaned_count} temporary files")
            
            if failed_count > 0:
                self.logger.warning(f"⚠️ Failed to clean up {failed_count} files (likely locked by MoviePy)")
                
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup of resources."""
        try:
            # Force cleanup of any remaining files
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                try:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                except Exception:
                    pass  # Ignore cleanup errors in destructor
        except Exception:
            pass  # Ignore all errors in destructor 

    async def _create_individual_silent_clips(self, scenes: List[Dict[str, Any]]) -> List[str]:
        """Create individual silent clip files from scenes using MoviePy."""
        clip_files = []
        failed_clips = 0
        
        for i, scene in enumerate(scenes):
            try:
                video_path = scene.get('video_path')
                start_time = scene.get('start_time', 0)
                end_time = scene.get('end_time', 0)
                
                # Validate scene data
                if not video_path or not os.path.exists(video_path):
                    self.logger.warning(f"⚠️ Scene {i+1}: Video file not found - {video_path}")
                    failed_clips += 1
                    continue
                
                # Create temporary clip file
                clip_filename = f"silent_clip_{i:03d}_{int(start_time)}_{int(end_time)}.mp4"
                clip_path = os.path.join(self.temp_dir, clip_filename)
                
                # Extract silent clip using MoviePy
                await self._extract_silent_clip_moviepy(video_path, start_time, end_time, clip_path)
                
                # Verify clip was created successfully
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    clip_files.append(clip_path)
                    self.logger.info(f"✅ Silent clip {i+1} created: {os.path.basename(clip_path)}")
                else:
                    self.logger.warning(f"⚠️ Silent clip {i+1} creation failed - file is empty or missing")
                    failed_clips += 1
                    
            except Exception as e:
                self.logger.error(f"❌ Silent clip {i+1} creation error: {str(e)}")
                failed_clips += 1
        
        self.logger.info(f"📊 Silent clip creation summary: {len(clip_files)} successful, {failed_clips} failed")
        return clip_files

    async def _extract_silent_clip_moviepy(self, video_path: str, start_time: float, end_time: float, output_path: str) -> None:
        """Extract a silent clip from video using MoviePy."""
        try:
            def extract_silent_clip():
                try:
                    # Load video clip
                    video_clip = VideoFileClip(video_path)
                    
                    # Extract subclip (no audio)
                    subclip = video_clip.subclip(start_time, end_time)
                    
                    # Write silent video (no audio)
                    subclip.write_videofile(
                        output_path,
                        codec='libx264',
                        audio=False,  # No audio
                        verbose=False,
                        logger=None
                    )
                    
                    # Clean up
                    video_clip.close()
                    subclip.close()
                    
                except Exception as e:
                    self.logger.error(f"MoviePy silent clip extraction error: {str(e)}")
                    raise
            
            # Run with timeout
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, extract_silent_clip),
                timeout=self.timeout_seconds
            )
            
        except asyncio.TimeoutError:
            raise Exception(f"Silent clip extraction timed out after {self.timeout_seconds} seconds")
        except Exception as e:
            raise Exception(f"Silent clip extraction failed: {str(e)}")

    async def _concatenate_silent_clips_chunked(self, clip_files: List[str], output_path: str) -> str:
        """Concatenate silent clips using chunked processing to handle large numbers."""
        try:
            if len(clip_files) <= self.max_clips_per_batch:
                # Small number of clips - direct concatenation
                self.logger.info(f"📦 Direct silent concatenation for {len(clip_files)} clips")
                return await self._concatenate_silent_clips_moviepy(clip_files, output_path)
            else:
                # Large number of clips - chunked processing
                self.logger.info(f"📦 Chunked silent concatenation for {len(clip_files)} clips (max {self.max_clips_per_batch} per chunk)")
                
                # Process in chunks
                chunks = [clip_files[i:i + self.max_clips_per_batch] 
                         for i in range(0, len(clip_files), self.max_clips_per_batch)]
                
                temp_chunk_files = []
                
                # Create intermediate chunk files
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"chunk_{i:03d}.mp4"
                    chunk_path = os.path.join(self.temp_dir, chunk_filename)
                    
                    self.logger.info(f"📦 Processing chunk {i+1}/{len(chunks)} with {len(chunk)} clips")
                    chunk_file = await self._concatenate_silent_clips_moviepy(chunk, chunk_path)
                    temp_chunk_files.append(chunk_file)
                
                # Concatenate all chunks into final video
                self.logger.info(f"📦 Concatenating {len(temp_chunk_files)} chunks into final video")
                final_video = await self._concatenate_silent_clips_moviepy(temp_chunk_files, output_path)
                
                return final_video
                
        except Exception as e:
            self.logger.error(f"💥 Chunked silent concatenation failed: {str(e)}")
            raise

    async def _concatenate_silent_clips_moviepy(self, clip_files: List[str], output_path: str) -> str:
        """Concatenate silent clips using MoviePy."""
        try:
            def concatenate_silent_clips():
                try:
                    # Load all video clips (no audio)
                    video_clips = []
                    for clip_path in clip_files:
                        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                            clip = VideoFileClip(clip_path)
                            video_clips.append(clip)
                    
                    if not video_clips:
                        raise Exception("No valid video clips to concatenate")
                    
                    # Concatenate clips (no audio)
                    final_clip = concatenate_videoclips(video_clips, method="compose")
                    
                    # Write silent video
                    final_clip.write_videofile(
                        output_path,
                        codec='libx264',
                        audio=False,  # No audio
                        verbose=False,
                        logger=None
                    )
                    
                    # Clean up
                    for clip in video_clips:
                        clip.close()
                    final_clip.close()
                    
                except Exception as e:
                    self.logger.error(f"MoviePy silent concatenation error: {str(e)}")
                    raise
            
            # Run with timeout
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, concatenate_silent_clips),
                timeout=self.timeout_seconds
            )
            
            return output_path
            
        except asyncio.TimeoutError:
            raise Exception(f"Silent concatenation timed out after {self.timeout_seconds} seconds")
        except Exception as e:
            raise Exception(f"Silent concatenation failed: {str(e)}") 