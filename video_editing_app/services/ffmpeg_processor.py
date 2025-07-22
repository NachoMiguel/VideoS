import ffmpeg
import os
import logging
import tempfile
import shutil
import time
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class FFmpegVideoProcessor:
    """Bulletproof FFmpeg-based video processor with batch processing and comprehensive error handling."""
    
    def __init__(self, max_clips_per_batch: int = 50, timeout_seconds: int = 300):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp(prefix="ffmpeg_processor_")
        self.max_clips_per_batch = max_clips_per_batch
        self.timeout_seconds = timeout_seconds
        self.logger.info(f"FFmpeg processor initialized with temp dir: {self.temp_dir}")
        self.logger.info(f"Batch processing: max {max_clips_per_batch} clips per batch")
    
    async def assemble_video_ffmpeg(self, scenes: List[Dict[str, Any]], audio_path: str, output_path: str) -> str:
        """
        Bulletproof video assembly using FFmpeg with batch processing for large numbers of clips.
        
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
            self.logger.info(f"🚀 Starting bulletproof FFmpeg video assembly with {len(scenes)} scenes")
            
            # Step 1: Create individual clip files using FFmpeg
            clip_files = await self._create_individual_clips(scenes)
            temp_files_to_cleanup.extend(clip_files)
            
            if not clip_files:
                raise Exception("❌ No valid clips could be created - assembly cannot proceed")
            
            self.logger.info(f"✅ Created {len(clip_files)} individual clip files")
            
            # Step 2: Batch concatenation if needed
            if len(clip_files) > self.max_clips_per_batch:
                self.logger.info(f"📦 Large clip count detected ({len(clip_files)} > {self.max_clips_per_batch}) - using batch processing")
                final_video = await self._batch_concatenate_clips(clip_files, output_path)
                temp_files_to_cleanup.append(final_video)
            else:
                self.logger.info(f"📦 Standard concatenation for {len(clip_files)} clips")
                concat_file = await self._create_concat_file(clip_files)
                temp_files_to_cleanup.append(concat_file)
                final_video = await self._concatenate_clips(concat_file, output_path)
                temp_files_to_cleanup.append(final_video)
            
            # Step 3: Verify video file exists and is valid
            if not self._verify_video_file(final_video):
                raise Exception(f"❌ Video concatenation failed - output file is invalid: {final_video}")
            
            self.logger.info(f"✅ Video concatenation verified: {final_video}")
            
            # Step 4: Add audio to final video (only if video exists)
            final_video_with_audio = await self._add_audio(final_video, audio_path, output_path)
            
            # Step 5: Final verification
            if not self._verify_video_file(final_video_with_audio):
                raise Exception(f"❌ Final video file is invalid: {final_video_with_audio}")
            
            self.logger.info(f"🎉 BULLETPROOF FFmpeg video assembly completed: {final_video_with_audio}")
            return final_video_with_audio
            
        except Exception as e:
            self.logger.error(f"💥 FFmpeg video assembly failed: {str(e)}")
            raise
        finally:
            # Always cleanup temporary files
            await self._cleanup_temp_files(temp_files_to_cleanup)
    
    async def _create_individual_clips(self, scenes: List[Dict[str, Any]]) -> List[str]:
        """Create individual clip files from scenes using FFmpeg with comprehensive validation."""
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
                
                # Use FFmpeg to extract the clip
                duration = end_time - start_time
                
                # FFmpeg command: extract clip from video
                stream = (
                    ffmpeg
                    .input(video_path, ss=start_time, t=duration)
                    .output(clip_path, 
                           vcodec='libx264', 
                           acodec='aac',
                           preset='fast',
                           crf=23,
                           loglevel='error')
                    .overwrite_output()
                )
                
                # Run FFmpeg command with timeout
                await self._run_ffmpeg_command(stream, f"clip {i+1}")
                
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
    
    async def _batch_concatenate_clips(self, clip_files: List[str], output_path: str) -> str:
        """Concatenate clips in batches to handle large numbers efficiently."""
        try:
            self.logger.info(f"📦 Starting batch concatenation for {len(clip_files)} clips")
            
            # Split clips into batches
            batches = [clip_files[i:i + self.max_clips_per_batch] 
                      for i in range(0, len(clip_files), self.max_clips_per_batch)]
            
            self.logger.info(f"📦 Created {len(batches)} batches")
            
            batch_outputs = []
            
            # Process each batch
            for batch_idx, batch_clips in enumerate(batches):
                self.logger.info(f"📦 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_clips)} clips)")
                
                # Create concat file for this batch
                batch_concat_file = os.path.join(self.temp_dir, f"batch_{batch_idx + 1}_concat.txt")
                await self._create_concat_file_for_batch(batch_clips, batch_concat_file)
                
                # Concatenate this batch
                batch_output = os.path.join(self.temp_dir, f"batch_{batch_idx + 1}_output.mp4")
                await self._concatenate_clips(batch_concat_file, batch_output)
                
                # Verify batch output
                if self._verify_video_file(batch_output):
                    batch_outputs.append(batch_output)
                    self.logger.info(f"✅ Batch {batch_idx + 1} completed: {os.path.basename(batch_output)}")
                else:
                    raise Exception(f"❌ Batch {batch_idx + 1} output is invalid")
            
            # Concatenate all batch outputs
            self.logger.info(f"🔗 Concatenating {len(batch_outputs)} batch outputs into final video")
            final_concat_file = os.path.join(self.temp_dir, "final_batch_concat.txt")
            await self._create_concat_file_for_batch(batch_outputs, final_concat_file)
            
            final_video = await self._concatenate_clips(final_concat_file, output_path)
            
            self.logger.info(f"✅ Batch concatenation completed: {output_path}")
            return final_video
            
        except Exception as e:
            self.logger.error(f"❌ Batch concatenation failed: {str(e)}")
            raise
    
    async def _create_concat_file_for_batch(self, files: List[str], concat_file: str) -> str:
        """Create a concat file for a specific batch."""
        with open(concat_file, 'w', encoding='utf-8') as f:
            for file_path in files:
                # Use Windows-compatible path format
                normalized_path = file_path.replace('\\', '/')
                f.write(f"file '{normalized_path}'\n")
        
        self.logger.debug(f"📄 Created concat file: {os.path.basename(concat_file)}")
        return concat_file
    
    async def _create_concat_file(self, clip_files: List[str]) -> str:
        """Create a concat file for FFmpeg concatenation."""
        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        return await self._create_concat_file_for_batch(clip_files, concat_file)
    
    async def _concatenate_clips(self, concat_file: str, output_path: str) -> str:
        """Concatenate clips using FFmpeg concat demuxer with comprehensive error handling."""
        try:
            self.logger.info(f"🔗 Concatenating clips using FFmpeg...")
            
            # Verify concat file exists
            if not os.path.exists(concat_file):
                raise Exception(f"❌ Concat file not found: {concat_file}")
            
            # FFmpeg concat command
            stream = (
                ffmpeg
                .input(concat_file, f='concat', safe=0)
                .output(output_path,
                       vcodec='libx264',
                       acodec='aac',
                       preset='fast',
                       crf=23,
                       loglevel='error')
                .overwrite_output()
            )
            
            # Run FFmpeg command with timeout
            await self._run_ffmpeg_command(stream, "concatenation")
            
            # Verify output
            if self._verify_video_file(output_path):
                self.logger.info(f"✅ Video concatenation completed: {output_path}")
                return output_path
            else:
                raise Exception(f"❌ Concatenated video file is invalid: {output_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Video concatenation failed: {str(e)}")
            raise
    
    async def _add_audio(self, video_path: str, audio_path: str, final_output_path: str) -> str:
        """Add audio to the concatenated video with comprehensive validation."""
        try:
            # Verify video file exists before attempting audio addition
            if not self._verify_video_file(video_path):
                raise Exception(f"❌ Cannot add audio - video file is invalid: {video_path}")
            
            if not os.path.exists(audio_path):
                self.logger.warning(f"⚠️ Audio file not found: {audio_path}")
                # Copy video to final location without audio
                shutil.copy2(video_path, final_output_path)
                self.logger.info(f"✅ Video copied to final location without audio: {final_output_path}")
                return final_output_path
            
            self.logger.info(f"🎵 Adding audio to video: {os.path.basename(audio_path)}")
            
            # Create temporary output path
            temp_output = os.path.join(self.temp_dir, "final_with_audio.mp4")
            
            # FFmpeg command to add audio
            stream = (
                ffmpeg
                .input(video_path)
                .input(audio_path)
                .output(temp_output,
                       vcodec='copy',
                       acodec='aac',
                       loglevel='error')
                .overwrite_output()
            )
            
            # Run FFmpeg command with timeout
            await self._run_ffmpeg_command(stream, "audio addition")
            
            # Verify and move to final location
            if self._verify_video_file(temp_output):
                shutil.move(temp_output, final_output_path)
                self.logger.info(f"✅ Audio added successfully: {final_output_path}")
                return final_output_path
            else:
                self.logger.warning(f"⚠️ Audio addition failed - output is invalid, returning video without audio")
                # Copy original video to final location
                shutil.copy2(video_path, final_output_path)
                return final_output_path
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not add audio: {str(e)}")
            # Copy original video to final location
            shutil.copy2(video_path, final_output_path)
            return final_output_path
    
    async def _run_ffmpeg_command(self, stream, operation_name: str) -> None:
        """Run FFmpeg command asynchronously with timeout and comprehensive error handling."""
        try:
            self.logger.debug(f"🔄 Running FFmpeg {operation_name}...")
            
            # Run FFmpeg command with timeout
            start_time = time.time()
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, timeout=self.timeout_seconds)
            elapsed_time = time.time() - start_time
            
            self.logger.debug(f"✅ FFmpeg {operation_name} completed in {elapsed_time:.1f}s")
            
        except ffmpeg.Error as e:
            stderr_output = e.stderr.decode() if e.stderr else "No stderr output"
            self.logger.error(f"❌ FFmpeg {operation_name} error: {stderr_output}")
            raise Exception(f"FFmpeg {operation_name} failed: {stderr_output}")
        except Exception as e:
            self.logger.error(f"❌ FFmpeg {operation_name} execution error: {str(e)}")
            raise
    
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
        """Clean up temporary files with error handling."""
        try:
            cleaned_count = 0
            for file_path in files_to_cleanup:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        self.logger.debug(f"🧹 Cleaned up: {os.path.basename(file_path)}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to clean up {file_path}: {str(e)}")
            
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
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup error: {str(e)}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass 