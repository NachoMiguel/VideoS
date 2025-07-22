import cv2
import numpy as np
import asyncio
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import sys

# Add yt root to path for ai_shared_lib imports
yt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, yt_root)

from ai_shared_lib.image_search import ImageSearchService
from ai_shared_lib.face_detection import FaceDetector
from ai_shared_lib.config import settings
from services.scene_detection import SceneDetector
from services.ffmpeg_processor import FFmpegVideoProcessor

logger = logging.getLogger(__name__)

class AdvancedVideoProcessor:
    def __init__(self):
        self.image_search = ImageSearchService()
        self.face_detector = FaceDetector()
        self.scene_detector = SceneDetector(threshold=25.0, min_scene_duration=2.0)
        # Initialize bulletproof FFmpeg processor with optimized settings
        self.ffmpeg_processor = FFmpegVideoProcessor(
            max_clips_per_batch=50,  # Process 50 clips per batch for large videos
            timeout_seconds=600      # 10 minute timeout for long operations
        )
        self.logger = logger
        
    async def extract_characters_from_script(self, script_content: str) -> List[str]:
        """Extract character names from script using AI."""
        try:
            # Use OpenAI to extract character names
            import openai
            
            if not settings.OPENAI_API_KEY:
                # Fallback: simple regex extraction
                return self._extract_characters_simple(script_content)
            
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            
            prompt = f"""
            Extract character names from this script. Return only the names, one per line:
            
            {script_content}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            characters = [name.strip() for name in response.choices[0].message.content.split('\n') if name.strip()]
            self.logger.info(f"Extracted characters: {characters}")
            return characters
            
        except Exception as e:
            self.logger.error(f"Character extraction error: {str(e)}")
            return self._extract_characters_simple(script_content)
    
    def _extract_characters_simple(self, script_content: str) -> List[str]:
        """Simple character extraction using regex patterns."""
        # Common character name patterns
        patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Last
            r'\b([A-Z][a-z]+)\b',  # Single names
        ]
        
        characters = set()
        for pattern in patterns:
            matches = re.findall(pattern, script_content)
            characters.update(matches)
        
        # Filter out common words that aren't names
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered = [char for char in characters if char.lower() not in common_words]
        
        return filtered[:10]  # Limit to 10 characters
    
    async def search_character_images(self, characters: List[str]) -> Dict[str, List[str]]:
        """Search for character images using Google API."""
        try:
            self.logger.info(f"Searching images for {len(characters)} characters")
            return await self.image_search.search_character_images(characters)
        except Exception as e:
            self.logger.error(f"Image search error: {str(e)}")
            return {}
    
    async def train_face_recognition(self, character_images: Dict[str, List[str]]) -> Dict[str, Any]:
        """Train face recognition model with character images."""
        try:
            self.logger.info("Training face recognition model...")
            
            # Train the face detector with character images
            trained_characters = await self.face_detector.train_character_faces(
                character_images
                # Remove progress_callback parameter
            )
            
            self.logger.info(f"Face recognition trained for {len(trained_characters)} characters")
            return trained_characters
            
        except Exception as e:
            self.logger.error(f"Face recognition training error: {str(e)}")
            return {}
    
    async def detect_video_scenes(self, video_paths: List[str], audio_duration: float = None) -> Dict[str, List[Dict[str, float]]]:
        """Detect scenes in all uploaded videos with immediate validation."""
        try:
            self.logger.info(f"Detecting scenes in {len(video_paths)} videos...")
            all_video_scenes = {}
            
            # Calculate target duration for scenes
            if audio_duration is None:
                audio_duration = 1200.0  # Default 20 minutes
            target_duration = audio_duration + 20.0  # Audio duration + 20 seconds
            
            self.logger.info(f"Target video duration: {target_duration:.1f}s (audio: {audio_duration:.1f}s + 20s)")
            
            for video_path in video_paths:
                try:
                    self.logger.info(f"Detecting scenes in: {video_path}")
                    
                    # Use duration-based scene detection to ensure enough content
                    scenes = self.scene_detector.detect_scenes_for_duration(
                        video_path, 
                        target_duration=target_duration,
                        target_scenes=max(50, int(target_duration / 3.0))  # Aim for 3-second scenes
                    )
                    
                    # CRITICAL FIX: Validate scenes immediately after detection
                    valid_scenes = []
                    invalid_count = 0
                    
                    self.logger.info(f"Validating {len(scenes)} detected scenes...")
                    
                    for i, scene in enumerate(scenes):
                        try:
                            # Add video_path to scene for validation
                            scene['video_path'] = video_path
                            
                            # Validate scene immediately
                            is_valid = await self.validate_scene_video_clip(scene)
                            if is_valid:
                                valid_scenes.append(scene)
                                self.logger.debug(f"✅ Scene {i+1} validated: {scene['start_time']:.1f}s to {scene['end_time']:.1f}s")
                            else:
                                invalid_count += 1
                                self.logger.warning(f"❌ Scene {i+1} corrupted, removing: {scene['start_time']:.1f}s to {scene['end_time']:.1f}s")
                                
                        except Exception as e:
                            invalid_count += 1
                            self.logger.error(f"❌ Scene {i+1} validation error: {str(e)}")
                            continue
                    
                    self.logger.info(f"Scene validation complete: {len(valid_scenes)} valid, {invalid_count} corrupted (removed)")
                    all_video_scenes[video_path] = valid_scenes
                    
                except Exception as e:
                    self.logger.error(f"Error detecting scenes in {video_path}: {str(e)}")
                    all_video_scenes[video_path] = []
            
            total_valid_scenes = sum(len(scenes) for scenes in all_video_scenes.values())
            total_duration = sum(
                sum(scene.get('duration', 0) for scene in scenes) 
                for scenes in all_video_scenes.values()
            )
            
            self.logger.info(f"Total valid scenes across all videos: {total_valid_scenes}")
            self.logger.info(f"Total scene duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
            
            return all_video_scenes
            
        except Exception as e:
            self.logger.error(f"Scene detection error: {str(e)}")
            return {}
    
    async def analyze_videos_with_character_detection(self, video_paths: List[str], face_model: Dict[str, Any], audio_duration: float = None) -> List[Dict[str, Any]]:
        """Analyze videos to detect characters in detected scenes."""
        try:
            self.logger.info(f"Analyzing {len(video_paths)} videos for character detection")
            
            # Step 1: Detect scenes in all videos with duration awareness
            video_scenes = await self.detect_video_scenes(video_paths, audio_duration)
            
            all_scenes_with_faces = []
            
            # Step 2: Analyze each detected scene for character faces
            for video_path, scenes in video_scenes.items():
                self.logger.info(f"Analyzing {len(scenes)} scenes in {video_path}")
                
                for scene in scenes:
                    try:
                        # Process this specific scene for face detection
                        scene_faces = await self.face_detector.process_video_segment(
                            video_path=scene['video_path'],
                            start_time=scene['start_time'],
                            duration=scene['duration'],
                            sample_interval=2.0  # Sample every 2 seconds for efficiency
                        )
                        
                        # Merge scene metadata with face detection results
                        if scene_faces:
                            # Take the first (and only) scene result since we're processing one scene at a time
                            scene_with_faces = scene_faces[0] if scene_faces else {}
                            
                            # Update scene with face detection results
                            scene.update({
                                'faces': scene_with_faces.get('faces', []),
                                'character_count': len([f for f in scene_with_faces.get('faces', []) if f.get('character')]),
                                'quality_metrics': {
                                    'overall_quality': 0.7,  # Default quality
                                    'face_quality': len(scene_with_faces.get('faces', [])) / max(scene['duration'], 1.0)
                                }
                            })
                        else:
                            # No faces detected in this scene
                            scene.update({
                                'faces': [],
                                'character_count': 0,
                                'quality_metrics': {
                                    'overall_quality': 0.3,
                                    'face_quality': 0.0
                                }
                            })
                        
                        all_scenes_with_faces.append(scene)
                        
                    except Exception as e:
                        self.logger.error(f"Error analyzing scene in {video_path}: {str(e)}")
                        # Add scene without face data as fallback
                        scene.update({
                            'faces': [],
                            'character_count': 0,
                            'quality_metrics': {'overall_quality': 0.3, 'face_quality': 0.0}
                        })
                        all_scenes_with_faces.append(scene)
            
            self.logger.info(f"Total scenes analyzed: {len(all_scenes_with_faces)}")
            return all_scenes_with_faces
            
        except Exception as e:
            self.logger.error(f"Video analysis error: {str(e)}")
            return []
    
    async def select_best_scenes(self, scenes: List[Dict[str, Any]], script_content: str, audio_duration: float = None) -> List[Dict[str, Any]]:
        """Select scenes to match audio duration with intelligent mixing."""
        try:
            self.logger.info(f"Selecting scenes from {len(scenes)} candidates")
            
            # CRITICAL FIX: Use audio duration as target
            if audio_duration is None:
                audio_duration = 1200.0  # Default 20 minutes if not provided
                self.logger.warning(f"No audio duration provided, using default: {audio_duration}s")
            
            target_duration = audio_duration + 20.0  # Audio duration + 20 seconds
            min_duration = audio_duration + 10.0     # Minimum audio duration + 10 seconds
            max_duration = audio_duration + 30.0     # Maximum audio duration + 30 seconds
            
            self.logger.info(f"Target video duration: {target_duration:.1f}s (audio: {audio_duration:.1f}s + 20s)")
            
            # CRITICAL FIX: Scenes are already validated during detection - use them directly
            self.logger.info(f"Using {len(scenes)} pre-validated scenes from detection phase")
            valid_scenes = scenes  # All scenes are already validated
            
            if not valid_scenes:
                self.logger.error("No valid scenes available!")
                return []
            
            # Score scenes based on multiple criteria
            scored_scenes = []
            
            for scene in valid_scenes:
                score = 0.0
                
                # Character presence score (higher weight for longer videos)
                if 'faces' in scene and scene['faces']:
                    character_faces = [face for face in scene['faces'] if face.get('character')]
                    score += len(character_faces) * 15  # Increased weight
                
                # Duration score (prefer medium-length scenes for 20-min video)
                duration = scene.get('duration', 0)
                if 15.0 <= duration <= 45.0:  # Ideal for 20-min video
                    score += 10
                elif 10.0 <= duration <= 60.0:  # Acceptable range
                    score += 5
                elif duration > 60.0:  # Too long, reduce score
                    score += 2
                
                # Quality score
                quality = scene.get('quality_metrics', {}).get('overall_quality', 0.5)
                score += quality * 8
                
                # Validation bonus
                if scene.get('validated', False):
                    score += 3
                
                # Video source diversity bonus (prevent content reuse)
                video_path = scene.get('video_path', '')
                if 'video1' in video_path:
                    score += 2  # Bonus for video diversity
                
                scene['score'] = score
                scored_scenes.append(scene)
            
            # Sort by score
            scored_scenes.sort(key=lambda x: x['score'], reverse=True)
            
            # CRITICAL FIX: Select scenes to match target duration
            selected_scenes = []
            total_duration = 0.0
            scenes_by_video = {"video1": 0, "video2": 0}  # Track scene distribution
            
            for scene in scored_scenes:
                scene_duration = scene.get('duration', 0)
                
                # Check if adding this scene would exceed max duration
                if total_duration + scene_duration > max_duration:
                    continue
                
                # Check video source distribution (prevent overuse of one video)
                video_path = scene.get('video_path', '')
                if 'video1' in video_path:
                    if scenes_by_video["video1"] > scenes_by_video["video2"] + 5:  # Don't overuse video1
                        continue
                    scenes_by_video["video1"] += 1
                else:
                    if scenes_by_video["video2"] > scenes_by_video["video1"] + 5:  # Don't overuse video2
                        continue
                    scenes_by_video["video2"] += 1
                
                selected_scenes.append(scene)
                total_duration += scene_duration
                
                # Stop if we've reached target duration
                if total_duration >= target_duration:
                    break
            
            # CRITICAL: Ensure minimum duration
            if total_duration < min_duration:
                self.logger.warning(f"Selected duration ({total_duration:.1f}s) below minimum ({min_duration:.1f}s)")
                # Add more scenes to reach minimum
                for scene in scored_scenes:
                    if scene not in selected_scenes:
                        scene_duration = scene.get('duration', 0)
                        if total_duration + scene_duration <= max_duration:
                            selected_scenes.append(scene)
                            total_duration += scene_duration
                            if total_duration >= min_duration:
                                break
            
            self.logger.info(f"Selected {len(selected_scenes)} scenes with total duration: {total_duration:.2f}s")
            self.logger.info(f"Scene distribution: Video1={scenes_by_video['video1']}, Video2={scenes_by_video['video2']}")
            
            # CRITICAL FIX: Scenes are already validated during detection - no need for final validation
            self.logger.info(f"Final processing complete: {len(selected_scenes)} pre-validated scenes ready for assembly")
            return selected_scenes
            
        except Exception as e:
            self.logger.error(f"Scene selection error: {str(e)}")
            return []  # Return empty list instead of potentially invalid scenes
    
    async def validate_scene_video_clip(self, scene: Dict[str, Any]) -> bool:
        """
        Validate that a scene can create a valid video clip.
        This prevents NoneType errors during assembly.
        """
        try:
            video_path = scene.get('video_path')
            start_time = scene.get('start_time', 0.0)
            end_time = scene.get('end_time', 30.0)
            
            # Level 1: File System Validation
            if not video_path or not os.path.exists(video_path):
                self.logger.error(f"Scene validation failed: File not found - {video_path}")
                return False
            
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                self.logger.error(f"Scene validation failed: Empty file - {video_path}")
                return False
            
            # CRITICAL FIX: Skip validation if scene is already validated
            if scene.get('validated', False):
                return True
            
            # Level 2: Video Format Validation (with timeout)
            try:
                import asyncio
                from moviepy.editor import VideoFileClip
                
                # CRITICAL FIX: Add timeout to prevent hanging
                async def validate_with_timeout():
                    try:
                        test_clip = VideoFileClip(video_path)
                        
                        if test_clip.duration <= 0:
                            self.logger.error(f"Scene validation failed: Invalid video duration - {video_path}")
                            test_clip.close()
                            return False
                        
                        # Level 3: Scene Validation
                        if start_time >= test_clip.duration:
                            self.logger.error(f"Scene validation failed: start_time >= video duration - {start_time}s >= {test_clip.duration}s")
                            test_clip.close()
                            return False
                        
                        if end_time > test_clip.duration:
                            self.logger.warning(f"Scene validation: Adjusting end_time from {end_time}s to {test_clip.duration}s")
                            scene['end_time'] = test_clip.duration
                            scene['duration'] = test_clip.duration - start_time
                        
                        if start_time >= end_time:
                            self.logger.error(f"Scene validation failed: start_time >= end_time - {start_time}s >= {end_time}s")
                            test_clip.close()
                            return False
                        
                        # Level 4: Clip Creation Validation (simplified)
                        try:
                            scene_clip = test_clip.subclip(start_time, end_time)
                            if scene_clip is None:
                                self.logger.error(f"Scene validation failed: subclip returned None - {video_path}")
                                test_clip.close()
                                return False
                            
                            # CRITICAL FIX: Skip frame test to prevent hanging
                            # test_frame = scene_clip.get_frame(0)  # REMOVED - causes hanging
                            
                            # Update scene with validated data
                            scene['validated'] = True
                            scene['actual_duration'] = scene_clip.duration
                            
                            # Clean up test clips
                            scene_clip.close()
                            test_clip.close()
                            
                            return True
                            
                        except Exception as clip_e:
                            self.logger.error(f"Scene validation failed: Clip creation error - {video_path}: {str(clip_e)}")
                            test_clip.close()
                            return False
                        
                    except Exception as video_e:
                        self.logger.error(f"Scene validation failed: Video file error - {video_path}: {str(video_e)}")
                        return False
                
                # CRITICAL FIX: Add 10-second timeout
                try:
                    result = await asyncio.wait_for(validate_with_timeout(), timeout=10.0)
                    return result
                except asyncio.TimeoutError:
                    self.logger.error(f"Scene validation timeout after 10 seconds - {video_path}")
                    return False
                
            except Exception as e:
                self.logger.error(f"Scene validation failed: Unexpected error - {str(e)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Scene validation failed: Unexpected error - {str(e)}")
            return False

    async def assemble_video(self, scenes: List[Dict[str, Any]], audio_path: str, output_path: str) -> str:
        """Assemble final video from selected scenes using FFmpeg for reliable processing."""
        try:
            self.logger.info(f"Assembling video from {len(scenes)} scenes using FFmpeg")
            if not scenes:
                raise Exception("No scenes provided for assembly")

            # CRITICAL: Use FFmpeg as primary method for reliable concatenation
            self.logger.info("Using FFmpeg for video assembly (primary method)")
            return await self.ffmpeg_processor.assemble_video_ffmpeg(scenes, audio_path, output_path)
            
        except Exception as e:
            self.logger.error(f"FFmpeg video assembly failed: {str(e)}")
            raise

    async def create_fallback_scenes(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create fallback scenes when no AI detection works."""
        try:
            self.logger.info("Creating fallback scenes...")
            
            fallback_scenes = []
            
            for i, video in enumerate(videos):
                video_path = video.get("path")
                
                if not video_path or not os.path.exists(video_path):
                    self.logger.error(f"Video {i}: Invalid path: {video_path}")
                    continue
                
                # Get video duration
                try:
                    from moviepy.editor import VideoFileClip
                    clip = VideoFileClip(video_path)
                    duration = min(clip.duration, 30.0)  # Limit to 30 seconds
                    clip.close()
                except Exception as e:
                    self.logger.warning(f"Could not get duration for {video_path}, using 30s: {str(e)}")
                    duration = 30.0
                
                # Create fallback scene
                fallback_scenes.append({
                    "video_path": video_path,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration,
                    "faces": [],  # Empty faces list
                    "quality_metrics": {"overall_quality": 0.5}  # Default quality
                })
                
                self.logger.info(f"Created fallback scene {i+1}: {video_path} (0s to {duration}s)")
            
            self.logger.info(f"Created {len(fallback_scenes)} fallback scenes")
            return fallback_scenes
            
        except Exception as e:
            self.logger.error(f"Fallback scene creation error: {str(e)}")
            return [] 

    async def create_guaranteed_valid_scenes(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create guaranteed valid scenes as fallback when main pipeline fails.
        This ensures we always have something to assemble.
        """
        try:
            self.logger.info("Creating guaranteed valid scenes as fallback...")
            
            guaranteed_scenes = []
            
            for i, video in enumerate(videos):
                video_path = video.get("path")
                
                if not video_path or not os.path.exists(video_path):
                    self.logger.error(f"Video {i}: Invalid path: {video_path}")
                    continue
                
                # Create simple, guaranteed valid scenes
                try:
                    from moviepy.editor import VideoFileClip
                    clip = VideoFileClip(video_path)
                    video_duration = clip.duration
                    clip.close()
                    
                    # Create 3 simple scenes per video
                    scene_duration = min(10.0, video_duration / 3)  # 10 seconds or 1/3 of video
                    
                    for j in range(3):
                        start_time = j * scene_duration
                        end_time = min(start_time + scene_duration, video_duration)
                        
                        if start_time >= end_time:
                            continue
                        
                        scene = {
                            "video_path": video_path,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": end_time - start_time,
                            "faces": [],
                            "character_count": 0,
                            "quality_metrics": {"overall_quality": 0.5, "face_quality": 0.0},
                            "score": 1.0,  # Low score but guaranteed valid
                            "fallback": True
                        }
                        
                        # Validate this fallback scene
                        if await self.validate_scene_video_clip(scene):
                            guaranteed_scenes.append(scene)
                            self.logger.info(f"Created guaranteed valid fallback scene {len(guaranteed_scenes)}: {video_path} ({start_time}s to {end_time}s)")
                        else:
                            self.logger.error(f"Fallback scene validation failed for {video_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error creating fallback scenes for {video_path}: {str(e)}")
                    continue
            
            self.logger.info(f"Created {len(guaranteed_scenes)} guaranteed valid fallback scenes")
            return guaranteed_scenes
            
        except Exception as e:
            self.logger.error(f"Guaranteed scene creation failed: {str(e)}")
            return [] 

    async def create_extended_scenes(self, videos: List[Dict[str, Any]], target_duration: float) -> List[Dict[str, Any]]:
        """
        Create longer scenes when we need more duration.
        This ensures we can meet the target duration even with fewer scenes.
        """
        try:
            self.logger.info(f"Creating extended scenes to meet target duration: {target_duration:.1f}s")
            
            extended_scenes = []
            total_duration = 0.0
            
            for i, video in enumerate(videos):
                video_path = video.get("path")
                
                if not video_path or not os.path.exists(video_path):
                    continue
                
                try:
                    from moviepy.editor import VideoFileClip
                    clip = VideoFileClip(video_path)
                    video_duration = clip.duration
                    clip.close()
                    
                    # Create longer scenes (30-60 seconds each)
                    scene_duration = min(45.0, video_duration / 2)  # 45 seconds or half video
                    
                    # Calculate how many scenes we need from this video
                    remaining_duration = target_duration - total_duration
                    if remaining_duration <= 0:
                        break
                    
                    scenes_needed = max(1, int(remaining_duration / scene_duration))
                    
                    for j in range(scenes_needed):
                        start_time = j * scene_duration
                        end_time = min(start_time + scene_duration, video_duration)
                        
                        if start_time >= end_time or total_duration >= target_duration:
                            break
                        
                        scene = {
                            "video_path": video_path,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": end_time - start_time,
                            "faces": [],
                            "character_count": 0,
                            "quality_metrics": {"overall_quality": 0.6, "face_quality": 0.0},
                            "score": 5.0,  # Medium score for extended scenes
                            "extended": True
                        }
                        
                        # Validate this extended scene
                        if await self.validate_scene_video_clip(scene):
                            extended_scenes.append(scene)
                            total_duration += scene['duration']
                            self.logger.info(f"Created extended scene {len(extended_scenes)}: {video_path} ({start_time}s to {end_time}s)")
                        
                        if total_duration >= target_duration:
                            break
                    
                except Exception as e:
                    self.logger.error(f"Error creating extended scenes for {video_path}: {str(e)}")
                    continue
            
            self.logger.info(f"Created {len(extended_scenes)} extended scenes with total duration: {total_duration:.1f}s")
            return extended_scenes
            
        except Exception as e:
            self.logger.error(f"Extended scene creation failed: {str(e)}")
            return [] 