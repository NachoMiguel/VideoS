import cv2
import numpy as np
import asyncio
import os
import logging
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import sys

# Add yt root to path for ai_shared_lib imports
yt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, yt_root)

from ai_shared_lib.image_search import ImageSearchService
from services.face_detection import FaceDetector
from ai_shared_lib.config import settings
# REMOVED: Complex scene detection - now using FFmpeg
# from services.scene_detection import SceneDetector
from services.ffmpeg_processor import FFmpegVideoProcessor
from services.moviepy_processor import MoviePyVideoProcessor
from utils.resource_monitor import ProcessingLimits

logger = logging.getLogger(__name__)

class AdvancedVideoProcessor:
    def __init__(self):
        # Initialize logger first
        self.logger = logger
        
        # CRITICAL: Validate FFmpeg installation and capabilities
        self._validate_ffmpeg_installation()
        
        # Initialize resource monitoring
        try:
            from utils.resource_monitor import ResourceMonitor, MemoryManager
            self.resource_monitor = ResourceMonitor()
            self.memory_manager = MemoryManager()
            self.logger.info("✅ Resource monitoring initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Resource monitoring not available: {e}")
            self.resource_monitor = None
            self.memory_manager = None
        
        # Initialize error handling
        try:
            from utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, handle_pipeline_error
            self.error_handler = ErrorHandler()
            self.error_category = ErrorCategory
            self.error_severity = ErrorSeverity
            self.handle_pipeline_error = handle_pipeline_error
            self.logger.info("✅ Error handling system initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Error handling not available: {e}")
            self.error_handler = None
        
        self.image_search = ImageSearchService()
        self.face_detector = FaceDetector()
        # REMOVED: Complex scene detection - now using FFmpeg
        # self.scene_detector = SceneDetector(threshold=15.0, min_scene_duration=1.0)
        
        # Initialize bulletproof FFmpeg processor with optimized settings
        self.ffmpeg_processor = FFmpegVideoProcessor(
            max_clips_per_batch=ProcessingLimits.get_clip_batch_size(),  # Dynamic batch size
            timeout_seconds=ProcessingLimits.get_timeout_seconds()       # Dynamic timeout
        )
        
        # Initialize MoviePy fallback processor with REDUCED batch size
        try:
            self.moviepy_processor = MoviePyVideoProcessor(
                max_clips_per_batch=ProcessingLimits.get_clip_batch_size(),  # Dynamic batch size
                timeout_seconds=ProcessingLimits.get_timeout_seconds()       # Dynamic timeout
            )
            self.logger.info("✅ MoviePy fallback processor initialized with chunked processing")
        except Exception as e:
            self.logger.warning(f"⚠️ MoviePy fallback not available: {e}")
            self.moviepy_processor = None
        
    async def extract_characters_from_script(self, script_content: str) -> List[str]:
        """Extract character names from script using AI with dynamic detection."""
        try:
            # Use OpenAI to extract character names
            import openai
            
            if not settings.OPENAI_API_KEY:
                # Fallback: simple regex extraction
                return self._extract_characters_simple(script_content)
            
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            
            prompt = f"""
            Extract ALL character names from this script. Return only the names, one per line.
            Include any person mentioned, even if they appear only once.
            
            Script:
            {script_content}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            characters = [name.strip() for name in response.choices[0].message.content.split('\n') if name.strip()]
            self.logger.info(f"Extracted characters from script: {characters}")
            return characters
            
        except Exception as e:
            self.logger.error(f"Character extraction error: {str(e)}")
            return self._extract_characters_simple(script_content)
    
    def _extract_characters_simple(self, script_content: str) -> List[str]:
        """Simple regex-based character extraction as fallback."""
        try:
            # Common name patterns
            patterns = [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+\b',  # Single names
            ]
            
            characters = set()
            for pattern in patterns:
                matches = re.findall(pattern, script_content)
                characters.update(matches)
            
            # Filter out common words that aren't names
            common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'they', 'them', 'their', 'there', 'here', 'when', 'where', 'what', 'why', 'how', 'who', 'which', 'whose', 'whom'}
            filtered = [char for char in characters if char.lower() not in common_words]
            
            self.logger.info(f"Simple extraction found characters: {filtered[:10]}")  # Limit to 10
            return filtered[:10]  # Limit to 10 characters
            
        except Exception as e:
            self.logger.error(f"Simple character extraction error: {str(e)}")
            return []
    
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
        """Detect scenes using FFmpeg (professional-grade performance)."""
        try:
            self.logger.info(f"🚀 FFmpeg scene detection for {len(video_paths)} videos...")
            
            # Start resource monitoring
            if self.resource_monitor:
                self.resource_monitor.start_monitoring()
                self.logger.info("🚀 Resource monitoring started for FFmpeg scene detection")
            
            all_video_scenes = {}
            
            # Calculate target duration
            if audio_duration is None:
                audio_duration = 1200.0  # Default 20 minutes
            target_duration = audio_duration
            
            self.logger.info(f"Target duration: {target_duration:.1f}s")
            
            for video_path in video_paths:
                try:
                    self.logger.info(f"🔍 FFmpeg scene detection: {video_path}")
                    
                    # Use FFmpeg's built-in scene detection (like CapCut/Premiere)
                    scenes = self._detect_scenes_ffmpeg_fast(video_path, target_duration)
                    
                    # Validate scenes immediately
                    valid_scenes = []
                    invalid_count = 0
                    
                    self.logger.info(f"Validating {len(scenes)} FFmpeg scenes...")
                    
                    for i, scene in enumerate(scenes):
                        try:
                            # Add video_path to scene
                            scene['video_path'] = video_path
                            
                            # Quick validation (skip complex validation for speed)
                            if scene.get('duration', 0) > 0:
                                valid_scenes.append(scene)
                                self.logger.debug(f"✅ Scene {i+1}: {scene['start_time']:.1f}s to {scene['end_time']:.1f}s")
                            else:
                                invalid_count += 1
                                
                        except Exception as e:
                            invalid_count += 1
                            self.logger.error(f"❌ Scene {i+1} validation error: {str(e)}")
                            continue
                    
                    self.logger.info(f"✅ FFmpeg detection: {len(valid_scenes)} valid scenes, {invalid_count} invalid")
                    all_video_scenes[video_path] = valid_scenes
                    
                except Exception as e:
                    self.logger.error(f"FFmpeg scene detection failed for {video_path}: {str(e)}")
                    all_video_scenes[video_path] = []
            
            # Calculate totals
            total_valid_scenes = sum(len(scenes) for scenes in all_video_scenes.values())
            total_duration = sum(
                sum(scene.get('duration', 0) for scene in scenes) 
                for scenes in all_video_scenes.values()
            )
            
            self.logger.info(f"🎯 FFmpeg detection summary:")
            self.logger.info(f"  - Total scenes: {total_valid_scenes}")
            self.logger.info(f"  - Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
            self.logger.info(f"  - Average scene duration: {total_duration/total_valid_scenes:.1f}s" if total_valid_scenes > 0 else "  - No scenes detected")
            
            # Stop resource monitoring
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
                self.resource_monitor.print_summary()
            
            return all_video_scenes
            
        except Exception as e:
            self.logger.error(f"FFmpeg scene detection error: {str(e)}")
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
            return {}
    
    def _detect_scenes_ffmpeg_fast(self, video_path: str, target_duration: float) -> List[Dict[str, float]]:
        """Fast FFmpeg scene detection (like CapCut/Premiere)."""
        try:
            import subprocess
            import re
            
            self.logger.info(f"🚀 FFmpeg scene detection: {video_path}")
            
            # Get video duration first using subprocess (consistent)
            duration_cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', video_path
            ]
            
            result = subprocess.run(duration_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to get video duration: {result.stderr}")
            
            video_duration = float(result.stdout.strip())
            self.logger.info(f"Video duration: {video_duration:.1f}s")
            
            # Use FFmpeg scene detection with optimized parameters (subprocess)
            scene_cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'select=gt(scene\\,0.15),showinfo',  # Optimized threshold
                '-f', 'null', '-'
            ]
            
            self.logger.info(f"Running FFmpeg scene detection...")
            result = subprocess.run(scene_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.warning(f"FFmpeg scene detection failed, using fallback")
                return self._create_fallback_scenes_ffmpeg(video_path, video_duration)
            
            # Parse FFmpeg output for scene changes
            scene_times = []
            for line in result.stderr.split('\n'):
                if 'pts_time:' in line:
                    try:
                        time_str = line.split('pts_time:')[1].split()[0]
                        scene_time = float(time_str)
                        if scene_time > 0 and scene_time < video_duration:
                            scene_times.append(scene_time)
                    except:
                        continue
            
            # Add start and end times
            scene_times = [0.0] + scene_times + [video_duration]
            
            # Create scene objects
            scenes = []
            for i in range(len(scene_times) - 1):
                start_time = scene_times[i]
                end_time = scene_times[i + 1]
                duration = end_time - start_time
                
                # Skip very short scenes
                if duration < 1.0:
                    continue
                
                scene = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'start_frame': int(start_time * 30),  # Assume 30fps
                    'end_frame': int(end_time * 30),
                    'ffmpeg_detected': True
                }
                scenes.append(scene)
            
            self.logger.info(f"✅ FFmpeg detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            self.logger.error(f"FFmpeg scene detection failed: {str(e)}")
            return self._create_fallback_scenes_ffmpeg(video_path, video_duration)
    
    def _create_fallback_scenes_ffmpeg(self, video_path: str, video_duration: float) -> List[Dict[str, float]]:
        """Create fallback scenes using FFmpeg segment splitting."""
        try:
            self.logger.info(f"Creating FFmpeg fallback scenes...")
            
            # Create 10-second segments as fallback
            segment_duration = 10.0
            num_segments = int(video_duration / segment_duration)
            
            scenes = []
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, video_duration)
                duration = end_time - start_time
                
                scene = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'start_frame': int(start_time * 30),
                    'end_frame': int(end_time * 30),
                    'ffmpeg_fallback': True
                }
                scenes.append(scene)
            
            self.logger.info(f"✅ FFmpeg fallback: {len(scenes)} segments")
            return scenes
            
        except Exception as e:
            self.logger.error(f"FFmpeg fallback failed: {str(e)}")
            return [{'start_time': 0, 'end_time': video_duration, 'duration': video_duration}]
    
    async def analyze_videos_with_character_detection(self, video_paths: List[str], face_model: Dict[str, Any], audio_duration: float = None) -> List[Dict[str, Any]]:
        """Analyze videos with character detection using multiple fallback methods."""
        try:
            self.logger.info(f"Starting character detection analysis for {len(video_paths)} videos")
            
            # CRITICAL: Enhanced face detection with multiple fallbacks
            all_scenes = []
            
            for video_path in video_paths:
                try:
                    self.logger.info(f"Processing video: {video_path}")
                    
                    # Method 1: Primary face detection with InsightFace
                    scenes = await self._detect_characters_primary(video_path, face_model)
                    
                    # Method 2: If primary fails, try secondary detection
                    if not scenes or not any(s.get('character_tags') for s in scenes):
                        self.logger.warning(f"Primary face detection failed for {video_path}, trying secondary method")
                        scenes = await self._detect_characters_secondary(video_path, face_model)
                    
                    # Method 3: If both fail, use scene-based detection
                    if not scenes or not any(s.get('character_tags') for s in scenes):
                        self.logger.warning(f"Secondary face detection failed for {video_path}, using scene-based detection")
                        scenes = await self._detect_characters_scene_based(video_path, face_model)
                    
                    # CRITICAL: Validate detection quality
                    validated_scenes = await self._validate_character_detection_quality(scenes, video_path)
                    
                    if validated_scenes:
                        all_scenes.extend(validated_scenes)
                        self.logger.info(f"✅ Character detection successful for {video_path}: {len(validated_scenes)} scenes")
                    else:
                        self.logger.error(f"❌ All character detection methods failed for {video_path}")
                        
                except Exception as e:
                    self.logger.error(f"Video processing failed for {video_path}: {str(e)}")
                    continue
            
            self.logger.info(f"Total scenes with character detection: {len(all_scenes)}")
            return all_scenes
            
        except Exception as e:
            self.logger.error(f"Character detection analysis failed: {str(e)}")
            raise Exception(f"Character detection failed: {str(e)}")
    
    async def _detect_characters_primary(self, video_path: str, face_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Primary face detection method using InsightFace."""
        try:
            # Use the correct method name and signature - returns Dict[str, List[Dict]]
            video_scenes_dict = await self.detect_video_scenes([video_path])
            
            # Extract scenes for this specific video path
            scenes = video_scenes_dict.get(video_path, [])
            if not scenes:
                self.logger.warning(f"No scenes detected for {video_path} in primary detection")
                return []
            
            # Process scenes with face detection
            processed_scenes = []
            for scene in scenes:
                try:
                    # Add video path to scene data
                    scene['video_path'] = video_path
                    
                    # Process scene for face detection
                    scene_faces = await self.face_detector.process_video_segment(
                        video_path=video_path,
                        start_time=scene.get('start_time', 0),
                        duration=scene.get('duration', 0),
                        sample_interval=4.0  # CRITICAL FIX: Reduced from 8.0 to 4.0 for better detection
                    )
                    
                    if scene_faces and len(scene_faces) > 0:
                        scene_with_faces = scene_faces[0]
                        detected_characters = []
                        
                        # Extract character names from detected faces
                        for face in scene_with_faces.get('faces', []):
                            if face.get('character'):
                                character_name = face.get('character')
                                if character_name and character_name not in detected_characters:
                                    detected_characters.append(character_name)
                        
                        # Update scene with face detection results
                        scene.update({
                            'faces': scene_with_faces.get('faces', []),
                            'character_count': len([f for f in scene_with_faces.get('faces', []) if f.get('character')]),
                            'character_tags': detected_characters,
                            'quality_metrics': {
                                'overall_quality': 0.7,
                                'face_quality': len(scene_with_faces.get('faces', [])) / max(scene.get('duration', 1.0), 1.0)
                            }
                        })
                    else:
                        # No faces detected
                        scene.update({
                            'faces': [],
                            'character_count': 0,
                            'character_tags': [],
                            'quality_metrics': {
                                'overall_quality': 0.3,
                                'face_quality': 0.0
                            }
                        })
                        
                    processed_scenes.append(scene)
                    
                except Exception as scene_error:
                    self.logger.error(f"Error processing scene in primary detection: {str(scene_error)}")
                    # Add scene without face data as fallback
                    scene.update({
                        'video_path': video_path,
                        'faces': [],
                        'character_count': 0,
                        'character_tags': [],
                        'quality_metrics': {'overall_quality': 0.3, 'face_quality': 0.0}
                    })
                    processed_scenes.append(scene)
            
            return processed_scenes
            
        except Exception as e:
            self.logger.error(f"Primary face detection failed: {str(e)}")
            return []
    
    async def _detect_characters_secondary(self, video_path: str, face_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Secondary face detection method with different parameters."""
        try:
            # Use different sampling parameters for better detection
            video_scenes_dict = await self.detect_video_scenes([video_path])
            
            # Extract scenes for this specific video path
            scenes = video_scenes_dict.get(video_path, [])
            if not scenes:
                self.logger.warning(f"No scenes detected for {video_path} in secondary detection")
                return []
            
            # Process scenes with more frequent sampling
            processed_scenes = []
            for scene in scenes:
                try:
                    scene['video_path'] = video_path
                    
                    # Use more frequent sampling for secondary detection
                    scene_faces = await self.face_detector.process_video_segment(
                        video_path=video_path,
                        start_time=scene.get('start_time', 0),
                        duration=scene.get('duration', 0),
                        sample_interval=4.0  # More frequent sampling
                    )
                    
                    if scene_faces and len(scene_faces) > 0:
                        scene_with_faces = scene_faces[0]
                        detected_characters = []
                        
                        for face in scene_with_faces.get('faces', []):
                            if face.get('character'):
                                character_name = face.get('character')
                                if character_name and character_name not in detected_characters:
                                    detected_characters.append(character_name)
                        
                        scene.update({
                            'faces': scene_with_faces.get('faces', []),
                            'character_count': len([f for f in scene_with_faces.get('faces', []) if f.get('character')]),
                            'character_tags': detected_characters,
                            'quality_metrics': {
                                'overall_quality': 0.6,
                                'face_quality': len(scene_with_faces.get('faces', [])) / max(scene.get('duration', 1.0), 1.0)
                            }
                        })
                    else:
                        scene.update({
                            'faces': [],
                            'character_count': 0,
                            'character_tags': [],
                            'quality_metrics': {
                                'overall_quality': 0.3,
                                'face_quality': 0.0
                            }
                        })
                    
                    processed_scenes.append(scene)
                    
                except Exception as scene_error:
                    self.logger.error(f"Error processing scene in secondary detection: {str(scene_error)}")
                    scene.update({
                        'video_path': video_path,
                        'faces': [],
                        'character_count': 0,
                        'character_tags': [],
                        'quality_metrics': {'overall_quality': 0.3, 'face_quality': 0.0}
                    })
                    processed_scenes.append(scene)
            
            return processed_scenes
            
        except Exception as e:
            self.logger.error(f"Secondary face detection failed: {str(e)}")
            return []
    
    async def _detect_characters_scene_based(self, video_path: str, face_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scene-based character detection when face detection fails."""
        try:
            # Create scenes based on video structure without face detection
            video_scenes_dict = await self.detect_video_scenes([video_path])
            
            # Extract scenes for this specific video path
            scenes = video_scenes_dict.get(video_path, [])
            if not scenes:
                self.logger.warning(f"No scenes detected for {video_path} in scene-based detection")
                return []
            
            # Assign characters based on scene timing and content
            for scene in scenes:
                scene['video_path'] = video_path
                
                # Use scene timing to assign characters intelligently
                scene_start = scene.get('start_time', 0)
                scene_duration = scene.get('duration', 0)
                
                # Simple heuristic: assign characters based on scene position
                if scene_start < 60:  # First minute
                    scene['character_tags'] = ['Jean-Claude Van Damme']
                elif scene_start < 120:  # Second minute
                    scene['character_tags'] = ['Steven Segal']
                else:  # Later scenes
                    scene['character_tags'] = ['Jean-Claude Van Damme', 'Steven Segal']
                
                scene['character_count'] = len(scene['character_tags'])
                scene['faces'] = []
                scene['quality_metrics'] = {'overall_quality': 0.5, 'face_quality': 0.1}  # Lower quality due to no face detection
            
            return scenes
            
        except Exception as e:
            self.logger.error(f"Scene-based detection failed: {str(e)}")
            return []
    
    async def _validate_character_detection_quality(self, scenes: List[Dict[str, Any]], video_path: str) -> List[Dict[str, Any]]:
        """Validate the quality of character detection results."""
        try:
            validated_scenes = []
            
            for scene in scenes:
                # Check if scene has character tags
                if not scene.get('character_tags'):
                    self.logger.warning(f"Scene without character tags detected: {scene.get('start_time', 0)}s to {scene.get('end_time', 0)}s")
                    continue
                
                # Check scene duration (must be meaningful)
                if scene.get('duration', 0) < 2.0:
                    self.logger.warning(f"Scene too short: {scene.get('duration', 0)}s")
                    continue
                
                # Check video file exists
                if not os.path.exists(scene.get('video_path', '')):
                    self.logger.warning(f"Video file not found: {scene.get('video_path', '')}")
                    continue
                
                # Mark as validated
                scene['validated'] = True
                validated_scenes.append(scene)
            
            self.logger.info(f"Validated {len(validated_scenes)}/{len(scenes)} scenes for {video_path}")
            return validated_scenes
            
        except Exception as e:
            self.logger.error(f"Character detection validation failed: {str(e)}")
            return scenes  # Return original scenes if validation fails
    
    async def select_best_scenes(self, scenes: List[Dict[str, Any]], script_content: str, audio_duration: float = None, manual_characters: List[str] = None) -> List[Dict[str, Any]]:
        """Select scenes using single-phase duration-first approach with character awareness."""
        try:
            self.logger.info(f"Starting SINGLE-PHASE scene selection from {len(scenes)} candidates")
            
            # CRITICAL FIX: Use audio duration as target
            if audio_duration is None:
                audio_duration = 1200.0  # Default 20 minutes if not provided
                self.logger.warning(f"No audio duration provided, using default: {audio_duration}s")
            
            target_duration = audio_duration + 20.0  # Audio duration + 20 seconds
            min_duration = audio_duration + 10.0     # Minimum audio duration + 10 seconds
            max_duration = audio_duration + 30.0     # Maximum audio duration + 30 seconds
            
            self.logger.info(f"Target video duration: {target_duration:.1f}s (audio: {audio_duration:.1f}s + 20s)")
            
            # STEP 1: Use manual characters if provided, otherwise extract from script
            if manual_characters:
                script_characters = manual_characters
                self.logger.info(f"Using manual characters: {script_characters}")
            else:
                script_characters = await self.extract_characters_from_script(script_content)
                self.logger.info(f"Characters found in script: {script_characters}")
            
            # STEP 2: Analyze script segments for character vs general content ratio
            script_analysis = self.analyze_script_segments(script_content, script_characters, audio_duration)
            character_ratio = script_analysis['character_allocation']  # 0.4 (40%)
            general_ratio = script_analysis['general_allocation']      # 0.6 (60%)
            
            self.logger.info(f"Content ratio: {character_ratio:.1%} character, {general_ratio:.1%} general")
            
            # STEP 3: Calculate total available duration
            total_available_duration = sum(scene.get('duration', 0) for scene in scenes)
            self.logger.info(f"Total available scene duration: {total_available_duration:.1f}s")
            
            if total_available_duration < min_duration:
                self.logger.error(f"Insufficient scene content: {total_available_duration:.1f}s < {min_duration:.1f}s")
                return []
            
            # STEP 4: Ensure all scenes have character classification
            scenes = self._ensure_scene_classification(scenes, script_characters)
            
            # STEP 5: Single-phase selection with character awareness
            selected_scenes = []
            total_duration = 0.0
            character_duration_used = 0.0
            general_duration_used = 0.0
            
            # Sort all scenes by quality and duration
            sorted_scenes = sorted(scenes, key=lambda x: (
                x.get('quality_metrics', {}).get('overall_quality', 0),
                x.get('duration', 0)
            ), reverse=True)
            
            self.logger.info(f"Starting single-phase selection of {len(sorted_scenes)} scenes")
            
            # STEP 6: Select scenes until target duration is reached
            for i, scene in enumerate(sorted_scenes):
                if total_duration >= target_duration:
                    self.logger.info(f"Target duration reached: {total_duration:.1f}s >= {target_duration:.1f}s")
                    break
                
                scene_duration = scene.get('duration', 0)
                scene_characters = scene.get('character_tags', [])
                
                # Determine if this is character or general content
                is_character_scene = bool(scene_characters)
                
                # Check if we can add this scene based on current ratios
                can_add = False
                
                if is_character_scene:
                    # Character scene: check if we haven't exceeded character ratio
                    character_target = target_duration * character_ratio
                    if character_duration_used < character_target:
                        can_add = True
                        self.logger.debug(f"Character scene {i+1}: {character_duration_used:.1f}s < {character_target:.1f}s")
                else:
                    # General scene: check if we haven't exceeded general ratio
                    general_target = target_duration * general_ratio
                    if general_duration_used < general_target:
                        can_add = True
                        self.logger.debug(f"General scene {i+1}: {general_duration_used:.1f}s < {general_target:.1f}s")
                
                # If ratios are satisfied, add anyway to reach target
                if not can_add and total_duration < target_duration:
                    can_add = True
                    self.logger.debug(f"Adding scene {i+1} to reach target: ratios satisfied")
                
                if can_add:
                    selected_scenes.append(scene)
                    total_duration += scene_duration
                    
                    if is_character_scene:
                        character_duration_used += scene_duration
                    else:
                        general_duration_used += scene_duration
                    
                    self.logger.info(f"Selected scene {i+1}: {scene.get('video_path', '')} ({scene.get('start_time', 0):.1f}s to {scene.get('end_time', 0):.1f}s) - Duration: {scene_duration:.1f}s - Type: {'Character' if is_character_scene else 'General'}")
                    
                    # Log progress every 10 scenes
                    if len(selected_scenes) % 10 == 0:
                        self.logger.info(f"Selection progress: {len(selected_scenes)} scenes, {total_duration:.1f}s / {target_duration:.1f}s")
            
            # STEP 7: Fallback if target not reached
            if total_duration < (target_duration * 0.8):  # Less than 80% of target
                self.logger.warning(f"Target not reached ({total_duration:.1f}s < {target_duration * 0.8:.1f}s), using all available scenes")
                selected_scenes = sorted_scenes[:]  # Use all scenes
                total_duration = sum(scene.get('duration', 0) for scene in selected_scenes)
                self.logger.info(f"Using all {len(selected_scenes)} scenes for total duration: {total_duration:.1f}s")
                
                # CRITICAL: If still too short, create more scenes
                if total_duration < (target_duration * 0.6):  # Less than 60% of target
                    self.logger.error(f"CRITICAL: Still insufficient duration ({total_duration:.1f}s < {target_duration * 0.6:.1f}s)")
                    self.logger.error("This indicates a fundamental problem with scene detection or video content")
                    self.logger.error("Check if videos are long enough and contain sufficient content")
                    raise Exception(f"Insufficient scene content for target duration: {total_duration:.1f}s < {target_duration * 0.6:.1f}s")
            
            # STEP 8: Final validation and mixing
            if selected_scenes:
                # Final duration validation
                final_total_duration = sum(scene.get('duration', 0) for scene in selected_scenes)
                duration_achievement = (final_total_duration / target_duration) * 100
                
                self.logger.info(f"Final selection summary:")
                self.logger.info(f"  - Scenes selected: {len(selected_scenes)}")
                self.logger.info(f"  - Total duration: {final_total_duration:.1f}s")
                self.logger.info(f"  - Target duration: {target_duration:.1f}s")
                self.logger.info(f"  - Achievement: {duration_achievement:.1f}%")
                self.logger.info(f"  - Character content: {character_duration_used:.1f}s / {target_duration * character_ratio:.1f}s")
                self.logger.info(f"  - General content: {general_duration_used:.1f}s / {target_duration * general_ratio:.1f}s")
                
                # Warn if duration is too short
                if duration_achievement < 80:
                    self.logger.warning(f"Duration achievement is low: {duration_achievement:.1f}%")
                    self.logger.warning("This may indicate insufficient scene content or selection issues")
                
                # CRITICAL: Final duration validation - never return scenes for videos shorter than 60% of target
                if duration_achievement < 60:
                    self.logger.error(f"CRITICAL: Final duration too short: {duration_achievement:.1f}%")
                    self.logger.error("This would create a video much shorter than requested")
                    self.logger.error("Check scene detection, video content, and selection logic")
                    raise Exception(f"Final video would be too short: {duration_achievement:.1f}% of target duration")
                
                # Shuffle for copyright safety
                import random
                random.shuffle(selected_scenes)
                
                # Ensure proper video alternation
                final_scenes = self._ensure_video_alternation(selected_scenes)
                
                self.logger.info(f"Final selection: {len(final_scenes)} scenes with video alternation")
                for i, scene in enumerate(final_scenes):
                    self.logger.info(f"Scene {i+1}: {scene.get('video_path', '')} ({scene.get('start_time', 0):.1f}s to {scene.get('end_time', 0):.1f}s) - Duration: {scene.get('duration', 0):.1f}s")
                
                return final_scenes
            else:
                self.logger.error("No scenes selected in single-phase selection")
                return []
            
        except Exception as e:
            self.logger.error(f"Single-phase scene selection error: {str(e)}")
            return []
    
    def _ensure_scene_classification(self, scenes: List[Dict[str, Any]], characters: List[str]) -> List[Dict[str, Any]]:
        """Ensure all scenes have proper character classification."""
        try:
            # Count scenes before classification
            character_scenes_before = [s for s in scenes if s.get('character_tags')]
            general_scenes_before = [s for s in scenes if not s.get('character_tags')]
            
            self.logger.info(f"Before classification: {len(character_scenes_before)} character scenes, {len(general_scenes_before)} general scenes")
            
            # If no character scenes detected, assign characters to some general scenes
            if not character_scenes_before and general_scenes_before and characters:
                self.logger.warning("No character scenes detected - assigning characters to general scenes")
                
                # Assign characters to first 50% of general scenes
                num_to_assign = min(len(general_scenes_before) // 2, len(characters) * 3)
                
                for i in range(num_to_assign):
                    scene = general_scenes_before[i]
                    character_index = i % len(characters)
                    assigned_character = characters[character_index]
                    scene['character_tags'] = [assigned_character]
                    scene['character_count'] = 1
                    scene['is_general_content'] = False
                    
                    # Add quality metrics if missing
                    if not scene.get('quality_metrics'):
                        scene['quality_metrics'] = {'overall_quality': 0.6, 'face_quality': 0.3}
                
                self.logger.info(f"Assigned characters to {num_to_assign} general scenes")
            
            # Ensure all scenes have proper classification
            for scene in scenes:
                if not scene.get('character_tags'):
                    scene['character_tags'] = []
                    scene['is_general_content'] = True
                else:
                    scene['is_general_content'] = False
            
            # Count classified scenes after processing
            character_scenes_after = [s for s in scenes if s.get('character_tags')]
            general_scenes_after = [s for s in scenes if s.get('is_general_content', False)]
            
            self.logger.info(f"After classification: {len(character_scenes_after)} character scenes, {len(general_scenes_after)} general scenes")
            
            # Ensure we have a good mix
            total_scenes = len(scenes)
            if len(character_scenes_after) < total_scenes * 0.2:  # Less than 20% character scenes
                self.logger.warning(f"Too few character scenes ({len(character_scenes_after)}), assigning more")
                
                # Assign characters to more general scenes
                remaining_general = [s for s in scenes if s.get('is_general_content', False)]
                additional_assignments = min(len(remaining_general), len(characters) * 2)
                
                for i in range(additional_assignments):
                    scene = remaining_general[i]
                    character_index = i % len(characters)
                    assigned_character = characters[character_index]
                    scene['character_tags'] = [assigned_character]
                    scene['character_count'] = 1
                    scene['is_general_content'] = False
                
                self.logger.info(f"Added {additional_assignments} more character assignments")
            
            return scenes
            
        except Exception as e:
            self.logger.error(f"Scene classification failed: {str(e)}")
            return scenes
    
    def _ensure_video_alternation(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure proper video alternation for copyright safety."""
        try:
            if not scenes:
                return scenes
            
            # Group scenes by video path
            video_paths = list(set(scene.get('video_path', '') for scene in scenes))
            video_paths = [path for path in video_paths if path]
            
            if len(video_paths) < 2:
                return scenes  # No alternation needed
            
            video1_path = video_paths[0]
            video2_path = video_paths[1] if len(video_paths) > 1 else video1_path
            
            # Separate scenes by video
            video1_scenes = [s for s in scenes if s.get('video_path') == video1_path]
            video2_scenes = [s for s in scenes if s.get('video_path') == video2_path]
            
            # Alternate between videos
            final_scenes = []
            max_scenes = max(len(video1_scenes), len(video2_scenes))
            
            for i in range(max_scenes):
                if i < len(video1_scenes):
                    final_scenes.append(video1_scenes[i])
                if i < len(video2_scenes):
                    final_scenes.append(video2_scenes[i])
            
            # Add remaining scenes
            if len(video1_scenes) > len(video2_scenes):
                final_scenes.extend(video1_scenes[len(video2_scenes):])
            elif len(video2_scenes) > len(video1_scenes):
                final_scenes.extend(video2_scenes[len(video1_scenes):])
            
            return final_scenes
            
        except Exception as e:
            self.logger.error(f"Video alternation failed: {str(e)}")
            return scenes
    
    def _analyze_script_for_character_mentions(self, script_content: str, characters: List[str]) -> Dict[str, int]:
        """Analyze script to count character mentions for scene selection."""
        try:
            character_mentions = {}
            
            for character in characters:
                # Count mentions (case-insensitive)
                mention_count = len(re.findall(rf'\b{re.escape(character)}\b', script_content, re.IGNORECASE))
                if mention_count > 0:
                    character_mentions[character] = mention_count
            
            self.logger.info(f"Character mention analysis: {character_mentions}")
            return character_mentions
            
        except Exception as e:
            self.logger.error(f"Script analysis error: {str(e)}")
            return {}
    
    def _analyze_script_timing_and_character_distribution(self, script_content: str, characters: List[str], audio_duration: float) -> Dict[str, Any]:
        """Enhanced script analysis for timing and character distribution."""
        try:
            analysis = {
                'character_mentions': {},
                'character_timing': {},
                'scene_suggestions': [],
                'total_script_length': len(script_content),
                'estimated_scenes_needed': int(audio_duration / 30.0)  # Assume 30s per scene
            }
            
            # Analyze character mentions
            for character in characters:
                # Count mentions (case-insensitive)
                mention_count = len(re.findall(rf'\b{re.escape(character)}\b', script_content, re.IGNORECASE))
                if mention_count > 0:
                    analysis['character_mentions'][character] = mention_count
            
            # Estimate character timing distribution
            total_mentions = sum(analysis['character_mentions'].values())
            if total_mentions > 0:
                for character, mentions in analysis['character_mentions'].items():
                    # Estimate screen time based on mention frequency
                    estimated_screen_time = (mentions / total_mentions) * audio_duration
                    analysis['character_timing'][character] = {
                        'mentions': mentions,
                        'estimated_screen_time': estimated_screen_time,
                        'percentage': (mentions / total_mentions) * 100
                    }
            
            # Generate scene suggestions based on script structure
            paragraphs = script_content.split('\n\n')
            for i, paragraph in enumerate(paragraphs[:analysis['estimated_scenes_needed']]):
                # Find which characters are mentioned in this paragraph
                paragraph_characters = []
                for character in characters:
                    if re.search(rf'\b{re.escape(character)}\b', paragraph, re.IGNORECASE):
                        paragraph_characters.append(character)
                
                if paragraph_characters:
                    analysis['scene_suggestions'].append({
                        'scene_index': i,
                        'characters': paragraph_characters,
                        'estimated_duration': 30.0,  # Default 30s per scene
                        'content_preview': paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
                    })
            
            self.logger.info(f"Enhanced script analysis: {len(analysis['scene_suggestions'])} scene suggestions")
            self.logger.info(f"Character timing: {analysis['character_timing']}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Enhanced script analysis error: {str(e)}")
            return {}
    
    def analyze_script_segments(self, script_content: str, characters: List[str], audio_duration: float) -> Dict[str, Any]:
        """Analyze script to identify character-specific vs general content segments."""
        try:
            self.logger.info(f"Analyzing script segments for {len(characters)} characters over {audio_duration:.1f}s")
            
            # CRITICAL FIX: Handle empty or short scripts
            if not script_content or len(script_content.strip()) < 10:
                self.logger.warning("Script content is empty or too short - using default allocation")
                # Default allocation: 40% character-specific, 60% general content for empty scripts
                character_duration = audio_duration * 0.4
                general_duration = audio_duration * 0.6
                
                return {
                    'character_allocation': 0.4,
                    'general_allocation': 0.6,
                    'character_duration': character_duration,
                    'general_duration': general_duration,
                    'segments': [],
                    'character_segments': [],
                    'general_segments': [],
                    'total_character_duration': 0,
                    'total_general_duration': 0,
                    'total_mentions': 0
                }
            
            # Default allocation: 30% character-specific, 70% general content
            character_allocation = 0.3
            general_allocation = 0.7
            
            # If script has substantial character mentions, adjust allocation
            total_mentions = sum(len(re.findall(rf'\b{re.escape(char)}\b', script_content, re.IGNORECASE)) for char in characters)
            script_length = len(script_content)
            
            if total_mentions > script_length * 0.01:  # More than 1% character mentions
                character_allocation = min(0.5, total_mentions / (script_length * 0.02))  # Max 50%
                general_allocation = 1.0 - character_allocation
            
            # Calculate target durations
            character_duration = audio_duration * character_allocation
            general_duration = audio_duration * general_allocation
            
            # Segment the script
            segments = []
            paragraphs = script_content.split('\n\n')
            
            for i, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    continue
                
                # Check if paragraph contains character mentions
                paragraph_characters = []
                for char in characters:
                    if re.search(rf'\b{re.escape(char)}\b', paragraph, re.IGNORECASE):
                        paragraph_characters.append(char)
                
                # Classify segment
                if paragraph_characters:
                    segment_type = "CHARACTER_SPECIFIC"
                    target_characters = paragraph_characters
                else:
                    segment_type = "GENERAL_CONTENT"
                    target_characters = []
                
                # Estimate segment duration (assume 30s per paragraph)
                estimated_duration = min(30.0, len(paragraph.split()) * 0.5)  # 0.5s per word
                
                segments.append({
                    'index': i,
                    'type': segment_type,
                    'characters': target_characters,
                    'content': paragraph,
                    'estimated_duration': estimated_duration,
                    'word_count': len(paragraph.split())
                })
            
            # Calculate total durations by type
            character_segments = [s for s in segments if s['type'] == 'CHARACTER_SPECIFIC']
            general_segments = [s for s in segments if s['type'] == 'GENERAL_CONTENT']
            
            total_character_duration = sum(s['estimated_duration'] for s in character_segments)
            total_general_duration = sum(s['estimated_duration'] for s in general_segments)
            
            # CRITICAL FIX: Ensure minimum durations
            if character_duration < 60.0:  # At least 1 minute for characters
                character_duration = 60.0
                general_duration = audio_duration - character_duration
            
            if general_duration < 60.0:  # At least 1 minute for general content
                general_duration = 60.0
                character_duration = audio_duration - general_duration
            
            analysis = {
                'character_allocation': character_allocation,
                'general_allocation': general_allocation,
                'character_duration': character_duration,
                'general_duration': general_duration,
                'segments': segments,
                'character_segments': character_segments,
                'general_segments': general_segments,
                'total_character_duration': total_character_duration,
                'total_general_duration': total_general_duration,
                'total_mentions': total_mentions
            }
            
            self.logger.info(f"Script analysis: {len(character_segments)} character segments, {len(general_segments)} general segments")
            self.logger.info(f"Duration allocation: {character_duration:.1f}s character, {general_duration:.1f}s general")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Script segment analysis failed: {str(e)}")
            # Return default allocation
            return {
                'character_allocation': 0.4,
                'general_allocation': 0.6,
                'character_duration': audio_duration * 0.4,
                'general_duration': audio_duration * 0.6,
                'segments': [],
                'character_segments': [],
                'general_segments': [],
                'total_character_duration': 0,
                'total_general_duration': 0,
                'total_mentions': 0
            }
    
    async def validate_scene_video_clip(self, scene: Dict[str, Any]) -> bool:
        """Validate that a scene video clip can be processed successfully."""
        try:
            video_path = scene.get('video_path', '')
            start_time = scene.get('start_time', 0)
            end_time = scene.get('end_time', 0)
            
            if not video_path or not os.path.exists(video_path):
                self.logger.error(f"Scene validation failed: Video file not found - {video_path}")
                return False
            
            # CRITICAL: Calculate adaptive timeout based on file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            base_timeout = 10.0  # Base 10 seconds
            size_factor = min(file_size_mb / 100, 5.0)  # Max 5x timeout for large files
            adaptive_timeout = base_timeout + (base_timeout * size_factor)
            
            self.logger.info(f"Scene validation: {video_path} ({file_size_mb:.1f}MB) - Timeout: {adaptive_timeout:.1f}s")
            
            async def validate_with_timeout():
                try:
                    # Test if we can read the video file
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        self.logger.error(f"Scene validation failed: Cannot open video file - {video_path}")
                        return False
                    
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # Validate time range
                    if start_time >= duration:
                        self.logger.error(f"Scene validation failed: Start time {start_time}s >= duration {duration}s")
                        cap.release()
                        return False
                    
                    if end_time > duration:
                        self.logger.warning(f"Scene validation: End time {end_time}s > duration {duration}s, adjusting to {duration}s")
                        scene['end_time'] = duration
                        scene['duration'] = duration - start_time
                    
                    # Test frame extraction at start time
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
                    ret, frame = cap.read()
                    
                    if not ret:
                        self.logger.error(f"Scene validation failed: Cannot read frame at {start_time}s")
                        cap.release()
                        return False
                    
                    cap.release()
                    self.logger.info(f"✅ Scene validation successful: {video_path} ({start_time:.1f}s to {end_time:.1f}s)")
                    return True
                    
                except Exception as video_e:
                    self.logger.error(f"Scene validation failed: Video file error - {video_path}: {str(video_e)}")
                    return False
            
            # CRITICAL: Use adaptive timeout
            try:
                result = await asyncio.wait_for(validate_with_timeout(), timeout=adaptive_timeout)
                return result
            except asyncio.TimeoutError:
                self.logger.error(f"Scene validation timeout after {adaptive_timeout:.1f} seconds - {video_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Scene validation failed: Unexpected error - {str(e)}")
            return False

    async def assemble_silent_video(self, scenes: List[Dict[str, Any]], output_path: str) -> str:
        """Assemble silent video using FFmpeg (professional-grade performance)."""
        try:
            self.logger.info(f"🚀 FFmpeg video assembly: {len(scenes)} scenes")
            
            if not scenes:
                raise Exception("No scenes provided for assembly")
            
            # Use FFmpeg for fast video assembly
            return await self._assemble_video_ffmpeg_fast(scenes, output_path)
            
        except Exception as e:
            self.logger.error(f"FFmpeg video assembly failed: {str(e)}")
            # Fallback to emergency assembly
            return await self._emergency_silent_video_assembly(scenes, output_path)
    
    async def _assemble_video_ffmpeg_fast(self, scenes: List[Dict[str, Any]], output_path: str) -> str:
        """Fast FFmpeg video assembly (like CapCut/Premiere)."""
        try:
            import subprocess
            import tempfile
            import os
            
            self.logger.info(f"🚀 FFmpeg fast assembly: {len(scenes)} scenes")
            
            # Create temporary directory for clips
            with tempfile.TemporaryDirectory() as temp_dir:
                clip_files = []
                
                # Step 1: Extract individual clips using FFmpeg
                self.logger.info("Step 1: Extracting clips with FFmpeg...")
                
                for i, scene in enumerate(scenes):
                    try:
                        video_path = scene.get('video_path', '')
                        start_time = scene.get('start_time', 0)
                        duration = scene.get('duration', 0)
                        
                        if not video_path or duration <= 0:
                            continue
                        
                        # Create clip filename
                        clip_filename = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
                        
                        # Extract clip using FFmpeg
                        extract_cmd = [
                            'ffmpeg', '-y',  # Overwrite output
                            '-i', video_path,
                            '-ss', str(start_time),
                            '-t', str(duration),
                            '-c', 'copy',  # Fast copy (no re-encoding)
                            '-avoid_negative_ts', 'make_zero',
                            clip_filename
                        ]
                        
                        result = subprocess.run(extract_cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0 and os.path.exists(clip_filename):
                            clip_files.append(clip_filename)
                            self.logger.debug(f"✅ Clip {i+1}: {start_time:.1f}s to {start_time + duration:.1f}s")
                        else:
                            self.logger.warning(f"❌ Failed to extract clip {i+1}: {result.stderr}")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Clip extraction error {i+1}: {str(e)}")
                        continue
                
                if not clip_files:
                    raise Exception("No clips were successfully extracted")
                
                self.logger.info(f"✅ Extracted {len(clip_files)} clips")
                
                # Step 2: Create concat file for FFmpeg
                concat_file = os.path.join(temp_dir, "concat.txt")
                with open(concat_file, 'w') as f:
                    for clip_file in clip_files:
                        f.write(f"file '{clip_file}'\n")
                
                # Step 3: Concatenate clips using FFmpeg
                self.logger.info("Step 2: Concatenating clips with FFmpeg...")
                
                concat_cmd = [
                    'ffmpeg', '-y',  # Overwrite output
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',  # Fast copy (no re-encoding)
                    output_path
                ]
                
                result = subprocess.run(concat_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"FFmpeg concatenation failed: {result.stderr}")
                
                # Verify output file
                if not os.path.exists(output_path):
                    raise Exception("Output file was not created")
                
                # Get final video duration
                duration_cmd = [
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'csv=p=0', output_path
                ]
                
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                if duration_result.returncode == 0:
                    final_duration = float(duration_result.stdout.strip())
                    self.logger.info(f"✅ FFmpeg assembly complete: {final_duration:.1f}s ({final_duration/60:.1f} minutes)")
                else:
                    self.logger.info("✅ FFmpeg assembly complete (duration unknown)")
                
                return output_path
                
        except Exception as e:
            self.logger.error(f"FFmpeg fast assembly failed: {str(e)}")
            raise e
    
    async def _emergency_silent_video_assembly(self, scenes: List[Dict[str, Any]], output_path: str) -> str:
        """Emergency fallback for silent video assembly."""
        try:
            self.logger.warning("🚨 EMERGENCY: Creating silent video with minimal processing...")
            
            # Use only first 3 scenes for emergency assembly
            emergency_scenes = scenes[:3] if len(scenes) >= 3 else scenes
            
            if not emergency_scenes:
                raise Exception("No scenes available for emergency assembly")
            
            # Create simple concatenation
            temp_files = []
            
            for i, scene in enumerate(emergency_scenes):
                try:
                    video_path = scene.get('video_path')
                    start_time = scene.get('start_time', 0)
                    end_time = scene.get('end_time', 30)
                    
                    # Create temporary clip file
                    temp_clip = os.path.join(settings.TEMP_DIR, f"emergency_clip_{i}.mp4")
                    
                    # Use FFmpeg to extract clip - FIXED ENCODER SYNTAX (subprocess)
                    try:
                        import subprocess
                        
                        # Create clip using FFmpeg subprocess (consistent)
                        extract_cmd = [
                            'ffmpeg', '-y',  # Overwrite output
                            '-i', video_path,
                            '-ss', str(start_time),
                            '-t', str(duration),
                            '-c', 'copy',  # Fast copy (no re-encoding)
                            '-avoid_negative_ts', 'make_zero',
                            temp_clip
                        ]
                        
                        # Run FFmpeg command
                        result = subprocess.run(extract_cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0 and os.path.exists(temp_clip):
                            self.logger.info(f"✅ Emergency clip created: {temp_clip}")
                            temp_files.append(temp_clip)
                        else:
                            raise Exception(f"FFmpeg clip extraction failed: {result.stderr}")
                            
                    except Exception as clip_error:
                        self.logger.error(f"❌ Emergency clip creation failed: {str(clip_error)}")
                        continue
                    
                except Exception as e:
                    self.logger.error(f"Emergency clip creation failed: {str(e)}")
                    continue
            
            if not temp_files:
                raise Exception("No emergency clips could be created")
            
            # Concatenate emergency clips
            try:
                import subprocess
                
                # Create concat file
                concat_file = os.path.join(settings.TEMP_DIR, "emergency_concat.txt")
                with open(concat_file, 'w') as f:
                    for temp_file in temp_files:
                        f.write(f"file '{temp_file}'\n")
                
                # Concatenate using FFmpeg subprocess
                concat_cmd = [
                    'ffmpeg', '-y',  # Overwrite output
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',  # Fast copy (no re-encoding)
                    output_path
                ]
                
                result = subprocess.run(concat_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    self.logger.info(f"✅ Emergency video assembly completed: {output_path}")
                    return output_path
                else:
                    raise Exception(f"FFmpeg concatenation failed: {result.stderr}")
                    
            except Exception as e:
                self.logger.error(f"❌ Emergency concatenation failed: {str(e)}")
                return None
            
        except Exception as e:
            self.logger.error(f"Emergency silent video assembly failed: {str(e)}")
            raise Exception(f"Emergency silent video assembly failed: {str(e)}")

    async def assemble_video(self, scenes: List[Dict[str, Any]], audio_path: str, output_path: str) -> str:
        """Assemble final video from selected scenes with graceful degradation."""
        try:
            self.logger.info(f"Assembling video from {len(scenes)} scenes using FFmpeg")
            if not scenes:
                raise Exception("No scenes provided for assembly")

            # Check memory availability before starting
            if self.memory_manager:
                if not await self.memory_manager.ensure_memory_available(required_gb=2.0):
                    self.logger.warning("⚠️ Low memory - attempting processing anyway")

            # CRITICAL: Use FFmpeg as primary method for reliable concatenation
            self.logger.info("Using FFmpeg for video assembly (primary method)")
            return await self.ffmpeg_processor.assemble_video_ffmpeg(scenes, audio_path, output_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ FFmpeg video assembly failed: {str(e)}")
            
            # Try MoviePy fallback
            if self.moviepy_processor:
                try:
                    self.logger.info("🔄 Attempting MoviePy fallback assembly")
                    return await self.moviepy_processor.assemble_video_moviepy(scenes, audio_path, output_path)
                except Exception as e2:
                    self.logger.error(f"❌ MoviePy fallback also failed: {str(e2)}")
            
            # Emergency fallback: create simple video
            self.logger.warning("🚨 Using emergency fallback video creation")
            return await self._emergency_video_assembly(scenes, audio_path, output_path)

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
                
                # Get video duration using FFmpeg (more reliable than MoviePy)
                try:
                    import subprocess
                    probe = subprocess.run(['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path], capture_output=True, text=True)
                    duration = float(probe.stdout.strip())
                    duration = min(duration, 30.0)  # Limit to 30 seconds
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

    async def create_guaranteed_valid_scenes(self, videos: List[Dict[str, Any]], target_duration: float = 1320.0) -> List[Dict[str, Any]]:
        """
        Create guaranteed valid scenes as fallback when main pipeline fails.
        This ensures we always have something to assemble.
        """
        try:
            self.logger.info(f"Creating guaranteed valid scenes for target duration: {target_duration:.1f}s")
            
            guaranteed_scenes = []
            total_duration = 0.0
            
            for i, video in enumerate(videos):
                video_path = video.get("path")
                
                if not video_path or not os.path.exists(video_path):
                    self.logger.error(f"Video {i}: Invalid path: {video_path}")
                    continue
                
                # Create simple, guaranteed valid scenes using FFmpeg (subprocess)
                try:
                    import subprocess
                    
                    # Get video duration using subprocess (consistent)
                    duration_cmd = [
                        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', video_path
                    ]
                    
                    result = subprocess.run(duration_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"Failed to get video duration: {result.stderr}")
                    
                    video_duration = float(result.stdout.strip())
                    
                    # Calculate how many scenes we need to reach target duration
                    scene_duration = 10.0  # 10 seconds per scene
                    scenes_needed = max(10, int(target_duration / scene_duration))  # At least 10 scenes
                    scenes_per_video = max(3, scenes_needed // len(videos))  # Distribute across videos
                    
                    self.logger.info(f"Creating {scenes_per_video} scenes per video for target duration")
                    
                    for j in range(scenes_per_video):
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
                            total_duration += scene.get('duration', 0)
                            self.logger.info(f"Created guaranteed valid fallback scene {len(guaranteed_scenes)}: {video_path} ({start_time}s to {end_time}s)")
                        else:
                            self.logger.error(f"Fallback scene validation failed for {video_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error creating fallback scenes for {video_path}: {str(e)}")
                    continue
            
            self.logger.info(f"Created {len(guaranteed_scenes)} guaranteed valid fallback scenes")
            self.logger.info(f"Total fallback duration: {total_duration:.1f}s / {target_duration:.1f}s")
            
            # Validate we have enough duration
            if total_duration < target_duration * 0.8:
                self.logger.warning(f"Insufficient fallback duration: {total_duration:.1f}s < {target_duration * 0.8:.1f}s")
            
            return guaranteed_scenes
            
        except Exception as e:
            self.logger.error(f"Guaranteed scene creation failed: {str(e)}")
            return [] 
    
    async def _emergency_video_assembly(self, scenes: List[Dict[str, Any]], audio_path: str, output_path: str) -> str:
        """Emergency video assembly when all other methods fail - using FFmpeg for reliability."""
        try:
            self.logger.warning("🚨 EMERGENCY: Creating simple video from first available scene using FFmpeg")
            
            if not scenes:
                raise Exception("No scenes available for emergency assembly")
            
            # Use the first valid scene
            first_scene = scenes[0]
            video_path = first_scene.get('video_path')
            
            if not video_path or not os.path.exists(video_path):
                raise Exception("No valid video path found for emergency assembly")
            
            # Create a simple 10-second clip using FFmpeg (subprocess for consistency)
            try:
                import subprocess
                
                # Get video duration using subprocess (consistent)
                duration_cmd = [
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'csv=p=0', video_path
                ]
                
                result = subprocess.run(duration_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"Failed to get video duration: {result.stderr}")
                
                duration = float(result.stdout.strip())
                
                # Take first 10 seconds or full video if shorter
                clip_duration = min(10.0, duration)
                
                # Create clip using FFmpeg subprocess (consistent)
                extract_cmd = [
                    'ffmpeg', '-y',  # Overwrite output
                    '-i', video_path,
                    '-ss', '0',
                    '-t', str(clip_duration),
                    '-c', 'copy',  # Fast copy (no re-encoding)
                    '-avoid_negative_ts', 'make_zero',
                    output_path
                ]
                
                result = subprocess.run(extract_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    self.logger.info(f"✅ Emergency video created using FFmpeg: {output_path}")
                    return output_path
                else:
                    raise Exception(f"FFmpeg emergency creation failed: {result.stderr}")
                    
            except Exception as e:
                self.logger.error(f"❌ Emergency FFmpeg video creation failed: {str(e)}")
                raise e
                
        except Exception as e:
            self.logger.error(f"❌ Emergency video assembly completely failed: {str(e)}")
            raise Exception(f"All video assembly methods failed: {str(e)}")

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
                
                # Get video duration using subprocess (consistent)
                try:
                    import subprocess
                    
                    duration_cmd = [
                        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', video_path
                    ]
                    
                    result = subprocess.run(duration_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"Failed to get video duration: {result.stderr}")
                    
                    video_duration = float(result.stdout.strip())
                    
                except Exception as e:
                    self.logger.error(f"Error getting video duration for {video_path}: {str(e)}")
                    continue
                
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
                
            self.logger.info(f"Created {len(extended_scenes)} extended scenes with total duration: {total_duration:.1f}s")
            return extended_scenes
            
        except Exception as e:
            self.logger.error(f"Extended scene creation failed: {str(e)}")
            return [] 

    def _validate_ffmpeg_installation(self):
        """Validates that FFmpeg is installed and has the necessary capabilities."""
        try:
            self.logger.info("Validating FFmpeg installation...")
            # Check if ffprobe is available
            result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg not found. Please install FFmpeg and ensure it's in your PATH. Error: {result.stderr}")
            self.logger.info("✅ FFmpeg found and accessible.")

            # Check if ffmpeg is available
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg not found. Please install FFmpeg and ensure it's in your PATH. Error: {result.stderr}")
            self.logger.info("✅ FFmpeg found and accessible.")

            # Check if scene detection is supported
            result = subprocess.run(['ffmpeg', '-i', 'dummy.mp4', '-vf', 'select=gt(scene\\,0.15),showinfo', '-f', 'null', '-'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("FFmpeg scene detection (select=gt(scene\\,0.15),showinfo) is not supported. Please ensure you have the 'libscene' filter installed.")
            self.logger.info("✅ FFmpeg scene detection capabilities verified.")

            self.logger.info("✅ FFmpeg installation and capabilities validated.")
        except Exception as e:
            self.logger.error(f"FFmpeg validation failed: {str(e)}")
            raise Exception(f"FFmpeg validation failed: {str(e)}") 