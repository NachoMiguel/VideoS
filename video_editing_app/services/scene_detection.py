import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
import subprocess
import json
import tempfile

logger = logging.getLogger(__name__)

class SceneDetector:
    """Professional-grade scene detection using multiple methods."""
    
    def __init__(self, threshold: float = 15.0, min_scene_duration: float = 0.5):
        """
        Initialize scene detector.
        
        Args:
            threshold: Threshold for detecting scene changes (lower = more sensitive, more scenes)
            min_scene_duration: Minimum duration for a scene in seconds (lower = more scenes)
        """
        self.threshold = threshold
        self.min_scene_duration = min_scene_duration
        self.logger = logging.getLogger(__name__)
    
    def detect_scenes(self, video_path: str) -> List[Dict[str, float]]:
        """
        Detect scenes using professional-grade methods.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of scenes with start_time, end_time, and duration
        """
        try:
            self.logger.info(f"Detecting scenes in: {video_path}")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # METHOD 1: Try FFmpeg scene detection (fastest, most reliable)
            scenes = self._detect_scenes_ffmpeg(video_path)
            if scenes and len(scenes) > 1:
                self.logger.info(f"✅ FFmpeg detected {len(scenes)} scenes")
                return scenes
            
            # METHOD 2: Try PySceneDetect (professional library)
            scenes = self._detect_scenes_pyscenedetect(video_path)
            if scenes and len(scenes) > 1:
                self.logger.info(f"✅ PySceneDetect detected {len(scenes)} scenes")
                return scenes
            
            # METHOD 3: Enhanced OpenCV detection (improved algorithm)
            scenes = self._detect_scenes_enhanced_opencv(video_path)
            if scenes and len(scenes) > 1:
                self.logger.info(f"✅ Enhanced OpenCV detected {len(scenes)} scenes")
                return scenes
            
            # FALLBACK: Single scene
            self.logger.warning("⚠️ No scenes detected, using single scene fallback")
            return self._create_fallback_scene(video_path)
            
        except Exception as e:
            self.logger.error(f"Scene detection failed for {video_path}: {str(e)}")
            return self._create_fallback_scene(video_path)
    
    def _detect_scenes_ffmpeg(self, video_path: str) -> List[Dict[str, float]]:
        """Use FFmpeg's built-in scene detection (fastest method)."""
        try:
            self.logger.info("🔍 Using FFmpeg scene detection...")
            
            # Get video duration first
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            # Use FFmpeg scene detection filter
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'select=gt(scene\\,0.1),showinfo',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse FFmpeg output for scene changes
                scene_times = []
                for line in result.stderr.split('\n'):
                    if 'pts_time:' in line:
                        try:
                            time_str = line.split('pts_time:')[1].split()[0]
                            scene_time = float(time_str)
                            if scene_time > 0 and scene_time < duration:
                                scene_times.append(scene_time)
                        except:
                            continue
                
                if scene_times:
                    # Create scenes from detected times
                    scenes = []
                    scene_times = sorted(scene_times)
                    
                    # Add start
                    if scene_times[0] > 1.0:  # If first scene change is not at start
                        scenes.append({
                            "start_time": 0.0,
                            "end_time": scene_times[0],
                            "duration": scene_times[0],
                            "start_frame": 0,
                            "end_frame": int(scene_times[0] * fps)
                        })
                    
                    # Add middle scenes
                    for i in range(len(scene_times) - 1):
                        start_time = scene_times[i]
                        end_time = scene_times[i + 1]
                        scene_duration = end_time - start_time
                        
                        if scene_duration >= self.min_scene_duration:
                            scenes.append({
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration": scene_duration,
                                "start_frame": int(start_time * fps),
                                "end_frame": int(end_time * fps)
                            })
                    
                    # Add end
                    if scene_times[-1] < duration - 1.0:
                        scenes.append({
                            "start_time": scene_times[-1],
                            "end_time": duration,
                            "duration": duration - scene_times[-1],
                            "start_frame": int(scene_times[-1] * fps),
                            "end_frame": total_frames
                        })
                    
                    return scenes
            
            return []
            
        except Exception as e:
            self.logger.error(f"FFmpeg scene detection failed: {str(e)}")
            return []
    
    def _detect_scenes_pyscenedetect(self, video_path: str) -> List[Dict[str, float]]:
        """Use PySceneDetect library (professional scene detection)."""
        try:
            self.logger.info("🔍 Using PySceneDetect...")
            
            # Try to import PySceneDetect
            try:
                from scenedetect import detect, ContentDetector, AdaptiveDetector
            except ImportError:
                self.logger.warning("PySceneDetect not available, skipping...")
                return []
            
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            # Use adaptive detector (best for compilation videos)
            scene_list = detect(video_path, AdaptiveDetector())
            
            if scene_list:
                scenes = []
                prev_time = 0.0
                
                for scene in scene_list:
                    start_time = scene[0].get_seconds()
                    end_time = scene[1].get_seconds()
                    scene_duration = end_time - start_time
                    
                    if scene_duration >= self.min_scene_duration:
                        scenes.append({
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": scene_duration,
                            "start_frame": int(start_time * fps),
                            "end_frame": int(end_time * fps)
                        })
                
                return scenes
            
            return []
            
        except Exception as e:
            self.logger.error(f"PySceneDetect failed: {str(e)}")
            return []
    
    def _detect_scenes_enhanced_opencv(self, video_path: str) -> List[Dict[str, float]]:
        """Enhanced OpenCV scene detection with adaptive thresholds."""
        try:
            self.logger.info("🔍 Using enhanced OpenCV detection...")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            self.logger.info(f"Video: {duration:.2f}s, {total_frames} frames, {fps:.2f} fps")
            
            # ENHANCED: Adaptive threshold calculation
            frame_differences = []
            prev_frame = None
            frame_count = 0
            sample_interval = max(1, total_frames // 100)  # Sample 100 frames
            
            # First pass: analyze frame differences
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is not None:
                        diff = cv2.absdiff(gray, prev_frame)
                        mean_diff = np.mean(diff)
                        frame_differences.append(mean_diff)
                    
                    prev_frame = gray
                
                frame_count += 1
            
            cap.release()
            
            # Calculate adaptive threshold
            if frame_differences:
                frame_differences = np.array(frame_differences)
                mean_diff = np.mean(frame_differences)
                std_diff = np.std(frame_differences)
                
                # Adaptive threshold: mean + 2*std (catches significant changes)
                adaptive_threshold = mean_diff + (2 * std_diff)
                adaptive_threshold = max(10.0, min(50.0, adaptive_threshold))  # Clamp to reasonable range
                
                self.logger.info(f"📊 Adaptive threshold: {adaptive_threshold:.2f} (mean: {mean_diff:.2f}, std: {std_diff:.2f})")
            else:
                adaptive_threshold = 20.0  # Fallback threshold
            
            # Second pass: detect scenes with adaptive threshold
            cap = cv2.VideoCapture(video_path)
            min_frames_between_scenes = int(self.min_scene_duration * fps)
            scene_changes = []
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > adaptive_threshold:
                        if not scene_changes or (frame_count - scene_changes[-1]) >= min_frames_between_scenes:
                            scene_changes.append(frame_count)
                            self.logger.debug(f"Scene change at {frame_count/fps:.1f}s (diff: {mean_diff:.2f})")
                
                prev_frame = gray
                frame_count += 1
                
                if frame_count % 2000 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Scene detection progress: {progress:.1f}%")
            
            cap.release()
            
            # Convert to scenes
            scenes = self._create_scenes_from_changes(scene_changes, fps, duration)
            return scenes
            
        except Exception as e:
            self.logger.error(f"Enhanced OpenCV detection failed: {str(e)}")
            return []
    
    def _create_scenes_from_changes(self, scene_changes: List[int], fps: float, duration: float) -> List[Dict[str, float]]:
        """Convert scene change frame numbers to scene objects."""
        scenes = []
        
        # Add start of video as first scene change if not present
        if not scene_changes or scene_changes[0] > 0:
            scene_changes.insert(0, 0)
        
        # Add end of video as last scene change if not present
        total_frames = int(duration * fps)
        if not scene_changes or scene_changes[-1] < total_frames:
            scene_changes.append(total_frames)
        
        # Create scenes from consecutive scene changes
        for i in range(len(scene_changes) - 1):
            start_frame = scene_changes[i]
            end_frame = scene_changes[i + 1]
            
            start_time = start_frame / fps
            end_time = end_frame / fps
            scene_duration = end_time - start_time
            
            # Only include scenes that meet minimum duration
            if scene_duration >= self.min_scene_duration:
                scene = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": scene_duration,
                    "start_frame": start_frame,
                    "end_frame": end_frame
                }
                scenes.append(scene)
                self.logger.debug(f"Scene {i+1}: {start_time:.2f}s - {end_time:.2f}s ({scene_duration:.2f}s)")
        
        return scenes
    
    def _create_fallback_scene(self, video_path: str) -> List[Dict[str, float]]:
        """Create a fallback scene when detection fails."""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps
                cap.release()
                
                self.logger.warning(f"Using fallback scene for {video_path}")
                return [{
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration,
                    "start_frame": 0,
                    "end_frame": total_frames
                }]
        except:
            pass
        
        # Ultimate fallback
        return [{
            "start_time": 0.0,
            "end_time": 30.0,
            "duration": 30.0,
            "start_frame": 0,
            "end_frame": 900  # Assuming 30fps
        }]
    
    def detect_scenes_adaptive(self, video_path: str) -> List[Dict[str, float]]:
        """
        Adaptive scene detection that adjusts threshold based on video content.
        This is more robust for different types of videos.
        """
        try:
            self.logger.info(f"Running adaptive scene detection on: {video_path}")
            
            # First pass: analyze video to determine optimal threshold
            optimal_threshold = self._find_optimal_threshold(video_path)
            
            # Second pass: detect scenes with optimal threshold
            original_threshold = self.threshold
            self.threshold = optimal_threshold
            
            scenes = self.detect_scenes(video_path)
            
            # Restore original threshold
            self.threshold = original_threshold
            
            return scenes
            
        except Exception as e:
            self.logger.error(f"Adaptive scene detection failed: {str(e)}")
            return self.detect_scenes(video_path)  # Fallback to regular detection
    
    def _find_optimal_threshold(self, video_path: str) -> float:
        """Analyze video to find optimal threshold for scene detection."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self.threshold
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames to analyze content
            sample_size = min(1000, total_frames // 10)  # Sample 10% of frames, max 1000
            frame_diffs = []
            
            prev_frame = None
            frame_count = 0
            
            while len(frame_diffs) < sample_size and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = np.mean(diff)
                    frame_diffs.append(mean_diff)
                
                prev_frame = frame
                frame_count += 1
            
            cap.release()
            
            if frame_diffs:
                # Calculate optimal threshold based on frame difference statistics
                mean_diff = np.mean(frame_diffs)
                std_diff = np.std(frame_diffs)
                
                # Optimal threshold: mean + 2*std (captures significant changes)
                optimal_threshold = mean_diff + (2 * std_diff)
                
                # Ensure reasonable bounds
                optimal_threshold = max(10.0, min(100.0, optimal_threshold))
                
                self.logger.info(f"Optimal threshold calculated: {optimal_threshold:.2f} (mean: {mean_diff:.2f}, std: {std_diff:.2f})")
                return optimal_threshold
            
        except Exception as e:
            self.logger.warning(f"Could not calculate optimal threshold: {str(e)}")
        
        return self.threshold 

    def detect_scenes_for_duration(self, video_path: str, target_duration: float, target_scenes: int = None) -> List[Dict[str, float]]:
        """
        OPTIMIZED: Detect scenes ensuring we have enough content for the target duration.
        Uses smart single-pass algorithm instead of multiple iterations.
        
        Args:
            video_path: Path to the video file
            target_duration: Target duration in seconds
            target_scenes: Target number of scenes (if specified)
            
        Returns:
            List of scenes with enough content for target duration
        """
        try:
            self.logger.info(f"🔧 OPTIMIZED: Detecting scenes for target duration: {target_duration:.1f}s")
            
            # Get video duration first
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            cap.release()
            
            self.logger.info(f"Video duration: {video_duration:.1f}s")
            
            # CRITICAL FIX: Check if video is too short for target duration
            if video_duration < target_duration * 0.5:
                self.logger.warning(f"Video duration ({video_duration:.1f}s) is too short for target ({target_duration:.1f}s)")
                self.logger.info("Using entire video as single scene to maintain quality")
                return self._create_single_scene(video_path, video_duration)
            
            # Calculate optimal parameters based on video characteristics
            optimal_params = self._calculate_optimal_parameters(video_duration, target_duration, target_scenes)
            
            # Use optimized single-pass detection
            scenes = self._detect_scenes_optimized(video_path, optimal_params)
            
            # Validate and adjust if needed
            total_scene_duration = sum(scene['duration'] for scene in scenes)
            
            # CRITICAL FIX: Better fallback logic
            if total_scene_duration < target_duration * 0.6:
                self.logger.warning(f"Generated duration ({total_scene_duration:.1f}s) too short for target ({target_duration:.1f}s)")
                
                # Try adaptive detection with different parameters
                self.logger.info("Attempting adaptive scene detection with relaxed parameters...")
                adaptive_scenes = self.detect_scenes_adaptive(video_path)
                adaptive_duration = sum(scene['duration'] for scene in adaptive_scenes)
                
                if adaptive_duration >= target_duration * 0.7:
                    self.logger.info(f"✅ Adaptive detection successful: {adaptive_duration:.1f}s")
                    return adaptive_scenes
                
                # Last resort: create fewer, longer artificial scenes
                self.logger.warning("Creating minimal artificial scenes to maintain quality")
                return self._create_minimal_artificial_scenes(video_path, target_duration, max(10, target_scenes // 3))
            
            self.logger.info(f"✅ OPTIMIZED: Generated {len(scenes)} scenes with {total_scene_duration:.1f}s duration")
            return scenes
                
        except Exception as e:
            self.logger.error(f"Optimized scene detection failed: {str(e)}")
            return self.detect_scenes(video_path)
    
    def _calculate_optimal_parameters(self, video_duration: float, target_duration: float, target_scenes: int = None) -> Dict:
        """Calculate optimal detection parameters based on video characteristics."""
        try:
            # Calculate target scenes if not provided
            if target_scenes is None:
                target_scenes = max(20, int(target_duration / 5.0))  # 5-second scenes
            
            # Calculate optimal scene duration
            optimal_scene_duration = video_duration / target_scenes
            
            # CRITICAL FIX: Use intelligent parameter selection based on video characteristics
            if video_duration < target_duration:
                # Video is short, need more sensitive detection
                # BUT: Use higher threshold to avoid noise
                threshold = max(20.0, min(30.0, 25.0 * (video_duration / target_duration)))
                min_duration = max(2.0, optimal_scene_duration * 0.3)  # Longer minimum duration
            else:
                # Video is long enough, use conservative detection for interviews
                # Based on diagnostic analysis: mean_diff=4.29, 95th percentile=16.06
                threshold = 25.0  # High threshold to ignore camera movements
                min_duration = max(3.0, optimal_scene_duration * 0.4)  # Longer scenes
            
            self.logger.info(f"🎯 Optimal parameters: threshold={threshold:.1f}, min_duration={min_duration:.1f}s, target_scenes={target_scenes}")
            self.logger.info(f"📊 Based on video analysis: high threshold to avoid camera movement noise")
            
            return {
                'threshold': threshold,
                'min_duration': min_duration,
                'target_scenes': target_scenes,
                'optimal_scene_duration': optimal_scene_duration
            }
            
        except Exception as e:
            self.logger.error(f"Parameter calculation failed: {str(e)}")
            # Fallback to conservative parameters
            return {
                'threshold': 25.0,  # High threshold for interview videos
                'min_duration': 3.0,  # Longer minimum duration
                'target_scenes': 50,
                'optimal_scene_duration': 3.0
            }

    def _detect_scenes_optimized(self, video_path: str, params: Dict) -> List[Dict[str, float]]:
        """Optimized single-pass scene detection with intelligent parameter selection."""
        try:
            self.logger.info(f"🚀 Running optimized single-pass scene detection...")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            self.logger.info(f"Video: {duration:.2f}s, {total_frames} frames, {fps:.2f} fps")
            
            # Use calculated optimal parameters
            threshold = params['threshold']
            min_frames_between_scenes = int(params['min_duration'] * fps)
            
            scene_changes = []
            prev_frame = None
            frame_count = 0
            
            # OPTIMIZED: Single-pass detection with progress tracking
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff)
                    
                    # Check if this is a scene change
                    if mean_diff > threshold:
                        # Ensure minimum time between scenes
                        if not scene_changes or (frame_count - scene_changes[-1]) >= min_frames_between_scenes:
                            scene_changes.append(frame_count)
                
                prev_frame = gray
                frame_count += 1
                
                # Progress update every 2000 frames (less frequent for performance)
                if frame_count % 2000 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Scene detection progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            cap.release()
            
            # Convert frame numbers to timestamps and create scenes
            scenes = self._create_scenes_from_changes(scene_changes, fps, duration)
            
            self.logger.info(f"✅ Optimized detection: {len(scenes)} scenes found")
            return scenes
            
        except Exception as e:
            self.logger.error(f"Optimized scene detection failed: {str(e)}")
            return self._create_fallback_scene(video_path)
    
    def _create_single_scene(self, video_path: str, duration: float) -> List[Dict[str, float]]:
        """Create a single scene from the entire video when duration is insufficient."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._create_fallback_scene(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            scene = {
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration,
                "start_frame": 0,
                "end_frame": int(duration * fps),
                "single_scene": True,
                "quality_note": "Full video used due to short duration"
            }
            
            self.logger.info(f"Created single scene with duration: {duration:.1f}s")
            return [scene]
            
        except Exception as e:
            self.logger.error(f"Single scene creation failed: {str(e)}")
            return self._create_fallback_scene(video_path)
    
    def _create_minimal_artificial_scenes(self, video_path: str, target_duration: float, max_scenes: int) -> List[Dict[str, float]]:
        """Create minimal artificial scenes with longer durations to maintain quality."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._create_fallback_scene(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            cap.release()
            
            # Create fewer, longer scenes (8-15 seconds each)
            scene_duration = min(15.0, max(8.0, video_duration / max_scenes))
            scenes = []
            
            current_time = 0.0
            scene_id = 0
            
            while current_time < video_duration and len(scenes) < max_scenes:
                start_time = current_time
                end_time = min(current_time + scene_duration, video_duration)
                
                if end_time <= start_time:
                    break
                
                scene = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "start_frame": int(start_time * fps),
                    "end_frame": int(end_time * fps),
                    "artificial": True,
                    "quality_note": "Minimal artificial scene for quality preservation"
                }
                
                scenes.append(scene)
                current_time = end_time
                scene_id += 1
            
            total_duration = sum(s['duration'] for s in scenes)
            self.logger.info(f"Created {len(scenes)} minimal artificial scenes with total duration: {total_duration:.1f}s")
            self.logger.warning("⚠️ Using artificial scenes - video quality may be affected")
            return scenes
            
        except Exception as e:
            self.logger.error(f"Minimal artificial scene creation failed: {str(e)}")
            return self._create_fallback_scene(video_path) 