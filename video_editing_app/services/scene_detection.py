import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SceneDetector:
    """Free scene detection using OpenCV-based shot boundary detection."""
    
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
        Detect scenes in a video using frame difference analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of scenes with start_time, end_time, and duration
        """
        try:
            self.logger.info(f"Detecting scenes in: {video_path}")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            self.logger.info(f"Video: {duration:.2f}s, {total_frames} frames, {fps:.2f} fps")
            
            # Scene detection parameters
            min_frames_between_scenes = int(self.min_scene_duration * fps)
            scene_changes = []
            prev_frame = None
            frame_count = 0
            
            # Process frames to detect scene changes
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
                    if mean_diff > self.threshold:
                        # Ensure minimum time between scenes
                        if not scene_changes or (frame_count - scene_changes[-1]) >= min_frames_between_scenes:
                            scene_changes.append(frame_count)
                            self.logger.debug(f"Scene change detected at frame {frame_count} (diff: {mean_diff:.2f})")
                
                prev_frame = gray
                frame_count += 1
                
                # Progress update every 1000 frames
                if frame_count % 1000 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Scene detection progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            cap.release()
            
            # Convert frame numbers to timestamps and create scenes
            scenes = self._create_scenes_from_changes(scene_changes, fps, duration)
            
            self.logger.info(f"Detected {len(scenes)} scenes in {video_path}")
            return scenes
            
        except Exception as e:
            self.logger.error(f"Scene detection failed for {video_path}: {str(e)}")
            # Return a single scene as fallback
            return self._create_fallback_scene(video_path)
    
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
        Detect scenes ensuring we have enough content for the target duration.
        
        Args:
            video_path: Path to the video file
            target_duration: Target duration in seconds
            target_scenes: Target number of scenes (if specified)
            
        Returns:
            List of scenes with enough content for target duration
        """
        try:
            self.logger.info(f"Detecting scenes for target duration: {target_duration:.1f}s")
            
            # Get video duration first
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            cap.release()
            
            self.logger.info(f"Video duration: {video_duration:.1f}s")
            
            # If video is shorter than target, we need to create more scenes
            if video_duration < target_duration:
                self.logger.warning(f"Video duration ({video_duration:.1f}s) < target ({target_duration:.1f}s) - will create more scenes")
                
                # Calculate how many scenes we need
                if target_scenes is None:
                    # Aim for 2-4 second scenes to reach target duration
                    target_scenes = max(50, int(target_duration / 3.0))
                
                self.logger.info(f"Target scenes: {target_scenes}")
                
                # Try different thresholds to get more scenes
                thresholds = [10.0, 8.0, 5.0, 3.0]  # More sensitive thresholds
                min_durations = [0.3, 0.2, 0.1]  # Shorter minimum durations
                
                for threshold in thresholds:
                    for min_duration in min_durations:
                        self.logger.info(f"Trying threshold={threshold}, min_duration={min_duration}")
                        
                        # Temporarily change parameters
                        original_threshold = self.threshold
                        original_min_duration = self.min_scene_duration
                        
                        self.threshold = threshold
                        self.min_scene_duration = min_duration
                        
                        # Detect scenes
                        scenes = self.detect_scenes(video_path)
                        
                        # Restore original parameters
                        self.threshold = original_threshold
                        self.min_scene_duration = original_min_duration
                        
                        # Check if we have enough scenes
                        total_scene_duration = sum(scene['duration'] for scene in scenes)
                        
                        self.logger.info(f"Generated {len(scenes)} scenes with total duration: {total_scene_duration:.1f}s")
                        
                        if len(scenes) >= target_scenes and total_scene_duration >= target_duration * 0.8:
                            self.logger.info(f"✅ Found sufficient scenes: {len(scenes)} scenes, {total_scene_duration:.1f}s")
                            return scenes
                
                # If we still don't have enough, create artificial scenes
                self.logger.warning("Creating artificial scenes to meet target duration")
                return self._create_artificial_scenes(video_path, target_duration, target_scenes)
            
            else:
                # Video is long enough, use normal detection
                return self.detect_scenes_adaptive(video_path)
                
        except Exception as e:
            self.logger.error(f"Duration-based scene detection failed: {str(e)}")
            return self.detect_scenes(video_path)
    
    def _create_artificial_scenes(self, video_path: str, target_duration: float, target_scenes: int) -> List[Dict[str, float]]:
        """Create artificial scenes by dividing video into equal segments."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._create_fallback_scene(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            cap.release()
            
            # Create scenes of 2-4 seconds each
            scene_duration = min(4.0, max(2.0, video_duration / target_scenes))
            scenes = []
            
            current_time = 0.0
            scene_id = 0
            
            while current_time < video_duration and len(scenes) < target_scenes:
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
                    "artificial": True
                }
                
                scenes.append(scene)
                current_time = end_time
                scene_id += 1
            
            self.logger.info(f"Created {len(scenes)} artificial scenes with total duration: {sum(s['duration'] for s in scenes):.1f}s")
            return scenes
            
        except Exception as e:
            self.logger.error(f"Artificial scene creation failed: {str(e)}")
            return self._create_fallback_scene(video_path) 