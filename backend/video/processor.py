import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
import os
import json
from pathlib import Path
import asyncio
import time

from core.config import settings
from core.logger import logger
from core.exceptions import VideoProcessingError
from .face_detection import FaceDetector

class VideoProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.face_detector = FaceDetector()
        
    async def process_videos_parallel(
        self,
        video_paths: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Process multiple videos in parallel for scene detection and face recognition."""
        try:
            if settings.parallel_processing and len(video_paths) > 1:
                # Process videos in parallel
                futures = []
                for video_path in video_paths:
                    future = self.executor.submit(self._process_single_video, video_path)
                    futures.append(future)
                
                results = []
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if progress_callback:
                            progress = int(((i + 1) / len(video_paths)) * 100)
                            await progress_callback(
                                f"Processed video {i + 1}/{len(video_paths)}",
                                progress
                            )
                    except Exception as e:
                        logger.error(f"Error processing video: {str(e)}")
                        continue
                
                return {
                    "videos_processed": len(results),
                    "total_scenes": sum(len(r["scenes"]) for r in results),
                    "results": results
                }
            else:
                # Process videos sequentially
                results = []
                for i, video_path in enumerate(video_paths):
                    try:
                        result = await self._process_single_video_async(video_path)
                        results.append(result)
                        
                        if progress_callback:
                            progress = int(((i + 1) / len(video_paths)) * 100)
                            await progress_callback(
                                f"Processed video {i + 1}/{len(video_paths)}",
                                progress
                            )
                    except Exception as e:
                        logger.error(f"Error processing video {video_path}: {str(e)}")
                        continue
                
                return {
                    "videos_processed": len(results),
                    "total_scenes": sum(len(r["scenes"]) for r in results),
                    "results": results
                }
                
        except Exception as e:
            logger.error(f"Error in parallel video processing: {str(e)}")
            raise VideoProcessingError(f"Parallel processing failed: {str(e)}")
    
    async def _process_single_video_async(self, video_path: str) -> Dict[str, Any]:
        """Async wrapper for single video processing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._process_single_video, video_path)
    
    def _process_single_video(self, video_path: str) -> Dict[str, Any]:
        """Process a single video for scene detection and face recognition."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            scenes = []
            faces_detected = []
            current_scene = None
            frame_count = 0
            scene_id = 0
            
            # Enhanced scene detection variables
            frame_history = []
            luminosity_history = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Enhanced scene transition detection
                is_transition = self._enhanced_scene_transition_detection(
                    frame, frame_count, frame_history, luminosity_history
                )
                
                # Process frames in batches for efficiency
                if frame_count % settings.face_detection_batch_size == 0:
                    # Scene detection
                    if current_scene is None:
                        current_scene = {
                            "id": f"scene_{scene_id}",
                            "start_frame": frame_count,
                            "faces": [],
                            "video_path": video_path,
                            "detected_actions": []
                        }
                        scene_id += 1
                    
                    # Advanced face detection using InsightFace
                    faces = self._detect_faces_insightface_sync(frame, frame_count / fps)
                    if faces:
                        current_scene["faces"].extend(faces)
                        faces_detected.append({
                            "frame": frame_count,
                            "timestamp": frame_count / fps,
                            "faces": faces
                        })
                    
                    # Scene transition detection
                    if is_transition and current_scene and frame_count > current_scene["start_frame"] + (fps * 2):  # Minimum 2 seconds
                        current_scene["end_frame"] = frame_count
                        current_scene["duration"] = (
                            current_scene["end_frame"] - current_scene["start_frame"]
                        ) / fps
                        current_scene["start_time"] = current_scene["start_frame"] / fps
                        current_scene["end_time"] = current_scene["end_frame"] / fps
                        
                        # Add scene quality metrics
                        current_scene["quality_metrics"] = self._calculate_scene_metrics(current_scene)
                        
                        scenes.append(current_scene)
                        current_scene = None
                
                frame_count += 1
            
            # Add final scene if exists
            if current_scene:
                current_scene["end_frame"] = frame_count
                current_scene["duration"] = (
                    current_scene["end_frame"] - current_scene["start_frame"]
                ) / fps
                current_scene["start_time"] = current_scene["start_frame"] / fps
                current_scene["end_time"] = current_scene["end_frame"] / fps
                current_scene["quality_metrics"] = self._calculate_scene_metrics(current_scene)
                scenes.append(current_scene)
            
            cap.release()
            
            return {
                "video_path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "duration": total_frames / fps,
                "scenes": scenes,
                "faces_detected": faces_detected,
                "total_scenes": len(scenes)
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise VideoProcessingError(f"Processing failed: {str(e)}")

    def _detect_faces_insightface_sync(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """Synchronous wrapper for InsightFace detection."""
        try:
            # Use InsightFace for detection
            faces = self.face_detector._detect_faces_insightface(frame)
            
            # Add timestamp and enhance face data
            enhanced_faces = []
            for i, face in enumerate(faces):
                enhanced_face = {
                    "id": f"face_{timestamp}_{i}",
                    "timestamp": timestamp,
                    "bbox": face["bbox"],
                    "confidence": face["confidence"],
                    "embedding": face["embedding"].tolist(),  # Convert numpy array to list for JSON serialization
                    "quality_score": face["quality_score"],
                    "character": None  # Will be filled by character identification
                }
                
                # Try to identify character if we have trained embeddings
                if self.face_detector.known_faces:
                    character_result = self.face_detector.identify_character(face["embedding"])
                    if character_result:
                        enhanced_face["character"] = character_result[0]
                        enhanced_face["character_confidence"] = character_result[1]
                
                enhanced_faces.append(enhanced_face)
            
            return enhanced_faces
            
        except Exception as e:
            logger.error(f"InsightFace detection error: {str(e)}")
            # Fallback to basic detection
            return self._detect_faces_basic(frame, timestamp)

    async def _detect_faces_advanced(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """Use advanced face detection with character recognition."""
        try:
            # Use the InsightFace detector
            result = await self.face_detector._analyze_frame(frame, timestamp)
            if result and result.get('faces'):
                # Enhance faces with character identification
                enhanced_faces = []
                for i, face in enumerate(result['faces']):
                    enhanced_face = {
                        "id": f"face_{timestamp}_{i}",
                        "timestamp": timestamp,
                        "bbox": face["bbox"],
                        "confidence": face["confidence"],
                        "embedding": face["embedding"].tolist(),
                        "quality_score": face["quality_score"],
                        "character": None
                    }
                    
                    # Try to identify character
                    if self.face_detector.known_faces:
                        character_result = self.face_detector.identify_character(face["embedding"])
                        if character_result:
                            enhanced_face["character"] = character_result[0]
                            enhanced_face["character_confidence"] = character_result[1]
                    
                    enhanced_faces.append(enhanced_face)
                
                return enhanced_faces
            return []
        except Exception as e:
            logger.error(f"Advanced face detection error: {str(e)}")
            # Fallback to basic detection
            return self._detect_faces_basic(frame, timestamp)
    
    def _detect_faces_basic(self, frame: np.ndarray, timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """Fallback basic face detection using Haar Cascade."""
        try:
            if not hasattr(self, 'face_cascade'):
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return [
                {
                    "id": f"face_{timestamp}_{i}",
                    "timestamp": timestamp,
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": 0.8,  # Default confidence for Haar Cascade
                    "embedding": None,  # No embedding available
                    "quality_score": 0.5,  # Default quality score
                    "character": None
                }
                for i, (x, y, w, h) in enumerate(faces)
            ]
        except Exception as e:
            logger.error(f"Basic face detection error: {str(e)}")
            return []

    def _enhanced_scene_transition_detection(
        self, 
        frame: np.ndarray, 
        frame_count: int,
        frame_history: List[np.ndarray],
        luminosity_history: List[float]
    ) -> bool:
        """Enhanced scene transition detection using multiple methods."""
        try:
            # Convert to grayscale for analysis
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Frame difference (existing)
            frame_diff_transition = self._frame_difference_detection(gray_frame)
            
            # Method 2: Histogram analysis
            hist_transition = self._histogram_based_detection(frame, frame_history)
            
            # Method 3: Luminosity analysis
            luminosity_transition = self._luminosity_based_detection(gray_frame, luminosity_history)
            
            # Method 4: Edge density analysis
            edge_transition = self._edge_density_detection(gray_frame, frame_history)
            
            # Combine methods with weights
            transition_score = (
                0.3 * frame_diff_transition +
                0.3 * hist_transition +
                0.2 * luminosity_transition +
                0.2 * edge_transition
            )
            
            # Update history
            frame_history.append(gray_frame)
            if len(frame_history) > 5:  # Keep last 5 frames
                frame_history.pop(0)
            
            luminosity_history.append(np.mean(gray_frame))
            if len(luminosity_history) > 10:  # Keep last 10 luminosity values
                luminosity_history.pop(0)
            
            return transition_score > 0.6  # Threshold for scene transition
            
        except Exception as e:
            logger.error(f"Enhanced scene transition detection error: {str(e)}")
            return False

    def _frame_difference_detection(self, gray_frame: np.ndarray, threshold: float = 30.0) -> float:
        """Frame difference based transition detection."""
        try:
            if not hasattr(self, '_prev_frame'):
                self._prev_frame = gray_frame
                return 0.0
            
            frame_diff = cv2.absdiff(gray_frame, self._prev_frame)
            mean_diff = np.mean(frame_diff)
            self._prev_frame = gray_frame
            
            return min(mean_diff / threshold, 1.0)
        except:
            return 0.0

    def _histogram_based_detection(self, frame: np.ndarray, frame_history: List[np.ndarray]) -> float:
        """Histogram-based scene transition detection."""
        try:
            if len(frame_history) == 0:
                return 0.0
            
            # Calculate histogram for current frame
            hist_current = cv2.calcHist([frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            
            # Compare with previous frame
            prev_frame = frame_history[-1]
            if len(prev_frame.shape) == 2:  # Convert grayscale to BGR for histogram
                prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2BGR)
            
            hist_prev = cv2.calcHist([prev_frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            
            # Calculate correlation
            correlation = cv2.compareHist(hist_current, hist_prev, cv2.HISTCMP_CORREL)
            
            return 1.0 - correlation  # Higher difference = higher transition score
        except:
            return 0.0

    def _luminosity_based_detection(self, gray_frame: np.ndarray, luminosity_history: List[float]) -> float:
        """Luminosity-based scene transition detection."""
        try:
            if len(luminosity_history) < 3:
                return 0.0
            
            current_luminosity = np.mean(gray_frame)
            avg_luminosity = np.mean(luminosity_history[-3:])
            
            # Calculate relative change
            if avg_luminosity > 0:
                change = abs(current_luminosity - avg_luminosity) / avg_luminosity
                return min(change * 2, 1.0)  # Scale and cap at 1.0
            
            return 0.0
        except:
            return 0.0

    def _edge_density_detection(self, gray_frame: np.ndarray, frame_history: List[np.ndarray]) -> float:
        """Edge density based scene transition detection."""
        try:
            if len(frame_history) == 0:
                return 0.0
            
            # Calculate edge density for current frame
            edges_current = cv2.Canny(gray_frame, 50, 150)
            density_current = np.sum(edges_current > 0) / edges_current.size
            
            # Calculate edge density for previous frame
            prev_frame = frame_history[-1]
            edges_prev = cv2.Canny(prev_frame, 50, 150)
            density_prev = np.sum(edges_prev > 0) / edges_prev.size
            
            # Calculate relative change in edge density
            if density_prev > 0:
                change = abs(density_current - density_prev) / density_prev
                return min(change * 3, 1.0)  # Scale and cap at 1.0
            
            return 0.0
        except:
            return 0.0

    def _calculate_scene_metrics(self, scene: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for a scene."""
        try:
            metrics = {
                "face_count": len(scene.get("faces", [])),
                "face_confidence_avg": 0.0,
                "duration_score": 0.0,
                "stability_score": 0.8  # Default stability score
            }
            
            # Calculate average face confidence
            faces = scene.get("faces", [])
            if faces:
                confidences = [face.get("confidence", 0.0) for face in faces]
                metrics["face_confidence_avg"] = np.mean(confidences)
            
            # Calculate duration score (ideal: 5-15 seconds)
            duration = scene.get("duration", 0)
            if duration < 5:
                metrics["duration_score"] = duration / 5
            elif duration > 15:
                metrics["duration_score"] = 15 / duration
            else:
                metrics["duration_score"] = 1.0
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating scene metrics: {str(e)}")
            return {"face_count": 0, "face_confidence_avg": 0.0, "duration_score": 0.0, "stability_score": 0.0}

    def _create_transition(self, width: int, height: int, fps: float, duration: float = 0.5) -> np.ndarray:
        """Create a simple fade transition."""
        n_frames = int(fps * duration)
        transition = []
        for i in range(n_frames):
            alpha = i / n_frames
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 1-alpha, frame, alpha, 0)
            transition.append(frame)
        return np.array(transition)

    async def create_compilation(
        self,
        selected_scenes: List[Dict[str, Any]],
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> str:
        """Create video compilation from selected scenes with transitions."""
        try:
            # Sort scenes by timestamp
            selected_scenes.sort(key=lambda x: x.get("timestamp", 0))
            
            # Process scenes in parallel
            futures = []
            temp_files = []
            
            # Enforce 30-second limit per scene
            for scene in selected_scenes:
                duration = scene["end_frame"] - scene["start_frame"]
                fps = cv2.VideoCapture(scene["video_path"]).get(cv2.CAP_PROP_FPS)
                if duration / fps > 30:
                    scene["end_frame"] = scene["start_frame"] + int(30 * fps)
            
            for i, scene in enumerate(selected_scenes):
                temp_output = f"{settings.temp_dir}/scene_{i}.mp4"
                future = self.executor.submit(
                    self._extract_scene,
                    scene["video_path"],
                    scene["start_frame"],
                    scene["end_frame"],
                    temp_output
                )
                futures.append((temp_output, future))
                temp_files.append(temp_output)
            
            # Wait for all scenes to be extracted
            total_scenes = len(selected_scenes)
            completed = 0
            
            for temp_file, future in futures:
                try:
                    future.result()
                    completed += 1
                    
                    if progress_callback:
                        progress = int((completed / total_scenes) * 50)
                        await progress_callback(
                            f"Extracted scene {completed}/{total_scenes}",
                            progress
                        )
                except Exception as e:
                    logger.error(f"Error extracting scene: {str(e)}")
            
            # Combine scenes with transitions
            await self._combine_scenes_with_transitions(temp_files, output_path, progress_callback)
            
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Error removing temp file {temp_file}: {str(e)}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating compilation: {str(e)}")
            raise VideoProcessingError(f"Compilation failed: {str(e)}")
    
    def _extract_scene(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        output_path: str
    ) -> None:
        """Extract a scene from video to temporary file."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
            
            while frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
        except Exception as e:
            logger.error(f"Error extracting scene: {str(e)}")
            raise VideoProcessingError(f"Scene extraction failed: {str(e)}")
    
    async def _combine_scenes_with_transitions(
        self,
        scene_files: List[str],
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> None:
        """Combine scenes with transitions every 5 scenes."""
        try:
            # Get video properties from first scene
            cap = cv2.VideoCapture(scene_files[0])
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Create transition frames
            transition = self._create_transition(width, height, fps)
            
            total_scenes = len(scene_files)
            for i, scene_file in enumerate(scene_files):
                # Read and write current scene
                cap = cv2.VideoCapture(scene_file)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
                
                # Add transition every 5 scenes (except for the last scene)
                if i < total_scenes - 1 and (i + 1) % 5 == 0:
                    for trans_frame in transition:
                        out.write(trans_frame)
                
                if progress_callback:
                    progress = 50 + int((i + 1) / total_scenes * 50)
                    await progress_callback(
                        f"Combining scene {i + 1}/{total_scenes}",
                        progress
                    )
            
            out.release()
            
        except Exception as e:
            logger.error(f"Error combining scenes: {str(e)}")
            raise VideoProcessingError(f"Scene combination failed: {str(e)}")

# Global processor instance
processor = VideoProcessor()
