import cv2
import numpy as np
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import pickle
import time
import os
import json

from ..core.config import settings
from ..core.exceptions import VideoProcessingError, FaceDetectionError
from ..core.parallel import parallel_processor, parallel_task

logger = logging.getLogger(__name__)

class FaceDetector:
    """Enhanced face detection with OpenCV and parallel processing."""
    
    def __init__(self):
        self.cache_dir = Path(settings.cache_dir) / "faces"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Parallel processing settings
        self.parallel_enabled = settings.parallel_face_detection
        self.max_workers = settings.max_workers
        self.batch_size = settings.face_detection_batch_size
        
        # Face detection settings
        self.min_confidence = settings.min_face_confidence
        
        # Initialize OpenCV face detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_net = self._load_dnn_face_detector()
        
        # Caching
        self.enable_caching = settings.enable_caching
        self.cache_ttl = settings.cache_ttl_hours * 3600  # Convert to seconds
        
        # Known characters storage
        self.known_faces = {}  # character_name -> face_features
        self.character_images = {}  # character_name -> image_paths
        
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        
        logger.info(f"Face detector initialized: parallel={self.parallel_enabled}, caching={self.enable_caching}")

    def _load_dnn_face_detector(self):
        """Load DNN-based face detector for better accuracy."""
        model_file = "backend/models/res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "backend/models/deploy.prototxt"
        
        # If model files don't exist, download them
        if not os.path.exists(model_file):
            os.makedirs("backend/models", exist_ok=True)
            # You'll need to download these files from:
            # https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
            # https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt
            logger.warning("DNN model files not found. Falling back to Haar Cascade classifier.")
            return None
            
        return cv2.dnn.readNet(model_file, config_file)

    def _detect_faces_dnn(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using DNN for better accuracy."""
        if self.face_net is None:
            return self._detect_faces_haar(frame)
            
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        h, w = frame.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                face_region = frame[y1:y2, x1:x2]
                if face_region.size > 0:
                    faces.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'confidence': float(confidence),
                        'features': self._extract_face_features(face_region)
                    })
        
        return faces

    def _detect_faces_haar(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback face detection using Haar Cascade classifier."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [
            {
                'bbox': (x, y, w, h),
                'confidence': self._calculate_confidence(frame[y:y+h, x:x+w]),
                'features': self._extract_face_features(frame[y:y+h, x:x+w])
            }
            for (x, y, w, h) in faces
        ]

    def _extract_face_features(self, face_region: np.ndarray) -> np.ndarray:
        """Extract face features using OpenCV."""
        # Resize for consistent feature extraction
        face_region = cv2.resize(face_region, (128, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for lighting invariance
        gray = cv2.equalizeHist(gray)
        
        # Extract HOG features
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(gray)
        
        return features.flatten()

    def _calculate_confidence(self, face_region: np.ndarray) -> float:
        """Calculate confidence score for detected face."""
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate face confidence based on image quality metrics
        # 1. Variance of Laplacian (focus measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_measure = np.var(laplacian)
        
        # 2. Contrast measure
        contrast = np.std(gray)
        
        # 3. Face size score
        height, width = face_region.shape[:2]
        size_score = min(1.0, (width * height) / (200 * 200))
        
        # Combine metrics into final confidence score
        confidence = (
            0.4 * min(1.0, focus_measure / 1000) +  # Focus weight
            0.3 * min(1.0, contrast / 100) +        # Contrast weight
            0.3 * size_score                        # Size weight
        )
        
        return float(confidence)

    async def process_video_segment(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        sample_interval: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Process video segment with face detection."""
        cache_path = self._get_cache_path(video_path, start_time, duration)
        
        # Try to load from cache first
        cached_results = self._load_from_cache(cache_path)
        if cached_results is not None:
            return cached_results
        
        try:
            # Extract frames from video segment
            frames = await self._extract_frames(video_path, start_time, duration, sample_interval)
            
            # Process frames
            if self.parallel_enabled and len(frames) > 1:
                results = await self._process_segment_parallel(frames)
            else:
                results = await self._process_segment_sequential(frames)
            
            # Cache results
            self._save_to_cache(cache_path, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video segment: {str(e)}")
            raise VideoProcessingError(f"Failed to process video segment: {str(e)}")

    async def _process_segment_parallel(
        self,
        frames: List[Tuple[np.ndarray, float]]
    ) -> List[Dict[str, Any]]:
        """Process video segment frames in parallel."""
        tasks = []
        for frame, timestamp in frames:
            tasks.append({
                'id': f'frame_{timestamp}',
                'type': 'cpu_intensive',
                'func': self._analyze_frame,
                'args': [frame, timestamp],
                'kwargs': {}
            })
        
        results = await parallel_processor.execute_parallel_tasks(tasks)
        return [r.result for r in results if r.success and r.result]

    async def _process_segment_sequential(
        self,
        frames: List[Tuple[np.ndarray, float]]
    ) -> List[Dict[str, Any]]:
        """Process video segment frames sequentially."""
        results = []
        for frame, timestamp in frames:
            try:
                frame_result = await self._analyze_frame(frame, timestamp)
                if frame_result:
                    results.append(frame_result)
            except Exception as e:
                logger.error(f"Error analyzing frame at {timestamp}: {str(e)}")
        return results

    @parallel_task('cpu_intensive')
    async def _analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single frame for faces."""
        try:
            # Use DNN detector first, fall back to Haar if needed
            faces = self._detect_faces_dnn(frame) if self.face_net else self._detect_faces_haar(frame)
            
            if not faces:
                return None
            
            return {
                'timestamp': timestamp,
                'faces': faces
            }
            
        except Exception as e:
            logger.error(f"Frame analysis error: {str(e)}")
            return None

    async def _extract_frames(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        sample_interval: float
    ) -> List[Tuple[np.ndarray, float]]:
        """Extract frames from video segment."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        try:
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
            end_time = start_time + duration
            current_time = start_time
            
            while current_time < end_time:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append((frame, current_time))
                
                # Skip to next sample interval
                current_time += sample_interval
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                
        finally:
            cap.release()
            
        return frames

    def compare_faces(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compare two face feature vectors."""
        # Normalize features
        features1 = features1 / np.linalg.norm(features1)
        features2 = features2 / np.linalg.norm(features2)
        
        # Calculate cosine similarity
        similarity = np.dot(features1, features2)
        
        return float(similarity)

    async def train_character_faces(
        self,
        character_images: Dict[str, List[str]],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[np.ndarray]]:
        """Train face features for known characters."""
        trained_faces = {}
        total_characters = len(character_images)
        
        for i, (character_name, image_paths) in enumerate(character_images.items()):
            try:
                features = []
                for image_path in image_paths:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    faces = self._detect_faces_dnn(image) if self.face_net else self._detect_faces_haar(image)
                    if faces:
                        features.append(faces[0]['features'])
                
                if features:
                    trained_faces[character_name] = features
                    self.known_faces[character_name] = features
                
                if progress_callback:
                    progress = ((i + 1) / total_characters) * 100
                    await progress_callback(f"Trained {character_name}", progress)
                    
            except Exception as e:
                logger.error(f"Error training {character_name}: {str(e)}")
        
        return trained_faces

    def _get_cache_path(self, video_path: str, start_time: float, duration: float) -> Path:
        """Generate cache file path for video segment."""
        video_name = Path(video_path).stem
        cache_key = f"{video_name}_{start_time:.2f}_{duration:.2f}"
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is valid and not expired."""
        if not cache_path.exists():
            return False
        
        if not self.enable_caching:
            return False
        
        # Check if cache is not expired
        cache_time = cache_path.stat().st_mtime
        current_time = time.time()
        
        return (current_time - cache_time) < self.cache_ttl
    
    def _save_to_cache(self, cache_path: Path, data: Any):
        """Save data to cache file."""
        if not self.enable_caching:
            return
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved face detection cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {str(e)}")
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Any]:
        """Load data from cache file."""
        if not self.enable_caching or not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded face detection cache: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {str(e)}")
            return None

# Global detector instance
detector = FaceDetector() 