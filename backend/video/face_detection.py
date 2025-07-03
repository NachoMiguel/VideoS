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
import insightface

# Handle both relative and absolute imports
try:
    from ..core.config import settings
    from ..core.exceptions import VideoProcessingError, FaceDetectionError
    from ..core.parallel import parallel_processor, parallel_task
except ImportError:
    # Fallback for absolute imports
    try:
        from core.config import settings
        from core.exceptions import VideoProcessingError, FaceDetectionError
        from core.parallel import parallel_processor, parallel_task
    except ImportError:
        # Create minimal fallback classes if imports fail
        class VideoProcessingError(Exception):
            pass
        
        class FaceDetectionError(Exception):
            pass
        
        # Create a minimal settings object
        class Settings:
            cache_dir = "cache"
            parallel_face_detection = True
            max_workers = 4
            face_detection_batch_size = 10
            insightface_det_thresh = 0.5
            insightface_rec_thresh = 0.6
            insightface_model_name = "buffalo_l"
            insightface_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            insightface_det_size = (640, 640)
            insightface_face_align = True
            insightface_gpu_memory_fraction = 0.5
            enable_caching = True
            cache_ttl_hours = 24
            enable_face_quality_filter = True
            face_quality_threshold = 0.3
            max_faces_per_frame = 10
            min_character_images = 3
            max_character_images = 20
            face_similarity_threshold = 0.6
        
        settings = Settings()
        
        def parallel_task(task_type):
            def decorator(func):
                return func
            return decorator
        
        class ParallelProcessor:
            async def execute_parallel_tasks(self, tasks):
                return []
        
        parallel_processor = ParallelProcessor()

logger = logging.getLogger(__name__)

class FaceDetector:
    """Enhanced face detection and recognition with InsightFace."""
    
    def __init__(self):
        self.cache_dir = Path(settings.cache_dir) / "faces"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Parallel processing settings
        self.parallel_enabled = settings.parallel_face_detection
        self.max_workers = settings.max_workers
        self.batch_size = settings.face_detection_batch_size
        
        # Face detection settings
        self.min_confidence = settings.insightface_det_thresh
        self.recognition_threshold = settings.insightface_rec_thresh
        
        # Initialize InsightFace models
        self.app = None
        self._initialize_insightface()
        
        # Caching
        self.enable_caching = settings.enable_caching
        self.cache_ttl = settings.cache_ttl_hours * 3600  # Convert to seconds
        
        # Known characters storage
        self.known_faces = {}  # character_name -> face_embeddings
        self.character_images = {}  # character_name -> image_paths
        
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        
        logger.info(f"Face detector initialized: parallel={self.parallel_enabled}, caching={self.enable_caching}")

    def _initialize_insightface(self):
        """Initialize InsightFace models."""
        try:
            self.app = insightface.app.FaceAnalysis(
                name=settings.insightface_model_name,
                providers=settings.insightface_providers
            )
            self.app.prepare(ctx_id=0, det_size=settings.insightface_det_size)
            logger.info(f"InsightFace initialized with model: {settings.insightface_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {str(e)}")
            raise FaceDetectionError(f"InsightFace initialization failed: {str(e)}")

    def _detect_faces_insightface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using InsightFace."""
        if self.app is None:
            raise FaceDetectionError("InsightFace not initialized")
            
        try:
            faces = self.app.get(frame)
            detected_faces = []
            
            for face in faces:
                # Extract bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Calculate confidence and quality metrics
                confidence = float(face.det_score)
                
                if confidence >= self.min_confidence:
                    # Extract face embedding
                    embedding = face.normed_embedding
                    
                    # Calculate face quality score
                    quality_score = self._calculate_face_quality(frame, bbox)
                    
                    detected_faces.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'confidence': confidence,
                        'embedding': embedding,
                        'quality_score': quality_score,
                        'landmarks': face.kps if hasattr(face, 'kps') else None
                    })
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"InsightFace detection error: {str(e)}")
            return []

    def _calculate_face_quality(self, frame: np.ndarray, bbox: np.ndarray) -> float:
        """Calculate face quality score based on various metrics."""
        x1, y1, x2, y2 = bbox
        face_region = frame[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return 0.0
        
        # Convert to grayscale for quality assessment
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # 2. Contrast
        contrast = np.std(gray)
        
        # 3. Face size score
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(1.0, face_area / (100 * 100))  # Normalize to 100x100 minimum
        
        # 4. Brightness check (avoid overexposed/underexposed faces)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        
        # Combine metrics
        quality_score = (
            0.3 * min(1.0, sharpness / 1000) +
            0.25 * min(1.0, contrast / 100) +
            0.25 * size_score +
            0.2 * brightness_score
        )
        
        return float(quality_score)

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
            # Use InsightFace for detection and recognition
            faces = self._detect_faces_insightface(frame)
            
            if not faces:
                return None
            
            # Filter faces by quality if enabled
            if settings.enable_face_quality_filter:
                faces = [f for f in faces if f['quality_score'] >= settings.face_quality_threshold]
            
            # Limit number of faces per frame
            if len(faces) > settings.max_faces_per_frame:
                faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)[:settings.max_faces_per_frame]
            
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

    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity."""
        # Normalize embeddings (InsightFace embeddings are already normalized)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)

    async def train_character_faces(
        self,
        character_images: Dict[str, List[str]],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[np.ndarray]]:
        """Train face embeddings for known characters."""
        trained_faces = {}
        total_characters = len(character_images)
        
        for i, (character_name, image_paths) in enumerate(character_images.items()):
            try:
                embeddings = []
                valid_images = 0
                
                for image_path in image_paths:
                    if valid_images >= settings.max_character_images:
                        break
                        
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    faces = self._detect_faces_insightface(image)
                    if faces:
                        # Use the highest confidence face
                        best_face = max(faces, key=lambda x: x['confidence'])
                        if best_face['quality_score'] >= settings.face_quality_threshold:
                            embeddings.append(best_face['embedding'])
                            valid_images += 1
                
                if len(embeddings) >= settings.min_character_images:
                    trained_faces[character_name] = embeddings
                    self.known_faces[character_name] = embeddings
                    logger.info(f"Trained {character_name} with {len(embeddings)} face embeddings")
                else:
                    logger.warning(f"Insufficient quality images for {character_name}: {len(embeddings)}/{settings.min_character_images}")
                
                if progress_callback:
                    progress = ((i + 1) / total_characters) * 100
                    await progress_callback(f"Trained {character_name}", progress)
                    
            except Exception as e:
                logger.error(f"Error training {character_name}: {str(e)}")
        
        return trained_faces

    def identify_character(self, face_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Identify a character from face embedding."""
        best_match = None
        best_similarity = 0.0
        
        for character_name, character_embeddings in self.known_faces.items():
            for embedding in character_embeddings:
                similarity = self.compare_faces(face_embedding, embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = character_name
        
        if best_similarity >= settings.face_similarity_threshold:
            return best_match, best_similarity
        
        return None

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