import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

from config import settings

# Define missing exception
class FaceDetectionError(Exception):
    pass

logger = logging.getLogger(__name__)

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available - using OpenCV fallback only")

class FaceDetector:
    """Enhanced face detection and recognition with InsightFace."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing InsightFace...")
        
        # CRITICAL FIX: Initialize known_faces attribute
        self.known_faces = {}
        self.face_cascade = None

        try:
            self.app = insightface.app.FaceAnalysis(
                name=getattr(settings, 'INSIGHTFACE_MODEL_NAME', 'buffalo_l'),
                providers=["CPUExecutionProvider"]  # or ["CUDAExecutionProvider"] if you have GPU
            )
            self.app.prepare(ctx_id=0, det_size=getattr(settings, 'INSIGHTFACE_DET_SIZE', (640, 640)))
            self.logger.info("✅ InsightFace initialized successfully.")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize InsightFace: {str(e)}")
            raise FaceDetectionError(f"InsightFace initialization failed: {e}")
    
    def _detect_faces_insightface(self, frame):
        if self.app is None:
            raise RuntimeError("InsightFace not initialized")
        try:
            faces = self.app.get(frame)
            # Convert InsightFace format to our expected format
            result = []
            for face in faces:
                result.append({
                    'bbox': face.bbox.astype(int).tolist(),
                    'confidence': face.det_score,
                    'embedding': face.embedding,
                    'quality_score': face.det_score,
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None
                })
            return result
        except Exception as e:
            self.logger.error(f"InsightFace detection error: {str(e)}")
            raise
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback face detection using OpenCV."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            detected_faces = []
            for (x, y, w, h) in faces:
                detected_faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.8,
                    'embedding': None,  # OpenCV doesn't provide embeddings
                    'quality_score': 0.5,
                    'landmarks': None
                })
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {str(e)}")
            return []
    
    def _calculate_face_quality(self, face) -> float:
        """Calculate face quality score."""
        try:
            # Simple quality assessment based on detection confidence
            quality = face.det_score
            
            # Additional quality factors could be added here
            # - Face size
            # - Blur detection
            # - Lighting assessment
            # - Pose estimation
            
            return quality
            
        except Exception:
            return 0.5
    
    async def train_character_faces(self, character_images: Dict[str, List[str]], progress_callback=None) -> Dict[str, List[np.ndarray]]:
        """Train face recognition model with character images."""
        try:
            logger.info(f"Training face recognition for {len(character_images)} characters")
            
            # CRITICAL FIX: Store trained embeddings in self.known_faces
            self.known_faces = {}
            
            for character_name, image_paths in character_images.items():
                try:
                    logger.info(f"Training for character: {character_name}")
                    
                    # Load and process character images
                    embeddings = []
                    for image_path in image_paths[:5]:  # Limit to 5 images per character
                        try:
                            # Load image
                            image = cv2.imread(image_path)
                            if image is None:
                                continue
                            
                            # Detect faces in image
                            faces = self._detect_faces_insightface(image)
                            
                            # Get embeddings for detected faces
                            for face in faces:
                                embedding = face.get('embedding')
                                if embedding is not None:
                                    embeddings.append(embedding)
                            
                            if progress_callback:
                                progress_callback(f"Processed image for {character_name}")
                                
                        except Exception as e:
                            logger.error(f"Error processing image {image_path}: {str(e)}")
                            continue
                    
                    if embeddings:
                        # CRITICAL FIX: Store in self.known_faces for later use
                        self.known_faces[character_name] = embeddings
                        logger.info(f"Trained {len(embeddings)} embeddings for {character_name}")
                    else:
                        logger.warning(f"No valid embeddings found for {character_name}")
                        
                except Exception as e:
                    logger.error(f"Error training character {character_name}: {str(e)}")
                    continue
            
            logger.info(f"Face recognition training completed for {len(self.known_faces)} characters")
            return self.known_faces
            
        except Exception as e:
            logger.error(f"Face recognition training error: {str(e)}")
            return {}
    
    async def _download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image from URL."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.read()
                        nparr = np.frombuffer(data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        return image
            return None
            
        except Exception as e:
            logger.error(f"Failed to download image {url}: {str(e)}")
            return None
    
    def identify_character(self, face_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Identify a character from face embedding."""
        if not self.known_faces or face_embedding is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for character_name, character_embeddings in self.known_faces.items():
            for embedding in character_embeddings:
                if embedding is not None:
                    similarity = self.compare_faces(face_embedding, embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = character_name
        
        # Use default threshold if not set
        threshold = getattr(settings, 'FACE_SIMILARITY_THRESHOLD', 0.6)
        if best_similarity >= threshold:
            return best_match, best_similarity
        
        return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity."""
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Face comparison failed: {str(e)}")
            return 0.0
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in a frame (compatibility method)."""
        faces = self._detect_faces_insightface(frame)
        
        # Convert to expected format
        result = []
        for face in faces:
            result.append({
                "bbox": face['bbox'],
                "confidence": face['confidence'],
                "center": (face['bbox'][0] + face['bbox'][2]//2, face['bbox'][1] + face['bbox'][3]//2),
                "embedding": face['embedding']
            })
        
        return result 

    async def process_video_segment(self, video_path: str, start_time: float, duration: float, sample_interval: float = 10.0) -> List[Dict[str, Any]]:
        """Process video segment with SMART sampling - not every frame."""
        try:
            logger.info(f"Processing video segment: {video_path}")
            
            scenes = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return scenes
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            
            # SMART SAMPLING: Only process key moments
            # - Sample every 10 seconds (not every frame)
            # - Process beginning, middle, end
            # - Skip most frames entirely
            
            # Limit processing to first 5 minutes maximum
            MAX_PROCESSING_DURATION = 300.0  # 5 minutes
            duration = min(duration, MAX_PROCESSING_DURATION)
            
            logger.info(f"Video: {video_duration:.2f}s, Processing: {start_time:.2f}s to {start_time + duration:.2f}s")
            logger.info(f"SMART SAMPLING: Every {sample_interval} seconds (not every frame)")
            
            # Calculate frame range
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            
            current_scene = {
                "start_time": start_time,
                "end_time": start_time + duration,
                "duration": duration,
                "video_path": video_path,
                "faces": []
            }
            
            # SMART SAMPLING: Only process every N seconds
            frames_to_process = []
            
            # Add key moments: beginning, middle, end
            key_timestamps = [
                start_time,  # Beginning
                start_time + duration * 0.25,  # Quarter
                start_time + duration * 0.5,   # Middle
                start_time + duration * 0.75,  # Three quarters
                start_time + duration - 10     # End (last 10 seconds)
            ]
            
            # Add regular intervals
            current_time = start_time
            while current_time < start_time + duration:
                frames_to_process.append(int(current_time * fps))
                current_time += sample_interval
            
            # Add key timestamps
            for timestamp in key_timestamps:
                frame = int(timestamp * fps)
                if start_frame <= frame < end_frame:
                    frames_to_process.append(frame)
            
            # Remove duplicates and sort
            frames_to_process = sorted(list(set(frames_to_process)))
            
            logger.info(f"SMART SAMPLING: Processing {len(frames_to_process)} frames out of {end_frame - start_frame} total frames")
            
            # Process only the selected frames
            for i, frame_number in enumerate(frames_to_process):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_number / fps
                    
                    # CRITICAL FIX: Use InsightFace instead of OpenCV fallback
                    try:
                        faces = self._detect_faces_insightface(frame)
                    except Exception as e:
                        logger.warning(f"InsightFace detection failed for frame {frame_number}, using OpenCV fallback: {e}")
                        faces = self._detect_faces_opencv_fallback(frame)
                    
                    if faces:
                        for face in faces:
                            # CRITICAL FIX: Proper character identification
                            character_result = None
                            if face.get('embedding') is not None:
                                character_result = self.identify_character(face['embedding'])
                            
                            character_name = None
                            if character_result:
                                character_name, confidence = character_result
                            
                            face_data = {
                                "timestamp": timestamp,
                                "bbox": face["bbox"],
                                "confidence": face["confidence"],
                                "character": character_name,
                                "embedding": face.get("embedding")
                            }
                            current_scene["faces"].append(face_data)
                    
                    # Progress update
                    if i % 5 == 0:
                        progress = (i / len(frames_to_process)) * 100
                        logger.info(f"Video processing: {progress:.1f}% ({i+1}/{len(frames_to_process)} frames)")
            
            cap.release()
            
            if current_scene["faces"]:
                scenes.append(current_scene)
                logger.info(f"Found {len(current_scene['faces'])} faces in video segment")
            
            return scenes
            
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            return [] 

    def _detect_faces_opencv_fallback(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """OpenCV fallback for face detection."""
        try:
            # Initialize cascade if not already done
            if not hasattr(self, 'face_cascade') or self.face_cascade is None:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
                if self.face_cascade.empty():
                    logger.error("Failed to load OpenCV face cascade")
                    return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            detected_faces = []
            for (x, y, w, h) in faces:
                detected_faces.append({
                    "bbox": (x, y, w, h),
                    "confidence": 0.8,
                    "embedding": None
                })
            
            if detected_faces:
                logger.info(f"OpenCV detected {len(detected_faces)} faces")
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"OpenCV face detection error: {str(e)}")
            return [] 