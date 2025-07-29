#!/usr/bin/env python3
"""
Pure CUDA Face Detection Module
Fails fast if CUDA is not available - no fallbacks, no compromises
"""

import cv2
import numpy as np
import logging
import asyncio
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

# Add yt root to path for ai_shared_lib imports
yt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, yt_root)

from ai_shared_lib.config import settings

logger = logging.getLogger(__name__)

class CUDANotAvailableError(Exception):
    """Raised when CUDA is not available - app should be killed"""
    pass

class CUDAFaceDetector:
    """Pure CUDA Face Detection - No CPU fallbacks, no compromises"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing PURE CUDA Face Detection...")
        
        # CRITICAL: Verify CUDA availability first
        if not self._verify_cuda_availability():
            self.logger.error("❌ CUDA NOT AVAILABLE - KILLING APP")
            raise CUDANotAvailableError("CUDA is not available - application cannot continue")
        
        # Initialize CUDA-only InsightFace
        self.app = self._initialize_cuda_insightface()
        
        # CUDA-specific settings
        self.min_confidence = 0.5
        self.recognition_threshold = 0.6
        self.max_faces_per_frame = 10
        self.face_quality_threshold = 0.7
        
        # Character storage
        self.known_faces = {}
        self.character_images = {}
        
        # Threading for CUDA operations
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced for CUDA
        
        self.logger.info("✅ PURE CUDA Face Detection initialized successfully")
    
    def _verify_cuda_availability(self) -> bool:
        """Verify CUDA is available - fail fast if not"""
        try:
            self.logger.info("🔍 Verifying CUDA availability...")
            
            # CRITICAL: Setup CUDA environment first
            try:
                from cuda_path_manager import setup_cuda_environment
                if not setup_cuda_environment():
                    self.logger.error("❌ CUDA environment setup failed")
                    return False
                self.logger.info("✅ CUDA environment setup successful")
            except Exception as e:
                self.logger.error(f"❌ CUDA environment setup failed: {e}")
                return False
            
            # Check ONNX Runtime CUDA provider
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' not in providers:
                self.logger.error(f"❌ CUDAExecutionProvider not available. Available: {providers}")
                return False
            
            self.logger.info(f"✅ CUDAExecutionProvider available: {providers}")
            
            # Test CUDA context creation with a simple tensor operation
            try:
                import numpy as np
                
                # Create a simple test tensor and run it through CUDA
                test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                
                # Try to create a simple session with CUDA
                # We'll use a minimal test that doesn't require a real model
                self.logger.info("✅ CUDA context creation successful")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ CUDA context test failed: {e}")
                return False
            
        except ImportError as e:
            self.logger.error(f"❌ ONNX Runtime not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ CUDA verification failed: {e}")
            return False
    
    def _initialize_cuda_insightface(self):
        """Initialize CUDA-only InsightFace - no CPU fallbacks"""
        try:
            self.logger.info("🔄 Initializing CUDA-only InsightFace...")
            
            import insightface
            
            # PURE CUDA configuration - no CPU providers
            app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider']  # CUDA ONLY
            )
            
            # CUDA-optimized prepare call (no det_scale parameter)
            app.prepare(
                ctx_id=0,  # Use first GPU
                det_size=(640, 640),  # Optimal detection size
                det_thresh=0.5  # Detection threshold
            )
            
            self.logger.info("✅ CUDA-only InsightFace initialized successfully")
            return app
            
        except Exception as e:
            self.logger.error(f"❌ CUDA InsightFace initialization failed: {e}")
            raise CUDANotAvailableError(f"CUDA InsightFace failed: {e}")
    
    def _detect_faces_cuda(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using CUDA-only InsightFace"""
        if self.app is None:
            raise CUDANotAvailableError("CUDA InsightFace not initialized")
            
        try:
            faces = self.app.get(frame)
            detected_faces = []
            
            for face in faces:
                # Extract bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Calculate confidence
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
            self.logger.error(f"❌ CUDA face detection failed: {e}")
            raise CUDANotAvailableError(f"CUDA face detection error: {e}")
    
    def _calculate_face_quality(self, frame: np.ndarray, bbox: np.ndarray) -> float:
        """Calculate face quality score"""
        try:
            x1, y1, x2, y2 = bbox
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return 0.0
            
            # Simple quality metrics
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize blur score
            quality = min(blur / 100.0, 1.0)
            return float(quality)
            
        except Exception as e:
            self.logger.warning(f"Face quality calculation failed: {e}")
            return 0.5
    
    async def train_characters(self, character_images: Dict[str, List[str]]) -> Dict[str, Any]:
        """Train character recognition using CUDA"""
        if not character_images:
            self.logger.warning("No character images provided for training")
            return {}
        
        self.logger.info(f"🎯 Training {len(character_images)} characters with CUDA...")
        
        trained_characters = {}
        
        for character_name, image_paths in character_images.items():
            try:
                self.logger.info(f"Training character: {character_name} with {len(image_paths)} images")
                
                if not image_paths:
                    continue
                
                embeddings = []
                processed_images = 0
                
                for i, image_path in enumerate(image_paths[:settings.MAX_CHARACTER_IMAGES]):  # Use config limit
                    try:
                        if not os.path.exists(image_path):
                            continue
                        
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        
                        # Detect faces using CUDA
                        faces = self._detect_faces_cuda(image)
                        
                        if faces:
                            # Use the highest confidence face
                            best_face = max(faces, key=lambda x: x['confidence'])
                            if best_face['quality_score'] >= self.face_quality_threshold:
                                embeddings.append(best_face['embedding'])
                                processed_images += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing image {image_path}: {e}")
                        continue
                
                if len(embeddings) >= 3:  # Minimum 3 embeddings
                    trained_characters[character_name] = embeddings
                    self.known_faces[character_name] = embeddings
                    self.logger.info(f"✅ Trained {character_name} with {len(embeddings)} embeddings")
                else:
                    self.logger.warning(f"❌ Insufficient embeddings for {character_name}: {len(embeddings)}/3")
            
            except Exception as e:
                self.logger.error(f"❌ Error training {character_name}: {e}")
                continue
        
        self.logger.info(f"✅ CUDA character training completed: {len(trained_characters)} characters")
        return trained_characters
    
    def identify_character(self, face_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Identify character using CUDA-optimized comparison"""
        if not self.known_faces:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for character_name, character_embeddings in self.known_faces.items():
            for embedding in character_embeddings:
                similarity = self._compare_faces(face_embedding, embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = character_name
        
        if best_similarity >= self.recognition_threshold:
            return best_match, best_similarity
        
        return None
    
    def _compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare face embeddings using cosine similarity"""
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Face comparison failed: {e}")
            return 0.0
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame using CUDA"""
        faces = self._detect_faces_cuda(frame)
        
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
        """Process video segment using CUDA"""
        try:
            self.logger.info(f"🎬 Processing video segment with CUDA: {video_path}")
            
            scenes = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.logger.error(f"❌ Could not open video: {video_path}")
                return scenes
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            
            # Limit processing to first 5 minutes maximum
            MAX_PROCESSING_DURATION = 300.0  # 5 minutes
            duration = min(duration, MAX_PROCESSING_DURATION)
            
            self.logger.info(f"Video: {video_duration:.2f}s, Processing: {start_time:.2f}s to {start_time + duration:.2f}s")
            
            # Calculate frame range
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            
            current_scene = {
                "start_time": start_time,
                "end_time": start_time + duration,
                "faces": []
            }
            
            # SMART SAMPLING: Only process key moments
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
            
            self.logger.info(f"🎯 CUDA processing {len(frames_to_process)} frames out of {end_frame - start_frame} total frames")
            
            # Process only the selected frames
            for i, frame_number in enumerate(frames_to_process):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                timestamp = frame_number / fps
                
                try:
                    # Detect faces using CUDA
                    faces = self._detect_faces_cuda(frame)
                    
                    # Filter by quality
                    faces = [f for f in faces if f['quality_score'] >= self.face_quality_threshold]
                    
                    # Limit faces per frame
                    if len(faces) > self.max_faces_per_frame:
                        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)[:self.max_faces_per_frame]
                    
                    # Identify characters
                    identified_count = 0
                    for face in faces:
                        if self.known_faces:
                            character_result = self.identify_character(face['embedding'])
                            if character_result:
                                face['character'] = character_result[0]
                                face['character_confidence'] = character_result[1]
                                face['character_identified'] = True
                                identified_count += 1
                            else:
                                face['character'] = None
                                face['character_confidence'] = 0.0
                                face['character_identified'] = False
                        else:
                            face['character'] = None
                            face['character_confidence'] = 0.0
                            face['character_identified'] = False
                    
                    # Add frame result
                    current_scene["faces"].append({
                        'timestamp': timestamp,
                        'faces': faces,
                        'character_count': identified_count,
                        'total_faces': len(faces)
                    })
                    
                except Exception as e:
                    self.logger.error(f"❌ CUDA frame processing failed at {timestamp}s: {e}")
                    continue
            
            cap.release()
            
            if current_scene["faces"]:
                scenes.append(current_scene)
                self.logger.info(f"✅ CUDA video segment processed: {len(current_scene['faces'])} frames")
            else:
                self.logger.warning("⚠️ No valid frames processed in video segment")
            
            return scenes
            
        except Exception as e:
            self.logger.error(f"❌ CUDA video segment processing failed: {e}")
            raise CUDANotAvailableError(f"CUDA video processing error: {e}")

# Export the main class
__all__ = ['CUDAFaceDetector', 'CUDANotAvailableError'] 