import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import concurrent.futures
from pathlib import Path
import logging
from core.parallel import ParallelProcessor
from core.exceptions import VideoProcessingError, FaceDetectionError, ValidationError
from core.config import VideoProcessingConfig, settings
from core.session import SessionManager
import os
from core.credit_manager import credit_manager, ServiceType
from core.parallel_error_handler import parallel_error_handler, OperationType, OperationResult
from video.face_detection import detector as main_face_detector

class VideoProcessor:
    def __init__(self, config: VideoProcessingConfig, session_manager: SessionManager):
        self.config = config
        self.session_manager = session_manager
        self.face_detector = main_face_detector  # Use the main InsightFace detector
        self.parallel_processor = ParallelProcessor(max_workers=config.max_workers)
        self.logger = logging.getLogger(__name__)

    async def process_video(self, video_path: str, session_id: str) -> Dict:
        """Process a video file with parallel frame processing and face detection."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Failed to open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Update session with video metadata
            await self.session_manager.update_session(session_id, {
                'total_frames': total_frames,
                'fps': fps,
                'status': 'processing'
            })

            frames_data = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                frame_futures = []
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    future = executor.submit(self._process_frame, frame, frame_count)
                    frame_futures.append(future)
                    frame_count += 1

                    # Process results in batches
                    if len(frame_futures) >= self.config.batch_size:
                        frames_data.extend(self._process_futures_batch(frame_futures))
                        frame_futures = []
                        
                        # Update progress
                        progress = (frame_count / total_frames) * 100
                        await self.session_manager.update_session(session_id, {
                            'progress': progress
                        })

                # Process remaining frames
                if frame_futures:
                    frames_data.extend(self._process_futures_batch(frame_futures))

            cap.release()

            # Update session with completion status
            await self.session_manager.update_session(session_id, {
                'status': 'completed',
                'progress': 100
            })

            return {
                'frames_processed': len(frames_data),
                'frames_data': frames_data,
                'video_metadata': {
                    'fps': fps,
                    'total_frames': total_frames
                }
            }

        except Exception as e:
            await self.session_manager.update_session(session_id, {
                'status': 'error',
                'error': str(e)
            })
            raise VideoProcessingError(f"Video processing failed: {str(e)}")

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """Process a single frame with face detection and analysis."""
        try:
            # Detect faces in the frame
            faces = self.face_detector.detect_faces(frame)
            
            # Analyze frame content
            frame_data = {
                'frame_number': frame_number,
                'faces': faces,
                'timestamp': frame_number / self.config.fps if hasattr(self.config, 'fps') else 0,
                'has_faces': len(faces) > 0
            }

            return frame_data

        except Exception as e:
            self.logger.error(f"Frame processing failed for frame {frame_number}: {str(e)}")
            return {
                'frame_number': frame_number,
                'error': str(e)
            }

    def _process_futures_batch(self, futures: List[concurrent.futures.Future]) -> List[Dict]:
        """Process a batch of frame futures and return their results."""
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process frame batch: {str(e)}")
        return sorted(results, key=lambda x: x['frame_number'])

    async def extract_scene(self, video_path: str, start_time: float, end_time: float, output_path: str) -> str:
        """Extract a scene from the video between start_time and end_time."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Failed to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame

            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current_frame += 1

            cap.release()
            out.release()

            return output_path

        except Exception as e:
            raise VideoProcessingError(f"Scene extraction failed: {str(e)}")

    async def process_video_pipeline(self, session_id: str, video_path: str, script_content: str, 
                                   websocket_manager=None, test_mode: bool = False, 
                                   test_settings: dict = None) -> dict:
        """
        Complete video processing pipeline with enhanced error handling and credit management.
        """
        try:
            # Initialize error handling and credit management for this session
            parallel_error_handler.start_session(session_id)
            credit_manager.start_session(session_id)
            
            logger.info(f"Starting enhanced video processing pipeline for session {session_id}")
            
            # Define all parallel operations
            operations = []
            
            # Phase 1: Character Extraction (35-45%)
            if not test_mode or not test_settings.get('useKnownCharacters', False):
                operations.append((
                    OperationType.CHARACTER_EXTRACTION,
                    self._extract_characters_with_error_handling,
                    (script_content,),
                    {'session_id': session_id}
                ))
            
            # Execute character extraction first (required for other operations)
            if operations:
                character_results = await parallel_error_handler.execute_parallel_operations(operations)
                characters = character_results.get(OperationType.CHARACTER_EXTRACTION, OperationResult(
                    operation_type=OperationType.CHARACTER_EXTRACTION,
                    success=True,
                    result=['Jean Claude Vandamme', 'Steven Seagal']  # Fallback
                )).result
            else:
                characters = ['Jean Claude Vandamme', 'Steven Seagal']  # Test mode fallback
            
            if websocket_manager:
                await websocket_manager.send_progress(session_id, 40, "Character extraction completed")
            
            # Phase 2: Parallel Operations (45-75%)
            parallel_operations = []
            
            # Image Search (45-55%)
            if not test_mode or not test_settings.get('useKnownCharacters', False):
                parallel_operations.append((
                    OperationType.IMAGE_SEARCH,
                    self._search_character_images_with_error_handling,
                    (characters,),
                    {'session_id': session_id}
                ))
            
            # Face Training (55-65%)
            parallel_operations.append((
                OperationType.FACE_TRAINING,
                self._train_face_recognition_with_error_handling,
                (characters,),
                {'session_id': session_id}
            ))
            
            # Audio Generation (55-75%)
            if not test_mode or not test_settings.get('use_saved_audio', False):
                parallel_operations.append((
                    OperationType.AUDIO_GENERATION,
                    self._generate_audio_with_error_handling,
                    (script_content,),
                    {'session_id': session_id}
                ))
            
            # Execute parallel operations
            parallel_results = await parallel_error_handler.execute_parallel_operations(parallel_operations)
            
            # Extract results
            character_images = parallel_results.get(OperationType.IMAGE_SEARCH, OperationResult(
                operation_type=OperationType.IMAGE_SEARCH,
                success=True,
                result={}
            )).result or {}
            
            face_training_result = parallel_results.get(OperationType.FACE_TRAINING, OperationResult(
                operation_type=OperationType.FACE_TRAINING,
                success=True,
                result=True
            )).result
            
            audio_path = parallel_results.get(OperationType.AUDIO_GENERATION, OperationResult(
                operation_type=OperationType.AUDIO_GENERATION,
                success=True,
                result=None
            )).result
            
            if websocket_manager:
                await websocket_manager.send_progress(session_id, 75, "Parallel operations completed")
            
            # Phase 3: Scene Analysis (75-85%)
            scene_analysis_result = await parallel_error_handler.execute_with_error_handling(
                OperationType.SCENE_ANALYSIS,
                self._analyze_scenes_with_error_handling,
                video_path,
                session_id=session_id
            )
            
            scenes = scene_analysis_result.result if scene_analysis_result.success else []
            
            if websocket_manager:
                await websocket_manager.send_progress(session_id, 85, "Scene analysis completed")
            
            # Phase 4: Scene Selection (85-90%)
            scene_selection_result = await parallel_error_handler.execute_with_error_handling(
                OperationType.SCENE_SELECTION,
                self._select_best_scenes_with_error_handling,
                scenes, script_content,
                session_id=session_id
            )
            
            selected_scenes = scene_selection_result.result if scene_selection_result.success else scenes[:10]
            
            if websocket_manager:
                await websocket_manager.send_progress(session_id, 90, "Scene selection completed")
            
            # Phase 5: Video Assembly (90-100%)
            assembly_result = await parallel_error_handler.execute_with_error_handling(
                OperationType.VIDEO_ASSEMBLY,
                self._assemble_video_with_error_handling,
                selected_scenes, audio_path,
                session_id=session_id
            )
            
            final_video_path = assembly_result.result if assembly_result.success else None
            
            if websocket_manager:
                await websocket_manager.send_progress(session_id, 100, "Video assembly completed")
            
            # Generate session summary
            processing_summary = parallel_error_handler.get_session_summary()
            credit_summary = credit_manager.get_session_usage_summary()
            
            # Save session logs
            await credit_manager.save_session_log()
            
            # Prepare final result
            result = {
                'success': True,
                'video_path': final_video_path,
                'characters': characters,
                'character_images': character_images,
                'scenes': selected_scenes,
                'audio_path': audio_path,
                'processing_summary': processing_summary,
                'credit_usage': credit_summary,
                'session_id': session_id
            }
            
            logger.info(f"✅ Enhanced video processing pipeline completed successfully for session {session_id}")
            return result
            
        except CreditExhaustionError as e:
            error_message = f"Processing stopped due to credit exhaustion: {str(e)}"
            logger.error(error_message)
            
            if websocket_manager:
                await websocket_manager.send_error(session_id, error_message)
            
            return {
                'success': False,
                'error': error_message,
                'error_type': 'credit_exhausted',
                'session_id': session_id
            }
            
        except Exception as e:
            error_message = f"Enhanced pipeline processing failed: {str(e)}"
            logger.error(error_message)
            
            if websocket_manager:
                await websocket_manager.send_error(session_id, error_message)
            
            # Clean up resources
            await parallel_error_handler.cleanup_resources()
            
            return {
                'success': False,
                'error': error_message,
                'session_id': session_id
            }
    
    async def _extract_characters_with_error_handling(self, script_content: str, session_id: str, account_info=None):
        """Character extraction with error handling wrapper."""
        from services.openai import openai_service
        return await openai_service.extract_characters_from_script(script_content, account_info)
    
    async def _search_character_images_with_error_handling(self, characters: List[str], session_id: str, account_info=None):
        """Character image search with error handling wrapper."""
        from services.image_search import image_search_service
        return await image_search_service.search_character_images(characters)
    
    async def _train_face_recognition_with_error_handling(self, characters: List[str], session_id: str):
        """Face recognition training with error handling wrapper."""
        # This is a local operation, so no account needed
        return await self.face_detector.train_face_recognition_async(characters)
    
    async def _generate_audio_with_error_handling(self, script_content: str, session_id: str, account_info=None):
        """Audio generation with error handling wrapper."""
        from services.elevenlabs import elevenlabs_service
        output_path = f"output/audio_{session_id}.mp3"
        return await elevenlabs_service.generate_audio_from_script(script_content, output_path)
    
    async def _analyze_scenes_with_error_handling(self, video_path: str, session_id: str):
        """Scene analysis with error handling wrapper."""
        return await self.extract_scenes(video_path)
    
    async def _select_best_scenes_with_error_handling(self, scenes: List[dict], script_content: str, session_id: str):
        """Scene selection with error handling wrapper."""
        return await self.select_best_scenes(scenes, script_content)
    
    async def _assemble_video_with_error_handling(self, scenes: List[dict], audio_path: str, session_id: str):
        """Video assembly with error handling wrapper."""
        output_path = f"output/final_video_{session_id}.mp4"
        return await self.create_compilation(scenes, output_path, audio_path)

def validate_video_file(file_path: str) -> bool:
    """Validate video file size and other constraints."""
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > settings.max_video_file_size_bytes:
            raise ValidationError(f"Video file size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum limit of 400MB")
        return True
    except Exception as e:
        raise ValidationError(f"Video file validation failed: {str(e)}") 