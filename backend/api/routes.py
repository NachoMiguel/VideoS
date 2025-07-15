from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json
import uuid
import os
from datetime import datetime
import asyncio
import time
from pathlib import Path
import aiofiles
import shutil

from core.config import settings
from core.logger import logger, api_logger
from core.exceptions import *
from api.websocket import manager
# Temporarily disabled due to InsightFace dependency conflicts
# from video.processor import processor  
# from video.face_detection import detector
import services.openai as openai_module
from services.elevenlabs import elevenlabs_service
from services.image_search import image_search_service
from core.credit_manager import credit_manager, CreditExhaustionError
from core.parallel_error_handler import parallel_error_handler
from core.session import manager as session_manager
from services.youtube import YouTubeService
from services.openai import OpenAIService
from services.elevenlabs import ElevenLabsService
from services.image_search import ImageSearchService
from services.topic_driven_generator import TopicDrivenScriptGenerator

router = APIRouter()

# In-memory session storage (in production, use Redis or database)
active_sessions = {}


# Request models
class ProcessVideoRequest(BaseModel):
    """Request model for video processing endpoint."""
    video_path: str
    script_content: str
    test_mode: bool = False
    test_settings: Optional[Dict[str, Any]] = None

# Session models
class SessionData:
    def __init__(self, session_id: str):
        self.id = session_id
        self.created_at = datetime.now()
        self.status = "initialized"
        self.progress = 0
        self.videos = []
        self.script = None
        self.audio_file = None
        self.characters = []
        self.output_file = None
        self.error = None
        
        # Test mode settings
        self.test_mode = settings.test_mode_enabled
        self.use_saved_script = settings.use_saved_script
        self.use_known_characters = settings.use_known_characters
        self.use_saved_audio = settings.use_saved_audio

# ====== SCRIPT PROCESSING ENDPOINTS ======

@router.post("/extract-transcript")
async def extract_transcript(
    youtube_url: str = Form(...),
    use_default_prompt: bool = Form(True),
    custom_prompt: Optional[str] = Form(None),
    use_saved_script: bool = Form(False)
):
    """Complete working version."""
    try:
        # All the working logic (we know this works)
        youtube_service = YouTubeService()
        video_id = youtube_service.extract_video_id(youtube_url)
        
        session_id = str(uuid.uuid4())
        session_data = {
            "test_mode": use_saved_script and settings.test_mode_enabled,
            "youtube_url": youtube_url,
            "use_default_prompt": use_default_prompt,
            "custom_prompt": custom_prompt,
            "use_saved_script": use_saved_script
        }
        
        initial_data = {
            "status": "extracting_transcript",
            **session_data
        }
        
        session = await session_manager.create_session(session_id=session_id, initial_data=initial_data)
        
        transcript_data = await youtube_service.get_transcript(video_id)
        transcript_text = ""
        for entry in transcript_data:
            transcript_text += entry.get('text', '') + " "
        transcript = transcript_text.strip()
        
        if use_default_prompt:
            prompt = settings.DEFAULT_SCRIPT_REWRITE_PROMPT
        else:
            prompt = custom_prompt
            
        openai_service = openai_module.get_openai_service()
        #  NEW: Pass video_id to generate_script
        rewritten_script = await openai_service.generate_script(transcript, prompt, video_id)
        
        #  NEW: Format script for clean display
        from services.text_cleaner import text_cleaner
        formatted_script = text_cleaner.format_script_for_display(rewritten_script)
        
        # 🎯 NEW: Save the generated script to script_tests folder
        try:
            # Create script_tests directory if it doesn't exist
            script_tests_dir = Path(__file__).parent.parent.parent / "script_tests"
            script_tests_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp and video info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_video_id = video_id.replace('-', '_').replace(' ', '_')[:20]
            filename = f"generated_script_{timestamp}_{safe_video_id}.txt"
            filepath = script_tests_dir / filename
            
            # Save CLEAN script content (no metadata)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(formatted_script)
            
            logger.info(f"✅ Clean script saved to: {filepath}")
            
        except Exception as save_error:
            logger.warning(f"Failed to save script to script_tests folder: {str(save_error)}")
            # Don't fail the entire request if saving fails
        
        # Complete the session update with formatted script
        script_data = {
            "content": formatted_script,  # Use formatted version
            "source": "generated",
            "youtube_url": youtube_url,
            "original_transcript": transcript,
            "prompt_used": prompt,
            "video_id": video_id
        }
        
        await session_manager.update_session(
            session_id=session_id,
            status="script_ready",
            script=script_data
        )
        
        # Get updated session for response
        updated_session = await session_manager.get_session(session_id)
        
        return {
            "session_id": session_id,
            "status": updated_session.status,
            "script": updated_session.metadata.get("script", {}),
            "test_mode": updated_session.metadata.get("test_mode", False)
        }
        
    except Exception as e:
        logger.error(f"Extract transcript error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/modify-script")
async def modify_script(request: Request):
    """Modify script text with context awareness."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        action = data.get("action")
        selected_text = data.get("selected_text")
        context_before = data.get("context_before", "")
        context_after = data.get("context_after", "")
        
        # Get session from session manager
        session = await session_manager.get_session(session_id)
        
        # Use OpenAI service for context-aware modification
        # openai_service = OpenAIService()  # Comment out - use imported global instance
        
        # Check if session is in test mode from metadata
        test_mode = session.metadata.get('test_mode', False)
        
        # Skip API call in test mode
        if test_mode:
            # Mock modification for test mode
            mock_modifications = {
                'shorten': f"[SHORTENED] {selected_text[:len(selected_text)//2]}...",
                'expand': f"{selected_text} [EXPANDED with more detail and examples]",
                'rewrite': f"[REWRITTEN] {selected_text.replace('.', ' with enhanced engagement.')}",
                'make_engaging': f"🔥 {selected_text} - This is absolutely incredible! 🚀",
                'delete': ""
            }
            modified_text = mock_modifications.get(action, selected_text)
        else:
            # Real API call for normal mode
            openai_service = openai_module.get_openai_service()
            modified_text = await openai_service.modify_script_context_aware(
                action=action,
                selected_text=selected_text,
                context_before=context_before,
                context_after=context_after
            )
            
            # Log API usage
            api_logger.log_api_usage("OpenAI", f"script_modify_{action}", 1)
        
        return {
            "success": True,
            "modified_text": modified_text,
            "action": action,
            "original_text": selected_text
        }
        
    except Exception as e:
        logger.error(f"Script modification error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/bulk-modify-script")
async def bulk_modify_script(request: Request):
    """Process multiple script modifications in parallel."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        modifications = data.get("modifications", [])
        
        # Get session from session manager
        session = await session_manager.get_session(session_id)
        
        # Use OpenAI service for bulk modifications
        # openai_service = OpenAIService()  # Comment out - use imported global instance
        
        # Check if session is in test mode from metadata
        test_mode = session.metadata.get('test_mode', False)
        
        if test_mode:
            # Mock bulk modifications for test mode
            results = []
            for i, mod in enumerate(modifications):
                results.append({
                    'id': mod.get('id', f'mod_{i}'),
                    'action': mod['action'],
                    'original_text': mod['selected_text'],
                    'modified_text': f"[TEST] Modified: {mod['selected_text']}",
                    'success': True,
                    'error': None,
                    'execution_time': 0.1
                })
        else:
            # Real bulk processing
            openai_service = openai_module.get_openai_service()
            results = await openai_service.bulk_modify_script(
                modifications=modifications,
                progress_callback=None  # Could add WebSocket progress here
            )
            
            # Log API usage
            api_logger.log_api_usage("OpenAI", "script_bulk_modify", len(modifications))
        
        return {
            "success": True,
            "results": results,
            "total_processed": len(results)
        }
        
    except Exception as e:
        logger.error(f"Bulk script modification error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ====== VIDEO UPLOAD ENDPOINTS ======

@router.post("/upload-videos")
async def upload_videos(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
    use_known_characters: bool = Form(False),
    use_saved_audio: bool = Form(False),
    selected_audio_file: Optional[str] = Form(None)
):
    """Upload and validate video files for processing."""
    try:
        if session_id not in active_sessions:
            raise SessionNotFoundError("Session not found")
        
        session = active_sessions[session_id]
        
        # Validate number of files
        if len(files) > settings.MAX_VIDEO_FILES:
            raise VideoValidationError(
                f"Maximum {settings.MAX_VIDEO_FILES} videos allowed",
                "TOO_MANY_FILES"
            )
        
        session.status = "uploading_videos"
        validated_videos = []
        
        for i, file in enumerate(files):
            await manager.send_progress(
                session_id, 
                int((i / len(files)) * 70), 
                "uploading_videos", 
                f"Validating video {i+1}/{len(files)}..."
            )
            
            # Validate file
            await validate_video_file(file)
            
            # Save file
            file_path = os.path.join(settings.UPLOAD_DIR, f"{session_id}_{file.filename}")
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            video_info = {
                "filename": file.filename,
                "path": file_path,
                "size": len(content),
                "content_type": file.content_type
            }
            validated_videos.append(video_info)
        
        session.videos = validated_videos
        session.use_known_characters = use_known_characters
        session.use_saved_audio = use_saved_audio
        
        if use_saved_audio and selected_audio_file:
            session.audio_file = selected_audio_file
        
        session.status = "videos_uploaded"
        
        await manager.send_progress(session_id, 100, "videos_uploaded", "Videos uploaded successfully!")
        
        return {
            "status": "videos_uploaded",
            "videos_count": len(validated_videos),
            "test_settings": {
                "use_known_characters": use_known_characters,
                "use_saved_audio": use_saved_audio,
                "selected_audio_file": selected_audio_file
            }
        }
        
    except Exception as e:
        logger.error(f"Video upload error: {str(e)}")
        if session_id in active_sessions:
            active_sessions[session_id].error = str(e)
            await manager.send_error(session_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def validate_video_file(file: UploadFile) -> None:
    """Validate video file size."""
    try:
        # Check file size using seek
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset position
        
        if size > settings.max_video_file_size_bytes:
            raise VideoValidationError(
                f"Video file size ({size / 1024 / 1024:.1f}MB) exceeds maximum limit of 400MB"
            )
    except Exception as e:
        raise VideoValidationError(f"Failed to validate video file: {str(e)}")

# ====== PROCESSING ENDPOINTS ======

@router.post("/start-processing")
async def start_processing(session_id: str = Form(...)):
    """Start the complete video processing pipeline."""
    try:
        if session_id not in active_sessions:
            raise SessionNotFoundError("Session not found")
        
        session = active_sessions[session_id]
        
        if not session.script or not session.videos:
            raise HTTPException(status_code=400, detail="Script and videos required")
        
        session.status = "processing"
        session.progress = 0
        
        # Start processing in background
        asyncio.create_task(process_video_pipeline(session_id))
        
        return {
            "status": "processing_started",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Processing start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def analyze_video_scenes(videos: List[str], characters: List[Dict] = None) -> List[Dict]:
    """Analyze video scenes for content and quality with character identification."""
    try:
        results = []
        for video_path in videos:
            # Process video for scenes with character identification
            scenes = await processor.process_videos_parallel([video_path])
            
            # Analyze each scene
            for scene in scenes:
                # Calculate scene quality score
                quality_score = await _calculate_scene_quality(
                    scene,
                    face_weight=0.4,
                    duration_weight=0.3,
                    stability_weight=0.3
                )
                
                # Add quality score to scene data
                scene['quality_score'] = quality_score
                
                # Add character recognition data
                if scene.get('faces'):
                    scene['characters'] = await _identify_characters(scene['faces'])
                    
                    # Add character summary for easy access
                    scene['character_summary'] = {
                        'total_characters': len(scene['characters']),
                        'identified_characters': len([c for c in scene['characters'] if c['character_id'] != 'unknown']),
                        'character_names': list(set([c['name'] for c in scene['characters'] if c['character_id'] != 'unknown'])),
                        'avg_confidence': sum([c['confidence'] for c in scene['characters'] if c['character_id'] != 'unknown']) / 
                                       max(len([c for c in scene['characters'] if c['character_id'] != 'unknown']), 1)
                    }
                else:
                    scene['characters'] = []
                    scene['character_summary'] = {
                        'total_characters': 0,
                        'identified_characters': 0,
                        'character_names': [],
                        'avg_confidence': 0.0
                    }
                
                results.append(scene)
        
        return results
    except Exception as e:
        logger.error(f"Scene analysis error: {str(e)}")
        raise SceneAnalysisError(str(e))

async def _calculate_scene_quality(
    scene: Dict,
    face_weight: float = 0.4,
    duration_weight: float = 0.3,
    stability_weight: float = 0.3
) -> float:
    """Calculate quality score for a scene based on multiple factors."""
    try:
        # Face detection score (0-1)
        face_score = min(len(scene.get('faces', [])) / 3, 1.0)
        
        # Duration score (0-1), assuming ideal duration is 5-15 seconds
        duration = scene.get('duration', 0)
        if duration < 5:
            duration_score = duration / 5
        elif duration > 15:
            duration_score = 15 / duration
        else:
            duration_score = 1.0
        
        # Stability score (0-1)
        stability_score = 1.0 - (
            len(scene.get('scene_transitions', [])) / 
            max(scene.get('total_frames', 1), 1)
        )
        
        # Calculate weighted average
        quality_score = (
            face_weight * face_score +
            duration_weight * duration_score +
            stability_weight * stability_score
        )
        
        return min(max(quality_score, 0.0), 1.0)
    except Exception as e:
        logger.error(f"Quality calculation error: {str(e)}")
        return 0.0

async def _identify_characters(faces: List[Dict], known_characters: List[Dict] = None) -> List[Dict]:
    """Identify characters in detected faces using trained face recognition."""
    try:
        identified_characters = []
        for face in faces:
            # Skip if face already has character identification from frame analysis
            if face.get('character_identified', False):
                identified_characters.append({
                    'character_id': face.get('character', 'unknown'),
                    'name': face.get('character', 'unknown'),
                    'confidence': face.get('character_confidence', 0.0),
                    'face_data': face
                })
                continue
            
            # Calculate face embeddings if not present
            if 'embedding' not in face:
                try:
                    face_embedding = await detector.get_face_embedding(face)
                except Exception as e:
                    logger.warning(f"Could not extract embedding for face: {str(e)}")
                    continue
            else:
                face_embedding = face['embedding']
            
            # Use the trained face recognition system
            character_result = detector.identify_character(face_embedding)
            
            if character_result:
                character_name, confidence = character_result
                identified_characters.append({
                    'character_id': character_name,
                    'name': character_name,
                    'confidence': confidence,
                    'face_data': face
                })
            else:
                # No character match found
                identified_characters.append({
                    'character_id': 'unknown',
                    'name': 'unknown',
                    'confidence': 0.0,
                    'face_data': face
                })
        
        return identified_characters
    except Exception as e:
        logger.error(f"Character identification error: {str(e)}")
        return []

async def select_scenes_for_script(
    scenes: List[Dict],
    script: Dict,
    audio_file: Optional[str] = None
) -> List[Dict]:
    """Select best scenes based on script context and quality."""
    try:
        selected_scenes = []
        script_segments = script.get('segments', [])
        
        for segment in script_segments:
            # Find best matching scenes for segment
            matching_scenes = await _find_matching_scenes(segment, scenes)
            
            # Sort by quality score
            matching_scenes.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # Select best scene
            if matching_scenes:
                selected_scenes.append({
                    'scene': matching_scenes[0],
                    'script_segment': segment,
                    'audio_timestamp': None  # Will be set during assembly
                })
        
        return selected_scenes
    except Exception as e:
        logger.error(f"Scene selection error: {str(e)}")
        raise SceneSelectionError(str(e))

async def _find_matching_scenes(script_segment: Dict, scenes: List[Dict]) -> List[Dict]:
    """Find scenes that match script segment context."""
    try:
        matching_scenes = []
        
        # Extract key information from script segment
        required_characters = script_segment.get('characters', [])
        required_actions = script_segment.get('actions', [])
        
        for scene in scenes:
            score = 0
            max_score = len(required_characters) + len(required_actions)
            
            # Check for character matches
            scene_characters = [
                char['name'] 
                for char in scene.get('characters', [])
            ]
            for character in required_characters:
                if character in scene_characters:
                    score += 1
            
            # Check for action matches (simplified)
            scene_actions = scene.get('detected_actions', [])
            for action in required_actions:
                if any(action.lower() in scene_action.lower() 
                      for scene_action in scene_actions):
                    score += 1
            
            # Calculate match percentage
            if max_score > 0:
                match_score = score / max_score
                if match_score > 0.5:  # Minimum match threshold
                    scene['match_score'] = match_score
                    matching_scenes.append(scene)
        
        return matching_scenes
    except Exception as e:
        logger.error(f"Scene matching error: {str(e)}")
        return []

async def process_video_pipeline(session_id: str):
    """Complete video processing pipeline (runs in background)."""
    try:
        session = active_sessions[session_id]
        
        # Phase A: Character Extraction (5-15%)
        if not session.use_known_characters:
            await manager.send_progress(session_id, 5, "extracting_characters", "Extracting characters from script...")
            
            if session.script and session.script.get("content"):
                characters = await openai_module.openai_service.extract_characters_from_script(session.script["content"])
            else:
                characters = settings.test_known_characters  # Use centralized config
        else:
            characters = settings.test_known_characters
        
        session.characters = characters
        await manager.send_progress(session_id, 15, "characters_extracted", f"Found {len(characters)} characters")
        
        # Phase B: Character Image Search (15-25%)
        await manager.send_progress(session_id, 20, "searching_images", "Finding character images...")
        
        try:
            character_images = await image_search_service.search_character_images(characters)
            logger.info(f"Found images for {len(character_images)} characters")
        except Exception as e:
            logger.error(f"Image search failed: {str(e)}")
            character_images = {}
        
        # Phase C: Face Model Training (25-35%)
        await manager.send_progress(session_id, 30, "training_face_model", "Training face recognition model...")
        
        try:
            if character_images:
                face_model = await detector.train_character_faces(character_images)
                logger.info(f"Trained face recognition for {len(face_model)} characters")
            else:
                logger.warning("No character images available for training")
                face_model = {}
        except Exception as e:
            logger.error(f"Face model training failed: {str(e)}")
            face_model = {}
        
        # Phase D: Audio Generation (35-55%)
        if not session.use_saved_audio:
            await manager.send_progress(session_id, 40, "generating_audio", "Creating audio with ElevenLabs...")
            
            try:
                if session.script and session.script.get("content"):
                    audio_file = f"output/{session_id}_audio.mp3"
                    generated_audio = await elevenlabs_service.generate_audio_from_script(
                        session.script["content"], 
                        audio_file
                    )
                    session.audio_file = generated_audio
                    logger.info(f"Generated audio: {generated_audio}")
                else:
                    logger.warning("No script content available for audio generation")
                    session.audio_file = None
            except Exception as e:
                logger.error(f"Audio generation failed: {str(e)}")
                session.audio_file = None
        else:
            # Use saved audio file
            if hasattr(session, 'selected_audio_file') and session.selected_audio_file:
                session.audio_file = session.selected_audio_file
            else:
                session.audio_file = None
        
        await manager.send_progress(session_id, 55, "audio_ready", "Audio generation completed")
        
        # Phase E: Scene Analysis (55-75%)
        await manager.send_progress(session_id, 60, "analyzing_scenes", "Analyzing video scenes...")
        
        try:
            scenes = await analyze_video_scenes(session.videos, session.characters)
            logger.info(f"Analyzed {len(scenes)} scenes from {len(session.videos)} videos")
        except Exception as e:
            logger.error(f"Scene analysis failed: {str(e)}")
            scenes = []
        
        # Phase F: Scene Selection (75-85%)
        await manager.send_progress(session_id, 80, "selecting_scenes", "Selecting best scenes...")
        
        try:
            if scenes and session.script:
                selected_scenes = await select_scenes_for_script(scenes, session.script, session.audio_file)
                logger.info(f"Selected {len(selected_scenes)} scenes for final video")
            else:
                logger.warning("No scenes or script available for selection")
                selected_scenes = []
        except Exception as e:
            logger.error(f"Scene selection failed: {str(e)}")
            selected_scenes = []
        
        # Phase G: Video Assembly (85-100%)
        await manager.send_progress(session_id, 90, "assembling_video", "Assembling final video...")
        
        try:
            if selected_scenes:
                output_video_path = f"output/{session_id}_final.mp4"
                
                # Prepare scenes for compilation
                compilation_scenes = []
                for selected_scene in selected_scenes:
                    scene = selected_scene.get('scene', selected_scene)
                    compilation_scenes.append(scene)
                
                output_video = await processor.create_compilation(
                    compilation_scenes, 
                    output_video_path,
                    progress_callback=lambda msg, prog: manager.send_progress(
                        session_id, 
                        90 + int(prog * 0.1), 
                        "assembling_video", 
                        msg
                    )
                )
                session.output_file = output_video
                logger.info(f"Created final video: {output_video}")
            else:
                logger.error("No scenes available for video assembly")
                session.output_file = None
                raise VideoProcessingError("No scenes available for video assembly")
        except Exception as e:
            logger.error(f"Video assembly failed: {str(e)}")
            session.output_file = None
            raise VideoProcessingError(f"Video assembly failed: {str(e)}")
        
        session.status = "completed"
        session.progress = 100
        
        await manager.send_progress(session_id, 100, "completed", "Video processing completed!")
        
    except Exception as e:
        logger.error(f"Processing pipeline error: {str(e)}")
        session.status = "failed"
        session.error = str(e)
        await manager.send_error(session_id, str(e))

# ====== SESSION MANAGEMENT ======

@router.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Get current session status and progress."""
    if session_id not in active_sessions:
        raise SessionNotFoundError("Session not found")
    
    session = active_sessions[session_id]
    
    return {
        "session_id": session.id,
        "status": session.status,
        "progress": session.progress,
        "created_at": session.created_at.isoformat(),
        "videos_count": len(session.videos),
        "has_script": session.script is not None,
        "has_audio": session.audio_file is not None,
        "characters": session.characters,
        "output_file": session.output_file,
        "error": session.error,
        "test_mode": session.test_mode
    }

@router.get("/download/{session_id}")
async def download_video(session_id: str):
    """Download the processed video."""
    try:
        # Get session from session manager
        session = await session_manager.get_session(session_id)
        
        if session.status != "completed":
            raise HTTPException(status_code=400, detail="Video not ready for download")
        
        # Get output file from video results
        video_results = session.metadata.get('video_results', {})
        output_file = video_results.get('video_path')
        
        if not output_file:
            raise HTTPException(status_code=404, detail="Output file not found in session")
        
        if not os.path.exists(output_file):
            raise HTTPException(status_code=404, detail="Output file not found on disk")
        
        return FileResponse(
            output_file,
            media_type="video/mp4",
            filename=f"ai_video_slicer_{session_id[:8]}.mp4"
        )
    
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Session not found")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# ====== TEST MODE UTILITIES ======

async def load_saved_script() -> str:
    """Load a saved script for test mode."""
    script_file = os.path.join(settings.TEST_SCRIPTS_DIR, "sample_script.txt")
    
    if os.path.exists(script_file):
        with open(script_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Create a sample script if none exists
        sample_script = """
        Welcome to this amazing video! Today we're going to explore the incredible world of action movies.
        
        Jean Claude Vandamme was known for his incredible martial arts skills and flexibility. His movies were always full of intense fight scenes.
        
        Steven Seagal brought a different style to action movies with his aikido background. His calm demeanor contrasted with explosive action sequences.
        
        Both of these action stars defined a generation of martial arts cinema and continue to influence modern action movies.
        """
        
        os.makedirs(settings.TEST_SCRIPTS_DIR, exist_ok=True)
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(sample_script)
        
        return sample_script

@router.get("/test-data/scripts")
async def list_saved_scripts():
    """List available saved scripts for test mode."""
    if not settings.test_mode_enabled:
        raise HTTPException(status_code=403, detail="Test mode not enabled")
    
    scripts_dir = settings.TEST_SCRIPTS_DIR
    if not os.path.exists(scripts_dir):
        return {"scripts": []}
    
    scripts = []
    for filename in os.listdir(scripts_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(scripts_dir, filename)
            scripts.append({
                "filename": filename,
                "size": os.path.getsize(filepath),
                "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            })
    
    return {"scripts": scripts}

@router.get("/test-data/audio")
async def list_saved_audio():
    """List available saved audio files for test mode."""
    if not settings.test_mode_enabled:
        raise HTTPException(status_code=403, detail="Test mode not enabled")
    
    audio_dir = settings.TEST_AUDIO_DIR
    if not os.path.exists(audio_dir):
        return {"audio_files": []}
    
    audio_files = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.mp3', '.wav', '.m4a')):
            filepath = os.path.join(audio_dir, filename)
            audio_files.append({
                "filename": filename,
                "size": os.path.getsize(filepath),
                "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            })
    
    return {"audio_files": audio_files}

# Phase 4: Enhanced Video Processing Endpoints

@router.post("/api/v1/analyze-videos")
async def analyze_videos(request: Request):
    """Analyze multiple videos in parallel for scenes and faces."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        video_paths = data.get("video_paths", [])
        
        if not video_paths:
            raise HTTPException(status_code=400, detail="No video paths provided")
        
        if session_id not in active_sessions:
            raise SessionNotFoundError("Session not found")
        
        session = active_sessions[session_id]
        
        async def progress_callback(message: str, progress: int):
            await manager.send_progress(
                session_id,
                progress,
                "analyzing_videos",
                message
            )
        
        # Process videos in parallel
        results = await processor.process_videos_parallel(
            video_paths,
            progress_callback=progress_callback
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.post("/api/v1/select-scenes")
async def select_scenes(request: Request):
    """Select scenes from analyzed videos based on criteria."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        video_results = data.get("video_results", {})
        criteria = data.get("criteria", {})
        
        if not video_results:
            raise HTTPException(status_code=400, detail="No video results provided")
        
        if session_id not in active_sessions:
            raise SessionNotFoundError("Session not found")
        
        session = active_sessions[session_id]
        
        # Filter scenes based on criteria
        selected_scenes = []
        for video_path, result in video_results.items():
            if "scenes" not in result:
                continue
                
            for scene in result["scenes"]:
                # Apply filtering criteria
                if _scene_matches_criteria(scene, criteria):
                    scene["video_path"] = video_path
                    selected_scenes.append(scene)
        
        # Sort scenes by criteria
        selected_scenes.sort(
            key=lambda x: (
                -len(x.get("faces", [])),  # More faces first
                -x.get("duration", 0)  # Longer scenes first
            )
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "selected_scenes": selected_scenes,
            "total_scenes": len(selected_scenes)
        }
        
    except Exception as e:
        logger.error(f"Scene selection error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.post("/api/v1/create-compilation")
async def create_compilation(request: Request):
    """Create video compilation from selected scenes."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        selected_scenes = data.get("selected_scenes", [])
        
        if not selected_scenes:
            raise HTTPException(status_code=400, detail="No scenes selected")
        
        if session_id not in active_sessions:
            raise SessionNotFoundError("Session not found")
        
        session = active_sessions[session_id]
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            settings.output_dir,
            f"compilation_{session_id}_{timestamp}.mp4"
        )
        
        async def progress_callback(message: str, progress: int):
            await manager.send_progress(
                session_id,
                progress,
                "creating_compilation",
                message
            )
        
        # Create compilation
        output_file = await processor.create_compilation(
            selected_scenes,
            output_path,
            progress_callback=progress_callback
        )
        
        session.output_file = output_file
        session.status = "completed"
        
        # Send completion notification
        await manager.send_completion(session_id, output_file)
        
        return {
            "success": True,
            "session_id": session_id,
            "output_file": output_file
        }
        
    except Exception as e:
        logger.error(f"Compilation error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.post("/api/v1/train-characters")
async def train_characters(request: Request):
    """Train character recognition from video samples."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        characters = data.get("characters", [])
        
        if not characters:
            raise HTTPException(status_code=400, detail="No characters provided")
        
        if session_id not in active_sessions:
            raise SessionNotFoundError("Session not found")
        
        session = active_sessions[session_id]
        
        async def progress_callback(message: str, progress: int):
            await manager.send_progress(
                session_id,
                progress,
                "training_characters",
                message
            )
        
        results = {}
        for character in characters:
            name = character.get("name")
            video_path = character.get("video_path")
            
            if not name or not video_path:
                continue
            
            try:
                character_data = await detector.train_character(
                    name,
                    video_path,
                    progress_callback
                )
                results[name] = character_data
            except Exception as e:
                logger.error(f"Error training character {name}: {str(e)}")
                results[name] = {"error": str(e)}
        
        return {
            "success": True,
            "session_id": session_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Character training error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def _scene_matches_criteria(scene: dict, criteria: dict) -> bool:
    """Check if scene matches selection criteria."""
    try:
        # Duration criteria
        min_duration = criteria.get("min_duration", settings.min_scene_duration)
        max_duration = criteria.get("max_duration", settings.max_scene_duration)
        
        if not (min_duration <= scene.get("duration", 0) <= max_duration):
            return False
        
        # Face criteria
        min_faces = criteria.get("min_faces", 0)
        max_faces = criteria.get("max_faces", float("inf"))
        face_count = len(scene.get("faces", []))
        
        if not (min_faces <= face_count <= max_faces):
            return False
        
        # Face confidence criteria
        min_confidence = criteria.get("min_face_confidence", 0.0)
        if scene.get("faces"):
            avg_confidence = sum(
                face.get("confidence", 0)
                for face in scene["faces"]
            ) / len(scene["faces"])
            
            if avg_confidence < min_confidence:
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error matching scene criteria: {str(e)}")
        return False

@router.get("/api/v1/processing-stats")
async def get_processing_stats():
    """Get current video processing statistics."""
    try:
        active_count = len(active_sessions)
        completed_count = sum(
            1 for session in active_sessions.values()
            if session.status == "completed"
        )
        error_count = sum(
            1 for session in active_sessions.values()
            if session.error is not None
        )
        
        return {
            "success": True,
            "stats": {
                "active_sessions": active_count,
                "completed_sessions": completed_count,
                "error_sessions": error_count,
                "parallel_processing": settings.parallel_processing_enabled,
                "max_workers": settings.max_workers
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting processing stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.delete("/api/v1/cleanup-session/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up session data and temporary files."""
    try:
        if session_id not in active_sessions:
            raise SessionNotFoundError("Session not found")
        
        session = active_sessions[session_id]
        
        # Clean up temporary files
        if session.output_file and os.path.exists(session.output_file):
            os.remove(session.output_file)
        
        # Remove session
        del active_sessions[session_id]
        
        return {
            "success": True,
            "message": f"Session {session_id} cleaned up successfully"
        }
        
    except Exception as e:
        logger.error(f"Session cleanup error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

async def validate_file_size(file: UploadFile) -> None:
    """Validate file size using a temporary file."""
    temp_file = Path(settings.temp_dir) / f"size_check_{uuid.uuid4()}.tmp"
    try:
        # Write file to temp location to get size
        async with aiofiles.open(temp_file, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Check file size
        file_size = temp_file.stat().st_size
        if file_size > settings.max_video_file_size_bytes:
            raise VideoValidationError(
                f"Video file size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum limit of 400MB"
            )
        
        # Reset file position for later use
        await file.seek(0)
        
    finally:
        # Cleanup temp file
        if temp_file.exists():
            temp_file.unlink()

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Upload video file endpoint."""
    try:
        # Validate session
        if session_id not in active_sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        session = active_sessions[session_id]
        
        # Check number of videos
        if len(session.videos) >= settings.max_videos_per_session:
            raise VideoValidationError(f"Maximum {settings.max_videos_per_session} videos allowed per session")
            
        # Validate file size
        await validate_file_size(file)

        # Generate unique filename
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        file_path = os.path.join(settings.upload_dir, filename)

        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        video_info = {
            "filename": file.filename,
            "path": file_path,
            "size": len(content),
            "content_type": file.content_type
        }
        session.videos.append(video_info)
        
        session.status = "videos_uploaded"
        
        await manager.send_progress(session_id, 100, "videos_uploaded", "Videos uploaded successfully!")
        
        return {
            "status": "videos_uploaded",
            "videos_count": len(session.videos),
            "video_info": video_info
        }
        
    except Exception as e:
        logger.error(f"Video upload error: {str(e)}")
        if session_id in active_sessions:
            active_sessions[session_id].error = str(e)
            await manager.send_error(session_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-video")
async def process_video(request: ProcessVideoRequest):
    """Enhanced video processing endpoint with credit management and error handling."""
    session_id = None
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        logger.info(f"Starting enhanced video processing for session: {session_id}")
        
        # Initialize session
        session_manager.create_session(session_id, {
            'video_path': request.video_path,
            'script_content': request.script_content,
            'test_mode': request.test_mode,
            'test_settings': request.test_settings
        })
        
        # Check account status before starting
        account_status = credit_manager.get_account_status()
        logger.info(f"Account status at processing start: {account_status}")
        
        # Process video with enhanced pipeline
        video_processor = processor
        result = await video_processor.process_video_pipeline(
            session_id=session_id,
            video_path=request.video_path,
            script_content=request.script_content,
            websocket_manager=manager,
            test_mode=request.test_mode,
            test_settings=request.test_settings or {}
        )
        
        if result['success']:
            # Update session with results
            session_manager.update_session(session_id, {
                'status': 'completed',
                'result': result,
                'processing_summary': result.get('processing_summary'),
                'credit_usage': result.get('credit_usage')
            })
            
            return {
                "success": True,
                "session_id": session_id,
                "message": "Video processing completed successfully",
                "video_path": result.get('video_path'),
                "characters": result.get('characters', []),
                "scenes_count": len(result.get('scenes', [])),
                "credit_usage": result.get('credit_usage', {}),
                "processing_summary": result.get('processing_summary', {})
            }
        else:
            # Handle processing failure
            error_type = result.get('error_type', 'processing_error')
            error_message = result.get('error', 'Unknown processing error')
            
            session_manager.update_session(session_id, {
                'status': 'failed',
                'error': error_message,
                'error_type': error_type
            })
            
            if error_type == 'credit_exhausted':
                return HTTPException(
                    status_code=402,  # Payment Required
                    detail={
                        "error": "Credit Limit Reached",
                        "message": error_message,
                        "error_type": "credit_exhausted",
                        "session_id": session_id,
                        "account_status": credit_manager.get_account_status()
                    }
                )
            else:
                return HTTPException(
                    status_code=500,
                    detail={
                        "error": "Processing Failed",
                        "message": error_message,
                        "session_id": session_id
                    }
                )
    
    except CreditExhaustionError as e:
        error_message = f"Credit exhaustion during processing: {str(e)}"
        logger.error(error_message)
        
        if session_id:
            session_manager.update_session(session_id, {
                'status': 'failed',
                'error': error_message,
                'error_type': 'credit_exhausted'
            })
        
        return HTTPException(
            status_code=402,  # Payment Required
            detail={
                "error": "Credit Limit Reached",
                "message": "Operations paused to prevent overage charges. Please check your account limits.",
                "error_type": "credit_exhausted",
                "session_id": session_id,
                "account_status": credit_manager.get_account_status()
            }
        )
    
    except Exception as e:
        error_message = f"Unexpected error in video processing: {str(e)}"
        logger.error(error_message)
        
        if session_id:
            session_manager.update_session(session_id, {
                'status': 'failed',
                'error': error_message,
                'error_type': 'unexpected_error'
            })
            
            # Clean up resources
            await parallel_error_handler.cleanup_resources()
        
        return HTTPException(
            status_code=500,
            detail={
                "error": "Processing Failed",
                "message": "An unexpected error occurred during processing",
                "session_id": session_id
            }
        )

@router.get("/session/{session_id}/credit-usage")
async def get_session_credit_usage(session_id: str):
    """Get detailed credit usage for a specific session."""
    try:
        # Set the session ID in credit manager to get the right data
        credit_manager.current_session_id = session_id
        
        usage_summary = credit_manager.get_session_usage_summary()
        account_status = credit_manager.get_account_status()
        
        return {
            "success": True,
            "session_id": session_id,
            "usage_summary": usage_summary,
            "account_status": account_status
        }
    
    except Exception as e:
        logger.error(f"Error getting credit usage for session {session_id}: {str(e)}")
        return HTTPException(
            status_code=500,
            detail=f"Failed to get credit usage: {str(e)}"
        )

@router.get("/accounts/status")
async def get_accounts_status():
    """Get current status of all API accounts."""
    try:
        account_status = credit_manager.get_account_status()
        
        # Add health indicators
        for service, accounts in account_status.items():
            for account in accounts:
                if account['usage_percentage'] >= 0.95:
                    account['health'] = 'critical'
                    account['message'] = 'Account at 95%+ limit - operations paused'
                elif account['usage_percentage'] >= 0.90:
                    account['health'] = 'warning'
                    account['message'] = 'Account approaching limit'
                else:
                    account['health'] = 'healthy'
                    account['message'] = 'Account operating normally'
        
        return {
            "success": True,
            "accounts": account_status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting account status: {str(e)}")
        return HTTPException(
            status_code=500,
            detail=f"Failed to get account status: {str(e)}"
        )

@router.get("/session/{session_id}/processing-summary")
async def get_processing_summary(session_id: str):
    """Get detailed processing summary for a session."""
    try:
        # Set the session ID in parallel error handler
        parallel_error_handler.session_id = session_id
        
        processing_summary = parallel_error_handler.get_session_summary()
        
        # Get session data
        session_data = session_manager.get_session(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "processing_summary": processing_summary,
            "session_data": session_data
        }
    
    except Exception as e:
        logger.error(f"Error getting processing summary for session {session_id}: {str(e)}")
        return HTTPException(
            status_code=500,
            detail=f"Failed to get processing summary: {str(e)}"
        ) 

