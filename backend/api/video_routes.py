from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
import os
from pathlib import Path
import uuid
import aiofiles
import cv2
from datetime import datetime
import asyncio
from core.config import settings
from core.session import manager as session_manager
from video.processor import VideoProcessor
from core.exceptions import VideoProcessingError

# Import the unified WebSocket manager
from api.websocket import manager as websocket_manager

router = APIRouter()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> Dict:
    """Upload a video file for processing."""
    session_id = None
    file_path = None
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size
        if hasattr(file, 'size') and file.size > settings.max_file_size:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {settings.max_file_size} bytes")
        
        # Check file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.allowed_video_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {settings.allowed_video_extensions}")
        
        # Generate unique session ID and file path
        session_id = str(uuid.uuid4())
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{session_id}{file_extension}"
        
        # Save uploaded file with progress tracking
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Verify file was saved correctly
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

        # Initialize session with comprehensive metadata
        await session_manager.create_session(session_id, {
            'filename': file.filename,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'file_extension': file_extension,
            'status': 'uploaded',
            'upload_timestamp': str(datetime.now()),
            'progress': 0
        })

        return {
            'session_id': session_id,
            'status': 'uploaded',
            'filename': file.filename,
            'file_size': file_path.stat().st_size
        }

    except HTTPException:
        # Clean up on HTTP errors
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        raise
    except Exception as e:
        # Clean up on any other errors
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        if session_id:
            try:
                await session_manager.cleanup_session(session_id)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# WebSocket endpoint is now handled by api/websocket.py with /api/video prefix

@router.post("/process/{session_id}")
async def process_video(session_id: str) -> Dict:
    """Start video processing for a session."""
    try:
        # Get session data
        session = await session_manager.get_session(session_id)
        
        # Validate session status
        if session.status not in ['uploaded', 'ready']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot process video in status: {session.status}"
            )
        
        # Validate file exists
        file_path = session.metadata.get('file_path') or getattr(session, 'file_path', None)
        if not file_path or not Path(file_path).exists():
            await session_manager.update_session(session_id, status='error', error='Video file not found')
            raise HTTPException(status_code=404, detail="Video file not found")

        # Update session to processing status
        await session_manager.update_session(session_id, 
            status='processing', 
            progress=5,
            processing_start_time=str(datetime.now())
        )
        
        # Send initial progress update
        await websocket_manager.send_progress(session_id, 5, 'processing', 'Starting video analysis...')

        # Initialize video processor
        processor = VideoProcessor()

        # Start processing in background task with comprehensive error handling
        try:
            # Create a background task for processing
            task = asyncio.create_task(
                process_video_background(session_id, file_path, processor, websocket_manager, session_manager)
            )
            
            return {
                'status': 'processing_started',
                'session_id': session_id,
                'message': 'Video processing has started. Check WebSocket for progress updates.'
            }
            
        except VideoProcessingError as e:
            await session_manager.update_session(session_id, status='error', error=str(e))
            await websocket_manager.send_error(session_id, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        await session_manager.update_session(session_id, status='error', error=str(e))
        await websocket_manager.send_error(session_id, str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def process_video_background(session_id: str, file_path: str, processor: VideoProcessor, 
                                  websocket_manager, session_manager):
    """Background task for video processing with real-time updates."""
    try:
        # Send progress updates throughout processing
        await websocket_manager.send_progress(session_id, 10, 'processing', 'Analyzing video structure...')
        
        result = await processor._process_single_video_async(file_path)
        
        # Send scenes immediately to frontend for analysis
        if result.get('scenes'):
            await websocket_manager.send_scenes(session_id, result['scenes'])
        
        # Update session with results
        await session_manager.update_session(session_id, 
            status='completed',
            progress=100,
            video_results=result,
            processing_end_time=str(datetime.now())
        )
        
        # Send completion notification
        await websocket_manager.send_completion(session_id, {
            'video_path': result['video_path'],
            'scenes': result.get('scenes', []),
            'scenes_detected': result.get('total_scenes', 0),
            'processing_time': result.get('duration', 0)
        })
        
    except Exception as e:
        await session_manager.update_session(session_id, status='error', error=str(e))
        await websocket_manager.send_error(session_id, str(e))

from pydantic import BaseModel

class ExtractSceneRequest(BaseModel):
    start_time: float
    end_time: float

@router.post("/extract-scene/{session_id}")
async def extract_scene(
    session_id: str,
    request: ExtractSceneRequest
) -> Dict:
    """Extract a scene from the video."""
    try:
        # Validate request parameters
        if request.start_time < 0 or request.end_time <= request.start_time:
            raise HTTPException(status_code=400, detail="Invalid time range")
        
        if request.end_time - request.start_time > 300:  # 5 minutes max
            raise HTTPException(status_code=400, detail="Scene too long. Maximum 5 minutes allowed")
        
        # Get session data
        session = await session_manager.get_session(session_id)
        
        # Validate session has processed video
        if session.status not in ['completed', 'processing']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot extract scene from video in status: {session.status}"
            )
        
        # Get file path
        file_path = session.metadata.get('file_path') or getattr(session, 'file_path', None)
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Source video file not found")

        # Validate video duration
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            raise HTTPException(status_code=500, detail="Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        if request.end_time > video_duration:
            raise HTTPException(
                status_code=400, 
                detail=f"End time {request.end_time}s exceeds video duration {video_duration:.2f}s"
            )

        # Initialize video processor
        processor = VideoProcessor()

        # Create output directory with session-specific folder
        output_dir = Path(settings.output_dir) / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"scene_{timestamp}_{request.start_time}_{request.end_time}.mp4"
        output_path = output_dir / output_filename

        # Convert time to frames
        start_frame = int(request.start_time * fps)
        end_frame = int(request.end_time * fps)

        # Extract scene with error handling
        try:
            # Send progress update
            await websocket_manager.send_progress(session_id, 0, 'extracting', 'Starting scene extraction...')
            
            processor._extract_scene(
                file_path,
                start_frame,
                end_frame,
                str(output_path)
            )
            
            # Verify output file was created
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise VideoProcessingError("Scene extraction failed - no output file created")
            
            # Update session with extracted scene info
            scene_info = {
                'start_time': request.start_time,
                'end_time': request.end_time,
                'duration': request.end_time - request.start_time,
                'output_path': str(output_path),
                'file_size': output_path.stat().st_size,
                'extracted_at': str(datetime.now())
            }
            
            await session_manager.update_session(session_id, 
                extracted_scenes=session.metadata.get('extracted_scenes', []) + [scene_info]
            )
            
            # Send completion notification
            await websocket_manager.send_progress(session_id, 100, 'completed', 'Scene extraction completed')
            
            return {
                'status': 'success',
                'output_path': str(output_path),
                'scene_info': scene_info,
                'download_url': f"/api/v1/download/{session_id}/{output_filename}"
            }
            
        except VideoProcessingError as e:
            await websocket_manager.send_error(session_id, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        await websocket_manager.send_error(session_id, str(e))
        raise HTTPException(status_code=500, detail=f"Scene extraction failed: {str(e)}")

@router.get("/status/{session_id}")
async def get_session_status(session_id: str) -> Dict:
    """Get the current status of a processing session."""
    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 