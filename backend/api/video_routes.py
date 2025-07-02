from fastapi import APIRouter, UploadFile, File, WebSocket, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
import os
from pathlib import Path
import uuid
import aiofiles
from ..core.config import get_video_config
from ..core.session import SessionManager
from ..core.websocket import WebSocketManager
from ..services.video_processor import VideoProcessor
from ..core.exceptions import VideoProcessingError

router = APIRouter()
websocket_manager = WebSocketManager()
session_manager = SessionManager()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> Dict:
    """Upload a video file for processing."""
    try:
        # Generate unique session ID and file path
        session_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{session_id}{file_extension}"
        
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Initialize session
        await session_manager.create_session(session_id, {
            'filename': file.filename,
            'file_path': str(file_path),
            'status': 'uploaded'
        })

        return {
            'session_id': session_id,
            'status': 'uploaded',
            'filename': file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time processing updates."""
    try:
        await websocket_manager.connect(websocket, session_id)
        
        try:
            while True:
                data = await websocket.receive_text()
                # Handle any incoming WebSocket messages if needed
                pass
                
        except Exception as e:
            await websocket_manager.send_error(session_id, str(e))
            
        finally:
            await websocket_manager.disconnect(websocket, session_id)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/{session_id}")
async def process_video(session_id: str) -> Dict:
    """Start video processing for a session."""
    try:
        # Get session data
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Initialize video processor
        config = get_video_config()
        processor = VideoProcessor(config, session_manager)

        # Start processing in background task
        file_path = session['file_path']
        try:
            result = await processor.process_video(file_path, session_id)
            return result
        except VideoProcessingError as e:
            await websocket_manager.send_error(session_id, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        await websocket_manager.send_error(session_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-scene/{session_id}")
async def extract_scene(
    session_id: str,
    start_time: float,
    end_time: float
) -> Dict:
    """Extract a scene from the video."""
    try:
        # Get session data
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Initialize video processor
        config = get_video_config()
        processor = VideoProcessor(config, session_manager)

        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"scene_{session_id}_{start_time}_{end_time}.mp4"

        # Extract scene
        try:
            result_path = await processor.extract_scene(
                session['file_path'],
                start_time,
                end_time,
                str(output_path)
            )
            return {
                'status': 'success',
                'output_path': result_path
            }
        except VideoProcessingError as e:
            await websocket_manager.send_error(session_id, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        await websocket_manager.send_error(session_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))

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