from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from pathlib import Path
from typing import List
import asyncio
import logging

from video_processor import AdvancedVideoProcessor
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Editing App",
    description="Standalone video editing with scene analysis and assembly",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.temp_dir, exist_ok=True)

app.mount("/downloads", StaticFiles(directory=settings.output_dir), name="downloads")

# Global state for current processing
current_videos = []
current_audio = None
processing_status = "idle"
processing_progress = 0

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Video Editing App"}

@app.post("/upload-videos")
async def upload_videos(files: List[UploadFile] = File(...)):
    """Upload video files for processing."""
    global current_videos
    
    try:
        uploaded_videos = []
        
        for i, file in enumerate(files):
            if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                raise HTTPException(status_code=400, detail=f"Invalid video format: {file.filename}")
            
            file_path = os.path.join(settings.upload_dir, f"video_{i}_{file.filename}")
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_videos.append({
                "filename": file.filename,
                "path": file_path,
                "size": len(content)
            })
        
        current_videos = uploaded_videos
        logger.info(f"Uploaded {len(uploaded_videos)} videos")
        
        return {
            "status": "success",
            "videos_count": len(uploaded_videos),
            "videos": uploaded_videos
        }
        
    except Exception as e:
        logger.error(f"Video upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for processing."""
    global current_audio
    
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg')):
            raise HTTPException(status_code=400, detail=f"Invalid audio format: {file.filename}")
        
        file_path = os.path.join(settings.upload_dir, f"audio_{file.filename}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        current_audio = {
            "filename": file.filename,
            "path": file_path,
            "size": len(content)
        }
        
        logger.info(f"Uploaded audio: {file.filename}")
        
        return {
            "status": "success",
            "audio_file": current_audio
        }
        
    except Exception as e:
        logger.error(f"Audio upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-editing")
async def start_editing():
    """Start the video editing process."""
    global current_videos, current_audio, processing_status, processing_progress
    
    try:
        if not current_videos:
            raise HTTPException(status_code=400, detail="No videos uploaded")
        
        if not current_audio:
            raise HTTPException(status_code=400, detail="No audio file uploaded")
        
        processing_status = "processing"
        processing_progress = 0
        
        # Start processing in background
        asyncio.create_task(run_editing_pipeline())
        
        return {"status": "processing_started"}
        
    except Exception as e:
        logger.error(f"Start editing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_editing_pipeline():
    """Run the complete advanced editing pipeline."""
    global current_videos, current_audio, processing_status, processing_progress
    
    try:
        processor = AdvancedVideoProcessor()
        
        # Phase 1: Character Extraction (0-10%)
        processing_progress = 5
        logger.info("Phase 1: Extracting characters from script...")
        
        # Use a sample script content
        script_content = """
        In the world of action cinema, a rivalry simmered quietly for decades. 
        It involved two larger-than-life figures: Jean-Claude, known for his grace and power, 
        and Seagal, the enigmatic martial artist with a penchant for wry comments.
        """
        
        characters = await processor.extract_characters_from_script(script_content)
        logger.info(f"Extracted characters: {characters}")
        
        # Phase 2: Image Search (10-25%)
        processing_progress = 15
        logger.info("Phase 2: Searching for character images...")
        
        character_images = await processor.search_character_images(characters)
        logger.info(f"Found images for {len(character_images)} characters")
        
        # Phase 3: Face Recognition Training (25-40%)
        processing_progress = 30
        logger.info("Phase 3: Training face recognition model...")
        
        face_model = await processor.train_face_recognition(character_images)
        logger.info(f"Face recognition trained for {len(face_model)} characters")
        
        # Phase 4: Video Analysis (40-70%)
        processing_progress = 45
        logger.info("Phase 4: Analyzing videos with character detection...")
        
        scenes = await processor.analyze_videos_with_character_detection(
            [v['path'] for v in current_videos], 
            face_model
        )
        logger.info(f"Found {len(scenes)} scenes with characters")
        
        # Phase 5: Scene Selection (70-85%)
        processing_progress = 75
        logger.info("Phase 5: Selecting best scenes...")
        
        selected_scenes = await processor.select_best_scenes(scenes, script_content)
        logger.info(f"Selected {len(selected_scenes)} best scenes")
        
        # Phase 6: Video Assembly (85-100%)
        processing_progress = 90
        logger.info("Phase 6: Assembling final video...")
        
        output_path = os.path.join(settings.output_dir, "final_video.mp4")
        
        final_video = await processor.assemble_video(
            selected_scenes, 
            current_audio["path"], 
            output_path
        )
        
        processing_progress = 100
        processing_status = "completed"
        
        logger.info(f"Advanced video editing completed: {final_video}")
        
    except Exception as e:
        logger.error(f"Advanced editing pipeline error: {str(e)}")
        processing_status = "error"

@app.get("/status")
async def get_status():
    """Get processing status."""
    return {
        "status": processing_status,
        "progress": processing_progress,
        "videos_count": len(current_videos),
        "has_audio": current_audio is not None
    }

@app.get("/download")
async def download_video():
    """Download the final video."""
    if processing_status != "completed":
        raise HTTPException(status_code=400, detail="Video not ready for download")
    
    output_path = os.path.join(settings.output_dir, "final_video.mp4")
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="edited_video.mp4"
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    ) 