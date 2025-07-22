import sys
import os
# Add ai_shared_lib to Python path
yt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, yt_root)
print(f"Added to Python path: {yt_root}")

# Add error handling for problematic imports
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")
    sys.exit(1)

try:
    import cv2
    print("OpenCV imported successfully")
except ImportError as e:
    print(f"OpenCV import error: {e}")
    sys.exit(1)

# Now import the rest
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from typing import List
import asyncio
import logging
import insightface

# Import services with error handling
try:
    from video_processor import AdvancedVideoProcessor
    print("Video processor imported successfully")
    
    # Use importlib to import ai_shared_lib modules
    import importlib.util
    
    # Import config
    config_spec = importlib.util.spec_from_file_location("config", os.path.join(yt_root, "ai_shared_lib", "config.py"))
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    settings = config_module.settings
    print("Settings imported successfully via importlib")
    
    print("Services imported successfully")
except ImportError as e:
    print(f"Service import error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"yt_root: {yt_root}")
    print(f"Path exists: {os.path.exists(yt_root)}")
    print(f"yt_root contents: {os.listdir(yt_root) if os.path.exists(yt_root) else 'N/A'}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Editing App",
    description="Standalone video editing with scene analysis and assembly",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the config attribute names
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.CACHE_DIR, exist_ok=True)

app.mount("/downloads", StaticFiles(directory=settings.OUTPUT_DIR), name="downloads")

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
        logger.info(f"Starting video upload for {len(files)} files")
        uploaded_videos = []
        
        for i, file in enumerate(files):
            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                raise HTTPException(status_code=400, detail=f"Invalid video format: {file.filename}")
            
            file_path = os.path.join(settings.UPLOAD_DIR, f"video_{i}_{file.filename}")
            logger.info(f"Saving to: {file_path}")
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"File {file.filename} saved successfully ({len(content)} bytes)")
            
            uploaded_videos.append({
                "filename": file.filename,
                "path": file_path,
                "size": len(content)
            })
        
        current_videos = uploaded_videos
        logger.info(f"✅ Uploaded {len(uploaded_videos)} videos successfully")
        
        return {
            "status": "success",
            "videos_count": len(uploaded_videos),
            "videos": uploaded_videos
        }
        
    except Exception as e:
        logger.error(f"❌ Video upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for processing."""
    global current_audio
    
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg')):
            raise HTTPException(status_code=400, detail=f"Invalid audio format: {file.filename}")
        
        file_path = os.path.join(settings.UPLOAD_DIR, f"audio_{file.filename}")
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
async def start_editing(characters: str = Form("Jean-Claude Van Damme, Steven Seagal")):  # Default value
    """Start the video editing process with manual character input."""
    global current_videos, current_audio, processing_status, processing_progress
    
    try:
        logger.info("START-EDITING BUTTON CLICKED - MANUAL CHARACTERS")
        
        if not current_videos:
            raise HTTPException(status_code=400, detail="No videos uploaded")
        
        if not current_audio:
            raise HTTPException(status_code=400, detail="No audio file uploaded")
        
        # Parse characters from input - FIX THE PARSING
        character_list = []
        for char in characters.split(','):
            char = char.strip()
            if char:
                # Remove any trailing periods and clean up
                char = char.rstrip('.').strip()
                if char:  # Only add if not empty after cleaning
                    character_list.append(char)
        
        logger.info(f"Manual characters: {character_list}")
        
        processing_status = "processing"
        processing_progress = 0
        
        # Start processing with manual characters
        asyncio.create_task(run_editing_pipeline(character_list))
        
        return {
            "status": "processing_started",
            "message": "Video editing pipeline started successfully"
        }
        
    except Exception as e:
        logger.error(f"Start editing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_editing_pipeline(manual_characters: List[str]):
    """Run the complete advanced editing pipeline with manual characters."""
    global current_videos, current_audio, processing_status, processing_progress
    
    try:
        logger.info("Starting REAL advanced editing pipeline with manual characters...")
        processor = AdvancedVideoProcessor()
        
        # Phase 1: Use manual characters (0-10%)
        processing_progress = 5
        logger.info(f"Phase 1: Using manual characters: {manual_characters}")
        characters = manual_characters
        
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
        
        # CRITICAL FIX: Get audio duration for scene selection
        audio_duration = None
        try:
            from moviepy.editor import AudioFileClip
            audio_clip = AudioFileClip(current_audio["path"])
            audio_duration = audio_clip.duration
            audio_clip.close()
            logger.info(f"Audio duration detected: {audio_duration:.1f}s ({audio_duration/60:.1f} minutes)")
        except Exception as e:
            logger.warning(f"Could not detect audio duration: {str(e)}, using default 20 minutes")
            audio_duration = 1200.0  # 20 minutes default
        
        # Get all scenes from all videos
        all_scenes = await processor.analyze_videos_with_character_detection(
            [v['path'] for v in current_videos], 
            face_model,
            audio_duration
        )
        
        # CRITICAL FIX: Pass audio duration to scene selection
        selected_scenes = await processor.select_best_scenes(all_scenes, "", audio_duration)
        logger.info(f"Selected {len(selected_scenes)} scenes for {audio_duration/60:.1f} minute audio")
        
        # CRITICAL FIX: Multi-level fallback system
        if not selected_scenes:
            logger.warning("No scenes selected from main pipeline - trying fallback scenes")
            selected_scenes = await processor.create_fallback_scenes(current_videos)
            logger.info(f"Created {len(selected_scenes)} fallback scenes")
        
        # ULTIMATE FALLBACK: Guaranteed valid scenes
        if not selected_scenes:
            logger.warning("Fallback scenes failed - creating guaranteed valid scenes")
            selected_scenes = await processor.create_guaranteed_valid_scenes(current_videos)
            logger.info(f"Created {len(selected_scenes)} guaranteed valid scenes")
        
        # DURATION FALLBACK: Create extended scenes if duration is too short
        if selected_scenes:
            total_duration = sum(scene.get('duration', 0) for scene in selected_scenes)
            if total_duration < audio_duration + 10:  # Less than audio duration + 10 seconds
                logger.warning(f"Selected duration ({total_duration:.1f}s) too short for audio ({audio_duration:.1f}s + 20s) - creating extended scenes")
                extended_scenes = await processor.create_extended_scenes(current_videos, audio_duration + 20)
                if extended_scenes:
                    selected_scenes = extended_scenes
                    logger.info(f"Replaced with {len(extended_scenes)} extended scenes")
        
        # FINAL VALIDATION: Ensure we have something to work with
        if not selected_scenes:
            logger.error("CRITICAL: No valid scenes available after all fallback attempts!")
            raise Exception("No valid scenes available for video assembly after all fallback attempts")
        
        # CRITICAL: Pre-assembly validation
        logger.info(f"Pre-assembly validation: {len(selected_scenes)} scenes ready")
        for i, scene in enumerate(selected_scenes):
            if not scene.get('validated', False):
                logger.warning(f"Scene {i+1} not validated, attempting validation now")
                try:
                    is_valid = await processor.validate_scene_video_clip(scene)
                    if not is_valid:
                        logger.error(f"Scene {i+1} failed pre-assembly validation")
                except Exception as e:
                    logger.error(f"Scene {i+1} pre-assembly validation error: {str(e)}")
        
        # Phase 6: Video Assembly (85-100%)
        processing_progress = 90
        logger.info("Phase 6: Assembling final video...")
        
        output_path = os.path.join(settings.OUTPUT_DIR, "final_video.mp4")
        
        try:
            final_video = await processor.assemble_video(
                selected_scenes, 
                current_audio["path"], 
                output_path
            )
            
            processing_progress = 100
            processing_status = "completed"
            
            logger.info(f"REAL advanced video editing completed: {final_video}")
            
        except Exception as assembly_error:
            logger.error(f"Video assembly failed: {str(assembly_error)}")
            
            # FINAL EMERGENCY FALLBACK: Create simple video
            logger.warning("Attempting emergency fallback video creation...")
            try:
                emergency_scenes = await processor.create_guaranteed_valid_scenes(current_videos)
                if emergency_scenes:
                    final_video = await processor.assemble_video(
                        emergency_scenes[:3],  # Use only first 3 scenes
                        current_audio["path"], 
                        output_path
                    )
                    processing_progress = 100
                    processing_status = "completed"
                    logger.info(f"Emergency fallback video created: {final_video}")
                else:
                    raise Exception("Emergency fallback also failed")
            except Exception as emergency_error:
                logger.error(f"Emergency fallback failed: {str(emergency_error)}")
                processing_status = "error"
                raise Exception(f"All video assembly attempts failed: {str(assembly_error)}")
        
    except Exception as e:
        logger.error(f"Advanced editing pipeline error: {str(e)}")
        processing_status = "error"

@app.get("/status")
async def get_status():
    """Get real processing status."""
    global processing_status, processing_progress, current_videos, current_audio
    
    return {
        "status": processing_status,
        "progress": processing_progress,
        "videos_count": len(current_videos),
        "has_audio": current_audio is not None,
        "message": f"Phase {processing_progress//15 + 1}/6: {processing_status}"
    }

@app.get("/download")
async def download_video():
    """Download the final video."""
    if processing_status != "completed":
        raise HTTPException(status_code=400, detail="Video not ready for download")
    
    output_path = os.path.join(settings.OUTPUT_DIR, "final_video.mp4")
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="edited_video.mp4"
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
        access_log=False
    ) 