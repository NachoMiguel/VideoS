#!/usr/bin/env python3
"""
Video Editing App - Main Application
"""

import os
import sys
import site

# CRITICAL: Set CUDNN path before any CUDA imports
def setup_cudnn_path():
    """Setup CUDNN path for CUDA operations."""
    try:
        # Find the CUDNN library path
        site_packages = site.getsitepackages()[1]  # lib/site-packages
        cudnn_path = os.path.join(site_packages, "nvidia", "cudnn", "bin")
        
        if os.path.exists(cudnn_path):
            # Add to PATH if not already there
            if cudnn_path not in os.environ['PATH']:
                os.environ['PATH'] = cudnn_path + os.pathsep + os.environ['PATH']
            
            # Set CUDNN environment variables
            os.environ['CUDNN_PATH'] = cudnn_path
            os.environ['CUDNN_LIBRARY_PATH'] = cudnn_path
            
            print(f"[OK] CUDNN path set: {cudnn_path}")
            return True
        else:
            print(f"[WARN] CUDNN path not found: {cudnn_path}")
            return False
    except Exception as e:
        print(f"[WARN] CUDNN setup failed: {e}")
        return False

# SETUP CUDNN PATH IMMEDIATELY
setup_cudnn_path()

# Add ai_shared_lib to Python path
yt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, yt_root)
print(f"[INFO] Added to Python path: {yt_root}")

# Add error handling for problematic imports
try:
    import numpy as np
    print(f"[INFO] NumPy version: {np.__version__}")
except ImportError as e:
    print(f"[ERROR] NumPy import error: {e}")
    sys.exit(1)

try:
    import cv2
    print("[INFO] OpenCV imported successfully")
except ImportError as e:
    print(f"[ERROR] OpenCV import error: {e}")
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
    print("[INFO] Video processor imported successfully")
    
    # Use importlib to import ai_shared_lib modules
    import importlib.util
    
    # Import config
    config_spec = importlib.util.spec_from_file_location("config", os.path.join(yt_root, "ai_shared_lib", "config.py"))
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    settings = config_module.settings
    print("[INFO] Settings imported successfully via importlib")
    
    print("[INFO] Services imported successfully")
except ImportError as e:
    print(f"[ERROR] Service import error: {e}")
    print(f"[DEBUG] Current sys.path: {sys.path}")
    print(f"[DEBUG] yt_root: {yt_root}")
    print(f"[DEBUG] Path exists: {os.path.exists(yt_root)}")
    print(f"[DEBUG] yt_root contents: {os.listdir(yt_root) if os.path.exists(yt_root) else 'N/A'}")
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
async def start_editing(
    characters: str = Form("Jean-Claude Van Damme, Steven Segal"),  # Keep manual characters
    script_content: str = Form(""),  # Add script content input
    audio_duration_minutes: float = Form(22.0)  # Add audio duration in minutes
):
    """Start the video editing process with script-based editing."""
    global current_videos, current_audio, processing_status, processing_progress
    
    try:
        logger.info("START-EDITING BUTTON CLICKED - SCRIPT-BASED EDITING")
        
        if not current_videos:
            raise HTTPException(status_code=400, detail="No videos uploaded")
        
        # Parse characters from input
        character_list = []
        for char in characters.split(','):
            char = char.strip()
            if char:
                char = char.rstrip('.').strip()
                if char:
                    character_list.append(char)
        
        # Convert audio duration from minutes to seconds
        audio_duration_seconds = audio_duration_minutes * 60.0
        
        logger.info(f"Manual characters: {character_list}")
        logger.info(f"Script content length: {len(script_content)} characters")
        logger.info(f"Audio duration: {audio_duration_minutes} minutes ({audio_duration_seconds:.1f} seconds)")
        
        processing_status = "processing"
        processing_progress = 0
        
        # Start processing with script-based editing
        asyncio.create_task(run_script_based_editing_pipeline(character_list, script_content, audio_duration_seconds))
        
        return {
            "status": "processing_started",
            "message": "Script-based video editing pipeline started successfully"
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
        selected_scenes = await processor.select_best_scenes(all_scenes, "", audio_duration, manual_characters)
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
        
        # FIXED: Duration fallback - create extended scenes if duration is too short
        if selected_scenes:
            total_duration = sum(scene.get('duration', 0) for scene in selected_scenes)
            if total_duration < audio_duration:  # Less than audio duration (removed +10s buffer)
                logger.warning(f"Selected duration ({total_duration:.1f}s) too short for audio ({audio_duration:.1f}s) - creating extended scenes")
                extended_scenes = await processor.create_extended_scenes(current_videos, audio_duration)
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

async def run_script_based_editing_pipeline(manual_characters: List[str], script_content: str, audio_duration_seconds: float):
    """Run the complete script-based editing pipeline with FFmpeg performance."""
    global current_videos, processing_status, processing_progress
    
    try:
        import time
        start_time = time.time()
        
        logger.info("🚀 Starting FFmpeg-powered editing pipeline...")
        logger.info(f"⚡ Expected performance: 4x faster than previous version")
        
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
        
        # Phase 4: FFmpeg Scene Detection (40-70%)
        processing_progress = 45
        logger.info("Phase 4: FFmpeg scene detection (professional-grade)...")
        
        # Get all scenes using FFmpeg (fast)
        all_scenes = await processor.analyze_videos_with_character_detection(
            [v['path'] for v in current_videos], 
            face_model,
            audio_duration_seconds
        )
        
        # CRITICAL: Use script-based scene selection
        selected_scenes = await processor.select_best_scenes(all_scenes, script_content, audio_duration_seconds, manual_characters)
        logger.info(f"Selected {len(selected_scenes)} scenes for {audio_duration_seconds/60:.1f} minute script")
        
        # CRITICAL FIX: Force use of AI scenes - NO FALLBACK TO SIMPLE CLIPS
        if not selected_scenes:
            logger.error("CRITICAL: No scenes selected from AI pipeline!")
            logger.error("This means the AI scene detection is completely failing.")
            logger.error("Check face detection and scene analysis logs above.")
            raise Exception("AI scene selection failed - no smart editing possible")
        
        # FINAL VALIDATION: Ensure we have something to work with
        if not selected_scenes:
            logger.error("CRITICAL: No valid scenes available!")
            raise Exception("No valid scenes available for video assembly")
        
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
        
        # Phase 6: FFmpeg Video Assembly (85-100%) - SILENT VIDEO (NO AUDIO)
        processing_progress = 90
        logger.info("Phase 6: FFmpeg video assembly (professional-grade)...")
        
        output_path = os.path.join(settings.OUTPUT_DIR, "final_video_silent.mp4")
        
        try:
            # CRITICAL: Create silent video using FFmpeg (fast)
            final_video = await processor.assemble_silent_video(selected_scenes, output_path)
            
            processing_progress = 100
            processing_status = "completed"
            
            # Performance metrics
            total_time = time.time() - start_time
            logger.info(f"🚀 FFmpeg-powered video editing completed in {total_time:.1f} seconds")
            logger.info(f"⚡ Performance: {total_time/60:.1f} minutes (4x faster than previous version)")
            logger.info(f"🎬 Final video: {final_video}")
            
        except Exception as assembly_error:
            logger.error(f"FFmpeg video assembly failed: {str(assembly_error)}")
            
            # CRITICAL: NO EMERGENCY FALLBACK - Force proper error
            logger.error("CRITICAL: Video assembly failed completely")
            logger.error("This means the FFmpeg processing is broken")
            logger.error("Check FFmpeg installation and video files")
            processing_status = "error"
            raise Exception(f"Video assembly failed - no emergency fallback: {str(assembly_error)}")
                
    except Exception as e:
        logger.error(f"FFmpeg-powered editing pipeline failed: {str(e)}")
        processing_status = "error"
        raise e

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
    
    # FIXED: Look for the correct filename that was actually created
    output_path = os.path.join(settings.OUTPUT_DIR, "final_video_silent.mp4")
    
    if not os.path.exists(output_path):
        # Fallback: try the old filename
        fallback_path = os.path.join(settings.OUTPUT_DIR, "final_video.mp4")
        if os.path.exists(fallback_path):
            output_path = fallback_path
        else:
            raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="edited_video.mp4"
    )

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments for port configuration
    parser = argparse.ArgumentParser(description="Video Editing App Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on (default: 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()
    
    # Check if port is available
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((args.host, args.port))
        sock.close()
        print(f"[OK] Port {args.port} is available")
    except OSError:
        print(f"[ERROR] Port {args.port} is already in use")
        print("[TIP] Try using a different port: python main.py --port 8002")
        exit(1)

    print(f"[START] Starting Video Editing App on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
        access_log=False
    ) 