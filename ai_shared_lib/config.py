import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # Google API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID", "")
    google_custom_search_api_key: str = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "")
    google_custom_search_engine_id: str = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID", "")
    
    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Directories
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "cache")
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # InsightFace Configuration
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    INSIGHTFACE_DET_THRESH: float = 0.5
    INSIGHTFACE_REC_THRESH: float = 0.6
    INSIGHTFACE_DET_SIZE: tuple = (640, 640)
    FACE_QUALITY_THRESHOLD: float = 0.7
    FACE_SIMILARITY_THRESHOLD: float = 0.6
    CHARACTER_CONFIDENCE_THRESHOLD: float = 0.7
    
    # Processing Settings
    MIN_CHARACTER_IMAGES: int = 3
    MAX_CHARACTER_IMAGES: int = 10
    MIN_SCENE_DURATION: float = 2.0
    MAX_SCENE_DURATION: float = 30.0
    FACE_DETECTION_BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4
    ENABLE_CACHING: bool = True
    CACHE_TTL_HOURS: int = 24

    # Fix InsightFace configuration naming
    insightface_model_name: str = "buffalo_l"  # Add lowercase version
    insightface_det_thresh: float = 0.5
    insightface_rec_thresh: float = 0.6
    insightface_det_size: tuple = (640, 640)

settings = Settings() 