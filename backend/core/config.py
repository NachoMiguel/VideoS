import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project Settings
    PROJECT_NAME: str = "AI Video Slicer"
    API_V1_STR: str = "/api/v1"
    
    # API Keys
    openai_api_key: str = ""
    elevenlabs_api_keys: List[str] = []
    google_custom_search_api_key: str = ""
    google_custom_search_engine_id: str = ""
    youtube_api_key: str = ""
    
    # File Upload Settings
    max_file_size: int = 400 * 1024 * 1024  # 400MB
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    upload_dir: str = "uploads"
    output_dir: str = "output"
    temp_dir: str = "temp"
    cache_dir: str = "cache"
    
    # Processing Settings
    max_video_duration: int = 1800  # 30 minutes
    min_scene_duration: float = 2.0  # seconds
    max_video_duration_seconds: int = 1800  # 30 minutes
    
    # Face Detection Settings
    min_face_confidence: float = 0.5  # Minimum confidence for face detection
    parallel_face_detection: bool = True  # Enable parallel processing
    face_detection_batch_size: int = 10  # Number of frames to process in parallel
    enable_caching: bool = True  # Enable caching of face detection results
    cache_ttl_hours: int = 24  # Cache time-to-live in hours
    max_workers: int = 4  # Maximum number of parallel workers
    
    # Test Mode
    test_mode_enabled: bool = False
    development_skip_mode: bool = False
    
    # Parallel Processing Settings
    parallel_processing: bool = True
    parallel_audio_generation: bool = True
    batch_size: int = 100
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    api_usage_log: str = "logs/api_usage.log"
    error_log: str = "logs/errors.log"
    
    # Video processing settings
    max_video_file_size_bytes: int = 419430400  # 400MB
    max_videos_per_session: int = 3
    
    # Scene settings
    concurrent_api_calls: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Parse ElevenLabs API keys from environment
        for i in range(1, 5):  # Support up to 4 accounts
            key = os.getenv(f"ELEVENLABS_API_KEY_{i}")
            if key:
                self.elevenlabs_api_keys.append(key)
        
        # Fallback to single key if no numbered keys found
        if not self.elevenlabs_api_keys:
            single_key = os.getenv("ELEVENLABS_API_KEY")
            if single_key:
                self.elevenlabs_api_keys.append(single_key)
    
    @property
    def has_required_keys(self) -> bool:
        """Check if all required API keys are present."""
        return bool(
            self.openai_api_key and 
            self.elevenlabs_api_keys and
            self.youtube_api_key
        )
    
    @property
    def parallel_processing_enabled(self) -> bool:
        """Check if parallel processing is enabled and feasible."""
        return (
            self.parallel_processing and 
            self.max_workers > 1
        )

# Global settings instance
settings = Settings() 