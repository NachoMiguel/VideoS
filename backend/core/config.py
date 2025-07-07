import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project Settings
    PROJECT_NAME: str = "AI Video Slicer"
    API_V1_STR: str = "/api/v1"
    
    # API Keys
    openai_api_key: str = ""
    elevenlabs_api_key_1: str = ""
    elevenlabs_api_key_2: str = ""
    elevenlabs_api_key_3: str = ""
    elevenlabs_api_key_4: str = ""
    google_custom_search_api_key: str = ""
    google_custom_search_engine_id: str = ""
    youtube_api_key: str = ""
    
    # ElevenLabs Service Settings
    elevenlabs_voice_id: str = "nPczCjzI2devNBz1zQrb"  # Default Bella voice
    elevenlabs_model_id: str = "eleven_monolingual_v1"
    elevenlabs_timeout: int = 30
    elevenlabs_retry_attempts: int = 3
    max_credits_per_account: int = 10000
    credit_warning_threshold: float = 0.8
    
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
    
    # InsightFace Settings
    insightface_model_name: str = "buffalo_l"  # Model to use (buffalo_l, buffalo_m, buffalo_s)
    insightface_providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # ONNX providers
    insightface_det_size: tuple = (640, 640)  # Detection input size
    insightface_det_thresh: float = 0.5  # Detection threshold
    insightface_rec_thresh: float = 0.6  # Recognition threshold
    insightface_face_align: bool = True  # Enable face alignment
    insightface_gpu_memory_fraction: float = 0.5  # GPU memory fraction to use
    
    # Face Detection Settings (Legacy - keeping for backward compatibility)
    min_face_confidence: float = 0.5  # Minimum confidence for face detection
    parallel_face_detection: bool = True  # Enable parallel processing
    face_detection_batch_size: int = 10  # Number of frames to process in parallel
    enable_caching: bool = True  # Enable caching of face detection results
    cache_ttl_hours: int = 24  # Cache time-to-live in hours
    max_workers: int = 4  # Maximum number of parallel workers
    
    # Face Recognition Settings
    face_embedding_size: int = 512  # Size of face embeddings
    face_similarity_threshold: float = 0.6  # Threshold for face matching
    max_faces_per_frame: int = 10  # Maximum faces to detect per frame
    face_quality_threshold: float = 0.3  # Minimum face quality score
    enable_face_quality_filter: bool = True  # Filter low quality faces
    
    # Character Training Settings
    min_character_images: int = 3  # Minimum images needed to train character
    max_character_images: int = 20  # Maximum images to use per character
    character_embedding_cache_size: int = 1000  # Max character embeddings to cache
    character_training_batch_size: int = 5  # Batch size for character training
    
    # Test Mode
    test_mode_enabled: bool = False
    development_skip_mode: bool = False
    
    # Test Mode Configuration
    use_saved_script: bool = False
    use_known_characters: bool = False
    use_saved_audio: bool = False
    
    # Test Data Directories
    test_scripts_dir: str = "test_data/scripts"
    test_audio_dir: str = "test_data/audio"
    test_characters_dir: str = "test_data/characters"
    
    # Default Prompts
    default_script_rewrite_prompt: str = """
Rewrite the following video transcript into an engaging, natural-sounding script suitable for video narration.
Make it more conversational and add appropriate pauses and emphasis.
Keep the core message intact while making it more engaging for viewers.

Transcript:
{transcript}

Please provide only the rewritten script without any additional commentary.
"""
    
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
    
    # Sentry Configuration
    sentry_dsn: str = ""
    sentry_environment: str = "development"
    sentry_release: Optional[str] = None
    sentry_sample_rate: float = 1.0
    sentry_traces_sample_rate: float = 0.1
    sentry_enabled: bool = True
    
    # Video processing settings
    max_video_file_size_bytes: int = 419430400  # 400MB
    max_videos_per_session: int = 3
    max_concurrent_processing: int = 3  # Maximum concurrent video processing tasks
    
    # Scene settings
    concurrent_api_calls: int = 3
    
    # Session Management
    session_timeout_minutes: int = 60  # Session timeout in minutes
    cleanup_interval_minutes: int = 30  # Cleanup interval in minutes
    max_active_sessions: int = 50  # Maximum number of active sessions
    
    # Performance Monitoring
    performance_monitoring_enabled: bool = True
    performance_report_interval_minutes: int = 60  # Generate performance report every hour
    max_operation_history: int = 1000  # Keep last 1000 operations in memory
    
    # System Resource Limits
    max_memory_usage_percent: float = 85.0  # Maximum memory usage before warnings
    max_disk_usage_percent: float = 90.0  # Maximum disk usage before warnings
    max_cpu_usage_percent: float = 90.0  # Maximum CPU usage before warnings
    
    class Config:
        env_file = Path(__file__).parent.parent / ".env"  # Absolute path to backend/.env
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env
        
    @property
    def elevenlabs_api_keys(self) -> List[str]:
        """Get all configured ElevenLabs API keys as a list."""
        keys = []
        for key in [self.elevenlabs_api_key_1, self.elevenlabs_api_key_2, 
                    self.elevenlabs_api_key_3, self.elevenlabs_api_key_4]:
            if key:  # Only add non-empty keys
                keys.append(key)
        return keys
    
    @property
    def has_required_keys(self) -> bool:
        """Check if all required API keys are present."""
        return bool(
            self.openai_api_key and 
            self.elevenlabs_api_keys and
            self.youtube_api_key
        )
    
    @property
    def has_partial_elevenlabs_config(self) -> bool:
        """Check if at least one ElevenLabs API key is configured."""
        return len(self.elevenlabs_api_keys) > 0
    
    @property
    def elevenlabs_config_status(self) -> dict:
        """Get detailed ElevenLabs configuration status."""
        return {
            "total_keys": len(self.elevenlabs_api_keys),
            "has_keys": len(self.elevenlabs_api_keys) > 0,
            "optimal_keys": len(self.elevenlabs_api_keys) >= 2,  # At least 2 for rotation
            "max_keys": len(self.elevenlabs_api_keys) == 4,  # Maximum supported
            "missing_keys": 4 - len(self.elevenlabs_api_keys),
            "voice_id_configured": bool(self.elevenlabs_voice_id),
            "model_id_configured": bool(self.elevenlabs_model_id),
        }
    
    def validate_elevenlabs_config(self) -> tuple[bool, str]:
        """Validate ElevenLabs configuration and return status."""
        if not self.elevenlabs_api_keys:
            return False, "No ElevenLabs API keys configured. Please set ELEVENLABS_API_KEY_1 through ELEVENLABS_API_KEY_4 in your .env file."
        
        if len(self.elevenlabs_api_keys) == 1:
            return True, f"ElevenLabs configured with 1 API key (limited throughput). Consider adding more keys for better performance."
        
        if len(self.elevenlabs_api_keys) < 4:
            return True, f"ElevenLabs configured with {len(self.elevenlabs_api_keys)} API keys. You can add up to {4 - len(self.elevenlabs_api_keys)} more for optimal performance."
        
        return True, f"ElevenLabs optimally configured with {len(self.elevenlabs_api_keys)} API keys."
    
    @property
    def parallel_processing_enabled(self) -> bool:
        """Check if parallel processing is enabled and feasible."""
        return (
            self.parallel_processing and 
            self.max_workers > 1
        )
    
    @property
    def insightface_gpu_enabled(self) -> bool:
        """Check if GPU is available and enabled for InsightFace."""
        return "CUDAExecutionProvider" in self.insightface_providers
    
    # Backward compatibility properties for screaming snake case
    @property
    def TEST_MODE_ENABLED(self) -> bool:
        """Backward compatibility property for TEST_MODE_ENABLED."""
        return self.test_mode_enabled
    
    @property
    def USE_SAVED_SCRIPT(self) -> bool:
        """Backward compatibility property for USE_SAVED_SCRIPT."""
        return self.use_saved_script
    
    @property
    def USE_KNOWN_CHARACTERS(self) -> bool:
        """Backward compatibility property for USE_KNOWN_CHARACTERS."""
        return self.use_known_characters
    
    @property
    def USE_SAVED_AUDIO(self) -> bool:
        """Backward compatibility property for USE_SAVED_AUDIO."""
        return self.use_saved_audio
    
    @property
    def DEFAULT_SCRIPT_REWRITE_PROMPT(self) -> str:
        """Backward compatibility property for DEFAULT_SCRIPT_REWRITE_PROMPT."""
        return self.default_script_rewrite_prompt

# Global settings instance
settings = Settings() 