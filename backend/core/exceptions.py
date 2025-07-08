"""
Custom exceptions for the AI Video Slicer application.
These exceptions help with proper error handling and user feedback.
"""

from typing import Optional

class AIVideoSlicerException(Exception):
    """Base exception class for AI Video Slicer."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        
        # REMOVED: Sentry integration (was causing import errors)
        # set_context("custom_exception", {
        #     "exception_type": type(self).__name__,
        #     "message": message,
        #     "error_code": error_code
        # })
        # set_tag("exception_category", "ai_video_slicer")
        # 
        # # Only capture if Sentry is enabled
        # if sentry_sdk.Hub.current.client:
        #     capture_exception(self)
            
        super().__init__(self.message)

class VideoProcessingError(AIVideoSlicerException):
    """Raised when video processing fails."""
    def __init__(self, message: str):
        super().__init__(message, "VIDEO_PROCESSING_ERROR")

class VideoValidationError(AIVideoSlicerException):
    """Raised when uploaded videos fail validation."""
    pass

class VideoCorruptionError(VideoValidationError):
    """Raised when video files are corrupted."""
    pass

class VideoFormatError(VideoValidationError):
    """Raised when video format is not supported."""
    pass

class VideoSizeError(VideoValidationError):
    """Raised when video exceeds size limits."""
    pass

class VideoDurationError(VideoValidationError):
    """Raised when video exceeds duration limits."""
    pass

class VideoResolutionError(VideoValidationError):
    """Raised when video resolution is too low."""
    pass

class FaceDetectionError(AIVideoSlicerException):
    """Raised when face detection fails."""
    def __init__(self, message: str):
        super().__init__(message, "FACE_DETECTION_ERROR")

class FaceRecognitionError(AIVideoSlicerException):
    """Raised when face recognition fails."""
    pass

class CharacterTrainingError(AIVideoSlicerException):
    """Raised when character model training fails."""
    pass

class ScriptGenerationError(AIVideoSlicerException):
    """Raised when script generation fails."""
    pass

class ScriptProcessingError(AIVideoSlicerException):
    """Raised when script processing fails."""
    pass

class TranscriptExtractionError(AIVideoSlicerException):
    """Raised when YouTube transcript extraction fails."""
    pass

class AudioProcessingError(AIVideoSlicerException):
    """Raised when audio processing fails."""
    pass

class AudioGenerationError(AudioProcessingError):
    """Raised when TTS audio generation fails."""
    pass

class AudioStitchingError(AudioProcessingError):
    """Raised when audio stitching fails."""
    pass

class SceneAnalysisError(AIVideoSlicerException):
    """Raised when scene analysis fails."""
    pass

class SceneSelectionError(AIVideoSlicerException):
    """Raised when scene selection fails."""
    pass

class VideoAssemblyError(AIVideoSlicerException):
    """Raised when final video assembly fails."""
    pass

class APIError(AIVideoSlicerException):
    """Base class for external API errors."""
    def __init__(self, message: str, service: str, error_code: str = None):
        self.service = service
        super().__init__(message, error_code)

class OpenAIError(APIError):
    """Raised when OpenAI API calls fail."""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, "OpenAI", error_code)

class ElevenLabsError(APIError):
    """Raised when ElevenLabs API calls fail."""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, "ElevenLabs", error_code)

class GoogleSearchError(APIError):
    """Raised when Google Custom Search API calls fail."""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, "Google Search", error_code)

class YouTubeAPIError(APIError):
    """Raised when YouTube API calls fail."""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, "YouTube", error_code)

class CreditLimitError(APIError):
    """Raised when API credit limits are exceeded."""
    def __init__(self, service: str, message: str = None):
        if not message:
            message = f"{service} API credit limit exceeded"
        super().__init__(message, service, "CREDIT_LIMIT_EXCEEDED")

class FileValidationError(AIVideoSlicerException):
    """Raised when uploaded files are invalid."""
    pass

class MemoryError(AIVideoSlicerException):
    """Raised when memory management fails."""
    pass

class SessionError(Exception):
    """Raised when there are session-related issues."""
    pass

class SessionNotFoundError(AIVideoSlicerException):
    """Raised when session is not found."""
    def __init__(self, message: str):
        super().__init__(message, "SESSION_NOT_FOUND")

class SessionExpiredError(SessionError):
    """Raised when a session has expired."""
    pass

class TestModeError(AIVideoSlicerException):
    """Raised when test mode operations fail."""
    pass

class ConfigurationError(Exception):
    """Raised when there are configuration issues."""
    pass

class ParallelProcessingError(AIVideoSlicerException):
    """Raised when parallel processing fails."""
    def __init__(self, message: str):
        super().__init__(message, "PARALLEL_PROCESSING_ERROR")

class VideoNotFoundError(Exception):
    """Raised when a video cannot be found or accessed."""
    pass

class TranscriptNotFoundError(Exception):
    """Raised when video transcript cannot be found."""
    pass

class AIGenerationError(Exception):
    """Raised when AI generation (OpenAI, ElevenLabs) fails."""
    pass

class VoiceNotFoundError(Exception):
    """Raised when a voice cannot be found or created."""
    pass

class CharacterNotFoundError(Exception):
    """Raised when a character cannot be found in the video."""
    pass

class InvalidScriptError(Exception):
    """Raised when the script format is invalid."""
    pass

class APILimitError(Exception):
    """Raised when API rate limits are exceeded."""
    pass

class InvalidFileError(Exception):
    """Raised when an uploaded file is invalid."""
    pass

class WebSocketError(Exception):
    """Raised when WebSocket communication fails."""
    pass

class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass

class CacheError(Exception):
    """Raised when cache operations fail."""
    pass

class SceneDetectionError(AIVideoSlicerException):
    """Raised when scene detection fails."""
    def __init__(self, message: str):
        super().__init__(message, "SCENE_DETECTION_ERROR")

class VideoCompilationError(AIVideoSlicerException):
    """Raised when video compilation fails."""
    def __init__(self, message: str):
        super().__init__(message, "VIDEO_COMPILATION_ERROR")

class InvalidVideoError(AIVideoSlicerException):
    """Raised when video file is invalid or corrupted."""
    def __init__(self, message: str):
        super().__init__(message, "INVALID_VIDEO_ERROR")

class ImageSearchError(AIVideoSlicerException):
    """Exception raised during image search operations."""
    def __init__(self, message: str):
        super().__init__(message, "IMAGE_SEARCH_ERROR")

class CreditExhaustionError(AIVideoSlicerException):
    """Raised when API credits are exhausted."""
    def __init__(self, message: str):
        super().__init__(message, "CREDIT_EXHAUSTED")

# Error code mappings for frontend
ERROR_CODES = {
    # Video Errors
    "VIDEO_CORRUPTED": "The uploaded video file is corrupted and cannot be processed.",
    "VIDEO_TOO_LARGE": "The video file exceeds the maximum size limit of 400MB.",
    "VIDEO_TOO_LONG": "The video duration exceeds the maximum limit of 40 minutes.",
    "VIDEO_RESOLUTION_LOW": "The video resolution is below the minimum requirement of 720p.",
    "VIDEO_FORMAT_UNSUPPORTED": "The video format is not supported. Please use MP4, AVI, or MOV.",
    
    # API Errors
    "OPENAI_CREDIT_LIMIT": "OpenAI API credit limit exceeded. Please try again later.",
    "ELEVENLABS_CREDIT_LIMIT": "ElevenLabs API credit limit exceeded. Please try again later.",
    "API_KEY_INVALID": "API key is invalid or expired.",
    "API_RATE_LIMIT": "API rate limit exceeded. Please wait before retrying.",
    
    # Processing Errors
    "FACE_DETECTION_FAILED": "Face detection failed. Falling back to simple video editing.",
    "CHARACTER_TRAINING_FAILED": "Character recognition training failed.",
    "AUDIO_GENERATION_FAILED": "Audio generation failed. Please try again.",
    "SCENE_ANALYSIS_FAILED": "Scene analysis failed during processing.",
    "VIDEO_ASSEMBLY_FAILED": "Final video assembly failed.",
    
    # Session Errors
    "SESSION_NOT_FOUND": "Session not found or expired. Please start a new session.",
    "SESSION_EXPIRED": "Session has expired. Please start a new session.",
    
    # General Errors
    "PROCESSING_FAILED": "Video processing failed. Please try again.",
    "MEMORY_ERROR": "Insufficient memory to process the request.",
    "CONFIGURATION_ERROR": "System configuration error. Please contact support.",
}

def get_user_friendly_message(error_code: str) -> str:
    """Get user-friendly error message for error codes."""
    messages = {
        "VIDEO_PROCESSING_ERROR": "Failed to process the video. Please try again or use a different video file.",
        "SCENE_DETECTION_ERROR": "Could not detect scenes in the video. The video might be too short or have insufficient variation.",
        "FACE_DETECTION_ERROR": "Failed to detect faces in the video. Please ensure the video has clear, well-lit faces.",
        "VIDEO_COMPILATION_ERROR": "Failed to create the final video compilation. Please try again with different scenes.",
        "INVALID_VIDEO_ERROR": "The video file is invalid or corrupted. Please upload a valid video file.",
        "SESSION_NOT_FOUND": "Session not found or expired. Please start a new session.",
        "PARALLEL_PROCESSING_ERROR": "Failed to process videos in parallel. Please try again with fewer videos."
    }
    return messages.get(error_code, "An unexpected error occurred. Please try again.") 