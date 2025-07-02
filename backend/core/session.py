from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path

from core.config import settings
from core.logger import logger
from core.exceptions import SessionNotFoundError, SessionExpiredError

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.status = "initialized"
        self.error = None
        self.output_file = None
        self.video_results = {}
        self.selected_scenes = []
        self.metadata = {}
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """Check if session has expired."""
        return (datetime.now() - self.last_activity) > timedelta(minutes=timeout_minutes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "status": self.status,
            "error": str(self.error) if self.error else None,
            "output_file": self.output_file,
            "video_results": self.video_results,
            "selected_scenes": self.selected_scenes,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create session from dictionary."""
        session = cls(data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.status = data["status"]
        session.error = data["error"]
        session.output_file = data["output_file"]
        session.video_results = data["video_results"]
        session.selected_scenes = data["selected_scenes"]
        session.metadata = data["metadata"]
        return session

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.cleanup_interval_minutes = 60
        self.session_timeout_minutes = 60
        self._load_sessions()
    
    def create_session(self) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        self.sessions[session_id] = session
        self._save_sessions()
        return session
    
    def get_session(self, session_id: str) -> Session:
        """Get session by ID."""
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        if session.is_expired(self.session_timeout_minutes):
            self.cleanup_session(session_id)
            raise SessionExpiredError(f"Session {session_id} has expired")
        
        session.update_activity()
        self._save_sessions()
        return session
    
    def update_session(self, session_id: str, **updates) -> Session:
        """Update session attributes."""
        session = self.get_session(session_id)
        
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.update_activity()
        self._save_sessions()
        return session
    
    def cleanup_session(self, session_id: str):
        """Clean up session and its resources."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Clean up output file
        if session.output_file and Path(session.output_file).exists():
            try:
                Path(session.output_file).unlink()
            except Exception as e:
                logger.error(f"Failed to delete output file: {e}")
        
        # Remove session
        del self.sessions[session_id]
        self._save_sessions()
    
    def cleanup_expired_sessions(self):
        """Clean up all expired sessions."""
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if session.is_expired(self.session_timeout_minutes)
        ]
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
    
    def _save_sessions(self):
        """Save sessions to file."""
        try:
            sessions_file = Path(settings.temp_dir) / "sessions.json"
            sessions_data = {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            }
            
            with open(sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    def _load_sessions(self):
        """Load sessions from file."""
        try:
            sessions_file = Path(settings.temp_dir) / "sessions.json"
            if not sessions_file.exists():
                return
            
            with open(sessions_file, 'r') as f:
                sessions_data = json.load(f)
            
            self.sessions = {
                session_id: Session.from_dict(data)
                for session_id, data in sessions_data.items()
            }
            
            # Clean up expired sessions on load
            self.cleanup_expired_sessions()
            
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            self.sessions = {}

# Global session manager instance
manager = SessionManager() 