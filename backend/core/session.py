from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path

from core.config import settings
from core.logger import logger
from core.exceptions import SessionNotFoundError, SessionExpiredError
import aiofiles
import asyncio

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
    def __init__(self, persist_sessions=False):  # Add parameter to control persistence
        self.sessions: Dict[str, Session] = {}
        self.cleanup_interval_minutes = 60
        self.session_timeout_minutes = 60
        self.persist_sessions = persist_sessions  # Control persistence behavior
        
        if self.persist_sessions:
            self._load_sessions()
    
    async def create_session(self, session_id: str = None, initial_data: Dict[str, Any] = None) -> Session:
        """Create a new session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session = Session(session_id)
        
        # Apply initial data if provided
        if initial_data:
            for key, value in initial_data.items():
                if hasattr(session, key):
                    setattr(session, key, value)
                else:
                    session.metadata[key] = value
        
        self.sessions[session_id] = session
        await self._save_sessions()
        logger.info(f"Created session {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Session:
        """Get session by ID."""
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        if session.is_expired(self.session_timeout_minutes):
            await self.cleanup_session(session_id)
            raise SessionExpiredError(f"Session {session_id} has expired")
        
        session.update_activity()
        await self._save_sessions()
        return session
    
    async def update_session(self, session_id: str, **updates) -> Session:
        """Update session attributes."""
        session = await self.get_session(session_id)
        
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
            else:
                session.metadata[key] = value
        
        session.update_activity()
        await self._save_sessions()
        logger.debug(f"Updated session {session_id}: {list(updates.keys())}")
        return session
    
    async def cleanup_session(self, session_id: str):
        """Clean up session and its resources."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Clean up output file (run in thread pool)
        if session.output_file and Path(session.output_file).exists():
            try:
                await asyncio.to_thread(Path(session.output_file).unlink)
                logger.info(f"Cleaned up output file: {session.output_file}")
            except Exception as e:
                logger.error(f"Failed to delete output file: {e}")
        
        # Clean up temp directory (run in thread pool)
        temp_dir = Path(settings.temp_dir) / session_id
        if temp_dir.exists():
            try:
                import shutil
                await asyncio.to_thread(shutil.rmtree, temp_dir)
                logger.info(f"Cleaned up session temp directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to delete session temp directory: {e}")
        
        # Remove session
        del self.sessions[session_id]
        await self._save_sessions()  # Make this async call
        logger.info(f"Cleaned up session {session_id}")
    
    async def cleanup_expired_sessions(self):
        """Clean up all expired sessions."""
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if session.is_expired(self.session_timeout_minutes)
        ]
        
        for session_id in expired_sessions:
            await self.cleanup_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _save_sessions(self):
        """Save sessions to file asynchronously."""
        if not self.persist_sessions:
            return  # Skip saving if persistence is disabled
        try:
            sessions_file = Path(settings.temp_dir) / "sessions.json"
            sessions_data = {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            }
            
            # Use async file writing
            async with aiofiles.open(sessions_file, 'w') as f:
                await f.write(json.dumps(sessions_data, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    def _load_sessions(self):
        """Load sessions from file."""
        if not self.persist_sessions:
            return  # Skip loading if persistence is disabled
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
            
            logger.info(f"Loaded {len(self.sessions)} sessions from file")
            
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            self.sessions = {}

# Global session manager instance with ephemeral sessions
manager = SessionManager(persist_sessions=False)  # Disable persistence 