from fastapi import WebSocket
from typing import Dict, List, Optional
import json
import logging
import asyncio
from .exceptions import WebSocketError

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket, session_id: str):
        """Connect a new WebSocket client."""
        try:
            await websocket.accept()
            if session_id not in self.active_connections:
                self.active_connections[session_id] = []
            self.active_connections[session_id].append(websocket)
            self.logger.info(f"New WebSocket connection for session {session_id}")
        except Exception as e:
            self.logger.error(f"Failed to connect WebSocket for session {session_id}: {str(e)}")
            raise WebSocketError(f"WebSocket connection failed: {str(e)}")

    async def disconnect(self, websocket: WebSocket, session_id: str):
        """Disconnect a WebSocket client."""
        try:
            if session_id in self.active_connections:
                self.active_connections[session_id].remove(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            self.logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            self.logger.error(f"Error during WebSocket disconnect for session {session_id}: {str(e)}")

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast a message to all clients in a session."""
        if session_id not in self.active_connections:
            return

        disconnected = []
        for websocket in self.active_connections[session_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                self.logger.error(f"Failed to send message to WebSocket in session {session_id}: {str(e)}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            await self.disconnect(websocket, session_id)

    async def send_progress_update(self, session_id: str, progress: float, status: str, metadata: Optional[Dict] = None):
        """Send a progress update to all clients in a session."""
        message = {
            'type': 'progress',
            'progress': progress,
            'status': status
        }
        if metadata:
            message['metadata'] = metadata

        await self.broadcast_to_session(session_id, message)

    async def send_error(self, session_id: str, error: str):
        """Send an error message to all clients in a session."""
        message = {
            'type': 'error',
            'error': error
        }
        await self.broadcast_to_session(session_id, message)

    async def send_completion(self, session_id: str, data: Dict):
        """Send a completion message to all clients in a session."""
        message = {
            'type': 'completion',
            'result': data
        }
        await self.broadcast_to_session(session_id, message)

    async def send_scenes(self, session_id: str, scenes: List[Dict]):
        """Send scene analysis data to all clients in a session."""
        message = {
            'type': 'scenes',
            'scenes': scenes
        }
        await self.broadcast_to_session(session_id, message)

    def get_active_sessions(self) -> List[str]:
        """Get a list of all active session IDs."""
        return list(self.active_connections.keys())

    async def close_all_connections(self):
        """Close all active WebSocket connections."""
        for session_id in list(self.active_connections.keys()):
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.close()
                except Exception as e:
                    self.logger.error(f"Error closing WebSocket in session {session_id}: {str(e)}")
            del self.active_connections[session_id] 