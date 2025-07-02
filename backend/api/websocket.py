from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Set, Optional, Any
import json
import asyncio
from datetime import datetime
from core.logger import logger

websocket_router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Connect a client to a session."""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        
        self.active_connections[session_id].add(websocket)
        logger.info(f"Client connected to session {session_id}")
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Disconnect a client from a session."""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
            
            logger.info(f"Client disconnected from session {session_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
    
    async def broadcast(self, message: str, session_id: str):
        """Broadcast a message to all clients in a session."""
        if session_id not in self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast message: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection, session_id)
    
    async def send_progress(
        self,
        session_id: str,
        progress: int,
        task: str,
        message: str
    ):
        """Send a progress update to all clients in a session."""
        data = {
            "type": "progress",
            "task": task,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(json.dumps(data), session_id)
    
    async def send_completion(
        self,
        session_id: str,
        output_file: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send a completion notification to all clients in a session."""
        data = {
            "type": "completed",
            "output_file": output_file,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(json.dumps(data), session_id)
    
    async def send_error(
        self,
        session_id: str,
        error: str,
        error_code: Optional[str] = None
    ):
        """Send an error notification to all clients in a session."""
        data = {
            "type": "error",
            "error": error,
            "error_code": error_code,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(json.dumps(data), session_id)
    
    async def send_status(
        self,
        session_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Send a status update to all clients in a session."""
        data = {
            "type": "status",
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(json.dumps(data), session_id)

# Global connection manager instance
manager = ConnectionManager()

@websocket_router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            message = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if message == "ping":
                await websocket.send_text("pong")
            else:
                # Handle other client messages if needed
                logger.debug(f"Received message from {session_id}: {message}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        logger.info(f"WebSocket disconnected normally for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
        manager.disconnect(websocket, session_id) 