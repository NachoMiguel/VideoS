#!/usr/bin/env python3
"""
Server Management Script for Video Editing App
Handles port conflicts, process management, and graceful shutdowns.
"""

import os
import sys
import subprocess
import time
import signal
import socket
import psutil
from pathlib import Path

class ServerManager:
    def __init__(self, default_port=8001):
        self.default_port = default_port
        self.server_process = None
        
    def find_available_port(self, start_port=8001, max_attempts=10):
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            if self.is_port_available(port):
                return port
        return None
    
    def is_port_available(self, port):
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False
    
    def kill_process_on_port(self, port):
        """Kill any process using the specified port."""
        try:
            # Find process using the port
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    connections = proc.info['connections']
                    for conn in connections:
                        if conn.laddr.port == port:
                            print(f"🔄 Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                            proc.terminate()
                            proc.wait(timeout=5)
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
        except Exception as e:
            print(f"⚠️ Error killing process on port {port}: {e}")
        return False
    
    def start_server(self, port=None, host="0.0.0.0"):
        """Start the video editing server."""
        if port is None:
            port = self.find_available_port(self.default_port)
            if port is None:
                print("❌ No available ports found")
                return False
        
        if not self.is_port_available(port):
            print(f"⚠️ Port {port} is in use, attempting to free it...")
            if self.kill_process_on_port(port):
                time.sleep(2)  # Wait for process to fully terminate
                if not self.is_port_available(port):
                    print(f"❌ Could not free port {port}")
                    return False
            else:
                print(f"❌ Could not kill process on port {port}")
                return False
        
        print(f"🚀 Starting Video Editing App on {host}:{port}")
        
        try:
            # Start the server process
            cmd = [sys.executable, "main.py", "--port", str(port), "--host", host]
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment to see if it starts successfully
            time.sleep(3)
            
            if self.server_process.poll() is None:
                print(f"✅ Server started successfully on port {port}")
                print(f"📊 Process ID: {self.server_process.pid}")
                print(f"🌐 Access URL: http://localhost:{port}")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                print(f"❌ Server failed to start:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the running server."""
        if self.server_process and self.server_process.poll() is None:
            print("🛑 Stopping server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                print("✅ Server stopped successfully")
                return True
            except subprocess.TimeoutExpired:
                print("⚠️ Server didn't stop gracefully, forcing...")
                self.server_process.kill()
                return True
            except Exception as e:
                print(f"❌ Error stopping server: {e}")
                return False
        else:
            print("ℹ️ No server process running")
            return True
    
    def restart_server(self, port=None, host="0.0.0.0"):
        """Restart the server."""
        print("🔄 Restarting server...")
        self.stop_server()
        time.sleep(2)
        return self.start_server(port, host)
    
    def get_server_status(self):
        """Get the current server status."""
        if self.server_process and self.server_process.poll() is None:
            return {
                "status": "running",
                "pid": self.server_process.pid,
                "port": self.default_port
            }
        else:
            return {
                "status": "stopped",
                "pid": None,
                "port": None
            }

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Editing App Server Manager")
    parser.add_argument("action", choices=["start", "stop", "restart", "status"], 
                       help="Action to perform")
    parser.add_argument("--port", type=int, default=8001, 
                       help="Port to use (default: 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    manager = ServerManager(args.port)
    
    if args.action == "start":
        success = manager.start_server(args.port, args.host)
        if success:
            print("🎉 Server management completed successfully")
        else:
            print("💥 Server management failed")
            sys.exit(1)
    
    elif args.action == "stop":
        manager.stop_server()
    
    elif args.action == "restart":
        success = manager.restart_server(args.port, args.host)
        if not success:
            sys.exit(1)
    
    elif args.action == "status":
        status = manager.get_server_status()
        print(f"Server Status: {status['status']}")
        if status['pid']:
            print(f"Process ID: {status['pid']}")
            print(f"Port: {status['port']}")

if __name__ == "__main__":
    main() 