"""
Performance Monitoring System
Tracks system performance, memory usage, and processing times for optimization.
"""

import psutil
import time
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from core.config import settings
from core.logger import logger

@dataclass
class PerformanceMetrics:
    """Performance metrics for a session or operation."""
    session_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    file_size_mb: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    
    def complete(self, success: bool = True, error: str = None):
        """Mark the operation as complete."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error

class PerformanceMonitor:
    """Monitors system performance and tracks metrics."""
    
    def __init__(self):
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.completed_operations: List[PerformanceMetrics] = []
        self.system_metrics: List[Dict] = []
        self.max_history = 1000  # Keep last 1000 operations
        
    def start_operation(self, session_id: str, operation: str, file_size_mb: float = None) -> str:
        """Start tracking an operation."""
        operation_key = f"{session_id}_{operation}_{int(time.time())}"
        
        metrics = PerformanceMetrics(
            session_id=session_id,
            operation=operation,
            start_time=time.time(),
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage(),
            file_size_mb=file_size_mb
        )
        
        self.active_operations[operation_key] = metrics
        logger.debug(f"Started tracking operation: {operation_key}")
        return operation_key
    
    def complete_operation(self, operation_key: str, success: bool = True, error: str = None):
        """Complete an operation and move to history."""
        if operation_key not in self.active_operations:
            logger.warning(f"Operation key not found: {operation_key}")
            return
        
        metrics = self.active_operations[operation_key]
        metrics.complete(success, error)
        
        # Update final metrics
        metrics.memory_usage_mb = max(metrics.memory_usage_mb, self._get_memory_usage())
        metrics.cpu_usage_percent = self._get_cpu_usage()
        
        # Move to completed operations
        self.completed_operations.append(metrics)
        del self.active_operations[operation_key]
        
        # Maintain history limit
        if len(self.completed_operations) > self.max_history:
            self.completed_operations = self.completed_operations[-self.max_history:]
        
        logger.info(f"Completed operation {operation_key}: "
                   f"duration={metrics.duration:.2f}s, "
                   f"success={metrics.success}, "
                   f"memory={metrics.memory_usage_mb:.1f}MB")
    
    def get_session_performance(self, session_id: str) -> Dict:
        """Get performance metrics for a specific session."""
        session_ops = [
            op for op in self.completed_operations 
            if op.session_id == session_id
        ]
        
        if not session_ops:
            return {'session_id': session_id, 'operations': 0, 'total_duration': 0}
        
        total_duration = sum(op.duration or 0 for op in session_ops)
        avg_memory = sum(op.memory_usage_mb for op in session_ops) / len(session_ops)
        success_rate = sum(1 for op in session_ops if op.success) / len(session_ops)
        
        return {
            'session_id': session_id,
            'operations': len(session_ops),
            'total_duration': total_duration,
            'average_memory_mb': avg_memory,
            'success_rate': success_rate,
            'operations_detail': [asdict(op) for op in session_ops]
        }
    
    def get_system_performance(self) -> Dict:
        """Get current system performance metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'memory_usage_mb': self._get_memory_usage(),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'cpu_usage_percent': self._get_cpu_usage(),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'active_operations': len(self.active_operations),
            'total_completed_operations': len(self.completed_operations)
        }
        
        # Add to system metrics history
        self.system_metrics.append(metrics)
        
        # Keep only last hour of system metrics (assuming 1 per minute)
        if len(self.system_metrics) > 60:
            self.system_metrics = self.system_metrics[-60:]
        
        return metrics
    
    def get_performance_summary(self) -> Dict:
        """Get a comprehensive performance summary."""
        if not self.completed_operations:
            return {'status': 'no_data', 'message': 'No operations completed yet'}
        
        # Calculate aggregated metrics
        durations = [op.duration for op in self.completed_operations if op.duration]
        memory_usage = [op.memory_usage_mb for op in self.completed_operations]
        success_count = sum(1 for op in self.completed_operations if op.success)
        
        # Group by operation type
        operation_stats = {}
        for op in self.completed_operations:
            if op.operation not in operation_stats:
                operation_stats[op.operation] = {
                    'count': 0,
                    'total_duration': 0,
                    'success_count': 0,
                    'avg_memory': 0
                }
            
            stats = operation_stats[op.operation]
            stats['count'] += 1
            stats['total_duration'] += op.duration or 0
            stats['success_count'] += 1 if op.success else 0
            stats['avg_memory'] += op.memory_usage_mb
        
        # Calculate averages
        for op_type, stats in operation_stats.items():
            if stats['count'] > 0:
                stats['avg_duration'] = stats['total_duration'] / stats['count']
                stats['success_rate'] = stats['success_count'] / stats['count']
                stats['avg_memory'] = stats['avg_memory'] / stats['count']
        
        return {
            'summary': {
                'total_operations': len(self.completed_operations),
                'success_rate': success_count / len(self.completed_operations),
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'avg_memory_usage_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0
            },
            'operation_stats': operation_stats,
            'current_system': self.get_system_performance(),
            'active_operations_count': len(self.active_operations)
        }
    
    def check_system_health(self) -> Dict:
        """Check system health and resource availability."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        warnings = []
        critical = []
        
        # Memory checks
        if memory.percent > 85:
            critical.append(f"High memory usage: {memory.percent:.1f}%")
        elif memory.percent > 70:
            warnings.append(f"Elevated memory usage: {memory.percent:.1f}%")
        
        # Disk checks
        if disk.percent > 90:
            critical.append(f"High disk usage: {disk.percent:.1f}%")
        elif disk.percent > 80:
            warnings.append(f"Elevated disk usage: {disk.percent:.1f}%")
        
        # CPU checks (if we have recent data)
        cpu_percent = self._get_cpu_usage()
        if cpu_percent > 90:
            critical.append(f"High CPU usage: {cpu_percent:.1f}%")
        elif cpu_percent > 75:
            warnings.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
        
        # Active operations check
        if len(self.active_operations) > 10:
            warnings.append(f"Many active operations: {len(self.active_operations)}")
        
        status = 'critical' if critical else 'warning' if warnings else 'healthy'
        
        return {
            'status': status,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'cpu_percent': cpu_percent,
            'active_operations': len(self.active_operations),
            'warnings': warnings,
            'critical': critical,
            'recommendations': self._get_recommendations(status, warnings, critical)
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    def _get_recommendations(self, status: str, warnings: List[str], critical: List[str]) -> List[str]:
        """Get recommendations based on system status."""
        recommendations = []
        
        if any('memory' in item.lower() for item in warnings + critical):
            recommendations.append("Consider reducing batch sizes or enabling memory optimization")
            recommendations.append("Clean up old sessions and temporary files")
        
        if any('disk' in item.lower() for item in warnings + critical):
            recommendations.append("Clean up output files and temporary directories")
            recommendations.append("Implement automatic file cleanup for old sessions")
        
        if any('cpu' in item.lower() for item in warnings + critical):
            recommendations.append("Reduce parallel processing workers")
            recommendations.append("Implement request throttling")
        
        if len(self.active_operations) > 5:
            recommendations.append("Implement request queuing to limit concurrent operations")
        
        return recommendations
    
    async def save_performance_report(self) -> str:
        """Save performance report to file."""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': self.get_performance_summary(),
                'system_health': self.check_system_health(),
                'recent_operations': [
                    asdict(op) for op in self.completed_operations[-50:]  # Last 50 operations
                ]
            }
            
            reports_dir = Path(settings.temp_dir) / "performance_reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"performance_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")
            raise

# Global performance monitor instance
monitor = PerformanceMonitor() 