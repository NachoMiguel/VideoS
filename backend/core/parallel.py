import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Coroutine
from functools import wraps
import time
from dataclasses import dataclass

from .config import settings
from .exceptions import ProcessingError

logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Result of a parallel task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0

class ParallelProcessor:
    """Enhanced parallel processing manager for AI Video Slicer."""
    
    def __init__(self):
        self.max_workers = settings.max_workers
        self.concurrent_api_calls = settings.concurrent_api_calls
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.api_semaphore = asyncio.Semaphore(self.concurrent_api_calls)
        self._active_tasks = {}
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the parallel processor."""
        self.executor.shutdown(wait=True)
        logger.info("Parallel processor shutdown complete")
    
    async def execute_parallel_tasks(
        self,
        tasks: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> List[TaskResult]:
        """Execute multiple tasks in parallel with progress tracking."""
        if not tasks:
            return []
        
        logger.info(f"Starting parallel execution of {len(tasks)} tasks")
        start_time = time.time()
        
        # Create coroutines for all tasks
        coroutines = []
        for i, task in enumerate(tasks):
            task_id = task.get('id', f'task_{i}')
            coroutine = self._execute_single_task(task_id, task)
            coroutines.append(coroutine)
        
        # Execute tasks with progress tracking
        results = []
        completed = 0
        
        for coro in asyncio.as_completed(coroutines):
            try:
                result = await coro
                results.append(result)
                completed += 1
                
                # Report progress
                if progress_callback:
                    progress = (completed / len(tasks)) * 100
                    await progress_callback(f"Completed {completed}/{len(tasks)} tasks", progress)
                    
            except Exception as e:
                logger.error(f"Task execution failed: {str(e)}")
                results.append(TaskResult(
                    task_id=f"failed_{completed}",
                    success=False,
                    error=e
                ))
                completed += 1
        
        execution_time = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        
        logger.info(
            f"Parallel execution completed: {success_count}/{len(tasks)} successful "
            f"in {execution_time:.2f}s"
        )
        
        return results
    
    async def _execute_single_task(self, task_id: str, task: Dict[str, Any]) -> TaskResult:
        """Execute a single task with error handling and timing."""
        start_time = time.time()
        
        try:
            task_type = task.get('type')
            task_func = task.get('func')
            task_args = task.get('args', [])
            task_kwargs = task.get('kwargs', {})
            
            if not task_func:
                raise ValueError(f"Task {task_id} missing 'func' parameter")
            
            # Execute based on task type
            if task_type == 'api_call':
                async with self.api_semaphore:  # Rate limit API calls
                    result = await self._execute_api_task(task_func, task_args, task_kwargs)
            elif task_type == 'cpu_intensive':
                result = await self._execute_cpu_task(task_func, task_args, task_kwargs)
            elif task_type == 'io_bound':
                result = await self._execute_io_task(task_func, task_args, task_kwargs)
            else:
                # Default execution
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(*task_args, **task_kwargs)
                else:
                    result = task_func(*task_args, **task_kwargs)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {str(e)}")
            
            return TaskResult(
                task_id=task_id,
                success=False,
                error=e,
                execution_time=execution_time
            )
    
    async def _execute_api_task(self, func: Callable, args: List, kwargs: Dict) -> Any:
        """Execute API call with rate limiting."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def _execute_cpu_task(self, func: Callable, args: List, kwargs: Dict) -> Any:
        """Execute CPU-intensive task in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def _execute_io_task(self, func: Callable, args: List, kwargs: Dict) -> Any:
        """Execute I/O-bound task."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def parallel_video_analysis(
        self,
        video_paths: List[str],
        analysis_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[TaskResult]:
        """Parallel video analysis for multiple videos."""
        tasks = []
        
        for i, video_path in enumerate(video_paths):
            tasks.append({
                'id': f'video_analysis_{i}',
                'type': 'cpu_intensive',
                'func': analysis_func,
                'args': [video_path],
                'kwargs': {}
            })
        
        return await self.execute_parallel_tasks(tasks, progress_callback)
    
    async def parallel_api_calls(
        self,
        api_calls: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> List[TaskResult]:
        """Execute multiple API calls in parallel with rate limiting."""
        tasks = []
        
        for i, api_call in enumerate(api_calls):
            tasks.append({
                'id': f'api_call_{i}',
                'type': 'api_call',
                'func': api_call['func'],
                'args': api_call.get('args', []),
                'kwargs': api_call.get('kwargs', {})
            })
        
        return await self.execute_parallel_tasks(tasks, progress_callback)
    
    async def parallel_audio_generation(
        self,
        audio_segments: List[Dict[str, Any]],
        generation_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[TaskResult]:
        """Generate multiple audio segments in parallel."""
        tasks = []
        
        for i, segment in enumerate(audio_segments):
            tasks.append({
                'id': f'audio_gen_{i}',
                'type': 'api_call',
                'func': generation_func,
                'args': [segment],
                'kwargs': {}
            })
        
        return await self.execute_parallel_tasks(tasks, progress_callback)

# Global parallel processor instance
parallel_processor = ParallelProcessor()

def parallel_task(task_type: str = 'default'):
    """Decorator to mark functions for parallel execution."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            task = {
                'id': func.__name__,
                'type': task_type,
                'func': func,
                'args': args,
                'kwargs': kwargs
            }
            
            results = await parallel_processor.execute_parallel_tasks([task])
            if results and results[0].success:
                return results[0].result
            elif results and results[0].error:
                raise results[0].error
            else:
                raise ProcessingError(f"Parallel task {func.__name__} failed")
        
        return wrapper
    return decorator 