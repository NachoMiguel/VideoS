"""
Enhanced Parallel Operation Error Handler
Manages coordinated error handling, retries, and resource cleanup for parallel operations.
"""
import asyncio
import traceback
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

from .logger import logger
from .exceptions import *
from .credit_manager import credit_manager, ServiceType

class OperationType(Enum):
    CHARACTER_EXTRACTION = "character_extraction"
    IMAGE_SEARCH = "image_search"
    FACE_TRAINING = "face_training"
    AUDIO_GENERATION = "audio_generation"
    SCENE_ANALYSIS = "scene_analysis"
    SCENE_SELECTION = "scene_selection"
    VIDEO_ASSEMBLY = "video_assembly"

class OperationPriority(Enum):
    CRITICAL = "critical"  # Must succeed for pipeline to continue
    IMPORTANT = "important"  # Should succeed but pipeline can continue without
    OPTIONAL = "optional"  # Nice to have but not required

@dataclass
class OperationResult:
    """Result of a parallel operation."""
    operation_type: OperationType
    success: bool
    result: Any = None
    error: Exception = None
    execution_time: float = 0.0
    retry_count: int = 0
    account_used: str = None

@dataclass
class ParallelOperationConfig:
    """Configuration for a parallel operation."""
    operation_type: OperationType
    priority: OperationPriority
    service_type: ServiceType
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 300.0  # 5 minutes default
    requires_account: bool = True

class ParallelErrorHandler:
    """Handles errors and coordination for parallel operations."""
    
    def __init__(self):
        self.operation_configs = self._initialize_operation_configs()
        self.active_operations: Dict[str, asyncio.Task] = {}
        self.completed_operations: List[OperationResult] = []
        self.session_id: Optional[str] = None
    
    def _initialize_operation_configs(self) -> Dict[OperationType, ParallelOperationConfig]:
        """Initialize default configurations for all operations."""
        return {
            OperationType.CHARACTER_EXTRACTION: ParallelOperationConfig(
                operation_type=OperationType.CHARACTER_EXTRACTION,
                priority=OperationPriority.CRITICAL,
                service_type=ServiceType.OPENAI,
                max_retries=2,
                timeout=120.0
            ),
            OperationType.IMAGE_SEARCH: ParallelOperationConfig(
                operation_type=OperationType.IMAGE_SEARCH,
                priority=OperationPriority.CRITICAL,
                service_type=ServiceType.GOOGLE_SEARCH,
                max_retries=2,
                timeout=180.0
            ),
            OperationType.FACE_TRAINING: ParallelOperationConfig(
                operation_type=OperationType.FACE_TRAINING,
                priority=OperationPriority.CRITICAL,
                service_type=None,  # Local operation
                max_retries=1,
                timeout=300.0,
                requires_account=False
            ),
            OperationType.AUDIO_GENERATION: ParallelOperationConfig(
                operation_type=OperationType.AUDIO_GENERATION,
                priority=OperationPriority.CRITICAL,
                service_type=ServiceType.ELEVENLABS,
                max_retries=2,
                timeout=240.0
            ),
            OperationType.SCENE_ANALYSIS: ParallelOperationConfig(
                operation_type=OperationType.SCENE_ANALYSIS,
                priority=OperationPriority.CRITICAL,
                service_type=None,  # Local operation
                max_retries=1,
                timeout=600.0,
                requires_account=False
            ),
            OperationType.SCENE_SELECTION: ParallelOperationConfig(
                operation_type=OperationType.SCENE_SELECTION,
                priority=OperationPriority.CRITICAL,
                service_type=None,  # Local operation
                max_retries=1,
                timeout=120.0,
                requires_account=False
            ),
            OperationType.VIDEO_ASSEMBLY: ParallelOperationConfig(
                operation_type=OperationType.VIDEO_ASSEMBLY,
                priority=OperationPriority.CRITICAL,
                service_type=None,  # Local operation
                max_retries=1,
                timeout=900.0,  # 15 minutes for video assembly
                requires_account=False
            )
        }
    
    def start_session(self, session_id: str):
        """Start a new error handling session."""
        self.session_id = session_id
        self.active_operations.clear()
        self.completed_operations.clear()
        logger.info(f"Started parallel error handler session: {session_id}")
    
    async def execute_with_error_handling(
        self,
        operation_type: OperationType,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> OperationResult:
        """Execute an operation with comprehensive error handling."""
        config = self.operation_configs[operation_type]
        start_time = time.time()
        retry_count = 0
        last_error = None
        account_used = None
        
        logger.info(f"Starting {operation_type.value} with {config.max_retries} max retries")
        
        while retry_count <= config.max_retries:
            try:
                # Get account if required
                if config.requires_account and config.service_type:
                    account = credit_manager.get_available_account(config.service_type)
                    account_used = account.account_id
                    
                    # Add account info to kwargs if needed
                    if 'account_info' not in kwargs:
                        kwargs['account_info'] = account
                
                # Execute operation with timeout
                result = await asyncio.wait_for(
                    operation_func(*args, **kwargs),
                    timeout=config.timeout
                )
                
                # Record successful usage
                if config.requires_account and config.service_type:
                    await credit_manager.record_usage(
                        service=config.service_type,
                        account_id=account_used,
                        operation=operation_type.value,
                        cost_estimate=self._estimate_operation_cost(operation_type, result),
                        success=True
                    )
                
                execution_time = time.time() - start_time
                operation_result = OperationResult(
                    operation_type=operation_type,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    retry_count=retry_count,
                    account_used=account_used
                )
                
                logger.info(f"✅ {operation_type.value} completed successfully "
                           f"(attempt {retry_count + 1}, {execution_time:.2f}s)")
                
                return operation_result
                
            except CreditExhaustionError as e:
                # Don't retry on credit exhaustion
                logger.error(f"❌ {operation_type.value} failed due to credit exhaustion: {str(e)}")
                last_error = e
                break
                
            except asyncio.TimeoutError:
                last_error = VideoProcessingError(f"{operation_type.value} timed out after {config.timeout}s")
                logger.warning(f"⏰ {operation_type.value} timed out (attempt {retry_count + 1})")
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ {operation_type.value} failed (attempt {retry_count + 1}): {str(e)}")
                
                # Record failed usage
                if config.requires_account and config.service_type and account_used:
                    await credit_manager.record_usage(
                        service=config.service_type,
                        account_id=account_used,
                        operation=operation_type.value,
                        cost_estimate=0,
                        success=False,
                        error_message=str(e)
                    )
            
            retry_count += 1
            
            # Wait before retry (exponential backoff)
            if retry_count <= config.max_retries:
                delay = config.retry_delay * (2 ** (retry_count - 1))
                logger.info(f"Retrying {operation_type.value} in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # Operation failed after all retries
        execution_time = time.time() - start_time
        operation_result = OperationResult(
            operation_type=operation_type,
            success=False,
            error=last_error,
            execution_time=execution_time,
            retry_count=retry_count - 1,
            account_used=account_used
        )
        
        logger.error(f"❌ {operation_type.value} failed after {retry_count - 1} retries: {str(last_error)}")
        
        return operation_result
    
    async def execute_parallel_operations(
        self,
        operations: List[Tuple[OperationType, Callable, tuple, dict]]
    ) -> Dict[OperationType, OperationResult]:
        """Execute multiple operations in parallel with coordinated error handling."""
        logger.info(f"Starting {len(operations)} parallel operations")
        
        # Create tasks for all operations
        tasks = {}
        for operation_type, func, args, kwargs in operations:
            task = asyncio.create_task(
                self.execute_with_error_handling(operation_type, func, *args, **kwargs)
            )
            tasks[operation_type] = task
            self.active_operations[operation_type.value] = task
        
        # Wait for all operations to complete
        results = {}
        try:
            completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for (operation_type, _), result in zip(operations, completed_tasks):
                if isinstance(result, Exception):
                    # Task raised an exception
                    results[operation_type] = OperationResult(
                        operation_type=operation_type,
                        success=False,
                        error=result
                    )
                else:
                    results[operation_type] = result
                
                # Remove from active operations
                self.active_operations.pop(operation_type.value, None)
        
        except Exception as e:
            logger.error(f"Error in parallel operation coordination: {str(e)}")
            # Clean up any remaining tasks
            for task in tasks.values():
                if not task.done():
                    task.cancel()
        
        # Analyze results and determine if pipeline should continue
        await self._analyze_parallel_results(results)
        
        return results
    
    async def _analyze_parallel_results(self, results: Dict[OperationType, OperationResult]):
        """Analyze parallel operation results and determine next steps."""
        critical_failures = []
        important_failures = []
        successes = []
        
        for operation_type, result in results.items():
            config = self.operation_configs[operation_type]
            
            if result.success:
                successes.append(operation_type)
            else:
                if config.priority == OperationPriority.CRITICAL:
                    critical_failures.append((operation_type, result.error))
                elif config.priority == OperationPriority.IMPORTANT:
                    important_failures.append((operation_type, result.error))
        
        # Log summary
        logger.info(f"Parallel operations summary: "
                   f"{len(successes)} succeeded, "
                   f"{len(critical_failures)} critical failures, "
                   f"{len(important_failures)} important failures")
        
        # Check if pipeline should continue
        if critical_failures:
            # Based on user preference: continue with partial results if parallel operations partially fail
            logger.warning(f"Critical operations failed but continuing as per configuration: "
                          f"{[op.value for op, _ in critical_failures]}")
            
            # Store failed operations for user notification
            self.completed_operations.extend(results.values())
            
            # Don't raise exception - continue with partial results
        else:
            logger.info("All critical operations completed successfully")
    
    def _estimate_operation_cost(self, operation_type: OperationType, result: Any) -> float:
        """Estimate the cost of an operation for usage tracking."""
        # These are rough estimates - you can refine based on actual usage
        cost_estimates = {
            OperationType.CHARACTER_EXTRACTION: 0.001,  # ~$0.001 per request
            OperationType.IMAGE_SEARCH: 1.0,  # 1 query unit
            OperationType.AUDIO_GENERATION: 100.0,  # ~100 characters estimate
            OperationType.FACE_TRAINING: 0.0,  # Local operation
            OperationType.SCENE_ANALYSIS: 0.0,  # Local operation
            OperationType.SCENE_SELECTION: 0.0,  # Local operation
            OperationType.VIDEO_ASSEMBLY: 0.0,  # Local operation
        }
        
        base_cost = cost_estimates.get(operation_type, 0.0)
        
        # Adjust based on result if possible
        if operation_type == OperationType.AUDIO_GENERATION and isinstance(result, str):
            # Estimate based on text length
            return len(result)
        elif operation_type == OperationType.IMAGE_SEARCH and isinstance(result, dict):
            # Count number of searches performed
            return sum(len(images) for images in result.values()) * 0.1
        
        return base_cost
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session's operations."""
        if not self.completed_operations:
            return {"total_operations": 0, "success_rate": 0.0}
        
        successful = len([op for op in self.completed_operations if op.success])
        total = len(self.completed_operations)
        
        return {
            "session_id": self.session_id,
            "total_operations": total,
            "successful_operations": successful,
            "failed_operations": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "operations": [
                {
                    "type": op.operation_type.value,
                    "success": op.success,
                    "execution_time": op.execution_time,
                    "retry_count": op.retry_count,
                    "error": str(op.error) if op.error else None
                }
                for op in self.completed_operations
            ]
        }
    
    async def cleanup_resources(self):
        """Clean up any remaining resources and active operations."""
        logger.info("Cleaning up parallel operation resources")
        
        # Cancel any still-active operations
        for operation_name, task in self.active_operations.items():
            if not task.done():
                logger.warning(f"Cancelling active operation: {operation_name}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error during cleanup of {operation_name}: {str(e)}")
        
        self.active_operations.clear()

# Global parallel error handler instance
parallel_error_handler = ParallelErrorHandler() 