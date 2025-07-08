#!/usr/bin/env python3
"""
Stage 3 Integration Test - Session Management, Error Handling & Performance
Tests the comprehensive Stage 3 implementation for production readiness.
"""

import asyncio
import aiohttp
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add backend to Python path
sys.path.append(str(Path(__file__).parent))

from core.session import manager as session_manager
from core.performance_monitor import monitor as performance_monitor
from core.background_tasks import manager as background_manager
from core.config import settings

class Stage3IntegrationTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, success: bool, message: str = "", details: Dict = None):
        """Log test results."""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': time.time(),
            'details': details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"   → {message}")
        if details:
            print(f"   → Details: {json.dumps(details, indent=2)}")
        print()
    
    async def run_all_tests(self):
        """Run all Stage 3 integration tests."""
        print("🎯 Starting Stage 3 Integration Tests")
        print("=" * 60)
        
        # Test 1: Session Management
        await self.test_session_management()
        
        # Test 2: Performance Monitoring - DISABLED (performance monitor removed)
        # await self.test_performance_monitoring()
        
        # Test 3: Error Handling
        await self.test_error_handling()
        
        # Test 4: Background Tasks
        await self.test_background_tasks()
        
        # Test 5: Resource Management
        await self.test_resource_management()
        
        # Test 6: API Integration
        await self.test_api_integration()
        
        # Generate summary
        self.generate_test_summary()
        
        return self.test_results
    
    async def test_session_management(self):
        """Test comprehensive session management."""
        print("🔄 Testing Session Management...")
        
        try:
            # Test session creation
            session = await session_manager.create_session(
                initial_data={'test': 'session_management'}
            )
            
            self.log_test(
                "Session Creation",
                session is not None,
                f"Created session: {session.session_id}",
                {'session_id': session.session_id}
            )
            
            # Test session retrieval
            retrieved = await session_manager.get_session(session.session_id)
            self.log_test(
                "Session Retrieval",
                retrieved.session_id == session.session_id,
                "Successfully retrieved session"
            )
            
            # Test session update
            updated = await session_manager.update_session(
                session.session_id,
                status='testing',
                progress=50
            )
            self.log_test(
                "Session Update",
                updated.status == 'testing' and updated.progress == 50,
                "Successfully updated session"
            )
            
            # Test session cleanup
            await session_manager.cleanup_session(session.session_id)
            
            try:
                await session_manager.get_session(session.session_id)
                self.log_test("Session Cleanup", False, "Session still exists after cleanup")
            except:
                self.log_test("Session Cleanup", True, "Session successfully cleaned up")
                
        except Exception as e:
            self.log_test("Session Management", False, f"Error: {str(e)}")
    
    async def test_performance_monitoring(self):
        """Test performance monitoring system."""
        print("📊 Testing Performance Monitoring...")
        
        try:
            # Start an operation
            operation_key = performance_monitor.start_operation(
                "test_session", "test_operation", 10.5
            )
            
            self.log_test(
                "Performance Operation Start",
                operation_key is not None,
                f"Started operation: {operation_key}"
            )
            
            # Simulate work
            await asyncio.sleep(0.1)
            
            # Complete the operation
            performance_monitor.complete_operation(operation_key, success=True)
            
            self.log_test(
                "Performance Operation Complete",
                operation_key not in performance_monitor.active_operations,
                "Operation completed successfully"
            )
            
            # Test performance summary
            summary = performance_monitor.get_performance_summary()
            self.log_test(
                "Performance Summary",
                summary.get('status') != 'no_data',
                f"Generated summary with {summary.get('summary', {}).get('total_operations', 0)} operations"
            )
            
            # Test system health check
            health = performance_monitor.check_system_health()
            self.log_test(
                "System Health Check",
                health.get('status') in ['healthy', 'warning', 'critical'],
                f"System status: {health.get('status')}"
            )
            
        except Exception as e:
            self.log_test("Performance Monitoring", False, f"Error: {str(e)}")
    
    async def test_error_handling(self):
        """Test comprehensive error handling."""
        print("🛡️ Testing Error Handling...")
        
        try:
            # Test session not found error
            try:
                await session_manager.get_session("nonexistent_session")
                self.log_test("Session Not Found Error", False, "Should have raised error")
            except Exception as e:
                self.log_test(
                    "Session Not Found Error",
                    "not found" in str(e).lower(),
                    "Correctly handled missing session"
                )
            
            # Test performance monitoring error handling
            performance_monitor.complete_operation("nonexistent_operation", success=False)
            self.log_test(
                "Performance Error Handling",
                True,
                "Gracefully handled nonexistent operation"
            )
            
            # Test configuration validation
            valid_config = (
                hasattr(settings, 'max_concurrent_processing') and
                hasattr(settings, 'session_timeout_minutes') and
                hasattr(settings, 'performance_monitoring_enabled')
            )
            
            self.log_test(
                "Configuration Validation",
                valid_config,
                "All required configuration settings present"
            )
            
        except Exception as e:
            self.log_test("Error Handling", False, f"Error: {str(e)}")
    
    async def test_background_tasks(self):
        """Test background task management."""
        print("⚙️ Testing Background Tasks...")
        
        try:
            # Check if background manager exists
            task_status = background_manager.get_task_status()
            
            self.log_test(
                "Background Task Manager",
                isinstance(task_status, dict),
                f"Background manager active with {len(task_status)} tasks"
            )
            
            # Test task status structure
            for task_name, status in task_status.items():
                valid_status = (
                    'status' in status and
                    'running' in status and
                    isinstance(status['running'], bool)
                )
                
                self.log_test(
                    f"Background Task: {task_name}",
                    valid_status,
                    f"Status: {status['status']}, Running: {status['running']}"
                )
            
        except Exception as e:
            self.log_test("Background Tasks", False, f"Error: {str(e)}")
    
    async def test_resource_management(self):
        """Test resource management and limits."""
        print("🔧 Testing Resource Management...")
        
        try:
            # Test file size limits
            valid_limits = (
                settings.max_file_size > 0 and
                settings.max_concurrent_processing > 0 and
                settings.session_timeout_minutes > 0
            )
            
            self.log_test(
                "Resource Limits Configuration",
                valid_limits,
                f"Max file size: {settings.max_file_size}, Max concurrent: {settings.max_concurrent_processing}"
            )
            
            # Test directory creation
            for directory in [settings.upload_dir, settings.output_dir, settings.temp_dir]:
                dir_path = Path(directory)
                dir_path.mkdir(exist_ok=True)
                
                self.log_test(
                    f"Directory Creation: {directory}",
                    dir_path.exists(),
                    f"Directory exists: {dir_path.absolute()}"
                )
            
        except Exception as e:
            self.log_test("Resource Management", False, f"Error: {str(e)}")
    
    async def test_api_integration(self):
        """Test API integration with new features."""
        print("🔌 Testing API Integration...")
        
        try:
            # Test API endpoints exist
            endpoints_to_test = [
                "/api/video/upload",
                "/api/video/process/test",
                "/api/video/status/test",
                "/api/v1/modify-script"
            ]
            
            for endpoint in endpoints_to_test:
                try:
                    async with self.session.get(f"{self.base_url}{endpoint}") as response:
                        # We expect various status codes, but not 404
                        success = response.status != 404
                        
                        self.log_test(
                            f"API Endpoint: {endpoint}",
                            success,
                            f"Status: {response.status}"
                        )
                        
                except Exception as e:
                    self.log_test(
                        f"API Endpoint: {endpoint}",
                        False,
                        f"Connection error: {str(e)}"
                    )
            
        except Exception as e:
            self.log_test("API Integration", False, f"Error: {str(e)}")
    
    def generate_test_summary(self):
        """Generate comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📋 STAGE 3 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n🔥 FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['message']}")
        
        print("\n🎯 STAGE 3 IMPLEMENTATION STATUS:")
        
        # Critical systems check
        critical_systems = [
            'Session Creation',
            'Session Retrieval', 
            'Session Update',
            'Performance Operation Start',
            'Performance Operation Complete',
            'Background Task Manager'
        ]
        
        critical_passed = sum(1 for r in self.test_results 
                            if r['test'] in critical_systems and r['success'])
        
        if critical_passed == len(critical_systems):
            print("✅ All critical systems operational")
            print("🚀 READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("❌ Some critical systems failing")
            print("⚠️  NOT READY FOR PRODUCTION")
        
        # Performance insights
        performance_tests = [r for r in self.test_results if 'Performance' in r['test']]
        if performance_tests:
            perf_passed = sum(1 for r in performance_tests if r['success'])
            print(f"📊 Performance Monitoring: {perf_passed}/{len(performance_tests)} tests passed")
        
        print("\n" + "=" * 60)

async def main():
    """Run Stage 3 integration tests."""
    print("🎬 AI Video Slicer - Stage 3 Integration Tests")
    print("Testing: Session Management, Error Handling & Performance")
    print()
    
    async with Stage3IntegrationTester() as tester:
        results = await tester.run_all_tests()
        
        # Save results to file
        results_file = Path(__file__).parent / "stage3_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Test results saved to: {results_file}")
        
        # Return appropriate exit code
        failed_tests = sum(1 for r in results if not r['success'])
        return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 