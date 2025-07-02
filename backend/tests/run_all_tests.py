#!/usr/bin/env python3
"""
Test runner for the AI Video Slicer project.
Runs all test suites and provides comprehensive reporting.
"""
import unittest
import sys
import os
from pathlib import Path
import time
from io import StringIO

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

def run_test_suite(test_module_name, description):
    """Run a specific test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    # Capture test output
    test_output = StringIO()
    
    # Load and run the test suite
    try:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_module_name)
        
        runner = unittest.TextTestRunner(
            stream=test_output,
            verbosity=2,
            failfast=False
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Print results
        output = test_output.getvalue()
        print(output)
        
        # Summary
        duration = end_time - start_time
        print(f"\n{description} Summary:")
        print(f"  Tests run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Duration: {duration:.2f}s")
        
        if result.failures:
            print(f"  Failed tests:")
            for test, traceback in result.failures:
                print(f"    - {test}")
        
        if result.errors:
            print(f"  Error tests:")
            for test, traceback in result.errors:
                print(f"    - {test}")
        
        return {
            'name': description,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'duration': duration,
            'success': len(result.failures) == 0 and len(result.errors) == 0
        }
        
    except Exception as e:
        print(f"❌ Failed to run {description}: {str(e)}")
        return {
            'name': description,
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'duration': 0,
            'success': False,
            'error': str(e)
        }

def run_integration_tests():
    """Run the comprehensive implementation tests."""
    print(f"\n{'='*60}")
    print("Running Integration Tests")
    print(f"{'='*60}")
    
    try:
        # Import and run the implementation tests
        from test_implementations import main as run_implementation_tests
        
        start_time = time.time()
        success = False
        
        # Run in a try-catch to handle any issues
        try:
            import asyncio
            success = asyncio.run(run_implementation_tests())
        except Exception as e:
            print(f"❌ Integration tests failed: {str(e)}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'name': 'Integration Tests',
            'tests_run': 6,  # We know there are 6 integration tests
            'failures': 0 if success else 6,
            'errors': 0,
            'duration': duration,
            'success': success
        }
        
    except Exception as e:
        print(f"❌ Failed to run integration tests: {str(e)}")
        return {
            'name': 'Integration Tests',
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'duration': 0,
            'success': False,
            'error': str(e)
        }

def main():
    """Run all test suites."""
    print("🚀 AI Video Slicer - Comprehensive Test Suite")
    print("=" * 80)
    
    # Define test suites
    test_suites = [
        ('test_core', 'Core Functionality Tests'),
        ('test_services', 'Service Module Tests'),
        ('test_video', 'Video Processing Tests'),
        ('test_phase7', 'Phase 7: Error Handling & Credit Protection Tests'),
    ]
    
    # Run all test suites
    results = []
    total_start_time = time.time()
    
    # Run unit tests
    for module_name, description in test_suites:
        result = run_test_suite(module_name, description)
        results.append(result)
    
    # Run integration tests
    integration_result = run_integration_tests()
    results.append(integration_result)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = sum(r['tests_run'] for r in results)
    total_failures = sum(r['failures'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    successful_suites = sum(1 for r in results if r['success'])
    
    print(f"Total Test Suites: {len(results)}")
    print(f"Successful Suites: {successful_suites}")
    print(f"Total Tests Run: {total_tests}")
    print(f"Total Failures: {total_failures}")
    print(f"Total Errors: {total_errors}")
    print(f"Total Duration: {total_duration:.2f}s")
    
    print(f"\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {result['name']:<25} {status} ({result['tests_run']} tests, {result['duration']:.2f}s)")
        
        if not result['success']:
            if result['failures'] > 0:
                print(f"    └─ {result['failures']} failures")
            if result['errors'] > 0:
                print(f"    └─ {result['errors']} errors")
            if 'error' in result:
                print(f"    └─ Error: {result['error']}")
    
    # Overall result
    all_passed = total_failures == 0 and total_errors == 0
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! The AI Video Slicer is ready for production.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Please review the failures above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 