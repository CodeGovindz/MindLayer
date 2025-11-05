#!/usr/bin/env python3
"""Test runner for integration and performance validation tests."""

import subprocess
import sys
import os
from pathlib import Path

def run_test_suite(test_file, description):
    """Run a test suite and report results."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v", 
            "--tb=short",
            "--no-header"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run all integration and performance tests."""
    print("Universal Memory Layer - Integration & Performance Test Suite")
    print("=" * 60)
    
    # Test suites to run
    test_suites = [
        ("tests/test_end_to_end_integration.py", "End-to-End Integration Tests"),
        ("tests/test_performance_validation.py", "Performance Validation Tests"),
        ("tests/test_error_handling_recovery.py", "Error Handling & Recovery Tests"),
    ]
    
    results = []
    
    for test_file, description in test_suites:
        success = run_test_suite(test_file, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<40} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())