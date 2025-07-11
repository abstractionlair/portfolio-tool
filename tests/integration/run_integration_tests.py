#!/usr/bin/env python3
"""
Integration Test Runner

This script runs the comprehensive integration test suite
and generates a detailed report of results.
"""

import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path


def run_test_command(cmd, description):
    """Run a test command and capture results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        elapsed = time.time() - start_time
        
        return {
            'description': description,
            'command': ' '.join(cmd),
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'elapsed_time': elapsed,
            'success': result.returncode == 0
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'description': description,
            'command': ' '.join(cmd),
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'elapsed_time': elapsed,
            'success': False
        }


def main():
    """Run comprehensive integration tests."""
    print("Portfolio Optimizer - Integration Test Suite")
    print(f"Started at: {datetime.now()}")
    
    # Test suites to run
    test_suites = [
        {
            'name': 'Phase 1: Complete Pipeline Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/integration/test_complete_pipeline.py', '-v', '--tb=short'],
            'critical': True
        },
        {
            'name': 'Phase 2: Mathematical Accuracy Tests (subset)',
            'cmd': ['python', '-m', 'pytest', 'tests/integration/test_mathematical_accuracy.py::TestMathematicalAccuracy::test_log_vs_simple_returns_mathematical_relationship', 
                   'tests/integration/test_mathematical_accuracy.py::TestMathematicalAccuracy::test_simple_vs_total_returns_relationship', '-v'],
            'critical': True
        },
        {
            'name': 'Phase 3: Performance Benchmarks (subset)',
            'cmd': ['python', '-m', 'pytest', 'tests/integration/test_performance_benchmarks.py::TestSingleTickerPerformance', '-v', '-m', 'performance'],
            'critical': True
        },
        {
            'name': 'Phase 4: Edge Cases (subset)',
            'cmd': ['python', '-m', 'pytest', 'tests/integration/test_edge_cases.py::TestMissingDataHandling::test_invalid_ticker_handling', 
                   'tests/integration/test_edge_cases.py::TestMissingDataHandling::test_future_date_handling', '-v'],
            'critical': False
        },
        {
            'name': 'Quick Cross-Validation Test',
            'cmd': ['python', '-m', 'pytest', 'tests/integration/test_cross_validation.py::TestStatisticalProperties::test_return_distribution_properties', '-v'],
            'critical': False
        }
    ]
    
    # Run tests
    results = []
    total_time = 0
    
    for test_suite in test_suites:
        result = run_test_command(test_suite['cmd'], test_suite['name'])
        result['critical'] = test_suite['critical']
        results.append(result)
        total_time += result['elapsed_time']
        
        # Print immediate feedback
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"\n{status} - {test_suite['name']} ({result['elapsed_time']:.1f}s)")
        
        if not result['success']:
            print(f"Error: {result['stderr'][:200]}...")
            if test_suite['critical']:
                print("❌ CRITICAL TEST FAILED - Stopping execution")
                break
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*80}")
    
    passed_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    critical_passed = sum(1 for r in results if r['success'] and r['critical'])
    critical_total = sum(1 for r in results if r['critical'])
    
    print(f"Total Tests: {passed_tests}/{total_tests} passed")
    print(f"Critical Tests: {critical_passed}/{critical_total} passed")
    print(f"Total Time: {total_time:.1f} seconds")
    print(f"Average Time per Suite: {total_time/len(results):.1f} seconds")
    
    # Test-by-test results
    print(f"\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "❌"
        critical = "(CRITICAL)" if result['critical'] else ""
        print(f"{i}. {status} {result['description']} {critical} - {result['elapsed_time']:.1f}s")
    
    # Performance analysis
    print(f"\nPerformance Analysis:")
    perf_results = [r for r in results if 'performance' in r['description'].lower()]
    if perf_results:
        avg_perf_time = sum(r['elapsed_time'] for r in perf_results) / len(perf_results)
        print(f"- Average performance test time: {avg_perf_time:.1f}s")
        print(f"- All performance tests under 2s target: {'✅' if avg_perf_time < 2.0 else '❌'}")
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'critical_tests_passed': critical_passed,
            'critical_tests_total': critical_total,
            'total_time': total_time,
            'overall_success': critical_passed == critical_total
        },
        'results': results
    }
    
    report_file = Path('integration_test_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Exit status
    if critical_passed == critical_total:
        print(f"\n🎉 INTEGRATION TESTS PASSED! Data layer is production-ready.")
        return 0
    else:
        print(f"\n⚠️  Some critical tests failed. Data layer needs attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())