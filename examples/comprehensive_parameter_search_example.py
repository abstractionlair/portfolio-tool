"""
Example usage of comprehensive parameter search functionality.

This demonstrates how to use the new comprehensive parameter optimization
that tests the complete pipeline: data loading + decomposition + estimation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from optimization.comprehensive_parameter_search import (
    ComprehensiveParameterSearchEngine,
    ComprehensiveParameterEstimator,
    analyze_search_results
)
from optimization.risk_premium_estimator import RiskPremiumEstimator
from data.return_decomposition import ReturnDecomposer
from data.exposure_universe import ExposureUniverse


def demonstrate_comprehensive_search():
    """Demonstrate comprehensive parameter search on exposure universe."""
    
    print("🚀 COMPREHENSIVE PARAMETER SEARCH DEMONSTRATION")
    print("=" * 60)
    
    # Load exposure universe
    universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
    universe = ExposureUniverse.from_yaml(str(universe_path))
    
    # Initialize framework
    return_decomposer = ReturnDecomposer()
    risk_estimator = RiskPremiumEstimator(universe, return_decomposer)
    estimation_date = datetime.now()
    
    print(f"✅ Loaded {len(universe)} exposures from universe")
    print(f"📅 Estimation date: {estimation_date.date()}")
    print()
    
    # Create search engine
    search_engine = ComprehensiveParameterSearchEngine(
        risk_estimator=risk_estimator,
        estimation_date=estimation_date
    )
    
    # Demonstrate single exposure optimization
    print("1. SINGLE EXPOSURE OPTIMIZATION")
    print("-" * 40)
    
    # Test on first exposure
    test_exposure = 'us_large_equity'
    print(f"Testing comprehensive optimization on: {test_exposure}")
    
    result = search_engine.optimize_single_exposure(
        exposure_id=test_exposure,
        method='randomized',
        n_iter=20,  # Small number for demo
        constrained=True,  # Use stable parameter ranges
        n_jobs=1  # Single thread for demo
    )
    
    if result:
        print(f"✅ Optimization successful!")
        print(f"   Best score: {result.best_score:.6f}")
        print(f"   Best parameters: {result.best_params}")
        print(f"   Method: {result.method}")
        print(f"   Combinations tested: {result.n_combinations_tested}")
        print(f"   Time: {result.elapsed_time:.1f} seconds")
    else:
        print(f"❌ Optimization failed")
    
    print()
    
    # Demonstrate method comparison
    print("2. SEARCH METHOD COMPARISON")
    print("-" * 40)
    
    comparison_results = search_engine.compare_search_methods(
        exposure_id=test_exposure,
        n_iter=15,  # Small number for demo
        constrained=True
    )
    
    if comparison_results:
        print(f"✅ Method comparison complete!")
        print(f"{'Method':<15} {'Score':<12} {'Time':<8} {'Tests':<6}")
        print("-" * 45)
        
        for method_name, result in comparison_results.items():
            print(f"{method_name:<15} {result.best_score:<12.6f} "
                  f"{result.elapsed_time:<8.1f} {result.n_combinations_tested:<6}")
        
        # Find best method
        best_method = min(comparison_results.items(), 
                         key=lambda x: x[1].best_score)
        print(f"\n🏆 Best method: {best_method[0]} "
              f"(score: {best_method[1].best_score:.6f})")
    else:
        print(f"❌ Method comparison failed")
    
    print()
    
    # Demonstrate multi-exposure optimization
    print("3. MULTI-EXPOSURE OPTIMIZATION")
    print("-" * 40)
    
    # Test on first few exposures
    test_exposures = ['us_large_equity', 'us_small_equity']
    print(f"Testing on exposures: {test_exposures}")
    
    multi_results = search_engine.optimize_multiple_exposures(
        exposure_ids=test_exposures,
        method='randomized',
        n_iter=15,  # Small number for demo
        constrained=True,
        n_jobs=1
    )
    
    if multi_results:
        print(f"✅ Multi-exposure optimization complete!")
        
        # Analyze results
        analysis = analyze_search_results(multi_results)
        
        if analysis:
            print(f"\n📊 ANALYSIS RESULTS:")
            summary = analysis['summary']
            print(f"   Exposures optimized: {summary['num_exposures']}")
            print(f"   Average score: {summary['avg_score']:.6f}")
            print(f"   Total combinations: {summary['total_combinations']}")
            print(f"   Average time: {summary['avg_time']:.1f} seconds")
            
            print(f"\n   Method preferences:")
            for method, count in analysis['method_preferences'].items():
                print(f"     {method}: {count} exposures")
            
            print(f"\n   Parameter statistics:")
            for param, stats in analysis['parameter_stats'].items():
                if stats['mean'] is not None:
                    print(f"     {param}: {stats['min']}-{stats['max']} "
                          f"(avg: {stats['mean']:.0f})")
    else:
        print(f"❌ Multi-exposure optimization failed")
    
    print()
    print("=" * 60)
    print("🎯 DEMONSTRATION COMPLETE!")
    print()
    print("KEY FEATURES DEMONSTRATED:")
    print("• Complete pipeline optimization (data + estimation)")
    print("• Multiple search methods (grid vs randomized)")
    print("• Multi-exposure batch optimization")
    print("• Comprehensive result analysis")
    print("• 64k+ parameter combination capability")


def demonstrate_direct_usage():
    """Show direct usage of the estimator (without sklearn optimization)."""
    
    print("\n🔬 DIRECT ESTIMATOR USAGE")
    print("=" * 40)
    
    # This shows how the estimator works without sklearn optimization
    print("Testing direct parameter estimation...")
    
    # Mock setup for demonstration
    from unittest.mock import Mock
    
    mock_risk_estimator = Mock()
    
    # Mock successful data
    dates = pd.date_range('2020-01-01', periods=30, freq='ME')
    mock_decomposition = pd.DataFrame({
        'spread': np.random.normal(0.005, 0.015, 30),
    }, index=dates)
    mock_risk_estimator.load_and_decompose_exposure_returns.return_value = mock_decomposition
    
    # Mock successful estimate
    mock_estimate = Mock()
    mock_estimate.risk_premium_volatility = 0.045
    mock_estimate.sample_size = 28
    mock_risk_estimator.estimate_risk_premium_volatility.return_value = mock_estimate
    
    # Test different parameter combinations
    test_params = [
        {'method': 'historical', 'window': 10, 'horizon': 21},
        {'method': 'historical', 'window': 20, 'horizon': 42},
        {'method': 'ewma', 'lambda_param': 0.94, 'horizon': 21},
        {'method': 'ewma', 'lambda_param': 0.97, 'horizon': 42},
    ]
    
    print(f"{'Method':<12} {'Window/Lambda':<12} {'Horizon':<8} {'Score':<12}")
    print("-" * 50)
    
    for params in test_params:
        estimator = ComprehensiveParameterEstimator(
            exposure_id='test_exposure',
            risk_estimator=mock_risk_estimator,
            estimation_date=datetime.now(),
            lookback_days=1260,
            frequency='monthly',
            **params
        )
        
        score = estimator.score()
        
        param_str = (f"{params.get('window', params.get('lambda_param', 'N/A'))}")
        print(f"{params['method']:<12} {param_str:<12} "
              f"{params['horizon']:<8} {score:<12.6f}")
    
    print("\n✅ Direct usage demonstration complete!")


if __name__ == "__main__":
    try:
        demonstrate_comprehensive_search()
        demonstrate_direct_usage()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory and have all dependencies.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is a demonstration - some components may need real data to work fully.")