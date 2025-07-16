#!/usr/bin/env python3
"""
Demo script for return prediction optimization in portfolio-level framework.

This script demonstrates:
1. Portfolio-level optimization with return prediction
2. Combined scoring (volatility + return accuracy)
3. Return prediction methods (historical, EWMA, momentum, mean reversion)
4. Multi-horizon optimization with return prediction accuracy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run return prediction optimization demo."""
    
    print("="*80)
    print("RETURN PREDICTION OPTIMIZATION DEMO")
    print("="*80)
    print()
    
    # Load exposure universe
    try:
        from data.exposure_universe import ExposureUniverse
        from optimization.parameter_optimization import ParameterOptimizer
    except ImportError:
        from src.data.exposure_universe import ExposureUniverse
        from src.optimization.parameter_optimization import ParameterOptimizer
    
    # Load universe
    universe_path = Path("config/exposure_universe.yaml")
    if not universe_path.exists():
        print(f"ERROR: Universe file not found at {universe_path}")
        return
    
    print(f"Loading exposure universe from {universe_path}")
    universe = ExposureUniverse.from_yaml(universe_path)
    
    print(f"✓ Loaded {len(universe.exposures)} exposures")
    print(f"✓ Available exposures: {list(universe.exposures.keys())}")
    print()
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(universe)
    
    # Run portfolio-level optimization with return prediction
    # Test smaller set of horizons for faster demo
    candidate_horizons = [21, 63, 126]  # Monthly, quarterly, semi-annual
    print(f"Testing forecast horizons: {candidate_horizons} days")
    print("- 21 days: Monthly rebalancing")
    print("- 63 days: Quarterly rebalancing") 
    print("- 126 days: Semi-annual rebalancing")
    print()
    print("Optimizing with return prediction integration...")
    print("This combines volatility prediction accuracy with return prediction accuracy")
    print("Combined scoring: 70% volatility accuracy + 30% return directional accuracy")
    print()
    
    try:
        results = optimizer.optimize_portfolio_level(
            candidate_horizons=candidate_horizons,
            start_date=datetime(2020, 1, 1),  # Shorter period for demo
            end_date=datetime(2024, 12, 31)
        )
        
        print("✓ Portfolio-level optimization with return prediction completed!")
        print()
        
        # Display results
        optimal_horizon = results['optimal_horizon']
        optimal_params = results['optimal_parameters']
        
        print(f"🎯 OPTIMAL HORIZON: {optimal_horizon} days")
        print(f"📊 Goodness Score: {optimal_params.goodness_score:.6f}")
        print(f"📈 Volatility RMSE: {optimal_params.validation_metrics.get('vol_rmse', 'N/A'):.4f}")
        print(f"🎯 Return Accuracy: {optimal_params.validation_metrics.get('return_accuracy', 'N/A'):.1%}")
        print(f"🔍 Test Cases: {optimal_params.validation_metrics.get('n_tests', 'N/A')}")
        print()
        
        # Show return prediction methods by exposure
        print("📋 RETURN PREDICTION METHODS BY EXPOSURE:")
        print("-" * 50)
        return_methods = {}
        for exp_id, params in optimal_params.return_params.items():
            method = params['method']
            score = params['validation_score']
            return_methods[method] = return_methods.get(method, 0) + 1
            print(f"  {exp_id:25} | {method:15} | {score:.3f}")
        
        print()
        print("📊 RETURN METHOD DISTRIBUTION:")
        print("-" * 30)
        for method, count in sorted(return_methods.items()):
            percentage = count / len(optimal_params.return_params) * 100
            print(f"  {method:15} | {count:2d} ({percentage:4.1f}%)")
        
        print()
        print("🔄 HORIZON COMPARISON:")
        print("-" * 40)
        for horizon, result in results['all_horizon_results'].items():
            vol_rmse = result.validation_metrics.get('vol_rmse', 0)
            return_acc = result.validation_metrics.get('return_accuracy', 0)
            print(f"  {horizon:3d} days | Score: {result.goodness_score:8.6f} | "
                  f"Vol RMSE: {vol_rmse:.4f} | Return Acc: {return_acc:.1%}")
        
        print()
        print("✅ SUCCESS: Return prediction optimization completed!")
        print(f"📁 Results saved to: config/optimal_parameters_portfolio_level.yaml")
        print(f"📁 Detailed results: output/portfolio_level_optimization/")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("="*80)
    print("DEMO COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()