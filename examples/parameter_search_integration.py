"""
Example of integrating CompatibilityValidationFramework with ParameterSearchEngine.

This demonstrates how to use the new modular validation framework with existing
parameter search functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from src.validation import CompatibilityValidationFramework
from src.search.parameter_search import ParameterSearchEngine, SearchConfiguration
from data.multi_frequency import Frequency

def main():
    """Demonstrate parameter search integration with modular validation."""
    print("=== Parameter Search Integration Demo ===\n")
    
    # Create sample exposure data
    np.random.seed(42)
    exposure_data = {}
    available_exposures = ['SPY_QQQ', 'IWM_TLT', 'GLD_VIX']
    
    for exposure_id in available_exposures:
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        spread = pd.Series(np.random.normal(0, 0.02, 200), index=dates)
        exposure_data[exposure_id] = {'spread': spread}
    
    print(f"Created sample data for {len(available_exposures)} exposures")
    print(f"Each exposure has {len(exposure_data[available_exposures[0]]['spread'])} periods\n")
    
    # === Method 1: Direct replacement (recommended) ===
    print("1. DIRECT REPLACEMENT METHOD")
    print("-" * 40)
    
    # Create validation framework using compatibility adapter
    validation_framework = CompatibilityValidationFramework(adaptive_mode=True)
    print("✅ Created CompatibilityValidationFramework")
    
    # Create search configuration for demonstration
    search_config = SearchConfiguration(
        history_lengths=[100, 150],  # Smaller for demo
        frequencies=[Frequency.WEEKLY],  # Single frequency for demo
        horizons=[21, 42],  # Two horizons
        methods={
            'historical': {
                'description': 'Historical Volatility',
                'parameters': [{'window': 20}, {'window': 40}]
            }
        }
    )
    
    print(f"Search configuration:")
    print(f"  - Methods: {list(search_config.methods.keys())}")
    print(f"  - Horizons: {search_config.horizons}")
    print(f"  - History lengths: {search_config.history_lengths}")
    print(f"  - Total combinations: {search_config.get_total_combinations()}")
    
    # Create and run parameter search
    search_engine = ParameterSearchEngine(
        exposure_data=exposure_data,
        available_exposures=available_exposures,
        validation_framework=validation_framework,  # Use compatibility framework
        search_config=search_config
    )
    
    print("\n🔍 Running parameter search...")
    results = search_engine.run_search(
        estimation_date=pd.Timestamp('2021-01-01'),
        save_results=False,
        report_interval=2
    )
    
    print(f"\n📊 Results Summary:")
    print(f"  - Total combinations tested: {results['summary']['total_combinations']}")
    print(f"  - Successful combinations: {results['summary']['successful']}")
    print(f"  - Success rate: {results['summary']['success_rate']:.1%}")
    
    # Show best result
    if results['results']:
        best_result = search_engine.get_best_combination(
            results['results'], 'mean_mse', minimize=True
        )
        print(f"\n🏆 Best Result:")
        print(f"  - Method: {best_result['combination']['method']}")
        print(f"  - Parameters: {best_result['combination']['parameters']}")
        print(f"  - Horizon: {best_result['combination']['horizon']}")
        print(f"  - MSE: {best_result['aggregate_metrics']['mean_mse']:.6f}")
        print(f"  - Success rate: {best_result['success_rate_across_exposures']:.1%}")
    
    print("\n=== Integration Complete ===")
    print("\nKey benefits of using CompatibilityValidationFramework:")
    print("- Drop-in replacement for ParameterValidationFramework")
    print("- Uses new modular architecture underneath")
    print("- Same interface, improved performance and modularity")
    print("- Better suited for AI-assisted development")

if __name__ == "__main__":
    main()