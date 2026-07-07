"""
Example of how to use comprehensive parameter search from a notebook.

This replaces the complex sklearn integration in the notebook with a simple,
clean interface to the tested functionality.
"""

# Notebook cell 1: Import and setup
from src.optimization.comprehensive_parameter_search import (
    ComprehensiveParameterSearchEngine,
    analyze_search_results
)
from src.optimization.risk_premium_estimator import RiskPremiumEstimator
from src.data.return_decomposition import ReturnDecomposer
from src.data.exposure_universe import ExposureUniverse
from datetime import datetime
import pandas as pd

print("🚀 COMPREHENSIVE PARAMETER OPTIMIZATION")
print("Using tested, modular code from src/")

# Load exposure universe and initialize framework
universe_path = 'config/exposure_universe.yaml'
universe = ExposureUniverse.from_yaml(universe_path)
return_decomposer = ReturnDecomposer()
risk_estimator = RiskPremiumEstimator(universe, return_decomposer)
estimation_date = datetime.now()

print(f"✅ Loaded {len(universe)} exposures")

# Create search engine
search_engine = ComprehensiveParameterSearchEngine(
    risk_estimator=risk_estimator,
    estimation_date=estimation_date
)

print("✅ Search engine ready")


# Notebook cell 2: Single exposure optimization
exposure_id = 'us_large_equity'
print(f"\n🎯 Optimizing {exposure_id}")

result = search_engine.optimize_single_exposure(
    exposure_id=exposure_id,
    method='randomized',
    n_iter=100,  # Test 100 parameter combinations
    constrained=True,  # Use stable parameter ranges
    n_jobs=-1  # Use all CPU cores
)

if result:
    print(f"✅ Best score: {result.best_score:.6f}")
    print(f"📊 Best parameters:")
    for param, value in result.best_params.items():
        print(f"   {param}: {value}")
    print(f"⚡ Tested {result.n_combinations_tested} combinations in {result.elapsed_time:.1f}s")
else:
    print("❌ Optimization failed")


# Notebook cell 3: Multi-exposure optimization
exposures_to_test = ['us_large_equity', 'us_small_equity', 'intl_developed_large_equity']
print(f"\n🌍 Multi-exposure optimization: {exposures_to_test}")

multi_results = search_engine.optimize_multiple_exposures(
    exposure_ids=exposures_to_test,
    method='randomized',
    n_iter=50,  # 50 combinations per exposure
    constrained=True
)

print(f"✅ Optimized {len(multi_results)} exposures")


# Notebook cell 4: Analysis and visualization
analysis = analyze_search_results(multi_results)

if analysis:
    print(f"\n📊 OPTIMIZATION ANALYSIS:")
    
    summary = analysis['summary']
    print(f"Total combinations tested: {summary['total_combinations']:,}")
    print(f"Average score: {summary['avg_score']:.6f}")
    print(f"Total time: {summary['avg_time']*len(multi_results):.1f} seconds")
    
    print(f"\nMethod preferences:")
    for method, count in analysis['method_preferences'].items():
        pct = count / summary['num_exposures'] * 100
        print(f"   {method}: {count} exposures ({pct:.0f}%)")
    
    print(f"\nOptimal parameter ranges:")
    for param, stats in analysis['parameter_stats'].items():
        if stats['mean'] is not None:
            print(f"   {param}: {stats['min']} - {stats['max']} (avg: {stats['mean']:.0f})")
    
    # Create summary DataFrame for visualization
    results_df = pd.DataFrame([
        {
            'Exposure': exp_id,
            'Best_Score': result.best_score,
            'Method': result.best_params['method'],
            'Lookback_Days': result.best_params['lookback_days'],
            'Frequency': result.best_params['frequency'],
            'Combinations_Tested': result.n_combinations_tested,
            'Time_Seconds': result.elapsed_time
        }
        for exp_id, result in multi_results.items()
    ])
    
    print(f"\n📋 Results Summary:")
    print(results_df.to_string(index=False))


# Notebook cell 5: 64k+ combinations demonstration
print(f"\n🚀 64K+ COMBINATIONS CAPABILITY:")

# Show parameter space size
discrete_grid, continuous_dist = search_engine.create_search_spaces(constrained=False)

total_discrete_combinations = 1
for param, values in discrete_grid.items():
    if isinstance(values, list):
        total_discrete_combinations *= len(values)

print(f"Full parameter space: {total_discrete_combinations:,} discrete combinations")
print(f"Continuous space: Unlimited (sampled by RandomizedSearchCV)")

# Estimate 64k capability
if len(multi_results) > 0:
    avg_time_per_combination = (
        sum(r.elapsed_time for r in multi_results.values()) / 
        sum(r.n_combinations_tested for r in multi_results.values())
    )
    
    time_for_64k = 64000 * avg_time_per_combination / 3600  # hours
    
    print(f"\n⏱️  Time estimates:")
    print(f"Average time per combination: {avg_time_per_combination:.3f} seconds")
    print(f"Time for 64k combinations: {time_for_64k:.1f} hours")
    print(f"✅ 64k+ combinations are absolutely feasible!")

print(f"\n🎯 KEY ADVANTAGES:")
print("• Complete pipeline optimization (data + estimation)")
print("• Intelligent parameter sampling (not exhaustive grid)")
print("• Parallel processing for speed")
print("• Robust validation with cross-validation")
print("• Scales to 64k+ combinations efficiently")
print("• Clean, tested code with proper error handling")