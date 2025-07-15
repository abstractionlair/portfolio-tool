"""Demonstrate portfolio-level parameter optimization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
from datetime import datetime
from optimization.parameter_optimization import ParameterOptimizer
from data.exposure_universe import ExposureUniverse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run portfolio-level optimization demo."""
    
    print("=" * 80)
    print("PORTFOLIO-LEVEL PARAMETER OPTIMIZATION")
    print("=" * 80)
    print()
    print("This demo implements a two-level optimization:")
    print("1. For each candidate horizon, optimize parameters for all exposures")
    print("2. Select the horizon that gives best portfolio-level prediction accuracy")
    print()
    
    # Initialize
    print("Loading exposure universe...")
    universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
    print(f"Loaded {len(universe.exposures)} exposures")
    
    optimizer = ParameterOptimizer(universe)
    
    # Run two-level optimization with comprehensive horizon testing
    candidate_horizons = [5, 10, 21, 42, 63, 126, 189, 252, 365]
    print(f"\nTesting forecast horizons: {candidate_horizons} days")
    print("- 5 days: Weekly rebalancing")
    print("- 10 days: Bi-weekly rebalancing")
    print("- 21 days: Monthly rebalancing") 
    print("- 42 days: Bi-monthly rebalancing")
    print("- 63 days: Quarterly rebalancing")
    print("- 126 days: Semi-annual rebalancing")
    print("- 189 days: Tri-annual rebalancing")
    print("- 252 days: Annual rebalancing")
    print("- 365 days: Full year rebalancing")
    print()
    print("Optimizing for portfolio-level prediction accuracy...")
    print("This will test multiple portfolio compositions and select parameters")
    print("that best predict portfolio volatility, not just individual asset volatility.")
    print()
    
    try:
        results = optimizer.optimize_portfolio_level(
            candidate_horizons=candidate_horizons,
            start_date=datetime(2020, 1, 1),  # Shorter period for demo
            end_date=datetime(2024, 12, 31)
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)
        
        print(f"\nOptimal Forecast Horizon: {results['optimal_horizon']} days")
        
        optimal = results['optimal_parameters']
        print(f"Portfolio-Level Goodness Score: {optimal.goodness_score:.6f}")
        print(f"Portfolio RMSE: {optimal.validation_metrics.get('rmse', 0.0):.4f}")
        print(f"Number of Portfolio Tests: {optimal.validation_metrics.get('n_tests', 0)}")
        
        print("\nMethod Selection by Exposure:")
        methods = {}
        for exp_id, params in optimal.volatility_params.items():
            method = params['method']
            methods[method] = methods.get(method, 0) + 1
            validation_score = params.get('validation_score', 0.0)
            print(f"  {exp_id:25} : {method:10} (score: {validation_score:.6f})")
        
        print(f"\nMethod Summary:")
        for method, count in methods.items():
            print(f"  {method:10}: {count:2d} exposures")
        
        corr_params = optimal.correlation_params.get('parameters', {})
        corr_method = corr_params.get('method', 'unknown')
        corr_score = optimal.correlation_params.get('validation_score', 0.0)
        print(f"\nCorrelation Method: {corr_method} (score: {corr_score:.6f})")
        
        print("\nHorizon Comparison:")
        print("Horizon | Goodness Score | Portfolio RMSE")
        print("--------|----------------|---------------")
        for horizon, result in results['all_horizon_results'].items():
            rmse = result.validation_metrics.get('rmse', 0.0)
            print(f"  {horizon:2d} days |    {result.goodness_score:8.6f}  |     {rmse:.4f}")
        
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        print(f"✓ Optimal horizon selected: {results['optimal_horizon']} days")
        print("✓ Each exposure has individually optimized parameters")
        print("✓ Method selection varies by exposure characteristics:")
        
        # Analyze method selection patterns
        equity_methods = []
        bond_methods = []
        alt_methods = []
        
        for exp_id, params in optimal.volatility_params.items():
            method = params['method']
            if 'equity' in exp_id:
                equity_methods.append(method)
            elif any(x in exp_id for x in ['ust', 'bond', 'tips']):
                bond_methods.append(method)
            elif exp_id in ['trend_following', 'factor_style_equity', 'factor_style_other']:
                alt_methods.append(method)
        
        if equity_methods:
            print(f"  - Equity exposures: mostly {max(set(equity_methods), key=equity_methods.count)}")
        if bond_methods:
            print(f"  - Bond exposures: mostly {max(set(bond_methods), key=bond_methods.count)}")
        if alt_methods:
            print(f"  - Alternative exposures: mostly {max(set(alt_methods), key=alt_methods.count)}")
        
        print("✓ Portfolio-level validation ensures parameters work together")
        print(f"✓ Results saved to config/optimal_parameters_portfolio_level.yaml")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)