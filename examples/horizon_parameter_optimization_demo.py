#!/usr/bin/env python3
"""
Global Forecast Horizon Parameter Optimization Demo

This script demonstrates the new horizon-specific parameter optimization framework
that ensures mathematical consistency by optimizing all parameters for the same
forecast horizon.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import pandas as pd
from datetime import datetime, timedelta
from src.data.exposure_universe import ExposureUniverse
from src.optimization.parameter_optimization import ParameterOptimizer, OptimizationConfig
from src.optimization.horizon_validator import HorizonValidator, validate_horizon_consistency
from src.optimization.exposure_risk_estimator import ExposureRiskEstimator

def main():
    """Demonstrate horizon-specific parameter optimization."""
    print("🎯 Global Forecast Horizon Parameter Optimization Demo")
    print("=" * 60)
    
    # Step 1: Load exposure universe
    print("\n1️⃣ Loading Exposure Universe...")
    try:
        exposure_universe = ExposureUniverse.from_yaml("config/exposure_universe.yaml")
        print(f"   ✅ Loaded {len(exposure_universe)} exposures")
        
        # Get subset for demo
        demo_exposures = ['us_large_equity', 'us_small_equity', 'government_bonds']
        available_exposures = [exp_id for exp_id in demo_exposures 
                             if exposure_universe.get_exposure(exp_id)]
        print(f"   📊 Demo exposures: {available_exposures}")
        
    except Exception as e:
        print(f"   ❌ Error loading exposure universe: {e}")
        print("   Using mock universe for demonstration...")
        exposure_universe = None
        available_exposures = ['us_large_equity', 'us_small_equity']
    
    # Step 2: Configure optimization for specific horizon
    print(f"\n2️⃣ Configuring Optimization for 21-Day Horizon...")
    target_horizon = 21  # Monthly rebalancing
    
    print(f"   🎯 Target forecast horizon: {target_horizon} days")
    print(f"   📅 Mathematical consistency: All parameters optimized for same horizon")
    print(f"   🔄 Recommended rebalancing: Every {target_horizon} days")
    
    # Step 3: Demonstrate parameter optimization (simulated)
    print(f"\n3️⃣ Horizon-Specific Parameter Optimization...")
    print("   (Simulating optimization process - would use real data in production)")
    
    # Create example optimization result
    optimization_result = {
        'global_settings': {
            'forecast_horizon': target_horizon,
            'optimization_date': datetime.now().strftime('%Y-%m-%d'),
            'total_combinations_tested': 100,
            'best_combined_score': 0.523,
            'mathematical_consistency': True
        },
        f'horizon_{target_horizon}_parameters': {
            'volatility': {
                'us_large_equity': {
                    'method': 'ewma',
                    'lambda': 0.94,
                    'min_periods': 30,
                    'validation_score': 0.023,
                    'sample_size': 500
                },
                'us_small_equity': {
                    'method': 'ewma',
                    'lambda': 0.92,  # More responsive for small cap
                    'min_periods': 30,
                    'validation_score': 0.031,
                    'sample_size': 480
                },
                'government_bonds': {
                    'method': 'ewma',
                    'lambda': 0.96,  # More stable for bonds
                    'min_periods': 30,
                    'validation_score': 0.018,
                    'sample_size': 520
                }
            },
            'correlation': {
                'method': 'ewma',
                'lambda': 0.95,  # Single lambda for all correlations
                'min_periods': 60,
                'validation_score': 0.142,  # Frobenius norm
                'sample_size': 450
            }
        },
        'validation_summary': {
            'volatility_performance': {
                'best_mse': 0.018,
                'worst_mse': 0.031,
                'average_mse': 0.024,
                'exposures_improved': 3
            },
            'correlation_performance': {
                'best_frobenius_norm': 0.142,
                'improvement_vs_sample': 0.23
            }
        }
    }
    
    print(f"   ✅ Optimization completed!")
    print(f"      Combinations tested: {optimization_result['global_settings']['total_combinations_tested']}")
    print(f"      Best combined score: {optimization_result['global_settings']['best_combined_score']:.3f}")
    print(f"      All parameters optimized for {target_horizon}-day forecasts")
    
    # Step 4: Save optimized configuration
    print(f"\n4️⃣ Saving Horizon-Specific Configuration...")
    config_path = "config/optimal_parameters_v2_demo.yaml"
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(optimization_result, f, default_flow_style=False, sort_keys=False)
        print(f"   ✅ Configuration saved to {config_path}")
        
        # Show parameter details
        vol_params = optimization_result[f'horizon_{target_horizon}_parameters']['volatility']
        corr_params = optimization_result[f'horizon_{target_horizon}_parameters']['correlation']
        
        print(f"\n   📊 Optimized Parameters for {target_horizon}-Day Horizon:")
        print(f"   Volatility Parameters:")
        for exposure, params in vol_params.items():
            print(f"     {exposure:<20}: λ={params['lambda']:.3f}, MSE={params['validation_score']:.3f}")
        
        print(f"   Correlation Parameters:")
        print(f"     {'all_pairs':<20}: λ={corr_params['lambda']:.3f}, Frobenius={corr_params['validation_score']:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error saving configuration: {e}")
        config_path = None
    
    # Step 5: Validate horizon consistency
    print(f"\n5️⃣ Validating Horizon Consistency...")
    
    if config_path:
        try:
            # Use the validator to check consistency
            result = validate_horizon_consistency(
                exposures=available_exposures,
                target_horizon=target_horizon,
                config_path=config_path
            )
            
            if result.is_consistent:
                print(f"   ✅ Horizon consistency validation PASSED")
                print(f"      All parameters optimized for {result.target_horizon}-day horizon")
                print(f"      Coverage: {result.horizon_coverage*100:.1f}% of exposures")
                print(f"      Configuration consistency: {'✅' if result.configuration_consistency else '❌'}")
                print(f"      Parameter consistency: {'✅' if result.parameter_consistency else '❌'}")
            else:
                print(f"   ❌ Horizon consistency validation FAILED")
                print(f"      Violations: {len(result.violations)}")
                for violation in result.violations:
                    print(f"        - {violation}")
                    
        except Exception as e:
            print(f"   ⚠️  Validation error: {e}")
    else:
        print(f"   ⚠️  Skipping validation (no config file)")
    
    # Step 6: Demonstrate usage in risk estimation
    print(f"\n6️⃣ Using Horizon-Consistent Parameters in Risk Estimation...")
    
    if config_path and exposure_universe:
        try:
            # Initialize risk estimator with fixed horizon
            risk_estimator = ExposureRiskEstimator(
                exposure_universe=exposure_universe,
                forecast_horizon=target_horizon,
                parameter_config_path=config_path
            )
            
            print(f"   ✅ Risk estimator initialized with {target_horizon}-day horizon")
            print(f"      All risk estimates will be for {target_horizon}-day forecasts")
            print(f"      Parameters loaded from horizon-specific configuration")
            print(f"      Mathematical consistency: GUARANTEED")
            
            # Show that the estimator enforces consistent horizons
            print(f"\n   🔒 Horizon Enforcement:")
            print(f"      Fixed forecast horizon: {risk_estimator.forecast_horizon} days")
            print(f"      No per-call horizon parameter allowed")
            print(f"      All estimates mathematically consistent")
            
        except Exception as e:
            print(f"   ❌ Error creating risk estimator: {e}")
    else:
        print(f"   ⚠️  Skipping risk estimator demo (dependencies missing)")
    
    # Step 7: Show benefits of horizon consistency
    print(f"\n7️⃣ Benefits of Global Forecast Horizon Framework:")
    print(f"   🎯 Mathematical Consistency:")
    print(f"      • All volatility estimates are {target_horizon}-day forecasts")
    print(f"      • All correlation estimates are {target_horizon}-day forecasts") 
    print(f"      • Portfolio optimization problem is well-defined")
    print(f"      • Risk estimates can be directly combined in optimization")
    
    print(f"\n   📈 Improved Accuracy:")
    print(f"      • Parameters optimized specifically for {target_horizon}-day horizon")
    print(f"      • Better prediction accuracy vs one-size-fits-all parameters")
    print(f"      • Validation ensures parameters work for intended horizon")
    
    print(f"\n   🔧 Operational Benefits:")
    print(f"      • Clear rebalancing schedule (every {target_horizon} days)")
    print(f"      • Consistent risk measurement across all assets") 
    print(f"      • Automatic validation of configuration consistency")
    print(f"      • Easy to switch between different horizon configurations")
    
    # Step 8: Usage recommendations
    print(f"\n8️⃣ Usage Recommendations:")
    print(f"   📅 Horizon Selection:")
    print(f"      • 5 days: High-frequency trading strategies")
    print(f"      • 21 days: Monthly rebalancing (recommended for most portfolios)")
    print(f"      • 63 days: Quarterly rebalancing for long-term strategies")
    print(f"      • 252 days: Annual rebalancing for very stable portfolios")
    
    print(f"\n   🔄 Implementation Steps:")
    print(f"      1. Choose target forecast horizon based on rebalancing frequency")
    print(f"      2. Run parameter optimization for that specific horizon")
    print(f"      3. Validate horizon consistency across all components")
    print(f"      4. Initialize risk estimator with fixed horizon")
    print(f"      5. Ensure portfolio optimization uses same horizon")
    
    print(f"\n   ⚠️  Critical Requirements:")
    print(f"      • ALL risk estimates must use the SAME forecast horizon")
    print(f"      • Parameters must be optimized FOR the target horizon")
    print(f"      • Configuration must enforce mathematical consistency")
    print(f"      • Regular validation ensures ongoing consistency")
    
    print(f"\n🎉 Demo completed! Horizon-specific optimization ensures mathematical")
    print(f"    consistency and improved accuracy for portfolio optimization.")
    
    if config_path:
        print(f"\n📁 Generated files:")
        print(f"    {config_path} - Horizon-specific parameter configuration")

if __name__ == "__main__":
    main()