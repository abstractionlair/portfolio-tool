#!/usr/bin/env python
"""
Script to understand how parameter optimization is incorporated into risk estimation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from datetime import datetime, timedelta
from data.exposure_universe import ExposureUniverse
from optimization.exposure_risk_estimator import ExposureRiskEstimator
from optimization.parameter_optimization import ParameterOptimizer

# Load exposure universe
universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
universe = ExposureUniverse.from_yaml(str(universe_path))

# Create risk estimator
risk_estimator = ExposureRiskEstimator(universe)

# Test exposures
test_exposures = ['us_large_equity', 'intl_developed_large_equity', 'intl_developed_small_equity']

# Check what parameters are used for different methods
print("Parameter Investigation")
print("=" * 50)

# Get estimation parameters for different methods
for method in ['optimal', 'ewma', 'historical']:
    print(f"\nMethod: {method}")
    
    # Check what parameters are returned
    params = risk_estimator._get_estimation_parameters(method, forecast_horizon=252)
    print(f"Parameters: {params}")

# Now compare volatility estimates
print("\n\nVolatility Estimates by Method (1-year horizon)")
print("=" * 50)

estimation_date = datetime.now()
forecast_horizon = 252

for method in ['historical', 'ewma', 'optimal']:
    print(f"\n{method.upper()} Method:")
    
    try:
        risk_estimates = risk_estimator.estimate_exposure_risks(
            test_exposures,
            estimation_date,
            forecast_horizon=forecast_horizon,
            method=method
        )
        
        for exp_id, estimate in risk_estimates.items():
            exp_name = universe.get_exposure(exp_id).name
            print(f"  {exp_name}: {estimate.volatility:.2%}")
            if method == 'optimal':
                print(f"    Parameters used: {estimate.parameters}")
    except Exception as e:
        print(f"  Error: {e}")

# Check if parameter optimizer exists
print("\n\nParameter Optimizer Status:")
print("=" * 50)
if risk_estimator.parameter_optimizer:
    print("Parameter optimizer is available")
    print("Optimal parameters would be loaded from previous optimization")
else:
    print("No parameter optimizer - using default parameters")
    print(f"Default params: {risk_estimator._default_params}")
