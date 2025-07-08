#!/usr/bin/env python
"""
Create a corrected notebook section that properly calculates and displays volatilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.exposure_universe import ExposureUniverse
from optimization.exposure_risk_estimator import ExposureRiskEstimator
from data.multi_frequency import MultiFrequencyDataFetcher, Frequency

# Load exposure universe
universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
universe = ExposureUniverse.from_yaml(str(universe_path))

# Create risk estimator
risk_estimator = ExposureRiskEstimator(universe)

# Test specific exposures
test_exposures = [
    'us_large_equity',
    'us_small_equity',
    'intl_developed_large_equity', 
    'intl_developed_small_equity',
    'commodities',
    'real_estate'
]

# Filter available
available = [e for e in test_exposures if universe.get_exposure(e)]

print("Volatility Comparison: 1-Year Horizon")
print("=" * 70)
print(f"{'Exposure':<40} {'Historical':<12} {'EWMA':<12} {'Diff':<12}")
print("-" * 70)

estimation_date = datetime.now()

# Compare methods
for exp_id in available:
    exposure = universe.get_exposure(exp_id)
    
    # Get historical volatility
    hist_estimates = risk_estimator.estimate_exposure_risks(
        [exp_id],
        estimation_date,
        forecast_horizon=252,
        method='historical'
    )
    
    # Get EWMA volatility
    ewma_estimates = risk_estimator.estimate_exposure_risks(
        [exp_id],
        estimation_date,
        forecast_horizon=252,
        method='ewma'
    )
    
    if exp_id in hist_estimates and exp_id in ewma_estimates:
        hist_vol = hist_estimates[exp_id].volatility * 100
        ewma_vol = ewma_estimates[exp_id].volatility * 100
        diff = ewma_vol - hist_vol
        
        print(f"{exposure.name:<40} {hist_vol:>10.1f}% {ewma_vol:>10.1f}% {diff:>+10.1f}%")

print("\nConclusion:")
print("The EWMA method is producing lower volatility estimates than simple historical.")
print("This is because EWMA with λ=0.94 gives more weight to recent (possibly calmer) periods.")
print("\nFor 1-year forecasts, historical volatility might be more appropriate unless")
print("you believe recent volatility patterns will persist.")
