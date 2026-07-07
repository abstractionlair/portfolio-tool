#!/usr/bin/env python
"""
Diagnose the discrepancy between direct volatility calculation and risk estimator output.
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

# Test exposures
test_exposures = ['intl_developed_large_equity', 'intl_developed_small_equity']

# Set up dates
estimation_date = datetime.now()
lookback_days = 1260  # 5 years
start_date = estimation_date - timedelta(days=lookback_days)
forecast_horizon = 252  # 1 year

print("Volatility Calculation Diagnosis")
print("=" * 60)
print(f"Estimation date: {estimation_date.strftime('%Y-%m-%d')}")
print(f"Lookback period: {lookback_days} days")
print(f"Forecast horizon: {forecast_horizon} days")

# Method 1: Direct calculation (like verification script)
print("\n1. DIRECT CALCULATION (from ETF returns):")
print("-" * 60)

fetcher = MultiFrequencyDataFetcher()
direct_vols = {}

for exp_id in test_exposures:
    exposure = universe.get_exposure(exp_id)
    impl = exposure.get_preferred_implementation()
    
    print(f"\n{exposure.name}:")
    ticker_vols = []
    
    for ticker in impl.tickers:
        try:
            returns = fetcher._fetch_single_ticker_returns(
                ticker, start_date, estimation_date, Frequency.DAILY, validate=True
            )
            if len(returns) > 252:
                vol = returns.std() * np.sqrt(252)
                ticker_vols.append(vol)
                print(f"  {ticker}: {vol:.2%} ({len(returns)} days)")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
    
    if ticker_vols:
        avg_vol = np.mean(ticker_vols)
        direct_vols[exp_id] = avg_vol
        print(f"  Average: {avg_vol:.2%}")

# Method 2: Risk estimator
print("\n\n2. RISK ESTIMATOR OUTPUT:")
print("-" * 60)

risk_estimator = ExposureRiskEstimator(universe)

for method in ['historical', 'ewma', 'optimal']:
    print(f"\n{method.upper()} method:")
    
    risk_estimates = risk_estimator.estimate_exposure_risks(
        test_exposures,
        estimation_date,
        lookback_days=lookback_days,
        forecast_horizon=forecast_horizon,
        method=method
    )
    
    for exp_id, estimate in risk_estimates.items():
        exposure = universe.get_exposure(exp_id)
        print(f"  {exposure.name}: {estimate.volatility:.2%}")
        if method == 'historical':
            # Store for comparison
            risk_estimator_vol = estimate.volatility

# Method 3: Check the risk estimator's internal calculation
print("\n\n3. DEBUGGING RISK ESTIMATOR INTERNALS:")
print("-" * 60)

# Manually trace through what the risk estimator is doing
for exp_id in test_exposures:
    exposure = universe.get_exposure(exp_id)
    print(f"\n{exposure.name}:")
    
    # Load returns the same way risk estimator does
    returns = risk_estimator._load_exposure_returns(exp_id, estimation_date, lookback_days)
    
    if returns is not None:
        print(f"  Loaded {len(returns)} returns")
        print(f"  Date range: {returns.index[0]} to {returns.index[-1]}")
        
        # Calculate volatility different ways
        daily_vol = returns.std()
        annual_vol_252 = daily_vol * np.sqrt(252)
        annual_vol_365 = daily_vol * np.sqrt(365)
        
        print(f"  Daily volatility: {daily_vol:.4%}")
        print(f"  Annualized (√252): {annual_vol_252:.2%}")
        print(f"  Annualized (√365): {annual_vol_365:.2%}")
        
        # Check if it's using a different forecast adjustment
        forecast_adjustment = np.sqrt(forecast_horizon / 252)
        adjusted_vol = annual_vol_252 * forecast_adjustment
        print(f"  With forecast adjustment (√{forecast_horizon}/252): {adjusted_vol:.2%}")
    else:
        print("  Failed to load returns!")

# Summary
print("\n\n4. SUMMARY:")
print("=" * 60)

for exp_id in test_exposures:
    exposure = universe.get_exposure(exp_id)
    direct = direct_vols.get(exp_id, 0) * 100
    
    print(f"\n{exposure.name}:")
    print(f"  Direct calculation: {direct:.1f}%")
    print(f"  Risk estimator: ???")  # Would need to store from above
    print(f"  Discrepancy: ???")
