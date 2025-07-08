#!/usr/bin/env python
"""
Demonstrate risk estimation on decomposed risk premia vs raw returns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.exposure_universe import ExposureUniverse
from data.return_decomposition import ReturnDecomposer
from data.total_returns import TotalReturnFetcher
from data.multi_frequency import MultiFrequencyDataFetcher, Frequency

# Load universe
universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
universe = ExposureUniverse.from_yaml(str(universe_path))

# Initialize components
decomposer = ReturnDecomposer()
fetcher = MultiFrequencyDataFetcher()

# Test exposures
test_exposures = ['us_large_equity', 'intl_developed_large_equity', 'commodities']

# Date range
end_date = datetime.now()
start_date = end_date - timedelta(days=1260)  # 5 years

print("Risk Premium Decomposition Analysis")
print("=" * 80)
print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print()

for exp_id in test_exposures:
    exposure = universe.get_exposure(exp_id)
    if not exposure:
        continue
        
    print(f"\n{exposure.name}")
    print("-" * 60)
    
    # Get implementation
    impl = exposure.get_preferred_implementation()
    if not impl:
        print("  No implementation found")
        continue
    
    # For simplicity, use first ticker if multiple
    ticker = impl.tickers[0] if impl.tickers else impl.ticker
    if not ticker:
        print("  No ticker found")
        continue
        
    try:
        # Fetch raw returns
        raw_returns = fetcher._fetch_single_ticker_returns(
            ticker, start_date, end_date, Frequency.DAILY, validate=True
        )
        
        if len(raw_returns) < 252:
            print(f"  Insufficient data: {len(raw_returns)} days")
            continue
            
        # Decompose returns
        decomposition = decomposer.decompose_returns(
            raw_returns,
            start_date,
            end_date,
            frequency="daily",
            inflation_series="cpi_all",
            risk_free_maturity="3m"
        )
        
        # Calculate volatilities
        raw_vol = raw_returns.std() * np.sqrt(252)
        
        # Extract components
        if 'spread' in decomposition.columns:
            spread_vol = decomposition['spread'].std() * np.sqrt(252)
            inflation_vol = decomposition['inflation'].std() * np.sqrt(252)
            
            print(f"  Ticker: {ticker}")
            print(f"  Total Return Volatility: {raw_vol:.2%}")
            print(f"  Risk Premium Volatility: {spread_vol:.2%}")
            print(f"  Inflation Volatility: {inflation_vol:.2%}")
            print(f"  Difference (Total - Risk Premium): {(raw_vol - spread_vol):.2%}")
            
            # Show average returns
            total_return_avg = raw_returns.mean() * 252
            risk_premium_avg = decomposition['spread'].mean() * 252
            
            print(f"\n  Average Annual Total Return: {total_return_avg:.2%}")
            print(f"  Average Annual Risk Premium: {risk_premium_avg:.2%}")
            
        else:
            print("  Decomposition failed - no spread component")
            
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("Risk premium volatility should be similar to total return volatility for equities")
print("but may differ significantly for bonds and other assets where risk-free rate")
print("changes contribute substantial volatility.")
