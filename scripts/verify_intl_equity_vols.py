#!/usr/bin/env python
"""
Quick script to verify volatility estimates for international developed equities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from datetime import datetime, timedelta
from data.exposure_universe import ExposureUniverse
from data.multi_frequency import MultiFrequencyDataFetcher, Frequency

# Load exposure universe
universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
universe = ExposureUniverse.from_yaml(str(universe_path))

# Get the two exposures
large_cap = universe.get_exposure('intl_developed_large_equity')
small_cap = universe.get_exposure('intl_developed_small_equity')

print("International Developed Equity Comparison")
print("=" * 50)
print(f"\nLarge Cap: {large_cap.name}")
print(f"Tickers: {large_cap.get_preferred_implementation().tickers}")
print(f"\nSmall Cap: {small_cap.name}")
print(f"Tickers: {small_cap.get_preferred_implementation().tickers}")

# Fetch data
fetcher = MultiFrequencyDataFetcher()
end_date = datetime.now()
start_date = end_date - timedelta(days=1260)  # 5 years

print("\nFetching returns data...")

# Get returns for each
results = {}
for exp_name, exp in [('Large Cap', large_cap), ('Small Cap', small_cap)]:
    impl = exp.get_preferred_implementation()
    if impl.tickers:
        print(f"\n{exp_name} ETFs:")
        ticker_vols = {}
        
        for ticker in impl.tickers:
            try:
                returns = fetcher._fetch_single_ticker_returns(
                    ticker, start_date, end_date, Frequency.DAILY, validate=True
                )
                if len(returns) > 252:
                    # Calculate annualized volatility
                    vol = returns.std() * (252 ** 0.5)
                    ticker_vols[ticker] = vol
                    print(f"  {ticker}: {vol:.2%} annualized volatility ({len(returns)} days)")
            except Exception as e:
                print(f"  {ticker}: Failed - {e}")
        
        if ticker_vols:
            avg_vol = sum(ticker_vols.values()) / len(ticker_vols)
            results[exp_name] = avg_vol
            print(f"  Average: {avg_vol:.2%}")

print("\n" + "=" * 50)
print("SUMMARY:")
if 'Large Cap' in results and 'Small Cap' in results:
    print(f"Large Cap Vol: {results['Large Cap']:.2%}")
    print(f"Small Cap Vol: {results['Small Cap']:.2%}")
    print(f"Difference: {results['Small Cap'] - results['Large Cap']:.2%}")
    
    if results['Small Cap'] < results['Large Cap']:
        print("\n⚠️  WARNING: Small cap showing LOWER volatility than large cap!")
        print("This is counterintuitive and may indicate:")
        print("1. Data quality issues with one of the ETFs")
        print("2. Different geographic/sector exposures")
        print("3. Currency hedging differences")
else:
    print("Could not compare - missing data")
