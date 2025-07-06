#!/usr/bin/env python3
"""
Correct Real Returns Calculation Example

This script demonstrates the CORRECT way to calculate real returns to avoid
the common mistake of comparing monthly returns to annualized inflation rates.

COMMON MISTAKE: Using annualized inflation with monthly returns
CORRECT APPROACH: Match inflation frequency to return frequency
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from src.data import (
    ExposureUniverse, TotalReturnFetcher, FREDDataFetcher, 
    ReturnEstimationFramework
)

warnings.filterwarnings('ignore', category=FutureWarning)


def demonstrate_correct_calculation():
    """Show the correct way to calculate real returns."""
    print("=" * 80)
    print("CORRECT REAL RETURNS CALCULATION")
    print("=" * 80)
    
    # Setup
    universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
    fetcher = TotalReturnFetcher()
    fred_fetcher = FREDDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)  # 2 years
    frequency = "monthly"
    
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Frequency: {frequency}")
    print()
    
    # Step 1: Get nominal returns for a test exposure
    test_exposure = universe.get_exposure('us_large_equity')
    nominal_returns, _ = fetcher.fetch_returns_for_exposure(
        test_exposure, start_date, end_date, frequency
    )
    
    print("STEP 1: Nominal Returns")
    print("-" * 30)
    if not nominal_returns.empty:
        print(f"Fetched {len(nominal_returns)} monthly returns")
        print(f"Sample monthly return: {nominal_returns.iloc[5]:.4f} ({nominal_returns.iloc[5]*100:.2f}%)")
        print(f"Average monthly return: {nominal_returns.mean():.4f} ({nominal_returns.mean()*100:.2f}%)")
        print(f"Annualized return: {nominal_returns.mean() * 12:.4f} ({nominal_returns.mean() * 12 * 100:.2f}%)")
    print()
    
    # Step 2: Get inflation rates the CORRECT way
    print("STEP 2: Inflation Rates (CORRECT METHOD)")
    print("-" * 40)
    
    # ✅ CORRECT: Use the convenience method that matches frequencies
    inflation_rates = fred_fetcher.get_inflation_rates_for_returns(
        start_date, end_date, return_frequency=frequency
    )
    
    if not inflation_rates.empty:
        print(f"Sample monthly inflation: {inflation_rates.iloc[5]:.6f} ({inflation_rates.iloc[5]*100:.4f}%)")
        print(f"Average monthly inflation: {inflation_rates.mean():.6f} ({inflation_rates.mean()*100:.4f}%)")
        print(f"Approximate annual inflation: {inflation_rates.mean()*12:.4f} ({inflation_rates.mean()*12*100:.2f}%)")
    print()
    
    # Step 3: Calculate real returns
    print("STEP 3: Real Returns Calculation")
    print("-" * 35)
    
    if not nominal_returns.empty and not inflation_rates.empty:
        real_returns = fred_fetcher.convert_to_real_returns(
            nominal_returns, inflation_rates, method="exact"
        )
        
        if not real_returns.empty:
            print(f"Sample monthly real return: {real_returns.iloc[5]:.4f} ({real_returns.iloc[5]*100:.2f}%)")
            print(f"Average monthly real return: {real_returns.mean():.4f} ({real_returns.mean()*100:.2f}%)")
            print(f"Annualized real return: {real_returns.mean() * 12:.4f} ({real_returns.mean() * 12 * 100:.2f}%)")
            
            # Validation
            nominal_annual = nominal_returns.mean() * 12
            real_annual = real_returns.mean() * 12
            inflation_annual = inflation_rates.mean() * 12
            
            print()
            print("VALIDATION:")
            print(f"  Nominal annual return: {nominal_annual:.2%}")
            print(f"  Real annual return: {real_annual:.2%}")
            print(f"  Annual inflation: {inflation_annual:.2%}")
            print(f"  Difference (nominal - real): {nominal_annual - real_annual:.2%}")
            print(f"  Should approximately equal inflation: {inflation_annual:.2%}")
            
            # Check if the calculation makes sense
            difference = abs((nominal_annual - real_annual) - inflation_annual)
            if difference < 0.005:  # Within 0.5%
                print("  ✅ Calculation looks correct!")
            else:
                print(f"  ⚠️  Calculation may be incorrect (difference: {difference:.3%})")


def demonstrate_wrong_calculation():
    """Show the WRONG way that causes negative real returns."""
    print("\n" + "=" * 80)
    print("WRONG CALCULATION (FOR COMPARISON)")
    print("=" * 80)
    
    fred_fetcher = FREDDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Get monthly CPI data
    inflation_index = fred_fetcher.fetch_inflation_data(
        start_date, end_date, "cpi_all", "monthly"
    )
    
    if not inflation_index.empty:
        # ❌ WRONG: Annualize monthly inflation rates
        wrong_inflation_rates = fred_fetcher.calculate_inflation_rate(
            inflation_index, periods=1, annualize=True  # This is WRONG for monthly returns!
        )
        
        print("❌ WRONG METHOD: Annualized inflation with monthly returns")
        print(f"Sample 'monthly' rate: {wrong_inflation_rates.iloc[5]:.4f} ({wrong_inflation_rates.iloc[5]*100:.2f}%)")
        print(f"Average 'monthly' rate: {wrong_inflation_rates.mean():.4f} ({wrong_inflation_rates.mean()*100:.2f}%)")
        print()
        print("This creates huge 'inflation' values that make all real returns negative!")
        print("The 'monthly' rate is actually an annualized rate (4-5% instead of 0.2-0.4%)")


def main():
    """Run the demonstration."""
    demonstrate_correct_calculation()
    demonstrate_wrong_calculation()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS FOR NOTEBOOK USAGE")
    print("=" * 80)
    print()
    print("✅ CORRECT: Use FREDDataFetcher.get_inflation_rates_for_returns()")
    print("   This automatically matches inflation frequency to your return frequency")
    print()
    print("✅ CORRECT: Or use calculate_inflation_rate(annualize=False) for monthly returns")
    print()
    print("❌ WRONG: Don't use calculate_inflation_rate(annualize=True) with monthly returns")
    print("   This creates artificially high inflation rates causing negative real returns")
    print()
    print("📝 NOTEBOOK CODE TO USE:")
    print("```python")
    print("# Replace this WRONG code:")
    print("# inflation_rates = fred_fetcher.calculate_inflation_rate(")
    print("#     inflation_index, periods=1, annualize=True")
    print("#")
    print("# With this CORRECT code:")
    print("inflation_rates = fred_fetcher.get_inflation_rates_for_returns(")
    print("    start_date, end_date, return_frequency='monthly'")
    print(")")
    print("```")


if __name__ == "__main__":
    main()