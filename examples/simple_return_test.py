#!/usr/bin/env python3
"""
Simple test to isolate the return calculation issue.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import date
from src.data.interfaces import RawDataType
from src.data.providers import RawDataProviderCoordinator, TransformedDataProvider

def test_simple_return_calculation():
    """Test return calculation with step-by-step debugging."""
    print("🧪 Simple Return Calculation Test")
    print("=" * 40)
    
    # Get real price data
    raw_coordinator = RawDataProviderCoordinator()
    transformed_provider = TransformedDataProvider(raw_coordinator)
    
    symbol = 'AAPL'
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)  # Just January for simplicity
    
    print(f"📊 Testing {symbol} for January 2023")
    
    # Get raw price data
    try:
        prices = transformed_provider.get_data(
            data_type=RawDataType.ADJUSTED_CLOSE,
            start=start_date,
            end=end_date,
            ticker=symbol,
            frequency="daily"
        )
        print(f"✅ Got {len(prices)} price observations")
        print(f"   First 5 prices: {prices.head().values}")
        print(f"   Price data type: {prices.dtype}")
        print(f"   Any null prices? {prices.isnull().any()}")
        
    except Exception as e:
        print(f"❌ Failed to get prices: {str(e)}")
        return
    
    # Manual return calculation (simplified)
    print(f"\n🧮 Manual return calculation...")
    
    # Step 1: Remove nulls
    clean_prices = prices.dropna()
    print(f"   After dropna: {len(clean_prices)} prices")
    
    if len(clean_prices) < 2:
        print("   ❌ Not enough data for returns")
        return
    
    # Step 2: Calculate simple returns manually
    prev_prices = clean_prices.shift(1)
    manual_returns = clean_prices / prev_prices - 1
    
    print(f"   Manual returns calculated: {len(manual_returns)}")
    print(f"   Non-null manual returns: {manual_returns.count()}")
    print(f"   First 5 returns: {manual_returns.head().values}")
    
    # Step 3: Test with our return calculator
    print(f"\n🔧 Testing ReturnCalculator...")
    from src.data.providers.calculators import ReturnCalculator
    calculator = ReturnCalculator()
    
    # Test simple returns first (no dividends)
    try:
        simple_returns = calculator.calculate_simple_returns(prices)
        print(f"   Simple returns: {len(simple_returns)} values")
        print(f"   Non-null simple returns: {simple_returns.count()}")
        print(f"   First 5 simple returns: {simple_returns.head().values}")
        
    except Exception as e:
        print(f"   ❌ Simple returns failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Test total returns (with None dividends - should be same as simple)
    try:
        total_returns_no_div = calculator.calculate_total_returns(prices, None)
        print(f"   Total returns (no div): {len(total_returns_no_div)} values")
        print(f"   Non-null total returns: {total_returns_no_div.count()}")
        print(f"   First 5 total returns: {total_returns_no_div.head().values}")
        
    except Exception as e:
        print(f"   ❌ Total returns (no div) failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare results
    print(f"\n🔍 Comparison:")
    print(f"   Manual: {manual_returns.count()} valid returns")
    print(f"   Simple: {simple_returns.count()} valid returns")
    print(f"   Total (no div): {total_returns_no_div.count()} valid returns")
    
    if manual_returns.count() > 0:
        print(f"   ✅ Manual calculation works!")
    if simple_returns.count() > 0:
        print(f"   ✅ Simple returns work!")
    if total_returns_no_div.count() > 0:
        print(f"   ✅ Total returns work!")
    
    # If they're different, investigate
    if (manual_returns.count() > 0 and 
        (simple_returns.count() == 0 or total_returns_no_div.count() == 0)):
        print(f"\n🚨 Issue found in ReturnCalculator implementation!")

if __name__ == "__main__":
    test_simple_return_calculation()