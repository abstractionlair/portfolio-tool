#!/usr/bin/env python3
"""
Test to check if dividend data is causing the total returns issue.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from datetime import date, timedelta
from src.data.interfaces import RawDataType
from src.data.providers import RawDataProviderCoordinator, TransformedDataProvider
from src.data.providers.calculators import ReturnCalculator

def test_dividend_issue():
    """Test if dividend data is causing the issue."""
    print("🧪 Dividend Data Issue Test")
    print("=" * 40)
    
    # Setup
    raw_coordinator = RawDataProviderCoordinator()
    transformed_provider = TransformedDataProvider(raw_coordinator)
    calculator = ReturnCalculator()
    
    symbol = 'AAPL'
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    extended_start = start_date - timedelta(days=7)
    
    print(f"📊 Testing {symbol} dividend issue")
    
    # Get price data
    try:
        prices = transformed_provider.raw_provider.get_data(
            RawDataType.ADJUSTED_CLOSE, extended_start, end_date, symbol, "daily"
        )
        print(f"✅ Prices: {len(prices)} values, timezone: {getattr(prices.index, 'tz', 'None')}")
        
    except Exception as e:
        print(f"❌ Failed to get prices: {str(e)}")
        return
    
    # Get dividend data
    try:
        dividends = transformed_provider.raw_provider.get_data(
            RawDataType.DIVIDEND, extended_start, end_date, symbol, "daily"
        )
        print(f"✅ Dividends: {len(dividends)} values, timezone: {getattr(dividends.index, 'tz', 'None')}")
        print(f"   Non-zero dividends: {len(dividends[dividends > 0])}")
        print(f"   Total dividend amount: ${dividends.sum():.4f}")
        
    except Exception as e:
        print(f"❌ Failed to get dividends: {str(e)}")
        dividends = None
    
    # Test 1: Total returns with None dividends (should work like simple returns)
    print(f"\n1️⃣ Testing total returns with None dividends...")
    try:
        returns_no_div = calculator.calculate_total_returns(prices, None)
        print(f"   ✅ No dividends: {len(returns_no_div)} values, {returns_no_div.count()} non-null")
        
    except Exception as e:
        print(f"   ❌ No dividends failed: {str(e)}")
    
    # Test 2: Total returns with actual dividend data (the problematic case)
    if dividends is not None:
        print(f"\n2️⃣ Testing total returns with dividend data...")
        try:
            returns_with_div = calculator.calculate_total_returns(prices, dividends)
            print(f"   ✅ With dividends: {len(returns_with_div)} values, {returns_with_div.count()} non-null")
            
            if returns_with_div.count() == 0:
                print(f"   🚨 FOUND IT: Dividend data is causing the issue!")
                print(f"   Price index sample: {prices.index[:3]}")
                print(f"   Dividend index sample: {dividends.index[:3]}")
                
                # Test with timezone-aligned dividend data
                print(f"\n   🔧 Testing with timezone-aligned dividends...")
                if hasattr(prices.index, 'tz') and prices.index.tz is not None:
                    aligned_dividends = dividends.copy()
                    if aligned_dividends.index.tz is None:
                        aligned_dividends.index = aligned_dividends.index.tz_localize(prices.index.tz)
                    else:
                        aligned_dividends.index = aligned_dividends.index.tz_convert(prices.index.tz)
                    
                    returns_aligned = calculator.calculate_total_returns(prices, aligned_dividends)
                    print(f"   ✅ Aligned dividends: {len(returns_aligned)} values, {returns_aligned.count()} non-null")
                    
                    if returns_aligned.count() > 0:
                        print(f"   🎉 SUCCESS: Timezone alignment fixed the issue!")
                
        except Exception as e:
            print(f"   ❌ With dividends failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test 3: Compare with simple returns
    print(f"\n3️⃣ Comparing with simple returns...")
    try:
        simple_returns = calculator.calculate_simple_returns(prices)
        print(f"   ✅ Simple returns: {len(simple_returns)} values, {simple_returns.count()} non-null")
        
    except Exception as e:
        print(f"   ❌ Simple returns failed: {str(e)}")

if __name__ == "__main__":
    test_dividend_issue()