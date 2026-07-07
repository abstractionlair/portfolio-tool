#!/usr/bin/env python3
"""
Test the full pipeline to find where the returns are getting lost.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from datetime import date
from src.data.interfaces import LogicalDataType, RawDataType
from src.data.providers import RawDataProviderCoordinator, TransformedDataProvider

def test_pipeline():
    """Test each step of the total returns pipeline."""
    print("🔬 Full Pipeline Test")
    print("=" * 40)
    
    # Setup
    raw_coordinator = RawDataProviderCoordinator()
    transformed_provider = TransformedDataProvider(raw_coordinator)
    
    symbol = 'AAPL'
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    
    print(f"📊 Testing {symbol} pipeline for January 2023")
    
    # Step 1: Test _compute_total_returns method directly by accessing it
    print(f"\n1️⃣ Testing _compute_total_returns method...")
    try:
        # Access the private method for testing
        direct_result = transformed_provider._compute_total_returns(
            start=start_date,
            end=end_date,
            ticker=symbol,
            frequency="daily"
        )
        print(f"   ✅ Direct method result: {len(direct_result)} values")
        print(f"   Non-null values: {direct_result.count()}")
        if direct_result.count() > 0:
            print(f"   Sample values: {direct_result.dropna().head(3).values}")
        else:
            print(f"   ❌ All values are null!")
            
    except Exception as e:
        print(f"   ❌ Direct method failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Step 2: Test the public API
    print(f"\n2️⃣ Testing public get_data API...")
    try:
        api_result = transformed_provider.get_data(
            data_type=LogicalDataType.TOTAL_RETURN,
            start=start_date,
            end=end_date,
            ticker=symbol,
            frequency="daily"
        )
        print(f"   ✅ API result: {len(api_result)} values")
        print(f"   Non-null values: {api_result.count()}")
        if api_result.count() > 0:
            print(f"   Sample values: {api_result.dropna().head(3).values}")
        else:
            print(f"   ❌ All values are null!")
            
    except Exception as e:
        print(f"   ❌ API failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Compare with simple returns (should work)
    print(f"\n3️⃣ Testing simple returns for comparison...")
    try:
        simple_result = transformed_provider.get_data(
            data_type=LogicalDataType.SIMPLE_RETURN,
            start=start_date,
            end=end_date,
            ticker=symbol,
            frequency="daily"
        )
        print(f"   ✅ Simple returns: {len(simple_result)} values")
        print(f"   Non-null values: {simple_result.count()}")
        if simple_result.count() > 0:
            print(f"   Sample values: {simple_result.dropna().head(3).values}")
        
    except Exception as e:
        print(f"   ❌ Simple returns failed: {str(e)}")
    
    # Step 4: Test the constituent parts separately
    print(f"\n4️⃣ Testing constituent parts...")
    
    # Get raw prices with extended date range (like _compute_total_returns does)
    from datetime import timedelta
    extended_start = start_date - timedelta(days=7)  # Extend start date
    
    try:
        extended_prices = transformed_provider.raw_provider.get_data(
            data_type=RawDataType.ADJUSTED_CLOSE,
            start=extended_start,
            end=end_date,
            ticker=symbol,
            frequency="daily"
        )
        print(f"   Raw prices (extended): {len(extended_prices)} values, {extended_prices.count()} non-null")
        
        # Test the return calculator on extended data
        from src.data.providers.calculators import ReturnCalculator
        calculator = ReturnCalculator()
        
        calc_result = calculator.calculate_total_returns(extended_prices, None)
        print(f"   Calculator on extended: {len(calc_result)} values, {calc_result.count()} non-null")
        
        # Test the trimming function
        trimmed_result = transformed_provider._trim_and_convert(
            calc_result, start_date, end_date, "daily", "daily", "return"
        )
        print(f"   After trimming: {len(trimmed_result)} values, {trimmed_result.count()} non-null")
        
        if trimmed_result.count() == 0 and calc_result.count() > 0:
            print(f"   🚨 FOUND THE ISSUE: Trimming is removing all data!")
            print(f"   Original date range: {calc_result.index.min()} to {calc_result.index.max()}")
            print(f"   Requested range: {start_date} to {end_date}")
        
    except Exception as e:
        print(f"   ❌ Constituent parts test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()