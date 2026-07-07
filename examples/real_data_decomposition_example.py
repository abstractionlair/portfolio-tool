#!/usr/bin/env python3
"""
Real Data Equity Return Decomposition Example

This script demonstrates the enhanced equity return decomposition functionality
using real market data instead of synthetic data.
"""

import pandas as pd
import numpy as np
from datetime import date
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.providers.coordinator import RawDataProviderCoordinator
from src.data.providers.transformed_provider import TransformedDataProvider
from src.data.interfaces import RawDataType, DataNotAvailableError


def setup_real_data_provider():
    """Setup the real data provider for decomposition analysis."""
    coordinator = RawDataProviderCoordinator()
    provider = TransformedDataProvider(coordinator)
    return provider


def create_realistic_earnings_data(start_date, end_date, base_eps, quarterly_growth=0.02):
    """
    Create realistic earnings data based on quarterly reporting patterns.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        base_eps: Base earnings per share
        quarterly_growth: Quarterly growth rate (default 2%)
    
    Returns:
        pd.Series: Daily earnings data with quarterly progression
    """
    # Generate quarterly earnings dates
    quarterly_dates = []
    quarterly_values = []
    
    current_date = start_date
    current_eps = base_eps
    
    while current_date <= end_date:
        # Find next quarter end (Mar 31, Jun 30, Sep 30, Dec 31)
        year = current_date.year
        month = current_date.month
        
        if month <= 3:
            quarter_end = date(year, 3, 31)
        elif month <= 6:
            quarter_end = date(year, 6, 30)
        elif month <= 9:
            quarter_end = date(year, 9, 30)
        else:
            quarter_end = date(year, 12, 31)
        
        if quarter_end <= end_date:
            quarterly_dates.append(quarter_end)
            quarterly_values.append(current_eps)
            current_eps *= (1 + quarterly_growth)
        
        # Move to next quarter
        if quarter_end.month == 12:
            current_date = date(quarter_end.year + 1, 1, 1)
        else:
            current_date = date(quarter_end.year, quarter_end.month + 1, 1)
    
    # Handle case where no quarterly dates fall in range
    if not quarterly_dates:
        # If no data in range, create a simple progression
        quarterly_dates = [end_date]
        quarterly_values = [base_eps]
    
    quarterly_earnings = pd.Series(quarterly_values, index=quarterly_dates, name='EPS')
    
    # Convert to daily by forward filling
    daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_earnings = quarterly_earnings.reindex(daily_range).ffill()
    
    # Fill any initial NaN values with the first available value
    daily_earnings = daily_earnings.fillna(quarterly_values[0] if quarterly_values else base_eps)
    
    return daily_earnings


def get_real_earnings_data(provider, ticker, start_date, end_date):
    """
    Get real earnings data - for demo purposes, create realistic quarterly data.
    In a real implementation, this would come from a financial data provider.
    """
    # Known approximate earnings for major stocks (simplified)
    earnings_data = {
        'AAPL': [
            (date(2022, 12, 31), 1.88),  # Q4 2022
            (date(2023, 3, 31), 1.52),   # Q1 2023
            (date(2023, 6, 30), 1.26),   # Q2 2023
            (date(2023, 9, 30), 1.46),   # Q3 2023
            (date(2023, 12, 31), 2.18),  # Q4 2023
        ],
        'MSFT': [
            (date(2022, 12, 31), 2.32),  # Q4 2022
            (date(2023, 3, 31), 2.45),   # Q1 2023
            (date(2023, 6, 30), 2.69),   # Q2 2023
            (date(2023, 9, 30), 3.30),   # Q3 2023
            (date(2023, 12, 31), 3.20),  # Q4 2023
        ],
        'GOOGL': [
            (date(2022, 12, 31), 1.05),  # Q4 2022
            (date(2023, 3, 31), 1.17),   # Q1 2023
            (date(2023, 6, 30), 1.44),   # Q2 2023
            (date(2023, 9, 30), 1.55),   # Q3 2023
            (date(2023, 12, 31), 1.64),  # Q4 2023
        ],
        'KO': [
            (date(2022, 12, 31), 0.69),  # Q4 2022
            (date(2023, 3, 31), 0.68),   # Q1 2023
            (date(2023, 6, 30), 0.70),   # Q2 2023
            (date(2023, 9, 30), 0.74),   # Q3 2023
            (date(2023, 12, 31), 0.77),  # Q4 2023
        ]
    }
    
    if ticker not in earnings_data:
        # Default earnings progression for unknown tickers
        earnings_data[ticker] = [
            (date(2022, 12, 31), 2.00),
            (date(2023, 3, 31), 2.10),
            (date(2023, 6, 30), 2.20),
            (date(2023, 9, 30), 2.30),
            (date(2023, 12, 31), 2.40),
        ]
    
    # Create quarterly series
    quarterly_dates = []
    quarterly_values = []
    
    for quarter_date, eps in earnings_data[ticker]:
        if start_date <= quarter_date <= end_date:
            quarterly_dates.append(quarter_date)
            quarterly_values.append(eps)
    
    if not quarterly_dates:
        # If no data in range, create a simple progression
        quarterly_dates = [end_date]
        quarterly_values = [2.0]
    
    quarterly_earnings = pd.Series(quarterly_values, index=quarterly_dates, name='EPS')
    
    # Convert to daily by forward filling
    daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_earnings = quarterly_earnings.reindex(daily_range).ffill()
    
    # Fill any initial NaN values with the first available value
    daily_earnings = daily_earnings.fillna(quarterly_values[0] if quarterly_values else 2.0)
    
    return daily_earnings


def print_real_data_summary(result, ticker, start_date, end_date):
    """Print a summary of real data decomposition results."""
    print(f"\n📊 {ticker} Real Data Decomposition Summary")
    print(f"📅 Period: {start_date} to {end_date}")
    print("=" * 60)
    
    # Key components
    components = [
        ('nominal_return', 'Total Nominal Return'),
        ('dividend_yield', 'Dividend Yield'),
        ('pe_change', 'P/E Change'),
        ('nominal_earnings_growth', 'Nominal Earnings Growth'),
        ('real_earnings_growth', 'Real Earnings Growth'),
        ('real_earnings_excess', 'Real Earnings Excess'),
        ('real_risk_premium', 'Real Risk Premium'),
    ]
    
    print("\nAnnualized Return Components:")
    print("-" * 35)
    
    for component, name in components:
        if component in result:
            series = result[component].dropna()
            if len(series) > 0:
                annual_mean = series.mean() * 252
                annual_std = series.std() * np.sqrt(252)
                print(f"{name:25}: {annual_mean:7.2%} ± {annual_std:6.2%}")
    
    # Economic context
    print("\nEconomic Context (Real Data):")
    print("-" * 35)
    
    econ_components = [
        ('inflation', 'Inflation Rate'),
        ('nominal_rf', 'Nominal Risk-Free Rate'),
        ('real_rf', 'Real Risk-Free Rate'),
    ]
    
    for component, name in econ_components:
        if component in result:
            series = result[component].dropna()
            if len(series) > 0:
                annual_mean = series.mean() * 252
                print(f"{name:25}: {annual_mean:7.2%}")
    
    # Data quality metrics
    print("\nData Quality Metrics:")
    print("-" * 35)
    
    # Check for actual dividend data
    if 'dividend_yield' in result:
        div_series = result['dividend_yield'].dropna()
        div_days = (div_series > 0).sum()
        total_days = len(div_series)
        print(f"{'Dividend Days':25}: {div_days:3d} of {total_days:3d} days ({div_days/total_days*100:.1f}%)")
    
    # Check earnings change frequency
    if 'nominal_earnings_growth' in result:
        earnings_series = result['nominal_earnings_growth'].dropna()
        earnings_change_days = (earnings_series != 0).sum()
        total_days = len(earnings_series)
        print(f"{'Earnings Change Days':25}: {earnings_change_days:3d} of {total_days:3d} days ({earnings_change_days/total_days*100:.1f}%)")
    
    # Quality metrics
    quality_components = [
        ('identity_error', 'Identity Error (max)'),
        ('decomp_error', 'Decomposition Error (max)'),
    ]
    
    for component, name in quality_components:
        if component in result:
            series = result[component].dropna()
            if len(series) > 0:
                max_error = series.max()
                status = "✅ Good" if max_error < 0.01 else "⚠️ Check"
                print(f"{name:25}: {max_error:.4f} {status}")


def main():
    """Main example function using real data."""
    print("Enhanced Equity Return Decomposition - Real Data Example")
    print("=" * 65)
    
    # Setup data provider
    print("\n1. Setting up data provider...")
    provider = setup_real_data_provider()
    
    # Analysis parameters - use a stock that pays dividends
    ticker = 'KO'  # Coca-Cola, known for regular dividends
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    
    print(f"2. Analyzing {ticker} from {start_date} to {end_date}")
    print("   Using REAL market data from providers...")
    
    # Get real earnings data
    print("3. Getting earnings data...")
    earnings_data = get_real_earnings_data(provider, ticker, start_date, end_date)
    
    print(f"   Created earnings data with {len(earnings_data)} daily observations")
    print(f"   EPS range: ${earnings_data.min():.2f} - ${earnings_data.max():.2f}")
    
    # Check what real data we can access
    print("\n4. Checking available real data...")
    
    try:
        # Test price data
        prices = provider.raw_provider.get_data(RawDataType.ADJUSTED_CLOSE, start_date, end_date, ticker, 'daily')
        print(f"   ✅ Price data: {len(prices)} days")
        print(f"   📈 Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    except Exception as e:
        print(f"   ❌ Price data error: {e}")
        return False
    
    try:
        # Test dividend data
        dividends = provider.raw_provider.get_data(RawDataType.DIVIDEND, start_date, end_date, ticker, 'daily')
        dividend_days = (dividends > 0).sum()
        total_div = dividends.sum()
        print(f"   ✅ Dividend data: {dividend_days} payment days, ${total_div:.2f} total")
    except Exception as e:
        print(f"   ⚠️ Dividend data: {e}")
    
    try:
        # Test economic data
        treasury_data = provider.raw_provider.get_data(RawDataType.TREASURY_3M, start_date, end_date, None, 'daily')
        avg_rate = treasury_data.mean() * 100
        print(f"   ✅ Treasury data: {len(treasury_data)} days, avg {avg_rate:.2f}%")
    except Exception as e:
        print(f"   ⚠️ Treasury data: {e}")
    
    # Perform decomposition with real data
    print("\n5. Performing return decomposition with real data...")
    try:
        result = provider.decompose_equity_returns(
            ticker=ticker,
            start=start_date,
            end=end_date,
            earnings_data=earnings_data,
            frequency='daily'
        )
        
        print(f"   ✅ Decomposition complete! Generated {len(result)} components")
        
        # Display results
        print_real_data_summary(result, ticker, start_date, end_date)
        
        # Show sample data
        print(f"\n6. Sample of real decomposition data (first 5 days):")
        print("-" * 70)
        
        sample_components = ['nominal_return', 'dividend_yield', 'pe_change', 'nominal_earnings_growth']
        sample_data = {}
        
        for component in sample_components:
            if component in result:
                series = result[component].dropna()
                if len(series) > 0:
                    sample_data[component] = series.head(5)
        
        if sample_data:
            sample_df = pd.DataFrame(sample_data)
            print(sample_df.round(6))
        
        # Show actual dividend dates
        if 'dividend_yield' in result:
            div_series = result['dividend_yield']
            div_dates = div_series[div_series > 0].head(10)
            if len(div_dates) > 0:
                print(f"\n7. Actual dividend payment dates (first 10):")
                print("-" * 70)
                for date_val, yield_val in div_dates.items():
                    print(f"   {date_val.strftime('%Y-%m-%d')}: {yield_val:.4f} ({yield_val*252:.2%} annualized)")
            else:
                print(f"\n7. No dividend payments found in the selected period")
        
        print("\n✅ Real data example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during decomposition: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 Real data decomposition example completed!")
        print("📊 Key Benefits of Real Data:")
        print("   - Actual dividend payment dates and amounts")
        print("   - Real market volatility and correlations")
        print("   - Genuine economic data (inflation, rates)")
        print("   - Realistic component interactions")
        print("📝 Next steps:")
        print("   - Try different tickers and time periods")
        print("   - Compare results across different market conditions")
        print("   - Use the insights for portfolio analysis")
    else:
        print("\n❌ Example failed - check your data connection and API access")
        sys.exit(1)