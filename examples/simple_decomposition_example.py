#!/usr/bin/env python3
"""
Simple Equity Return Decomposition Example

This script provides a simple, standalone example of how to use the
enhanced equity return decomposition functionality.
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
from unittest.mock import patch


def create_sample_earnings(start_date, end_date, base_eps=2.0, growth_rate=0.08):
    """Create sample earnings data for demonstration."""
    # Create quarterly earnings dates
    quarters = pd.date_range(start=start_date, end=end_date, freq='QE')
    
    # Generate earnings with growth
    eps_values = []
    for i, quarter in enumerate(quarters):
        eps = base_eps * (1 + growth_rate) ** (i * 0.25)
        eps_values.append(eps)
    
    # Create quarterly series
    quarterly_earnings = pd.Series(eps_values, index=quarters, name='EPS')
    
    # Convert to daily
    daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_earnings = quarterly_earnings.reindex(daily_range).ffill()
    daily_earnings = daily_earnings.fillna(base_eps)
    
    return daily_earnings


def create_realistic_economic_data(dates, annual_inflation=0.03, annual_rf=0.05):
    """Create realistic economic data for demonstration."""
    # Convert annual rates to daily rates
    daily_inflation = annual_inflation / 252
    daily_rf = annual_rf / 252
    daily_real_rf = daily_rf - daily_inflation
    
    # Create series with small random variations
    np.random.seed(42)  # For reproducibility
    
    inflation_series = pd.Series(
        daily_inflation + np.random.normal(0, daily_inflation * 0.1, len(dates)),
        index=dates,
        name='inflation'
    )
    
    rf_series = pd.Series(
        daily_rf + np.random.normal(0, daily_rf * 0.05, len(dates)),
        index=dates,
        name='nominal_rf'
    )
    
    real_rf_series = pd.Series(
        daily_real_rf + np.random.normal(0, daily_real_rf * 0.1, len(dates)),
        index=dates,
        name='real_rf'
    )
    
    return inflation_series, rf_series, real_rf_series


def print_decomposition_summary(result, ticker):
    """Print a summary of decomposition results."""
    print(f"\n📊 {ticker} Return Decomposition Summary")
    print("=" * 50)
    
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
    print("-" * 30)
    
    for component, name in components:
        if component in result:
            series = result[component].dropna()
            if len(series) > 0:
                annual_mean = series.mean() * 252
                annual_std = series.std() * np.sqrt(252)
                print(f"{name:25}: {annual_mean:6.2%} ± {annual_std:5.2%}")
    
    # Economic context
    print("\nEconomic Context:")
    print("-" * 30)
    
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
                print(f"{name:25}: {annual_mean:6.2%}")
    
    # Quality metrics
    print("\nQuality Metrics:")
    print("-" * 30)
    
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
    """Main example function."""
    print("Enhanced Equity Return Decomposition - Simple Example")
    print("=" * 60)
    
    # Setup data provider
    print("\n1. Setting up data provider...")
    coordinator = RawDataProviderCoordinator()
    provider = TransformedDataProvider(coordinator)
    
    # Analysis parameters
    ticker = 'AAPL'
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    
    print(f"2. Analyzing {ticker} from {start_date} to {end_date}")
    
    # Create sample earnings data
    print("3. Creating sample earnings data...")
    earnings_data = create_sample_earnings(
        start_date, end_date, 
        base_eps=1.50,  # Apple-like starting EPS
        growth_rate=0.12  # 12% annual growth
    )
    
    print(f"   Created earnings data with {len(earnings_data)} daily observations")
    print(f"   EPS range: ${earnings_data.min():.2f} - ${earnings_data.max():.2f}")
    
    # Create realistic economic data
    print("4. Creating realistic economic data...")
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    inflation_data, rf_data, real_rf_data = create_realistic_economic_data(date_range)
    
    print(f"   Annual inflation: {inflation_data.mean() * 252:.2%}")
    print(f"   Annual risk-free rate: {rf_data.mean() * 252:.2%}")
    print(f"   Annual real risk-free rate: {real_rf_data.mean() * 252:.2%}")
    
    # Mock the economic data methods to return realistic values
    def mock_get_economic_data(start, end, frequency, inflation_measure="CPI", rf_tenor="3M"):
        return {
            'inflation': inflation_data,
            'nominal_rf': rf_data,
            'real_rf': real_rf_data
        }
    
    # Perform decomposition
    print("5. Performing return decomposition...")
    try:
        with patch.object(provider, '_get_economic_data_for_decomposition', side_effect=mock_get_economic_data):
            result = provider.decompose_equity_returns(
                ticker=ticker,
                start=start_date,
                end=end_date,
                earnings_data=earnings_data,
                frequency='daily'
            )
        
        print(f"   Decomposition complete! Generated {len(result)} components")
        
        # Display results
        print_decomposition_summary(result, ticker)
        
        # Show sample data
        print(f"\n6. Sample of decomposition data (first 5 days):")
        print("-" * 60)
        
        sample_components = ['nominal_return', 'dividend_yield', 'pe_change', 'nominal_earnings_growth']
        sample_data = {}
        
        for component in sample_components:
            if component in result:
                series = result[component].dropna()
                if len(series) > 0:
                    sample_data[component] = series.head(5)
        
        if sample_data:
            sample_df = pd.DataFrame(sample_data)
            print(sample_df)
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during decomposition: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 Simple decomposition example completed!")
        print("📝 Next steps:")
        print("   - Try different tickers")
        print("   - Experiment with different time periods")
        print("   - Use the Jupyter notebook for visualization")
        print("   - Explore the advanced examples")
    else:
        print("\n❌ Example failed - check your data connection")
        sys.exit(1)