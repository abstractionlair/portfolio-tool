#!/usr/bin/env python3
"""
Enhanced Equity Return Decomposition Demo

This script demonstrates the enhanced equity return decomposition functionality
that separates nominal returns into economically meaningful components.

Key Features:
- Decomposes returns into dividend yield, P/E change, and real earnings excess
- Adjusts earnings growth for inflation and real risk-free rate
- Provides economic insight into return drivers
- Validates decomposition identity and quality
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.providers.coordinator import RawDataProviderCoordinator
from src.data.providers.transformed_provider import TransformedDataProvider
from src.data.interfaces import RawDataType
from unittest.mock import patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_data_providers():
    """Setup data providers for the demonstration."""
    coordinator = RawDataProviderCoordinator()
    provider = TransformedDataProvider(coordinator)
    
    return provider


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

def get_sample_earnings_data(ticker: str, start_date: date, end_date: date) -> pd.Series:
    """
    Create sample earnings data for demonstration.
    
    In a real implementation, this would come from a financial data provider.
    """
    # Create quarterly earnings with realistic growth
    if ticker == 'AAPL':
        # Apple-like earnings progression
        quarterly_dates = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq='QE'
        )
        
        # Sample EPS progression with growth
        base_eps = 1.50
        growth_rate = 0.08  # 8% quarterly growth
        
        eps_values = []
        for i, date in enumerate(quarterly_dates):
            eps = base_eps * (1 + growth_rate) ** i
            eps_values.append(eps)
        
        quarterly_earnings = pd.Series(eps_values, index=quarterly_dates, name='EPS')
        
    elif ticker == 'SPY':
        # S&P 500 earnings
        quarterly_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq='QE'
        )
        
        # S&P 500 approximate EPS
        base_eps = 45.0  # Approximate S&P 500 EPS
        growth_rate = 0.06  # 6% quarterly growth
        
        eps_values = []
        for i, date in enumerate(quarterly_dates):
            eps = base_eps * (1 + growth_rate) ** i
            eps_values.append(eps)
        
        quarterly_earnings = pd.Series(eps_values, index=quarterly_dates, name='EPS')
        
    else:
        # Generic earnings
        quarterly_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq='QE'
        )
        
        base_eps = 2.0
        growth_rate = 0.05  # 5% quarterly growth
        
        eps_values = []
        for i, date in enumerate(quarterly_dates):
            eps = base_eps * (1 + growth_rate) ** i
            eps_values.append(eps)
        
        quarterly_earnings = pd.Series(eps_values, index=quarterly_dates, name='EPS')
    
    # Create daily series by forward-filling
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_earnings = quarterly_earnings.reindex(full_date_range).ffill()
    
    return daily_earnings

def analyze_decomposition(result: dict, ticker: str) -> None:
    """Analyze and display decomposition results."""
    
    print(f"\n{ticker} Return Decomposition Analysis")
    print("=" * 50)
    
    # Calculate annualized statistics
    trading_days = 252
    
    # Get valid data (drop NaN values)
    valid_data = {}
    for key, series in result.items():
        valid_data[key] = series.dropna()
    
    if len(valid_data['nominal_return']) == 0:
        print("No valid data available for analysis")
        return
    
    # Annualized returns
    print("\nAnnualized Return Components:")
    print("-" * 30)
    
    components = [
        ('Total Nominal Return', 'nominal_return'),
        ('  - Dividend Yield', 'dividend_yield'),
        ('  - P/E Change', 'pe_change'),
        ('  - Nominal Earnings Growth', 'nominal_earnings_growth'),
    ]
    
    for name, key in components:
        if key in valid_data and len(valid_data[key]) > 0:
            annualized = valid_data[key].mean() * trading_days
            print(f"{name:30}: {annualized:6.2%}")
    
    print("\nReal Risk Premium Analysis:")
    print("-" * 30)
    
    real_components = [
        ('Real Risk Premium', 'real_risk_premium'),
        ('  - Dividend Yield', 'dividend_yield'),
        ('  - P/E Change', 'pe_change'),
        ('  - Real Earnings Excess', 'real_earnings_excess'),
    ]
    
    for name, key in real_components:
        if key in valid_data and len(valid_data[key]) > 0:
            annualized = valid_data[key].mean() * trading_days
            print(f"{name:30}: {annualized:6.2%}")
    
    print("\nEconomic Context:")
    print("-" * 30)
    
    economic_data = [
        ('Inflation Rate', 'inflation'),
        ('Nominal Risk-Free Rate', 'nominal_rf'),
        ('Real Risk-Free Rate', 'real_rf'),
    ]
    
    for name, key in economic_data:
        if key in valid_data and len(valid_data[key]) > 0:
            annualized = valid_data[key].mean() * trading_days
            print(f"{name:30}: {annualized:6.2%}")
    
    # Quality metrics
    print("\nDecomposition Quality:")
    print("-" * 30)
    
    quality_metrics = [
        ('Identity Error (max)', 'identity_error'),
        ('Decomposition Error (max)', 'decomp_error'),
    ]
    
    for name, key in quality_metrics:
        if key in valid_data and len(valid_data[key]) > 0:
            max_error = valid_data[key].max()
            print(f"{name:30}: {max_error:8.4f}")
    
    # Volatility analysis
    print("\nVolatility Analysis (Annualized):")
    print("-" * 30)
    
    vol_components = [
        ('Total Return Volatility', 'nominal_return'),
        ('Dividend Yield Volatility', 'dividend_yield'),
        ('P/E Change Volatility', 'pe_change'),
        ('Earnings Growth Volatility', 'nominal_earnings_growth'),
    ]
    
    for name, key in vol_components:
        if key in valid_data and len(valid_data[key]) > 0:
            annualized_vol = valid_data[key].std() * np.sqrt(trading_days)
            print(f"{name:30}: {annualized_vol:6.2%}")

def run_demonstration():
    """Run the enhanced equity return decomposition demonstration."""
    
    print("Enhanced Equity Return Decomposition Demo")
    print("=" * 50)
    
    # Setup
    provider = setup_data_providers()
    
    # Analysis period
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    
    # Test with different assets
    test_assets = ['AAPL', 'SPY', 'MSFT']
    
    for ticker in test_assets:
        try:
            print(f"\nAnalyzing {ticker}...")
            
            # Get sample earnings data
            earnings_data = get_sample_earnings_data(ticker, start_date, end_date)
            
            # Create realistic economic data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            inflation_data, rf_data, real_rf_data = create_realistic_economic_data(date_range)
            
            # Mock the economic data methods to return realistic values
            def mock_get_economic_data(start, end, frequency, inflation_measure="CPI", rf_tenor="3M"):
                return {
                    'inflation': inflation_data,
                    'nominal_rf': rf_data,
                    'real_rf': real_rf_data
                }
            
            # Perform decomposition with realistic economic data
            with patch.object(provider, '_get_economic_data_for_decomposition', side_effect=mock_get_economic_data):
                result = provider.decompose_equity_returns(
                    ticker=ticker,
                    start=start_date,
                    end=end_date,
                    earnings_data=earnings_data,
                    frequency='daily'
                )
            
            # Analyze results
            analyze_decomposition(result, ticker)
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
    
    # Demonstrate monthly frequency
    print("\n" + "=" * 70)
    print("Monthly Frequency Analysis")
    print("=" * 70)
    
    try:
        ticker = 'AAPL'
        earnings_data = get_sample_earnings_data(ticker, start_date, end_date)
        
        # Create realistic economic data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        inflation_data, rf_data, real_rf_data = create_realistic_economic_data(date_range)
        
        # Mock the economic data methods to return realistic values
        def mock_get_economic_data(start, end, frequency, inflation_measure="CPI", rf_tenor="3M"):
            return {
                'inflation': inflation_data,
                'nominal_rf': rf_data,
                'real_rf': real_rf_data
            }
        
        # Perform decomposition with realistic economic data
        with patch.object(provider, '_get_economic_data_for_decomposition', side_effect=mock_get_economic_data):
            result = provider.decompose_equity_returns(
                ticker=ticker,
                start=start_date,
                end=end_date,
                earnings_data=earnings_data,
                frequency='monthly'
            )
        
        analyze_decomposition(result, f"{ticker} (Monthly)")
        
    except Exception as e:
        print(f"Error with monthly analysis: {e}")
        logger.error(f"Error with monthly analysis: {e}", exc_info=True)

def main():
    """Main function."""
    try:
        run_demonstration()
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()