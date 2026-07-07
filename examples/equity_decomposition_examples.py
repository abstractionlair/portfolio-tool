#!/usr/bin/env python3
"""
Equity Return Decomposition Examples

This module provides focused examples of equity return decomposition
for different scenarios and use cases.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.providers.coordinator import RawDataProviderCoordinator
from src.data.providers.transformed_provider import TransformedDataProvider
from src.data.interfaces import DataNotAvailableError

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def setup_provider():
    """Setup the data provider."""
    coordinator = RawDataProviderCoordinator()
    return TransformedDataProvider(coordinator)


def create_realistic_economic_data(
    dates: pd.DatetimeIndex,
    annual_inflation: float = 0.03,
    annual_rf: float = 0.05
) -> tuple:
    """Create realistic economic data for the given dates."""
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


def create_realistic_earnings_data(
    ticker: str, 
    start_date: date, 
    end_date: date,
    base_eps: float = 2.0,
    growth_rate: float = 0.08
) -> pd.Series:
    """
    Create realistic earnings data with quarterly reporting.
    
    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        base_eps: Base earnings per share
        growth_rate: Annual growth rate
        
    Returns:
        Daily earnings series with quarterly updates
    """
    # Create quarterly earnings dates
    quarterly_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Find next quarter end
        if current_date.month <= 3:
            quarter_end = date(current_date.year, 3, 31)
        elif current_date.month <= 6:
            quarter_end = date(current_date.year, 6, 30)
        elif current_date.month <= 9:
            quarter_end = date(current_date.year, 9, 30)
        else:
            quarter_end = date(current_date.year, 12, 31)
        
        if quarter_end <= end_date:
            quarterly_dates.append(quarter_end)
        
        # Move to next quarter
        if current_date.month <= 3:
            current_date = date(current_date.year, 4, 1)
        elif current_date.month <= 6:
            current_date = date(current_date.year, 7, 1)
        elif current_date.month <= 9:
            current_date = date(current_date.year, 10, 1)
        else:
            current_date = date(current_date.year + 1, 1, 1)
    
    # Create EPS progression
    eps_values = []
    for i, quarter_date in enumerate(quarterly_dates):
        # Calculate quarterly growth
        quarterly_growth = (1 + growth_rate) ** (i * 0.25)
        eps = base_eps * quarterly_growth
        eps_values.append(eps)
    
    # Create quarterly series
    quarterly_earnings = pd.Series(eps_values, index=quarterly_dates, name='EPS')
    
    # Convert to daily by forward filling
    daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_earnings = quarterly_earnings.reindex(daily_range).ffill()
    
    # Fill any initial NaN values
    daily_earnings = daily_earnings.fillna(base_eps)
    
    return daily_earnings


def analyze_single_stock(
    provider: TransformedDataProvider,
    ticker: str,
    start_date: date,
    end_date: date,
    frequency: str = 'daily'
) -> Dict:
    """
    Analyze a single stock's return decomposition.
    
    Args:
        provider: Data provider
        ticker: Stock ticker
        start_date: Analysis start date
        end_date: Analysis end date
        frequency: Data frequency
        
    Returns:
        Dictionary with decomposition results and analysis
    """
    from unittest.mock import patch
    
    # Stock-specific earnings parameters
    earnings_params = {
        'AAPL': {'base_eps': 1.50, 'growth_rate': 0.12},
        'MSFT': {'base_eps': 2.80, 'growth_rate': 0.10},
        'GOOGL': {'base_eps': 4.50, 'growth_rate': 0.15},
        'AMZN': {'base_eps': 1.20, 'growth_rate': 0.18},
        'SPY': {'base_eps': 45.0, 'growth_rate': 0.08},
        'QQQ': {'base_eps': 38.0, 'growth_rate': 0.12},
    }
    
    params = earnings_params.get(ticker, {'base_eps': 2.0, 'growth_rate': 0.08})
    
    # Create earnings data
    earnings_data = create_realistic_earnings_data(
        ticker, start_date, end_date, 
        base_eps=params['base_eps'],
        growth_rate=params['growth_rate']
    )
    
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
    
    # Patch the provider's economic data method
    with patch.object(provider, '_get_economic_data_for_decomposition', side_effect=mock_get_economic_data):
        # Perform decomposition
        result = provider.decompose_equity_returns(
            ticker=ticker,
            start=start_date,
            end=end_date,
            earnings_data=earnings_data,
            frequency=frequency
        )
    
    # Calculate summary statistics
    trading_days = 252 if frequency == 'daily' else 12
    
    summary = {}
    for key, series in result.items():
        valid_data = series.dropna()
        if len(valid_data) > 0:
            summary[key] = {
                'mean_annualized': valid_data.mean() * trading_days,
                'std_annualized': valid_data.std() * np.sqrt(trading_days),
                'min': valid_data.min(),
                'max': valid_data.max(),
                'count': len(valid_data)
            }
    
    return {
        'ticker': ticker,
        'period': f"{start_date} to {end_date}",
        'frequency': frequency,
        'raw_data': result,
        'summary': summary,
        'earnings_params': params
    }


def compare_stocks_decomposition(
    provider: TransformedDataProvider,
    tickers: list,
    start_date: date,
    end_date: date,
    frequency: str = 'daily'
) -> Dict:
    """
    Compare return decomposition across multiple stocks.
    
    Args:
        provider: Data provider
        tickers: List of stock tickers
        start_date: Analysis start date
        end_date: Analysis end date
        frequency: Data frequency
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for ticker in tickers:
        try:
            results[ticker] = analyze_single_stock(
                provider, ticker, start_date, end_date, frequency
            )
        except Exception as e:
            logger.warning(f"Failed to analyze {ticker}: {e}")
            results[ticker] = {'error': str(e)}
    
    return results


def analyze_time_series_properties(
    provider: TransformedDataProvider,
    ticker: str,
    start_date: date,
    end_date: date
) -> Dict:
    """
    Analyze time series properties of return components.
    
    Args:
        provider: Data provider
        ticker: Stock ticker
        start_date: Analysis start date
        end_date: Analysis end date
        
    Returns:
        Dictionary with time series analysis
    """
    # Get decomposition
    earnings_data = create_realistic_earnings_data(ticker, start_date, end_date)
    
    result = provider.decompose_equity_returns(
        ticker=ticker,
        start=start_date,
        end=end_date,
        earnings_data=earnings_data,
        frequency='daily'
    )
    
    # Analyze each component
    analysis = {}
    
    key_components = [
        'nominal_return', 'dividend_yield', 'pe_change', 
        'nominal_earnings_growth', 'real_earnings_excess'
    ]
    
    for component in key_components:
        if component in result:
            series = result[component].dropna()
            if len(series) > 10:
                analysis[component] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis(),
                    'autocorr_1': series.autocorr(1) if len(series) > 1 else np.nan,
                    'autocorr_5': series.autocorr(5) if len(series) > 5 else np.nan,
                    'sharpe_ratio': series.mean() / series.std() if series.std() > 0 else np.nan,
                    'max_drawdown': (series.cumsum() - series.cumsum().expanding().max()).min(),
                    'volatility_clustering': series.rolling(20).std().std() if len(series) > 20 else np.nan
                }
    
    return {
        'ticker': ticker,
        'period': f"{start_date} to {end_date}",
        'component_analysis': analysis,
        'correlation_matrix': pd.DataFrame({
            k: v for k, v in result.items() 
            if k in key_components and k in result
        }).corr().to_dict()
    }


def sector_comparison_example(
    provider: TransformedDataProvider,
    start_date: date,
    end_date: date
) -> Dict:
    """
    Compare return decomposition across different sectors.
    
    Args:
        provider: Data provider
        start_date: Analysis start date
        end_date: Analysis end date
        
    Returns:
        Dictionary with sector comparison
    """
    sector_stocks = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL'],
        'Consumer Discretionary': ['AMZN', 'TSLA'],
        'Financial': ['JPM', 'BAC'],
        'Healthcare': ['JNJ', 'PFE'],
        'Broad Market': ['SPY', 'QQQ']
    }
    
    sector_results = {}
    
    for sector, stocks in sector_stocks.items():
        try:
            stock_results = compare_stocks_decomposition(
                provider, stocks[:2], start_date, end_date  # Limit to 2 stocks per sector
            )
            
            # Calculate sector averages
            sector_summary = {}
            component_keys = [
                'nominal_return', 'dividend_yield', 'pe_change', 
                'nominal_earnings_growth', 'real_earnings_excess'
            ]
            
            for component in component_keys:
                values = []
                for stock, data in stock_results.items():
                    if 'summary' in data and component in data['summary']:
                        values.append(data['summary'][component]['mean_annualized'])
                
                if values:
                    sector_summary[component] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            sector_results[sector] = {
                'stocks': stock_results,
                'sector_summary': sector_summary
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze sector {sector}: {e}")
            sector_results[sector] = {'error': str(e)}
    
    return sector_results


def monthly_vs_daily_comparison(
    provider: TransformedDataProvider,
    ticker: str,
    start_date: date,
    end_date: date
) -> Dict:
    """
    Compare decomposition results between monthly and daily frequencies.
    
    Args:
        provider: Data provider
        ticker: Stock ticker
        start_date: Analysis start date
        end_date: Analysis end date
        
    Returns:
        Dictionary with frequency comparison
    """
    results = {}
    
    for frequency in ['daily', 'monthly']:
        try:
            results[frequency] = analyze_single_stock(
                provider, ticker, start_date, end_date, frequency
            )
        except Exception as e:
            logger.warning(f"Failed to analyze {ticker} at {frequency} frequency: {e}")
            results[frequency] = {'error': str(e)}
    
    # Create comparison summary
    comparison = {}
    if 'daily' in results and 'monthly' in results:
        if 'summary' in results['daily'] and 'summary' in results['monthly']:
            for component in ['nominal_return', 'dividend_yield', 'pe_change']:
                if (component in results['daily']['summary'] and 
                    component in results['monthly']['summary']):
                    
                    daily_mean = results['daily']['summary'][component]['mean_annualized']
                    monthly_mean = results['monthly']['summary'][component]['mean_annualized']
                    
                    comparison[component] = {
                        'daily_mean': daily_mean,
                        'monthly_mean': monthly_mean,
                        'difference': daily_mean - monthly_mean,
                        'relative_difference': (daily_mean - monthly_mean) / abs(monthly_mean) if monthly_mean != 0 else np.nan
                    }
    
    return {
        'ticker': ticker,
        'period': f"{start_date} to {end_date}",
        'frequency_results': results,
        'comparison': comparison
    }


# Example usage functions that can be called from notebooks
def run_basic_example():
    """Run a basic single stock example."""
    provider = setup_provider()
    
    return analyze_single_stock(
        provider=provider,
        ticker='AAPL',
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        frequency='daily'
    )


def run_stock_comparison():
    """Run a stock comparison example."""
    provider = setup_provider()
    
    return compare_stocks_decomposition(
        provider=provider,
        tickers=['AAPL', 'MSFT', 'SPY'],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        frequency='daily'
    )


def run_time_series_analysis():
    """Run time series properties analysis."""
    provider = setup_provider()
    
    return analyze_time_series_properties(
        provider=provider,
        ticker='AAPL',
        start_date=date(2022, 1, 1),
        end_date=date(2023, 12, 31)
    )


def run_sector_comparison():
    """Run sector comparison analysis."""
    provider = setup_provider()
    
    return sector_comparison_example(
        provider=provider,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31)
    )


def run_frequency_comparison():
    """Run frequency comparison analysis."""
    provider = setup_provider()
    
    return monthly_vs_daily_comparison(
        provider=provider,
        ticker='AAPL',
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31)
    )


if __name__ == "__main__":
    print("Equity Return Decomposition Examples")
    print("=" * 50)
    
    # Run basic example
    print("\n1. Basic Single Stock Analysis:")
    basic_result = run_basic_example()
    print(f"Analyzed {basic_result['ticker']} for {basic_result['period']}")
    
    # Run comparison
    print("\n2. Stock Comparison:")
    comparison_result = run_stock_comparison()
    print(f"Compared {len(comparison_result)} stocks")
    
    print("\nExamples completed successfully!")