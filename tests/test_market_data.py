"""
Test basic functionality of the market data fetcher.
"""

import pytest
from datetime import datetime, timedelta

from src.data.market_data import MarketDataFetcher, calculate_returns


def test_market_data_fetcher_initialization():
    """Test that MarketDataFetcher can be initialized."""
    fetcher = MarketDataFetcher()
    assert fetcher is not None
    assert fetcher.cache_dir is None


def test_fetch_current_prices():
    """Test fetching current prices for common tickers."""
    fetcher = MarketDataFetcher()
    
    # Test with common, liquid tickers
    tickers = ["SPY", "AAPL"]
    prices = fetcher.fetch_current_prices(tickers)
    
    assert isinstance(prices, dict)
    assert len(prices) > 0
    
    for ticker in prices:
        assert isinstance(prices[ticker], float)
        assert prices[ticker] > 0


def test_fetch_price_history():
    """Test fetching historical price data."""
    fetcher = MarketDataFetcher()
    
    # Fetch last 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    result = fetcher.fetch_price_history(
        "SPY",
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(result, dict)
    assert "SPY" in result
    
    spy_data = result["SPY"]
    assert not spy_data.empty
    assert all(col in spy_data.columns for col in ["Open", "High", "Low", "Close", "Volume"])
    assert len(spy_data) > 10  # Should have at least 10 trading days in 30 days


def test_calculate_returns():
    """Test return calculation functionality."""
    fetcher = MarketDataFetcher()
    
    # Fetch some data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    result = fetcher.fetch_price_history(
        "SPY",
        start_date=start_date,
        end_date=end_date
    )
    
    spy_data = result["SPY"]
    
    # Calculate daily returns
    daily_returns = calculate_returns(spy_data, period='daily')
    assert len(daily_returns) == len(spy_data)
    assert daily_returns.iloc[0] != daily_returns.iloc[0]  # First value should be NaN
    
    # Calculate total return
    total_return = calculate_returns(spy_data, period='total')
    assert isinstance(total_return, float)


if __name__ == "__main__":
    # Run basic tests
    print("Testing market data fetcher...")
    test_market_data_fetcher_initialization()
    print("✓ Initialization test passed")
    
    test_fetch_current_prices()
    print("✓ Current prices test passed")
    
    test_fetch_price_history()
    print("✓ Price history test passed")
    
    test_calculate_returns()
    print("✓ Returns calculation test passed")
    
    print("\nAll tests passed!")
