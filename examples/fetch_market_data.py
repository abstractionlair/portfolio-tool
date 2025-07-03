#!/usr/bin/env python3
"""
Example usage of the portfolio optimizer market data functionality.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timedelta
from pprint import pprint

from src.data.market_data import MarketDataFetcher, calculate_returns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Demonstrate basic market data functionality."""
    print("Portfolio Optimizer - Market Data Demo\n")
    
    # Initialize the fetcher
    fetcher = MarketDataFetcher()
    
    # Example 1: Fetch current prices
    print("1. Fetching current prices for popular ETFs:")
    print("-" * 50)
    
    etfs = ["SPY", "QQQ", "TLT", "GLD", "VTI"]
    current_prices = fetcher.fetch_current_prices(etfs)
    
    for ticker, price in current_prices.items():
        print(f"{ticker}: ${price:.2f}")
    
    # Example 2: Fetch historical data
    print("\n2. Fetching 3 months of historical data for SPY:")
    print("-" * 50)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    historical_data = fetcher.fetch_price_history(
        "SPY",
        start_date=start_date,
        end_date=end_date
    )
    
    spy_data = historical_data["SPY"]
    print(f"Data shape: {spy_data.shape}")
    print(f"Date range: {spy_data.index[0].date()} to {spy_data.index[-1].date()}")
    print(f"\nLast 5 days:")
    print(spy_data.tail())
    
    # Example 3: Calculate returns
    print("\n3. Calculating returns:")
    print("-" * 50)
    
    daily_returns = calculate_returns(spy_data, period='daily')
    total_return = calculate_returns(spy_data, period='total')
    
    print(f"Total return over period: {total_return:.2%}")
    print(f"Average daily return: {daily_returns.mean():.4%}")
    print(f"Daily volatility: {daily_returns.std():.4%}")
    print(f"Annualized volatility: {daily_returns.std() * (252**0.5):.2%}")
    
    # Example 4: Fetch ticker information
    print("\n4. Fetching detailed ticker information:")
    print("-" * 50)
    
    ticker_info = fetcher.fetch_ticker_info("SPY")
    
    print("SPY Information:")
    for key, value in ticker_info.items():
        if value and key != 'symbol':
            print(f"  {key}: {value}")
    
    # Example 5: Multi-asset comparison
    print("\n5. Multi-asset performance comparison:")
    print("-" * 50)
    
    assets = ["SPY", "TLT", "GLD"]  # Stocks, Bonds, Gold
    asset_data = fetcher.fetch_price_history(
        assets,
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now()
    )
    
    print("1-Year Returns:")
    for ticker, data in asset_data.items():
        if not data.empty:
            total_ret = calculate_returns(data, period='total')
            vol = calculate_returns(data, period='daily').std() * (252**0.5)
            print(f"  {ticker}: Return={total_ret:.2%}, Volatility={vol:.2%}")


if __name__ == "__main__":
    main()
