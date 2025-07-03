"""
Market data fetcher module.

Provides unified interface for fetching market data from various sources.
Primary source is Yahoo Finance via yfinance, with fallbacks to other APIs.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches market data from various sources."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the market data fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded data. If None, caching is disabled.
        """
        self.cache_dir = cache_dir
        # TODO: Implement caching mechanism
        
    def fetch_price_history(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> Dict[str, DataFrame]:
        """
        Fetch historical price data for given tickers.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date for historical data (default: 5 years ago)
            end_date: End date for historical data (default: today)
            interval: Data interval - 1d, 1wk, 1mo, etc.
            
        Returns:
            Dictionary mapping ticker to DataFrame with columns:
            Open, High, Low, Close, Volume, Adj Close
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=5*365)
        if end_date is None:
            end_date = datetime.now()
            
        results = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Fetching data for {ticker}")
                ticker_data = yf.Ticker(ticker)
                hist = ticker_data.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False
                )
                
                if hist.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue
                    
                results[ticker] = hist
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
                
        return results
    
    def fetch_current_prices(self, tickers: Union[str, List[str]]) -> Dict[str, float]:
        """
        Fetch current market prices for given tickers.
        
        Args:
            tickers: Single ticker or list of tickers
            
        Returns:
            Dictionary mapping ticker to current price
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        results = {}
        
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                # Try different price fields in order of preference
                price = None
                for field in ['regularMarketPrice', 'currentPrice', 'price']:
                    if field in info and info[field] is not None:
                        price = info[field]
                        break
                
                if price is None:
                    # Fallback to last close from history
                    hist = ticker_obj.history(period="1d")
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]
                
                if price is not None:
                    results[ticker] = float(price)
                else:
                    logger.warning(f"Could not fetch price for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching price for {ticker}: {str(e)}")
                continue
                
        return results
    
    def fetch_ticker_info(self, ticker: str) -> Dict:
        """
        Fetch detailed information about a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker information including name, sector, market cap, etc.
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Extract relevant fields
            relevant_fields = {
                'symbol': ticker,
                'shortName': info.get('shortName', ''),
                'longName': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'marketCap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'quoteType': info.get('quoteType', ''),
                'dividendYield': info.get('dividendYield', 0),
                'trailingPE': info.get('trailingPE', None),
                'beta': info.get('beta', None),
            }
            
            return relevant_fields
            
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {str(e)}")
            return {'symbol': ticker, 'error': str(e)}


def calculate_returns(price_data: DataFrame, period: str = 'daily') -> Series:
    """
    Calculate returns from price data.
    
    Args:
        price_data: DataFrame with price data (expects 'Adj Close' column)
        period: 'daily', 'weekly', 'monthly', or 'total'
        
    Returns:
        Series of returns
    """
    if 'Adj Close' in price_data.columns:
        prices = price_data['Adj Close']
    elif 'Close' in price_data.columns:
        prices = price_data['Close']
    else:
        raise ValueError("Price data must have 'Close' or 'Adj Close' column")
    
    if period == 'daily':
        returns = prices.pct_change()
    elif period == 'weekly':
        returns = prices.resample('W').last().pct_change()
    elif period == 'monthly':
        returns = prices.resample('M').last().pct_change()
    elif period == 'total':
        returns = (prices.iloc[-1] / prices.iloc[0]) - 1
    else:
        raise ValueError(f"Unknown period: {period}")
        
    return returns
