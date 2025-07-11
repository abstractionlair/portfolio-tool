"""
Yahoo Finance data provider implementation.

This module provides market data for securities using the yfinance library,
implementing the RawDataProvider interface.
"""

import logging
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd
import yfinance as yf

from ..interfaces import (
    RawDataProvider, RawDataType, Frequency,
    validate_ticker_requirement, validate_date_range,
    DataNotAvailableError, InvalidTickerError
)
from typing import List

logger = logging.getLogger(__name__)


class YFinanceProvider(RawDataProvider):
    """Yahoo Finance implementation of RawDataProvider for securities data."""
    
    def __init__(self):
        """Initialize Yahoo Finance provider."""
        self.session = None  # Can be used for connection pooling
        
        # Security data types this provider supports
        self.supported_types = {
            RawDataType.OHLCV,
            RawDataType.ADJUSTED_CLOSE,
            RawDataType.VOLUME,
            RawDataType.DIVIDEND,
            RawDataType.SPLIT
        }
    
    def get_data(
        self,
        data_type: RawDataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        **kwargs
    ) -> pd.Series:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            data_type: Type of data to fetch
            start: Start date
            end: End date  
            ticker: Ticker symbol (required for securities data)
            frequency: Data frequency
            **kwargs: Additional arguments
            
        Returns:
            pandas Series with date index and data values
            
        Raises:
            InvalidTickerError: If ticker is required but not provided
            DataNotAvailableError: If data cannot be fetched
        """
        # Validate inputs
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # Raw providers should only accept RawDataType
        if not isinstance(data_type, RawDataType):
            raise ValueError(
                f"RawDataProvider only accepts RawDataType, got {type(data_type).__name__}"
            )
        
        # Check if we support this data type
        if data_type not in self.supported_types:
            raise DataNotAvailableError(
                f"YFinance provider does not support {data_type.value}"
            )
        
        # Convert frequency to yfinance interval
        interval = self._get_yfinance_interval(frequency)
        
        try:
            return self._fetch_yfinance_data(ticker, data_type, start, end, interval)
            
        except Exception as e:
            logger.error(f"Failed to fetch {data_type.value} for {ticker}: {e}")
            raise DataNotAvailableError(
                f"Could not fetch {data_type.value} for {ticker}: {e}"
            )
    
    def is_available(
        self,
        data_type: RawDataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Check if data is available from Yahoo Finance.
        
        Args:
            data_type: Type of data to check
            start: Start date
            end: End date
            ticker: Ticker symbol
            **kwargs: Additional arguments
            
        Returns:
            True if data should be available
        """
        try:
            validate_ticker_requirement(data_type, ticker)
            validate_date_range(start, end)
            return data_type in self.supported_types
        except Exception:
            # Catch all validation errors and return False
            return False
    
    def get_universe_data(
        self,
        data_type: RawDataType,
        tickers: List[str],
        start: date,
        end: date,
        frequency: Union[str, Frequency] = "daily",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data for multiple tickers.
        
        Args:
            data_type: Type of data to fetch
            tickers: List of ticker symbols
            start: Start date
            end: End date
            frequency: Data frequency
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with tickers as columns and dates as index
        """
        # Raw providers should only accept RawDataType
        if not isinstance(data_type, RawDataType):
            raise ValueError(
                f"RawDataProvider only accepts RawDataType, got {type(data_type).__name__}"
            )
        
        if data_type not in self.supported_types:
            raise DataNotAvailableError(
                f"YFinance provider does not support {data_type.value}"
            )
        
        # For optimal performance with yfinance, we can download multiple tickers at once
        # Convert frequency to yfinance interval
        interval = self._get_yfinance_interval(frequency)
        
        try:
            # Download all tickers at once
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                prepost=False,
                group_by='ticker'
            )
            
            if data.empty:
                raise DataNotAvailableError("No data returned for any ticker")
            
            # Extract the specific data type for each ticker
            result_data = {}
            
            for ticker in tickers:
                try:
                    # Extract data for this ticker
                    if len(tickers) == 1:
                        # Single ticker - data is not grouped
                        ticker_data = data
                    else:
                        # Multiple tickers - data is grouped by ticker
                        ticker_data = data[ticker] if ticker in data.columns.levels[0] else pd.DataFrame()
                    
                    if ticker_data.empty:
                        # Create empty series for missing data
                        result_data[ticker] = pd.Series(dtype=float, name=ticker)
                        continue
                    
                    # Extract the requested data type
                    if data_type == RawDataType.ADJUSTED_CLOSE:
                        if 'Adj Close' in ticker_data.columns:
                            series = ticker_data['Adj Close'].dropna()
                        else:
                            series = pd.Series(dtype=float, name=ticker)
                            
                    elif data_type == RawDataType.VOLUME:
                        if 'Volume' in ticker_data.columns:
                            series = ticker_data['Volume'].dropna()
                        else:
                            series = pd.Series(dtype=float, name=ticker)
                            
                    elif data_type == RawDataType.OHLCV:
                        if 'Close' in ticker_data.columns:
                            series = ticker_data['Close'].dropna()
                        else:
                            series = pd.Series(dtype=float, name=ticker)
                            
                    elif data_type in [RawDataType.DIVIDEND, RawDataType.SPLIT]:
                        # For dividends and splits, fall back to individual ticker calls
                        # since yf.download doesn't include this data
                        try:
                            series = self.get_data(data_type, start, end, ticker, frequency)
                        except DataNotAvailableError:
                            series = pd.Series(dtype=float, name=ticker)
                    else:
                        series = pd.Series(dtype=float, name=ticker)
                    
                    result_data[ticker] = series
                    
                except Exception as e:
                    logger.warning(f"Failed to get {data_type.value} for {ticker}: {e}")
                    result_data[ticker] = pd.Series(dtype=float, name=ticker)
            
            # Convert to DataFrame
            result_df = pd.DataFrame(result_data)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to fetch universe data for {data_type.value}: {e}")
            raise DataNotAvailableError(
                f"Could not fetch universe data for {data_type.value}: {e}"
            )
    
    def _get_yfinance_interval(self, frequency: Union[str, Frequency]) -> str:
        """Convert frequency to yfinance interval string."""
        if isinstance(frequency, Frequency):
            return frequency.yfinance_interval
        
        # Handle string frequencies
        mapping = {
            "daily": "1d",
            "weekly": "1wk", 
            "monthly": "1mo",
            "quarterly": "3mo",
            "annual": "1y"
        }
        return mapping.get(frequency, "1d")
    
    def _fetch_yfinance_data(
        self,
        ticker: str,
        data_type: RawDataType,
        start: date,
        end: date,
        interval: str
    ) -> pd.Series:
        """Fetch data from yfinance and extract the requested type."""
        
        # Create yfinance ticker object
        yf_ticker = yf.Ticker(ticker)
        
        # Fetch historical data
        hist = yf_ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,  # We want both Close and Adj Close
            prepost=False
        )
        
        if hist.empty:
            raise DataNotAvailableError(f"No data returned for {ticker}")
        
        # Extract the requested data type
        if data_type == RawDataType.ADJUSTED_CLOSE:
            if 'Adj Close' not in hist.columns:
                raise DataNotAvailableError(f"Adjusted close not available for {ticker}")
            series = hist['Adj Close']
            
        elif data_type == RawDataType.VOLUME:
            if 'Volume' not in hist.columns:
                raise DataNotAvailableError(f"Volume not available for {ticker}")
            series = hist['Volume']
            
        elif data_type == RawDataType.OHLCV:
            # For OHLCV, return the close price (most commonly used)
            # In a more complete implementation, this might return a MultiIndex series
            if 'Close' not in hist.columns:
                raise DataNotAvailableError(f"OHLCV data not available for {ticker}")
            series = hist['Close']
            
        elif data_type == RawDataType.DIVIDEND:
            # Get dividend data separately
            dividends = yf_ticker.dividends
            if dividends.empty:
                # Return empty series with correct date range
                return pd.Series(
                    dtype=float,
                    index=pd.date_range(start, end, freq='D'),
                    name=ticker
                )
            
            # Filter to date range and reindex to daily
            dividends = dividends[(dividends.index.date >= start) & 
                                 (dividends.index.date <= end)]
            
            # Reindex to daily frequency, filling with zeros
            daily_index = pd.date_range(start, end, freq='D')
            series = dividends.reindex(daily_index, fill_value=0.0)
            
        elif data_type == RawDataType.SPLIT:
            # Get stock split data
            splits = yf_ticker.splits
            if splits.empty:
                # Return series of 1.0 (no splits)
                return pd.Series(
                    1.0,
                    index=pd.date_range(start, end, freq='D'),
                    name=ticker
                )
            
            # Filter to date range
            splits = splits[(splits.index.date >= start) & 
                           (splits.index.date <= end)]
            
            # Reindex to daily frequency, filling with 1.0 (no split)
            daily_index = pd.date_range(start, end, freq='D')
            series = splits.reindex(daily_index, fill_value=1.0)
            
        else:
            raise DataNotAvailableError(f"Unsupported data type: {data_type.value}")
        
        # Ensure proper naming
        series.name = ticker
        
        # Remove any invalid data
        series = series.dropna()
        
        if series.empty:
            raise DataNotAvailableError(f"No valid data for {ticker} {data_type.value}")
        
        return series