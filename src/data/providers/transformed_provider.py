"""
Transformed Data Provider implementation.

This module provides both raw and computed data types by orchestrating
calculations on top of raw data providers.
"""

import logging
from datetime import date, timedelta
from typing import Optional, Union, List, Dict, Any
import pandas as pd

from ..interfaces import (
    DataProvider, RawDataProvider, RawDataType, LogicalDataType, DataType, Frequency,
    validate_ticker_requirement, validate_date_range,
    DataNotAvailableError, InvalidTickerError
)
from .calculators import ReturnCalculator, EconomicCalculator, FrequencyConverter

logger = logging.getLogger(__name__)


class TransformedDataProvider(DataProvider):
    """
    Provides both raw and computed data types.
    
    This provider acts as a computational layer on top of raw data providers,
    calculating derived metrics like returns, inflation rates, and risk-free rates.
    """
    
    def __init__(self, raw_provider: RawDataProvider):
        """
        Initialize the transformed provider.
        
        Args:
            raw_provider: Raw data provider for fetching base data
        """
        self.raw_provider = raw_provider
        self.return_calculator = ReturnCalculator()
        self.economic_calculator = EconomicCalculator()
        self.frequency_converter = FrequencyConverter()
        
        # Configuration for calculations
        self.config = {
            "inflation_method": "yoy",
            "risk_free_tenor": "3m",
            "use_adjusted_close": True,
            "dividend_reinvestment": True,
            "lookback_buffer_days": 400  # Extra days for calculations
        }
    
    def get_data(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        **kwargs
    ) -> pd.Series:
        """
        Fetch data, computing derived types as needed.
        
        Args:
            data_type: Type of data to fetch (raw or logical)
            start: Start date
            end: End date
            ticker: Ticker symbol (required for securities data)
            frequency: Data frequency
            **kwargs: Additional arguments
            
        Returns:
            pandas Series with date index and data values
            
        Raises:
            InvalidTickerError: If ticker requirements not met
            DataNotAvailableError: If data cannot be computed
        """
        # Validate inputs
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # Convert frequency to string if needed
        freq_str = frequency.value if hasattr(frequency, 'value') else frequency
        
        # Route based on data type
        if isinstance(data_type, RawDataType):
            # Pass through to raw provider
            return self.raw_provider.get_data(data_type, start, end, ticker, frequency, **kwargs)
            
        elif isinstance(data_type, LogicalDataType):
            # Compute the derived data
            return self._compute_logical_data(data_type, start, end, ticker, freq_str, **kwargs)
            
        else:
            raise ValueError(f"Unknown data type: {type(data_type).__name__}")
    
    def is_available(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Check if data is available (raw or can be computed).
        
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
            
            if isinstance(data_type, RawDataType):
                return self.raw_provider.is_available(data_type, start, end, ticker, **kwargs)
            elif isinstance(data_type, LogicalDataType):
                return self._can_compute_logical_data(data_type, start, end, ticker, **kwargs)
            else:
                return False
                
        except Exception:
            return False
    
    def get_universe_data(
        self,
        data_type: DataType,
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
        if isinstance(data_type, RawDataType):
            # Delegate to raw provider if it supports universe data
            if hasattr(self.raw_provider, 'get_universe_data'):
                return self.raw_provider.get_universe_data(data_type, tickers, start, end, frequency, **kwargs)
            else:
                # Fall back to individual calls
                return self._get_universe_data_individual(data_type, tickers, start, end, frequency, **kwargs)
                
        elif isinstance(data_type, LogicalDataType):
            # For logical data, always use individual calls
            return self._get_universe_data_individual(data_type, tickers, start, end, frequency, **kwargs)
        else:
            raise ValueError(f"Unknown data type: {type(data_type).__name__}")
    
    def _compute_logical_data(
        self,
        data_type: LogicalDataType,
        start: date,
        end: date,
        ticker: Optional[str],
        frequency: str,
        **kwargs
    ) -> pd.Series:
        """Compute logical data types from raw data."""
        
        if data_type == LogicalDataType.TOTAL_RETURN:
            return self._compute_total_returns(start, end, ticker, frequency, **kwargs)
            
        elif data_type == LogicalDataType.SIMPLE_RETURN:
            return self._compute_simple_returns(start, end, ticker, frequency, **kwargs)
            
        elif data_type == LogicalDataType.LOG_RETURN:
            return self._compute_log_returns(start, end, ticker, frequency, **kwargs)
            
        elif data_type == LogicalDataType.EXCESS_RETURN:
            return self._compute_excess_returns(start, end, ticker, frequency, **kwargs)
            
        elif data_type == LogicalDataType.INFLATION_RATE:
            return self._compute_inflation_rate(start, end, frequency, **kwargs)
            
        elif data_type == LogicalDataType.NOMINAL_RISK_FREE:
            return self._compute_nominal_risk_free(start, end, frequency, **kwargs)
            
        elif data_type == LogicalDataType.REAL_RISK_FREE:
            return self._compute_real_risk_free(start, end, frequency, **kwargs)
            
        else:
            raise DataNotAvailableError(f"Logical data type not implemented: {data_type.value}")
    
    def _compute_total_returns(
        self,
        start: date,
        end: date,
        ticker: str,
        frequency: str,
        **kwargs
    ) -> pd.Series:
        """Compute total returns including dividends."""
        # Extend date range to get previous price for return calculation
        extended_start = self._extend_start_date(start, frequency, periods=1)
        
        # Get prices (use adjusted close if configured)
        price_type = RawDataType.ADJUSTED_CLOSE if self.config["use_adjusted_close"] else RawDataType.OHLCV
        prices = self.raw_provider.get_data(price_type, extended_start, end, ticker, frequency)
        
        # Get dividends if dividend reinvestment is enabled
        dividends = None
        if self.config["dividend_reinvestment"]:
            try:
                dividends = self.raw_provider.get_data(RawDataType.DIVIDEND, extended_start, end, ticker, frequency)
            except DataNotAvailableError:
                logger.debug(f"No dividend data available for {ticker}, using price-only returns")
        
        # Calculate total returns
        total_returns = self.return_calculator.calculate_total_returns(prices, dividends)
        
        # Trim to requested date range and convert frequency if needed
        total_returns = self._trim_and_convert(total_returns, start, end, frequency, frequency, "return")
        
        return total_returns
    
    def _compute_simple_returns(
        self,
        start: date,
        end: date,
        ticker: str,
        frequency: str,
        **kwargs
    ) -> pd.Series:
        """Compute simple price returns."""
        extended_start = self._extend_start_date(start, frequency, periods=1)
        
        price_type = RawDataType.ADJUSTED_CLOSE if self.config["use_adjusted_close"] else RawDataType.OHLCV
        prices = self.raw_provider.get_data(price_type, extended_start, end, ticker, frequency)
        
        simple_returns = self.return_calculator.calculate_simple_returns(prices, frequency)
        
        return self._trim_and_convert(simple_returns, start, end, frequency, frequency, "return")
    
    def _compute_log_returns(
        self,
        start: date,
        end: date,
        ticker: str,
        frequency: str,
        **kwargs
    ) -> pd.Series:
        """Compute log returns."""
        extended_start = self._extend_start_date(start, frequency, periods=1)
        
        price_type = RawDataType.ADJUSTED_CLOSE if self.config["use_adjusted_close"] else RawDataType.OHLCV
        prices = self.raw_provider.get_data(price_type, extended_start, end, ticker, frequency)
        
        log_returns = self.return_calculator.calculate_log_returns(prices)
        
        return self._trim_and_convert(log_returns, start, end, frequency, frequency, "return")
    
    def _compute_excess_returns(
        self,
        start: date,
        end: date,
        ticker: str,
        frequency: str,
        **kwargs
    ) -> pd.Series:
        """Compute excess returns over risk-free rate."""
        # Get the underlying returns
        returns = self._compute_total_returns(start, end, ticker, frequency, **kwargs)
        
        # Get risk-free rate
        risk_free = self._compute_nominal_risk_free(start, end, frequency, **kwargs)
        
        # Calculate excess returns
        excess_returns = self.return_calculator.calculate_excess_returns(returns, risk_free)
        
        return excess_returns
    
    def _compute_inflation_rate(
        self,
        start: date,
        end: date,
        frequency: str,
        **kwargs
    ) -> pd.Series:
        """Compute inflation rate from price indices."""
        method = kwargs.get("method", self.config["inflation_method"])
        
        # For YoY inflation, need 12+ months of history
        if method == "yoy":
            extended_start = self._extend_start_date(start, frequency, periods=12)
        else:
            extended_start = self._extend_start_date(start, frequency, periods=2)
        
        # Try CPI first, then PCE
        try:
            cpi_data = self.raw_provider.get_data(RawDataType.CPI_INDEX, extended_start, end, None, frequency)
            inflation = self.economic_calculator.calculate_inflation_rate(cpi_data, method)
        except DataNotAvailableError:
            try:
                pce_data = self.raw_provider.get_data(RawDataType.PCE_INDEX, extended_start, end, None, frequency)
                inflation = self.economic_calculator.calculate_inflation_rate(pce_data, method)
            except DataNotAvailableError:
                raise DataNotAvailableError("No price index data available for inflation calculation")
        
        return self._trim_and_convert(inflation, start, end, frequency, frequency, "rate")
    
    def _compute_nominal_risk_free(
        self,
        start: date,
        end: date,
        frequency: str,
        **kwargs
    ) -> pd.Series:
        """Compute nominal risk-free rate."""
        tenor = kwargs.get("tenor", self.config["risk_free_tenor"])
        
        # Try to get various treasury rates
        available_rates = {}
        rate_types = [
            (RawDataType.TREASURY_3M, "3m"),
            (RawDataType.TREASURY_6M, "6m"),
            (RawDataType.TREASURY_1Y, "1y"),
            (RawDataType.TREASURY_2Y, "2y"),
            (RawDataType.FED_FUNDS, "fed_funds")
        ]
        
        for rate_type, rate_name in rate_types:
            try:
                rate_data = self.raw_provider.get_data(rate_type, start, end, None, frequency)
                available_rates[rate_name] = rate_data
            except DataNotAvailableError:
                continue
        
        if not available_rates:
            raise DataNotAvailableError("No treasury rate data available")
        
        # Select the best available rate
        risk_free_rate = self.economic_calculator.select_risk_free_rate(available_rates, tenor)
        
        return self._trim_and_convert(risk_free_rate, start, end, frequency, frequency, "rate")
    
    def _compute_real_risk_free(
        self,
        start: date,
        end: date,
        frequency: str,
        **kwargs
    ) -> pd.Series:
        """Compute real risk-free rate."""
        # Get nominal risk-free rate
        nominal_rate = self._compute_nominal_risk_free(start, end, frequency, **kwargs)
        
        # Get inflation rate  
        inflation_rate = self._compute_inflation_rate(start, end, frequency, **kwargs)
        
        # Calculate real rate using Fisher equation
        real_rate = self.economic_calculator.calculate_real_rate(nominal_rate, inflation_rate)
        
        return real_rate
    
    def _can_compute_logical_data(
        self,
        data_type: LogicalDataType,
        start: date,
        end: date,
        ticker: Optional[str],
        **kwargs
    ) -> bool:
        """Check if we can compute a logical data type."""
        try:
            if data_type in [LogicalDataType.TOTAL_RETURN, LogicalDataType.SIMPLE_RETURN, LogicalDataType.LOG_RETURN]:
                # Need price data
                price_type = RawDataType.ADJUSTED_CLOSE if self.config["use_adjusted_close"] else RawDataType.OHLCV
                return self.raw_provider.is_available(price_type, start, end, ticker)
                
            elif data_type == LogicalDataType.EXCESS_RETURN:
                # Need both price data and risk-free rate
                price_type = RawDataType.ADJUSTED_CLOSE if self.config["use_adjusted_close"] else RawDataType.OHLCV
                return (self.raw_provider.is_available(price_type, start, end, ticker) and
                        self._can_compute_risk_free_rate(start, end))
                
            elif data_type == LogicalDataType.INFLATION_RATE:
                # Need CPI or PCE data
                return (self.raw_provider.is_available(RawDataType.CPI_INDEX, start, end, None) or
                        self.raw_provider.is_available(RawDataType.PCE_INDEX, start, end, None))
                
            elif data_type in [LogicalDataType.NOMINAL_RISK_FREE, LogicalDataType.REAL_RISK_FREE]:
                return self._can_compute_risk_free_rate(start, end)
                
            else:
                return False
                
        except Exception:
            return False
    
    def _can_compute_risk_free_rate(self, start: date, end: date) -> bool:
        """Check if we can compute risk-free rate from available data."""
        rate_types = [RawDataType.TREASURY_3M, RawDataType.TREASURY_6M, 
                     RawDataType.TREASURY_1Y, RawDataType.FED_FUNDS]
        
        for rate_type in rate_types:
            if self.raw_provider.is_available(rate_type, start, end, None):
                return True
        return False
    
    def _extend_start_date(self, start: date, frequency: str, periods: int) -> date:
        """Extend start date backward to get enough data for calculations."""
        if frequency.lower() == "daily":
            return start - timedelta(days=periods * 7)  # Extra buffer for weekends
        elif frequency.lower() == "weekly":
            return start - timedelta(weeks=periods + 2)
        elif frequency.lower() == "monthly":
            # Approximate: 30 days per month with buffer
            return start - timedelta(days=(periods + 1) * 35)
        elif frequency.lower() == "quarterly":
            return start - timedelta(days=(periods + 1) * 100)
        elif frequency.lower() == "annual":
            return start - timedelta(days=(periods + 1) * 370)
        else:
            # Default: extend by lookback buffer
            return start - timedelta(days=self.config["lookback_buffer_days"])
    
    def _trim_and_convert(
        self,
        series: pd.Series,
        start: date,
        end: date,
        from_frequency: str,
        to_frequency: str,
        data_type: str
    ) -> pd.Series:
        """Trim series to date range and convert frequency if needed."""
        # First trim to requested date range
        if not series.empty:
            # Handle timezone-aware indices properly
            start_datetime = pd.Timestamp(start)
            end_datetime = pd.Timestamp(end)
            
            # If series index is timezone-aware, localize the comparison timestamps
            if hasattr(series.index, 'tz') and series.index.tz is not None:
                start_datetime = start_datetime.tz_localize(series.index.tz)
                end_datetime = end_datetime.tz_localize(series.index.tz)
            
            mask = (series.index >= start_datetime) & (series.index <= end_datetime)
            series = series.loc[mask]
        
        # Convert frequency if needed
        if from_frequency.lower() != to_frequency.lower():
            if self.frequency_converter.can_convert(from_frequency, to_frequency):
                series = self.frequency_converter.auto_convert(
                    series, from_frequency, to_frequency, data_type
                )
            else:
                logger.warning(f"Cannot convert from {from_frequency} to {to_frequency}")
        
        return series
    
    def _get_universe_data_individual(
        self,
        data_type: DataType,
        tickers: List[str],
        start: date,
        end: date,
        frequency: Union[str, Frequency],
        **kwargs
    ) -> pd.DataFrame:
        """Get universe data by making individual calls for each ticker."""
        result_data = {}
        
        for ticker in tickers:
            try:
                series = self.get_data(data_type, start, end, ticker, frequency, **kwargs)
                result_data[ticker] = series
            except (DataNotAvailableError, InvalidTickerError) as e:
                logger.warning(f"Failed to get {data_type.value} for {ticker}: {e}")
                # Create empty series for missing data
                result_data[ticker] = pd.Series(dtype=float, name=ticker)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(result_data)
        
        return result_df