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
            "use_adjusted_close": True,  # When True, dividends are implicit in prices
            "dividend_reinvestment": True,  # Only applies when use_adjusted_close=False
            "handle_splits": True,  # Only applies when use_adjusted_close=False
            "lookback_buffer_days": 400  # Extra days for calculations
        }
        
        # Validate configuration consistency
        self._validate_config()
    
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
        """
        Compute total returns including dividends.
        
        Handles adjusted vs unadjusted prices correctly:
        - If use_adjusted_close=True: dividends are already included in adjusted prices
        - If use_adjusted_close=False: explicit dividends are added to unadjusted prices
        """
        # Extend date range to get previous price for return calculation
        extended_start = self._extend_start_date(start, frequency, periods=1)
        
        if self.config["use_adjusted_close"]:
            # OPTION A: Use adjusted close (dividends already included)
            prices = self.raw_provider.get_data(RawDataType.ADJUSTED_CLOSE, extended_start, end, ticker, frequency)
            
            # Calculate returns from adjusted prices (no explicit dividends needed)
            total_returns = self.return_calculator.calculate_simple_returns(prices, frequency)
            
            logger.debug(f"Using adjusted close prices for {ticker} - dividends implicitly included")
            
        else:
            # OPTION B: Use unadjusted close with explicit dividends
            prices = self.raw_provider.get_data(RawDataType.OHLCV, extended_start, end, ticker, frequency)
            
            # Get corporate actions
            dividends = None
            splits = None
            
            if self.config["dividend_reinvestment"]:
                try:
                    dividends = self.raw_provider.get_data(RawDataType.DIVIDEND, extended_start, end, ticker, frequency)
                    logger.debug(f"Retrieved dividend data for {ticker}")
                except DataNotAvailableError:
                    logger.debug(f"No dividend data available for {ticker}")
            
            # Check if we need to handle splits
            try:
                splits = self.raw_provider.get_data(RawDataType.SPLIT, extended_start, end, ticker, frequency)
                logger.debug(f"Retrieved split data for {ticker}")
            except DataNotAvailableError:
                logger.debug(f"No split data available for {ticker}")
            
            # Use comprehensive method for unadjusted prices if we have splits
            if splits is not None and not splits.empty and (splits != 1.0).any():
                total_returns = self.return_calculator.calculate_comprehensive_total_returns(
                    prices, dividends, splits
                )
                logger.debug(f"Using comprehensive method for {ticker} with splits")
            else:
                # No splits, just handle dividends with regular method
                total_returns = self.return_calculator.calculate_total_returns(prices, dividends)
                logger.debug(f"Using regular method for {ticker} without splits")
        
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
        
        # Convert annualized inflation rates to daily rates for consistency with return components
        # YoY inflation is computed as annualized rates, but return components are daily changes
        if frequency.lower() == "daily":
            inflation = inflation / 252
        elif frequency.lower() == "monthly":
            inflation = inflation / 12
        elif frequency.lower() == "quarterly":
            inflation = inflation / 4
        # Annual rates stay as-is
        
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
        
        # Convert annualized Treasury rates to daily rates for consistency with return components
        # Treasury rates are provided as annualized rates, but return components are daily changes
        if frequency.lower() == "daily":
            risk_free_rate = risk_free_rate / 252
        elif frequency.lower() == "monthly":
            risk_free_rate = risk_free_rate / 12
        elif frequency.lower() == "quarterly":
            risk_free_rate = risk_free_rate / 4
        # Annual rates stay as-is
        
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
    
    def _validate_config(self):
        """Validate configuration consistency."""
        if self.config["use_adjusted_close"] and self.config.get("explicit_dividends", False):
            logger.warning("Both use_adjusted_close and explicit_dividends are True - "
                         "dividends are already in adjusted prices")
        
        # Log current configuration for debugging
        logger.debug(f"TransformedDataProvider configuration: {self.config}")
    
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
    
    def decompose_returns(
        self,
        ticker: str,
        start: date,
        end: date,
        earnings_data: Optional[pd.Series] = None,
        frequency: str = "daily"
    ) -> Dict[str, pd.Series]:
        """
        Decompose total returns into components.
        
        Without earnings data:
        - Total return = Dividend yield + Price appreciation
        
        With earnings data:
        - Total return = Dividend yield + Earnings growth + P/E change
        
        Args:
            ticker: Stock ticker
            start: Start date
            end: End date
            earnings_data: Optional earnings per share data
            frequency: Data frequency
            
        Returns:
            Dictionary with return components
        """
        # Extend date range for calculations
        extended_start = self._extend_start_date(start, frequency, periods=1)
        
        # Get raw data
        unadjusted_prices = self.raw_provider.get_data(RawDataType.OHLCV, extended_start, end, ticker, frequency)
        adjusted_prices = self.raw_provider.get_data(RawDataType.ADJUSTED_CLOSE, extended_start, end, ticker, frequency)
        
        try:
            dividends = self.raw_provider.get_data(RawDataType.DIVIDEND, extended_start, end, ticker, frequency)
        except DataNotAvailableError:
            dividends = pd.Series([], dtype=float, name='dividends')
        
        # Calculate total returns from adjusted prices
        total_returns = self.return_calculator.calculate_simple_returns(adjusted_prices, frequency)
        
        # Calculate dividend yield (dividends / previous price)
        dividend_yield = pd.Series(0.0, index=unadjusted_prices.index, name='dividend_yield')
        
        if not dividends.empty:
            # Align dividends with prices
            for div_date, div_amount in dividends.items():
                if div_date in unadjusted_prices.index:
                    # Find previous trading day
                    price_dates = unadjusted_prices.index
                    current_idx = price_dates.get_loc(div_date)
                    if current_idx > 0:
                        prev_price = unadjusted_prices.iloc[current_idx - 1]
                        if prev_price > 0:
                            dividend_yield.loc[div_date] = div_amount / prev_price
        
        # Trim to requested date range
        total_returns = self._trim_and_convert(total_returns, start, end, frequency, frequency, "return")
        dividend_yield = self._trim_and_convert(dividend_yield, start, end, frequency, frequency, "return")
        
        # Calculate price appreciation (total return - dividend yield)
        price_appreciation = total_returns - dividend_yield
        price_appreciation.name = 'price_appreciation'
        
        result = {
            'total_return': total_returns,
            'dividend_yield': dividend_yield,
            'price_appreciation': price_appreciation
        }
        
        # If earnings data provided, calculate P/E decomposition
        if earnings_data is not None:
            # Trim earnings data to same period
            earnings_trimmed = earnings_data.loc[start:end] if hasattr(earnings_data, 'loc') else earnings_data
            
            # Calculate P/E ratios using unadjusted prices
            unadjusted_trimmed = self._trim_and_convert(unadjusted_prices, start, end, frequency, frequency, "price")
            
            # Align earnings with prices - handle timezone issues
            if hasattr(unadjusted_trimmed.index, 'tz') and unadjusted_trimmed.index.tz is not None:
                # If price data has timezone info, localize earnings data
                if not hasattr(earnings_trimmed.index, 'tz') or earnings_trimmed.index.tz is None:
                    earnings_trimmed.index = earnings_trimmed.index.tz_localize(unadjusted_trimmed.index.tz)
            
            aligned_earnings = earnings_trimmed.reindex(unadjusted_trimmed.index).ffill()
            
            # Calculate P/E ratio
            pe_ratio = unadjusted_trimmed / aligned_earnings
            pe_ratio.name = 'pe_ratio'
            
            # Calculate earnings growth
            earnings_growth = aligned_earnings.pct_change()
            earnings_growth.name = 'earnings_growth'
            
            # Calculate P/E change
            pe_change = pe_ratio.pct_change()
            pe_change.name = 'pe_change'
            
            # Add to result
            result.update({
                'earnings_growth': earnings_growth,
                'pe_change': pe_change,
                'pe_ratio': pe_ratio
            })
            
            # Recalculate price appreciation to exclude dividends
            result['price_return_ex_div'] = total_returns - dividend_yield
        
        return result
    
    def decompose_equity_returns(
        self,
        ticker: str,
        start: date,
        end: date,
        earnings_data: pd.Series,
        frequency: str = "daily",
        inflation_measure: str = "CPI",
        rf_tenor: str = "3M"
    ) -> Dict[str, pd.Series]:
        """
        Decompose equity returns into economically meaningful components.
        
        This method separates total returns into components that have different
        economic interpretations and time series properties:
        
        - Dividend yield: Income component
        - P/E change: Multiple expansion/contraction  
        - Real earnings excess: Real earnings growth above real risk-free rate
        
        The key insight is that earnings growth needs to be adjusted for inflation
        and the real risk-free rate to get the true risk premium component.
        
        Args:
            ticker: Stock ticker
            start: Start date
            end: End date
            earnings_data: Earnings per share data (aligned with dates)
            frequency: Data frequency
            inflation_measure: Inflation measure to use ("CPI", "PCE")
            rf_tenor: Risk-free rate tenor ("3M", "6M", "1Y")
            
        Returns:
            Dictionary with return components:
            - nominal_return: Total nominal return
            - dividend_yield: Dividend yield component
            - pe_change: P/E multiple change component  
            - nominal_earnings_growth: Earnings growth in nominal terms
            - real_earnings_growth: Earnings growth adjusted for inflation
            - real_earnings_excess: Real earnings growth above real risk-free rate
            - inflation: Inflation rate over period
            - nominal_rf: Nominal risk-free rate
            - real_rf: Real risk-free rate
            - real_risk_premium: Total real risk premium
            - excess_return: Nominal return minus nominal risk-free rate
        """
        # Get base decomposition with earnings
        base_decomp = self.decompose_returns(
            ticker=ticker,
            start=start,
            end=end,
            earnings_data=earnings_data,
            frequency=frequency
        )
        
        # Get economic data
        economic_data = self._get_economic_data_for_decomposition(
            start=start,
            end=end,
            frequency=frequency,
            inflation_measure=inflation_measure,
            rf_tenor=rf_tenor
        )
        
        # Align all data
        aligned_data = self._align_decomposition_data(
            base_decomp=base_decomp,
            economic_data=economic_data,
            frequency=frequency
        )
        
        # Calculate real components
        final_data = self._calculate_real_components(aligned_data)
        
        # Convert back to series dictionary
        result = {}
        for col in final_data.columns:
            result[col] = final_data[col].copy()
            result[col].name = col
        
        return result
    
    def _get_economic_data_for_decomposition(
        self,
        start: date,
        end: date,
        frequency: str,
        inflation_measure: str = "CPI",
        rf_tenor: str = "3M"
    ) -> Dict[str, pd.Series]:
        """Fetch and align economic data needed for decomposition."""
        
        # Get inflation rate
        inflation = self._compute_inflation_rate(start, end, frequency)
        
        # Get nominal risk-free rate
        nominal_rf = self._compute_nominal_risk_free(start, end, frequency, tenor=rf_tenor)
        
        # Get real risk-free rate
        real_rf = self._compute_real_risk_free(start, end, frequency, tenor=rf_tenor)
        
        return {
            'inflation': inflation,
            'nominal_rf': nominal_rf,
            'real_rf': real_rf
        }
    
    def _align_decomposition_data(
        self,
        base_decomp: Dict[str, pd.Series],
        economic_data: Dict[str, pd.Series],
        frequency: str
    ) -> pd.DataFrame:
        """Align all data series for consistent calculation."""
        
        # Combine all data
        all_data = {}
        all_data.update(base_decomp)
        
        # Handle timezone differences between base and economic data
        if base_decomp and economic_data:
            # Get a representative series from base decomposition
            base_series = next(iter(base_decomp.values()))
            
            # Check if base data has timezone info
            if hasattr(base_series.index, 'tz') and base_series.index.tz is not None:
                # If base data has timezone, localize economic data
                for key, series in economic_data.items():
                    if not hasattr(series.index, 'tz') or series.index.tz is None:
                        series.index = series.index.tz_localize(base_series.index.tz)
                    all_data[key] = series
            else:
                # If base data has no timezone, remove timezone from economic data
                for key, series in economic_data.items():
                    if hasattr(series.index, 'tz') and series.index.tz is not None:
                        series.index = series.index.tz_localize(None)
                    all_data[key] = series
        else:
            all_data.update(economic_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Rename columns for clarity
        if 'total_return' in df.columns:
            df['nominal_return'] = df['total_return']
        if 'earnings_growth' in df.columns:
            df['nominal_earnings_growth'] = df['earnings_growth']
        
        # Forward fill economic data (rates don't change daily)
        # Also backward fill to handle cases where economic data starts later in the period
        for col in ['inflation', 'nominal_rf', 'real_rf']:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # Drop rows with NaN in critical columns (including economic data for real calculations)
        # Use total_return if nominal_return doesn't exist yet
        critical_cols = ['dividend_yield', 'nominal_earnings_growth', 'pe_change', 'inflation', 'real_rf']
        if 'nominal_return' in df.columns:
            critical_cols.append('nominal_return')
        elif 'total_return' in df.columns:
            critical_cols.append('total_return')
        
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        return df
    
    def _calculate_real_components(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate real risk premium components."""
        
        result = aligned_data.copy()
        
        # Real earnings growth (adjust for inflation)
        result['real_earnings_growth'] = result['nominal_earnings_growth'] - result['inflation']
        
        # Real earnings excess (real earnings growth above real risk-free rate)
        result['real_earnings_excess'] = result['real_earnings_growth'] - result['real_rf']
        
        # Total real risk premium (sum of real components)
        # Key insight: dividend yield and P/E change are already "real" ratios
        result['real_risk_premium'] = (
            result['dividend_yield'] + 
            result['pe_change'] + 
            result['real_earnings_excess']
        )
        
        # Excess return (nominal return minus nominal risk-free rate)
        result['excess_return'] = result['nominal_return'] - result['nominal_rf']
        
        # Verification: alternative calculation of real risk premium
        # Should equal: nominal return - inflation - real risk-free rate
        result['real_risk_premium_check'] = (
            result['nominal_return'] - result['inflation'] - result['real_rf']
        )
        
        # Decomposition quality check
        result['decomp_error'] = abs(
            result['real_risk_premium'] - result['real_risk_premium_check']
        )
        
        # Identity check: nominal return should equal sum of components
        result['identity_check'] = (
            result['dividend_yield'] + 
            result['pe_change'] + 
            result['nominal_earnings_growth']
        )
        
        result['identity_error'] = abs(
            result['nominal_return'] - result['identity_check']
        )
        
        return result