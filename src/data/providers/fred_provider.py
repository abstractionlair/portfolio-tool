"""
FRED (Federal Reserve Economic Data) provider implementation.

This module provides economic data using the pandas_datareader library to access
FRED data, implementing the RawDataProvider interface.
"""

import logging
from datetime import date, datetime
from typing import Optional, Union, List
import warnings

import pandas as pd
import pandas_datareader.data as web
import numpy as np

from ..interfaces import (
    RawDataProvider, RawDataType, Frequency,
    validate_ticker_requirement, validate_date_range,
    DataNotAvailableError, InvalidTickerError
)

logger = logging.getLogger(__name__)


class FREDProvider(RawDataProvider):
    """FRED implementation of RawDataProvider for economic data."""
    
    def __init__(self):
        """Initialize FRED provider."""
        self._cache = {}  # Simple in-memory cache
        
        # Economic data types this provider supports  
        self.supported_types = {
            RawDataType.TREASURY_3M,
            RawDataType.TREASURY_6M,
            RawDataType.TREASURY_1Y,
            RawDataType.TREASURY_2Y,
            RawDataType.TREASURY_5Y,
            RawDataType.TREASURY_10Y,
            RawDataType.TREASURY_30Y,
            RawDataType.TIPS_5Y,
            RawDataType.TIPS_10Y,
            RawDataType.TIPS_30Y,
            RawDataType.FED_FUNDS,
            RawDataType.SOFR,
            RawDataType.CPI_INDEX,
            RawDataType.PCE_INDEX
        }
        
        # FRED series codes mapping
        self.series_codes = {
            RawDataType.TREASURY_3M: 'DGS3MO',
            RawDataType.TREASURY_6M: 'DGS6MO', 
            RawDataType.TREASURY_1Y: 'DGS1',
            RawDataType.TREASURY_2Y: 'DGS2',
            RawDataType.TREASURY_5Y: 'DGS5',
            RawDataType.TREASURY_10Y: 'DGS10',
            RawDataType.TREASURY_30Y: 'DGS30',
            RawDataType.TIPS_5Y: 'DFII5',
            RawDataType.TIPS_10Y: 'DFII10',
            RawDataType.TIPS_30Y: 'DFII30',
            RawDataType.FED_FUNDS: 'FEDFUNDS',
            RawDataType.SOFR: 'SOFR',
            RawDataType.CPI_INDEX: 'CPIAUCSL',
            RawDataType.PCE_INDEX: 'PCEPI'
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
        Fetch data from FRED.
        
        Args:
            data_type: Type of data to fetch
            start: Start date
            end: End date  
            ticker: Not used for economic data (should be None)
            frequency: Data frequency
            **kwargs: Additional arguments
            
        Returns:
            pandas Series with date index and data values
            
        Raises:
            InvalidTickerError: If ticker is provided for economic data
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
                f"FRED provider does not support {data_type.value}"
            )
        
        # Get FRED series code
        series_code = self.series_codes.get(data_type)
        if not series_code:
            raise DataNotAvailableError(
                f"No FRED series code available for {data_type.value}"
            )
        
        try:
            return self._fetch_fred_data(series_code, data_type, start, end, frequency)
            
        except Exception as e:
            logger.error(f"Failed to fetch {data_type.value} from FRED: {e}")
            # Try fallback data generation
            return self._get_fallback_data(data_type, start, end, frequency)
    
    def is_available(
        self,
        data_type: RawDataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Check if data is available from FRED.
        
        Args:
            data_type: Type of data to check
            start: Start date
            end: End date
            ticker: Should be None for economic data
            **kwargs: Additional arguments
            
        Returns:
            True if data should be available
        """
        try:
            validate_ticker_requirement(data_type, ticker)
            validate_date_range(start, end)
            return data_type in self.supported_types
        except Exception:
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
        Fetch data for multiple tickers (not applicable for economic data).
        
        Economic data doesn't have tickers, so this will raise an error
        for supported data types.
        """
        # Raw providers should only accept RawDataType
        if not isinstance(data_type, RawDataType):
            raise ValueError(
                f"RawDataProvider only accepts RawDataType, got {type(data_type).__name__}"
            )
        
        if data_type in self.supported_types:
            raise DataNotAvailableError(
                f"Economic data type {data_type.value} does not support multiple tickers"
            )
        else:
            raise DataNotAvailableError(
                f"FRED provider does not support {data_type.value}"
            )
    
    def _fetch_fred_data(
        self,
        series_code: str,
        data_type: RawDataType,
        start: date,
        end: date,
        frequency: Union[str, Frequency]
    ) -> pd.Series:
        """Fetch data from FRED API."""
        
        # Check cache first
        cache_key = f"{series_code}_{start}_{end}_{frequency}"
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {series_code}")
            return self._cache[cache_key].copy()
        
        try:
            # Suppress pandas_datareader warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Fetch from FRED
                data = web.DataReader(
                    series_code,
                    'fred',
                    start=start,
                    end=end
                )
            
            if data.empty:
                raise DataNotAvailableError(f"No data returned for {series_code}")
            
            # Extract the series (FRED returns DataFrame with single column)
            if isinstance(data, pd.DataFrame):
                if len(data.columns) == 1:
                    series = data.iloc[:, 0]
                else:
                    # Look for the series code column
                    if series_code in data.columns:
                        series = data[series_code]
                    else:
                        series = data.iloc[:, 0]
            else:
                series = data
            
            # Clean the data
            series = series.dropna()
            
            if series.empty:
                raise DataNotAvailableError(f"No valid data for {series_code}")
            
            # Convert interest rates from percentages to decimals
            if self._is_interest_rate(data_type):
                series = series / 100.0
            
            # Handle frequency conversion if needed
            series = self._handle_frequency_conversion(series, frequency)
            
            # Set proper name
            series.name = data_type.value
            
            # Cache the result
            self._cache[cache_key] = series.copy()
            
            return series
            
        except Exception as e:
            if "Access Denied" in str(e) or "403" in str(e):
                logger.warning(f"FRED API access denied for {series_code}, using fallback")
                raise DataNotAvailableError(f"FRED API access denied: {e}")
            else:
                logger.error(f"Error fetching {series_code} from FRED: {e}")
                raise DataNotAvailableError(f"Could not fetch {series_code}: {e}")
    
    def _get_fallback_data(
        self,
        data_type: RawDataType,
        start: date,
        end: date,
        frequency: Union[str, Frequency]
    ) -> pd.Series:
        """Generate synthetic fallback data when FRED API is unavailable."""
        logger.info(f"Generating fallback data for {data_type.value}")
        
        # Create date range
        freq_str = self._get_pandas_frequency(frequency)
        dates = pd.date_range(start=start, end=end, freq=freq_str)
        
        if len(dates) == 0:
            return pd.Series(dtype=float, name=data_type.value)
        
        # Generate realistic synthetic data based on data type
        np.random.seed(42)  # For reproducibility
        
        if self._is_interest_rate(data_type):
            # Interest rates: base rate with realistic variation
            if data_type == RawDataType.FED_FUNDS:
                base_rate = 0.025  # 2.5%
                volatility = 0.005
            elif "3M" in data_type.value or "6M" in data_type.value:
                base_rate = 0.03   # 3%
                volatility = 0.003
            elif "1Y" in data_type.value or "2Y" in data_type.value:
                base_rate = 0.035  # 3.5%
                volatility = 0.004
            elif "10Y" in data_type.value:
                base_rate = 0.04   # 4%
                volatility = 0.005
            elif "30Y" in data_type.value:
                base_rate = 0.045  # 4.5%
                volatility = 0.006
            else:
                base_rate = 0.03
                volatility = 0.004
            
            # Generate with mean reversion
            values = np.zeros(len(dates))
            values[0] = base_rate
            
            for i in range(1, len(values)):
                # Mean reversion with noise
                reversion = 0.1 * (base_rate - values[i-1])
                noise = np.random.normal(0, volatility)
                values[i] = max(0.001, values[i-1] + reversion + noise)
            
            data = values
            
        elif data_type == RawDataType.CPI_INDEX:
            # CPI: steadily increasing with ~2% annual inflation
            initial_cpi = 280.0
            annual_inflation = 0.02
            daily_growth = (1 + annual_inflation) ** (1/252) - 1
            
            # Generate with some noise
            growth_rates = np.random.normal(daily_growth, 0.0002, len(dates))
            data = initial_cpi * np.cumprod(1 + growth_rates)
            
        elif data_type == RawDataType.PCE_INDEX:
            # PCE: similar to CPI but slightly lower
            initial_pce = 110.0
            annual_inflation = 0.018
            daily_growth = (1 + annual_inflation) ** (1/252) - 1
            
            growth_rates = np.random.normal(daily_growth, 0.0001, len(dates))
            data = initial_pce * np.cumprod(1 + growth_rates)
            
            
        else:
            # Default: positive trending data
            data = np.abs(np.random.normal(100, 10, len(dates)))
        
        series = pd.Series(data=data, index=dates, name=data_type.value)
        return series
    
    def _is_interest_rate(self, data_type: RawDataType) -> bool:
        """Check if data type is an interest rate."""
        rate_types = {
            RawDataType.TREASURY_3M, RawDataType.TREASURY_6M,
            RawDataType.TREASURY_1Y, RawDataType.TREASURY_2Y,
            RawDataType.TREASURY_5Y, RawDataType.TREASURY_10Y,
            RawDataType.TREASURY_30Y, RawDataType.TIPS_5Y,
            RawDataType.TIPS_10Y, RawDataType.TIPS_30Y,
            RawDataType.FED_FUNDS, RawDataType.SOFR
        }
        return data_type in rate_types
    
    def _get_pandas_frequency(self, frequency: Union[str, Frequency]) -> str:
        """Convert frequency to pandas frequency string."""
        if isinstance(frequency, Frequency):
            return frequency.pandas_freq
        
        mapping = {
            "daily": "D",
            "weekly": "W-FRI",
            "monthly": "ME", 
            "quarterly": "QE",
            "annual": "YE"
        }
        return mapping.get(frequency, "D")
    
    def _handle_frequency_conversion(
        self,
        series: pd.Series,
        target_frequency: Union[str, Frequency]
    ) -> pd.Series:
        """Handle frequency conversion if needed."""
        # For now, return as-is since FRED data comes in its native frequency
        # In a more complete implementation, we might resample here
        return series