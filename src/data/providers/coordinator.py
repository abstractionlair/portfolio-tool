"""
Raw Data Provider Coordinator.

This module provides a coordinator that routes data requests to the appropriate
data provider based on the data type.
"""

import logging
from datetime import date
from typing import Optional, Union, List

import pandas as pd

from ..interfaces import (
    RawDataProvider, RawDataType, DataTypeCategory, Frequency,
    validate_ticker_requirement, validate_date_range,
    DataNotAvailableError
)
from .yfinance_provider import YFinanceProvider
from .fred_provider import FREDProvider

logger = logging.getLogger(__name__)


class RawDataProviderCoordinator(RawDataProvider):
    """
    Coordinates multiple raw data providers to handle different data types.
    
    Routes requests to the appropriate provider based on the data type category:
    - Securities data -> YFinance provider
    - Economic data -> FRED provider
    """
    
    def __init__(self):
        """Initialize the coordinator with all available providers."""
        self.yfinance = YFinanceProvider()
        self.fred = FREDProvider()
        
        # Build routing map based on data type categories
        self._provider_map = {}
        
        # Map securities data to YFinance
        for data_type in RawDataType:
            if data_type.category == DataTypeCategory.SECURITY_RAW:
                self._provider_map[data_type] = self.yfinance
        
        # Map economic data to FRED
        for data_type in RawDataType:
            if data_type.category == DataTypeCategory.ECONOMIC_RAW:
                self._provider_map[data_type] = self.fred
        
        logger.info(f"Initialized coordinator with {len(self._provider_map)} data type mappings")
    
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
        Route data request to appropriate provider.
        
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
            DataNotAvailableError: If no provider supports the data type
        """
        # Validate inputs
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # Raw providers should only accept RawDataType
        if not isinstance(data_type, RawDataType):
            raise ValueError(
                f"RawDataProvider only accepts RawDataType, got {type(data_type).__name__}"
            )
        
        # Find the appropriate provider
        provider = self._provider_map.get(data_type)
        if not provider:
            raise DataNotAvailableError(
                f"No provider available for data type: {data_type.value}"
            )
        
        logger.debug(f"Routing {data_type.value} request to {provider.__class__.__name__}")
        
        try:
            return provider.get_data(data_type, start, end, ticker, frequency, **kwargs)
        except Exception as e:
            logger.error(f"Provider {provider.__class__.__name__} failed for {data_type.value}: {e}")
            raise
    
    def is_available(
        self,
        data_type: RawDataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Check if data is available from any provider.
        
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
            
            provider = self._provider_map.get(data_type)
            if not provider:
                return False
            
            return provider.is_available(data_type, start, end, ticker, **kwargs)
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
        Fetch data for multiple tickers from appropriate provider.
        
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
        
        # Find the appropriate provider
        provider = self._provider_map.get(data_type)
        if not provider:
            raise DataNotAvailableError(
                f"No provider available for data type: {data_type.value}"
            )
        
        logger.debug(f"Routing universe {data_type.value} request to {provider.__class__.__name__}")
        
        try:
            return provider.get_universe_data(data_type, tickers, start, end, frequency, **kwargs)
        except Exception as e:
            logger.error(f"Provider {provider.__class__.__name__} failed for universe {data_type.value}: {e}")
            raise
    
    def get_supported_data_types(self) -> List[RawDataType]:
        """Get list of all supported data types across all providers."""
        return list(self._provider_map.keys())
    
    def get_provider_for_data_type(self, data_type: RawDataType) -> Optional[RawDataProvider]:
        """Get the provider that handles a specific data type."""
        return self._provider_map.get(data_type)
    
    def get_provider_summary(self) -> dict:
        """Get summary of provider capabilities."""
        summary = {
            "total_data_types": len(self._provider_map),
            "providers": {}
        }
        
        # Count data types per provider
        provider_counts = {}
        for data_type, provider in self._provider_map.items():
            provider_name = provider.__class__.__name__
            if provider_name not in provider_counts:
                provider_counts[provider_name] = []
            provider_counts[provider_name].append(data_type.value)
        
        summary["providers"] = provider_counts
        return summary