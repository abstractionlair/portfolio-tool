"""
Frequency conversion utilities.

This module handles proper frequency conversion for financial time series,
with special attention to compounding for returns.
"""

import logging
from typing import Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FrequencyConverter:
    """Converts time series between frequencies with proper handling for different data types."""
    
    # Frequency hierarchy (higher number = higher frequency)
    FREQUENCY_ORDER = {
        "annual": 1,
        "quarterly": 4,
        "monthly": 12,
        "weekly": 52,
        "daily": 252  # Trading days
    }
    
    # Pandas frequency strings
    PANDAS_FREQUENCIES = {
        "daily": "D",
        "weekly": "W-FRI",  # Weekly ending Friday
        "monthly": "ME",    # Month end
        "quarterly": "QE",  # Quarter end
        "annual": "YE"      # Year end
    }
    
    def can_convert(self, from_freq: str, to_freq: str) -> bool:
        """
        Check if conversion is possible (only allow downsampling).
        
        Args:
            from_freq: Source frequency
            to_freq: Target frequency
            
        Returns:
            True if conversion is possible
        """
        from_order = self.FREQUENCY_ORDER.get(from_freq.lower())
        to_order = self.FREQUENCY_ORDER.get(to_freq.lower())
        
        if from_order is None or to_order is None:
            logger.warning(f"Unknown frequency: {from_freq} or {to_freq}")
            return False
        
        # Only allow downsampling (high frequency to low frequency)
        return from_order >= to_order
    
    def convert_returns(
        self,
        returns: pd.Series,
        from_frequency: str,
        to_frequency: str,
        method: str = "compound"
    ) -> pd.Series:
        """
        Convert return series between frequencies with proper compounding.
        
        Args:
            returns: Return series to convert
            from_frequency: Source frequency ("daily", "weekly", "monthly", etc.)
            to_frequency: Target frequency
            method: Conversion method:
                - "compound": Compound returns properly (default for returns)
                - "average": Simple average (not recommended for returns)
                - "sum": Sum values (for flow variables)
                
        Returns:
            Return series at target frequency
        """
        if not self.can_convert(from_frequency, to_frequency):
            raise ValueError(
                f"Cannot convert from {from_frequency} to {to_frequency}. "
                "Only downsampling (high to low frequency) is supported."
            )
        
        if from_frequency.lower() == to_frequency.lower():
            # No conversion needed
            return returns.copy()
        
        pandas_freq = self.PANDAS_FREQUENCIES.get(to_frequency.lower())
        if pandas_freq is None:
            raise ValueError(f"Unsupported target frequency: {to_frequency}")
        
        if method == "compound":
            # Compound returns: (1 + r1) * (1 + r2) * ... - 1
            compounded = (1 + returns.fillna(0)).resample(pandas_freq).prod() - 1
            
        elif method == "average":
            # Simple average (not recommended for returns)
            compounded = returns.resample(pandas_freq).mean()
            
        elif method == "sum":
            # Sum values (useful for flows like dividends)
            compounded = returns.resample(pandas_freq).sum()
            
        else:
            raise ValueError(f"Unsupported conversion method: {method}")
        
        compounded.name = f"{returns.name}_{to_frequency}" if returns.name else f"{to_frequency}_returns"
        return compounded
    
    def convert_prices(
        self,
        prices: pd.Series,
        from_frequency: str,
        to_frequency: str,
        method: str = "last"
    ) -> pd.Series:
        """
        Convert price series between frequencies.
        
        Args:
            prices: Price series to convert
            from_frequency: Source frequency
            to_frequency: Target frequency
            method: Conversion method:
                - "last": Use last price in period (default for stock prices)
                - "first": Use first price in period
                - "average": Average price in period
                - "high": Highest price in period
                - "low": Lowest price in period
                
        Returns:
            Price series at target frequency
        """
        if not self.can_convert(from_frequency, to_frequency):
            raise ValueError(
                f"Cannot convert from {from_frequency} to {to_frequency}. "
                "Only downsampling is supported."
            )
        
        if from_frequency.lower() == to_frequency.lower():
            return prices.copy()
        
        pandas_freq = self.PANDAS_FREQUENCIES.get(to_frequency.lower())
        if pandas_freq is None:
            raise ValueError(f"Unsupported target frequency: {to_frequency}")
        
        if method == "last":
            converted = prices.resample(pandas_freq).last()
        elif method == "first":
            converted = prices.resample(pandas_freq).first()
        elif method == "average":
            converted = prices.resample(pandas_freq).mean()
        elif method == "high":
            converted = prices.resample(pandas_freq).max()
        elif method == "low":
            converted = prices.resample(pandas_freq).min()
        else:
            raise ValueError(f"Unsupported price conversion method: {method}")
        
        converted.name = f"{prices.name}_{to_frequency}" if prices.name else f"{to_frequency}_prices"
        return converted
    
    def convert_rates(
        self,
        rates: pd.Series,
        from_frequency: str,
        to_frequency: str,
        method: str = "average"
    ) -> pd.Series:
        """
        Convert interest rate or similar series between frequencies.
        
        Args:
            rates: Interest rate series
            from_frequency: Source frequency
            to_frequency: Target frequency
            method: Conversion method:
                - "average": Average rate over period (default for rates)
                - "last": Last rate in period
                - "first": First rate in period
                
        Returns:
            Rate series at target frequency
        """
        if not self.can_convert(from_frequency, to_frequency):
            raise ValueError(
                f"Cannot convert from {from_frequency} to {to_frequency}. "
                "Only downsampling is supported."
            )
        
        if from_frequency.lower() == to_frequency.lower():
            return rates.copy()
        
        pandas_freq = self.PANDAS_FREQUENCIES.get(to_frequency.lower())
        if pandas_freq is None:
            raise ValueError(f"Unsupported target frequency: {to_frequency}")
        
        if method == "average":
            converted = rates.resample(pandas_freq).mean()
        elif method == "last":
            converted = rates.resample(pandas_freq).last()
        elif method == "first":
            converted = rates.resample(pandas_freq).first()
        else:
            raise ValueError(f"Unsupported rate conversion method: {method}")
        
        converted.name = f"{rates.name}_{to_frequency}" if rates.name else f"{to_frequency}_rates"
        return converted
    
    def convert_volumes(
        self,
        volumes: pd.Series,
        from_frequency: str,
        to_frequency: str,
        method: str = "sum"
    ) -> pd.Series:
        """
        Convert volume series between frequencies.
        
        Args:
            volumes: Volume series
            from_frequency: Source frequency
            to_frequency: Target frequency
            method: Conversion method:
                - "sum": Sum volumes over period (default)
                - "average": Average volume per period
                - "max": Maximum volume in period
                
        Returns:
            Volume series at target frequency
        """
        if not self.can_convert(from_frequency, to_frequency):
            raise ValueError(
                f"Cannot convert from {from_frequency} to {to_frequency}. "
                "Only downsampling is supported."
            )
        
        if from_frequency.lower() == to_frequency.lower():
            return volumes.copy()
        
        pandas_freq = self.PANDAS_FREQUENCIES.get(to_frequency.lower())
        if pandas_freq is None:
            raise ValueError(f"Unsupported target frequency: {to_frequency}")
        
        if method == "sum":
            converted = volumes.resample(pandas_freq).sum()
        elif method == "average":
            converted = volumes.resample(pandas_freq).mean()
        elif method == "max":
            converted = volumes.resample(pandas_freq).max()
        else:
            raise ValueError(f"Unsupported volume conversion method: {method}")
        
        converted.name = f"{volumes.name}_{to_frequency}" if volumes.name else f"{to_frequency}_volumes"
        return converted
    
    def auto_convert(
        self,
        series: pd.Series,
        from_frequency: str,
        to_frequency: str,
        data_type: str = "return"
    ) -> pd.Series:
        """
        Automatically convert series using appropriate method based on data type.
        
        Args:
            series: Time series to convert
            from_frequency: Source frequency
            to_frequency: Target frequency
            data_type: Type of data to determine conversion method:
                - "return": Use compounding
                - "price": Use last value
                - "rate": Use average
                - "volume": Use sum
                - "dividend": Use sum
                
        Returns:
            Converted series
        """
        if data_type == "return":
            return self.convert_returns(series, from_frequency, to_frequency, method="compound")
        elif data_type == "price":
            return self.convert_prices(series, from_frequency, to_frequency, method="last")
        elif data_type == "rate":
            return self.convert_rates(series, from_frequency, to_frequency, method="average")
        elif data_type in ["volume", "dividend"]:
            return self.convert_volumes(series, from_frequency, to_frequency, method="sum")
        else:
            logger.warning(f"Unknown data type '{data_type}', using average conversion")
            pandas_freq = self.PANDAS_FREQUENCIES.get(to_frequency.lower())
            return series.resample(pandas_freq).mean()
    
    def infer_frequency(self, series: pd.Series) -> str:
        """
        Attempt to infer the frequency of a time series.
        
        Args:
            series: Time series with datetime index
            
        Returns:
            Inferred frequency string
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have datetime index for frequency inference")
        
        if len(series) < 2:
            raise ValueError("Need at least 2 observations to infer frequency")
        
        # Get the most common time difference
        time_diffs = series.index.to_series().diff().dropna()
        mode_diff = time_diffs.mode()
        
        if len(mode_diff) == 0:
            raise ValueError("Cannot determine frequency from time differences")
        
        typical_diff = mode_diff.iloc[0]
        
        # Map time differences to frequency strings
        if typical_diff <= pd.Timedelta(days=1):
            return "daily"
        elif typical_diff <= pd.Timedelta(days=7):
            return "weekly"
        elif typical_diff <= pd.Timedelta(days=31):
            return "monthly"
        elif typical_diff <= pd.Timedelta(days=93):
            return "quarterly"
        else:
            return "annual"