"""
Data module for portfolio optimizer.

This module handles all data fetching and storage operations including:
- Market data from various sources (Yahoo Finance, Alpha Vantage, etc.)
- Portfolio data import/export
- Benchmark data
- Economic indicators
- Exposure universe configuration and management
- Total return data with proper dividend handling
- FRED economic data integration
- Return estimation framework
- Multi-frequency data support with proper return compounding
"""

# Import core interfaces first (no dependencies)
from .interfaces import (
    RawDataType, LogicalDataType, DataType, Frequency as DataFrequency,
    RawDataProvider, DataProvider, QualityMonitor, CacheManager,
    QualityReport, QualityIssue,
    InvalidTickerError, InvalidDateRangeError, DataNotAvailableError,
    validate_ticker_requirement, validate_date_range
)

# Import other modules with potential dependencies
_HAS_RETURN_ESTIMATION = False
ReturnEstimationFramework = None

# Skip return_estimation import to avoid scipy compatibility issues
# TODO: Fix scipy/numpy compatibility in return_estimation.py
# try:
#     from .return_estimation import ReturnEstimationFramework
#     _HAS_RETURN_ESTIMATION = True
# except ImportError:
#     _HAS_RETURN_ESTIMATION = False
#     ReturnEstimationFramework = None

# Import other modules
from .market_data import MarketDataFetcher, calculate_returns
from .exposure_universe import ExposureUniverse, Exposure, Implementation, ExposureUniverseConfig
from .total_returns import TotalReturnFetcher
from .fred_data import FREDDataFetcher
from .return_decomposition import ReturnDecomposer
from .multi_frequency import (
    Frequency, ReturnCompounding, MultiFrequencyDataFetcher, 
    FrequencyConverter, MultiFrequencyAnalyzer
)

__all__ = [
    # Interface definitions
    "RawDataType", "LogicalDataType", "DataType", "DataFrequency",
    "RawDataProvider", "DataProvider", "QualityMonitor", "CacheManager",
    "QualityReport", "QualityIssue",
    "InvalidTickerError", "InvalidDateRangeError", "DataNotAvailableError",
    "validate_ticker_requirement", "validate_date_range",
    
    # Legacy components  
    "MarketDataFetcher", 
    "calculate_returns",
    "ExposureUniverse",
    "Exposure", 
    "Implementation",
    "ExposureUniverseConfig",
    "TotalReturnFetcher",
    "FREDDataFetcher", 
    "ReturnDecomposer",
    "Frequency",
    "ReturnCompounding",
    "MultiFrequencyDataFetcher",
    "FrequencyConverter", 
    "MultiFrequencyAnalyzer"
]

# Conditionally add return estimation framework
if _HAS_RETURN_ESTIMATION:
    __all__.append("ReturnEstimationFramework")
