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

from .market_data import MarketDataFetcher, calculate_returns
from .exposure_universe import ExposureUniverse, Exposure, Implementation, ExposureUniverseConfig
from .total_returns import TotalReturnFetcher
from .fred_data import FREDDataFetcher
from .return_estimation import ReturnEstimationFramework
from .return_decomposition import ReturnDecomposer
from .multi_frequency import (
    Frequency, ReturnCompounding, MultiFrequencyDataFetcher, 
    FrequencyConverter, MultiFrequencyAnalyzer
)

__all__ = [
    "MarketDataFetcher", 
    "calculate_returns",
    "ExposureUniverse",
    "Exposure", 
    "Implementation",
    "ExposureUniverseConfig",
    "TotalReturnFetcher",
    "FREDDataFetcher", 
    "ReturnEstimationFramework",
    "ReturnDecomposer",
    "Frequency",
    "ReturnCompounding",
    "MultiFrequencyDataFetcher",
    "FrequencyConverter", 
    "MultiFrequencyAnalyzer"
]
