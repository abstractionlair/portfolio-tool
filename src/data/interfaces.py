"""
Data Layer Interface Specification

This module defines the complete interface for the portfolio optimizer's data layer.
These interfaces form the contracts that all implementations must follow.

The architecture supports:
- Multiple data sources with fallback
- Transparent caching
- Data quality monitoring
- Frequency conversion
- Both security and economic data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Protocol, Union
import pandas as pd


# ============================================================================
# Data Type Definitions
# ============================================================================

class DataTypeCategory(Enum):
    """Categories to help validate data requirements."""
    SECURITY_RAW = "security_raw"
    SECURITY_DERIVED = "security_derived"
    ECONOMIC_RAW = "economic_raw"
    ECONOMIC_DERIVED = "economic_derived"


class RawDataType(Enum):
    """Concrete data types that can be fetched from external sources."""
    
    # Security data (requires ticker)
    OHLCV = ("ohlcv", DataTypeCategory.SECURITY_RAW)
    ADJUSTED_CLOSE = ("adjusted_close", DataTypeCategory.SECURITY_RAW)
    DIVIDEND = ("dividend", DataTypeCategory.SECURITY_RAW)
    SPLIT = ("split", DataTypeCategory.SECURITY_RAW)
    VOLUME = ("volume", DataTypeCategory.SECURITY_RAW)
    
    # Economic data (no ticker)
    TREASURY_3M = ("treasury_3m", DataTypeCategory.ECONOMIC_RAW)
    TREASURY_6M = ("treasury_6m", DataTypeCategory.ECONOMIC_RAW)
    TREASURY_1Y = ("treasury_1y", DataTypeCategory.ECONOMIC_RAW)
    TREASURY_2Y = ("treasury_2y", DataTypeCategory.ECONOMIC_RAW)
    TREASURY_5Y = ("treasury_5y", DataTypeCategory.ECONOMIC_RAW)
    TREASURY_10Y = ("treasury_10y", DataTypeCategory.ECONOMIC_RAW)
    TREASURY_30Y = ("treasury_30y", DataTypeCategory.ECONOMIC_RAW)
    
    TIPS_5Y = ("tips_5y", DataTypeCategory.ECONOMIC_RAW)
    TIPS_10Y = ("tips_10y", DataTypeCategory.ECONOMIC_RAW)
    TIPS_30Y = ("tips_30y", DataTypeCategory.ECONOMIC_RAW)
    
    FED_FUNDS = ("fed_funds", DataTypeCategory.ECONOMIC_RAW)
    SOFR = ("sofr", DataTypeCategory.ECONOMIC_RAW)
    
    CPI_INDEX = ("cpi_index", DataTypeCategory.ECONOMIC_RAW)
    PCE_INDEX = ("pce_index", DataTypeCategory.ECONOMIC_RAW)
    
    def __init__(self, value: str, category: DataTypeCategory):
        self._value_ = value
        self.category = category
    
    @property
    def requires_ticker(self) -> bool:
        """Whether this data type requires a ticker parameter."""
        return self.category in [DataTypeCategory.SECURITY_RAW, 
                                DataTypeCategory.SECURITY_DERIVED]


class LogicalDataType(Enum):
    """Abstract data types that may require computation or interpretation."""
    
    # Security computations (requires ticker)
    TOTAL_RETURN = ("total_return", DataTypeCategory.SECURITY_DERIVED)
    SIMPLE_RETURN = ("simple_return", DataTypeCategory.SECURITY_DERIVED)
    LOG_RETURN = ("log_return", DataTypeCategory.SECURITY_DERIVED)
    EXCESS_RETURN = ("excess_return", DataTypeCategory.SECURITY_DERIVED)
    
    # Economic computations (no ticker)
    NOMINAL_RISK_FREE = ("nominal_risk_free", DataTypeCategory.ECONOMIC_DERIVED)
    REAL_RISK_FREE = ("real_risk_free", DataTypeCategory.ECONOMIC_DERIVED)
    INFLATION_RATE = ("inflation_rate", DataTypeCategory.ECONOMIC_DERIVED)
    TERM_PREMIUM = ("term_premium", DataTypeCategory.ECONOMIC_DERIVED)
    
    def __init__(self, value: str, category: DataTypeCategory):
        self._value_ = value
        self.category = category
    
    @property
    def requires_ticker(self) -> bool:
        """Whether this data type requires a ticker parameter."""
        return self.category in [DataTypeCategory.SECURITY_RAW,
                                DataTypeCategory.SECURITY_DERIVED]


# Union type for any data type
DataType = Union[RawDataType, LogicalDataType]


# ============================================================================
# Frequency Definitions
# ============================================================================

class Frequency(Enum):
    """Supported data frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    
    @property
    def pandas_freq(self) -> str:
        """Convert to pandas frequency string."""
        mapping = {
            Frequency.DAILY: "D",
            Frequency.WEEKLY: "W-FRI",
            Frequency.MONTHLY: "ME",
            Frequency.QUARTERLY: "QE",
            Frequency.ANNUAL: "YE"
        }
        return mapping[self]
    
    @property
    def yfinance_interval(self) -> str:
        """Get yfinance interval string."""
        mapping = {
            Frequency.DAILY: '1d',
            Frequency.WEEKLY: '1wk',
            Frequency.MONTHLY: '1mo',
            Frequency.QUARTERLY: '3mo',
            Frequency.ANNUAL: '1y'
        }
        return mapping[self]
    
    @property
    def annualization_factor(self) -> float:
        """Get annualization factor for this frequency."""
        mapping = {
            Frequency.DAILY: 252.0,      # Trading days per year
            Frequency.WEEKLY: 52.0,      # Weeks per year
            Frequency.MONTHLY: 12.0,     # Months per year
            Frequency.QUARTERLY: 4.0,    # Quarters per year
            Frequency.ANNUAL: 1.0        # Already annual
        }
        return mapping[self]
    
    def get_business_days_per_period(self) -> int:
        """Get typical number of business days per period."""
        mapping = {
            Frequency.DAILY: 1,
            Frequency.WEEKLY: 5,
            Frequency.MONTHLY: 21,
            Frequency.QUARTERLY: 63,
            Frequency.ANNUAL: 252
        }
        return mapping[self]
    
    def to_business_days_multiplier(self, target_freq: 'Frequency') -> float:
        """Get multiplier to convert from this frequency to target frequency."""
        return target_freq.get_business_days_per_period() / self.get_business_days_per_period()
    
    def can_convert_to(self, target: 'Frequency') -> bool:
        """Check if this frequency can be converted to target frequency."""
        # Can only convert to lower frequencies
        hierarchy = [
            Frequency.DAILY,
            Frequency.WEEKLY,
            Frequency.MONTHLY,
            Frequency.QUARTERLY,
            Frequency.ANNUAL
        ]
        try:
            source_idx = hierarchy.index(self)
            target_idx = hierarchy.index(target)
            return source_idx <= target_idx
        except ValueError:
            return False


# ============================================================================
# Core Data Provider Interface
# ============================================================================

class DataProvider(Protocol):
    """
    Core interface for data providers.
    
    This is the main interface that all data providers must implement.
    It provides a single entry point for all data retrieval.
    """
    
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
        Retrieve data of the specified type.
        
        Args:
            data_type: The type of data to retrieve
            start: Start date (inclusive)
            end: End date (inclusive)
            ticker: Security ticker (required for security data, None for economic data)
            frequency: Data frequency ('daily', 'monthly', etc. or Frequency enum)
            **kwargs: Additional parameters (e.g., tenor='3m' for risk-free rates)
            
        Returns:
            pandas Series with datetime index and requested data as values
            
        Raises:
            ValueError: If ticker requirement doesn't match data type
            DataNotAvailableError: If data cannot be retrieved
            InvalidDateRangeError: If date range is invalid
        """
        ...
    
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
        Retrieve data for multiple securities.
        
        Args:
            data_type: The type of data to retrieve
            tickers: List of security tickers
            start: Start date (inclusive)
            end: End date (inclusive)
            frequency: Data frequency
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with datetime index and tickers as columns
            
        Raises:
            ValueError: If data_type doesn't support multiple tickers
            DataNotAvailableError: If data cannot be retrieved for any ticker
        """
        ...
    
    def is_available(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Check if data is available without fetching it.
        
        Args:
            data_type: The type of data to check
            start: Start date
            end: End date
            ticker: Security ticker (if applicable)
            **kwargs: Additional parameters
            
        Returns:
            True if data is available, False otherwise
        """
        ...


# ============================================================================
# Raw Data Provider Interface
# ============================================================================

class RawDataProvider(ABC):
    """
    Abstract base class for raw data providers.
    
    This class provides raw data exactly as received from external sources,
    with no computation or transformation beyond basic cleaning.
    """
    
    @abstractmethod
    def get_data(
        self,
        data_type: RawDataType,  # Note: Only RawDataType, not LogicalDataType
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        **kwargs
    ) -> pd.Series:
        """
        Retrieve raw data from external sources.
        
        This method should only handle RawDataType values and return
        data exactly as provided by the source (with basic cleaning).
        
        Raises:
            ValueError: If data_type is not a RawDataType
            DataNotAvailableError: If data cannot be retrieved
        """
        pass
    
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
        Default implementation fetches each ticker sequentially.
        Subclasses may override for parallel fetching.
        """
        # Default implementation
        data_dict = {}
        for ticker in tickers:
            try:
                data_dict[ticker] = self.get_data(
                    data_type, start, end, ticker, frequency, **kwargs
                )
            except DataNotAvailableError:
                # Let caller decide how to handle missing tickers
                data_dict[ticker] = pd.Series(dtype=float)
        
        return pd.DataFrame(data_dict)
    
    @abstractmethod
    def is_available(
        self,
        data_type: RawDataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Check if raw data is available."""
        pass


# ============================================================================
# Exception Hierarchy
# ============================================================================

class DataError(Exception):
    """Base exception for all data-related errors."""
    pass


class DataNotAvailableError(DataError):
    """Raised when requested data is not available from any source."""
    pass


class InvalidDateRangeError(DataError):
    """Raised when date range is invalid (e.g., start > end)."""
    pass


class DataQualityError(DataError):
    """Raised when data quality checks fail."""
    pass


class InsufficientDataError(DataError):
    """Raised when some but not all requested data is available."""
    pass


class DataSourceError(DataError):
    """Raised when an external data source fails."""
    pass


class InvalidTickerError(DataError):
    """Raised when ticker is required but not provided, or vice versa."""
    pass


# ============================================================================
# Quality Monitoring Interface
# ============================================================================

@dataclass
class QualityIssue:
    """Represents a single data quality issue."""
    severity: str  # 'critical', 'warning', 'info'
    description: str
    affected_dates: List[date]
    can_auto_fix: bool = False


@dataclass
class QualityReport:
    """Summary of data quality check results."""
    ticker: Optional[str]
    data_type: DataType
    check_date: datetime
    total_issues: int
    critical_issues: int
    warning_issues: int
    info_issues: int
    issues: List[QualityIssue]
    data_points_checked: int
    data_points_fixed: int
    
    @property
    def quality_score(self) -> float:
        """Calculate quality score from 0-100."""
        if self.total_issues == 0:
            return 100.0
        
        # Weight by severity
        weighted_issues = (
            self.critical_issues * 10 +
            self.warning_issues * 3 +
            self.info_issues * 1
        )
        
        # Normalize by data points
        if self.data_points_checked == 0:
            return 0.0
            
        issue_rate = weighted_issues / self.data_points_checked
        return max(0.0, 100.0 * (1.0 - issue_rate))


class QualityMonitor(Protocol):
    """Interface for data quality monitoring."""
    
    def check_data(
        self,
        data: pd.Series,
        data_type: DataType,
        ticker: Optional[str] = None
    ) -> QualityReport:
        """
        Check data quality and return a report.
        
        Args:
            data: The data to check
            data_type: Type of data being checked
            ticker: Security ticker if applicable
            
        Returns:
            QualityReport with all issues found
        """
        ...
    
    def check_and_fix(
        self,
        data: pd.Series,
        data_type: DataType,
        ticker: Optional[str] = None
    ) -> tuple[pd.Series, QualityReport]:
        """
        Check data quality and attempt to fix issues.
        
        Args:
            data: The data to check and fix
            data_type: Type of data being checked
            ticker: Security ticker if applicable
            
        Returns:
            Tuple of (fixed_data, quality_report)
        """
        ...


# ============================================================================
# Cache Interface
# ============================================================================

class CacheManager(Protocol):
    """Interface for cache management."""
    
    def get(self, key: str) -> Optional[pd.Series]:
        """Retrieve data from cache if available."""
        ...
    
    def set(self, key: str, data: pd.Series, ttl: Optional[int] = None) -> None:
        """Store data in cache with optional time-to-live."""
        ...
    
    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern. Returns count deleted."""
        ...
    
    def clear(self) -> None:
        """Clear entire cache."""
        ...


# ============================================================================
# Helper Functions
# ============================================================================

def validate_ticker_requirement(data_type: DataType, ticker: Optional[str]) -> None:
    """
    Validate that ticker parameter matches data type requirements.
    
    Raises:
        InvalidTickerError: If requirements don't match
    """
    if data_type.requires_ticker and not ticker:
        raise InvalidTickerError(
            f"{data_type.name} requires a ticker but none was provided"
        )
    elif not data_type.requires_ticker and ticker is not None:
        raise InvalidTickerError(
            f"{data_type.name} does not accept a ticker but '{ticker}' was provided"
        )


def validate_date_range(start: date, end: date) -> None:
    """
    Validate that date range is valid.
    
    Raises:
        InvalidDateRangeError: If date range is invalid
    """
    if start > end:
        raise InvalidDateRangeError(
            f"Start date {start} is after end date {end}"
        )
    
    if end > date.today():
        # This might be okay for some use cases, but warn
        pass  # Or raise warning
