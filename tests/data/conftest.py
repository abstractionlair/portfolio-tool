"""
Shared fixtures for data layer tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Optional

from src.data import (
    RawDataType, LogicalDataType, DataType, DataFrequency as Frequency,
    QualityReport, QualityIssue, InvalidTickerError, 
    DataNotAvailableError
)


@pytest.fixture
def sample_date_range():
    """Standard date range for testing."""
    return date(2023, 1, 1), date(2023, 12, 31)


@pytest.fixture
def short_date_range():
    """Short date range for quick tests."""
    return date(2023, 1, 1), date(2023, 1, 31)


@pytest.fixture
def sample_daily_dates():
    """Daily date range as pandas DatetimeIndex."""
    return pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")


@pytest.fixture
def sample_monthly_dates():
    """Monthly date range as pandas DatetimeIndex."""
    return pd.date_range(start="2023-01-01", end="2023-12-31", freq="ME")


@pytest.fixture
def sample_price_series(sample_daily_dates):
    """Sample price data for testing."""
    # Generate realistic price movements starting at $100
    np.random.seed(42)  # Reproducible
    returns = np.random.normal(0.001, 0.02, len(sample_daily_dates))
    prices = 100 * np.exp(returns.cumsum())
    
    return pd.Series(
        data=prices,
        index=sample_daily_dates,
        name="AAPL"
    )


@pytest.fixture
def sample_return_series(sample_price_series):
    """Sample return data computed from prices."""
    return sample_price_series.pct_change().dropna()


@pytest.fixture
def sample_treasury_series(sample_monthly_dates):
    """Sample treasury rate data."""
    # 3-month treasury around 2-5% range
    np.random.seed(123)
    rates = 0.03 + 0.02 * np.random.randn(len(sample_monthly_dates))
    rates = np.clip(rates, 0.001, 0.1)  # Keep reasonable
    
    return pd.Series(
        data=rates,
        index=sample_monthly_dates,
        name="3M_TREASURY"
    )


@pytest.fixture
def sample_universe_data(sample_daily_dates):
    """Sample data for multiple tickers."""
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data = {}
    
    np.random.seed(456)
    for i, ticker in enumerate(tickers):
        # Each ticker starts at different price, similar vol
        base_price = 100 + i * 50
        returns = np.random.normal(0.001, 0.02, len(sample_daily_dates))
        prices = base_price * np.exp(returns.cumsum())
        data[ticker] = prices
    
    return pd.DataFrame(data, index=sample_daily_dates)


@pytest.fixture
def sample_quality_report():
    """Sample quality report for testing."""
    return QualityReport(
        ticker="AAPL",
        data_type=RawDataType.ADJUSTED_CLOSE,
        check_date=datetime.now(),
        total_issues=3,
        critical_issues=1,
        warning_issues=1,
        info_issues=1,
        issues=[
            QualityIssue(
                severity="critical",
                description="Missing data for 5 consecutive days",
                affected_dates=[date(2023, 1, 15), date(2023, 1, 16)],
                can_auto_fix=False
            ),
            QualityIssue(
                severity="warning", 
                description="Price spike detected",
                affected_dates=[date(2023, 2, 1)],
                can_auto_fix=True
            ),
            QualityIssue(
                severity="info",
                description="Volume below average",
                affected_dates=[date(2023, 3, 1)],
                can_auto_fix=False
            )
        ],
        data_points_checked=100,
        data_points_fixed=1
    )


@pytest.fixture
def all_raw_data_types():
    """All RawDataType enum values for parametrized tests."""
    return list(RawDataType)


@pytest.fixture 
def all_logical_data_types():
    """All LogicalDataType enum values for parametrized tests."""
    return list(LogicalDataType)


@pytest.fixture
def all_frequencies():
    """All Frequency enum values for parametrized tests."""
    return list(Frequency)


@pytest.fixture
def security_data_types():
    """Data types that require tickers."""
    return [dt for dt in RawDataType if dt.requires_ticker] + \
           [dt for dt in LogicalDataType if dt.requires_ticker]


@pytest.fixture  
def economic_data_types():
    """Data types that forbid tickers."""
    return [dt for dt in RawDataType if not dt.requires_ticker] + \
           [dt for dt in LogicalDataType if not dt.requires_ticker]


@pytest.fixture
def sample_returns_series(sample_daily_dates):
    """Sample return data for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(sample_daily_dates))
    return pd.Series(returns, index=sample_daily_dates, name="returns")


@pytest.fixture
def sample_quality_issues():
    """Sample quality issues for testing."""
    return [
        QualityIssue(
            severity="warning",
            description="Missing data",
            affected_dates=[date(2023, 1, 5)],
            can_auto_fix=True
        ),
        QualityIssue(
            severity="critical",
            description="Extreme value",
            affected_dates=[date(2023, 1, 10)],
            can_auto_fix=False
        )
    ]


@pytest.fixture
def mock_raw_provider():
    """Configured mock raw provider."""
    from .test_mock_providers import MockRawDataProvider
    return MockRawDataProvider()


@pytest.fixture
def mock_cache():
    """Configured mock cache."""
    from .test_cache_interface import MockCacheManager
    return MockCacheManager()


@pytest.fixture
def mock_quality_monitor():
    """Configured mock quality monitor."""
    from .test_quality_monitor import MockQualityMonitor
    return MockQualityMonitor()


class MockException(Exception):
    """Exception for testing error handling."""
    pass