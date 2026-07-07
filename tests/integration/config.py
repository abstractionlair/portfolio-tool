"""
Configuration for integration tests.

This module provides test data, benchmarks, and utility functions
for integration testing with real market data.
"""

from datetime import date
from typing import Dict, List, Tuple


# Test asset universes for different scenarios
TEST_TICKERS = {
    "large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "etfs": ["SPY", "AGG", "GLD", "VNQ", "EFA"],
    "high_dividend": ["T", "VZ", "XOM", "CVX", "JNJ"],
    "international": ["EFA", "EEM", "FXI", "VGK", "VWO"],
    "sectors": ["XLF", "XLK", "XLE", "XLV", "XLI"],
    "small_sample": ["AAPL", "MSFT", "SPY"]  # For quick tests
}

# Test time periods with known characteristics
TEST_PERIODS = {
    "recent": (date(2024, 10, 1), date(2024, 11, 30)),
    "full_year_2023": (date(2023, 1, 1), date(2023, 12, 31)),
    "covid_crash": (date(2020, 2, 1), date(2020, 4, 30)),
    "covid_recovery": (date(2020, 4, 1), date(2020, 12, 31)),
    "rate_hikes": (date(2022, 1, 1), date(2022, 12, 31)),
    "short_term": (date(2024, 11, 1), date(2024, 11, 30))
}

# Known benchmarks for validation
KNOWN_BENCHMARKS = {
    # SPY 2023 performance (approximate)
    "SPY_2023_return": 0.263,  # ~26.3% total return
    "SPY_2023_volatility": 0.12,  # ~12% annualized volatility
    
    # December 2023 inflation rate (YoY)
    "inflation_2023_12": 0.034,  # ~3.4% YoY inflation
    
    # COVID crash magnitude
    "SPY_covid_drawdown": -0.30,  # ~30% peak-to-trough
    
    # Treasury rates 2023 year-end
    "treasury_3m_2023": 0.052,  # ~5.2%
    "treasury_10y_2023": 0.043,  # ~4.3%
}

# Performance targets
PERFORMANCE_TARGETS = {
    "single_ticker_max_time": 2.0,  # seconds
    "five_tickers_max_time": 5.0,   # seconds
    "universe_max_time": 10.0,      # seconds for 20+ tickers
    "memory_limit_mb": 500,         # MB maximum memory usage
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    "max_daily_return": 0.15,      # 15% max daily return
    "min_trading_days_per_year": 250,  # Minimum trading days
    "max_missing_data_pct": 0.05,   # 5% max missing data
    "inflation_range": (0.0, 0.10), # 0-10% annual inflation
    "risk_free_range": (0.0, 0.08), # 0-8% risk-free rate
}

# Known dividend dates for testing
DIVIDEND_TEST_CASES = {
    "AAPL": {
        "ex_date": date(2024, 2, 9),
        "amount": 0.24,
        "test_period": (date(2024, 2, 5), date(2024, 2, 15))
    },
    "MSFT": {
        "ex_date": date(2024, 2, 14),
        "amount": 0.75,
        "test_period": (date(2024, 2, 10), date(2024, 2, 20))
    },
    "SPY": {
        "ex_date": date(2024, 3, 15),
        "amount": 1.57,  # Quarterly dividend
        "test_period": (date(2024, 3, 10), date(2024, 3, 20))
    }
}

# Economic data test cases
ECONOMIC_TEST_CASES = {
    "inflation_2023": {
        "period": (date(2023, 1, 1), date(2023, 12, 31)),
        "expected_range": (0.03, 0.06),  # 3-6% range
        "method": "yoy"
    },
    "treasury_rates_2023": {
        "period": (date(2023, 12, 1), date(2023, 12, 31)),
        "expected_3m": (0.050, 0.055),  # 5.0-5.5%
        "expected_10y": (0.040, 0.045)  # 4.0-4.5%
    }
}


def get_test_tickers(category: str = "small_sample") -> List[str]:
    """Get test tickers for a specific category."""
    return TEST_TICKERS.get(category, TEST_TICKERS["small_sample"])


def get_test_period(period_name: str = "recent") -> Tuple[date, date]:
    """Get test period dates."""
    return TEST_PERIODS.get(period_name, TEST_PERIODS["recent"])


def get_benchmark(benchmark_name: str) -> float:
    """Get known benchmark value."""
    return KNOWN_BENCHMARKS.get(benchmark_name)


def get_performance_target(target_name: str) -> float:
    """Get performance target."""
    return PERFORMANCE_TARGETS.get(target_name)


def get_quality_threshold(threshold_name: str):
    """Get data quality threshold."""
    return QUALITY_THRESHOLDS.get(threshold_name)


def is_market_hours() -> bool:
    """Check if current time is during market hours (approximate)."""
    from datetime import datetime
    import pytz
    
    # Simple check - market is open 9:30 AM - 4:00 PM ET on weekdays
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    # Weekend check
    if now_et.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Time check (approximate)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_et <= market_close


def create_production_provider():
    """Create a production-ready data provider for testing."""
    from src.data.providers import RawDataProviderCoordinator, TransformedDataProvider
    
    # Create coordinator (it creates providers internally)
    raw_coordinator = RawDataProviderCoordinator()
    
    # Create transformed provider
    transformed_provider = TransformedDataProvider(raw_coordinator)
    
    return transformed_provider