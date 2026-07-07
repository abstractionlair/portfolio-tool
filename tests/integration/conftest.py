"""
Configuration for integration tests.

This file configures pytest for integration testing with real data.
"""

import pytest
import logging
import os
from datetime import datetime


def pytest_configure(config):
    """Configure pytest for integration tests."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'integration_tests_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requiring real data)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests as stress/load tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow-running tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add integration marker to all tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to stress tests
        if "stress" in str(item.name) or "sustained_load" in str(item.name):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session", autouse=True)
def integration_test_setup():
    """Set up integration test environment."""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STARTING INTEGRATION TEST SUITE")
    logger.info("="*80)
    logger.info(f"Test session started at: {datetime.now()}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check environment
    logger.info("Environment checks:")
    logger.info(f"- Python executable: {os.sys.executable}")
    logger.info(f"- Current user: {os.getenv('USER', 'unknown')}")
    
    yield
    
    logger.info("="*80)
    logger.info("INTEGRATION TEST SUITE COMPLETED")
    logger.info("="*80)


@pytest.fixture
def skip_if_no_internet():
    """Skip test if no internet connection available."""
    import socket
    try:
        # Try to connect to a reliable server
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        pytest.skip("No internet connection available")


@pytest.fixture
def skip_if_market_hours():
    """Skip test if during market hours (to avoid excessive API calls)."""
    from .config import is_market_hours
    
    if is_market_hours():
        pytest.skip("Skipping during market hours to reduce API load")


# Custom assertion helpers
def assert_reasonable_returns(returns, max_daily_return=0.2):
    """Assert that returns are reasonable."""
    import pandas as pd
    import numpy as np
    
    assert isinstance(returns, pd.Series), "Returns should be a pandas Series"
    
    if not returns.empty:
        finite_returns = returns[np.isfinite(returns)]
        if len(finite_returns) > 0:
            assert finite_returns.abs().max() < max_daily_return, \
                f"Max return {finite_returns.abs().max():.3f} exceeds {max_daily_return:.1%}"


def assert_reasonable_correlation(corr_value, min_corr=-0.8, max_corr=0.8):
    """Assert that correlation is reasonable."""
    assert min_corr <= corr_value <= max_corr, \
        f"Correlation {corr_value:.3f} outside reasonable range [{min_corr}, {max_corr}]"


# Make assertion helpers available to tests
pytest.assert_reasonable_returns = assert_reasonable_returns
pytest.assert_reasonable_correlation = assert_reasonable_correlation