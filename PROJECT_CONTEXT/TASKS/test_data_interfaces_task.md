# Task: Write Tests for Data Layer Interfaces

**Status**: TODO  
**Priority**: HIGH  
**Estimated Time**: 1 day  
**Dependencies**: interfaces.py exists

## Overview

Implement comprehensive tests for the data layer interfaces defined in `src/data/interfaces.py`. These tests will verify that implementations correctly follow the contracts and will serve as the specification for implementing the actual data providers.

## Test Structure

Create the following test files:

### 1. `tests/data/test_interfaces.py`

Test the interface definitions themselves:
- Enum behavior (requires_ticker property)
- Helper functions (validate_ticker_requirement, validate_date_range)
- Exception hierarchy
- Basic protocol compliance

```python
# Example tests to write:

def test_raw_data_type_requires_ticker():
    """Test that security data types require ticker."""
    assert RawDataType.OHLCV.requires_ticker == True
    assert RawDataType.DIVIDEND.requires_ticker == True
    assert RawDataType.TREASURY_3M.requires_ticker == False
    assert RawDataType.CPI_INDEX.requires_ticker == False

def test_validate_ticker_requirement():
    """Test ticker validation logic."""
    # Should pass
    validate_ticker_requirement(RawDataType.OHLCV, "AAPL")
    validate_ticker_requirement(RawDataType.TREASURY_3M, None)
    
    # Should raise
    with pytest.raises(InvalidTickerError):
        validate_ticker_requirement(RawDataType.OHLCV, None)
    
    with pytest.raises(InvalidTickerError):
        validate_ticker_requirement(RawDataType.TREASURY_3M, "AAPL")

def test_frequency_conversion_hierarchy():
    """Test frequency conversion rules."""
    assert Frequency.DAILY.can_convert_to(Frequency.MONTHLY) == True
    assert Frequency.MONTHLY.can_convert_to(Frequency.DAILY) == False
    assert Frequency.WEEKLY.can_convert_to(Frequency.WEEKLY) == True

def test_quality_score_calculation():
    """Test quality report scoring."""
    report = QualityReport(
        ticker="AAPL",
        data_type=RawDataType.OHLCV,
        check_date=datetime.now(),
        total_issues=5,
        critical_issues=1,
        warning_issues=2,
        info_issues=2,
        issues=[],
        data_points_checked=1000,
        data_points_fixed=3
    )
    
    # Score = 100 * (1 - (1*10 + 2*3 + 2*1) / 1000)
    # Score = 100 * (1 - 18/1000) = 98.2
    assert abs(report.quality_score - 98.2) < 0.01
```

### 2. `tests/data/test_data_provider_contract.py`

Test the contract that ALL DataProvider implementations must follow:

```python
class DataProviderContractTest:
    """
    Base test class that all DataProvider implementations should pass.
    This defines the contract that must be upheld.
    """
    
    @pytest.fixture
    def provider(self):
        """Subclasses must provide a DataProvider instance."""
        raise NotImplementedError
    
    def test_get_security_data_requires_ticker(self, provider):
        """Security data types must have ticker."""
        with pytest.raises(InvalidTickerError):
            provider.get_data(
                RawDataType.OHLCV,
                date(2023, 1, 1),
                date(2023, 12, 31),
                ticker=None  # Missing required ticker
            )
    
    def test_get_economic_data_forbids_ticker(self, provider):
        """Economic data types must not have ticker."""
        with pytest.raises(InvalidTickerError):
            provider.get_data(
                RawDataType.TREASURY_3M,
                date(2023, 1, 1),
                date(2023, 12, 31),
                ticker="AAPL"  # Ticker not allowed
            )
    
    def test_returns_pandas_series(self, provider):
        """Data must be returned as pandas Series."""
        # Mock or use real data
        result = provider.get_data(
            RawDataType.TREASURY_3M,
            date(2023, 1, 1),
            date(2023, 1, 31),
            ticker=None
        )
        
        assert isinstance(result, pd.Series)
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_date_range_validation(self, provider):
        """Invalid date ranges should raise error."""
        with pytest.raises(InvalidDateRangeError):
            provider.get_data(
                RawDataType.TREASURY_3M,
                date(2023, 12, 31),  # Start after end
                date(2023, 1, 1),
                ticker=None
            )
    
    def test_get_universe_data_shape(self, provider):
        """Universe data should return DataFrame with correct shape."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = provider.get_universe_data(
            RawDataType.ADJUSTED_CLOSE,
            tickers,
            date(2023, 1, 1),
            date(2023, 1, 31)
        )
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == tickers
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_is_available_returns_bool(self, provider):
        """is_available must return boolean."""
        result = provider.is_available(
            RawDataType.TREASURY_3M,
            date(2023, 1, 1),
            date(2023, 12, 31)
        )
        assert isinstance(result, bool)
    
    def test_frequency_parameter_accepted(self, provider):
        """Test both string and Frequency enum accepted."""
        # String frequency
        data1 = provider.get_data(
            RawDataType.TREASURY_3M,
            date(2023, 1, 1),
            date(2023, 12, 31),
            frequency="monthly"
        )
        
        # Enum frequency  
        data2 = provider.get_data(
            RawDataType.TREASURY_3M,
            date(2023, 1, 1),
            date(2023, 12, 31),
            frequency=Frequency.MONTHLY
        )
        
        # Both should work
        assert len(data1) > 0
        assert len(data2) > 0
```

### 3. `tests/data/test_raw_provider_contract.py`

Specific tests for RawDataProvider implementations:

```python
class RawDataProviderContractTest(DataProviderContractTest):
    """
    Contract tests specific to RawDataProvider implementations.
    """
    
    def test_rejects_logical_data_types(self, provider):
        """Raw provider should not accept LogicalDataType."""
        with pytest.raises(ValueError):
            provider.get_data(
                LogicalDataType.TOTAL_RETURN,  # Not a raw type!
                date(2023, 1, 1),
                date(2023, 12, 31),
                ticker="AAPL"
            )
    
    def test_no_computation_performed(self, provider):
        """Raw provider should not compute returns from prices."""
        # If we ask for dividends, we should get dividend amounts,
        # not computed returns
        dividends = provider.get_data(
            RawDataType.DIVIDEND,
            date(2023, 1, 1),
            date(2023, 12, 31),
            ticker="AAPL"
        )
        
        # Dividends should be amounts (e.g., 0.24 for $0.24/share)
        # not returns (e.g., 0.0024 for 0.24% return)
        assert all(d >= 0 for d in dividends.dropna())
        assert any(d > 0.01 for d in dividends.dropna())  # Some dividend > 1 cent
```

### 4. `tests/data/test_mock_providers.py`

Create mock implementations for testing:

```python
class MockRawDataProvider(RawDataProvider):
    """Mock raw data provider for testing."""
    
    def __init__(self, data_map: Dict[str, pd.Series] = None):
        self.data_map = data_map or {}
        self.call_count = 0
    
    def get_data(self, data_type, start, end, ticker=None, frequency="daily", **kwargs):
        self.call_count += 1
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # Return mock data
        key = f"{data_type.value}_{ticker or 'no_ticker'}"
        if key in self.data_map:
            return self.data_map[key]
        
        # Generate fake data
        dates = pd.date_range(start, end, freq=Frequency(frequency).pandas_freq)
        return pd.Series(
            data=np.random.randn(len(dates)),
            index=dates,
            name=ticker or data_type.value
        )
    
    def is_available(self, data_type, start, end, ticker=None, **kwargs):
        return True


class MockDataProvider:
    """Mock complete data provider for testing."""
    
    def __init__(self, raw_provider: RawDataProvider = None):
        self.raw = raw_provider or MockRawDataProvider()
    
    def get_data(self, data_type, start, end, ticker=None, frequency="daily", **kwargs):
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # Handle logical types
        if data_type == LogicalDataType.TOTAL_RETURN:
            # Get prices and compute returns
            prices = self.raw.get_data(
                RawDataType.ADJUSTED_CLOSE, start, end, ticker, frequency
            )
            return prices.pct_change().dropna()
        
        # Delegate raw types
        if isinstance(data_type, RawDataType):
            return self.raw.get_data(data_type, start, end, ticker, frequency, **kwargs)
        
        raise NotImplementedError(f"Mock doesn't handle {data_type}")
```

### 5. `tests/data/test_quality_monitor.py`

Tests for quality monitoring interface:

```python
def test_quality_monitor_contract():
    """Test quality monitor interface."""
    
    class MockQualityMonitor:
        def check_data(self, data, data_type, ticker=None):
            # Always return one issue for testing
            return QualityReport(
                ticker=ticker,
                data_type=data_type,
                check_date=datetime.now(),
                total_issues=1,
                critical_issues=0,
                warning_issues=1,
                info_issues=0,
                issues=[
                    QualityIssue(
                        severity="warning",
                        description="Test issue",
                        affected_dates=[data.index[0]],
                        can_auto_fix=True
                    )
                ],
                data_points_checked=len(data),
                data_points_fixed=0
            )
        
        def check_and_fix(self, data, data_type, ticker=None):
            report = self.check_data(data, data_type, ticker)
            # "Fix" by adding 0.001
            fixed_data = data + 0.001
            report.data_points_fixed = 1
            return fixed_data, report
    
    monitor = MockQualityMonitor()
    data = pd.Series([1, 2, 3], index=pd.date_range("2023-01-01", periods=3))
    
    # Test check_data
    report = monitor.check_data(data, RawDataType.ADJUSTED_CLOSE, "AAPL")
    assert report.total_issues == 1
    assert report.quality_score < 100
    
    # Test check_and_fix
    fixed, report = monitor.check_and_fix(data, RawDataType.ADJUSTED_CLOSE, "AAPL")
    assert (fixed == data + 0.001).all()
    assert report.data_points_fixed == 1
```

## Testing Strategy

1. **Protocol Compliance**: Use `typing_extensions.runtime_checkable` to verify protocols
2. **Contract Tests**: Base test classes that all implementations must pass
3. **Mock Implementations**: For testing composed providers
4. **Property-Based Testing**: Consider using `hypothesis` for edge cases
5. **Parametrized Tests**: Test all enum values systematically

## Success Criteria

- [ ] 100% coverage of interface validation logic
- [ ] Contract tests that any implementation can inherit
- [ ] Mock providers that can be used in other tests
- [ ] Clear documentation of expected behavior
- [ ] All tests pass with meaningful assertions

## Notes for Implementation

1. Use `pytest` as the test framework
2. Create fixtures for common test data (date ranges, mock series)
3. Use `pytest.mark.parametrize` for testing all enum values
4. Consider creating a `conftest.py` with shared fixtures
5. Mock external data sources, don't make real API calls
6. Test edge cases: empty data, single data point, missing values

## Example Test Structure

```
tests/
└── data/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures
    ├── test_interfaces.py       # Direct interface tests
    ├── test_data_provider_contract.py  # Contract all providers must pass
    ├── test_raw_provider_contract.py   # Contract for raw providers
    ├── test_mock_providers.py   # Mock implementations
    ├── test_quality_monitor.py  # Quality monitoring tests
    └── test_cache_interface.py  # Cache interface tests
```

This comprehensive test suite will ensure that all implementations correctly follow the interfaces and make it easy to verify new implementations.
