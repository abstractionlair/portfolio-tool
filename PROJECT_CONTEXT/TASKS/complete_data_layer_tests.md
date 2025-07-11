# Task: Complete Data Layer Test Suite

**Status**: IN PROGRESS  
**Priority**: HIGH - Must complete before implementation  
**Estimated Time**: 1 day  
**Dependencies**: Initial tests created, needs completion

## Overview

Claude Code has created an excellent foundation for the data layer tests. This task completes the test suite by adding the missing components and ensuring all tests pass with mock implementations.

## Missing Components to Implement

### 1. Mock Providers (`test_mock_providers.py`)

Create comprehensive mock implementations that can be used throughout testing:

```python
# tests/data/test_mock_providers.py

class MockRawDataProvider(RawDataProvider):
    """
    Mock raw data provider for testing.
    
    Features:
    - Configurable data responses
    - Call tracking
    - Controllable failures
    - No external dependencies
    """
    
    def __init__(self, data_map: Dict[str, pd.Series] = None):
        self.data_map = data_map or {}
        self.call_count = 0
        self.call_history = []
        self._available_types = set()
        self._should_fail = False
    
    def get_data(self, data_type, start, end, ticker=None, frequency="daily", **kwargs):
        # Track calls
        self.call_count += 1
        self.call_history.append({
            'data_type': data_type,
            'start': start,
            'end': end,
            'ticker': ticker,
            'frequency': frequency,
            'kwargs': kwargs
        })
        
        # Validate inputs
        if not isinstance(data_type, RawDataType):
            raise ValueError(f"Raw provider only handles RawDataType, got {type(data_type)}")
        
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # Simulate failures if configured
        if self._should_fail:
            raise DataNotAvailableError("Simulated failure")
        
        # Return configured data or generate
        key = f"{data_type.value}:{ticker or 'none'}"
        if key in self.data_map:
            return self._filter_to_date_range(self.data_map[key], start, end)
        
        # Generate realistic fake data
        return self._generate_fake_data(data_type, start, end, ticker, frequency)


class MockDataProvider:
    """
    Complete mock data provider supporting both raw and logical types.
    
    This mock handles:
    - All computation for logical types
    - Delegation to raw provider
    - Configurable behavior
    """
    
    def __init__(self, raw_provider: RawDataProvider = None):
        self.raw = raw_provider or MockRawDataProvider()
        self.transform_call_count = 0
    
    def get_data(self, data_type, start, end, ticker=None, frequency="daily", **kwargs):
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # Handle logical types with computation
        if isinstance(data_type, LogicalDataType):
            self.transform_call_count += 1
            return self._compute_logical_type(data_type, start, end, ticker, frequency, **kwargs)
        
        # Delegate raw types
        return self.raw.get_data(data_type, start, end, ticker, frequency, **kwargs)


class ConfigurableMockProvider:
    """
    Highly configurable mock for specific test scenarios.
    
    Features:
    - Set specific return values
    - Configure delays
    - Simulate various failure modes
    - Track detailed call patterns
    """
    
    def __init__(self):
        self.responses = {}
        self.delays = {}
        self.failure_modes = {}
        self.call_log = []
    
    def configure_response(self, data_type, ticker, data):
        """Configure specific response for data_type/ticker combination."""
        key = (data_type, ticker)
        self.responses[key] = data
    
    def configure_failure(self, data_type, ticker, exception):
        """Configure specific failure for data_type/ticker combination."""
        key = (data_type, ticker)
        self.failure_modes[key] = exception
```

Also add tests for the mocks themselves:

```python
def test_mock_raw_provider_validates_inputs():
    """Mock should validate inputs just like real provider."""
    mock = MockRawDataProvider()
    
    # Should validate ticker requirement
    with pytest.raises(InvalidTickerError):
        mock.get_data(RawDataType.OHLCV, date(2023, 1, 1), date(2023, 1, 31), ticker=None)
    
    # Should reject logical types
    with pytest.raises(ValueError):
        mock.get_data(LogicalDataType.TOTAL_RETURN, date(2023, 1, 1), date(2023, 1, 31), ticker="AAPL")

def test_mock_call_tracking():
    """Mock should track calls for verification."""
    mock = MockRawDataProvider()
    
    mock.get_data(RawDataType.TREASURY_3M, date(2023, 1, 1), date(2023, 1, 31))
    
    assert mock.call_count == 1
    assert mock.call_history[0]['data_type'] == RawDataType.TREASURY_3M

def test_configurable_mock_responses():
    """Mock should return configured responses."""
    mock = MockRawDataProvider()
    test_data = pd.Series([1, 2, 3], index=pd.date_range("2023-01-01", periods=3))
    
    mock.data_map["treasury_3m:none"] = test_data
    
    result = mock.get_data(RawDataType.TREASURY_3M, date(2023, 1, 1), date(2023, 1, 31))
    pd.testing.assert_series_equal(result, test_data)
```

### 2. Quality Monitor Contract Tests (`test_quality_monitor.py`)

Complete the quality monitor tests:

```python
# tests/data/test_quality_monitor.py

from src.data.interfaces import QualityMonitor, QualityReport, QualityIssue

class QualityMonitorContractTest:
    """
    Base test class for QualityMonitor implementations.
    
    All QualityMonitor implementations should pass these tests.
    """
    
    @pytest.fixture
    def monitor(self):
        """Subclasses must provide a QualityMonitor instance."""
        raise NotImplementedError
    
    def test_check_data_returns_quality_report(self, monitor):
        """check_data must return a QualityReport."""
        data = pd.Series([100, 101, 102], index=pd.date_range("2023-01-01", periods=3))
        
        report = monitor.check_data(data, RawDataType.ADJUSTED_CLOSE, "AAPL")
        
        assert isinstance(report, QualityReport)
        assert report.ticker == "AAPL"
        assert report.data_type == RawDataType.ADJUSTED_CLOSE
        assert report.data_points_checked == 3
    
    def test_check_and_fix_returns_tuple(self, monitor):
        """check_and_fix must return (Series, QualityReport) tuple."""
        data = pd.Series([100, 101, 102], index=pd.date_range("2023-01-01", periods=3))
        
        fixed_data, report = monitor.check_and_fix(data, RawDataType.ADJUSTED_CLOSE, "AAPL")
        
        assert isinstance(fixed_data, pd.Series)
        assert isinstance(report, QualityReport)
        assert len(fixed_data) >= len(data)  # Might add data points


class MockQualityMonitor:
    """Mock quality monitor for testing."""
    
    def __init__(self, issues_to_generate=None):
        self.issues_to_generate = issues_to_generate or []
        self.check_count = 0
        self.fix_count = 0
    
    def check_data(self, data, data_type, ticker=None):
        self.check_count += 1
        
        return QualityReport(
            ticker=ticker,
            data_type=data_type,
            check_date=datetime.now(),
            total_issues=len(self.issues_to_generate),
            critical_issues=sum(1 for i in self.issues_to_generate if i.severity == "critical"),
            warning_issues=sum(1 for i in self.issues_to_generate if i.severity == "warning"),
            info_issues=sum(1 for i in self.issues_to_generate if i.severity == "info"),
            issues=self.issues_to_generate,
            data_points_checked=len(data),
            data_points_fixed=0
        )
    
    def check_and_fix(self, data, data_type, ticker=None):
        self.fix_count += 1
        report = self.check_data(data, data_type, ticker)
        
        # Simulate fixing by adding a small amount
        fixed_data = data.copy()
        if self.issues_to_generate:
            fixed_data = fixed_data + 0.001
            report.data_points_fixed = len(self.issues_to_generate)
        
        return fixed_data, report


# Tests for specific quality checks
def test_extreme_value_detection():
    """Test detection of extreme values."""
    monitor = MockQualityMonitor([
        QualityIssue(
            severity="critical",
            description="Extreme return: 50%",
            affected_dates=[date(2023, 1, 2)],
            can_auto_fix=False
        )
    ])
    
    data = pd.Series([100, 150, 100], index=pd.date_range("2023-01-01", periods=3))
    report = monitor.check_data(data, LogicalDataType.SIMPLE_RETURN, "TEST")
    
    assert report.critical_issues == 1
    assert report.quality_score < 100

def test_missing_data_detection():
    """Test detection of missing data gaps."""
    monitor = MockQualityMonitor([
        QualityIssue(
            severity="warning",
            description="Missing data gap: 5 days",
            affected_dates=[date(2023, 1, 5), date(2023, 1, 10)],
            can_auto_fix=True
        )
    ])
    
    # Create data with gap
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    data = pd.Series(range(10), index=dates)
    data = data.drop(dates[4:9])  # Remove middle dates
    
    report = monitor.check_data(data, RawDataType.ADJUSTED_CLOSE, "TEST")
    
    assert report.warning_issues == 1
    assert any("Missing data gap" in issue.description for issue in report.issues)
```

### 3. Cache Interface Tests (`test_cache_interface.py`)

Complete the cache tests:

```python
# tests/data/test_cache_interface.py

from src.data.interfaces import CacheManager

class CacheManagerContractTest:
    """
    Base test class for CacheManager implementations.
    """
    
    @pytest.fixture
    def cache(self):
        """Subclasses must provide a CacheManager instance."""
        raise NotImplementedError
    
    def test_get_returns_none_for_missing_key(self, cache):
        """get() should return None for missing keys."""
        result = cache.get("nonexistent_key")
        assert result is None
    
    def test_set_and_get_roundtrip(self, cache):
        """Data stored with set() should be retrievable with get()."""
        test_data = pd.Series([1, 2, 3], index=pd.date_range("2023-01-01", periods=3))
        
        cache.set("test_key", test_data)
        retrieved = cache.get("test_key")
        
        pd.testing.assert_series_equal(retrieved, test_data)
    
    def test_ttl_expiration(self, cache):
        """Data should expire after TTL."""
        import time
        
        test_data = pd.Series([1, 2, 3])
        cache.set("ttl_key", test_data, ttl=1)  # 1 second TTL
        
        # Should be available immediately
        assert cache.get("ttl_key") is not None
        
        # Should expire after TTL
        time.sleep(1.5)
        assert cache.get("ttl_key") is None
    
    def test_invalidate_pattern(self, cache):
        """invalidate() should remove matching keys."""
        # Set multiple keys
        cache.set("prefix:key1", pd.Series([1]))
        cache.set("prefix:key2", pd.Series([2]))
        cache.set("other:key3", pd.Series([3]))
        
        # Invalidate pattern
        count = cache.invalidate("prefix:*")
        
        assert count >= 2
        assert cache.get("prefix:key1") is None
        assert cache.get("prefix:key2") is None
        assert cache.get("other:key3") is not None
    
    def test_clear_removes_all(self, cache):
        """clear() should remove all cached data."""
        cache.set("key1", pd.Series([1]))
        cache.set("key2", pd.Series([2]))
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class MockCacheManager:
    """Mock cache for testing."""
    
    def __init__(self):
        self.store = {}
        self.expiry_times = {}
        self.get_count = 0
        self.set_count = 0
    
    def get(self, key: str) -> Optional[pd.Series]:
        self.get_count += 1
        
        # Check expiry
        if key in self.expiry_times:
            if datetime.now() > self.expiry_times[key]:
                del self.store[key]
                del self.expiry_times[key]
                return None
        
        return self.store.get(key)
    
    def set(self, key: str, data: pd.Series, ttl: Optional[int] = None):
        self.set_count += 1
        self.store[key] = data.copy()  # Copy to avoid reference issues
        
        if ttl:
            self.expiry_times[key] = datetime.now() + timedelta(seconds=ttl)
    
    def invalidate(self, pattern: str) -> int:
        import fnmatch
        keys_to_remove = [k for k in self.store.keys() if fnmatch.fnmatch(k, pattern)]
        
        for key in keys_to_remove:
            del self.store[key]
            self.expiry_times.pop(key, None)
        
        return len(keys_to_remove)
    
    def clear(self):
        self.store.clear()
        self.expiry_times.clear()


def test_mock_cache_manager():
    """Test the mock cache implementation."""
    cache = MockCacheManager()
    test = CacheManagerContractTest()
    
    # Mock should pass all contract tests
    test.test_get_returns_none_for_missing_key(cache)
    test.test_set_and_get_roundtrip(cache)
    test.test_clear_removes_all(cache)
```

### 4. Integration Test File

Create an integration test that uses all mocks together:

```python
# tests/data/test_integration.py

def test_full_provider_stack_with_mocks():
    """Test the full provider stack using mocks."""
    # Create the stack
    raw = MockRawDataProvider()
    transformed = MockDataProvider(raw)
    cache = MockCacheManager()
    cached = CachedDataProvider(transformed, cache)  # Will need to import/create this
    monitor = MockQualityMonitor()
    final = QualityAwareDataProvider(cached, monitor)  # Will need to import/create this
    
    # Make a request
    result = final.get_data(
        LogicalDataType.TOTAL_RETURN,
        date(2023, 1, 1),
        date(2023, 1, 31),
        ticker="AAPL"
    )
    
    # Verify the chain worked
    assert isinstance(result, pd.Series)
    assert raw.call_count > 0  # Raw provider was called
    assert cache.get_count > 0  # Cache was checked
    assert monitor.check_count > 0  # Quality was checked

def test_mock_provider_inheritance():
    """Verify mock providers pass contract tests."""
    
    class TestMockRawProvider(RawDataProviderContractTest):
        @pytest.fixture
        def provider(self):
            return MockRawDataProvider()
    
    class TestMockDataProvider(DataProviderContractTest):
        @pytest.fixture 
        def provider(self):
            return MockDataProvider()
    
    # The test classes should be instantiable and runnable
    # This verifies our mocks implement the contracts correctly
```

## Additional Fixtures to Add

Add to `conftest.py`:

```python
@pytest.fixture
def all_raw_data_types():
    """List of all RawDataType enum values."""
    return list(RawDataType)

@pytest.fixture
def all_logical_data_types():
    """List of all LogicalDataType enum values."""
    return list(LogicalDataType)

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
    return MockRawDataProvider()

@pytest.fixture
def mock_cache():
    """Configured mock cache."""
    return MockCacheManager()

@pytest.fixture
def mock_quality_monitor():
    """Configured mock quality monitor."""
    return MockQualityMonitor()
```

## Success Criteria

- [ ] All mock implementations created and tested
- [ ] Mock providers pass their respective contract tests
- [ ] Quality monitor contract tests complete
- [ ] Cache manager contract tests complete
- [ ] Integration test demonstrates full stack
- [ ] All tests pass with `pytest tests/data/`
- [ ] Test coverage > 95% for interfaces module
- [ ] Clear documentation in test docstrings

## Testing the Tests

Run these commands to verify:

```bash
# Run all data layer tests
pytest tests/data/ -v

# Check coverage
pytest tests/data/ --cov=src.data.interfaces --cov-report=html

# Run specific contract tests
pytest tests/data/test_data_provider_contract.py -v

# Verify mocks work
pytest tests/data/test_mock_providers.py -v
```

## Notes for Implementation

1. Focus on making mocks realistic but fast
2. Ensure mocks validate inputs just like real providers
3. Add helper methods to mocks for easy test setup
4. Use the mocks to test the composed providers later
5. Make sure all contract tests can be inherited
6. Document mock behavior clearly

## Next Steps

After completing this task:
1. All tests should pass with mock implementations
2. Ready to implement real providers using TDD
3. Each real provider should inherit from contract tests
4. Use mocks to test composed providers

This completes the test suite foundation before moving to implementation.
