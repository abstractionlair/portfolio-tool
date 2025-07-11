# Task: Implement Real Data Providers

**Status**: TODO  
**Priority**: HIGH  
**Estimated Time**: 3-4 days  
**Dependencies**: Data layer tests complete ✅

## Overview

The data layer test foundation is complete with 202 comprehensive tests. Now it's time to implement the real data providers that will fetch actual market and economic data.

## Implementation Order

### Phase 1: Raw Data Provider (Days 1-2)

#### 1.1 YFinance Integration
**File**: `src/data/providers/yfinance_source.py`
- [ ] Implement `YFinanceSource` class
- [ ] Handle OHLCV data fetching
- [ ] Handle dividend and split data
- [ ] Add retry logic and error handling
- [ ] Map frequency parameters correctly

#### 1.2 FRED Integration  
**File**: `src/data/providers/fred_source.py`
- [ ] Implement `FREDSource` class
- [ ] Map RawDataType enums to FRED series codes
- [ ] Handle rate data (Treasury, TIPS, Fed Funds)
- [ ] Handle price indices (CPI, PCE)
- [ ] Add FRED API key configuration

#### 1.3 CSV Fallback Source
**File**: `src/data/providers/csv_source.py`
- [ ] Implement `CSVSource` for manual data
- [ ] Support standard CSV format
- [ ] Date parsing and validation
- [ ] Configurable data directory

#### 1.4 Raw Provider Coordinator
**File**: `src/data/providers/raw_provider.py`
- [ ] Implement `DefaultRawDataProvider`
- [ ] Source routing logic
- [ ] Fallback handling
- [ ] Input validation
- [ ] Must pass all 202 contract tests!

### Phase 2: Transformed Data Provider (Day 3)

#### 2.1 Return Calculations
**File**: `src/data/providers/calculators/return_calculator.py`
- [ ] Simple returns from prices
- [ ] Total returns including dividends
- [ ] Log returns
- [ ] Proper frequency handling

#### 2.2 Economic Calculations
**File**: `src/data/providers/calculators/economic_calculator.py`
- [ ] Inflation rate from price indices
- [ ] Real rate calculations
- [ ] Term premium calculations

#### 2.3 Transformed Provider
**File**: `src/data/providers/transformed_provider.py`
- [ ] Implement `DefaultTransformedDataProvider`
- [ ] Route logical data types to calculators
- [ ] Delegate raw types to raw provider
- [ ] Frequency conversion logic

### Phase 3: Integration & Testing (Day 4)

#### 3.1 Provider Factory
**File**: `src/data/factory.py`
- [ ] Create production provider stack
- [ ] Configuration management
- [ ] Environment variable handling

#### 3.2 Integration Testing
- [ ] Test with real market data
- [ ] Verify FRED data access
- [ ] End-to-end data fetching
- [ ] Performance benchmarks

## Key Implementation Points

### 1. Use Existing Code Where Possible
The project already has:
- `src/data/market_data.py` - yfinance wrapper
- `src/data/fred_data.py` - FRED integration
- `src/data/total_returns.py` - return calculations

Refactor and adapt these into the new architecture.

### 2. Follow TDD Approach
Each provider must:
- Inherit from contract test classes
- Pass all applicable tests
- Add implementation-specific tests

### 3. Error Handling Strategy
```python
# Consistent error handling pattern
try:
    data = source.fetch_data(...)
except HTTPError as e:
    if e.response.status_code == 404:
        raise DataNotAvailableError(f"Ticker {ticker} not found")
    else:
        raise DataSourceError(f"API error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise DataSourceError(f"Failed to fetch data: {e}")
```

### 4. Configuration
Create `config/data_sources.yaml`:
```yaml
yfinance:
  session_timeout: 10
  max_retries: 3
  
fred:
  api_key: ${FRED_API_KEY}
  rate_limit: 120/minute
  
csv:
  data_dir: ./data/manual
  date_format: "%Y-%m-%d"
```

## Testing Strategy

### Unit Tests
```python
# Each source should have thorough unit tests
class TestYFinanceSource:
    def test_fetch_ohlcv_data(self, mock_yfinance):
        source = YFinanceSource()
        data = source.fetch_ohlcv("AAPL", start, end)
        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
```

### Contract Compliance
```python
# Providers must pass contract tests
class TestDefaultRawProvider(RawDataProviderContractTest):
    @pytest.fixture
    def provider(self):
        # Use mocked sources for fast tests
        return DefaultRawDataProvider(sources=[MockYFinance(), MockFRED()])
```

### Integration Tests
```python
# Separate integration tests with real APIs
@pytest.mark.integration
def test_real_yfinance_data():
    provider = DefaultRawDataProvider()
    data = provider.get_data(
        RawDataType.ADJUSTED_CLOSE,
        date(2024, 1, 1),
        date(2024, 1, 31),
        ticker="AAPL"
    )
    assert len(data) > 15  # Should have ~20 trading days
```

## Success Criteria

- [ ] All 202 contract tests pass with real providers
- [ ] Can fetch real market data from yfinance
- [ ] Can fetch real economic data from FRED
- [ ] CSV fallback works for missing data
- [ ] Return calculations are accurate
- [ ] Frequency conversion works correctly
- [ ] Performance targets met (<2s for single ticker)
- [ ] Clean error messages for all failure modes

## Configuration Setup

Before starting, ensure:
1. FRED API key is available (get from https://fred.stlouisfed.org/docs/api/api_key.html)
2. Test data CSVs prepared for fallback testing
3. Virtual environment has all dependencies

## Notes

1. Start with YFinance as it's most familiar
2. Use existing code as reference but adapt to new interface
3. Keep external API calls in integration tests only
4. Mock everything for unit tests
5. Document any API quirks discovered

The test foundation is solid - focus on making the implementations pass all the contract tests!
