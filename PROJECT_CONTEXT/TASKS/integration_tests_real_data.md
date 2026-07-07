# Task: Integration Tests with Real Data

**Status**: TODO  
**Priority**: HIGH  
**Estimated Time**: 1-2 days  
**Dependencies**: Data layer complete ✅

## Overview

The data layer is functionally complete with 420+ tests. Now we need comprehensive integration tests using real market data to validate mathematical accuracy, performance, and edge case handling before moving to production use.

## Test Implementation Plan

### Phase 1: End-to-End Data Flow Tests (Day 1 Morning)

#### 1.1 Complete Pipeline Tests
```python
# tests/data/test_integration_complete_pipeline.py

@pytest.mark.integration
class TestCompletePipeline:
    """Test complete data flow from raw API to computed metrics."""
    
    def test_stock_total_returns_pipeline(self):
        """Verify complete flow: API → prices → dividends → total returns."""
        provider = create_production_provider()
        
        # Get Apple total returns for last month
        returns = provider.get_data(
            LogicalDataType.TOTAL_RETURN,
            date(2024, 11, 1),
            date(2024, 11, 30),
            ticker="AAPL"
        )
        
        # Verify structure
        assert isinstance(returns, pd.Series)
        assert len(returns) >= 19  # ~20 trading days
        
        # Verify reasonable values
        assert returns.abs().max() < 0.10  # No 10%+ daily moves
        assert returns.mean() > -0.05  # Not crashing
        
    def test_economic_pipeline(self):
        """Test inflation calculation from CPI data."""
        provider = create_production_provider()
        
        # Get inflation for 2023
        inflation = provider.get_data(
            LogicalDataType.INFLATION_RATE,
            date(2023, 1, 1),
            date(2023, 12, 31),
            frequency="monthly"
        )
        
        # Should have 12 monthly observations
        assert len(inflation) >= 11  # Some months might be missing
        
        # Reasonable inflation (0-10% annually)
        assert 0 <= inflation.mean() <= 0.10
```

#### 1.2 Multi-Asset Tests
```python
def test_portfolio_data_retrieval():
    """Test fetching data for a realistic portfolio."""
    provider = create_production_provider()
    tickers = ["SPY", "AGG", "GLD", "VNQ", "EFA"]
    
    # Get returns for all assets
    results = {}
    for ticker in tickers:
        results[ticker] = provider.get_data(
            LogicalDataType.TOTAL_RETURN,
            date(2024, 1, 1),
            date(2024, 11, 30),
            ticker=ticker
        )
    
    # All should have data
    assert all(len(returns) > 200 for returns in results.values())
    
    # Correlations should be reasonable
    returns_df = pd.DataFrame(results)
    corr = returns_df.corr()
    assert corr.loc["SPY", "AGG"] < 0.5  # Stocks/bonds not perfectly correlated
```

### Phase 2: Mathematical Validation (Day 1 Afternoon)

#### 2.1 Return Calculation Verification
```python
# tests/data/test_mathematical_accuracy.py

def test_dividend_adjusted_returns():
    """Verify dividend adjustment calculations against known examples."""
    # Use a high-dividend stock during ex-dividend date
    provider = create_production_provider()
    
    # Get data around known dividend date
    # Example: MSFT paid $0.75 dividend on 2024-02-14
    prices = provider.get_data(
        RawDataType.ADJUSTED_CLOSE,
        date(2024, 2, 10),
        date(2024, 2, 20),
        ticker="MSFT"
    )
    
    dividends = provider.get_data(
        RawDataType.DIVIDEND,
        date(2024, 2, 10),
        date(2024, 2, 20),
        ticker="MSFT"
    )
    
    total_returns = provider.get_data(
        LogicalDataType.TOTAL_RETURN,
        date(2024, 2, 10),
        date(2024, 2, 20),
        ticker="MSFT"
    )
    
    # Manually calculate expected return on ex-div date
    # Verify our calculation matches
```

#### 2.2 Inflation Calculation Verification
```python
def test_yoy_inflation_calculation():
    """Verify YoY inflation matches published figures."""
    provider = create_production_provider()
    
    # Get inflation for a known period
    inflation = provider.get_data(
        LogicalDataType.INFLATION_RATE,
        date(2023, 12, 1),
        date(2023, 12, 31),
        frequency="monthly"
    )
    
    # December 2023 YoY inflation was ~3.4%
    assert 0.032 <= inflation.iloc[-1] <= 0.036
```

### Phase 3: Performance Benchmarking (Day 2 Morning)

#### 3.1 Response Time Tests
```python
# tests/data/test_performance_benchmarks.py

def test_single_ticker_performance():
    """Verify performance meets targets."""
    provider = create_production_provider()
    
    start_time = time.time()
    data = provider.get_data(
        LogicalDataType.TOTAL_RETURN,
        date(2024, 1, 1),
        date(2024, 11, 30),
        ticker="AAPL"
    )
    elapsed = time.time() - start_time
    
    assert elapsed < 2.0  # Should complete in under 2 seconds
    assert len(data) > 200  # Should have full data

def test_universe_performance():
    """Test performance with multiple tickers."""
    provider = create_production_provider()
    
    start_time = time.time()
    # This should use efficient batch fetching
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
        provider.get_data(
            LogicalDataType.TOTAL_RETURN,
            date(2024, 1, 1),
            date(2024, 11, 30),
            ticker=ticker
        )
    elapsed = time.time() - start_time
    
    assert elapsed < 5.0  # 5 tickers in under 5 seconds
```

### Phase 4: Edge Cases and Error Handling (Day 2 Afternoon)

#### 4.1 Missing Data Scenarios
```python
def test_delisted_stock_handling():
    """Test handling of delisted securities."""
    provider = create_production_provider()
    
    # Try to get data for a delisted stock
    try:
        data = provider.get_data(
            LogicalDataType.TOTAL_RETURN,
            date(2024, 1, 1),
            date(2024, 11, 30),
            ticker="LEHMAN"  # Doesn't exist
        )
        # Should either return empty or raise DataNotAvailableError
        assert len(data) == 0
    except DataNotAvailableError:
        pass  # This is acceptable

def test_partial_data_handling():
    """Test handling when data partially available."""
    provider = create_production_provider()
    
    # Request data before IPO
    data = provider.get_data(
        LogicalDataType.TOTAL_RETURN,
        date(2010, 1, 1),  # Before many tech IPOs
        date(2024, 11, 30),
        ticker="FB"  # META, IPO'd in 2012
    )
    
    # Should return data from IPO date forward
    assert data.index[0].year >= 2012
```

### Phase 5: Cross-Validation Tests

#### 5.1 Compare Against Known Sources
```python
def test_cross_validate_returns():
    """Compare calculated returns against other sources."""
    provider = create_production_provider()
    
    # Get SPY returns for a specific period
    returns = provider.get_data(
        LogicalDataType.TOTAL_RETURN,
        date(2023, 1, 1),
        date(2023, 12, 31),
        ticker="SPY"
    )
    
    # Calculate annual return
    total_return = (1 + returns).prod() - 1
    
    # SPY 2023 total return was approximately 26.3%
    assert 0.25 <= total_return <= 0.27
```

## Configuration for Tests

Create `tests/data/integration_config.py`:
```python
# Configuration for integration tests
TEST_TICKERS = {
    "large_cap": ["AAPL", "MSFT", "GOOGL"],
    "etfs": ["SPY", "AGG", "GLD", "VNQ"],
    "high_dividend": ["T", "VZ", "XOM"],
    "international": ["EFA", "EEM", "FXI"]
}

TEST_PERIODS = {
    "recent": (date(2024, 10, 1), date(2024, 11, 30)),
    "full_year": (date(2023, 1, 1), date(2023, 12, 31)),
    "covid": (date(2020, 1, 1), date(2020, 12, 31))
}
```

## Success Criteria

- [ ] All end-to-end pipeline tests pass
- [ ] Mathematical calculations verified against known values
- [ ] Performance meets targets (<2s single ticker, <5s for 5 tickers)
- [ ] Edge cases handled gracefully
- [ ] Cross-validation confirms accuracy
- [ ] No unexpected API failures
- [ ] Memory usage remains reasonable
- [ ] Documentation of any discovered quirks

## Running the Tests

```bash
# Run all integration tests
pytest tests/data/test_integration_*.py -v -m integration

# Run with performance profiling
pytest tests/data/test_performance_benchmarks.py -v --profile

# Run specific test suite
pytest tests/data/test_mathematical_accuracy.py -v

# Run with coverage
pytest tests/data/ -v -m integration --cov=src.data.providers
```

## Notes

1. **API Rate Limits**: Be mindful of rate limits, especially for yfinance
2. **Market Hours**: Some tests may behave differently during market hours
3. **Data Availability**: Recent data might not be immediately available
4. **Caching**: Consider implementing caching before running extensive tests
5. **CI/CD**: Mark these tests appropriately so they don't run on every commit

## Next Steps After Integration Tests

Once integration tests pass:
1. Implement caching layer for performance
2. Add data quality monitoring
3. Create provider factory for production use
4. Begin portfolio optimization integration

The integration tests will validate that our data layer is truly production-ready!
