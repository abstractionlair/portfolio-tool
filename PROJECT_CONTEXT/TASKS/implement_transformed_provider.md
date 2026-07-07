# Task: Implement Transformed Data Provider

**Status**: TODO  
**Priority**: HIGH  
**Estimated Time**: 1-2 days  
**Dependencies**: Raw data providers complete ✅

## Overview

The raw data providers are successfully implemented and passing all contract tests. Now we need to implement the TransformedDataProvider that computes derived data types (returns, inflation rates, etc.) from raw data.

## Implementation Plan

### 1. Return Calculator (`src/data/providers/calculators/return_calculator.py`)

Create calculations for various return types:

```python
class ReturnCalculator:
    """Calculates various types of returns from price and dividend data."""
    
    def calculate_simple_returns(
        self, prices: pd.Series, frequency: str = "daily"
    ) -> pd.Series:
        """Simple returns: (P_t - P_{t-1}) / P_{t-1}"""
        
    def calculate_total_returns(
        self, prices: pd.Series, dividends: Optional[pd.Series] = None
    ) -> pd.Series:
        """Total returns including dividends: (P_t + D_t) / P_{t-1} - 1"""
        
    def calculate_log_returns(
        self, prices: pd.Series
    ) -> pd.Series:
        """Log returns: ln(P_t / P_{t-1})"""
        
    def calculate_excess_returns(
        self, returns: pd.Series, risk_free_rate: pd.Series
    ) -> pd.Series:
        """Excess returns: R_t - RF_t"""
```

Key points:
- Handle missing dividends gracefully
- Align dividend dates with price dates
- Proper handling of frequency for return calculations
- Account for first observation (will be NaN)

### 2. Economic Calculator (`src/data/providers/calculators/economic_calculator.py`)

Create calculations for economic indicators:

```python
class EconomicCalculator:
    """Calculates economic indicators from raw data."""
    
    def calculate_inflation_rate(
        self, price_index: pd.Series, method: str = "yoy"
    ) -> pd.Series:
        """Calculate inflation from price indices (CPI/PCE)"""
        
    def calculate_real_rate(
        self, nominal_rate: pd.Series, inflation_rate: pd.Series
    ) -> pd.Series:
        """Real rate using Fisher equation: (1+nominal)/(1+inflation) - 1"""
        
    def calculate_term_premium(
        self, long_rate: pd.Series, short_rate: pd.Series
    ) -> pd.Series:
        """Term premium: long_rate - short_rate"""
        
    def select_risk_free_rate(
        self, available_rates: Dict[str, pd.Series], tenor: str = "3m"
    ) -> pd.Series:
        """Select appropriate risk-free rate based on tenor"""
```

Key points:
- YoY inflation calculation needs 12-month lookback
- Handle different inflation calculation methods (YoY, MoM annualized)
- Fisher equation for real rates
- Smart selection of risk-free rate based on available data

### 3. Transformed Provider (`src/data/providers/transformed_provider.py`)

Main provider that orchestrates calculations:

```python
class TransformedDataProvider:
    """Provides both raw and computed data types."""
    
    def __init__(self, raw_provider: RawDataProvider):
        self.raw_provider = raw_provider
        self.return_calculator = ReturnCalculator()
        self.economic_calculator = EconomicCalculator()
    
    def get_data(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        **kwargs
    ) -> pd.Series:
        # Route based on data type
        if isinstance(data_type, RawDataType):
            # Pass through to raw provider
            return self.raw_provider.get_data(...)
        elif isinstance(data_type, LogicalDataType):
            # Compute the derived data
            return self._compute_logical_data(...)
```

Key implementation methods:
- `_compute_total_returns()` - Fetch prices and dividends, calculate
- `_compute_inflation_rate()` - Fetch CPI/PCE, calculate YoY
- `_compute_nominal_risk_free()` - Select appropriate treasury rate
- `_compute_real_risk_free()` - Calculate from nominal and inflation
- `_extend_date_range()` - Get extra data for calculations

### 4. Frequency Conversion (`src/data/providers/calculators/frequency_converter.py`)

Handle proper frequency conversion for returns:

```python
class FrequencyConverter:
    """Converts return frequencies with proper compounding."""
    
    def convert_returns(
        self,
        returns: pd.Series,
        from_frequency: str,
        to_frequency: str
    ) -> pd.Series:
        """Convert return series between frequencies"""
        
    def can_convert(self, from_freq: str, to_freq: str) -> bool:
        """Check if conversion is possible (only downsampling)"""
```

Key points:
- Returns must be compounded, not averaged
- Only allow downsampling (daily → monthly), not upsampling
- Handle different return types appropriately

## Testing Strategy

### Unit Tests (`tests/data/test_transformed_provider.py`)

```python
class TestTransformedProvider(DataProviderContractTest):
    """Test that TransformedProvider passes all contracts."""
    
    @pytest.fixture
    def provider(self):
        raw = MockRawDataProvider(...)  # Configure with test data
        return TransformedDataProvider(raw)

class TestReturnCalculations:
    """Test specific return calculation logic."""
    
    def test_total_return_includes_dividends(self):
        prices = pd.Series([100, 102, 101])
        dividends = pd.Series([0, 1, 0])
        returns = calculator.calculate_total_returns(prices, dividends)
        # First return: (102 + 1) / 100 - 1 = 0.03
        assert abs(returns.iloc[0] - 0.03) < 0.0001
    
    def test_log_returns(self):
        prices = pd.Series([100, 110])
        log_returns = calculator.calculate_log_returns(prices)
        # ln(110/100) = ln(1.1) ≈ 0.0953
        assert abs(log_returns.iloc[0] - 0.0953) < 0.0001
```

### Integration Tests

```python
@pytest.mark.integration
def test_real_total_returns():
    """Test with real market data."""
    coordinator = RawDataProviderCoordinator()
    provider = TransformedDataProvider(coordinator)
    
    returns = provider.get_data(
        LogicalDataType.TOTAL_RETURN,
        date(2024, 1, 1),
        date(2024, 1, 31),
        ticker="AAPL",
        frequency="daily"
    )
    
    # Should have ~20 observations
    assert len(returns) > 15
    # Returns should be reasonable (-10% to +10% daily)
    assert returns.abs().max() < 0.10
```

## Key Implementation Considerations

### 1. Date Range Extension
Many calculations need data before the requested start date:
- Returns need previous price
- YoY inflation needs 12 months prior
- Handle this transparently

### 2. Data Alignment
When combining multiple series (prices + dividends):
- Use outer join to preserve all dates
- Fill dividends with 0 where missing
- Align on date index properly

### 3. Missing Data Handling
- If no dividend data, calculate price-only returns
- If insufficient history for YoY inflation, return shorter series
- Document behavior clearly

### 4. Performance
- Don't fetch more data than needed
- Cache intermediate calculations if possible
- Efficient pandas operations

## Success Criteria

- [ ] All LogicalDataType values implemented
- [ ] Passes all DataProvider contract tests
- [ ] Return calculations match expected values
- [ ] Inflation calculations are accurate
- [ ] Risk-free rate selection works correctly
- [ ] Frequency conversion preserves return properties
- [ ] Handles missing data gracefully
- [ ] Clear error messages
- [ ] Performance <100ms overhead vs raw provider

## Configuration

```yaml
# config/calculations.yaml
return_calculations:
  total_returns:
    use_adjusted_close: true
    dividend_reinvestment: true
    
inflation:
  default_method: yoy  # or mom_annualized
  default_index: cpi   # or pce
  
risk_free:
  default_tenor: 3m
  fallback_chain:
    3m: [TREASURY_3M, FED_FUNDS]
    1y: [TREASURY_1Y, TREASURY_6M]
```

## Next Steps

After this implementation:
1. Create Provider Factory for easy production setup
2. Add caching layer
3. Implement quality monitoring
4. Performance optimization

The transformed provider is the brain of the data layer - it makes raw data useful for portfolio optimization!
