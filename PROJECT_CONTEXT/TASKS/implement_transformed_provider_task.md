# Task: Implement Transformed Data Provider

**Status**: TODO  
**Priority**: HIGH  
**Estimated Time**: 2 days  
**Dependencies**: implement_raw_provider_task.md completed

## Overview

Implement the `TransformedDataProvider` that takes a `RawDataProvider` and adds computation capabilities for logical data types (returns, risk-free rates, inflation). This provider handles all derived calculations while delegating raw data fetching to the underlying provider.

## Implementation Plan

### 1. `src/data/providers/transformed_provider.py`

Main implementation that wraps a raw provider:

```python
class DefaultTransformedDataProvider:
    """
    Provides both raw and computed data types.
    
    This provider:
    - Delegates raw data requests to underlying RawDataProvider
    - Computes returns from prices
    - Computes inflation rates from price indices
    - Selects appropriate risk-free rates based on tenor
    - Handles frequency conversion properly
    """
    
    def __init__(self, raw_provider: RawDataProvider):
        self.raw_provider = raw_provider
        self._return_calculator = ReturnCalculator()
        self._economic_calculator = EconomicCalculator()
    
    def get_data(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        **kwargs
    ) -> pd.Series:
        """Get any type of data, computing if necessary."""
        # Validate inputs
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # Route based on data type
        if isinstance(data_type, RawDataType):
            # Just pass through to raw provider
            return self.raw_provider.get_data(data_type, start, end, ticker, frequency, **kwargs)
        
        elif isinstance(data_type, LogicalDataType):
            # Compute the logical type
            return self._compute_logical_data(data_type, start, end, ticker, frequency, **kwargs)
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _compute_logical_data(self, data_type, start, end, ticker, frequency, **kwargs):
        """Route logical data types to appropriate calculators."""
        if data_type == LogicalDataType.TOTAL_RETURN:
            return self._compute_total_return(ticker, start, end, frequency)
        elif data_type == LogicalDataType.SIMPLE_RETURN:
            return self._compute_simple_return(ticker, start, end, frequency)
        elif data_type == LogicalDataType.LOG_RETURN:
            return self._compute_log_return(ticker, start, end, frequency)
        elif data_type == LogicalDataType.EXCESS_RETURN:
            return self._compute_excess_return(ticker, start, end, frequency, **kwargs)
        elif data_type == LogicalDataType.NOMINAL_RISK_FREE:
            return self._get_nominal_risk_free(start, end, frequency, **kwargs)
        elif data_type == LogicalDataType.REAL_RISK_FREE:
            return self._get_real_risk_free(start, end, frequency, **kwargs)
        elif data_type == LogicalDataType.INFLATION_RATE:
            return self._compute_inflation_rate(start, end, frequency, **kwargs)
        elif data_type == LogicalDataType.TERM_PREMIUM:
            return self._compute_term_premium(start, end, frequency, **kwargs)
        else:
            raise NotImplementedError(f"Logical type {data_type} not implemented")
```

### 2. `src/data/providers/calculators/return_calculator.py`

Handle all return calculations:

```python
class ReturnCalculator:
    """Calculates various types of returns from price data."""
    
    def calculate_simple_returns(
        self,
        prices: pd.Series,
        frequency: str
    ) -> pd.Series:
        """
        Calculate simple returns from price series.
        
        Simple return = (P_t - P_{t-1}) / P_{t-1}
        """
        returns = prices.pct_change()
        returns.name = f"{prices.name}_simple_return" if prices.name else "simple_return"
        return returns.dropna()
    
    def calculate_log_returns(
        self,
        prices: pd.Series,
        frequency: str  
    ) -> pd.Series:
        """
        Calculate log returns from price series.
        
        Log return = ln(P_t / P_{t-1})
        """
        log_returns = np.log(prices / prices.shift(1))
        log_returns.name = f"{prices.name}_log_return" if prices.name else "log_return"
        return log_returns.dropna()
    
    def calculate_total_returns(
        self,
        prices: pd.Series,
        dividends: Optional[pd.Series] = None,
        splits: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate total returns including dividends and splits.
        
        Total return = (P_t + D_t) / P_{t-1} - 1
        where D_t is dividend paid at time t
        """
        if dividends is None or dividends.empty:
            # No dividends, just price return
            return self.calculate_simple_returns(prices, "daily")
        
        # Align dividends with prices
        aligned_div = dividends.reindex(prices.index, fill_value=0)
        
        # Calculate total return
        total_returns = (prices + aligned_div) / prices.shift(1) - 1
        total_returns.name = f"{prices.name}_total_return" if prices.name else "total_return"
        
        return total_returns.dropna()
```

### 3. `src/data/providers/calculators/economic_calculator.py`

Handle economic indicator calculations:

```python
class EconomicCalculator:
    """Calculates economic indicators from raw data."""
    
    def calculate_inflation_rate(
        self,
        price_index: pd.Series,
        method: str = "yoy"
    ) -> pd.Series:
        """
        Calculate inflation rate from price index.
        
        Args:
            price_index: CPI or PCE index series
            method: 'yoy' for year-over-year, 'mom' for month-over-month
        """
        if method == "yoy":
            # Year-over-year inflation
            inflation = price_index.pct_change(periods=12)
            inflation.name = "inflation_yoy"
        elif method == "mom":
            # Month-over-month (annualized)
            monthly = price_index.pct_change()
            inflation = (1 + monthly) ** 12 - 1
            inflation.name = "inflation_mom_annualized"
        else:
            raise ValueError(f"Unknown inflation method: {method}")
        
        return inflation.dropna()
    
    def calculate_real_rate(
        self,
        nominal_rate: pd.Series,
        inflation_rate: pd.Series
    ) -> pd.Series:
        """
        Calculate real interest rate from nominal rate and inflation.
        
        Fisher equation: (1 + nominal) = (1 + real) * (1 + inflation)
        Therefore: real = (1 + nominal) / (1 + inflation) - 1
        """
        # Align series
        aligned_nominal, aligned_inflation = nominal_rate.align(inflation_rate)
        
        # Fisher equation
        real_rate = (1 + aligned_nominal) / (1 + aligned_inflation) - 1
        real_rate.name = "real_rate"
        
        return real_rate.dropna()
```

### 4. Complete Implementation Methods

```python
class DefaultTransformedDataProvider:  # continued
    
    def _compute_total_return(self, ticker: str, start: date, end: date, frequency: str) -> pd.Series:
        """Compute total return including dividends."""
        # Determine fetch frequency
        fetch_freq = self._determine_fetch_frequency(frequency)
        
        # Need to extend start date for proper return calculation
        extended_start = self._extend_start_date(start, fetch_freq)
        
        # Fetch required data
        prices = self.raw_provider.get_data(
            RawDataType.ADJUSTED_CLOSE, extended_start, end, ticker, fetch_freq
        )
        
        # Check if we need dividends
        try:
            dividends = self.raw_provider.get_data(
                RawDataType.DIVIDEND, extended_start, end, ticker, fetch_freq
            )
        except DataNotAvailableError:
            # No dividend data available, use price returns only
            dividends = None
        
        # Calculate returns
        returns = self._return_calculator.calculate_total_returns(prices, dividends)
        
        # Convert frequency if needed
        if fetch_freq != frequency:
            returns = self._convert_return_frequency(returns, fetch_freq, frequency)
        
        # Trim to requested period
        return returns[start:end]
    
    def _get_nominal_risk_free(self, start: date, end: date, frequency: str, **kwargs) -> pd.Series:
        """Get appropriate nominal risk-free rate."""
        # Get tenor from kwargs
        tenor = kwargs.get('tenor', '3m')
        
        # Map to appropriate treasury
        tenor_map = {
            '3m': RawDataType.TREASURY_3M,
            '6m': RawDataType.TREASURY_6M,
            '1y': RawDataType.TREASURY_1Y,
            '2y': RawDataType.TREASURY_2Y,
            '5y': RawDataType.TREASURY_5Y,
            '10y': RawDataType.TREASURY_10Y,
            '30y': RawDataType.TREASURY_30Y,
        }
        
        if tenor not in tenor_map:
            # Could interpolate or use closest
            raise ValueError(f"Unsupported tenor: {tenor}")
        
        treasury_type = tenor_map[tenor]
        
        # Fetch the rate
        rate = self.raw_provider.get_data(
            treasury_type, start, end, ticker=None, frequency=frequency
        )
        
        # Convert from percentage to decimal if needed
        if rate.mean() > 1:  # Likely in percentage form
            rate = rate / 100
        
        rate.name = f"nominal_rf_{tenor}"
        return rate
    
    def _compute_inflation_rate(self, start: date, end: date, frequency: str, **kwargs) -> pd.Series:
        """Compute inflation rate from price index."""
        # Which index to use
        index_type = kwargs.get('index', 'cpi')  # or 'pce'
        
        # Need extra data for YoY calculation
        extended_start = start - timedelta(days=370)
        
        # Get the index
        if index_type == 'cpi':
            price_index = self.raw_provider.get_data(
                RawDataType.CPI_INDEX, extended_start, end, ticker=None, frequency=frequency
            )
        elif index_type == 'pce':
            price_index = self.raw_provider.get_data(
                RawDataType.PCE_INDEX, extended_start, end, ticker=None, frequency=frequency
            )
        else:
            raise ValueError(f"Unknown inflation index: {index_type}")
        
        # Calculate inflation
        inflation = self._economic_calculator.calculate_inflation_rate(
            price_index, method=kwargs.get('method', 'yoy')
        )
        
        # Trim to requested period
        return inflation[start:end]
    
    def _convert_return_frequency(
        self,
        returns: pd.Series,
        from_freq: str,
        to_freq: str
    ) -> pd.Series:
        """Convert return series frequency (compound properly)."""
        from_enum = Frequency(from_freq)
        to_enum = Frequency(to_freq)
        
        if not from_enum.can_convert_to(to_enum):
            raise ValueError(f"Cannot convert {from_freq} returns to {to_freq}")
        
        if from_freq == to_freq:
            return returns
        
        # Compound returns properly
        if to_enum == Frequency.MONTHLY:
            # Compound daily to monthly
            monthly = (1 + returns).resample('ME').prod() - 1
            return monthly
        elif to_enum == Frequency.QUARTERLY:
            # Compound to quarterly
            quarterly = (1 + returns).resample('QE').prod() - 1
            return quarterly
        # ... etc
```

## Key Implementation Points

### 1. Proper Return Calculation
```python
# Important: Need price before start date to calculate first return
def _extend_start_date(self, start: date, frequency: str) -> date:
    """Extend start date to ensure we can calculate returns from requested start."""
    if frequency == "daily":
        return start - timedelta(days=5)  # Extra for weekends
    elif frequency == "monthly":
        return start - timedelta(days=35)
    # ... etc
```

### 2. Handle Missing Dividends Gracefully
```python
# Don't fail if no dividend data
try:
    dividends = self.raw_provider.get_data(RawDataType.DIVIDEND, ...)
except DataNotAvailableError:
    logger.info(f"No dividend data for {ticker}, using price returns only")
    dividends = None
```

### 3. Frequency Conversion for Returns
```python
# Must compound returns, not average them!
# WRONG:
monthly_returns = daily_returns.resample('ME').mean()

# RIGHT:
monthly_returns = (1 + daily_returns).resample('ME').prod() - 1
```

### 4. Smart Data Fetching
```python
def _determine_fetch_frequency(self, requested_freq: str) -> str:
    """Determine optimal frequency to fetch data at."""
    # For returns, might need daily even if monthly requested
    # For rates, can often fetch at requested frequency
    # This avoids unnecessary data transfer
```

## Testing Requirements

Must pass all contract tests plus:

### Specific Calculation Tests
```python
def test_total_return_includes_dividends():
    """Verify dividends are properly included."""
    prices = pd.Series([100, 102, 101], 
                      index=pd.date_range("2023-01-01", periods=3))
    dividends = pd.Series([0, 1, 0],
                         index=pd.date_range("2023-01-01", periods=3))
    
    calculator = ReturnCalculator()
    returns = calculator.calculate_total_returns(prices, dividends)
    
    # First return: (102 + 1) / 100 - 1 = 0.03
    assert abs(returns.iloc[0] - 0.03) < 0.0001

def test_inflation_calculation():
    """Verify YoY inflation calculation."""
    # CPI: 100, 102, 103 (monthly)
    cpi = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
                    index=pd.date_range("2022-01-01", periods=13, freq="ME"))
    
    calculator = EconomicCalculator()
    inflation = calculator.calculate_inflation_rate(cpi, method="yoy")
    
    # YoY for month 13: 112/100 - 1 = 0.12
    assert abs(inflation.iloc[-1] - 0.12) < 0.0001

def test_frequency_conversion():
    """Test proper return compounding."""
    # Daily returns: 1%, 1%, 1%
    daily_returns = pd.Series([0.01, 0.01, 0.01],
                             index=pd.date_range("2023-01-01", periods=3))
    
    # Monthly should compound: (1.01 * 1.01 * 1.01) - 1 = 0.030301
    provider = DefaultTransformedDataProvider(mock_raw)
    monthly = provider._convert_return_frequency(daily_returns, "daily", "monthly")
    
    assert abs(monthly.iloc[0] - 0.030301) < 0.000001
```

## Success Criteria

- [ ] All LogicalDataType values are implemented
- [ ] Passes all contract tests
- [ ] Return calculations match expected values
- [ ] Frequency conversion preserves return properties
- [ ] Handles missing data gracefully
- [ ] Clear error messages
- [ ] Well-documented calculation methods
- [ ] Performance: <100ms overhead vs raw provider

## Configuration

```yaml
# config/transformed_provider.yaml
calculations:
  inflation:
    default_index: cpi  # or pce
    default_method: yoy
    
  risk_free:
    default_tenor: 3m
    fallback_tenors:  # If primary not available
      3m: [fed_funds, 6m]
      1y: [6m, 2y]
      
  returns:
    use_adjusted_close: true  # For simple returns
    compound_method: geometric  # or arithmetic
```

## Notes

1. This provider should be thoroughly tested as it's where bugs can compound
2. Document all calculation methods clearly
3. Consider adding calculation metadata to returned series
4. Log when falling back or making assumptions
5. Performance is less critical than correctness

This transformed provider enables all the advanced functionality while keeping raw data fetching separate and simple.
