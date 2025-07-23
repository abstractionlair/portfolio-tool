# Task: Implement Enhanced Equity Return Decomposition in Data Layer

**Status**: READY  
**Priority**: HIGH  
**Type**: Feature Implementation  
**Estimated Time**: 3-4 hours
**Dependencies**: Dividend double-counting fix (COMPLETE)

## Overview

Implement an enhanced equity return decomposition that properly separates real risk premium components. This builds on the basic return decomposition by adjusting earnings growth for inflation and the real risk-free rate.

## Economic Framework

The decomposition follows this logic:

```
r_nominal = r_dividend + r_pe_change + r_nominal_earnings

r_real_risk_premium = r_nominal - r_inflation - r_real_rf
                    = r_dividend + r_pe_change + r_real_earnings_excess

where:
r_real_earnings_excess = r_nominal_earnings - r_inflation - r_real_rf
```

Key insight: Dividend yield and P/E changes are already "real" ratios, but earnings growth is in currency units and needs adjustment.

## Implementation Design

### 1. New Method in TransformedDataProvider

```python
def decompose_equity_returns(
    self,
    ticker: str,
    start: date,
    end: date,
    earnings_data: pd.Series,
    frequency: str = "daily"
) -> Dict[str, pd.Series]:
    """
    Decompose equity returns into economically meaningful components.
    
    Returns dictionary with:
    - nominal_return: Total nominal return
    - dividend_yield: Dividend yield component
    - pe_change: P/E multiple change component  
    - nominal_earnings_growth: Earnings growth in nominal terms
    - real_earnings_growth: Earnings growth adjusted for inflation
    - real_earnings_excess: Real earnings growth above real risk-free rate
    - inflation: Inflation rate over period
    - nominal_rf: Nominal risk-free rate
    - real_rf: Real risk-free rate
    - real_risk_premium: Total real risk premium
    - excess_return: Nominal return minus nominal risk-free rate
    
    All components are aligned and calculated for the same periods.
    """
```

### 2. Enhanced Data Requirements

The method needs to fetch and align:

1. **Price and Dividend Data** (existing)
   - Unadjusted prices for P/E calculation
   - Adjusted prices for total return
   - Dividend data

2. **Earnings Data** (input parameter)
   - Quarterly or annual EPS data
   - Will need interpolation/alignment

3. **Economic Data** (new integration)
   - Inflation rate (CPI or PCE)
   - Nominal risk-free rate (Treasury rates)
   - Real risk-free rate (calculated or TIPS)

### 3. Implementation Steps

#### Step 1: Enhance Base Decomposition

Update the existing `decompose_returns` to ensure we have all needed components:

```python
# In decompose_returns method
if earnings_data is not None:
    # Ensure we calculate these components
    # - total_return (from adjusted prices)
    # - dividend_yield (dividends / lagged price)
    # - earnings_growth (earnings pct_change)
    # - pe_change (calculated as residual or directly)
    
    # Important: Verify the identity holds
    # total_return ≈ dividend_yield + earnings_growth + pe_change
```

#### Step 2: Add Economic Data Integration

```python
def _get_economic_data_for_decomposition(
    self,
    start: date,
    end: date,
    frequency: str,
    inflation_measure: str = "CPI",
    rf_tenor: str = "3M"
) -> Dict[str, pd.Series]:
    """Fetch and align economic data needed for decomposition."""
    
    # Get inflation
    if inflation_measure == "CPI":
        inflation = self._compute_inflation_rate(start, end, frequency)
    else:  # PCE or other measures
        inflation = self._compute_inflation_rate(start, end, frequency, method=inflation_measure.lower())
    
    # Get nominal risk-free rate
    nominal_rf = self._compute_nominal_risk_free(start, end, frequency, tenor=rf_tenor)
    
    # Get real risk-free rate
    real_rf = self._compute_real_risk_free(start, end, frequency, tenor=rf_tenor)
    
    return {
        'inflation': inflation,
        'nominal_rf': nominal_rf,
        'real_rf': real_rf
    }
```

#### Step 3: Implement Alignment Logic

```python
def _align_decomposition_data(
    self,
    base_decomp: Dict[str, pd.Series],
    economic_data: Dict[str, pd.Series],
    frequency: str
) -> pd.DataFrame:
    """Align all data series for consistent calculation."""
    
    # Create DataFrame with all components
    all_data = {}
    all_data.update(base_decomp)
    all_data.update(economic_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Handle frequency conversion if needed
    if frequency != "daily":
        # May need to aggregate or resample
        # E.g., for monthly: ensure all data is monthly
        pass
    
    # Forward fill economic data if needed (rates don't change daily)
    for col in ['inflation', 'nominal_rf', 'real_rf']:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')
    
    # Drop rows with any NaN in critical columns
    critical_cols = ['total_return', 'dividend_yield', 'earnings_growth', 'pe_change']
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    
    return df
```

#### Step 4: Calculate Real Components

```python
def _calculate_real_components(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate real risk premium components."""
    
    result = aligned_data.copy()
    
    # Real earnings growth
    result['real_earnings_growth'] = result['nominal_earnings_growth'] - result['inflation']
    
    # Real earnings excess (above real risk-free rate)
    result['real_earnings_excess'] = result['real_earnings_growth'] - result['real_rf']
    
    # Total real risk premium (sum of real components)
    result['real_risk_premium'] = (
        result['dividend_yield'] + 
        result['pe_change'] + 
        result['real_earnings_excess']
    )
    
    # Excess return for comparison
    result['excess_return'] = result['total_return'] - result['nominal_rf']
    
    # Verification: alternative calculation
    result['real_risk_premium_check'] = (
        result['total_return'] - result['inflation'] - result['real_rf']
    )
    
    # Add decomposition quality check
    result['decomp_error'] = abs(
        result['real_risk_premium'] - result['real_risk_premium_check']
    )
    
    return result
```

### 4. Data Sources and Challenges

#### Earnings Data Sources

1. **For ETFs/Indices**: 
   - Need index-level earnings (S&P 500 earnings)
   - Sources: S&P, Bloomberg, FRED (some series)

2. **For Individual Stocks**:
   - Yahoo Finance quarterly earnings
   - Need to handle reporting dates vs effective dates
   - Interpolation for daily/monthly frequency

3. **Implementation Approach**:
   ```python
   def get_earnings_data(
       self,
       ticker: str,
       start: date,
       end: date,
       frequency: str = "quarterly"
   ) -> pd.Series:
       """Fetch earnings data with appropriate source."""
       
       # Check if it's an index/ETF
       if ticker in ['SPY', 'IVV', 'VOO']:  # S&P 500
           # Use S&P 500 earnings from FRED or other source
           return self._get_sp500_earnings(start, end, frequency)
       
       elif ticker in self.KNOWN_ETFS:
           # Map to underlying index earnings
           return self._get_index_earnings(ticker, start, end, frequency)
       
       else:
           # Individual stock - use financial data API
           return self._get_stock_earnings(ticker, start, end, frequency)
   ```

#### Handling Frequency Mismatches

1. **Earnings**: Usually quarterly, need interpolation
2. **Economic Data**: May be monthly, need alignment
3. **Returns**: Can be daily, need aggregation

```python
def _interpolate_earnings(
    self,
    quarterly_earnings: pd.Series,
    target_dates: pd.DatetimeIndex,
    method: str = "time"
) -> pd.Series:
    """Interpolate quarterly earnings to target frequency."""
    
    # Create a complete date range
    full_range = pd.date_range(
        start=quarterly_earnings.index.min(),
        end=quarterly_earnings.index.max(),
        freq='D'
    )
    
    # Reindex to daily
    daily_earnings = quarterly_earnings.reindex(full_range)
    
    # Interpolate
    if method == "time":
        # Time-based interpolation
        daily_earnings = daily_earnings.interpolate(method='time')
    elif method == "flat":
        # Forward fill (flat between quarters)
        daily_earnings = daily_earnings.fillna(method='ffill')
    
    # Select target dates
    return daily_earnings.reindex(target_dates)
```

### 5. Testing Requirements

#### Unit Tests

```python
def test_equity_decomposition_identity():
    """Test that components sum to total return."""
    # Create synthetic data where we know the relationship
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='M')
    
    # Known components
    dividend_yield = pd.Series(0.02/12, index=dates)  # 2% annual
    earnings_growth = pd.Series(0.08/12, index=dates)  # 8% annual
    pe_change = pd.Series(0.05/12, index=dates)  # 5% annual
    
    # Total should be ~15% annualized
    total_return = dividend_yield + earnings_growth + pe_change
    
    # Test decomposition
    result = decompose_equity_returns(...)
    
    # Verify identity holds
    assert np.allclose(
        result['total_return'],
        result['dividend_yield'] + result['earnings_growth'] + result['pe_change'],
        rtol=1e-5
    )

def test_real_adjustment():
    """Test inflation and rf adjustment of earnings."""
    # With 3% inflation and 1% real rf
    # 8% nominal earnings -> 5% real -> 4% real excess
    
    nominal_earnings_growth = 0.08
    inflation = 0.03
    real_rf = 0.01
    
    expected_real_excess = nominal_earnings_growth - inflation - real_rf
    
    # Test the calculation
    result = calculate_real_components(...)
    assert np.isclose(result['real_earnings_excess'], expected_real_excess)
```

#### Integration Tests

```python
def test_spy_decomposition():
    """Test with real S&P 500 data."""
    # Use a period with known characteristics
    # E.g., 2019 - stable growth, ~2% dividends
    
    result = provider.decompose_equity_returns(
        'SPY',
        date(2019, 1, 1),
        date(2019, 12, 31),
        sp500_earnings_data,
        frequency='monthly'
    )
    
    # Check reasonable ranges
    assert 0.015 < result['dividend_yield'].mean() < 0.025  # 1.5-2.5%
    assert result['decomp_error'].max() < 0.001  # Identity holds
```

### 6. Example Usage

```python
# Example script: examples/enhanced_equity_decomposition_demo.py

# Initialize provider
provider = TransformedDataProvider(coordinator)

# Get S&P 500 earnings (from FRED or other source)
sp500_earnings = get_sp500_earnings_data(start_date, end_date)

# Decompose returns
decomp = provider.decompose_equity_returns(
    ticker='SPY',
    start=date(2020, 1, 1),
    end=date(2024, 12, 31),
    earnings_data=sp500_earnings,
    frequency='monthly'
)

# Display results
print("S&P 500 Return Decomposition (Annualized):")
print(f"Total Nominal Return: {decomp['nominal_return'].mean() * 12:.2%}")
print(f"  - Dividend Yield: {decomp['dividend_yield'].mean() * 12:.2%}")
print(f"  - P/E Change: {decomp['pe_change'].mean() * 12:.2%}")
print(f"  - Nominal Earnings Growth: {decomp['nominal_earnings_growth'].mean() * 12:.2%}")
print(f"\nReal Risk Premium: {decomp['real_risk_premium'].mean() * 12:.2%}")
print(f"  - Dividend Yield: {decomp['dividend_yield'].mean() * 12:.2%}")
print(f"  - P/E Change: {decomp['pe_change'].mean() * 12:.2%}")
print(f"  - Real Earnings Excess: {decomp['real_earnings_excess'].mean() * 12:.2%}")
print(f"\nEconomic Context:")
print(f"  - Inflation: {decomp['inflation'].mean() * 12:.2%}")
print(f"  - Real Risk-Free Rate: {decomp['real_rf'].mean() * 12:.2%}")
```

### 7. Success Criteria

- [ ] Base decomposition correctly splits returns into div yield, earnings growth, P/E change
- [ ] Identity holds: total return = sum of components (within 0.1% tolerance)
- [ ] Economic data properly integrated and aligned
- [ ] Real components correctly calculated
- [ ] Works with multiple frequencies (daily, monthly, quarterly)
- [ ] Handles missing data gracefully
- [ ] Example demonstrates economic intuition
- [ ] All tests pass

### 8. Future Enhancements

After this foundation is complete:

1. **Add Sector/Industry Decomposition**: Compare earnings growth across sectors
2. **Multi-Factor Attribution**: Decompose P/E changes into style factors
3. **International Markets**: Handle currency effects for international equities
4. **Forecast Integration**: Use decomposition for return prediction
5. **Risk Decomposition**: Separate volatility into component contributions

## Notes

- This provides the data foundation for sophisticated equity analysis
- Each component has different time series properties suitable for different models
- The framework extends naturally to other asset classes with appropriate modifications
- Focus on data quality and alignment - garbage in, garbage out
