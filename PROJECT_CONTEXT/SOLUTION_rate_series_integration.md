# Solution: Integrate FRED Rate Data with TotalReturnFetcher

## The Issue
You have:
1. ✅ `FREDDataFetcher` that can fetch risk-free rates
2. ✅ `TotalReturnFetcher` that handles ETF/stock price data
3. ❌ No connection between them for 'rate_series' implementation type

## Quick Solution

Add this method to your `TotalReturnFetcher` class or wherever you're fetching exposure data:

```python
def fetch_rate_series_returns(
    self,
    implementation: Dict,
    start_date: datetime,
    end_date: datetime
) -> pd.Series:
    """Convert rate series to returns.
    
    Args:
        implementation: Implementation dict with 'source' and 'series'
        start_date: Start date
        end_date: End date
        
    Returns:
        Series of returns based on the rate
    """
    source = implementation.get('source')
    series = implementation.get('series')
    
    if source == 'FRED':
        # Use your existing FREDDataFetcher
        fred_fetcher = FREDDataFetcher()
        
        # Map series codes to fetch method
        if series == 'DGS3MO':
            # Fetch 3-month Treasury rate
            rates = fred_fetcher.fetch_risk_free_rate(
                start_date, 
                end_date, 
                maturity='3m', 
                frequency='daily'
            )
            
            # FRED returns annualized rates as decimals (e.g., 0.0525 for 5.25%)
            # Convert to daily returns
            daily_returns = rates / 252  # Trading days per year
            
            # Name the series
            daily_returns.name = 'risk_free_rate'
            
            return daily_returns
        else:
            raise ValueError(f"Unsupported FRED series: {series}")
    else:
        raise ValueError(f"Unsupported rate source: {source}")
```

## Integration Point

In your exposure data fetching logic (probably in `exposure_universe.py`), add handling for 'rate_series':

```python
def fetch_implementation_data(
    self,
    implementation: Dict,
    start_date: datetime,
    end_date: datetime
) -> pd.Series:
    """Fetch data for a specific implementation."""
    
    impl_type = implementation.get('type')
    
    if impl_type == 'rate_series':
        # Special handling for rate data
        return self.fetch_rate_series_returns(implementation, start_date, end_date)
        
    elif impl_type in ['etf', 'etf_average', 'fund', 'fund_average']:
        # Existing price-based handling
        tickers = implementation.get('tickers', [implementation.get('ticker')])
        # ... existing code to fetch price returns
        
    else:
        raise ValueError(f"Unknown implementation type: {impl_type}")
```

## Even Simpler: Use ETFs Only

If you want to avoid this complexity for now, just modify the exposure to use only ETFs:

```python
# In your code where you load exposures
def get_cash_rate_exposure(universe):
    cash_rate = universe.get_exposure('cash_rate')
    
    # Filter out rate_series implementations for now
    cash_rate.implementations = [
        impl for impl in cash_rate.implementations 
        if impl['type'] != 'rate_series'
    ]
    
    return cash_rate
```

## Test It

```python
# Test that cash rate now works
universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
cash_rate = universe.get_exposure('cash_rate')

# This should now work with just ETFs
returns = fetcher.fetch_exposure_returns(
    cash_rate,
    datetime(2020, 1, 1),
    datetime(2023, 12, 31)
)

print(f"Cash rate annual return: {returns.mean() * 252:.2%}")
```

## Note on Returns

Remember:
- **ETF returns** (BIL, SHV): Already reflect the risk-free rate through price appreciation
- **Rate series**: Need to be converted from annual rates to period returns
- Both should give very similar results (within a few basis points)

The ETF approach is simpler and gives you the total return directly. The rate series approach is more "pure" but requires this conversion step.
