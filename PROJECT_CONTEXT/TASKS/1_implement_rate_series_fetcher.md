# Task 1: Implement Rate Series Support in TotalReturnFetcher

**Status**: TODO  
**Priority**: CRITICAL - Blocking all other data work  
**Estimated Time**: 2-3 hours  
**Dependencies**: None

## Context
The TotalReturnFetcher currently only handles price series from yfinance. We need to extend it to handle rate series from FRED (e.g., DGS3MO for risk-free rates). This is critical because the cash/risk-free rate exposure cannot be fetched without this functionality.

## Problem
- Cash/risk-free rate data comes from FRED as an annualized rate (e.g., 5.25%)
- This is fundamentally different from price data
- We need to convert rates to returns based on the requested frequency
- The exposure universe config already specifies rate_series implementations

## Requirements

### 1. Add Rate Series Method to TotalReturnFetcher
Location: `/src/data/total_returns.py`

Add this method:
```python
def fetch_rate_series_returns(
    self,
    series_code: str,
    source: str,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily"
) -> pd.Series:
    """Fetch returns from a rate series (e.g., FRED Treasury rates).
    
    Args:
        series_code: FRED series code (e.g., 'DGS3MO')
        source: Data source (currently only 'FRED' supported)
        start_date: Start date
        end_date: End date
        frequency: Return frequency ('daily', 'monthly', 'annual')
        
    Returns:
        Series of returns at the specified frequency
    """
```

### 2. Implementation Details
- Use the existing FREDDataFetcher to fetch the rate data
- FRED returns rates as percentages, convert to decimals (/100)
- Convert annualized rates to period returns:
  - Daily: rate / 252 (simple approximation is fine)
  - Monthly: (1 + rate)^(1/12) - 1 (geometric)
  - Annual: rate (already annualized)
- Forward fill missing values (weekends, holidays)
- Handle errors gracefully with logging

### 3. Update fetch_exposure_returns Method
In the same file, update the main method to handle rate_series:

```python
def fetch_exposure_returns(self, exposure_config: dict, ...) -> pd.Series:
    """Existing method - needs to handle 'rate_series' implementation type"""
    
    for impl in exposure_config['implementations']:
        if impl['type'] == 'rate_series':
            # NEW: Call fetch_rate_series_returns
            returns = self.fetch_rate_series_returns(
                series_code=impl['series'],
                source=impl['source'],
                start_date=start_date,
                end_date=end_date,
                frequency=frequency
            )
            if not returns.empty:
                return returns
```

### 4. Add Tests
Location: `/tests/test_total_returns.py`

Add test cases:
```python
def test_fetch_rate_series_returns():
    """Test fetching rate series from FRED."""
    # Test with DGS3MO
    # Test frequency conversions
    # Test error handling
    
def test_exposure_with_rate_series():
    """Test fetching exposure that uses rate_series implementation."""
    # Use cash_rate exposure config
    # Verify it fetches from FRED
```

## Testing Instructions
1. Run existing tests to ensure nothing breaks: `pytest tests/test_total_returns.py`
2. Test with real FRED data:
   ```python
   fetcher = TotalReturnFetcher()
   returns = fetcher.fetch_rate_series_returns(
       series_code="DGS3MO",
       source="FRED",
       start_date=datetime(2023, 1, 1),
       end_date=datetime(2023, 12, 31),
       frequency="daily"
   )
   print(f"Fetched {len(returns)} returns")
   print(f"Sample: {returns.iloc[0]:.6f} daily ({returns.iloc[0]*252*100:.2f}% annualized)")
   ```

3. Test the cash_rate exposure:
   ```python
   universe = ExposureUniverse()
   universe.load_config('/Users/scottmcguire/portfolio-tool/config/exposure_universe.yaml')
   cash_exposure = universe.get_exposure('cash_rate')
   
   returns = fetcher.fetch_exposure_returns(
       cash_exposure,
       start_date=datetime(2023, 1, 1),
       end_date=datetime(2023, 12, 31),
       frequency="monthly"
   )
   ```

## Success Criteria
- [ ] fetch_rate_series_returns method implemented and working
- [ ] FRED DGS3MO data successfully fetched and converted to returns
- [ ] Frequency conversions work correctly (daily, monthly, annual)
- [ ] Integration with fetch_exposure_returns working
- [ ] All existing tests still pass
- [ ] New tests for rate series functionality pass
- [ ] Cash/risk-free rate exposure can be fetched through ExposureUniverse

## Notes
- Reference implementation guide at: `/PROJECT_CONTEXT/TASKS/cash_rate_implementation_guide.md`
- The FREDDataFetcher already exists and works - just use it
- Keep it simple - don't over-engineer the frequency conversion
- This blocks all other data work, so prioritize getting it working over perfection
