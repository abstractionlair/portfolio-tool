# Task: Implement Rate Series Support in TotalReturnFetcher

**Status**: READY TO IMPLEMENT  
**Priority**: HIGH - Blocking exposure universe completion  
**Estimated Time**: 2-3 hours  
**Dependencies**: None (FRED fetcher already exists)

## Problem Statement
The TotalReturnFetcher currently only handles price series from yfinance. We need to extend it to handle rate series from FRED (e.g., DGS3MO for risk-free rate) which are fundamentally different:
- Price series: Returns calculated from price changes
- Rate series: Already represents returns (as annualized percentages)

## Implementation Requirements

### 1. Extend TotalReturnFetcher class
**File**: `/src/data/total_returns.py`

Add the following method:
```python
def fetch_rate_series_returns(
    self,
    series_code: str,
    source: str,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily"
) -> pd.Series:
    """Fetch returns from a rate series (e.g., FRED Treasury rates)."""
```

Implementation details:
- Only support "FRED" as source initially
- Use self.fred_fetcher.fetch_series() to get raw data
- Convert from percentage to decimal (divide by 100)
- Convert annualized rates to period returns based on frequency:
  - Daily: rate / 252
  - Monthly: (1 + rate)^(1/12) - 1
  - Quarterly: (1 + rate)^(1/4) - 1
  - Annual: rate (no conversion needed)
- Forward fill missing values (weekends/holidays)
- Return pd.Series with appropriate name

### 2. Update fetch_exposure_returns method
**File**: `/src/data/total_returns.py`

Modify to handle new implementation type:
```python
def fetch_exposure_returns(
    self,
    exposure_config: dict,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily"
) -> Dict[str, any]:
```

Add handling for `impl['type'] == 'rate_series'`:
- Extract series_code and source from implementation config
- Call fetch_rate_series_returns()
- Return in standard format with success flag

### 3. Add specialized cash rate method
**File**: `/src/data/total_returns.py`

Add convenience method with fallback logic:
```python
def fetch_cash_rate_returns(
    self,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily"
) -> pd.Series:
    """Fetch cash/risk-free rate with FRED priority and ETF fallback."""
```

Logic:
1. Try FRED DGS3MO first (most accurate)
2. If fails, try T-bill ETFs in order: BIL, SHV, SGOV
3. Log which source was used
4. Raise ValueError if all sources fail

### 4. Update ExposureUniverse integration
**File**: `/src/data/exposure_universe.py`

Ensure the exposure universe correctly identifies rate series implementations:
- Check that cash_rate exposure config includes rate_series implementation
- Verify the fetch_universe_returns method handles rate series

## Testing Requirements

### 1. Create unit tests
**File**: `/tests/test_total_returns_rate_series.py`

Test cases:
- `test_fetch_rate_series_returns_daily`: Verify daily conversion
- `test_fetch_rate_series_returns_monthly`: Verify monthly conversion
- `test_fetch_rate_series_returns_missing_data`: Test forward fill
- `test_cash_rate_fallback_logic`: Test FRED → ETF fallback
- `test_rate_series_in_exposure_universe`: Integration test

### 2. Create integration test
**File**: `/tests/test_exposure_universe_integration.py`

Test that cash_rate exposure works in full universe fetch:
```python
def test_universe_with_rate_series():
    universe = ExposureUniverse()
    universe.load_config('config/exposure_universe.yaml')
    results = fetcher.fetch_universe_returns(universe, start, end)
    assert 'cash_rate' in results
    assert results['cash_rate']['success']
```

## Validation Script
**File**: `/scripts/validate_rate_series.py`

Create a script that:
1. Fetches DGS3MO from FRED for last year
2. Fetches BIL ETF returns for comparison
3. Shows correlation and differences
4. Validates conversion math
5. Tests all frequencies

## Success Criteria
- [ ] Rate series returns correctly converted from annual to period
- [ ] Missing data handled appropriately (forward fill)
- [ ] Cash rate successfully fetched from FRED
- [ ] ETF fallback works when FRED unavailable
- [ ] All tests pass
- [ ] Validation script shows reasonable correlation between FRED and ETF

## Example Usage
```python
# After implementation, this should work:
fetcher = TotalReturnFetcher()

# Direct rate series fetch
rf_returns = fetcher.fetch_rate_series_returns(
    series_code="DGS3MO",
    source="FRED",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    frequency="monthly"
)

# Through exposure universe
universe = ExposureUniverse()
universe.load_config('config/exposure_universe.yaml')
results = fetcher.fetch_universe_returns(universe, start, end)
cash_returns = results['cash_rate']['returns']  # Should work!
```

## Implementation Notes
- Keep error messages descriptive for debugging
- Log at INFO level when using fallback sources
- Consider adding rate series caching since FRED has rate limits
- Document the conversion formulas clearly in docstrings
- Make sure the returned Series has a meaningful name attribute

## References
- FRED API docs: https://fred.stlouisfed.org/docs/api/fred/
- Existing FRED integration: `/src/data/fred_data.py`
- Cash rate implementation guide: `/PROJECT_CONTEXT/TASKS/cash_rate_implementation_guide.md`
