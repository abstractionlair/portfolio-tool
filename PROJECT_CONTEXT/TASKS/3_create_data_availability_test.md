# Task 3: Create Comprehensive Data Availability Test

**Status**: TODO  
**Priority**: HIGH - Validation and debugging  
**Estimated Time**: 2 hours  
**Dependencies**: Tasks 1 & 2

## Context
We need to validate which data sources are actually available and working. This will help identify any issues with mutual funds, FRED data, or other sources before we build more features on top.

## Problem
- Some mutual funds might not be in yfinance
- Need to verify FRED integration works
- Need to identify which exposures have data gaps
- Need to validate the fallback mechanism works

## Requirements

### 1. Create Data Availability Test Script
Location: `/scripts/test_data_availability.py`

Structure:
```python
#!/usr/bin/env python
"""
Comprehensive test of data availability for all exposures.
Tests each ticker and data source to identify what's available.
"""

def test_ticker_availability(ticker: str, start_date: datetime, end_date: datetime) -> dict:
    """Test if a ticker has data available."""
    # Returns: {
    #     'available': bool,
    #     'data_points': int,
    #     'date_range': tuple,
    #     'source': str,
    #     'error': Optional[str]
    # }

def test_exposure_availability(exposure: dict, fetcher: TotalReturnFetcher) -> dict:
    """Test all implementations for an exposure."""
    # Try each implementation in order
    # Return details about what worked

def generate_availability_report(results: dict) -> pd.DataFrame:
    """Generate a summary report of data availability."""
    # Create DataFrame with columns:
    # - Exposure ID
    # - Exposure Name
    # - Primary Implementation
    # - Data Available
    # - Date Range
    # - Fallback Used
    # - Notes

def main():
    """Run comprehensive data availability test."""
```

### 2. Test Categories

#### Individual Ticker Tests
```python
# Test each unique ticker mentioned in the config
tickers_to_test = extract_all_tickers(universe)
ticker_results = {}

for ticker in tickers_to_test:
    result = test_ticker_availability(ticker, start_date, end_date)
    ticker_results[ticker] = result
    print(f"{ticker}: {'✓' if result['available'] else '✗'} "
          f"({result.get('data_points', 0)} data points)")
```

#### FRED Data Tests
```python
# Test FRED series
fred_series = ['DGS3MO', 'DGS1', 'DGS10', 'DFII10', 'T10YIE']
fred_results = {}

for series in fred_series:
    try:
        data = fred_fetcher.fetch_series(series, start_date, end_date)
        fred_results[series] = {
            'available': not data.empty,
            'data_points': len(data)
        }
    except Exception as e:
        fred_results[series] = {
            'available': False,
            'error': str(e)
        }
```

#### Exposure-Level Tests
```python
# Test each exposure with all its implementations
exposure_results = {}

for exposure_id in universe.list_exposures():
    exposure = universe.get_exposure(exposure_id)
    result = test_exposure_availability(exposure, fetcher)
    exposure_results[exposure_id] = result
```

### 3. Output Format

Create multiple output formats:

#### Console Output
```
========================================
DATA AVAILABILITY TEST RESULTS
========================================

TICKER AVAILABILITY:
✓ SPY: 252 daily prices (2023-01-01 to 2023-12-31)
✓ AGG: 252 daily prices (2023-01-01 to 2023-12-31)
✗ ABYIX: No data available (Error: Ticker not found)
...

FRED DATA AVAILABILITY:
✓ DGS3MO: 252 observations
✓ DGS10: 252 observations
...

EXPOSURE AVAILABILITY:
✓ US Large Cap Equity: Using primary (SPY)
✓ International Developed: Using primary (EFA)
⚠ Trend Following: Using fallback (DBMF) - Primary failed
✗ Factor/Style - Other: No implementation available
...

SUMMARY:
- Total Exposures: 16
- Fully Available: 12 (75%)
- Using Fallback: 3 (19%)
- Not Available: 1 (6%)
```

#### CSV Report
Save to: `/data/data_availability_report.csv`

#### JSON Report
Save to: `/data/data_availability_report.json`
With full details for programmatic use.

### 4. Add Special Case Tests

```python
def test_special_cases():
    """Test specific known issues."""
    
    # Test cash rate with both FRED and ETF
    print("\nTesting Cash Rate implementations:")
    
    # Test FRED DGS3MO
    fred_cash = test_rate_series('DGS3MO', 'FRED')
    print(f"  FRED DGS3MO: {'✓' if fred_cash['available'] else '✗'}")
    
    # Test ETF fallbacks
    for ticker in ['BIL', 'SHV', 'SGOV']:
        etf_result = test_ticker_availability(ticker)
        print(f"  {ticker} ETF: {'✓' if etf_result['available'] else '✗'}")
```

## Testing Instructions

1. Run the full test:
```bash
python scripts/test_data_availability.py
```

2. Check specific exposures:
```bash
python scripts/test_data_availability.py --exposure trend_following
```

3. Test date ranges:
```bash
python scripts/test_data_availability.py --start 2020-01-01 --end 2023-12-31
```

## Success Criteria
- [ ] Script identifies all available and missing data sources
- [ ] Clear report showing which exposures have data
- [ ] Identifies which mutual funds are not in yfinance
- [ ] Validates FRED integration works
- [ ] Shows which exposures use fallbacks
- [ ] Generates actionable report for fixing data gaps
- [ ] Handles errors gracefully with clear messages
- [ ] Can be run repeatedly to check data status

## Expected Findings
Based on the configuration, we expect:
- Most ETFs should be available (SPY, AGG, etc.)
- Some mutual funds might fail (ABYIX, AHLIX, QMNIX, etc.)
- FRED data should work for all series
- Some exposures will need to use fallback implementations

## Next Steps Based on Results
1. For missing mutual funds: Implement alternative data sources
2. For failed exposures: Add more fallback options
3. For partial data: Document date range limitations
4. Update exposure config based on findings

## Notes
- This is a diagnostic tool - it should be runnable anytime
- Make the output clear and actionable
- Save results for documentation
- This will guide what additional data work is needed
