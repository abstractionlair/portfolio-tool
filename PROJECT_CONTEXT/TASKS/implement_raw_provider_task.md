# Task: Implement Raw Data Provider

**Status**: TODO  
**Priority**: HIGH  
**Estimated Time**: 2 days  
**Dependencies**: test_data_interfaces_task.md completed

## Overview

Implement the `RawDataProvider` that fetches data from external sources (yfinance, FRED) without any transformation. This provider should pass all tests defined in the test suite and serve as the foundation for other providers.

## Implementation Plan

### 1. `src/data/providers/raw_provider.py`

Create the main raw data provider that coordinates between different sources:

```python
class DefaultRawDataProvider(RawDataProvider):
    """
    Default implementation that fetches from yfinance and FRED.
    
    This provider:
    - Fetches security data from yfinance
    - Fetches economic data from FRED
    - Does NO computation (no return calculation)
    - Does basic data cleaning (remove nulls at edges)
    - Validates all inputs according to interface
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._init_sources()
    
    def _init_sources(self):
        """Initialize connections to data sources."""
        # Setup yfinance session with retries
        # Setup FRED connection
        # Setup any other sources
    
    def get_data(
        self,
        data_type: RawDataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        **kwargs
    ) -> pd.Series:
        """Implementation of main data fetching."""
        # 1. Validate inputs
        validate_ticker_requirement(data_type, ticker)
        validate_date_range(start, end)
        
        # 2. Route to appropriate source
        if data_type.category in [DataTypeCategory.SECURITY_RAW]:
            return self._fetch_security_data(data_type, ticker, start, end, frequency)
        else:
            return self._fetch_economic_data(data_type, start, end, frequency, **kwargs)
```

### 2. `src/data/providers/sources/yfinance_source.py`

Implement yfinance integration:

```python
class YFinanceSource:
    """Handles all yfinance data fetching."""
    
    def fetch_ohlcv(self, ticker: str, start: date, end: date, frequency: str) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance."""
        # Map frequency to yfinance interval
        interval_map = {
            "daily": "1d",
            "weekly": "1wk", 
            "monthly": "1mo"
        }
        
        # Fetch with retry logic
        # Handle yfinance quirks (timezone, adjusted close)
        # Return clean DataFrame
    
    def fetch_dividends(self, ticker: str, start: date, end: date) -> pd.Series:
        """Fetch dividend data."""
        # yfinance dividends need special handling
        # Make sure to return actual amounts, not adjusted
    
    def fetch_splits(self, ticker: str, start: date, end: date) -> pd.Series:
        """Fetch split data."""
        # Return split ratios
```

### 3. `src/data/providers/sources/fred_source.py`

Implement FRED integration:

```python
class FREDSource:
    """Handles all FRED data fetching."""
    
    # Mapping of our types to FRED series IDs
    SERIES_MAP = {
        RawDataType.TREASURY_3M: "DGS3MO",
        RawDataType.TREASURY_1Y: "DGS1",
        RawDataType.TREASURY_10Y: "DGS10",
        RawDataType.TIPS_5Y: "DFII5",
        RawDataType.CPI_INDEX: "CPIAUCSL",
        # ... etc
    }
    
    def fetch_series(self, data_type: RawDataType, start: date, end: date, frequency: str) -> pd.Series:
        """Fetch FRED series data."""
        series_id = self.SERIES_MAP.get(data_type)
        if not series_id:
            raise DataNotAvailableError(f"No FRED series for {data_type}")
        
        # Use existing FRED fetching logic
        # Handle FRED-specific issues (rate limiting, missing data)
        # Convert frequency if needed
```

### 4. `src/data/providers/sources/csv_source.py`

Implement CSV fallback source:

```python
class CSVSource:
    """Fallback source for data in CSV files."""
    
    def __init__(self, data_dir: str = "./data/manual"):
        self.data_dir = Path(data_dir)
    
    def is_available(self, ticker: str, data_type: RawDataType) -> bool:
        """Check if we have CSV data for this request."""
        filename = self._get_filename(ticker, data_type)
        return (self.data_dir / filename).exists()
    
    def fetch_data(self, ticker: str, data_type: RawDataType, start: date, end: date) -> pd.Series:
        """Load data from CSV."""
        filename = self._get_filename(ticker, data_type)
        filepath = self.data_dir / filename
        
        # Read CSV with proper date parsing
        # Filter to requested date range
        # Return as Series
```

## Key Implementation Points

### 1. NO Computation in Raw Provider
```python
# WRONG - Don't compute returns
def get_data(self, data_type, ...):
    if data_type == RawDataType.SIMPLE_RETURN:  # NO!
        prices = self._fetch_prices(...)
        return prices.pct_change()
        
# RIGHT - Raw provider only handles raw types
def get_data(self, data_type, ...):
    if not isinstance(data_type, RawDataType):
        raise ValueError(f"Raw provider only handles RawDataType, got {type(data_type)}")
```

### 2. Handle Missing Data Appropriately
```python
def _clean_series(self, series: pd.Series, start: date, end: date) -> pd.Series:
    """Clean and validate fetched data."""
    # Remove leading/trailing NaN
    series = series.dropna()
    
    # Check if we have any data in range
    if series.empty:
        raise DataNotAvailableError(f"No data available for date range")
    
    # Warn if partial data
    if series.index[0] > pd.Timestamp(start):
        logger.warning(f"Data only available from {series.index[0]}")
    
    return series
```

### 3. Proper Error Handling
```python
def _fetch_security_data(self, data_type, ticker, start, end, frequency):
    """Fetch security data with proper error handling."""
    try:
        if data_type == RawDataType.OHLCV:
            return self.yfinance_source.fetch_ohlcv(ticker, start, end, frequency)
        # ... other types
    except HTTPError as e:
        if e.response.status_code == 404:
            raise DataNotAvailableError(f"Ticker {ticker} not found")
        else:
            raise DataSourceError(f"YFinance error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching {ticker}: {e}")
        raise DataSourceError(f"Failed to fetch {ticker}: {e}")
```

### 4. Frequency Handling
```python
def _handle_frequency(self, data: pd.Series, requested_freq: str, source_freq: str) -> pd.Series:
    """Handle frequency conversion if source doesn't match request."""
    # Only downsample, never upsample
    freq_enum = Frequency(requested_freq)
    source_enum = Frequency(source_freq)
    
    if not source_enum.can_convert_to(freq_enum):
        logger.warning(
            f"Cannot convert {source_freq} to {requested_freq}, "
            f"returning {source_freq} data"
        )
        return data
    
    # Perform conversion
    if freq_enum == Frequency.MONTHLY:
        return data.resample('ME').last()
    # ... etc
```

## Testing Requirements

The implementation must pass all tests in:
- `test_data_provider_contract.py`
- `test_raw_provider_contract.py`

Additional implementation-specific tests to write:
- Test fallback between sources
- Test error handling for each source
- Test frequency conversion
- Test data cleaning logic
- Mock external APIs for unit tests

## Success Criteria

- [ ] Passes all contract tests
- [ ] Fetches real data from yfinance (integration test)
- [ ] Fetches real data from FRED (integration test)  
- [ ] Falls back to CSV when other sources fail
- [ ] Properly validates all inputs
- [ ] Clean error messages for all failure modes
- [ ] No return computation (raw data only)
- [ ] Handles missing data gracefully
- [ ] Performance: <2s for single ticker daily data

## Configuration

Create configuration for the provider:

```yaml
# config/data_sources.yaml
raw_data_provider:
  cache_dir: ~/.portfolio_optimizer/cache
  
  yfinance:
    session_timeout: 10
    max_retries: 3
    rate_limit: 2000/hour
    
  fred:
    api_key: ${FRED_API_KEY}  # From environment
    rate_limit: 120/minute
    
  csv:
    data_dir: ./data/manual
    date_column: Date
    date_format: "%Y-%m-%d"
```

## Notes

1. Start with yfinance + FRED, add other sources later
2. Use existing code where possible (fred_data.py, market_data.py)
3. Focus on reliability over performance initially
4. Log all data source selection decisions
5. Consider adding source attribution to returned data

This implementation will provide the foundation for all other data layer components.
