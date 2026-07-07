# TotalReturnFetcher Implementation Plan

**Status**: PLANNING  
**Priority**: Critical - Blocking exposure universe implementation  
**Complexity**: Medium  

## Overview

The TotalReturnFetcher needs to handle multiple data types:
1. **Price-based assets** (ETFs, mutual funds) - fetch prices and calculate returns
2. **Rate series** (risk-free rates from FRED) - rates ARE returns, need proper conversion
3. **Composite assets** (weighted combinations)
4. **Index data** (future extension)

## Design Specification

### Core Class Structure

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime

class DataSource(ABC):
    """Abstract base class for different data sources."""
    
    @abstractmethod
    def fetch_data(
        self, 
        identifier: str, 
        start_date: datetime, 
        end_date: datetime,
        **kwargs
    ) -> pd.Series:
        """Fetch raw data from source."""
        pass
    
    @abstractmethod
    def to_returns(
        self, 
        data: pd.Series, 
        frequency: str = "daily"
    ) -> pd.Series:
        """Convert raw data to returns."""
        pass


class PriceDataSource(DataSource):
    """Handles price-based assets (ETFs, stocks, mutual funds)."""
    
    def __init__(self, source: str = "yfinance"):
        self.source = source
        self.fetcher = self._get_fetcher(source)
    
    def fetch_data(self, ticker: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.Series:
        """Fetch adjusted close prices including dividends."""
        # Use yfinance to get Adjusted Close
        # Handle missing data gracefully
        # Validate data quality
        pass
    
    def to_returns(self, prices: pd.Series, frequency: str = "daily") -> pd.Series:
        """Convert prices to total returns."""
        # Standard price return calculation: (P_t / P_{t-1}) - 1
        # Already includes dividends via Adjusted Close
        pass


class RateDataSource(DataSource):
    """Handles rate series (FRED treasury rates, etc.)."""
    
    def __init__(self):
        from src.data.fred_data import FREDDataFetcher
        self.fred_fetcher = FREDDataFetcher()
    
    def fetch_data(self, series_code: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.Series:
        """Fetch rate series from FRED."""
        # Use existing FREDDataFetcher
        # Rates come as annualized percentages
        pass
    
    def to_returns(self, rates: pd.Series, frequency: str = "daily") -> pd.Series:
        """Convert annualized rates to period returns."""
        # CRITICAL: Rates are annualized percentages
        # Need to convert to period returns based on frequency
        # Example: 3% annual rate = 0.03/252 daily return
        pass


class TotalReturnFetcher:
    """Main class that orchestrates fetching total returns from any source."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.price_source = PriceDataSource()
        self.rate_source = RateDataSource()
        
    def fetch_returns(
        self,
        identifier: str,
        data_type: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily",
        **kwargs
    ) -> pd.Series:
        """
        Fetch total returns for any asset type.
        
        Args:
            identifier: Ticker, FRED series code, etc.
            data_type: "price", "rate_series", "composite"
            start_date: Start date
            end_date: End date
            frequency: "daily", "monthly", "annual"
            **kwargs: Additional parameters (e.g., components for composite)
            
        Returns:
            Series of total returns at specified frequency
        """
        if data_type == "price":
            return self._fetch_price_returns(identifier, start_date, end_date, frequency)
        elif data_type == "rate_series":
            return self._fetch_rate_returns(identifier, start_date, end_date, frequency)
        elif data_type == "composite":
            return self._fetch_composite_returns(kwargs['components'], start_date, end_date, frequency)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _fetch_price_returns(self, ticker: str, start: datetime, end: datetime, freq: str) -> pd.Series:
        """Fetch returns for price-based assets."""
        prices = self.price_source.fetch_data(ticker, start, end)
        returns = self.price_source.to_returns(prices, freq)
        return returns
    
    def _fetch_rate_returns(self, series_code: str, start: datetime, end: datetime, freq: str) -> pd.Series:
        """Fetch returns for rate series."""
        rates = self.rate_source.fetch_data(series_code, start, end)
        returns = self.rate_source.to_returns(rates, freq)
        return returns
    
    def _fetch_composite_returns(
        self, 
        components: List[Dict[str, Union[str, float]]], 
        start: datetime, 
        end: datetime, 
        freq: str
    ) -> pd.Series:
        """Calculate returns for weighted composite."""
        # Fetch returns for each component
        # Align dates
        # Calculate weighted average returns
        pass
```

## Implementation Details

### 1. Rate Series Conversion (Critical!)

```python
def convert_annualized_rate_to_returns(
    annualized_rates: pd.Series,  # e.g., 3.5 means 3.5% annual
    target_frequency: str
) -> pd.Series:
    """Convert annualized percentage rates to period returns."""
    
    # First convert percentage to decimal
    decimal_rates = annualized_rates / 100.0  # 3.5% -> 0.035
    
    # Then convert to target frequency
    if target_frequency == "daily":
        # Assuming 252 trading days per year
        daily_returns = decimal_rates / 252
    elif target_frequency == "monthly":
        # Convert annual to monthly
        # Using (1 + r)^(1/12) - 1 for geometric conversion
        monthly_returns = (1 + decimal_rates) ** (1/12) - 1
    elif target_frequency == "annual":
        # Already annual, just return decimal form
        annual_returns = decimal_rates
    else:
        raise ValueError(f"Unsupported frequency: {target_frequency}")
    
    return returns
```

### 2. Handling Missing Data

```python
class DataAvailability:
    """Check and handle data availability issues."""
    
    def check_ticker_availability(self, ticker: str) -> bool:
        """Quick check if ticker has data in yfinance."""
        try:
            data = yf.download(ticker, period="1d", progress=False)
            return not data.empty
        except:
            return False
    
    def get_fallback_tickers(self, primary_ticker: str) -> List[str]:
        """Get fallback tickers from configuration."""
        # Look up in exposure universe config
        pass
```

### 3. Data Alignment and Frequency Conversion

```python
def align_and_resample_returns(
    returns_dict: Dict[str, pd.Series],
    target_frequency: str
) -> pd.DataFrame:
    """Align multiple return series and convert to target frequency."""
    
    # Create DataFrame from dict
    df = pd.DataFrame(returns_dict)
    
    # Forward fill missing data (up to a limit)
    df = df.fillna(method='ffill', limit=5)
    
    # Resample if needed
    if target_frequency == "monthly":
        # Compound daily returns to monthly
        df = (1 + df).resample('M').prod() - 1
    elif target_frequency == "annual":
        # Compound to annual
        df = (1 + df).resample('A').prod() - 1
    
    return df
```

## Integration with Exposure Universe

The TotalReturnFetcher will be used by the ExposureUniverse class:

```python
class ExposureUniverse:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.fetcher = TotalReturnFetcher()
    
    def fetch_exposure_returns(
        self, 
        exposure_id: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily"
    ) -> pd.Series:
        """Fetch returns for a specific exposure."""
        
        exposure = self.get_exposure(exposure_id)
        
        # Try each implementation in order
        for impl in exposure['implementations']:
            try:
                if impl['type'] == 'etf_average':
                    returns = self._fetch_etf_average(impl['tickers'], start_date, end_date, frequency)
                elif impl['type'] == 'fund':
                    returns = self.fetcher.fetch_returns(
                        impl['ticker'], 'price', start_date, end_date, frequency
                    )
                elif impl['type'] == 'rate_series':
                    returns = self.fetcher.fetch_returns(
                        impl['series'], 'rate_series', start_date, end_date, frequency
                    )
                
                if returns is not None and not returns.empty:
                    return returns
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {impl}: {e}")
                continue
        
        raise ValueError(f"Could not fetch data for exposure {exposure_id}")
```

## Special Considerations

### 1. Cash/Risk-Free Rate
- FRED series DGS3MO provides 3-month Treasury rate
- Data is annualized percentage (e.g., 5.25 = 5.25% annual)
- May have missing values on weekends/holidays
- Need to forward-fill for daily returns

### 2. Mutual Fund Data
- Many mutual funds (ABYIX, AHLIX, etc.) not available in yfinance
- Need fallback strategy:
  1. Try primary mutual fund tickers
  2. Fall back to ETF alternatives
  3. Use composite if necessary
  4. Log what was actually used

### 3. Total Return Validation
```python
def validate_total_returns(price_series: pd.Series, ticker: str) -> bool:
    """Validate that we're getting total returns not just price returns."""
    # Check if 'Adj Close' was used
    # Compare with dividend dates if available
    # Log warnings if suspicious
```

## Testing Strategy

1. **Unit Tests**
   - Test rate conversion logic with known values
   - Test frequency conversion
   - Test missing data handling

2. **Integration Tests**
   - Test fetching SPY (known good ETF)
   - Test fetching DGS3MO (FRED rate series)
   - Test composite calculations

3. **Validation Tests**
   - Compare returns with manual calculations
   - Verify total returns include dividends
   - Check data alignment

## Example Usage

```python
# Initialize fetcher
fetcher = TotalReturnFetcher()

# Fetch ETF returns (price-based)
spy_returns = fetcher.fetch_returns(
    "SPY", 
    data_type="price",
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    frequency="daily"
)

# Fetch risk-free rate returns (rate series)
rf_returns = fetcher.fetch_returns(
    "DGS3MO",
    data_type="rate_series", 
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    frequency="daily"
)

# Fetch composite returns
trend_returns = fetcher.fetch_returns(
    "trend_composite",
    data_type="composite",
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    frequency="monthly",
    components=[
        {"ticker": "DBMF", "weight": 0.5},
        {"ticker": "KMLM", "weight": 0.5}
    ]
)
```

## Success Criteria

- [ ] Correctly fetches and converts FRED rate series to returns
- [ ] Handles missing mutual fund data with appropriate fallbacks
- [ ] Calculates true total returns (including dividends)
- [ ] Properly aligns data across different sources
- [ ] Converts between daily/monthly/annual frequencies correctly
- [ ] Integrates seamlessly with ExposureUniverse class
- [ ] Comprehensive test coverage
- [ ] Clear logging of data sources used

## Next Steps

1. Implement core TotalReturnFetcher class
2. Add comprehensive unit tests
3. Test with real data from exposure universe
4. Document any data quirks discovered
5. Integrate with ExposureUniverse class
