# Cash/Risk-Free Rate Implementation Guide

## The Specific Issue

The cash/risk-free rate exposure cannot be fetched like a normal ETF because:
1. It's a rate series (percentage) not a price series
2. The data comes from FRED (Federal Reserve Economic Data) not yfinance
3. The "returns" need to be calculated differently

## Current Configuration

From `/config/exposure_universe.yaml`:
```yaml
- id: cash_rate
  name: "Cash/Risk-Free Rate"
  implementations:
    - type: "etf_average"
      tickers: ["BIL", "SHV", "SGOV"]  # 0-3 month T-bills
    - type: "rate_series"
      source: "FRED"
      series: "DGS3MO"  # 3-Month Treasury Rate
```

## What We Already Have

The FRED integration already exists in `/src/data/fred_data.py`:
```python
# This already works!
fred_fetcher = FREDDataFetcher()
rates = fred_fetcher.fetch_risk_free_rate(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    maturity="3m",
    frequency="daily"
)
# Returns Series of rates in decimal form (5.25% -> 0.0525)
```

## What's Missing

The TotalReturnFetcher needs to:
1. Recognize when an implementation type is "rate_series"
2. Use the FRED fetcher instead of yfinance
3. Convert the rate appropriately for the requested return frequency

## Implementation Solution

### Step 1: Extend TotalReturnFetcher to Handle Rate Series

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
    if source != "FRED":
        raise ValueError(f"Unsupported rate series source: {source}")
    
    # Use existing FRED fetcher
    fred_fetcher = FREDDataFetcher()
    
    # Fetch the rate data (comes as annualized decimal, e.g., 0.0525 for 5.25%)
    rates = fred_fetcher.fetch_series(
        series_code=series_code,
        start_date=start_date,
        end_date=end_date,
        frequency='D'  # Always fetch daily, then convert
    )
    
    # FRED returns percentages, convert to decimal
    rates = rates / 100.0
    
    # Convert annualized rates to period returns
    if frequency == "daily":
        # Daily return from annualized rate
        # Assuming 252 trading days per year
        period_returns = rates / 252
    elif frequency == "monthly":
        # Monthly return from annualized rate
        # Using geometric conversion: (1 + r)^(1/12) - 1
        period_returns = (1 + rates) ** (1/12) - 1
    elif frequency == "annual":
        # Already annualized
        period_returns = rates
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    # Forward fill missing values (weekends, holidays)
    period_returns = period_returns.fillna(method='ffill')
    
    return period_returns
```

### Step 2: Integrate into Main Fetch Method

```python
def fetch_exposure_returns(
    self,
    exposure_config: dict,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily"
) -> pd.Series:
    """Fetch returns for an exposure using its configuration."""
    
    # Try each implementation in order
    for impl in exposure_config['implementations']:
        try:
            if impl['type'] == 'rate_series':
                # NEW: Handle rate series
                returns = self.fetch_rate_series_returns(
                    series_code=impl['series'],
                    source=impl['source'],
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency
                )
                if not returns.empty:
                    return returns
                    
            elif impl['type'] == 'etf_average':
                # Existing ETF handling
                returns = self.fetch_etf_average_returns(
                    tickers=impl['tickers'],
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency
                )
                if not returns.empty:
                    return returns
                    
        except Exception as e:
            logger.warning(f"Failed to fetch {impl['type']}: {e}")
            continue
    
    raise ValueError(f"Could not fetch data for exposure")
```

### Step 3: Handle the ETF Fallback

The cash_rate exposure also has ETF alternatives (BIL, SHV, SGOV). These can be used as fallback:

```python
def fetch_cash_rate_returns(
    self,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily"
) -> pd.Series:
    """Specialized method for cash/risk-free rate with fallback logic."""
    
    # First try FRED data (preferred for accuracy)
    try:
        returns = self.fetch_rate_series_returns(
            series_code="DGS3MO",
            source="FRED",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency
        )
        if not returns.empty:
            logger.info("Using FRED DGS3MO for cash rate")
            return returns
    except Exception as e:
        logger.warning(f"FRED data unavailable: {e}")
    
    # Fallback to T-bill ETFs
    tbill_etfs = ["BIL", "SHV", "SGOV"]
    for ticker in tbill_etfs:
        try:
            returns = self.fetch_price_returns(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency
            )
            if not returns.empty:
                logger.info(f"Using {ticker} ETF for cash rate")
                return returns
        except:
            continue
    
    raise ValueError("Could not fetch cash rate from any source")
```

## Testing the Implementation

```python
# Test script to verify cash rate fetching works
def test_cash_rate_fetching():
    fetcher = TotalReturnFetcher()
    
    # Test dates
    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31)
    
    # Test FRED rate series
    print("Testing FRED DGS3MO fetch...")
    try:
        rf_returns = fetcher.fetch_rate_series_returns(
            series_code="DGS3MO",
            source="FRED",
            start_date=start,
            end_date=end,
            frequency="daily"
        )
        print(f"✓ Fetched {len(rf_returns)} daily returns")
        print(f"  Sample rate: {rf_returns.iloc[0]:.6f} ({rf_returns.iloc[0]*252*100:.2f}% annualized)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test ETF fallback
    print("\nTesting BIL ETF fetch...")
    try:
        bil_returns = fetcher.fetch_price_returns(
            ticker="BIL",
            start_date=start,
            end_date=end,
            frequency="daily"
        )
        print(f"✓ Fetched {len(bil_returns)} daily returns")
        print(f"  Sample return: {bil_returns.iloc[100]:.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Compare if both available
    if 'rf_returns' in locals() and 'bil_returns' in locals():
        print("\nComparing FRED vs ETF returns...")
        # Align dates
        common_dates = rf_returns.index.intersection(bil_returns.index)
        correlation = rf_returns.loc[common_dates].corr(bil_returns.loc[common_dates])
        print(f"Correlation: {correlation:.4f}")
```

## Key Points to Remember

1. **FRED rates are annualized percentages** - must convert to period returns
2. **Forward fill missing values** - FRED data has gaps on weekends/holidays
3. **ETF fallback** - BIL, SHV, SGOV are good alternatives if FRED fails
4. **Leverage cost modeling** - This rate is used to calculate the cost of leverage in the optimization

## Success Validation

The implementation is correct when:
- Daily returns from FRED DGS3MO are approximately 1/252 of the annual rate
- Monthly returns use geometric conversion: (1 + annual_rate)^(1/12) - 1
- Missing values are handled appropriately
- ETF fallback produces similar (but not identical) returns
- The returns can be used in the optimization engine for leverage cost calculations
