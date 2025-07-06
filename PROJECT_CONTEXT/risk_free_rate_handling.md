# Handling Cash/Risk-Free Rate Data

## The Issue
Rate series from FRED provide interest rates (e.g., 5.25%), not prices that we can calculate returns from. This requires special handling.

## Understanding Rate vs Price Data

### Price Series (ETFs like BIL, SHV)
- Provides daily prices
- Calculate returns: (Price_t - Price_t-1) / Price_t-1
- Already incorporates the risk-free rate through price appreciation

### Rate Series (FRED DGS3MO)
- Provides annualized interest rates
- Need to convert to returns based on the period
- Example: 5.25% annual rate = 0.0525/252 daily return

## Implementation Solutions

### Option 1: Convert Rate to Return Series (Recommended)
```python
def convert_rate_to_returns(rate_series, frequency='daily'):
    """
    Convert interest rate series to return series.
    
    Args:
        rate_series: Series of annualized interest rates (e.g., 0.0525 for 5.25%)
        frequency: 'daily', 'monthly', or 'annual'
    
    Returns:
        Series of returns for the specified frequency
    """
    if frequency == 'daily':
        # Convert annual rate to daily return
        # Assuming 252 trading days per year
        daily_returns = rate_series / 100 / 252  # Divide by 100 if rates are in percentage
    elif frequency == 'monthly':
        # Convert annual rate to monthly return
        daily_returns = rate_series / 100 / 12
    else:  # annual
        daily_returns = rate_series / 100
        
    # Forward fill any missing values (rates don't change every day)
    daily_returns = daily_returns.fillna(method='ffill')
    
    return daily_returns
```

### Option 2: Create Synthetic Price Series
```python
def create_synthetic_price_series(rate_series, start_value=100):
    """
    Create a synthetic price series that grows at the risk-free rate.
    
    Args:
        rate_series: Series of annualized interest rates
        start_value: Starting price (default 100)
    
    Returns:
        Series of synthetic prices
    """
    daily_returns = convert_rate_to_returns(rate_series, 'daily')
    
    # Create cumulative price series
    price_series = start_value * (1 + daily_returns).cumprod()
    
    return price_series
```

### Option 3: Hybrid Approach (Best)
```python
class RiskFreeRateFetcher:
    """Specialized fetcher for risk-free rate data."""
    
    def fetch_returns(self, start_date, end_date):
        # First try ETF approach (BIL, SHV, SGOV)
        try:
            etf_data = self._fetch_etf_returns(['BIL', 'SHV', 'SGOV'], start_date, end_date)
            if etf_data is not None:
                return etf_data.mean(axis=1)  # Average of available ETFs
        except:
            pass
        
        # Fallback to FRED rate series
        try:
            import pandas_datareader as pdr
            
            # Fetch 3-month Treasury rate
            rate_data = pdr.get_data_fred('DGS3MO', start_date, end_date)
            
            # Convert to returns
            returns = self.convert_rate_to_returns(rate_data['DGS3MO'])
            
            return returns
            
        except:
            raise ValueError("Unable to fetch risk-free rate data")
```

## Quick Fix for Current Implementation

If you want to just use the ETF implementations for now:

```python
# In exposure_universe.yaml, temporarily comment out the rate_series implementation:
implementations:
  - type: "etf_average"
    tickers: ["BIL", "SHV", "SGOV"]  # These should work fine
  # - type: "rate_series"  # TODO: Implement rate series support
  #   source: "FRED"
  #   series: "DGS3MO"
```

## Long-term Solution

The TotalReturnFetcher should be extended to handle different data types:
1. Price series (current implementation)
2. Rate series (convert to returns)
3. Index levels (calculate returns from levels)
4. Direct return series (use as-is)

## For Leverage Cost Modeling

Remember that for leverage cost calculations, you need:
```python
leverage_cost = borrowed_amount * (risk_free_rate + spread)

# Where spread is typically 0.5% - 1.5% for ETFs
```

The risk-free rate is essential for accurate fund replication of Return Stacked products.
