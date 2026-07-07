# Quick Implementation for Rate Series Support

Add this to your TotalReturnFetcher or create a specialized handler:

```python
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

def fetch_risk_free_rate_returns(start_date, end_date, series='DGS3MO'):
    """
    Fetch risk-free rate and convert to return series.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        series: FRED series ID (default: DGS3MO for 3-month Treasury)
    
    Returns:
        pd.Series: Daily returns based on the risk-free rate
    """
    # Fetch rate data from FRED
    rate_data = pdr.get_data_fred(series, start_date, end_date)
    
    # FRED provides annualized rates in percentage (e.g., 5.25 for 5.25%)
    # Convert to decimal
    annual_rates = rate_data[series] / 100
    
    # Fill missing values (rates don't update every day)
    annual_rates = annual_rates.fillna(method='ffill')
    
    # Convert to daily returns
    # Using 252 trading days per year convention
    daily_returns = annual_rates / 252
    
    # Create a series with proper name
    daily_returns.name = 'risk_free_rate'
    
    return daily_returns

# Example usage:
# rf_returns = fetch_risk_free_rate_returns(datetime(2020, 1, 1), datetime(2023, 12, 31))
# print(f"Average daily risk-free rate: {rf_returns.mean():.4%}")
# print(f"Annualized: {rf_returns.mean() * 252:.2%}")
```

## Integration with Existing Code

In your data fetcher, add a special case for rate series:

```python
def fetch_implementation_data(self, implementation, start_date, end_date):
    impl_type = implementation['type']
    
    if impl_type == 'rate_series':
        # Special handling for interest rate data
        source = implementation['source']
        series = implementation['series']
        
        if source == 'FRED':
            return fetch_risk_free_rate_returns(start_date, end_date, series)
        else:
            raise ValueError(f"Unknown rate series source: {source}")
            
    elif impl_type in ['etf', 'etf_average', 'fund', 'fund_average']:
        # Existing price-based implementation
        return self.fetch_price_based_returns(implementation, start_date, end_date)
    
    # ... other implementation types
```

## Alternative: Just Use ETFs for Now

If you want to skip rate series for now and just use the ETF implementations:

```yaml
# In exposure_universe.yaml, modify cash_rate to only use ETFs:
- id: cash_rate
  name: "Cash/Risk-Free Rate"
  description: "Short-term risk-free rate for leverage cost modeling"
  category: "nominal_fixed_income"
  implementations:
    - type: "etf_average"
      tickers: ["BIL", "SHV", "SGOV"]  # 0-3 month T-bills
      # Rate series commented out until implemented
      # - type: "rate_series"
      #   source: "FRED"
      #   series: "DGS3MO"
```

The ETFs (BIL, SHV, SGOV) should give you very similar returns to the risk-free rate and are easier to work with since they're price series.
