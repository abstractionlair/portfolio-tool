# Task: Multi-Frequency Data Support

**Status**: TODO  
**Priority**: HIGH - Enables better analysis across time horizons  
**Estimated Time**: 6-8 hours  
**Dependencies**: Task 3 (Data Availability) should be complete

## Objective
Enable the system to work with data at different frequencies (daily, weekly, monthly, quarterly) with proper return compounding and frequency conversion.

## Motivation
- Different investment horizons benefit from different data frequencies
- Reduce noise in estimates for longer-term strategies
- Some patterns only visible at certain frequencies
- Match frequency to rebalancing schedule

## Implementation Tasks

### 1. Enhance Data Fetcher
**File**: `/src/data/total_returns.py`

Update fetch method to support frequencies:
```python
def fetch_total_returns(
    self,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily",  # NEW: daily, weekly, monthly, quarterly
    compound_method: str = "log"  # NEW: log, simple
) -> pd.Series:
    """
    Fetch total returns at specified frequency.
    
    Args:
        frequency: Data frequency - 'daily', 'weekly', 'monthly', 'quarterly'
        compound_method: How to compound returns - 'log' or 'simple'
    """
    # Always fetch daily data first
    daily_prices = self._fetch_daily_data(ticker, start_date, end_date)
    
    if frequency == "daily":
        return self._prices_to_returns(daily_prices, method=compound_method)
    
    # Convert to requested frequency
    freq_map = {
        'weekly': 'W-FRI',    # Week ending Friday
        'monthly': 'M',       # Month end
        'quarterly': 'Q'      # Quarter end
    }
    
    # Resample to requested frequency
    freq_prices = daily_prices.resample(freq_map[frequency]).last()
    return self._prices_to_returns(freq_prices, method=compound_method)
```

Add multi-frequency fetch:
```python
def fetch_multi_frequency(
    self,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    frequencies: List[str] = ['daily', 'weekly', 'monthly']
) -> Dict[str, pd.Series]:
    """Fetch same data at multiple frequencies for comparison."""
    results = {}
    
    # Fetch daily data once
    daily_data = self.fetch_total_returns(
        ticker, start_date, end_date, 'daily'
    )
    results['daily'] = daily_data
    
    # Convert to other frequencies
    if 'weekly' in frequencies:
        results['weekly'] = self._aggregate_returns(daily_data, 'W-FRI')
    if 'monthly' in frequencies:
        results['monthly'] = self._aggregate_returns(daily_data, 'M')
    if 'quarterly' in frequencies:
        results['quarterly'] = self._aggregate_returns(daily_data, 'Q')
    
    return results
```

### 2. Create Frequency Converter
**File**: `/src/data/frequency_converter.py` (NEW)

```python
import pandas as pd
import numpy as np
from typing import Optional, Dict, Union

class FrequencyConverter:
    """Utilities for converting between data frequencies."""
    
    @staticmethod
    def aggregate_returns(
        returns: pd.Series,
        target_freq: str,
        method: str = 'compound'
    ) -> pd.Series:
        """
        Aggregate returns to lower frequency.
        
        Args:
            returns: Daily or higher frequency returns
            target_freq: pandas frequency string ('W', 'M', 'Q', 'Y')
            method: 'compound' or 'simple'
        """
        if method == 'compound':
            # (1 + r1) * (1 + r2) * ... - 1
            return (1 + returns).resample(target_freq).prod() - 1
        else:
            # Simple sum (only valid for log returns)
            return returns.resample(target_freq).sum()
    
    @staticmethod
    def returns_to_prices(
        returns: pd.Series,
        initial_price: float = 100.0
    ) -> pd.Series:
        """Convert returns series to price series."""
        return initial_price * (1 + returns).cumprod()
    
    @staticmethod
    def align_frequencies(
        data: Dict[str, pd.Series],
        method: str = 'outer'
    ) -> pd.DataFrame:
        """
        Align multiple series of different frequencies.
        
        Args:
            data: Dict of frequency -> series
            method: 'outer' (all dates) or 'inner' (common dates)
        """
        # Convert all to daily frequency for alignment
        aligned = {}
        
        for freq, series in data.items():
            if freq == 'daily':
                aligned[freq] = series
            else:
                # Upscale to daily using forward fill
                daily_index = pd.date_range(
                    series.index[0], 
                    series.index[-1], 
                    freq='D'
                )
                aligned[freq] = series.reindex(daily_index).fillna(method='ffill')
        
        # Combine based on method
        df = pd.DataFrame(aligned)
        return df.dropna(how='any' if method == 'inner' else None)
    
    @staticmethod
    def calculate_frequency_ratio(
        high_freq_returns: pd.Series,
        low_freq_returns: pd.Series
    ) -> float:
        """Calculate information retention ratio between frequencies."""
        # Aggregate high frequency to match low frequency
        high_freq_period = high_freq_returns.index[-1] - high_freq_returns.index[0]
        low_freq_period = low_freq_returns.index[-1] - low_freq_returns.index[0]
        
        if high_freq_period != low_freq_period:
            raise ValueError("Series must cover same time period")
        
        # Calculate volatilities
        high_vol = high_freq_returns.std() * np.sqrt(252)
        periods_per_year = {
            'W': 52,
            'M': 12,
            'Q': 4
        }
        
        # Infer low frequency
        avg_days = (low_freq_returns.index[1:] - low_freq_returns.index[:-1]).mean().days
        if avg_days <= 10:
            scale = 252  # Daily
        elif avg_days <= 40:
            scale = 52   # Weekly
        elif avg_days <= 100:
            scale = 12   # Monthly
        else:
            scale = 4    # Quarterly
            
        low_vol = low_freq_returns.std() * np.sqrt(scale)
        
        return low_vol / high_vol  # Information retention ratio
```

### 3. Update Analytics for Multi-Frequency
**File**: `/src/portfolio/analytics.py`

Add frequency-aware analytics:
```python
def calculate_summary(
    self,
    frequency: str = 'daily',
    convert_to_annual: bool = True
) -> Dict:
    """
    Calculate portfolio summary statistics at specified frequency.
    
    Args:
        frequency: Analysis frequency
        convert_to_annual: Annualize statistics
    """
    # Get returns at requested frequency
    returns = self._get_returns_at_frequency(frequency)
    
    # Scaling factors for annualization
    scale_factors = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4
    }
    scale = scale_factors[frequency] if convert_to_annual else 1
    
    # Calculate statistics with proper scaling
    stats = {
        'frequency': frequency,
        'total_return': (1 + returns).prod() - 1,
        'annual_return': (1 + returns).prod() ** (scale / len(returns)) - 1,
        'volatility': returns.std() * np.sqrt(scale),
        'sharpe_ratio': self._calculate_sharpe(returns, scale),
        'max_drawdown': self._calculate_max_drawdown(returns),
        'observations': len(returns)
    }
    
    return stats
```

### 4. Create Frequency Analysis Tools
**File**: `/src/analysis/frequency_analyzer.py` (NEW)

```python
class FrequencyAnalyzer:
    """Analyze optimal data frequency for different strategies."""
    
    def __init__(self, fetcher: TotalReturnFetcher):
        self.fetcher = fetcher
        
    def compare_frequencies(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        frequencies: List[str] = ['daily', 'weekly', 'monthly']
    ) -> pd.DataFrame:
        """Compare key statistics across frequencies."""
        results = []
        
        for ticker in tickers:
            multi_freq = self.fetcher.fetch_multi_frequency(
                ticker, start_date, end_date, frequencies
            )
            
            for freq, returns in multi_freq.items():
                stats = self._calculate_frequency_stats(returns, freq)
                stats['ticker'] = ticker
                stats['frequency'] = freq
                results.append(stats)
        
        return pd.DataFrame(results)
    
    def find_optimal_frequency(
        self,
        ticker: str,
        metric: str = 'information_ratio',
        rebalance_frequency: Optional[str] = None
    ) -> str:
        """Find optimal data frequency for given metric."""
        # Test different frequencies
        frequencies = ['daily', 'weekly', 'monthly']
        
        if rebalance_frequency:
            # Match data frequency to rebalancing
            return rebalance_frequency
        
        # Otherwise optimize for signal/noise
        best_freq = None
        best_score = -np.inf
        
        for freq in frequencies:
            score = self._calculate_frequency_score(ticker, freq, metric)
            if score > best_score:
                best_score = score
                best_freq = freq
        
        return best_freq
```

### 5. Create Example Scripts
**File**: `/scripts/experiments/frequency_comparison.py` (NEW)

```python
#!/usr/bin/env python
"""Compare portfolio optimization results across data frequencies."""

def main():
    # Initialize
    fetcher = TotalReturnFetcher()
    analyzer = FrequencyAnalyzer(fetcher)
    
    # Define universe
    tickers = ['SPY', 'AGG', 'GLD', 'VNQ', 'EFA']
    
    # Compare frequencies
    print("Comparing data frequencies...")
    comparison = analyzer.compare_frequencies(
        tickers,
        datetime(2015, 1, 1),
        datetime(2023, 12, 31)
    )
    
    # Show results
    print("\nVolatility by Frequency:")
    pivot = comparison.pivot(index='ticker', columns='frequency', values='volatility')
    print(pivot)
    
    print("\nInformation Ratio by Frequency:")
    pivot = comparison.pivot(index='ticker', columns='frequency', values='info_ratio')
    print(pivot)
    
    # Find optimal frequencies
    print("\nOptimal Frequencies by Asset:")
    for ticker in tickers:
        opt_freq = analyzer.find_optimal_frequency(ticker)
        print(f"{ticker}: {opt_freq}")
    
    # Run optimization at different frequencies
    print("\nOptimization Results by Frequency:")
    for freq in ['daily', 'weekly', 'monthly']:
        result = run_optimization_at_frequency(tickers, freq)
        print(f"\n{freq.upper()} Frequency:")
        print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
        print(f"  Turnover: {result['turnover']:.1%}")
```

### 6. Add Tests
**File**: `/tests/test_frequency_converter.py` (NEW)

Test cases:
1. Return aggregation preserves total return
2. Price/return conversion round-trip
3. Frequency alignment handles missing data
4. Information ratio calculation
5. Proper annualization factors

Example test:
```python
def test_return_aggregation_preserves_total_return():
    """Aggregating returns should preserve total return."""
    # Create daily returns
    daily_returns = pd.Series(
        np.random.normal(0.0005, 0.01, 252),
        index=pd.date_range('2023-01-01', periods=252, freq='D')
    )
    
    # Calculate total returns
    daily_total = (1 + daily_returns).prod() - 1
    
    # Aggregate to monthly
    converter = FrequencyConverter()
    monthly_returns = converter.aggregate_returns(daily_returns, 'M')
    monthly_total = (1 + monthly_returns).prod() - 1
    
    # Should be very close (within floating point error)
    assert abs(daily_total - monthly_total) < 1e-10
```

## Success Criteria
- [ ] Can fetch data at daily/weekly/monthly/quarterly frequencies
- [ ] Proper return compounding when aggregating
- [ ] Statistics correctly scaled for each frequency
- [ ] Frequency comparison tools working
- [ ] Tests verify accuracy of conversions
- [ ] Example demonstrates frequency trade-offs

## Design Decisions
1. Always fetch daily data and downsample (ensures consistency)
2. Use geometric (compound) returns by default
3. Week ends on Friday (market convention)
4. Month/quarter ends follow market calendar
5. Handle missing data with forward fill

## Next Steps
After implementation:
1. Run frequency comparison on all exposure types
2. Document optimal frequencies for each
3. Update optimization to use recommended frequencies
4. Add frequency selection to web interface

## Progress Updates
- [ ] Started: [timestamp]
- [ ] Data fetcher enhanced: [status]
- [ ] Frequency converter created: [status]
- [ ] Analytics updated: [status]
- [ ] Analyzer tools built: [status]
- [ ] Tests written: [status]
- [ ] Example working: [status]
