# Advanced Analytics Enhancement Plan

**Created**: 2025-01-06 by Desktop Claude  
**Purpose**: Add sophisticated time-series analysis and real return optimization

## Overview

This plan adds four major enhancements to the portfolio optimizer:
1. Exponentially Weighted Moving Average (EWMA) support throughout
2. Multi-frequency data handling (daily, weekly, monthly, quarterly)
3. Parameter optimization framework for stability analysis
4. Real return tangent portfolio optimization

## Enhancement 1: EWMA Support

### Motivation
- Recent data should have more weight in estimates
- Better handling of regime changes
- Industry standard for risk modeling (RiskMetrics)

### Implementation Areas

#### 1.1 Return Estimation (`/src/data/return_estimation.py`)
```python
class ReturnEstimator:
    def estimate_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'historical',
        window: Optional[int] = None,
        ewma_halflife: Optional[int] = None,  # NEW
        ewma_com: Optional[float] = None,      # NEW
        ewma_alpha: Optional[float] = None     # NEW
    ) -> pd.Series:
        """
        Enhanced with EWMA options.
        
        Args:
            ewma_halflife: Half-life in periods (e.g., 60 days)
            ewma_com: Center of mass (RiskMetrics uses 94 for daily)
            ewma_alpha: Direct decay factor (0 < α ≤ 1)
        """
```

#### 1.2 Covariance Estimation (`/src/optimization/estimators.py`)
```python
def estimate_covariance(
    returns: pd.DataFrame,
    method: str = 'sample',
    ewma_params: Optional[Dict] = None  # NEW
) -> pd.DataFrame:
    """
    Add EWMA covariance estimation.
    
    Common EWMA parameters:
    - RiskMetrics daily: λ = 0.94 (com ≈ 94)
    - RiskMetrics monthly: λ = 0.97
    """
```

#### 1.3 Portfolio Analytics (`/src/portfolio/analytics.py`)
Add EWMA options to rolling metrics:
- Rolling volatility
- Rolling Sharpe ratio
- Rolling correlations

### Design Decisions
- Support all three pandas EWMA parameterizations (halflife, com, alpha)
- Default to RiskMetrics parameters where applicable
- Allow method chaining: `.with_ewma(halflife=60)`

## Enhancement 2: Multi-Frequency Data Support

### Motivation
- Different horizons need different frequencies
- Reduce noise while maintaining responsiveness
- Some strategies work better at different frequencies

### Implementation Plan

#### 2.1 Enhanced Data Fetcher (`/src/data/total_returns.py`)
```python
class TotalReturnFetcher:
    def fetch_total_returns(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily"  # ENHANCED: daily, weekly, monthly, quarterly
    ) -> pd.Series:
        """Fetch returns at specified frequency."""
        
    def fetch_multi_frequency(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        frequencies: List[str]
    ) -> Dict[str, pd.Series]:
        """Fetch same data at multiple frequencies for analysis."""
```

#### 2.2 Frequency Conversion Utilities
```python
class FrequencyConverter:
    @staticmethod
    def to_monthly(daily_returns: pd.Series, method: str = 'compound') -> pd.Series:
        """Convert daily to monthly returns."""
        
    @staticmethod
    def to_quarterly(returns: pd.Series, source_freq: str) -> pd.Series:
        """Convert any frequency to quarterly."""
        
    @staticmethod
    def align_frequencies(data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Align multiple series of different frequencies."""
```

### Key Considerations
- Handle compounding correctly when aggregating returns
- Preserve total return accuracy across frequencies
- Support both price and return series

## Enhancement 3: Parameter Optimization Framework

### Motivation
- Find optimal lookback windows empirically
- Determine best EWMA decay factors
- Identify most stable frequency for each asset class

### Implementation Structure

#### 3.1 Stability Analyzer (`/src/analysis/stability_analyzer.py`)
```python
class StabilityAnalyzer:
    def analyze_window_stability(
        self,
        returns: pd.DataFrame,
        window_sizes: List[int],
        metrics: List[str] = ['volatility', 'correlation', 'sharpe']
    ) -> pd.DataFrame:
        """Test stability of estimates across different windows."""
        
    def optimize_ewma_decay(
        self,
        returns: pd.DataFrame,
        objective: str = 'forecast_error',
        search_space: Dict = {'halflife': (20, 120)}
    ) -> Dict:
        """Find optimal EWMA parameters."""
        
    def frequency_stability_test(
        self,
        ticker: str,
        frequencies: List[str],
        metric: str = 'information_ratio'
    ) -> Dict:
        """Compare estimate stability across frequencies."""
```

#### 3.2 Experiment Scripts (`/scripts/experiments/`)
1. `optimize_estimation_windows.py`
   - Test windows from 20 to 252 days
   - Measure out-of-sample stability
   - Generate recommendations per asset class

2. `optimize_ewma_parameters.py`
   - Grid search over decay factors
   - Cross-validation approach
   - Compare to simple moving averages

3. `frequency_comparison.py`
   - Compare daily vs weekly vs monthly
   - Identify noise vs signal trade-offs
   - Recommend frequencies by exposure type

### Metrics for Optimization
- Out-of-sample forecast accuracy
- Estimate stability (rolling standard deviation)
- Information ratio of resulting portfolios
- Turnover implications

## Enhancement 4: Real Return Tangent Portfolio

### Motivation
- Maximize real returns, not nominal
- Properly account for inflation in optimization
- Find truly optimal risk/return trade-off

### Mathematical Framework
```
Real Return = (1 + Nominal Return) / (1 + Inflation) - 1
Risk Premium = Nominal Return - Risk-Free Rate
Real Risk Premium = Real Return - Real Risk-Free Rate

Tangent Portfolio: max(E[R_real] - R_f_real) / σ
```

### Implementation Plan

#### 4.1 Real Return Calculator (`/src/analysis/real_returns.py`)
```python
class RealReturnCalculator:
    def __init__(self, inflation_series: pd.Series):
        self.inflation = inflation_series
    
    def to_real_returns(
        self,
        nominal_returns: pd.Series,
        method: str = 'exact'  # or 'approximate'
    ) -> pd.Series:
        """Convert nominal to real returns."""
        
    def real_covariance_matrix(
        self,
        nominal_returns: pd.DataFrame,
        inflation_correlation: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Calculate covariance matrix in real terms."""
```

#### 4.2 Enhanced Optimization (`/src/optimization/real_return_optimizer.py`)
```python
class RealReturnOptimizer:
    def find_tangent_portfolio(
        self,
        expected_real_returns: pd.Series,
        real_covariance: pd.DataFrame,
        real_risk_free_rate: float,
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """Find tangent portfolio in real return space."""
        
    def decompose_returns(
        self,
        nominal_returns: pd.Series,
        inflation: pd.Series,
        risk_free: pd.Series
    ) -> Dict:
        """Decompose into inflation + real rf + risk premium."""
```

### Special Considerations
- Handle correlation between assets and inflation
- Account for inflation-hedging properties
- Consider geometric (compound) returns for long horizons

## Implementation Sequence

### Phase 1: Foundation (Week 1)
1. Add EWMA support to existing estimators
2. Implement frequency conversion utilities
3. Create real return calculator

### Phase 2: Analysis Tools (Week 2)
1. Build stability analyzer
2. Create parameter optimization framework
3. Write experiment scripts

### Phase 3: Integration (Week 3)
1. Add real return optimization
2. Update portfolio analytics with new options
3. Create demonstration notebooks

### Phase 4: Validation (Week 4)
1. Backtest parameter recommendations
2. Compare real vs nominal optimization results
3. Document findings and best practices

## File Structure
```
src/
├── analysis/           # NEW
│   ├── stability_analyzer.py
│   ├── real_returns.py
│   └── parameter_optimizer.py
├── data/
│   └── frequency_converter.py  # NEW
└── optimization/
    └── real_return_optimizer.py  # NEW

scripts/
└── experiments/       # NEW
    ├── optimize_estimation_windows.py
    ├── optimize_ewma_parameters.py
    ├── frequency_comparison.py
    └── real_return_tangent.py

notebooks/
└── experiments/       # NEW
    ├── parameter_optimization_results.ipynb
    └── real_vs_nominal_comparison.ipynb
```

## Success Criteria
1. **EWMA**: All estimators support exponential weighting
2. **Frequencies**: Can analyze at daily/weekly/monthly with proper compounding
3. **Optimization**: Find empirically optimal parameters for major asset classes
4. **Real Returns**: Tangent portfolio that maximizes real Sharpe ratio

## Research Questions to Answer
1. What EWMA halflife minimizes out-of-sample forecast error?
2. Which frequency provides best signal-to-noise for each exposure?
3. How different is real vs nominal tangent portfolio?
4. What window size gives most stable correlation estimates?

## Next Steps
1. Create detailed task specifications for each enhancement
2. Prioritize based on impact and dependencies
3. Begin with EWMA support as it enhances existing functionality
