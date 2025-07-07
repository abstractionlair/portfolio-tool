# Task: Implement EWMA Support

**Status**: TODO  
**Priority**: HIGH - Enhances all existing analytics  
**Estimated Time**: 4-6 hours  
**Dependencies**: None (enhances existing code)

## Objective
Add exponentially weighted moving average support throughout the system for better estimation of returns, volatilities, and correlations.

## Background
EWMA gives more weight to recent observations, which is crucial for:
- Capturing regime changes
- Reducing lag in estimates
- Industry-standard risk modeling (RiskMetrics)

## Implementation Tasks

### 1. Enhance Return Estimation
**File**: `/src/data/return_estimation.py`

Add EWMA parameter support:
```python
def estimate_returns(
    self,
    prices: pd.DataFrame,
    method: str = 'historical',
    window: Optional[int] = None,
    # NEW PARAMETERS
    ewma_halflife: Optional[int] = None,
    ewma_com: Optional[float] = None,
    ewma_alpha: Optional[float] = None
) -> pd.Series:
    """
    Estimate expected returns with optional EWMA.
    
    Args:
        ewma_halflife: Periods for 50% decay (e.g., 60 days)
        ewma_com: Center of mass (RiskMetrics: 94 for daily)
        ewma_alpha: Direct smoothing factor (0 < α ≤ 1)
        
    Only one EWMA parameter should be specified.
    """
    if method == 'historical':
        if any([ewma_halflife, ewma_com, ewma_alpha]):
            # Use EWMA for rolling mean
            returns = self._calculate_returns(prices)
            ewma_params = self._get_ewma_params(
                halflife=ewma_halflife,
                com=ewma_com,
                alpha=ewma_alpha
            )
            return returns.ewm(**ewma_params).mean().iloc[-1]
```

### 2. Add EWMA Covariance Estimation
**File**: `/src/optimization/estimators.py`

Create new estimation method:
```python
def estimate_covariance_ewma(
    returns: pd.DataFrame,
    halflife: Optional[int] = None,
    com: Optional[float] = 94,  # RiskMetrics default
    alpha: Optional[float] = None,
    min_periods: int = 30
) -> pd.DataFrame:
    """
    Calculate EWMA covariance matrix.
    
    Industry standard parameters:
    - Daily data: com=94 (λ=0.94)
    - Monthly data: com=32.33 (λ=0.97)
    """
    # Implementation approach:
    # 1. Calculate EWMA variances for each asset
    # 2. Calculate EWMA covariances for each pair
    # 3. Ensure positive semi-definite result
```

Update main covariance function:
```python
def estimate_covariance(
    returns: pd.DataFrame,
    method: str = 'sample',  # 'sample', 'shrinkage', 'ewma'
    **kwargs
) -> pd.DataFrame:
    if method == 'ewma':
        return estimate_covariance_ewma(returns, **kwargs)
```

### 3. Enhance Portfolio Analytics
**File**: `/src/portfolio/analytics.py`

Add EWMA options to rolling calculations:
```python
def calculate_rolling_volatility(
    self,
    window: int = 252,
    ewma_params: Optional[Dict] = None
) -> pd.Series:
    """Calculate rolling or EWMA volatility."""
    if ewma_params:
        return self.returns.ewm(**ewma_params).std() * np.sqrt(252)
    else:
        return self.returns.rolling(window).std() * np.sqrt(252)

def calculate_rolling_sharpe(
    self,
    risk_free_rate: float = 0.0,
    window: int = 252,
    ewma_params: Optional[Dict] = None
) -> pd.Series:
    """Calculate rolling or EWMA Sharpe ratio."""
    if ewma_params:
        roll_mean = self.returns.ewm(**ewma_params).mean()
        roll_std = self.returns.ewm(**ewma_params).std()
    else:
        roll_mean = self.returns.rolling(window).mean()
        roll_std = self.returns.rolling(window).std()
    
    excess_returns = roll_mean - risk_free_rate/252
    return excess_returns / roll_std * np.sqrt(252)
```

### 4. Create EWMA Utilities
**File**: `/src/utils/ewma_utils.py` (NEW)

```python
from typing import Optional, Dict
import pandas as pd

class EWMAConfig:
    """Configuration for EWMA calculations."""
    
    # Industry standard parameters
    RISKMETRICS_DAILY = {'com': 94}      # λ = 0.94
    RISKMETRICS_MONTHLY = {'com': 32.33}  # λ = 0.97
    
    @staticmethod
    def get_params(
        halflife: Optional[int] = None,
        com: Optional[float] = None,
        alpha: Optional[float] = None,
        span: Optional[float] = None
    ) -> Dict:
        """Convert between different EWMA parameterizations."""
        # Validate only one parameter specified
        params = {'halflife': halflife, 'com': com, 'alpha': alpha, 'span': span}
        specified = [k for k, v in params.items() if v is not None]
        
        if len(specified) != 1:
            raise ValueError(f"Specify exactly one EWMA parameter, got: {specified}")
        
        return {specified[0]: params[specified[0]]}
    
    @staticmethod
    def lambda_to_com(lambda_param: float) -> float:
        """Convert RiskMetrics λ to pandas center of mass."""
        return 1 / (1 - lambda_param) - 1
    
    @staticmethod
    def suggest_parameters(frequency: str, lookback_days: int) -> Dict:
        """Suggest EWMA parameters based on data frequency."""
        suggestions = {
            'daily': {'com': 94},      # 3-4 month effective window
            'weekly': {'com': 19},     # ~3-4 month effective window  
            'monthly': {'com': 32.33}  # ~3 year effective window
        }
        return suggestions.get(frequency, {'halflife': lookback_days})
```

### 5. Add Tests
**File**: `/tests/test_ewma_support.py` (NEW)

Test cases:
1. EWMA parameter validation (only one specified)
2. EWMA vs simple moving average comparison
3. Covariance matrix positive semi-definite check
4. Parameter conversion accuracy
5. Industry standard parameter validation

Example test:
```python
def test_ewma_reduces_lag():
    """EWMA should respond faster to changes than SMA."""
    # Create returns with regime change
    returns1 = np.random.normal(0.0001, 0.01, 100)
    returns2 = np.random.normal(0.0003, 0.02, 100)
    returns = pd.Series(np.concatenate([returns1, returns2]))
    
    # Calculate both estimates
    sma_vol = returns.rolling(60).std()
    ewma_vol = returns.ewm(halflife=30).std()
    
    # EWMA should detect the volatility increase faster
    sma_detection = np.where(sma_vol > 0.015)[0][0]
    ewma_detection = np.where(ewma_vol > 0.015)[0][0]
    
    assert ewma_detection < sma_detection
```

### 6. Create Example Script
**File**: `/examples/ewma_comparison.py` (NEW)

Demonstrate:
1. Fetching market data
2. Calculating returns/volatility with both methods
3. Visualizing the difference
4. Showing regime change detection
5. Comparing covariance stability

## Integration Points

1. **OptimizationEngine**: Accept EWMA parameters
2. **BacktestEngine**: Use EWMA for rolling optimization
3. **RiskMetrics**: Default to industry standards
4. **Reporting**: Show both EWMA and SMA estimates

## Success Criteria
- [ ] All return/risk estimators accept EWMA parameters
- [ ] EWMA covariance matrices are positive semi-definite
- [ ] Tests pass showing EWMA reduces lag
- [ ] Example demonstrates clear benefits
- [ ] Documentation includes parameter recommendations
- [ ] Backwards compatible (EWMA optional)

## Performance Considerations
- EWMA calculations are O(n) vs O(n²) for rolling windows
- Cache EWMA statistics when possible
- Use pandas built-in `.ewm()` for efficiency

## Questions for Desktop Claude
Add questions here during implementation.

## Progress Updates
- [ ] Started: [timestamp]
- [ ] Return estimation enhanced: [status]
- [ ] Covariance EWMA implemented: [status]
- [ ] Analytics updated: [status]
- [ ] Tests written: [status]
- [ ] Example created: [status]
- [ ] Documentation updated: [status]
