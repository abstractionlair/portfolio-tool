# Task: Exposure-Level Risk Estimation Framework

**Status**: TODO  
**Priority**: CRITICAL - Core requirement for portfolio optimization  
**Estimated Time**: 6-8 hours  
**Dependencies**: Parameter optimization framework (complete)

## Objective
Create a production-ready framework for estimating future volatilities and correlations at the **exposure level** (not individual securities), using the optimized parameters from the parameter optimization framework.

## Motivation
- Portfolio optimization requires forward-looking risk estimates for exposures
- We've validated parameters on historical data, now need to apply them
- Exposure-level estimation is more stable than individual security estimation
- This is the bridge between parameter optimization and portfolio construction

## Key Requirements

1. **Use Validated Optimal Parameters**
   - Leverage the parameter optimization results
   - Different parameters for different forecast horizons
   - Frequency-specific estimation

2. **Exposure-Level Focus**
   - Estimate volatility for each exposure (e.g., US Equity Large Cap)
   - Full correlation matrix between all exposures
   - Handle missing data gracefully

3. **Multiple Estimation Methods**
   - EWMA (already optimized)
   - GARCH (integrate into framework)
   - Simple historical
   - Shrinkage estimators

## Implementation Plan

### 1. Create Exposure Risk Estimator
**File**: `/src/optimization/exposure_risk_estimator.py` (NEW)

```python
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from ..data.exposure_universe import ExposureUniverse
from .parameter_optimization import ParameterOptimizer
from .ewma import EWMAEstimator, GARCHEstimator


@dataclass
class ExposureRiskEstimate:
    """Container for exposure-level risk estimates."""
    exposure_id: str
    volatility: float  # Annualized
    forecast_horizon: int
    estimation_date: datetime
    method: str
    confidence_interval: Optional[Tuple[float, float]] = None


class ExposureRiskEstimator:
    """Estimate forward-looking volatilities and correlations for exposures."""
    
    def __init__(
        self,
        exposure_universe: ExposureUniverse,
        parameter_optimizer: Optional[ParameterOptimizer] = None
    ):
        self.exposure_universe = exposure_universe
        self.parameter_optimizer = parameter_optimizer
        self._exposure_returns_cache = {}
        
    def estimate_exposure_risks(
        self,
        exposures: List[str],
        estimation_date: datetime,
        lookback_days: int = 756,  # 3 years default
        forecast_horizon: int = 21,  # 1 month default
        method: str = 'optimal'  # 'optimal', 'ewma', 'garch', 'historical'
    ) -> Dict[str, ExposureRiskEstimate]:
        """
        Estimate volatilities for multiple exposures.
        
        Args:
            exposures: List of exposure IDs
            estimation_date: Date for estimation
            lookback_days: Historical data to use
            forecast_horizon: Days ahead to forecast
            method: Estimation method
            
        Returns:
            Dictionary of exposure_id -> risk estimate
        """
        
    def estimate_exposure_correlation_matrix(
        self,
        exposures: List[str],
        estimation_date: datetime,
        lookback_days: int = 756,
        method: str = 'optimal'
    ) -> pd.DataFrame:
        """
        Estimate correlation matrix between exposures.
        
        Returns:
            DataFrame with exposure correlations
        """
        
    def get_risk_matrix(
        self,
        exposures: List[str],
        estimation_date: datetime,
        forecast_horizon: int = 21
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Get both volatilities and correlation matrix.
        
        Returns:
            Tuple of (volatilities Series, correlation DataFrame)
        """
```

### 2. Integrate GARCH into Parameter Optimization
**File**: `/src/optimization/parameter_optimization.py` (UPDATE)

Add GARCH parameter testing:
```python
# In OptimizationConfig, add:
test_garch: bool = True
garch_omega_values: List[float] = field(default_factory=lambda: [0.000001, 0.000005])
garch_alpha_values: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
garch_beta_values: List[float] = field(default_factory=lambda: [0.80, 0.85, 0.90])

# In parameter grid search, add GARCH testing
```

### 3. Create Comprehensive Risk Report
**File**: `/src/reports/exposure_risk_report.py` (NEW)

```python
class ExposureRiskReport:
    """Generate comprehensive risk reports for exposures."""
    
    def generate_risk_summary(
        self,
        exposures: List[str],
        estimation_date: datetime,
        horizons: List[int] = [1, 5, 21, 63]
    ) -> pd.DataFrame:
        """Generate risk summary across multiple horizons."""
        
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame):
        """Visualize exposure correlations."""
        
    def plot_volatility_term_structure(self, exposures: List[str]):
        """Show how volatility estimates change with horizon."""
```

### 4. Example Script
**File**: `/examples/exposure_risk_estimation_demo.py` (NEW)

```python
#!/usr/bin/env python
"""
Exposure-Level Risk Estimation Demo

Shows how to:
1. Use optimized parameters from parameter optimization
2. Estimate forward-looking volatilities for exposures
3. Build correlation matrices between exposures
4. Compare different estimation methods
"""

def main():
    # Load exposure universe
    universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
    
    # Load optimal parameters (from previous optimization)
    param_optimizer = ParameterOptimizer(universe)
    # ... load previous results ...
    
    # Create risk estimator
    risk_estimator = ExposureRiskEstimator(universe, param_optimizer)
    
    # Select exposures for portfolio
    portfolio_exposures = [
        'us_equity_large_cap',
        'developed_intl_equity',
        'us_treasury_long',
        'commodities_broad',
        'real_estate_us'
    ]
    
    # Estimate risks using optimal parameters
    volatilities, correlations = risk_estimator.get_risk_matrix(
        portfolio_exposures,
        estimation_date=datetime.now(),
        forecast_horizon=21  # 1 month
    )
    
    print("\nEstimated Volatilities (Annualized):")
    print(volatilities)
    
    print("\nEstimated Correlation Matrix:")
    print(correlations)
    
    # Compare methods
    for method in ['optimal', 'ewma', 'garch', 'historical']:
        vols = risk_estimator.estimate_exposure_risks(
            portfolio_exposures, 
            datetime.now(),
            method=method
        )
        print(f"\n{method.upper()} Method Volatilities:")
        for exp_id, estimate in vols.items():
            print(f"  {exp_id}: {estimate.volatility:.1%}")
```

### 5. Integration with Portfolio Optimization
**File**: `/src/optimization/engine.py` (UPDATE)

```python
def optimize_with_exposure_estimates(
    self,
    exposure_weights: Dict[str, float],  # Current exposure weights
    risk_estimator: ExposureRiskEstimator,
    estimation_date: datetime,
    forecast_horizon: int = 21,
    **kwargs
) -> OptimizationResult:
    """Optimize using forward-looking exposure risk estimates."""
    
    # Get risk estimates
    exposures = list(exposure_weights.keys())
    vols, corr_matrix = risk_estimator.get_risk_matrix(
        exposures, estimation_date, forecast_horizon
    )
    
    # Convert to covariance matrix
    cov_matrix = corr_matrix * np.outer(vols, vols)
    
    # Run optimization...
```

## Success Criteria
- [ ] Can estimate volatilities for all exposures using optimal parameters
- [ ] Full correlation matrix between exposures
- [ ] GARCH integrated into parameter selection
- [ ] Forward-looking estimates for multiple horizons
- [ ] Comprehensive risk report generation
- [ ] Integration with portfolio optimization
- [ ] Validation against realized values

## Testing Requirements
1. Test parameter pass-through from optimization
2. Verify estimation accuracy on out-of-sample data
3. Test handling of missing exposure data
4. Validate correlation matrix properties (PSD, bounded)
5. Compare methods on same data

## Next Steps After This Task
1. Expected return estimation (much simpler, lower priority)
2. Risk model attribution and decomposition
3. Scenario analysis on risk estimates
4. Time-varying correlation modeling

## Notes
- This bridges parameter optimization to actual portfolio construction
- Focus on robustness - this is production-critical
- Consider confidence intervals for estimates
- Document parameter choices clearly
