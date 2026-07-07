# Task: Integrate GARCH into Parameter Optimization

**Status**: TODO  
**Priority**: HIGH - Enhances volatility forecasting  
**Estimated Time**: 3-4 hours  
**Dependencies**: GARCH already implemented, parameter optimization framework exists

## Objective
Add GARCH model testing to the parameter optimization framework to compare against EWMA and potentially get better volatility forecasts, especially during volatility clustering periods.

## Quick Implementation Plan

### 1. Update OptimizationConfig
In `/src/optimization/parameter_optimization.py`:

```python
@dataclass
class OptimizationConfig:
    # ... existing fields ...
    
    # GARCH parameters
    test_garch: bool = True
    garch_omega_values: List[float] = field(default_factory=lambda: [0.000001, 0.000005, 0.00001])
    garch_alpha_values: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
    garch_beta_values: List[float] = field(default_factory=lambda: [0.80, 0.85, 0.90])
```

### 2. Add GARCH Validation Method
```python
def _validate_garch_parameters(
    self,
    returns: pd.Series,
    omega: float,
    alpha: float,
    beta: float,
    horizon: int,
    test_start: datetime,
    test_end: datetime
) -> Dict[str, float]:
    """Validate GARCH forecasting accuracy."""
    
    garch = GARCHEstimator(omega=omega, alpha=alpha, beta=beta)
    
    # Similar structure to EWMA validation
    # Return MSE, MAE, QLIKE, hit_rate metrics
```

### 3. Update Parameter Grid Search
Include GARCH in the main optimization loop alongside EWMA testing.

### 4. Comparison Report
Add method to compare GARCH vs EWMA performance by exposure type.

## Success Criteria
- [ ] GARCH parameters included in optimization grid
- [ ] Can compare GARCH vs EWMA for each exposure
- [ ] Results show which model works better for which exposures
- [ ] Integrated into existing parameter selection framework
