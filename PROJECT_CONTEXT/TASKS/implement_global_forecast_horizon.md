# Task: Implement Global Forecast Horizon for Parameter Optimization

**Status**: NOT STARTED  
**Priority**: HIGH  
**Estimated Time**: 2 days  
**Dependencies**: Existing parameter optimization framework

## Problem Statement

The current parameter optimization system optimizes parameters independently for each exposure without ensuring a consistent forecast horizon. This creates an inconsistent optimization problem where different assets might have parameters optimized for different forward-looking periods.

For portfolio optimization to be mathematically consistent, all risk estimates (volatilities and correlations) must be for the same forecast horizon.

## Current Issues

1. **No Global Horizon**: Parameters are optimized without a target forecast horizon
2. **Inconsistent Estimates**: Different exposures could have different implicit horizons
3. **Post-hoc Horizon Application**: The forecast horizon is only applied during risk estimation, not during parameter optimization

## Requirements

### 1. Global Forecast Horizon Configuration
- Add a global forecast horizon setting that applies to all exposures
- Common horizons: 1 day, 5 days (weekly), 21 days (monthly), 63 days (quarterly)
- This should be configurable but consistent across all assets

### 2. Horizon-Aware Parameter Optimization
Modify the parameter optimization to:
- Accept a target forecast horizon as input
- Optimize parameters specifically for predicting risk at that horizon
- Validate predictions against realized volatility/correlation at the target horizon

### 3. Updated Configuration Structure
Create a new configuration format that clearly shows:
- The global forecast horizon
- Parameters optimized for that specific horizon
- Validation metrics for the chosen horizon

## Implementation Plan

### Phase 1: Modify Parameter Optimization Framework

1. **Update `ParameterOptimizer` class**:
```python
class ParameterOptimizer:
    def optimize_for_horizon(
        self,
        exposures: List[str],
        target_horizon: int,  # e.g., 21 days
        start_date: datetime,
        end_date: datetime,
        validation_method: str = "walk_forward"
    ) -> Dict[str, OptimalParams]:
        """
        Optimize parameters for all exposures for a specific forecast horizon.
        
        Args:
            exposures: List of exposure IDs to optimize
            target_horizon: Forecast horizon in days (must be same for all)
            start_date: Start of historical data
            end_date: End of historical data
            validation_method: Method for validation
            
        Returns:
            Dictionary of exposure_id -> optimal parameters for the horizon
        """
```

2. **Update validation to use horizon-specific metrics**:
```python
def validate_volatility_forecast(
    self,
    returns: pd.Series,
    params: Dict,
    horizon: int
) -> Dict[str, float]:
    """Validate volatility forecast for specific horizon."""
    # Predict volatility h-days ahead
    # Compare with realized volatility over next h days
    # Return MSE, MAE, QLIKE for h-day forecasts
```

### Phase 2: Create Horizon-Specific Configuration

1. **New configuration structure** (`config/optimal_parameters_v2.yaml`):
```yaml
# Global settings - applies to entire portfolio
global_settings:
  forecast_horizon: 21  # All parameters optimized for 21-day ahead forecasts
  rebalance_frequency: "monthly"
  optimization_date: "2025-07-11"
  validation_period: ["2020-01-01", "2025-07-11"]

# Horizon-specific parameters for each exposure
horizon_21_parameters:
  volatility:
    us_large_equity:
      method: "ewma"
      lambda: 0.94  # Optimized for 21-day forecasts
      min_periods: 30
      lookback_days: 252
      validation_score: 0.023  # MSE for 21-day forecasts
      
    us_small_equity:
      method: "ewma"
      lambda: 0.92  # Different lambda for small-cap
      min_periods: 30
      lookback_days: 252
      validation_score: 0.031
      
  correlation:
    method: "ewma"
    lambda: 0.96  # Single lambda for all correlations
    min_periods: 60
    lookback_days: 504
    validation_score: 0.15  # Frobenius norm

# Alternative horizons (for future use)
alternative_horizons:
  horizon_5:  # Weekly rebalancing
    note: "Parameters optimized for 5-day forecasts"
  horizon_63:  # Quarterly rebalancing
    note: "Parameters optimized for 63-day forecasts"
```

### Phase 3: Update Risk Estimation Integration

1. **Modify `ExposureRiskEstimator`**:
```python
class ExposureRiskEstimator:
    def __init__(
        self,
        exposure_universe: ExposureUniverse,
        forecast_horizon: int,  # Set at initialization
        parameter_config_path: str = "config/optimal_parameters_v2.yaml"
    ):
        """Initialize with fixed forecast horizon."""
        self.forecast_horizon = forecast_horizon
        self.parameters = self._load_horizon_specific_parameters(forecast_horizon)
        
    def _load_horizon_specific_parameters(self, horizon: int) -> Dict:
        """Load parameters optimized for this specific horizon."""
        # Load from optimal_parameters_v2.yaml
        # Use parameters from horizon_{horizon}_parameters section
```

2. **Remove horizon parameter from estimation methods**:
```python
def estimate_exposure_risks(
    self,
    exposures: List[str],
    estimation_date: datetime,
    lookback_days: int = 756
    # No forecast_horizon parameter - uses self.forecast_horizon
) -> Dict[str, ExposureRiskEstimate]:
    """Estimate risks using globally configured horizon."""
```

## Validation Requirements

1. **Consistency Checks**:
   - Ensure all exposures use same forecast horizon
   - Verify parameters are optimized for the stated horizon
   - Check that risk estimates are all forward-looking to same period

2. **Performance Metrics**:
   - Track prediction accuracy for the chosen horizon
   - Compare with naive forecasts (historical volatility)
   - Ensure improvement over non-optimized parameters

3. **Integration Tests**:
   - Test full pipeline from parameter optimization to risk estimation
   - Verify portfolio optimization uses consistent horizons
   - Check that results are reproducible

## Success Criteria

- [ ] Parameter optimization accepts and uses target forecast horizon
- [ ] All exposures have parameters optimized for same horizon
- [ ] Configuration clearly shows global horizon setting
- [ ] Risk estimator enforces consistent horizon usage
- [ ] Validation proves parameters are optimal for chosen horizon
- [ ] Full integration test passes with consistent 21-day forecasts

## Files to Create/Modify

1. **Modify**:
   - `src/optimization/parameter_optimization.py` - Add horizon-aware optimization
   - `src/optimization/exposure_risk_estimator.py` - Enforce consistent horizon
   - `src/optimization/ewma.py` - Add horizon-specific validation

2. **Create**:
   - `config/optimal_parameters_v2.yaml` - New configuration format
   - `src/optimization/horizon_validator.py` - Consistency checks
   - `tests/test_horizon_consistency.py` - Integration tests

3. **Update**:
   - `examples/parameter_optimization_demo.py` - Show horizon selection
   - `notebooks/parameter_optimization_analysis.ipynb` - Analyze horizon impact

## Notes for Implementation

- Start with 21-day horizon as the default (monthly rebalancing is common)
- Keep the old parameter files for comparison
- Make the change backward compatible if possible
- Consider computational efficiency - don't re-optimize if horizon hasn't changed
- Document why consistent horizons are critical for portfolio optimization

This ensures mathematical consistency in portfolio optimization by guaranteeing all risk estimates look forward to the same time period.
