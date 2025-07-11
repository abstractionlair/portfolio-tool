# Component Optimization Implementation Guidance

**For**: Claude Code  
**Re**: Questions in current_task.md

## Answers to Implementation Questions

### 1. Should expected return optimization include factor models or just time series approaches?

**Answer**: Start with time series approaches, but design for extensibility.

**Phase 1** (Implement first):
- Historical mean returns
- EWMA returns
- Momentum (3, 6, 12 month)
- Mean reversion signals

**Phase 2** (Framework for future):
- Create `FactorReturnModel` base class
- Placeholder for value, quality, low vol factors
- Interface for external factor data

**Rationale**: Time series models can be implemented with existing data. Factor models require additional data sources we haven't integrated yet.

### 2. What validation periods should we use for optimization?

**Answer**: Use expanding window with these specifications:

- **Minimum training period**: 2 years (504 trading days)
- **Validation splits**: 5 periods of 6 months each
- **Total data requirement**: 5 years minimum
- **Walk-forward step**: 6 months

Example timeline:
- Train: 2019-2020, Validate: 2021 H1
- Train: 2019-2021 H1, Validate: 2021 H2
- Train: 2019-2021, Validate: 2022 H1
- etc.

**Special handling**: For correlation optimization, allow longer training periods (up to 5 years) if it improves stability.

### 3. Should we include regime detection in the optimization?

**Answer**: Not in Phase 1, but design for it.

**Phase 1**: 
- Single regime optimization
- Flag in results when parameters change significantly over time

**Future enhancement**:
- Add `regime_aware` parameter to optimizers
- Store regime-specific parameters in results
- Let production code choose based on current regime

**Rationale**: Regime detection adds complexity. Get single-regime working first.

### 4. Do you want automated re-optimization on a schedule?

**Answer**: Yes, but make it configurable.

Implement in `OptimizedRiskEstimator`:
```python
def __init__(self, 
             parameter_file="config/optimal_parameters.yaml",
             auto_reoptimize_days=90,  # Re-optimize quarterly
             min_data_change_percent=10):  # Only if >10% new data
```

Features:
- Check parameter age on initialization
- Log warning if parameters are stale (>180 days)
- Optional auto-reoptimization if requested
- Store optimization history for comparison

## Additional Implementation Guidance

### Priority Order
1. Volatility optimizer (most important)
2. Production interface (so it can be used immediately)
3. Correlation optimizer
4. Expected return optimizer
5. Full orchestrator

### Performance Considerations
- Cache all expensive calculations
- Use joblib.Memory for function-level caching
- Parallelize across exposures, not just methods
- Store intermediate results for debugging

### Testing Strategy
- Create synthetic data with known optimal parameters
- Verify optimizers find these parameters
- Test production interface with both real and synthetic data
- Include edge cases (short data, missing data, single exposure)

### Configuration Format
Optimal parameters YAML should be human-readable:
```yaml
version: "1.0"
optimization_date: "2025-07-10"
validation_period: 
  start: "2020-01-01"
  end: "2024-12-31"

volatility_parameters:
  us_large_equity:
    method: "ewma"
    lookback_days: 252
    frequency: "daily"
    parameters:
      lambda: 0.94
      min_periods: 63
    score: -0.0234  # negative MSE
    
correlation_parameters:
  method: "ewma"
  lookback_days: 756  # 3 years
  frequency: "daily"
  parameters:
    lambda: 0.97
    min_periods: 252
  score: 0.892  # stability score
```

### Error Handling
- If optimization fails for an exposure, use sensible defaults
- Log all failures with full context
- Never let one exposure failure stop the entire process
- Provide clear error messages in the production interface

## Success Metrics
Track these in the implementation:
1. Time to run full optimization (<30 minutes acceptable)
2. Memory usage (<4GB for full universe)
3. Parameter stability (shouldn't change drastically between runs)
4. Actual forecast improvements in backtesting
