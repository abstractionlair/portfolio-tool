# Task: Enable Exposure-Level Optimization in Existing Engine

**Status**: NOT STARTED  
**Priority**: HIGH  
**Estimated Time**: 1-2 days  
**Approach**: ADAPT EXISTING - The optimization engine already exists!

## Current System Inventory

### What Already Exists:

1. **Optimization Engine**:
   - `src/optimization/engine.py` - OptimizationEngine class
   - `src/optimization/methods.py` - Various optimization methods
   - Already supports: Max Sharpe, Min Volatility, Risk Parity, etc.
   - **This is sophisticated and complete - adapt it**

2. **Risk Estimation**:
   - `src/optimization/exposure_risk_estimator.py` - ExposureRiskEstimator
   - Already provides exposure-level covariance matrices
   - Has `get_risk_matrix()` method that returns covariances

3. **Constraints System**:
   - `src/optimization/constraints.py` - OptimizationConstraints class
   - Already handles min/max weights, sectors, etc.

4. **Portfolio Integration**:
   - `src/optimization/portfolio_optimizer.py` - PortfolioOptimizer
   - Currently works with tickers/symbols
   - **This is what needs modification**

## The Gap

The OptimizationEngine is general-purpose and can optimize anything with returns and covariances. The PortfolioOptimizer currently feeds it ticker-level data. We need to:
1. Create an exposure-level wrapper
2. Feed exposure data instead of ticker data

## Required Changes (MINIMAL)

### 1. Create Exposure Optimization Adapter

**New File**: `src/optimization/exposure_optimization_adapter.py`

```python
from .engine import OptimizationEngine, OptimizationConstraints, ObjectiveType
from .exposure_risk_estimator import ExposureRiskEstimator

class ExposureOptimizationAdapter:
    """Adapts existing OptimizationEngine to work with exposures.
    
    This is a thin wrapper that feeds exposure data to the existing engine.
    """
    
    def __init__(self, risk_estimator: ExposureRiskEstimator):
        self.risk_estimator = risk_estimator
        self.engine = OptimizationEngine()  # Use existing!
        
    def optimize_exposures(
        self,
        exposures: List[str],
        estimation_date: datetime,
        objective: ObjectiveType,  # Use existing enum
        constraints: OptimizationConstraints,  # Use existing class
        expected_returns: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """Optimize at exposure level using existing engine.
        
        This method:
        1. Gets exposure-level risk matrix from risk_estimator
        2. Formats data for existing OptimizationEngine
        3. Calls existing optimization methods
        4. Returns results with exposure labels
        """
        
        # Get risk matrix - this already exists!
        risk_matrix = self.risk_estimator.get_risk_matrix(
            exposures, estimation_date
        )
        
        # Convert to format expected by existing engine
        cov_matrix = risk_matrix.covariance_matrix.values
        
        # Use existing engine - no new optimization code!
        if objective == ObjectiveType.MAX_SHARPE:
            result = self.engine.optimize_max_sharpe(
                symbols=exposures,  # Just use exposures as symbols
                expected_returns=expected_returns or self._default_returns(exposures),
                covariance_matrix=cov_matrix,
                constraints=constraints
            )
        # ... handle other objectives
        
        return result
```

### 2. Add Exposure Constraints

**File**: `src/optimization/constraints.py`

**Add to existing OptimizationConstraints class**:
```python
# Existing class already has min/max weights, sectors, etc.
# Just add exposure-specific validations

def validate_for_exposures(self, exposures: List[str]) -> None:
    """Validate constraints make sense for exposures.
    
    E.g., sector constraints might map to exposure categories.
    """
    # Add any exposure-specific validation
    # But reuse all existing constraint logic!
```

### 3. Two-Stage Optimization Method

**File**: `src/optimization/portfolio_optimizer.py`

**Add method to existing PortfolioOptimizer**:
```python
def optimize_two_stage(
    self,
    target_exposures: List[str],
    available_funds: List[str],
    fund_exposure_map: FundExposureMap,  # From previous task
    start_date: date,
    end_date: date,
    exposure_constraints: OptimizationConstraints,
    fund_constraints: OptimizationConstraints
) -> TwoStageResult:
    """Two-stage optimization using existing components.
    
    Stage 1: Use ExposureOptimizationAdapter to find optimal exposures
    Stage 2: Use existing optimize_portfolio to find funds
    """
    
    # Stage 1: Optimize exposures
    exposure_adapter = ExposureOptimizationAdapter(self.risk_estimator)
    exposure_result = exposure_adapter.optimize_exposures(
        target_exposures, end_date, 
        ObjectiveType.MAX_SHARPE, exposure_constraints
    )
    
    # Stage 2: Find funds to match target exposures
    # This is a new optimization problem but uses existing engine
    # Minimize tracking error to target exposures
    # Subject to fund constraints
```

## What NOT to Do

1. **Don't rewrite the optimization engine** - It's already excellent
2. **Don't create new constraint systems** - Extend existing
3. **Don't duplicate risk calculations** - ExposureRiskEstimator exists
4. **Don't create new optimization algorithms** - Use existing methods

## Testing Approach

1. **Reuse existing optimization tests** as templates
2. Test that exposure optimization gives same results as equivalent fund optimization
3. Verify constraints work at exposure level
4. Test two-stage optimization convergence

## Success Criteria

- [ ] ExposureOptimizationAdapter wraps existing engine
- [ ] Can optimize with exposure-level data
- [ ] All existing optimization methods work (Sharpe, Risk Parity, etc.)
- [ ] Two-stage optimization implemented
- [ ] Results match single-stage when funds = exposures
- [ ] Existing tests still pass

## Example Usage After Implementation

```python
# Setup
risk_estimator = ExposureRiskEstimator(exposure_universe, forecast_horizon=21)
adapter = ExposureOptimizationAdapter(risk_estimator)

# Optimize exposures using existing constraint system
constraints = OptimizationConstraints(
    min_weight=0.05,
    max_weight=0.40,
    sector_constraints={'alternatives': (0.1, 0.3)}  # Works for exposure categories
)

result = adapter.optimize_exposures(
    exposures=['us_large_equity', 'broad_ust', 'trend_following'],
    estimation_date=datetime(2025, 7, 11),
    objective=ObjectiveType.MAX_SHARPE,
    constraints=constraints
)

# Result uses existing OptimizationResult structure
print(result.weights)  # {'us_large_equity': 0.4, 'broad_ust': 0.3, ...}
print(result.sharpe_ratio)  # 0.85
```

This approach maximizes reuse of the excellent existing optimization infrastructure.
