# Task: Build Exposure-Level Optimization Engine

**Status**: NOT STARTED  
**Priority**: HIGH  
**Estimated Time**: 2 days  
**Dependencies**: Global forecast horizon, Fund exposure mapping

## Overview

Create an optimization engine that works directly with exposures rather than individual securities. This is the core of the exposure-based portfolio optimization system, enabling optimization based on true economic exposures rather than fund labels.

## Motivation

Traditional portfolio optimization works with securities (funds/stocks/bonds). But what we really want to optimize are the underlying economic exposures. For example:
- RSST = 100% US Equity + 100% Trend Following (200% notional)
- Optimizing RSST weight doesn't directly optimize equity vs trend exposure

The exposure-level optimizer solves this by:
1. Optimizing exposure weights directly
2. Using exposure-level risk estimates
3. Handling leverage naturally at the exposure level

## Requirements

### 1. Exposure Optimization Engine

```python
# src/optimization/exposure_optimizer.py
class ExposureOptimizer:
    """Optimization engine that works directly with exposures."""
    
    def __init__(
        self,
        risk_estimator: ExposureRiskEstimator,
        return_estimator: Optional[ExposureReturnEstimator] = None
    ):
        """
        Initialize with risk and return estimators.
        
        Args:
            risk_estimator: Provides covariance matrix estimates
            return_estimator: Provides expected return estimates (optional)
        """
        self.risk_estimator = risk_estimator
        self.return_estimator = return_estimator
        
    def optimize(
        self,
        exposures: List[str],
        estimation_date: datetime,
        objective: ExposureObjective,
        constraints: ExposureConstraints,
        options: OptimizationOptions = None
    ) -> ExposureOptimizationResult:
        """
        Main optimization method.
        
        Args:
            exposures: List of exposure IDs to optimize
            estimation_date: Date for risk/return estimation
            objective: Optimization objective (MaxSharpe, MinVol, etc.)
            constraints: Constraints on exposures
            options: Additional optimization options
            
        Returns:
            Optimization result with exposure weights and metrics
        """
```

### 2. Exposure-Specific Objectives

```python
# src/optimization/exposure_objectives.py
class ExposureObjective(ABC):
    """Base class for exposure optimization objectives."""
    
    @abstractmethod
    def evaluate(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """Evaluate objective function."""
        
class MaxSharpeRatio(ExposureObjective):
    """Maximize risk-adjusted returns at exposure level."""
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
class MinVolatility(ExposureObjective):
    """Minimize portfolio volatility."""
    
class RiskParity(ExposureObjective):
    """Equal risk contribution from each exposure."""
    
class MaxDiversification(ExposureObjective):
    """Maximize diversification ratio."""
    
class TargetVolatility(ExposureObjective):
    """Achieve specific volatility target."""
    def __init__(self, target_vol: float):
        self.target_vol = target_vol
```

### 3. Exposure-Specific Constraints

```python
# src/optimization/exposure_constraints.py
@dataclass
class ExposureConstraints:
    """Constraints for exposure optimization."""
    
    # Basic weight constraints
    min_weights: Dict[str, float] = field(default_factory=dict)
    max_weights: Dict[str, float] = field(default_factory=dict)
    
    # Category constraints
    category_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # e.g., {'equity_beta': (0.3, 0.7), 'alternatives': (0.0, 0.3)}
    
    # Leverage constraints
    max_total_notional: float = 1.5  # 150% max leverage
    max_leverage_exposures: float = 0.5  # Max 50% in leveraged strategies
    
    # Risk constraints
    max_volatility: Optional[float] = None
    max_tracking_error: Optional[float] = None
    
    # Diversification constraints
    min_exposures: int = 3
    max_exposures: int = 10
    max_concentration: float = 0.4  # Max 40% in any exposure
    
    # Correlation constraints
    max_correlation_pairs: float = 0.8  # Avoid highly correlated exposures
    
    def validate(self, exposures: List[str]) -> None:
        """Validate constraints are feasible."""
```

### 4. Enhanced Risk Model Integration

```python
# src/optimization/exposure_risk_model.py
class ExposureRiskModel:
    """Enhanced risk model for exposure optimization."""
    
    def __init__(
        self,
        risk_estimator: ExposureRiskEstimator,
        confidence_level: float = 0.95
    ):
        self.risk_estimator = risk_estimator
        self.confidence_level = confidence_level
        
    def get_covariance_matrix(
        self,
        exposures: List[str],
        estimation_date: datetime
    ) -> np.ndarray:
        """Get covariance matrix with confidence adjustments."""
        
    def get_correlation_matrix(
        self,
        exposures: List[str],
        estimation_date: datetime
    ) -> np.ndarray:
        """Get correlation matrix."""
        
    def calculate_portfolio_risk(
        self,
        weights: Dict[str, float],
        include_specific_risk: bool = True
    ) -> float:
        """Calculate total portfolio risk."""
        
    def calculate_risk_contributions(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate marginal risk contributions."""
        
    def calculate_diversification_ratio(
        self,
        weights: Dict[str, float]
    ) -> float:
        """Calculate portfolio diversification ratio."""
```

### 5. Optimization Result Structure

```python
@dataclass
class ExposureOptimizationResult:
    """Results from exposure optimization."""
    
    # Optimal weights
    weights: Dict[str, float]
    
    # Risk metrics
    expected_return: float
    volatility: float
    sharpe_ratio: float
    
    # Risk decomposition
    risk_contributions: Dict[str, float]
    correlation_matrix: pd.DataFrame
    
    # Optimization details
    objective_value: float
    optimization_status: str
    solver_stats: Dict[str, Any]
    
    # Constraint analysis
    active_constraints: List[str]
    shadow_prices: Dict[str, float]
    
    def summary(self) -> str:
        """Generate readable summary of results."""
        
    def plot_allocation(self) -> None:
        """Visualize exposure allocation."""
        
    def plot_risk_contributions(self) -> None:
        """Visualize risk contributions."""
```

## Implementation Details

### Phase 1: Core Optimization Engine

1. **CVXPy Integration**:
```python
def _solve_convex_optimization(
    self,
    objective: ExposureObjective,
    constraints: ExposureConstraints,
    covariance: np.ndarray,
    returns: Optional[np.ndarray]
) -> np.ndarray:
    """Solve using CVXPy for convex problems."""
    
    n = len(self.exposures)
    weights = cp.Variable(n)
    
    # Build objective
    if isinstance(objective, MinVolatility):
        obj = cp.Minimize(cp.quad_form(weights, covariance))
    elif isinstance(objective, MaxSharpeRatio):
        # Transform to convex form
        ...
```

2. **Special Case Solvers**:
- Analytical solution for min variance
- Risk parity iterative solver
- Black-Litterman integration

### Phase 2: Advanced Features

1. **Multi-Period Optimization**:
```python
def optimize_multiperiod(
    self,
    exposures: List[str],
    rebalance_dates: List[datetime],
    transaction_costs: Dict[str, float]
) -> List[ExposureOptimizationResult]:
    """Optimize over multiple periods with transaction costs."""
```

2. **Scenario-Based Optimization**:
```python
def optimize_with_scenarios(
    self,
    exposures: List[str],
    scenarios: List[Scenario],
    scenario_probs: List[float]
) -> ExposureOptimizationResult:
    """Optimize considering multiple scenarios."""
```

3. **Robust Optimization**:
```python
def optimize_robust(
    self,
    exposures: List[str],
    uncertainty_sets: Dict[str, UncertaintySet]
) -> ExposureOptimizationResult:
    """Robust optimization with parameter uncertainty."""
```

## Testing Requirements

### Unit Tests
1. Test each objective function independently
2. Verify constraint validation and enforcement
3. Check numerical stability of solvers
4. Test edge cases (infeasible problems, etc.)

### Integration Tests
1. Full optimization workflow with real exposure data
2. Compare results across different objectives
3. Verify risk calculations match expected values
4. Test with various constraint combinations

### Performance Tests
1. Optimization time for various problem sizes
2. Memory usage for large correlation matrices
3. Numerical accuracy vs. speed tradeoffs

## Success Criteria

- [ ] Core optimization engine working with 5+ objectives
- [ ] Comprehensive constraint system implemented
- [ ] Risk model integration complete
- [ ] Results include full risk decomposition
- [ ] Performance: < 1 second for 20 exposures
- [ ] Numerical stability verified
- [ ] Clear documentation and examples
- [ ] Integration with existing risk estimator

## Example Usage

```python
# Setup
risk_estimator = ExposureRiskEstimator(
    exposure_universe,
    forecast_horizon=21
)
optimizer = ExposureOptimizer(risk_estimator)

# Define constraints
constraints = ExposureConstraints(
    min_weights={'cash_rate': 0.0},
    max_weights={'trend_following': 0.3},
    category_limits={
        'equity_beta': (0.3, 0.6),
        'alternatives': (0.1, 0.3)
    },
    max_total_notional=1.3,  # 130% leverage limit
    max_volatility=0.12  # 12% annual volatility target
)

# Optimize
result = optimizer.optimize(
    exposures=['us_large_equity', 'broad_ust', 'trend_following', 
               'real_estate', 'gold'],
    estimation_date=datetime(2025, 7, 11),
    objective=MaxSharpeRatio(risk_free_rate=0.04),
    constraints=constraints
)

# Examine results
print(result.summary())
# Optimal Exposure Weights:
# - us_large_equity: 45.2%
# - broad_ust: 23.1%
# - trend_following: 20.0%
# - real_estate: 8.3%
# - gold: 3.4%
# Expected Return: 8.2%
# Volatility: 11.8%
# Sharpe Ratio: 0.36

result.plot_risk_contributions()
```

## Notes

- This is the core of the exposure-based system
- Must integrate cleanly with risk estimator
- Consider computational efficiency for large problems
- Document mathematical formulations clearly
- Build in stages - start with basic objectives, add advanced later

This creates a mathematically rigorous optimization engine working directly with economic exposures rather than fund vehicles.
