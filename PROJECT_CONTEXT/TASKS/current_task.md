# Task: Implement Leverage-Aware Optimization Engine

**Status**: COMPLETED  
**Assigned**: Claude Code  
**Priority**: High  
**Dependencies**: Portfolio, Exposures, and Analytics modules must exist

## Objective
Implement a sophisticated portfolio optimization engine that properly handles leveraged funds and optimizes based on true underlying exposures rather than naive fund positions. This is the core algorithmic component that will generate optimal portfolio allocations.

## Requirements

### 1. Core Optimization Framework

Create an `OptimizationEngine` class with support for multiple optimization methods:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
import cvxpy as cp

class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MAX_SHARPE = "maximize_sharpe_ratio"
    MIN_VOLATILITY = "minimize_volatility"
    MAX_RETURN = "maximize_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "maximize_diversification"
    MIN_TRACKING_ERROR = "minimize_tracking_error"

@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0  # Minimum position weight
    max_weight: float = 1.0  # Maximum position weight
    max_total_notional: float = 2.0  # Maximum leverage
    target_volatility: Optional[float] = None
    max_exposure_per_type: Optional[Dict[ExposureType, float]] = None
    min_exposure_per_type: Optional[Dict[ExposureType, float]] = None
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None
    long_only: bool = True
    
class OptimizationEngine:
    def __init__(self, analytics: PortfolioAnalytics, fund_map: FundExposureMap):
        self.analytics = analytics
        self.fund_map = fund_map
        self.calculator = ExposureCalculator(fund_map)
```

### 2. Mean-Variance Optimization

Implement classic Markowitz optimization with leverage awareness:

```python
def optimize_mean_variance(
    self,
    symbols: List[str],
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    constraints: OptimizationConstraints,
    objective: ObjectiveType = ObjectiveType.MAX_SHARPE,
    risk_free_rate: float = 0.02
) -> OptimizationResult:
    """
    Perform mean-variance optimization.
    
    Must handle:
    - Leveraged funds properly (e.g., RSSB with 200% notional)
    - Exposure-based constraints
    - Various objectives (max Sharpe, min vol, etc.)
    """
```

### 3. Risk Parity Optimization

Implement risk parity that works with leveraged funds:

```python
def optimize_risk_parity(
    self,
    symbols: List[str],
    covariance_matrix: np.ndarray,
    constraints: OptimizationConstraints,
    use_leverage: bool = True
) -> OptimizationResult:
    """
    Allocate risk equally across assets/exposures.
    
    Should support:
    - Equal risk contribution
    - Hierarchical risk parity
    - Leverage to achieve target volatility
    """
```

### 4. Black-Litterman Model

Implement Black-Litterman for incorporating views:

```python
@dataclass
class MarketView:
    """Represents a view on asset returns."""
    assets: List[str]
    view_type: str  # 'absolute' or 'relative'
    expected_return: float
    confidence: float

def black_litterman_returns(
    self,
    symbols: List[str],
    market_weights: np.ndarray,
    covariance_matrix: np.ndarray,
    views: List[MarketView],
    risk_aversion: float = 2.5
) -> np.ndarray:
    """Calculate Black-Litterman expected returns."""
```

### 5. Exposure-Based Optimization

The key differentiator - optimize on true exposures:

```python
def optimize_exposures(
    self,
    symbols: List[str],
    target_exposures: Dict[ExposureType, float],
    constraints: OptimizationConstraints,
    minimize_cost: bool = True
) -> OptimizationResult:
    """
    Find fund weights that best match target exposure profile.
    
    Example: Target 100% equity + 50% managed futures
    Could achieve with: 67% RSST (150% notional) + 33% cash
    Or with: 50% SPY + 25% RSST + 25% cash
    
    Should minimize tracking error to target exposures
    while respecting constraints.
    """
```

### 6. Constraint System

Implement sophisticated constraints:

```python
def build_exposure_constraints(
    self,
    weights: cp.Variable,
    symbols: List[str],
    constraints: OptimizationConstraints
) -> List[cp.Constraint]:
    """
    Build constraints based on true exposures.
    
    Example constraints:
    - Max 150% equity exposure
    - Min 20% bond exposure
    - Max 200% total notional
    """
```

### 7. Return and Risk Estimation

Historical and forward-looking estimates:

```python
class ReturnEstimator:
    def estimate_expected_returns(
        self,
        symbols: List[str],
        method: str = 'historical',
        lookback_years: int = 5
    ) -> np.ndarray:
        """Estimate expected returns using various methods."""
    
    def estimate_covariance_matrix(
        self,
        symbols: List[str],
        method: str = 'sample',
        lookback_years: int = 5,
        frequency: str = 'daily'
    ) -> np.ndarray:
        """Estimate covariance matrix with shrinkage options."""
```

### 8. Optimization Results

Comprehensive result object:

```python
@dataclass
class OptimizationResult:
    """Results from portfolio optimization."""
    weights: Dict[str, float]
    objective_value: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    exposures: Dict[ExposureType, float]
    total_notional: float
    success: bool
    message: str
    
    # Additional diagnostics
    effective_assets: int  # Number of non-zero positions
    concentration_ratio: float  # Sum of squared weights
    diversification_ratio: float
    
    def to_trades(self, current_portfolio: Portfolio, prices: Dict[str, float]) -> List[Trade]:
        """Convert optimal weights to actual trades."""
```

### 9. Backtesting Integration

Test optimization strategies:

```python
class OptimizationBacktest:
    def backtest_strategy(
        self,
        optimization_method: Callable,
        rebalance_frequency: str = 'monthly',
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Backtest an optimization strategy."""
```

## Implementation Steps

1. Create `src/optimization/` package structure
2. Implement base `OptimizationEngine` class
3. Add mean-variance optimization with cvxpy
4. Implement risk parity methods
5. Add Black-Litterman model
6. Build exposure-based optimization
7. Create comprehensive constraint system
8. Add return/risk estimation utilities
9. Write thorough tests
10. Create example demonstrating various optimization approaches

## Test Cases

1. Basic mean-variance optimization
2. Optimization with leveraged funds
3. Exposure-based constraints working correctly
4. Risk parity with leverage
5. Black-Litterman with various views
6. Handling infeasible problems gracefully
7. Large portfolio optimization performance
8. Backtesting optimization strategies

## Key Considerations

- **Numerical Stability**: Use appropriate solvers for different problem types
- **Leverage Handling**: Ensure total notional exposure is tracked correctly
- **Transaction Costs**: Consider implementation costs in optimization
- **Rebalancing**: Support periodic rebalancing with bands
- **Solver Choice**: CVXPY supports multiple backends (OSQP, SCS, etc.)
- **Performance**: Optimize for portfolios with 100+ assets

## Success Criteria

- [x] All optimization methods produce valid portfolios
- [x] Leveraged funds handled correctly in all methods
- [x] Exposure constraints enforced properly
- [x] Results include comprehensive diagnostics
- [x] Performance acceptable for large portfolios
- [x] Clear documentation with examples
- [x] Comprehensive test suite implemented

## Implementation Summary

**Completed on**: 2025-07-05

**What was implemented**:
1. ✅ Complete `src/optimization/` package with all required modules
2. ✅ `OptimizationEngine` class with multiple optimization methods
3. ✅ Mean-variance optimization (max Sharpe, min volatility, max return) 
4. ✅ Risk parity optimization with leverage support
5. ✅ Black-Litterman model with market views
6. ✅ Exposure-based optimization (key differentiator)
7. ✅ Comprehensive constraint system with leverage awareness
8. ✅ Return and risk estimation utilities with multiple methods
9. ✅ Trade generation utilities (`TradeGenerator`, `Trade` classes)
10. ✅ Comprehensive test suite (`tests/test_optimization.py`)
11. ✅ Working demonstration script (`examples/optimization_demo.py`)

**Key features**:
- Proper handling of leveraged funds (e.g., 3x ETFs, balanced funds)
- Exposure-based constraints and optimization
- Integration with existing Portfolio, Analytics, and Exposures modules
- CVXPY-based convex optimization with fallback handling
- Comprehensive result diagnostics and trade generation
- Multiple estimation methods (historical, CAPM, shrinkage, exponential weighting)

**Files created**:
- `src/optimization/__init__.py`
- `src/optimization/engine.py`
- `src/optimization/methods.py` 
- `src/optimization/estimators.py`
- `src/optimization/constraints.py`
- `src/optimization/trades.py`
- `tests/test_optimization.py`
- `examples/optimization_demo.py`

## Example Usage

```python
# Setup
engine = OptimizationEngine(analytics, fund_map)
estimator = ReturnEstimator(market_data)

# Estimate returns and risk
symbols = ['SPY', 'TLT', 'RSSB', 'RSST', 'GLD']
expected_returns = estimator.estimate_expected_returns(symbols)
cov_matrix = estimator.estimate_covariance_matrix(symbols)

# Define constraints
constraints = OptimizationConstraints(
    min_weight=0.0,
    max_weight=0.4,
    max_total_notional=1.5,  # 150% gross exposure
    max_exposure_per_type={
        ExposureType.US_LARGE_EQUITY: 1.0,  # Max 100% equity
        ExposureType.MANAGED_FUTURES: 0.3,  # Max 30% managed futures
    }
)

# Optimize
result = engine.optimize_mean_variance(
    symbols=symbols,
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    constraints=constraints,
    objective=ObjectiveType.MAX_SHARPE
)

print(f"Optimal weights: {result.weights}")
print(f"Expected return: {result.expected_return:.2%}")
print(f"Expected volatility: {result.expected_volatility:.2%}")
print(f"Total notional exposure: {result.total_notional:.1%}")

# Show exposures
print("\nResulting exposures:")
for exp_type, exposure in result.exposures.items():
    print(f"  {exp_type.value}: {exposure:.1%}")
```

## Notes

- Start with mean-variance as it's the foundation
- Risk parity is particularly interesting with leveraged funds
- Exposure-based optimization is the key differentiator
- Consider adding more exotic objectives later (e.g., CVaR optimization)
- Integration with tax-aware features can come later
