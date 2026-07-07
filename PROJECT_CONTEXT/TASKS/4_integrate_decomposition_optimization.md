# Task 4: Integrate Return Decomposition with Optimization Engine

**Status**: TODO  
**Priority**: MEDIUM - Enhances optimization quality  
**Estimated Time**: 3-4 hours  
**Dependencies**: Tasks 1, 2, & 3 complete

## Context
The return decomposition is working, and we have an optimization engine, but they're not connected. We need to integrate them so the optimizer can use decomposed returns (real returns, risk premiums) for better optimization.

## Problem
- Optimizer currently uses nominal returns
- No easy way to optimize based on real returns
- Risk premiums (spreads) are more stable than total returns
- Need to maintain backwards compatibility

## Requirements

### 1. Extend OptimizationEngine with Decomposition Support
Location: `/src/optimization/engine.py`

Add initialization parameter:
```python
class OptimizationEngine:
    def __init__(
        self,
        analytics: PortfolioAnalytics,
        fund_map: Optional[FundMap] = None,
        return_decomposer: Optional[ReturnDecomposer] = None,  # NEW
        use_real_returns: bool = False  # NEW
    ):
```

### 2. Add Return Decomposition Methods

```python
def get_decomposed_returns(
    self,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    frequency: str = "monthly",
    component: str = "total_return"  # or "real_return", "spread"
) -> pd.DataFrame:
    """Get decomposed returns for optimization.
    
    Args:
        symbols: List of symbols to get returns for
        start_date: Start date
        end_date: End date  
        frequency: Return frequency
        component: Which component to return:
            - 'total_return': Nominal returns (default)
            - 'real_return': Total return - inflation
            - 'spread': Risk premium over risk-free rate
            - 'excess_real': Real return - real risk-free rate
    
    Returns:
        DataFrame with returns for each symbol
    """
```

### 3. Update Optimization Methods

#### Mean-Variance Optimization
```python
def optimize_mean_variance(
    self,
    symbols: List[str],
    expected_returns: Optional[pd.Series] = None,
    covariance_matrix: Optional[pd.DataFrame] = None,
    constraints: Optional[List[Constraint]] = None,
    objective: ObjectiveType = ObjectiveType.MAX_SHARPE,
    risk_free_rate: float = 0.02,
    use_decomposed_returns: bool = False,  # NEW
    return_component: str = "total_return"  # NEW
) -> OptimizationResult:
    """Existing method with new parameters."""
    
    if use_decomposed_returns and self.return_decomposer:
        # Get decomposed returns
        returns_df = self.get_decomposed_returns(
            symbols, start_date, end_date, 
            component=return_component
        )
        
        # Recalculate expected returns and covariance
        if expected_returns is None:
            expected_returns = self.estimate_expected_returns(
                returns_df, method="historical"
            )
        if covariance_matrix is None:
            covariance_matrix = self.estimate_covariance(
                returns_df, method="sample"
            )
```

### 4. Add Inflation-Aware Optimization

New method:
```python
def optimize_real_returns(
    self,
    symbols: List[str],
    target_real_return: float,
    constraints: Optional[List[Constraint]] = None,
    inflation_forecast: float = 0.02
) -> OptimizationResult:
    """Optimize for real (inflation-adjusted) returns.
    
    Args:
        symbols: Symbols to optimize
        target_real_return: Target real return
        constraints: Portfolio constraints
        inflation_forecast: Expected inflation rate
        
    Returns:
        Optimization result with weights targeting real return
    """
```

### 5. Create Example: Real Return Optimization

Location: `/examples/real_return_optimization.py`

```python
"""Example: Optimizing for real returns using decomposition."""

def main():
    # Load portfolio and data
    portfolio = load_sample_portfolio()
    
    # Initialize with decomposer
    decomposer = ReturnDecomposer()
    engine = OptimizationEngine(
        analytics=analytics,
        fund_map=fund_map,
        return_decomposer=decomposer,
        use_real_returns=True
    )
    
    # Example 1: Optimize using real returns
    result = engine.optimize_mean_variance(
        symbols=['SPY', 'AGG', 'GLD', 'VNQ'],
        objective=ObjectiveType.MAX_SHARPE,
        use_decomposed_returns=True,
        return_component="real_return"
    )
    
    # Example 2: Optimize using risk premiums
    result = engine.optimize_mean_variance(
        symbols=['SPY', 'TLT', 'GLD'],
        objective=ObjectiveType.MAX_SHARPE,
        use_decomposed_returns=True,
        return_component="spread"
    )
    
    # Example 3: Target real return
    result = engine.optimize_real_returns(
        symbols=['SPY', 'AGG', 'TIPS', 'VNQ'],
        target_real_return=0.05,  # 5% real
        inflation_forecast=0.025   # 2.5% expected inflation
    )
```

### 6. Add Tests

Location: `/tests/test_optimization_decomposition.py`

```python
def test_decomposed_optimization():
    """Test optimization with decomposed returns."""
    
def test_real_return_optimization():
    """Test optimizing for real returns."""
    
def test_risk_premium_optimization():
    """Test optimization using risk premiums."""
    
def test_backwards_compatibility():
    """Ensure existing optimization still works."""
```

## Testing Instructions

1. Test basic integration:
```python
# Verify decomposer integration
engine = OptimizationEngine(analytics, return_decomposer=decomposer)
assert engine.return_decomposer is not None
```

2. Compare nominal vs real optimization:
```python
# Optimize with nominal returns
nominal_result = engine.optimize_mean_variance(
    symbols=['SPY', 'AGG', 'GLD'],
    use_decomposed_returns=False
)

# Optimize with real returns  
real_result = engine.optimize_mean_variance(
    symbols=['SPY', 'AGG', 'GLD'],
    use_decomposed_returns=True,
    return_component="real_return"
)

# Compare allocations
print("Nominal allocation:", nominal_result.weights)
print("Real allocation:", real_result.weights)
```

3. Test inflation scenarios:
```python
# High inflation scenario
high_inflation_result = engine.optimize_real_returns(
    symbols=['TIPS', 'GLD', 'VNQ', 'DBC'],  # Inflation hedges
    target_real_return=0.03,
    inflation_forecast=0.05  # 5% inflation
)
```

## Success Criteria
- [ ] OptimizationEngine accepts ReturnDecomposer
- [ ] Can optimize using real returns
- [ ] Can optimize using risk premiums
- [ ] Real return targeting works correctly
- [ ] Backwards compatibility maintained
- [ ] Example demonstrates value of decomposition
- [ ] Tests cover all new functionality
- [ ] Documentation updated

## Benefits of This Integration
1. **More stable optimization**: Risk premiums are more stable than total returns
2. **Inflation-aware portfolios**: Explicitly optimize for real purchasing power
3. **Better risk-adjusted returns**: Separate market risk from inflation risk
4. **Scenario analysis**: Test portfolios under different inflation regimes

## Notes
- Keep backwards compatibility - decomposition should be optional
- Start with simple integration, can add more features later
- Real returns = Nominal returns - Inflation (approximate)
- For optimization, geometric differences matter less than relative differences
- This is a differentiating feature for the portfolio optimizer
