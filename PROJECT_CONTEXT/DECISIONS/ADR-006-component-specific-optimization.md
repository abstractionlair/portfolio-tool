# Architecture Decision Record: Component-Specific Parameter Optimization

**Date**: 2025-07-10  
**Status**: Proposed  
**Author**: Desktop Claude

## Context

The current parameter optimization framework uses the same objective function (minimize risk premium volatility forecasting error) for all components of the covariance matrix. However, different components of portfolio optimization have fundamentally different objectives:

- **Volatility**: Need accurate risk forecasts for position sizing
- **Correlation**: Need stable, well-conditioned matrices for optimization
- **Expected Returns**: Need directional accuracy and unbiased estimates

Using the same parameters for all components is theoretically suboptimal and practically limiting.

## Decision

We will implement a component-specific optimization framework that:

1. **Separates optimization objectives** for volatility, correlation, and expected returns
2. **Allows different parameters** for each component (method, lookback, frequency, etc.)
3. **Provides a unified interface** for downstream consumption
4. **Caches optimal parameters** for production use

## Architecture Design

### 1. Component Optimizers

```python
# Base class for all component optimizers
class ComponentOptimizer(ABC):
    @abstractmethod
    def optimize_parameters(self, exposure_ids: List[str], 
                          start_date: datetime, 
                          end_date: datetime) -> ComponentOptimalParameters
    
    @abstractmethod
    def get_optimization_objective(self) -> str

# Specific implementations
class VolatilityOptimizer(ComponentOptimizer):
    """Optimizes for volatility forecast accuracy"""
    optimization_objectives = ['mse', 'qlike', 'realized_vol_correlation']
    
class CorrelationOptimizer(ComponentOptimizer):
    """Optimizes for correlation matrix stability and accuracy"""
    optimization_objectives = ['matrix_stability', 'condition_number', 'eigenvalue_stability']
    
class ExpectedReturnOptimizer(ComponentOptimizer):
    """Optimizes for return prediction accuracy"""
    optimization_objectives = ['directional_accuracy', 'information_ratio', 'hit_rate']
```

### 2. Unified Parameter Set

```python
@dataclass
class ComponentOptimalParameters:
    """Optimal parameters for a specific component"""
    component_type: str  # 'volatility', 'correlation', 'expected_returns'
    exposure_id: str
    parameters: Dict[str, Any]
    score: float
    metadata: Dict[str, Any]

@dataclass 
class UnifiedOptimalParameters:
    """Complete parameter set for all components"""
    volatility_params: Dict[str, ComponentOptimalParameters]
    correlation_params: ComponentOptimalParameters  
    expected_return_params: Dict[str, ComponentOptimalParameters]
    optimization_date: datetime
    validation_period: Tuple[datetime, datetime]
```

### 3. Optimization Orchestrator

```python
class ComponentOptimizationOrchestrator:
    """Orchestrates optimization across all components"""
    
    def __init__(self):
        self.volatility_optimizer = VolatilityOptimizer()
        self.correlation_optimizer = CorrelationOptimizer()
        self.return_optimizer = ExpectedReturnOptimizer()
        
    def optimize_all_components(self, 
                               exposure_ids: List[str],
                               start_date: datetime,
                               end_date: datetime,
                               parallel: bool = True) -> UnifiedOptimalParameters:
        """Run optimization for all components"""
        
    def save_optimal_parameters(self, params: UnifiedOptimalParameters, 
                               path: str = "config/optimal_parameters.yaml"):
        """Save parameters for production use"""
        
    def load_optimal_parameters(self, path: str) -> UnifiedOptimalParameters:
        """Load previously optimized parameters"""
```

### 4. Production Interface (Entry Points)

```python
class OptimizedRiskEstimator:
    """
    Main entry point for production use.
    Uses pre-optimized parameters to provide best predictions.
    """
    
    def __init__(self, 
                 parameter_file: str = "config/optimal_parameters.yaml",
                 auto_optimize: bool = False):
        """
        Initialize with optimal parameters.
        
        Args:
            parameter_file: Path to optimized parameters
            auto_optimize: Whether to run optimization if params missing
        """
        self.optimal_params = self._load_or_optimize_parameters()
        
    def get_volatility_estimate(self, 
                               exposure_id: str,
                               estimation_date: datetime,
                               horizon: int = 21) -> VolatilityEstimate:
        """Get optimized volatility estimate for an exposure"""
        
    def get_correlation_matrix(self, 
                              exposure_ids: List[str],
                              estimation_date: datetime) -> pd.DataFrame:
        """Get optimized correlation matrix"""
        
    def get_expected_returns(self, 
                           exposure_ids: List[str],
                           estimation_date: datetime,
                           horizon: int = 21) -> pd.Series:
        """Get optimized expected return estimates"""
        
    def get_full_covariance_matrix(self,
                                  exposure_ids: List[str],
                                  estimation_date: datetime) -> pd.DataFrame:
        """
        Get complete covariance matrix using optimal parameters.
        Combines optimally estimated volatilities and correlations.
        """

# Convenience functions for simple access
def get_best_risk_estimates(exposure_ids: List[str], 
                          estimation_date: datetime) -> CovarianceMatrix:
    """Simple entry point - just get the best estimates"""
    estimator = OptimizedRiskEstimator()
    return estimator.get_full_covariance_matrix(exposure_ids, estimation_date)
```

### 5. Optimization Objectives

#### Volatility Optimization
- **Primary**: Minimize out-of-sample volatility forecast error
- **Secondary**: Maximize realized volatility correlation
- **Constraints**: Stable estimates, reasonable parameter ranges

#### Correlation Optimization  
- **Primary**: Maximize matrix stability over time
- **Secondary**: Minimize condition number (numerical stability)
- **Constraints**: Positive semi-definite, full rank

#### Expected Return Optimization
- **Primary**: Maximize directional accuracy
- **Secondary**: Maximize information ratio
- **Constraints**: Unbiased estimates, reasonable Sharpe ratios

## Implementation Plan

### Phase 1: Core Framework
1. Create base `ComponentOptimizer` class
2. Implement component-specific optimizers
3. Build optimization orchestrator
4. Design parameter storage format

### Phase 2: Optimization Implementation  
1. Implement volatility optimization with multiple objectives
2. Implement correlation optimization with stability metrics
3. Implement expected return optimization
4. Add parallel processing support

### Phase 3: Production Interface
1. Build `OptimizedRiskEstimator` class
2. Create convenience functions
3. Add parameter caching and loading
4. Build comprehensive tests

### Phase 4: Analysis Tools
1. Create analysis notebook
2. Build visualization tools
3. Add performance comparison
4. Document optimal parameters

## Benefits

1. **Theoretical Soundness**: Each component optimized for its actual use
2. **Better Performance**: Improved portfolio outcomes
3. **Flexibility**: Can update components independently  
4. **Production Ready**: Clean interface for downstream use
5. **Interpretability**: Clear why parameters differ

## Risks and Mitigations

**Risk**: Overfitting to validation period
- **Mitigation**: Use multiple validation periods, regularization

**Risk**: Increased complexity
- **Mitigation**: Clean interfaces, good documentation

**Risk**: Computational cost
- **Mitigation**: Cache results, parallel processing

## Success Metrics

1. Volatility estimates: >10% improvement in forecast accuracy
2. Correlation matrices: >20% improvement in stability
3. Expected returns: >15% improvement in directional accuracy
4. End-to-end portfolio performance: Improved Sharpe ratios
