# Task: Risk Premium Decomposition and Estimation Framework

**Status**: COMPLETE  
**Priority**: CRITICAL - Core requirement for proper risk estimation  
**Estimated Time**: 10-12 hours  
**Dependencies**: Return decomposition and parameter optimization exist but need integration

## Objective
Create a comprehensive framework that:
1. Decomposes exposure returns into fundamental components (inflation + real risk-free + risk premium)
2. Estimates volatilities and correlations on the RISK PREMIA, not total returns
3. Optimizes parameters specifically for risk premium estimation
4. Provides both risk premium AND total return estimates for portfolio construction

## Motivation
- Portfolio optimization should focus on compensation for risk (the premium), not total returns
- Risk-free rate volatility is not a risk that needs compensation
- Inflation volatility affects all assets similarly
- True diversification comes from uncorrelated risk premia, not uncorrelated total returns
- This approach aligns with academic asset pricing theory

## Core Requirements

### 1. Return Decomposition for All Exposures
```
Total Return = Inflation + Real Risk-Free Rate + Risk Premium (Spread)

Where:
- Inflation: From FRED (CPI, PCE, etc.)
- Real Risk-Free: TIPS yields or nominal - expected inflation  
- Risk Premium: The compensation for bearing risk
```

### 2. Multi-Method Risk Estimation
Support multiple estimation methods on the decomposed risk premia:
- **Naive/Historical**: Simple standard deviation
- **EWMA**: With various lambda parameters (0.90 to 0.98)
- **GARCH**: With optimizable omega, alpha, beta parameters
- **Shrinkage**: Ledoit-Wolf and other shrinkage estimators

### 3. Parameter Optimization on Risk Premia
- Test parameters on RISK PREMIUM volatility forecasting, not total returns
- Validate correlation stability of risk premia
- Find optimal parameters for different forecast horizons
- May differ from total return optimal parameters

### 4. Dual Output: Risk Premia AND Total Returns
After estimation on risk premia, recombine to provide:
- Risk premium volatilities and correlations (for understanding true risk)
- Total return volatilities and correlations (for implementation)

## Implementation Plan

### Phase 1: Exposure Return Decomposition
**File**: `/src/optimization/risk_premium_estimator.py` (NEW)

```python
class RiskPremiumEstimator(ExposureRiskEstimator):
    """Enhanced risk estimator that works on decomposed risk premia."""
    
    def __init__(
        self,
        exposure_universe: ExposureUniverse,
        return_decomposer: ReturnDecomposer,
        parameter_optimizer: Optional[ParameterOptimizer] = None
    ):
        super().__init__(exposure_universe, parameter_optimizer)
        self.return_decomposer = return_decomposer
        self._decomposed_returns_cache = {}
    
    def load_and_decompose_exposure_returns(
        self,
        exposure_id: str,
        estimation_date: datetime,
        lookback_days: int,
        frequency: str = 'daily'
    ) -> Dict[str, pd.Series]:
        """
        Load returns and decompose into components.
        
        Returns:
            Dict with keys: 'total', 'inflation', 'real_rf', 'risk_premium'
        """
        # Load raw returns
        raw_returns = self._load_exposure_returns(...)
        
        # Decompose
        decomposition = self.return_decomposer.decompose_returns(
            raw_returns,
            start_date,
            end_date,
            frequency=frequency
        )
        
        return {
            'total': raw_returns,
            'inflation': decomposition['inflation'],
            'real_rf': decomposition['real_rf_rate'],
            'risk_premium': decomposition['spread']
        }
```

### Phase 2: Multi-Method Estimation on Risk Premia
```python
def estimate_risk_premium_volatility(
    self,
    exposure_id: str,
    estimation_date: datetime,
    method: str,  # 'historical', 'ewma', 'garch', 'shrinkage'
    parameters: Dict[str, Any],
    forecast_horizon: int
) -> RiskEstimate:
    """Estimate volatility of risk premium component."""
    
    # Get decomposed returns
    components = self.load_and_decompose_exposure_returns(...)
    risk_premium_returns = components['risk_premium']
    
    # Apply chosen method
    if method == 'ewma':
        volatility = self._estimate_ewma_volatility(
            risk_premium_returns,
            lambda_=parameters.get('lambda', 0.94),
            ...
        )
    # etc for other methods
```

### Phase 3: Parameter Optimization for Risk Premia
**File**: `/src/optimization/risk_premium_parameter_optimization.py` (NEW)

```python
class RiskPremiumParameterOptimizer(ParameterOptimizer):
    """Optimize parameters specifically for risk premium estimation."""
    
    def validate_risk_premium_forecast(
        self,
        method: str,
        parameters: Dict,
        exposure_returns: Dict[str, pd.Series],  # Decomposed returns
        test_start: datetime,
        test_end: datetime,
        horizon: int
    ) -> ValidationResult:
        """Test forecasting accuracy on risk premia, not total returns."""
        
        # Use risk premium component for validation
        risk_premia = {
            exp_id: components['risk_premium'] 
            for exp_id, components in exposure_returns.items()
        }
        
        # Validate volatility and correlation forecasts
        # ...
```

### Phase 4: Recombination for Total Return Estimates
```python
def get_combined_risk_estimates(
    self,
    exposures: List[str],
    estimation_date: datetime,
    forecast_horizon: int
) -> CombinedRiskEstimates:
    """
    Provide both risk premium and total return estimates.
    
    Returns:
        Object containing:
        - risk_premium_volatilities
        - risk_premium_correlations
        - total_return_volatilities  
        - total_return_correlations
        - component_volatilities (inflation, real_rf)
    """
    
    # Estimate risk premia volatilities/correlations
    rp_estimates = self.estimate_risk_premium_matrix(...)
    
    # Get component volatilities
    inflation_vol = self.estimate_inflation_volatility(...)
    real_rf_vol = self.estimate_real_rf_volatility(...)
    
    # Recombine for total return estimates
    # This requires careful handling of correlations between components
    total_estimates = self.recombine_to_total_returns(
        rp_estimates, inflation_vol, real_rf_vol
    )
```

## Key Design Decisions

### 1. Component Correlation Structure
- Risk premia may have different correlations than total returns
- Inflation affects all assets similarly (high correlation)
- Real risk-free rate changes affect duration assets
- Need to model cross-component correlations

### 2. Frequency Considerations
- Daily decomposition may be noisy
- Consider weekly or monthly for more stable estimates
- But maintain daily data for short-term forecasts

### 3. Missing Data Handling
- FRED data has lags (CPI ~2 weeks)
- Need forward-fill or nowcasting for recent periods
- Different frequencies for different components

### 4. Asset-Specific Considerations
- **Equities**: Risk premium ≈ total return - rf rate
- **Bonds**: Duration component is large part of volatility
- **Commodities**: No direct cash flows, special handling
- **Real Assets**: Already inflation-adjusted?

## Success Criteria
- [ ] All exposure returns decomposed into components
- [ ] Risk premium volatilities computed with multiple methods
- [ ] Parameters optimized specifically for risk premia
- [ ] Correlation structure preserved and estimable
- [ ] Can recombine to get total return estimates
- [ ] Validation shows improved forecast accuracy
- [ ] Clear documentation of what each number means

## Testing Requirements
1. Verify decomposition adds up: sum(components) = total return
2. Test risk premium volatility < total return volatility for bonds
3. Validate correlation differences between RP and total
4. Ensure positive semi-definite correlation matrices
5. Test parameter stability across time periods

## Example Usage
```python
# Initialize with decomposition
risk_estimator = RiskPremiumEstimator(
    exposure_universe,
    return_decomposer,
    parameter_optimizer
)

# Get risk premium estimates
rp_matrix = risk_estimator.get_risk_premium_matrix(
    exposures=['us_equity', 'bonds', 'commodities'],
    method='optimal',  # Uses optimized parameters
    forecast_horizon=252
)

# Get combined estimates
combined = risk_estimator.get_combined_risk_estimates(
    exposures, 
    include_total_returns=True
)

# Access different views
print(f"Risk Premium Vol: {combined.risk_premium_volatilities}")
print(f"Total Return Vol: {combined.total_return_volatilities}")
print(f"RP Correlation: {combined.risk_premium_correlations}")
```

## Next Steps After Implementation
1. Validate that risk premium approach improves portfolio optimization
2. Study component correlation stability
3. Implement inflation hedging analysis
4. Build attribution from RP to total return performance
