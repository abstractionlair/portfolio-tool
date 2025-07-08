# Architecture Decision Record: Risk Premium Decomposition

**Date**: 2025-01-07  
**Status**: Accepted  
**Author**: Desktop Claude

## Context
The current risk estimation framework estimates volatilities and correlations on total returns. However, for portfolio optimization, we care about the compensation for risk (risk premium), not total returns which include risk-free rate changes and inflation.

## Decision
We will implement a comprehensive risk premium decomposition framework that:
1. Decomposes all exposure returns into: Inflation + Real Risk-Free Rate + Risk Premium
2. Estimates volatilities and correlations on the RISK PREMIUM component
3. Optimizes parameters specifically for risk premium forecasting
4. Provides both risk premium and total return estimates

## Rationale

### Theoretical Foundation
- Modern asset pricing theory prices risk premia, not total returns
- Risk-free rate volatility is not a compensated risk
- True diversification comes from uncorrelated risk premia
- Inflation is a common factor affecting all assets

### Practical Benefits
1. **Better Risk Understanding**: Separates compensated from uncompensated volatility
2. **Improved Optimization**: Portfolio weights based on true risk/return tradeoffs
3. **Clearer Attribution**: Can attribute returns to inflation, rates, or risk premia
4. **Academic Alignment**: Matches how institutional investors think

### Example Impact
For bonds:
- Total return volatility might be 5% (includes duration risk from rate changes)
- Risk premium volatility might be 2% (credit spread changes)
- Optimization should care about the 2%, not the 5%

## Implementation Approach

### Decomposition Formula
```
Total Return = Inflation + Real Risk-Free Rate + Risk Premium

Where:
- Inflation: CPI or PCE from FRED
- Real Risk-Free: TIPS yield or (Nominal RF - Expected Inflation)
- Risk Premium: Residual (includes all compensation for risk)
```

### Multi-Method Estimation
1. **Historical**: Simple standard deviation of risk premia
2. **EWMA**: Exponentially weighted with various lambdas
3. **GARCH**: For volatility clustering in risk premia
4. **Shrinkage**: For stable correlation estimates

### Parameter Optimization
- Optimize on risk premium forecast accuracy, not total returns
- May find different optimal parameters than total return optimization
- Validate on out-of-sample risk premium volatility

### Dual Output
Provide both:
- Risk premium volatilities/correlations (for optimization)
- Total return volatilities/correlations (for implementation)

## Consequences

### Positive
- More theoretically sound portfolio optimization
- Better understanding of risk sources
- Improved long-term forecasting
- Natural inflation hedging analysis

### Negative
- Increased complexity
- Requires more data (inflation, risk-free rates)
- Decomposition assumptions may not hold perfectly
- More computational overhead

### Risks
- Decomposition quality depends on FRED data availability
- Real-time estimation challenges (data lags)
- Component correlations need careful modeling
- May produce unintuitive results initially

## Alternatives Considered

1. **Status Quo**: Continue with total return estimation
   - Simpler but theoretically inferior
   
2. **Excess Returns Only**: Use returns minus risk-free rate
   - Misses inflation component
   - Still includes real rate volatility

3. **Factor Models**: Use statistical factor decomposition
   - More complex, less interpretable
   - Doesn't align with economic reasoning

## Implementation Plan

1. Create `RiskPremiumEstimator` extending current framework
2. Integrate with existing `ReturnDecomposer`
3. Add parameter optimization for risk premia
4. Update notebooks and examples
5. Validate improvements in backtesting

## Success Metrics

- Risk premium volatility < Total return volatility for bonds
- Improved out-of-sample forecast accuracy
- More stable correlation estimates
- Better portfolio performance in backtests
