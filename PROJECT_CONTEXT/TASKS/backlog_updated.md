# Updated Backlog - Risk Premium Focus

**Updated**: 2025-01-07 by Desktop Claude

## Immediate Priority: Risk Premium Decomposition

### 🔴 Critical Path (Must Do First)
1. **Risk Premium Decomposition Framework** (NEW - Task in progress)
   - Implement RiskPremiumEstimator class
   - Integrate return decomposition with risk estimation
   - Re-optimize parameters on risk premia
   - Create dual output (RP + total returns)

2. **Update Parameter Optimization for Risk Premia**
   - Re-run optimization on decomposed risk premia
   - Compare optimal parameters for RP vs total returns
   - Validate forecasting accuracy on risk premia

3. **Revise Notebooks for Risk Premium Analysis**
   - Update exposure risk estimation notebook
   - Show both RP and total return volatilities
   - Explain the decomposition visually

### 🟡 High Priority (After RP Implementation)

4. **GARCH Integration with Risk Premia**
   - Add GARCH to parameter optimization
   - Test on risk premia specifically
   - Compare GARCH vs EWMA for different exposures

5. **Real Return Optimization** (Builds on RP framework)
   - Now makes more sense with decomposed returns
   - Real RP = Nominal RP - Expected Inflation
   - Natural extension of decomposition

6. **Component Correlation Modeling**
   - Model correlations between inflation/RF/RP
   - Critical for recombining to total returns
   - Affects portfolio hedging properties

### 🟢 Medium Priority

7. **Risk Attribution System**
   - Decompose portfolio risk into sources
   - Inflation risk vs RF risk vs RP risk
   - Natural output from decomposition

8. **Enhanced Visualization**
   - Waterfall charts for return decomposition
   - Risk contribution by component
   - RP vs total return efficient frontiers

9. **Inflation Hedging Analysis**
   - Which exposures hedge inflation?
   - Based on component correlations
   - Natural from decomposition

### 🔵 Future Enhancements

10. **Dynamic Component Modeling**
    - Time-varying inflation expectations
    - Term structure of real rates
    - Regime-dependent correlations

11. **Factor Risk Premia**
    - Decompose factor returns
    - Pure factor risk premia
    - Cross-sectional RP analysis

12. **Scenario Analysis on Components**
    - Inflation shock scenarios
    - Real rate scenarios
    - Risk premium compression

## Removed/Deprioritized Items

- ~~Simple web interface~~ - Wait until RP framework proves value
- ~~Basic reporting~~ - Need RP-aware reports
- ~~Simple expected returns~~ - User explicitly deprioritized

## Key Insight
The shift to risk premium decomposition fundamentally changes many downstream tasks. Items that seemed ready for implementation now need reconsideration through the RP lens.

## Success Metrics
1. Risk premium volatility < Total return volatility for bonds ✓
2. More stable optimal portfolios over time
3. Better out-of-sample performance
4. Natural inflation hedging identification
5. Alignment with academic theory
