# History Optimization Changes

## Key Updates Made to Exposure Universe

### 1. Trend Following - Major Improvement
**Old**: ETFs with 2019+ inception (DBMF, KMLM, RSST, CTA)
**New**: Mutual funds with 10+ year history:
- **ABYIX** - Abbey Capital Futures Strategy Fund
- **AHLIX** - Aspect Core Diversified Program
- **AQMNX** - AQR Managed Futures Strategy Fund  
- **ASFYX** - AlphaSimplex Managed Futures Strategy Fund

These funds provide history back to 2010 or earlier, giving us 15+ years of data for trend following exposure.

### 2. General Principle Applied
Added configuration guideline:
```yaml
# History prioritization guidelines:
# - Always prefer funds/indices with 10+ years of history
# - Use mutual funds over ETFs when they provide longer history
# - Consider using indices for backfilling pre-fund-inception data
# - Document any data splicing or approximations clearly
```

### 3. Other Exposures Reviewed
- **Factor/Style - Equities**: QMNIX (2014) is reasonable; ETF composite limited to 2014
- **Factor/Style - Other**: QSPIX (2013) provides good history
- **Dynamic Global Bonds**: Noted PFIUX has history to 1993, PFUIX as alternative

### 4. Implementation Strategy
For each exposure, the config now prioritizes:
1. **Primary**: Funds/indices with longest history
2. **Secondary**: Newer ETFs for liquidity/trading
3. **Fallback**: Composites or proxies when needed

## Benefits
- Trend following now has 15+ years of data (vs. 5 years with ETFs)
- Better parameter estimation for optimization
- More reliable covariance matrices
- Ability to backtest through multiple market cycles

## Next Steps for Implementation
When loading data, the system should:
1. Use the longest available history source
2. Document any data transitions
3. Validate consistency across sources
4. Consider splicing methods for combining old/new data
