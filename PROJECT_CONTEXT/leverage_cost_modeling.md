# Leverage Cost Modeling - Risk-Free Rate Exposure

## Purpose
The Cash/Risk-Free Rate exposure serves critical functions in portfolio analysis:

### 1. Leverage Cost Calculation
When replicating leveraged funds (e.g., Return Stacked ETFs), the cost of leverage is typically:
- **Cost = Risk-Free Rate + Spread**
- Spread varies by fund (typically 0.5% - 1.5% for ETFs)
- Essential for accurate fund decomposition

### 2. Fund Replication Example
For a 150% leveraged fund like RSSB:
```
RSSB Returns = 
  + 100% Equity Returns
  + 100% Bond Returns  
  - 50% × (Risk-Free Rate + Leverage Spread)
```

### 3. Why Not Just Use Short-Term Treasuries?
- Need the absolute shortest duration (0-3 months)
- Must match funding rates used by leveraged funds
- SHY (1-3 year) is too long for accurate modeling
- Cash rate better represents actual borrowing costs

## Implementation Options

### ETF Options (0-3 Month T-Bills)
1. **BIL** - SPDR 1-3 Month T-Bill ETF
2. **SHV** - iShares 0-3 Month Treasury ETF  
3. **SGOV** - iShares 0-3 Month Treasury Bond ETF

### Direct Rate Data
- **FRED DGS3MO** - 3-Month Treasury Constant Maturity Rate
- **Fed Funds Rate** - Alternative for overnight funding
- **SOFR** - Secured Overnight Financing Rate (modern benchmark)

## Usage in Optimization

When optimizing portfolios with leveraged funds:
1. Include cash rate as a "negative" position (borrowing cost)
2. Constrain based on total leverage employed
3. Use for attribution: what returns come from leverage vs. selection

## Practical Considerations

- **Historical Analysis**: Use consistent rate series over time
- **International Funds**: May need currency-specific rates
- **Spread Estimation**: Can calibrate empirically or use fixed assumption
- **Tax Impact**: Interest expense may be deductible

## Connection to Your Research
Your empirical finding that "cost of leverage led to the best fit" validates this approach. By explicitly modeling the risk-free rate + spread, you can:
- Better decompose Return Stacked fund returns
- More accurately estimate the value added by leverage
- Optimize leverage usage across the portfolio
