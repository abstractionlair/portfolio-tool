# Exposure Universe - Implementation Decisions

## Futures Yield/Carry Exposure - Using QSPIX as Proxy

### Decision: Use QSPIX (AQR Style Premia Alternative Fund)

**Rationale:**
1. **Historical Data Available**: QSPIX launched October 31, 2013, providing ~11 years of actual returns
2. **Includes Carry**: One of four systematic factors in the fund (carry, value, momentum, defensive)
3. **Multi-Asset Implementation**: Applies carry across equities, bonds, currencies, commodities
4. **Professional Implementation**: AQR's systematic approach with risk management
5. **Investable**: Can actually allocate to this fund if desired

**Limitations to Accept:**
- Not pure carry (includes 3 other factors)
- According to fund documentation, QSPIX targets 10% volatility using leverage
- Higher expense ratio than ETFs (but captures sophisticated strategies)

**Performance Context:**
- 2018: -12.3%, 2019: -8.1%, 2020: -21.9%, 2021: +25.0%, 2022: +30.8%, 2023 (through Oct): +15.6%
- Shows negative correlation to equities during stress periods
- Experienced significant drawdown during value factor struggles (2018-2020)

### Alternative Considered but Rejected

**Pure Futures Carry Indices**: 
- Limited public access to bank indices
- No long-term historical data available
- Cannot invest directly

**Building Custom Carry Series**:
- Requires extensive futures data
- Complex implementation
- Results wouldn't be investable

### Implementation in Optimization

When using QSPIX as futures yield proxy:
1. Accept it represents a blend of strategies, not pure carry
2. Consider reducing allocation vs pure carry (since it includes other factors)
3. Monitor for overlap with other factor exposures
4. May need to adjust correlation assumptions

### Data Handling

```python
# When fetching returns for futures_yield exposure
if exposure_id == "futures_yield":
    # Use QSPIX data
    returns = fetch_fund_returns("QSPIX", start="2013-10-31")
    
    # Note: This is a multi-factor fund, not pure carry
    # Adjust expectations and allocations accordingly
```

## Other Adjustments Made

1. **Other Factors**: Now uses QRPRX instead of QSPIX to avoid double-counting
2. **Clear Documentation**: Added notes about the proxy relationship
3. **Start Date**: Explicitly marked as 2013-10-31 for QSPIX

This pragmatic approach allows us to include futures yield exposure with real, investable data while acknowledging the limitations of the proxy.
