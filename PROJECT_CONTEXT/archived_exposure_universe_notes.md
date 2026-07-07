# Exposure Universe Design Notes

## Your Exposure List - Refined

### ✅ Included as Requested
1. **US Large Cap Equity Beta** ✓
2. **US Small Cap Equity Beta** ✓  
3. **Developed Ex-US Large Cap Equity Beta** ✓
4. **Developed Ex-US Small Cap Equity Beta** ✓
5. **Emerging Markets Equity Beta** ✓
6. **Equity Factor Exposure** ✓ (using QMNIX as example)
7. **Other Factor Exposure** ✓ (using QSPIX as example)
8. **Trend Following** ✓ (kept separate from momentum)
9. **Short USTs** ✓
10. **Broad USTs** ✓
11. **Dynamic Global Bonds** ✓ (using PFIUX as example)
12. **TIPS** ✓
13. **Real Estate** ✓

### 🆕 Suggested Additions
1. **Futures Yield/Carry** - Added as separate exposure for RSSY component
2. **Commodities** - Broad commodity exposure (beyond just trend)
3. **Gold** - Often behaves differently than broad commodities

### 💡 Key Design Decisions

#### 1. Exposure Categorization
- **Equity Beta**: Traditional market beta exposures
- **Factors**: Systematic factor strategies (equity and non-equity)
- **Alternatives**: Trend, carry, and other non-traditional strategies
- **Fixed Income**: Various bond exposures
- **Real Assets**: Real estate, commodities, gold

#### 2. Bond Granularity Rationale
Your instinct is correct! Different Return Stacked funds use different bond strategies:
- **RSSB**: Uses broad intermediate treasuries
- **RSST**: Might use shorter duration for stability
- **RSSY**: Could use different duration profile
- **PIMCO funds**: Often use dynamic/unconstrained approach

Having Short UST, Broad UST, and Dynamic Global Bonds captures these differences.

#### 3. Futures Yield Strategy
Created a separate exposure for this because:
- It's primarily carry on futures (as you noted)
- Distinct from equity/bond carry in "Other Factors"
- Important component of RSSY
- May need special extraction methodology

#### 4. Factor Separation
Keeping "Equity Factors" and "Other Factors" separate makes sense:
- Different risk profiles
- Different correlation patterns
- Different implementation vehicles
- Clearer attribution

### 📊 Missing Exposures to Consider

Depending on your strategy evolution:
1. **Long-Duration Treasuries** - For barbell strategies
2. **Credit Spread** - IG corporate bonds vs treasuries
3. **High Yield** - Credit risk premium
4. **Currency Strategies** - If going beyond USD
5. **Volatility Premium** - Selling volatility

### 🔧 Implementation Notes

1. **Composite Calculations**: Some exposures average multiple ETFs for robustness
2. **Fund Extraction**: RSSY's futures yield needs regression-based extraction
3. **Index Fallbacks**: Where possible, include index data for longer history
4. **Dynamic Replication**: Some exposures may need time-varying weights

Would you like me to adjust any of these exposures or add any of the suggested missing ones?
