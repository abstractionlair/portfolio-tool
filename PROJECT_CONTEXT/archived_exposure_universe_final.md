# Exposure Universe - Final Structure

## Overview
The exposure universe is organized into 15 distinct exposures across 5 categories. Futures yield/carry is captured within "Factor/Style - Other" rather than as a standalone exposure.

## Complete Exposure List

### 1. Equity Beta (5 exposures)
- **US Large Cap Equity Beta** - Core US large cap exposure
- **US Small Cap Equity Beta** - US small cap risk premium
- **Developed Ex-US Large Cap Equity Beta** - International developed large caps
- **Developed Ex-US Small Cap Equity Beta** - International developed small caps
- **Emerging Markets Equity Beta** - Emerging markets exposure

### 2. Factor/Style (2 exposures)
- **Factor/Style - Equities** - Equity-only factor strategies (value, momentum, quality, low vol)
  - Example: QMNIX (AQR Equity Market Neutral)
  
- **Factor/Style - Other** - Multi-asset factor strategies across bonds, currencies, commodities
  - Example: QSPIX (AQR Style Premia Alternative)
  - Includes: Value, momentum, carry (including futures yield), and defensive factors

### 3. Alternative Strategies (1 exposure)
- **Trend Following** - Managed futures trend following
  - Kept separate from factors as it's conceptually distinct
  - Uses DBMF, KMLM average

### 4. Fixed Income (4 exposures)
- **Short-Term US Treasuries** - 1-3 year Treasury exposure
- **Broad US Treasuries** - Intermediate Treasury exposure
- **Dynamic Global Bonds** - Active global fixed income (PFIUX example)
- **TIPS** - Inflation-protected securities

### 5. Real Assets (3 exposures)
- **Real Estate** - REIT exposure
- **Commodities** - Broad commodity basket
- **Gold** - Precious metals exposure

## Key Design Decisions

### Factor/Style Split Rationale
1. **Equity Factors** remain equity-focused for clarity
2. **Other Factors** capture cross-asset strategies including:
   - Currency carry
   - Commodity carry (including futures)
   - Bond carry
   - Cross-asset value and momentum

### Why This Structure Works
- **Clean Categories**: Clear distinction between equity and multi-asset factors
- **Data Available**: QSPIX provides 11+ years of history for multi-asset factors
- **No Redundancy**: Avoids double-counting carry strategies
- **Implementation Ready**: All exposures have identifiable data sources

### QSPIX as Multi-Asset Factor Proxy
- Launched: October 31, 2013
- Includes four systematic styles: value, momentum, carry, defensive
- Applied across: equities, bonds, currencies, commodities, interest rates
- Targets 10% volatility with leverage
- Provides exposure to futures yield through its carry component

## Data Implementation Notes

```yaml
# Factor/Style - Other implementation
- Uses QSPIX which includes:
  * ~25% carry strategies (including futures)
  * ~25% value strategies
  * ~25% momentum strategies  
  * ~25% defensive strategies
- Across multiple asset classes
```

## Summary
This structure provides a comprehensive exposure universe that:
- ✅ Captures all major risk premia
- ✅ Has available historical data
- ✅ Avoids conceptual overlaps
- ✅ Maps cleanly to investable instruments
- ✅ Includes futures yield within multi-asset factors
