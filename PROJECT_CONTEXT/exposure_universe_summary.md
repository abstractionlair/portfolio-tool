# Exposure Universe - Ready for Implementation

## Final Exposure List (14 Total)

### Equity Beta (5)
1. **US Large Cap Equity Beta** - SPY/IVV/VOO
2. **US Small Cap Equity Beta** - IWM/IJR/VB
3. **Developed Ex-US Large Cap Equity Beta** - EFA/IEFA/VEA
4. **Developed Ex-US Small Cap Equity Beta** - SCZ/IEUS/VSS
5. **Emerging Markets Equity Beta** - EEM/IEMG/VWO

### Factor Exposures (2)
6. **Equity Factor Exposure** - QMNIX or composite of MTUM/VLUE/QUAL/USMV
7. **Alternative Risk Premia** - QRPRX (avoiding QSPIX overlap)

### Alternative Strategies (2)
8. **Trend Following** - DBMF/KMLM/RSST average
9. **Futures Yield/Carry** - QSPIX (multi-factor proxy including carry)

### Fixed Income (4)
10. **Short-Term US Treasuries** - SHY/SCHO/VGSH
11. **Broad US Treasuries** - IEF/IEI/GOVT
12. **Dynamic Global Bonds** - PFIUX or BNDX/EMB composite
13. **TIPS** - TIP/SCHP/VTIP

### Real Assets (1)
14. **Real Estate** - VNQ/XLRE/RWR/REET

## Implementation Approach

### Phase 1: Core Infrastructure
1. Create `ExposureUniverse` class to load YAML configuration
2. Enhance `MarketDataFetcher` for total returns (use Adj Close)
3. Add composite calculation support (weighted averages)
4. Implement basic data validation

### Phase 2: Data Collection
1. Add FRED integration for CPI data
2. Build inflation adjustment utilities
3. Create caching layer for external API calls
4. Handle different data start dates gracefully

### Phase 3: Return Estimation
1. Implement arithmetic to geometric return conversion
2. Build covariance estimation with shrinkage
3. Create parameter estimation framework
4. Connect to existing optimization engine

## Data Availability Notes

- **Most ETFs**: 10+ years of history available
- **QMNIX**: Limited access, use ETF composite as fallback
- **QSPIX**: Data from October 2013
- **DBMF**: Only from 2019 (handle short history)
- **PFIUX**: May need institutional access

## Ready to Implement

The exposure universe is now:
- ✅ Well-defined with clear categories
- ✅ Implementable with available data
- ✅ Pragmatic about data limitations (QSPIX for carry)
- ✅ Comprehensive enough for sophisticated optimization
- ✅ Extensible for future additions

Next step: Hand off to Claude Code for implementation!
