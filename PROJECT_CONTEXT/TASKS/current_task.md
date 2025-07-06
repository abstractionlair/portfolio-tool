# Current Task: Exposure Universe and Returns Infrastructure

**Status**: READY TO START  
**Priority**: Critical - Foundation for everything else  
**Type**: Infrastructure Milestone

## 🚀 Claude Code: Start Here
See: `/PROJECT_CONTEXT/START_HERE_CLAUDE_CODE.md`

## Overview
Before we can properly use the optimization engine, we need a robust data infrastructure that:
1. Defines our exposure universe (what risks/returns we want to capture)
2. Maps each exposure to reliable data sources
3. Ensures we get total returns (including dividends)
4. Provides inflation data for real return calculations
5. Estimates expected returns and covariances properly

## Detailed Specification
See: `/PROJECT_CONTEXT/TASKS/exposure_universe_infrastructure.md`

## Exposure Universe Definition
See: `/config/exposure_universe.yaml`

## Quick Implementation Path
See: `/PROJECT_CONTEXT/TASKS/exposure_universe_roadmap.md`

## Implementation Readiness
See: `/PROJECT_CONTEXT/implementation_readiness.md`

## Data Availability Test Script
See: `/test_data_availability.py` - Run this first to check which tickers work

## Why This Matters
The optimization engine is only as good as its inputs. We need:
- **Total returns** not just price returns
- **Real returns** to handle inflation properly  
- **Consistent data** across all exposures
- **Proper handling** of exposures with different histories
- **Flexible system** to add new exposures easily

## Success Criteria
- [ ] Exposure universe configuration system implemented
- [ ] Total returns properly calculated for all exposures
- [ ] Inflation data integrated
- [ ] Real geometric returns computed correctly
- [ ] System works with existing optimization engine
- [ ] Can easily add new exposures
- [ ] **Fallback data sources implemented for missing mutual funds**
- [ ] **FRED integration working for risk-free rate and CPI**
- [ ] **Cash/Risk-Free Rate (DGS3MO) properly fetched and converted to returns**

## Implementation Guides Created
1. **TotalReturnFetcher Implementation Plan**: `/PROJECT_CONTEXT/TASKS/total_return_fetcher_implementation.md`
   - Complete design for handling price series, rate series, and composites
   - Detailed implementation structure with code examples
   
2. **Cash/Risk-Free Rate Implementation Guide**: `/PROJECT_CONTEXT/TASKS/cash_rate_implementation_guide.md`
   - Specific solution for the reported DGS3MO issue
   - Shows how to integrate existing FRED fetcher
   - Includes rate-to-return conversion logic

## Key Decisions Made
- **Hierarchical Organization**: 5 clear categories (16 exposures total)
  - Equity Beta (5), Factor/Style (2), Alternatives (1)
  - Nominal Fixed Income (4) - includes Cash/Risk-Free Rate
  - Real Assets (4) - includes TIPS as inflation hedge
- **Cash/Risk-Free Rate Added**: Essential for leverage cost modeling
  - See `/PROJECT_CONTEXT/leverage_cost_modeling.md` for rationale
- **History Prioritization**: Use funds with longest available history
  - Trend following now uses mutual funds (10+ years) not ETFs (5 years)
  - See `/PROJECT_CONTEXT/history_optimization_notes.md`

## Key Exposures to Include
- US Large/Small Cap Equity Beta  
- International Developed/EM Equity Beta
- Factor/Style - Equities (QMNIX or factor ETF composite)
- Factor/Style - Other (QSPIX - includes carry/futures yield)
- Trend Following
- Cash/Risk-Free Rate (leverage cost modeling)
- Various Fixed Income Exposures
- Real Assets (RE, Commodities, Gold, TIPS)

## Next Steps After Completion
1. Portfolio visualization tools
2. Web API/interface
3. Backtesting framework
4. Tax-aware features
