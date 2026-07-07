# START HERE - Exposure Universe Implementation

## Your Mission
Implement the Exposure Universe infrastructure to load exposure definitions and fetch historical returns data. The exposure universe configuration is complete at `/config/exposure_universe.yaml`.

## First Steps
1. **Run the test script**: `python test_data_availability.py`
   - This will show which tickers are available in yfinance
   - Mutual funds (ABYIX, AHLIX, etc.) may not be available

2. **Start with what works**:
   - Major ETFs (SPY, TLT, VNQ, etc.) are reliable in yfinance
   - Build the core system with these first
   - Add mutual funds if available, otherwise use fallbacks

3. **Key files to read**:
   - `/config/exposure_universe.yaml` - The complete configuration
   - `/PROJECT_CONTEXT/implementation_readiness.md` - Data source strategies
   - `/PROJECT_CONTEXT/TASKS/current_task.md` - Full requirements

## Critical Implementation Notes

### Data Sources
```python
# You'll need multiple data sources:
import yfinance as yf  # For most ETFs
import pandas_datareader as pdr  # For FRED data
# May need: tiingo, alpha_vantage for mutual funds
```

### Fallback Strategy
```python
# Example for trend following
if "ABYIX" not in available_tickers:
    # Use ETF fallback
    use_tickers = ["DBMF", "KMLM"]  # Shorter history but available
```

### Total Returns
- **Always use Adjusted Close** from yfinance
- This includes dividends and splits
- Raw Close prices are NOT total returns

### Risk-Free Rate
- Not available in yfinance
- Use FRED: `pdr.get_data_fred('DGS3MO', start, end)`

## Expected Deliverables
1. `ExposureUniverse` class that loads the YAML config
2. Enhanced `MarketDataFetcher` with total return support
3. Fallback logic for missing mutual funds
4. FRED integration for rates and inflation
5. Working example showing all exposures loaded

## Remember
- Start simple with ETFs that work
- Build in flexibility for data source issues
- Document which sources actually work
- Test with `test_data_availability.py` first!

Good luck! The configuration is ready - you just need to build the infrastructure to use it.
