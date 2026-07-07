# Asset Universe Infrastructure - Implementation Roadmap

## Quick Start Path

### Step 1: Basic Asset Universe Loader (1-2 hours)
```python
# Create src/data/asset_universe.py
class AssetClass:
    """Represents a single asset class/strategy."""
    
class AssetUniverse:
    """Manages the collection of asset classes."""
    @classmethod
    def from_yaml(cls, filepath: str) -> 'AssetUniverse':
        """Load universe from YAML configuration."""
```

### Step 2: Enhance MarketDataFetcher for Total Returns (2-3 hours)
- Add dividend-adjusted returns
- Verify we're using 'Adj Close' from yfinance
- Add composite/average return calculations
- Add basic data validation

### Step 3: Add Inflation Data (1-2 hours)
- Use `pandas_datareader` for FRED access
- Cache inflation data locally
- Add real return conversion utilities

### Step 4: Return Estimation Module (2-3 hours)
- Arithmetic to geometric conversion
- Basic historical estimation
- Covariance matrix with Ledoit-Wolf shrinkage

### Step 5: Integration Example (1 hour)
- Load universe
- Fetch all returns
- Convert to real returns
- Estimate parameters
- Feed to optimizer

## Minimal First Implementation

Start with just 5 asset classes to prove the concept:
1. US Large Cap Equity (SPY)
2. US Bonds (AGG)
3. International Equity (EFA)
4. Trend Following (DBMF)
5. Commodities (DJP)

## Key Technical Decisions

1. **Total Returns**: yfinance 'Adj Close' includes dividends
2. **Frequency**: Daily data, converted as needed
3. **Real Returns**: Monthly CPI data, interpolated to daily
4. **Missing Data**: Forward fill up to 5 days, then drop
5. **Caching**: Simple file-based cache with 24hr TTL

## Next Actions

1. Review and refine the asset universe YAML
2. Implement basic AssetUniverse class
3. Extend data fetcher for total returns
4. Add inflation integration
5. Build return estimation framework

## Questions to Resolve

1. Should we add currency-hedged versions of international assets?
2. How granular on equity factors? (Single composite vs individual factors)
3. Include crypto as an asset class?
4. How to handle assets with limited history (e.g., DBMF from 2019)?
5. Benchmark indices vs investable ETFs only?
