# Task: Build Asset Universe and Returns Infrastructure

**Status**: PLANNING  
**Priority**: Critical  
**Type**: Infrastructure Milestone

## Objective
Create a comprehensive system for defining asset classes/strategies, mapping them to data sources, retrieving total returns inclusive of all distributions, and converting to real geometric returns for use in optimization.

## Components

### 1. Asset Universe Definition System

Create a structured way to define and persist asset classes and strategies:

```yaml
# src/data/asset_universe.yaml
asset_classes:
  - id: us_large_equity
    name: "US Large Cap Equity Beta"
    description: "Broad exposure to US large cap stocks"
    data_sources:
      - type: "etf"
        tickers: ["SPY", "IVV", "VOO"]
        weighting: "equal"  # or "market_cap" or custom weights
    benchmark: "SPY"
    
  - id: us_small_equity  
    name: "US Small Cap Equity Beta"
    description: "Broad exposure to US small cap stocks"
    data_sources:
      - type: "etf"
        tickers: ["IWM", "IJR"]
        weighting: "equal"
        
  - id: international_equity
    name: "International Developed Equity"
    description: "Developed markets ex-US equity exposure"
    data_sources:
      - type: "etf"
        tickers: ["EFA", "IEFA", "VEA"]
        weighting: "equal"
        
  - id: emerging_equity
    name: "Emerging Markets Equity"
    description: "Emerging markets equity exposure"
    data_sources:
      - type: "etf"
        tickers: ["EEM", "IEMG", "VWO"]
        weighting: "equal"
        
  - id: us_bonds
    name: "US Aggregate Bonds"
    description: "Broad US investment grade bond exposure"
    data_sources:
      - type: "etf"
        tickers: ["AGG", "BND"]
        weighting: "equal"
        
  - id: trend_following
    name: "Trend Following Strategy"
    description: "Managed futures trend following"
    data_sources:
      - type: "etf"
        tickers: ["DBMF", "KMLM", "CTA"]
        weighting: "equal"
      - type: "index"
        id: "SG_TREND_INDEX"  # If we can get it
        
  - id: equity_factors
    name: "Equity Factor Exposure"
    description: "Diversified equity factor premia"
    data_sources:
      - type: "composite"
        components:
          - ticker: "VMOT"  # Momentum
            weight: 0.25
          - ticker: "VLUE"  # Value
            weight: 0.25
          - ticker: "QUAL"  # Quality
            weight: 0.25
          - ticker: "USMV"  # Low Vol
            weight: 0.25
            
  - id: real_assets
    name: "Real Assets"
    description: "Commodities and real estate"
    data_sources:
      - type: "composite"
        components:
          - ticker: "DJP"   # Commodities
            weight: 0.5
          - ticker: "VNQ"   # REITs
            weight: 0.5
```

### 2. Total Return Data Collection

Implement robust total return calculation:

```python
class TotalReturnFetcher:
    """Ensures we get total returns including dividends and distributions."""
    
    def __init__(self, data_source: str = "yfinance"):
        self.data_source = data_source
        
    def fetch_total_returns(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily"
    ) -> pd.Series:
        """
        Fetch total returns including:
        - Price appreciation
        - Dividends/distributions
        - Stock splits
        - Other corporate actions
        """
        # Implementation should:
        # 1. Get adjusted close prices (handles splits)
        # 2. Get dividend data
        # 3. Calculate total returns properly
        # 4. Validate data quality
        
    def fetch_composite_returns(
        self,
        components: List[Dict[str, float]],
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """Calculate returns for a weighted composite."""
```

### 3. Inflation Data Integration

Add inflation data for real return calculations:

```python
class InflationDataFetcher:
    """Fetch inflation data from FRED or other sources."""
    
    def fetch_cpi_data(
        self,
        start_date: datetime,
        end_date: datetime,
        series: str = "CPIAUCSL"  # CPI-U All Items
    ) -> pd.Series:
        """Fetch CPI data from FRED."""
        
    def calculate_inflation_rate(
        self,
        cpi_series: pd.Series,
        frequency: str = "monthly"
    ) -> pd.Series:
        """Convert CPI levels to inflation rates."""
        
    def convert_to_real_returns(
        self,
        nominal_returns: pd.Series,
        inflation_rates: pd.Series
    ) -> pd.Series:
        """Convert nominal returns to real returns."""
```

### 4. Geometric Return and Risk Estimation

Build proper return estimation infrastructure:

```python
class GeometricReturnEstimator:
    """Estimate geometric returns and covariances."""
    
    def calculate_geometric_returns(
        self,
        arithmetic_returns: pd.Series,
        frequency: str = "daily"
    ) -> float:
        """
        Convert arithmetic to geometric returns.
        Handles compounding effects properly.
        """
        
    def estimate_real_returns(
        self,
        asset_universe: AssetUniverse,
        lookback_years: int = 10,
        method: str = "historical"
    ) -> pd.DataFrame:
        """
        Estimate expected real geometric returns.
        
        Methods:
        - historical: Simple historical average
        - shrinkage: Shrink to common mean
        - capm_adjusted: Use CAPM framework
        - regime_aware: Adjust for current regime
        """
        
    def estimate_covariance_matrix(
        self,
        returns: pd.DataFrame,
        method: str = "sample",
        frequency: str = "daily"
    ) -> np.ndarray:
        """
        Estimate covariance matrix with various methods.
        Should handle:
        - Different return frequencies
        - Missing data
        - Shrinkage methods
        - Factor models
        """
```

### 5. Data Quality and Validation

Ensure data integrity:

```python
class DataValidator:
    """Validate and clean financial data."""
    
    def validate_returns(self, returns: pd.Series) -> ValidationResult:
        """
        Check for:
        - Missing data
        - Outliers (e.g., >50% daily moves)
        - Data gaps
        - Suspicious patterns
        """
        
    def validate_total_returns(
        self,
        price_returns: pd.Series,
        total_returns: pd.Series
    ) -> bool:
        """Verify that total returns properly include distributions."""
```

## Implementation Plan

### Phase 1: Asset Universe Definition
1. Create YAML schema for asset universe
2. Build AssetUniverse class to load and manage definitions
3. Add validation for universe definitions
4. Create example universe with your target assets

### Phase 2: Enhanced Data Collection
1. Extend MarketDataFetcher for total returns
2. Add dividend/distribution handling
3. Implement composite return calculations
4. Add data validation and quality checks

### Phase 3: Inflation Integration
1. Add FRED API integration (or use pandas-datareader)
2. Build inflation data fetcher
3. Implement real return conversion
4. Cache inflation data appropriately

### Phase 4: Return Estimation Framework
1. Implement geometric return calculations
2. Build various estimation methods
3. Add covariance estimation with shrinkage
4. Create comprehensive testing suite

### Phase 5: Integration and Examples
1. Connect to existing optimization engine
2. Create example showing full workflow
3. Document best practices
4. Performance optimization

## Key Design Decisions

1. **Total Returns**: Always use adjusted close prices + dividends
2. **Data Sources**: Start with yfinance, add fallbacks later
3. **Frequency**: Support daily/monthly/annual conversions
4. **Real Returns**: Use CPI-U as default inflation measure
5. **Composites**: Allow weighted averages of multiple securities
6. **Caching**: Cache all external API calls with TTL

## Success Criteria

- [ ] Asset universe defined in configuration, not code
- [ ] Total returns properly include all distributions
- [ ] Inflation data integrated and cached
- [ ] Real geometric returns calculated correctly
- [ ] Covariance matrices estimated with multiple methods
- [ ] Data quality validation in place
- [ ] Comprehensive test coverage
- [ ] Example demonstrating full workflow

## Example Usage

```python
# Load asset universe
universe = AssetUniverse.from_yaml("asset_universe.yaml")

# Fetch total returns for all assets
returns_fetcher = TotalReturnFetcher()
returns_data = {}
for asset in universe.assets:
    returns_data[asset.id] = returns_fetcher.fetch_returns_for_asset(
        asset,
        start_date=datetime(2010, 1, 1),
        end_date=datetime.now()
    )

# Get inflation data
inflation_fetcher = InflationDataFetcher()
cpi_data = inflation_fetcher.fetch_cpi_data(start_date, end_date)
inflation_rates = inflation_fetcher.calculate_inflation_rate(cpi_data)

# Convert to real returns
real_returns = {}
for asset_id, nominal_returns in returns_data.items():
    real_returns[asset_id] = inflation_fetcher.convert_to_real_returns(
        nominal_returns,
        inflation_rates
    )

# Estimate expected returns and covariances
estimator = GeometricReturnEstimator()
expected_returns = estimator.estimate_real_returns(
    universe,
    lookback_years=10,
    method="shrinkage"
)
cov_matrix = estimator.estimate_covariance_matrix(
    pd.DataFrame(real_returns),
    method="ledoit_wolf"
)

# Ready for optimization!
engine = OptimizationEngine(analytics, fund_map)
result = engine.optimize_mean_variance(
    symbols=list(universe.assets),
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    constraints=constraints
)
```

## Notes

- Consider adding alternative data sources (Tiingo, Polygon.io)
- May want to add currency hedging considerations for international
- Factor exposures might need more granular decomposition later
- Consider adding inflation expectations vs realized inflation
- Think about regime detection for return estimation
