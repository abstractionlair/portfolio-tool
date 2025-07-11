# Task: Create Data Layer Integration

**Status**: TODO  
**Priority**: MEDIUM  
**Estimated Time**: 1 day  
**Dependencies**: Transformed provider complete

## Overview

Once the transformed provider is complete, we need to create the integration layer that brings everything together for production use. This includes factory patterns, configuration management, and convenience wrappers.

## Components to Implement

### 1. Provider Factory (`src/data/factory.py`)

Create a factory that builds the complete provider stack:

```python
class DataProviderFactory:
    """Factory for creating configured data provider stacks."""
    
    @staticmethod
    def create_production_provider(config: Optional[Dict] = None) -> DataProvider:
        """Create production-ready provider with all layers."""
        # Stack: Raw → Transformed → Cached → Quality
        
    @staticmethod
    def create_test_provider() -> DataProvider:
        """Create provider with mocked sources for testing."""
        
    @staticmethod
    def create_from_config(config_path: str) -> DataProvider:
        """Create provider from YAML configuration file."""
```

### 2. Configuration Management (`src/data/config.py`)

Handle configuration for all data sources:

```python
@dataclass
class DataLayerConfig:
    """Configuration for the data layer."""
    
    # Source configurations
    yfinance_config: YFinanceConfig
    fred_config: FREDConfig
    csv_config: Optional[CSVConfig] = None
    
    # Layer configurations
    cache_enabled: bool = True
    cache_config: Optional[CacheConfig] = None
    quality_enabled: bool = True
    quality_config: Optional[QualityConfig] = None
    
    # Calculation configurations
    calculation_config: CalculationConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'DataLayerConfig':
        """Load configuration from YAML file."""
        
    @classmethod
    def from_env(cls) -> 'DataLayerConfig':
        """Load configuration from environment variables."""
```

### 3. Convenience Wrapper (`src/data/client.py`)

High-level client for easy data access:

```python
class DataClient:
    """High-level client for data access."""
    
    def __init__(self, provider: Optional[DataProvider] = None):
        self.provider = provider or DataProviderFactory.create_production_provider()
    
    def get_stock_returns(
        self, ticker: str, start: date, end: date, 
        frequency: str = "daily", total: bool = True
    ) -> pd.Series:
        """Get stock returns with sensible defaults."""
        
    def get_risk_free_rate(
        self, start: date, end: date,
        tenor: str = "3m", real: bool = False
    ) -> pd.Series:
        """Get risk-free rate with automatic selection."""
        
    def get_inflation(
        self, start: date, end: date,
        index: str = "cpi", frequency: str = "monthly"
    ) -> pd.Series:
        """Get inflation rate."""
        
    def get_market_data(
        self, tickers: List[str], start: date, end: date,
        metrics: List[str] = ["returns", "volume"]
    ) -> pd.DataFrame:
        """Get multiple metrics for multiple tickers."""
```

### 4. CSV Source Implementation (`src/data/providers/csv_provider.py`)

Implement the CSV fallback source:

```python
class CSVProvider(RawDataProvider):
    """CSV file-based data provider for manual/fallback data."""
    
    def __init__(self, data_dir: str = "./data/manual"):
        self.data_dir = Path(data_dir)
        self._index_cache = {}  # Cache file listings
        
    def get_data(self, data_type, start, end, ticker=None, **kwargs):
        """Load data from CSV files."""
        # File naming convention: {ticker}_{data_type}.csv
        # or {data_type}.csv for economic data
```

### 5. Example Configurations

Create example configuration files:

**`config/data_layer.yaml`**:
```yaml
# Data source configurations
sources:
  yfinance:
    session_timeout: 10
    max_retries: 3
    rate_limit: 2000/hour
    
  fred:
    api_key: ${FRED_API_KEY}
    rate_limit: 120/minute
    fallback_enabled: true
    
  csv:
    enabled: true
    data_directory: ./data/manual
    date_column: Date
    date_format: "%Y-%m-%d"

# Caching configuration    
cache:
  enabled: true
  backend: sqlite  # or redis, memory
  location: ~/.portfolio_optimizer/cache
  ttl:
    recent: 300      # 5 minutes for recent data
    standard: 86400  # 1 day for standard data
    historical: 604800  # 1 week for old data

# Quality monitoring
quality:
  enabled: true
  min_quality_score: 80.0
  auto_fix: true
  checks:
    extreme_returns:
      enabled: true
      threshold: 0.25  # 25% daily move
    missing_data:
      enabled: true
      max_gap_days: 5
    stale_prices:
      enabled: true
      max_unchanged_days: 5

# Calculation settings
calculations:
  returns:
    use_adjusted_close: true
    compound_dividends: true
  inflation:
    default_index: cpi
    default_method: yoy
  risk_free:
    default_tenor: 3m
```

### 6. Integration Examples

Create example scripts showing usage:

**`examples/data_layer_usage.py`**:
```python
from datetime import date
from src.data import DataClient

# Simple usage
client = DataClient()

# Get Apple returns
returns = client.get_stock_returns(
    "AAPL", 
    date(2023, 1, 1), 
    date(2023, 12, 31)
)

# Get risk-free rate
rf_rate = client.get_risk_free_rate(
    date(2023, 1, 1),
    date(2023, 12, 31),
    tenor="3m"
)

# Get multiple metrics for portfolio
portfolio_data = client.get_market_data(
    ["AAPL", "MSFT", "GOOGL"],
    date(2023, 1, 1),
    date(2023, 12, 31),
    metrics=["total_returns", "volume", "volatility"]
)
```

## Testing Requirements

### Factory Tests
- Test creation with different configurations
- Test fallback behavior
- Test error handling

### Configuration Tests
- Test loading from YAML
- Test environment variable override
- Test validation

### Integration Tests
- End-to-end data fetching
- Performance benchmarks
- Multi-threading safety

## Success Criteria

- [ ] Factory creates properly configured provider stacks
- [ ] Configuration management from files and environment
- [ ] CSV provider implements fallback functionality
- [ ] Client provides intuitive high-level interface
- [ ] Example configurations work out of the box
- [ ] Documentation complete with examples
- [ ] Performance meets targets (<2s for typical requests)

## Next Steps

After integration layer:
1. Add cache and quality layers (already designed in interfaces)
2. Performance optimization (parallel fetching, smart caching)
3. Production deployment guide
4. Monitoring and alerting setup

This completes the data layer implementation!
