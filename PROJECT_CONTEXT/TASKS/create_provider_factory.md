# Task: Create Data Provider Factory

**Status**: TODO  
**Priority**: MEDIUM  
**Estimated Time**: 1 day  
**Dependencies**: Core providers, caching, and quality layers complete

## Overview

Create a factory pattern that assembles the complete data provider stack with proper configuration management. This will be the main entry point for using the data layer in production.

## Implementation Plan

### 1. Provider Factory (`src/data/factory.py`)

```python
from typing import Optional, Dict, Union
from pathlib import Path
import os
import yaml

from .interfaces import DataProvider
from .providers import RawDataProviderCoordinator, TransformedDataProvider
from .cache import CacheManager, CacheConfig
from .quality import DefaultQualityMonitor, QualityConfig
from .config import DataLayerConfig

class DataProviderFactory:
    """Factory for creating configured data provider stacks."""
    
    @staticmethod
    def create_production_provider(
        config: Optional[Union[str, Path, Dict]] = None
    ) -> DataProvider:
        """
        Create production-ready provider with all layers.
        
        Args:
            config: Configuration (path to yaml, dict, or None for defaults)
            
        Returns:
            Fully configured DataProvider
        """
        # Load configuration
        if config is None:
            config = DataLayerConfig.from_environment()
        elif isinstance(config, (str, Path)):
            config = DataLayerConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = DataLayerConfig.from_dict(config)
        else:
            config = DataLayerConfig()
        
        # Build provider stack
        provider = DataProviderFactory._build_provider_stack(config)
        
        # Warm cache if configured
        if config.cache.warm_on_startup:
            DataProviderFactory._warm_cache(provider, config)
        
        return provider
    
    @staticmethod
    def create_test_provider(
        use_mocks: bool = True,
        config: Optional[Dict] = None
    ) -> DataProvider:
        """Create provider for testing (no external calls)."""
        if use_mocks:
            from tests.data.test_mock_providers import MockRawDataProvider
            raw = MockRawDataProvider()
        else:
            raw = RawDataProviderCoordinator()
        
        transformed = TransformedDataProvider(raw)
        
        # Minimal config for tests
        test_config = DataLayerConfig(
            cache_enabled=False,
            quality_enabled=False
        )
        if config:
            test_config.update(config)
        
        return transformed
    
    @staticmethod
    def _build_provider_stack(config: DataLayerConfig) -> DataProvider:
        """Build the complete provider stack based on configuration."""
        # Start with raw providers
        raw_provider = DataProviderFactory._create_raw_provider(config)
        
        # Add transformation layer
        provider = TransformedDataProvider(raw_provider)
        
        # Add caching if enabled
        if config.cache_enabled:
            cache_manager = DataProviderFactory._create_cache_manager(config.cache)
            provider = CachedDataProvider(provider, cache_manager, config.cache)
        
        # Add quality monitoring if enabled
        if config.quality_enabled:
            monitor = DataProviderFactory._create_quality_monitor(config.quality)
            provider = QualityAwareDataProvider(provider, monitor, config.quality)
        
        return provider
    
    @staticmethod
    def _create_raw_provider(config: DataLayerConfig) -> RawDataProvider:
        """Create and configure raw data providers."""
        # For now, just use the coordinator
        # In future, could configure individual providers
        return RawDataProviderCoordinator()
    
    @staticmethod
    def _create_cache_manager(config: CacheConfig) -> CacheManager:
        """Create configured cache manager."""
        return CacheManager(config)
    
    @staticmethod
    def _create_quality_monitor(config: QualityConfig) -> QualityMonitor:
        """Create configured quality monitor."""
        return DefaultQualityMonitor(config)
```

### 2. Configuration Management (`src/data/config.py`)

```python
@dataclass
class DataLayerConfig:
    """Complete configuration for data layer."""
    
    # Layer toggles
    cache_enabled: bool = True
    quality_enabled: bool = True
    
    # Sub-configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    calculations: CalculationsConfig = field(default_factory=CalculationsConfig)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'DataLayerConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataLayerConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update with provided values
        if 'cache_enabled' in data:
            config.cache_enabled = data['cache_enabled']
        if 'quality_enabled' in data:
            config.quality_enabled = data['quality_enabled']
        
        # Sub-configurations
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'quality' in data:
            config.quality = QualityConfig(**data['quality'])
        if 'sources' in data:
            config.sources = SourcesConfig(**data['sources'])
        if 'calculations' in data:
            config.calculations = CalculationsConfig(**data['calculations'])
        
        return config
    
    @classmethod
    def from_environment(cls) -> 'DataLayerConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Check for config file in environment
        if config_path := os.environ.get('PORTFOLIO_CONFIG_PATH'):
            return cls.from_yaml(config_path)
        
        # Individual overrides
        if cache_enabled := os.environ.get('PORTFOLIO_CACHE_ENABLED'):
            config.cache_enabled = cache_enabled.lower() == 'true'
        
        if cache_dir := os.environ.get('PORTFOLIO_CACHE_DIR'):
            config.cache.cache_dir = Path(cache_dir)
        
        # API keys
        if fred_key := os.environ.get('FRED_API_KEY'):
            config.sources.fred_api_key = fred_key
        
        return config

@dataclass
class SourcesConfig:
    """Configuration for data sources."""
    # YFinance settings
    yfinance_session_timeout: int = 10
    yfinance_max_retries: int = 3
    
    # FRED settings
    fred_api_key: Optional[str] = None
    fred_use_fallback: bool = True
    
    # CSV settings
    csv_enabled: bool = False
    csv_directory: Path = Path("./data/manual")

@dataclass  
class CalculationsConfig:
    """Configuration for calculations."""
    # Return calculations
    use_adjusted_close: bool = True
    dividend_reinvestment: bool = True
    
    # Inflation calculations
    default_inflation_index: str = "cpi"  # or "pce"
    default_inflation_method: str = "yoy"  # or "mom_annualized"
    
    # Risk-free rate selection
    default_risk_free_tenor: str = "3m"
    risk_free_fallback_chain: Dict[str, List[str]] = field(default_factory=lambda: {
        "3m": ["TREASURY_3M", "FED_FUNDS"],
        "1y": ["TREASURY_1Y", "TREASURY_6M"],
    })
```

### 3. High-Level Client (`src/data/client.py`)

```python
class DataClient:
    """High-level client for easy data access."""
    
    def __init__(self, config: Optional[Union[str, Path, Dict]] = None):
        """Initialize with optional configuration."""
        self.provider = DataProviderFactory.create_production_provider(config)
    
    # Convenience methods for common use cases
    def get_stock_returns(
        self,
        ticker: str,
        start: date,
        end: date,
        frequency: str = "daily",
        total: bool = True
    ) -> pd.Series:
        """Get stock returns with sensible defaults."""
        data_type = LogicalDataType.TOTAL_RETURN if total else LogicalDataType.SIMPLE_RETURN
        return self.provider.get_data(data_type, start, end, ticker, frequency)
    
    def get_portfolio_returns(
        self,
        tickers: List[str],
        start: date,
        end: date,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """Get returns for multiple tickers as DataFrame."""
        returns = {}
        for ticker in tickers:
            try:
                returns[ticker] = self.get_stock_returns(ticker, start, end, frequency)
            except DataNotAvailableError:
                logger.warning(f"No data available for {ticker}")
        
        return pd.DataFrame(returns)
    
    def get_risk_free_rate(
        self,
        start: date,
        end: date,
        tenor: str = "3m",
        real: bool = False,
        frequency: str = "daily"
    ) -> pd.Series:
        """Get risk-free rate with automatic selection."""
        data_type = (LogicalDataType.REAL_RISK_FREE if real 
                    else LogicalDataType.NOMINAL_RISK_FREE)
        return self.provider.get_data(
            data_type, start, end, frequency=frequency, tenor=tenor
        )
    
    def get_market_data(
        self,
        ticker: str,
        start: date,
        end: date,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """Get multiple metrics for a ticker."""
        if metrics is None:
            metrics = ["adjusted_close", "volume", "total_return"]
        
        data = {}
        for metric in metrics:
            try:
                # Map string to data type
                data_type = self._get_data_type(metric)
                data[metric] = self.provider.get_data(
                    data_type, start, end, ticker
                )
            except Exception as e:
                logger.warning(f"Failed to get {metric} for {ticker}: {e}")
        
        return pd.DataFrame(data)
```

### 4. Example Configurations

**Default Production Config (`config/production.yaml`):**
```yaml
# Data layer production configuration
cache_enabled: true
quality_enabled: true

cache:
  cache_directory: ~/.portfolio_optimizer/cache
  memory_size_mb: 500
  ttl:
    recent_data: 300      # 5 minutes
    standard_data: 3600   # 1 hour
    historical_data: 86400 # 1 day
  warm_on_startup: true
  warm_tickers: ["SPY", "AGG", "GLD", "VNQ", "VXUS"]

quality:
  auto_fix: true
  min_quality_score: 80.0
  checks:
    missing_data:
      enabled: true
      max_gap_days: 5
    extreme_values:
      enabled: true
      return_threshold: 0.25

sources:
  fred_api_key: ${FRED_API_KEY}
  fred_use_fallback: true

calculations:
  use_adjusted_close: true
  default_inflation_index: cpi
  default_risk_free_tenor: 3m
```

**Development Config (`config/development.yaml`):**
```yaml
# Faster iteration for development
cache_enabled: true
quality_enabled: false  # Skip quality checks for speed

cache:
  cache_directory: ./cache_dev
  memory_size_mb: 100
  warm_on_startup: false

sources:
  fred_use_fallback: true  # Always use fallback in dev
```

### 5. Usage Examples

**Basic Usage:**
```python
from datetime import date
from portfolio_optimizer.data import DataClient

# Use default configuration
client = DataClient()

# Get Apple returns
returns = client.get_stock_returns("AAPL", date(2023, 1, 1), date(2023, 12, 31))

# Get portfolio data
portfolio = ["SPY", "AGG", "GLD"]
portfolio_returns = client.get_portfolio_returns(
    portfolio, 
    date(2023, 1, 1), 
    date(2023, 12, 31)
)
```

**Custom Configuration:**
```python
# Use custom config file
client = DataClient("config/my_config.yaml")

# Or use dict
client = DataClient({
    "cache_enabled": False,
    "quality": {
        "min_quality_score": 90.0
    }
})

# Or use environment
os.environ['PORTFOLIO_CONFIG_PATH'] = "config/production.yaml"
client = DataClient()  # Loads from environment
```

**Direct Provider Access:**
```python
from portfolio_optimizer.data import DataProviderFactory

# For advanced usage
provider = DataProviderFactory.create_production_provider()

# Get any data type
inflation = provider.get_data(
    LogicalDataType.INFLATION_RATE,
    date(2023, 1, 1),
    date(2023, 12, 31)
)
```

## Testing

### Factory Tests
```python
def test_factory_creates_correct_stack():
    """Test that factory builds provider stack correctly."""
    config = {
        "cache_enabled": True,
        "quality_enabled": True
    }
    
    provider = DataProviderFactory.create_production_provider(config)
    
    # Should have quality -> cache -> transformed -> raw
    assert isinstance(provider, QualityAwareDataProvider)
    assert isinstance(provider.provider, CachedDataProvider)
```

### Configuration Tests
```python
def test_config_from_yaml():
    """Test loading configuration from YAML."""
    config = DataLayerConfig.from_yaml("tests/fixtures/test_config.yaml")
    assert config.cache_enabled == True
    assert config.cache.memory_size_mb == 100
```

## Success Criteria

- [ ] Factory assembles complete provider stack
- [ ] Configuration from YAML, dict, and environment
- [ ] High-level client with convenience methods
- [ ] Example configurations for common scenarios
- [ ] Environment variable overrides
- [ ] Comprehensive documentation
- [ ] Unit tests for all components
- [ ] Usage examples

## Next Steps

After factory implementation:
1. Create production deployment guide
2. Add monitoring and metrics
3. Integrate with portfolio optimization
4. Build web API on top

The factory completes the data layer, making it easy to use in production!
