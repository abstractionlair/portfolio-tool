# Task: Implement Caching Layer

**Status**: TODO  
**Priority**: HIGH  
**Estimated Time**: 2 days  
**Dependencies**: Integration tests complete

## Overview

After validating the data layer with real market data, we need to implement a caching layer to improve performance and reduce API calls. This will make the system more responsive and prevent rate limiting issues.

## Implementation Plan

### 1. Cache Infrastructure (`src/data/cache/`)

#### 1.1 Cache Backend Interface
```python
# src/data/cache/backends/base.py
from abc import ABC, abstractmethod

class CacheBackend(ABC):
    """Abstract interface for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store item in cache with optional TTL in seconds."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove item from cache."""
        pass
    
    @abstractmethod
    def clear(self, pattern: Optional[str] = None):
        """Clear cache, optionally by pattern."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
```

#### 1.2 SQLite Cache Backend
```python
# src/data/cache/backends/sqlite_backend.py
import sqlite3
import pickle
from pathlib import Path

class SQLiteCache(CacheBackend):
    """SQLite-based persistent cache."""
    
    def __init__(self, db_path: str = "~/.portfolio_optimizer/cache.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database schema."""
        # Schema:
        # - key (TEXT PRIMARY KEY)
        # - value (BLOB) - pickled pandas objects
        # - created_at (TIMESTAMP)
        # - expires_at (TIMESTAMP)
        # - metadata (JSON) - series info, data type, etc.
```

#### 1.3 Memory Cache Backend
```python
# src/data/cache/backends/memory_backend.py
from collections import OrderedDict
import sys

class MemoryCache(CacheBackend):
    """In-memory LRU cache with size limits."""
    
    def __init__(self, max_size_mb: int = 500):
        self.max_size = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.size_tracker = {}
```

### 2. Cache Manager (`src/data/cache/manager.py`)

```python
class CacheManager:
    """Manages multi-tier caching with intelligent key generation."""
    
    def __init__(self, config: CacheConfig):
        self.l1_cache = MemoryCache(config.memory_size_mb)
        self.l2_cache = SQLiteCache(config.cache_dir)
        self.config = config
        self.stats = CacheStatistics()
    
    def get_market_data(
        self,
        data_type: DataType,
        ticker: str,
        start: date,
        end: date,
        frequency: str
    ) -> Optional[pd.Series]:
        """Get market data from cache with smart key generation."""
        # Generate cache key with date rounding
        cache_key = self._make_cache_key(data_type, ticker, start, end, frequency)
        
        # Check L1 (memory)
        if data := self.l1_cache.get(cache_key):
            self.stats.l1_hits += 1
            return self._slice_to_requested_range(data, start, end)
        
        # Check L2 (disk)
        if data := self.l2_cache.get(cache_key):
            self.stats.l2_hits += 1
            # Promote to L1
            self.l1_cache.set(cache_key, data)
            return self._slice_to_requested_range(data, start, end)
        
        self.stats.misses += 1
        return None
```

### 3. Cached Provider Wrapper (`src/data/providers/cached_provider.py`)

```python
class CachedDataProvider:
    """Wraps any DataProvider with caching functionality."""
    
    def __init__(
        self,
        provider: DataProvider,
        cache_manager: CacheManager,
        config: Optional[CacheConfig] = None
    ):
        self.provider = provider
        self.cache = cache_manager
        self.config = config or CacheConfig()
    
    def get_data(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        **kwargs
    ) -> pd.Series:
        """Get data with caching."""
        # Check if cacheable
        if not self._is_cacheable(data_type, **kwargs):
            return self.provider.get_data(data_type, start, end, ticker, frequency, **kwargs)
        
        # Try cache first
        if cached := self.cache.get_market_data(data_type, ticker, start, end, frequency):
            logger.debug(f"Cache hit for {data_type}:{ticker}")
            return cached
        
        # Fetch from provider with extended range
        fetch_start, fetch_end = self._extend_cache_range(start, end, frequency)
        data = self.provider.get_data(data_type, fetch_start, fetch_end, ticker, frequency, **kwargs)
        
        # Store in cache
        ttl = self._calculate_ttl(data_type, end)
        self.cache.set_market_data(data_type, ticker, fetch_start, fetch_end, frequency, data, ttl)
        
        # Return requested range
        return data[start:end]
```

### 4. Cache Configuration (`src/data/cache/config.py`)

```python
@dataclass
class CacheConfig:
    """Configuration for caching layer."""
    
    # Cache storage
    cache_dir: Path = Path("~/.portfolio_optimizer/cache")
    memory_size_mb: int = 500
    
    # TTL settings (seconds)
    ttl_recent: int = 300        # 5 minutes for data < 1 day old
    ttl_standard: int = 3600     # 1 hour for data < 1 week old
    ttl_historical: int = 86400  # 1 day for older data
    ttl_immutable: int = 604800  # 1 week for data > 1 year old
    
    # Fetching strategy
    cache_extension_days: Dict[str, int] = field(default_factory=lambda: {
        "daily": 30,      # Fetch extra month for daily
        "weekly": 90,     # Fetch extra quarter for weekly
        "monthly": 365,   # Fetch extra year for monthly
    })
    
    # What to cache
    cacheable_data_types: Set[DataType] = field(default_factory=lambda: {
        RawDataType.ADJUSTED_CLOSE,
        RawDataType.VOLUME,
        RawDataType.DIVIDEND,
        LogicalDataType.TOTAL_RETURN,
        LogicalDataType.SIMPLE_RETURN,
        # Don't cache: real-time quotes, news sentiment
    })
```

### 5. Cache Utilities

#### 5.1 Cache Warming
```python
# src/data/cache/warming.py
class CacheWarmer:
    """Pre-populate cache with common requests."""
    
    def warm_universe(
        self,
        provider: DataProvider,
        cache: CacheManager,
        tickers: List[str],
        lookback_years: int = 5
    ):
        """Warm cache for a universe of tickers."""
        end = date.today()
        start = end - timedelta(days=365 * lookback_years)
        
        for ticker in tqdm(tickers, desc="Warming cache"):
            for data_type in [RawDataType.ADJUSTED_CLOSE, LogicalDataType.TOTAL_RETURN]:
                try:
                    data = provider.get_data(data_type, start, end, ticker)
                    # Data is now cached
                except Exception as e:
                    logger.warning(f"Failed to warm cache for {ticker}: {e}")
```

#### 5.2 Cache Maintenance
```python
# src/data/cache/maintenance.py
class CacheMaintenance:
    """Cache maintenance utilities."""
    
    def cleanup_expired(self, cache: CacheManager):
        """Remove expired entries."""
        pass
    
    def optimize_db(self, cache: SQLiteCache):
        """Vacuum and analyze SQLite database."""
        pass
    
    def get_statistics(self, cache: CacheManager) -> Dict:
        """Get cache usage statistics."""
        pass
```

### 6. Integration Example

```python
# src/data/factory.py updates
def create_production_provider(config: Optional[Dict] = None) -> DataProvider:
    """Create production provider with all layers."""
    # Create base providers
    coordinator = RawDataProviderCoordinator()
    transformed = TransformedDataProvider(coordinator)
    
    # Add caching layer
    cache_config = CacheConfig.from_dict(config.get("cache", {}))
    cache_manager = CacheManager(cache_config)
    cached = CachedDataProvider(transformed, cache_manager)
    
    return cached
```

## Testing Strategy

### Unit Tests
- Test each cache backend independently
- Test cache key generation
- Test TTL calculation
- Test LRU eviction
- Test size limits

### Integration Tests
- Test cache hit/miss scenarios
- Test cache promotion between tiers
- Test concurrent access
- Test cache warming
- Benchmark performance improvements

### Performance Tests
```python
def test_cache_performance_improvement():
    """Verify cache provides significant speedup."""
    # First call - populate cache
    start_time = time.time()
    data1 = cached_provider.get_data(...)
    first_call_time = time.time() - start_time
    
    # Second call - from cache
    start_time = time.time()
    data2 = cached_provider.get_data(...)
    cache_call_time = time.time() - start_time
    
    assert cache_call_time < first_call_time * 0.1  # 10x speedup
```

## Success Criteria

- [ ] Memory cache with LRU eviction
- [ ] Persistent SQLite cache
- [ ] Smart cache key generation with date rounding
- [ ] Configurable TTL based on data age
- [ ] Cache statistics and monitoring
- [ ] 10x performance improvement on cache hits
- [ ] Cache warming utilities
- [ ] Thread-safe operations
- [ ] Comprehensive test coverage

## Configuration

```yaml
# config/cache.yaml
cache:
  # Storage settings
  cache_directory: ~/.portfolio_optimizer/cache
  memory_size_mb: 500
  
  # TTL configuration (seconds)
  ttl:
    recent_data: 300      # 5 minutes
    standard_data: 3600   # 1 hour
    historical_data: 86400 # 1 day
    immutable_data: 604800 # 1 week
  
  # Cache warming
  warming:
    enabled: true
    universe: ["SPY", "AGG", "GLD", "VNQ", "VXUS"]
    lookback_years: 5
  
  # Maintenance
  maintenance:
    cleanup_interval: 3600  # Run cleanup hourly
    max_cache_size_gb: 10
```

## Next Steps

After caching implementation:
1. Quality monitoring layer
2. Provider factory with full configuration
3. Production deployment guide
4. Performance optimization

The caching layer will dramatically improve user experience and reduce load on external APIs!
