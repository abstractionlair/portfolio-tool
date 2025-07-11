# Data Layer Improvement Task Specification

**Status**: READY FOR IMPLEMENTATION  
**Priority**: CRITICAL - Foundation for all other work  
**Estimated Time**: 2 weeks for full implementation  
**Created**: 2025-07-11

## Executive Summary

The data layer is the foundation of the portfolio optimizer. While functional, it has several limitations that impact production readiness. This task specification outlines a comprehensive improvement plan focusing on robustness, performance, and extensibility.

## Current State Analysis

### What Works Well
1. **Return Decomposition** - Successfully isolates risk premium from inflation/RF
2. **FRED Fallback** - Synthetic data prevents API failures
3. **Alignment Strategies** - 30x improvement in data retention
4. **Multi-frequency Support** - Handles daily/monthly/quarterly data

### Critical Gaps
1. **No Persistent Caching** - Refetches everything on each run
2. **Single Data Source** - yfinance only, no fallbacks
3. **Limited Asset Support** - Mutual funds often missing
4. **No Quality Monitoring** - Silent data issues
5. **Sequential Fetching** - Slow for large universes
6. **Basic Error Handling** - Fails hard on API issues

## Improvement Phases

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Data Source Abstraction Layer
**Goal**: Decouple from yfinance, enable multiple sources

```python
# src/data/sources/base.py
from abc import ABC, abstractmethod
from typing import Protocol, Optional, Dict, Any
import pandas as pd

class MarketDataSource(Protocol):
    """Protocol for market data sources."""
    
    def fetch_prices(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """Fetch price data (OHLCV + Adjusted Close)."""
        ...
    
    def fetch_dividends(
        self,
        ticker: str,
        start: datetime,
        end: datetime
    ) -> pd.Series:
        """Fetch dividend data."""
        ...
    
    def fetch_splits(
        self,
        ticker: str,
        start: datetime,
        end: datetime
    ) -> pd.Series:
        """Fetch stock split data."""
        ...
    
    def get_metadata(self, ticker: str) -> Dict[str, Any]:
        """Get security metadata."""
        ...
    
    def is_available(self, ticker: str) -> bool:
        """Check if ticker is available from this source."""
        ...
    
    @property
    def name(self) -> str:
        """Source name for logging."""
        ...
    
    @property
    def priority(self) -> int:
        """Priority for source selection (lower = higher priority)."""
        ...
```

**Implementation Sources**:
1. `YFinanceSource` (existing, priority=1)
2. `PolygonSource` (high quality, priority=2)
3. `AlphaVantageSource` (free tier, priority=3)
4. `TiingoSource` (good for mutual funds, priority=2)
5. `CSVSource` (manual fallback, priority=99)

#### 1.2 Intelligent Source Router
**Goal**: Automatically select best source per security

```python
# src/data/sources/router.py
class DataSourceRouter:
    """Routes requests to optimal data source."""
    
    def __init__(self, sources: List[MarketDataSource]):
        self.sources = sorted(sources, key=lambda s: s.priority)
        self._availability_cache = {}
    
    def get_best_source(self, ticker: str) -> Optional[MarketDataSource]:
        """Get best available source for ticker."""
        # Check cache
        if ticker in self._availability_cache:
            return self._availability_cache[ticker]
        
        # Test sources in priority order
        for source in self.sources:
            try:
                if source.is_available(ticker):
                    self._availability_cache[ticker] = source
                    logger.info(f"Selected {source.name} for {ticker}")
                    return source
            except Exception as e:
                logger.warning(f"{source.name} failed for {ticker}: {e}")
                continue
        
        return None
    
    def fetch_with_fallback(
        self,
        ticker: str,
        fetch_method: str,
        *args,
        **kwargs
    ) -> Any:
        """Fetch data with automatic fallback."""
        errors = []
        
        for source in self.sources:
            try:
                method = getattr(source, fetch_method)
                return method(ticker, *args, **kwargs)
            except Exception as e:
                errors.append(f"{source.name}: {str(e)}")
                continue
        
        raise DataUnavailableError(
            f"All sources failed for {ticker}.{fetch_method}: {errors}"
        )
```

#### 1.3 Persistent Cache System
**Goal**: Eliminate redundant API calls, improve speed

```python
# src/data/cache/base.py
class CacheBackend(ABC):
    """Abstract base for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """Remove from cache."""
        pass
    
    @abstractmethod
    def clear(self, pattern: Optional[str] = None):
        """Clear cache, optionally by pattern."""
        pass

# src/data/cache/sqlite_cache.py
class SQLiteCache(CacheBackend):
    """SQLite-based cache for development."""
    
    def __init__(self, db_path: str = "~/.portfolio_optimizer/cache.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    expires_at REAL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires 
                ON cache(expires_at)
            """)

# src/data/cache/manager.py
class CacheManager:
    """Manages multi-tier caching."""
    
    def __init__(self, config: CacheConfig):
        self.memory = MemoryCache(max_size=config.memory_size_mb * 1024 * 1024)
        self.disk = SQLiteCache(config.cache_dir / "market_data.db")
        self.config = config
    
    def get_market_data(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        data_type: str = "prices"
    ) -> Optional[pd.DataFrame]:
        """Get market data from cache."""
        key = self._make_key(ticker, start, end, data_type)
        
        # Try memory first
        if data := self.memory.get(key):
            logger.debug(f"Memory cache hit: {key}")
            return data
        
        # Try disk
        if data := self.disk.get(key):
            logger.debug(f"Disk cache hit: {key}")
            # Promote to memory
            self.memory.set(key, data)
            return data
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set_market_data(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        data: pd.DataFrame,
        data_type: str = "prices"
    ):
        """Store market data in cache."""
        key = self._make_key(ticker, start, end, data_type)
        
        # Determine TTL based on data age
        ttl = self._calculate_ttl(end)
        
        # Store in both tiers
        self.memory.set(key, data, ttl)
        self.disk.set(key, data, ttl)
        
        logger.debug(f"Cached {key} with TTL {ttl}s")
```

### Phase 2: Data Quality & Performance (Week 2)

#### 2.1 Data Quality Framework
**Goal**: Detect and handle data quality issues

```python
# src/data/quality/validators.py
class DataValidator:
    """Validates market data quality."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
    
    def validate_prices(self, df: pd.DataFrame, ticker: str) -> ValidationReport:
        """Validate price data quality."""
        issues = []
        
        # Check for missing data
        if missing := self._check_missing_data(df):
            issues.append(MissingDataIssue(ticker, missing))
        
        # Check for extreme returns
        if outliers := self._check_return_outliers(df):
            issues.append(ReturnOutlierIssue(ticker, outliers))
        
        # Check for stale prices
        if stale := self._check_stale_prices(df):
            issues.append(StalePriceIssue(ticker, stale))
        
        # Check for weekend/holiday data
        if weekend := self._check_weekend_data(df):
            issues.append(WeekendDataIssue(ticker, weekend))
        
        return ValidationReport(ticker, issues)
    
    def _check_return_outliers(self, df: pd.DataFrame) -> List[datetime]:
        """Find returns exceeding threshold."""
        returns = df['Adj Close'].pct_change()
        threshold = self.config.max_daily_return
        
        outliers = returns[returns.abs() > threshold]
        return outliers.index.tolist()

# src/data/quality/monitor.py
class DataQualityMonitor:
    """Real-time data quality monitoring."""
    
    def __init__(self):
        self.validator = DataValidator(QualityConfig())
        self.alerts = []
        self.stats = defaultdict(int)
    
    def check_data(self, ticker: str, data: pd.DataFrame) -> pd.DataFrame:
        """Check data quality and potentially fix issues."""
        report = self.validator.validate_prices(data, ticker)
        
        if report.has_issues():
            self.stats['tickers_with_issues'] += 1
            
            # Try to fix automatically
            fixed_data = self._auto_fix(data, report)
            
            # Alert if can't fix
            if report.has_critical_issues():
                self.alerts.append(DataQualityAlert(ticker, report))
            
            return fixed_data
        
        self.stats['clean_tickers'] += 1
        return data
```

#### 2.2 Parallel Data Fetching
**Goal**: 10x speed improvement for universe fetching

```python
# src/data/parallel/batch_fetcher.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp

class ParallelDataFetcher:
    """Fetches data in parallel with rate limiting."""
    
    def __init__(self, router: DataSourceRouter, max_workers: int = 10):
        self.router = router
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.rate_limiters = {}
    
    async def fetch_universe(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers in parallel."""
        
        # Group by optimal source
        by_source = self._group_by_source(tickers)
        
        # Create fetch tasks
        tasks = []
        for source_name, ticker_group in by_source.items():
            rate_limiter = self._get_rate_limiter(source_name)
            
            for ticker in ticker_group:
                task = self._fetch_with_rate_limit(
                    ticker, start, end, source_name, rate_limiter
                )
                tasks.append(task)
        
        # Execute with progress reporting
        results = {}
        for i, task in enumerate(asyncio.as_completed(tasks)):
            ticker, data = await task
            results[ticker] = data
            
            if progress_callback:
                progress_callback(i + 1, len(tasks))
        
        return results
    
    async def _fetch_with_rate_limit(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        source_name: str,
        rate_limiter: RateLimiter
    ) -> Tuple[str, pd.DataFrame]:
        """Fetch single ticker with rate limiting."""
        async with rate_limiter:
            # Run in thread pool (most finance APIs are sync)
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.executor,
                self._fetch_sync,
                ticker, start, end, source_name
            )
            return ticker, data
```

#### 2.3 Enhanced Metadata System
**Goal**: Rich security information for better analysis

```python
# src/data/metadata/security_master.py
class SecurityMaster:
    """Central repository for security metadata."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.enrichers = [
            YFinanceEnricher(),
            OpenFIGIEnricher(),
            ManualMappingEnricher()
        ]
    
    def get_security_info(self, ticker: str) -> SecurityInfo:
        """Get comprehensive security information."""
        # Check cache first
        if info := self.cache.get(f"metadata:{ticker}"):
            return info
        
        # Build from enrichers
        info = SecurityInfo(ticker=ticker)
        
        for enricher in self.enrichers:
            try:
                enricher.enrich(info)
            except Exception as e:
                logger.warning(f"Enricher {enricher} failed for {ticker}: {e}")
        
        # Cache for 7 days
        self.cache.set(f"metadata:{ticker}", info, ttl=7*24*3600)
        
        return info

@dataclass
class SecurityInfo:
    """Comprehensive security information."""
    ticker: str
    name: Optional[str] = None
    asset_class: Optional[str] = None
    security_type: Optional[str] = None  # ETF, Mutual Fund, Stock
    inception_date: Optional[datetime] = None
    
    # Fund-specific
    expense_ratio: Optional[float] = None
    aum: Optional[float] = None
    issuer: Optional[str] = None
    benchmark: Optional[str] = None
    
    # Identifiers
    cusip: Optional[str] = None
    isin: Optional[str] = None
    figi: Optional[str] = None
    
    # Historical
    previous_tickers: List[str] = field(default_factory=list)
    name_changes: List[Tuple[datetime, str]] = field(default_factory=list)
```

### Implementation Plan

#### Week 1: Core Infrastructure
**Monday-Tuesday**: Data Source Abstraction
- [ ] Implement base protocol and router
- [ ] Create YFinance adapter (refactor existing)
- [ ] Add Polygon source
- [ ] Add CSV fallback source
- [ ] Write comprehensive tests

**Wednesday-Thursday**: Cache System
- [ ] Implement cache backends (memory, SQLite)
- [ ] Create cache manager with TTL logic
- [ ] Add cache warming functionality
- [ ] Integration with data sources
- [ ] Performance benchmarks

**Friday**: Integration & Testing
- [ ] Update existing code to use new infrastructure
- [ ] End-to-end testing
- [ ] Performance comparison
- [ ] Documentation

#### Week 2: Quality & Performance
**Monday-Tuesday**: Data Quality
- [ ] Implement validators
- [ ] Create quality monitor
- [ ] Add auto-fix capabilities
- [ ] Build alerting system
- [ ] Quality dashboard

**Wednesday-Thursday**: Parallel Fetching
- [ ] Implement batch fetcher
- [ ] Add rate limiters per source
- [ ] Progress reporting
- [ ] Error handling
- [ ] Performance testing

**Friday**: Metadata & Polish
- [ ] Security master implementation
- [ ] Metadata enrichers
- [ ] Final integration
- [ ] Complete documentation
- [ ] Deployment guide

### Success Metrics

#### Performance
- [ ] 50-ticker universe fetch: <2 seconds (from ~20s)
- [ ] Cache hit rate: >90% for repeated analysis
- [ ] API calls: 80% reduction
- [ ] Memory usage: <500MB for 1000 tickers

#### Reliability
- [ ] Zero failures due to single API issues
- [ ] 100% data availability for supported securities
- [ ] <1% missing data after quality fixes
- [ ] Automatic recovery from transient errors

#### Quality
- [ ] Detect 100% of extreme returns (>25% daily)
- [ ] Flag stale price periods
- [ ] Identify and handle corporate actions
- [ ] Generate quality reports

### Testing Strategy

#### Unit Tests
```python
# tests/test_data/test_sources.py
def test_source_fallback():
    """Test automatic fallback between sources."""
    router = DataSourceRouter([
        MockFailingSource(priority=1),
        MockWorkingSource(priority=2)
    ])
    
    data = router.fetch_with_fallback("AAPL", "fetch_prices", ...)
    assert data is not None

# tests/test_data/test_cache.py
def test_cache_ttl():
    """Test cache expiration."""
    cache = SQLiteCache()
    cache.set("key", "value", ttl=1)
    assert cache.get("key") == "value"
    
    time.sleep(2)
    assert cache.get("key") is None

# tests/test_data/test_quality.py
def test_outlier_detection():
    """Test extreme return detection."""
    df = create_test_prices_with_outlier()
    validator = DataValidator()
    report = validator.validate_prices(df, "TEST")
    
    assert report.has_issues()
    assert any(isinstance(i, ReturnOutlierIssue) for i in report.issues)
```

### Migration Guide

1. **Update imports**:
```python
# Old
from src.data.market_data import MarketDataFetcher

# New
from src.data import DataLayer
data_layer = DataLayer()  # Automatically uses best configuration
```

2. **Fetch with automatic caching**:
```python
# Old
fetcher = MarketDataFetcher()
data = fetcher.fetch_price_history("AAPL", start, end)

# New
data = data_layer.get_prices("AAPL", start, end)
# Automatically cached, quality checked, and sourced optimally
```

3. **Parallel universe fetching**:
```python
# New capability
universe_data = await data_layer.fetch_universe_async(
    tickers=["AAPL", "MSFT", "GOOGL", ...],
    start=start,
    end=end,
    progress_callback=lambda done, total: print(f"{done}/{total}")
)
```

### Configuration

```yaml
# config/data_layer.yaml
cache:
  backend: sqlite  # or redis, memory
  location: ~/.portfolio_optimizer/cache
  memory_size_mb: 512
  ttl:
    recent: 300      # 5 minutes for recent data
    standard: 86400  # 1 day for standard data
    historical: 604800  # 1 week for old data

sources:
  - name: yfinance
    priority: 1
    rate_limit: 2000/hour
  - name: polygon
    priority: 2
    api_key: ${POLYGON_API_KEY}
    rate_limit: unlimited  # paid tier
  - name: csv
    priority: 99
    directory: ./data/manual

quality:
  max_daily_return: 0.25  # 25%
  max_unchanged_days: 5
  min_volume: 1000
  auto_fix: true

parallel:
  max_workers: 10
  chunk_size: 50
```

## Expected Outcomes

### Immediate Benefits
1. **No more data failures** - Multiple sources with fallbacks
2. **10x faster** - Parallel fetching and caching
3. **Better data** - Quality monitoring and fixes
4. **Production ready** - Robust error handling

### Long-term Benefits
1. **Extensible** - Easy to add new sources
2. **Maintainable** - Clean architecture
3. **Scalable** - Handles large universes
4. **Professional** - Enterprise-grade data layer

## Questions for Desktop Claude

1. Should we prioritize any specific data source integrations?
2. Is Redis caching important for production deployment?
3. Any specific quality checks based on your investment strategy?
4. Preferences on async vs threading for parallel fetching?

## Next Steps

1. Review and approve specification
2. Create feature branch
3. Begin Phase 1 implementation
4. Weekly progress reviews

The improved data layer will transform the portfolio optimizer from a prototype to a production-ready system.
