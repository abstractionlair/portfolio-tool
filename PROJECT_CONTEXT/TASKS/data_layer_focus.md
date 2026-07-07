# Data Layer Focus Areas

**Status**: IN PROGRESS  
**Created**: 2025-07-11  
**Priority**: HIGH - Critical foundation for all optimization work

## Overview

This document identifies orthogonal improvements to the data layer that would enhance the portfolio optimizer's robustness, flexibility, and production readiness.

## Current Data Layer Architecture

### Existing Components
1. **market_data.py** - Basic yfinance wrapper for price history
2. **total_returns.py** - Enhanced fetcher for total returns including dividends
3. **fred_data.py** - Economic data from FRED (with fallback synthetic data)
4. **return_decomposition.py** - Decomposes returns into inflation/RF/risk premium
5. **alignment_strategies.py** - Modular strategies for aligning different frequency data
6. **multi_frequency.py** - Handles daily/weekly/monthly/quarterly data
7. **exposure_universe.py** - Maps funds to exposures

### Current Strengths
- ✅ FRED fallback system prevents API rate limiting failures
- ✅ Modular alignment strategies for data retention (30x improvement)
- ✅ Multi-frequency support with proper compounding
- ✅ Return decomposition for risk premium isolation
- ✅ Total return fetching with dividend handling

### Current Limitations
- ❌ No persistent caching (only in-memory)
- ❌ Limited data source fallbacks (only yfinance for equities)
- ❌ No corporate action handling beyond dividends
- ❌ Missing mutual fund data support
- ❌ No data quality monitoring/alerting
- ❌ No batch/parallel fetching optimization
- ❌ Limited metadata management

## Focus Area 1: Robust Data Source Abstraction

### Problem
Current system is tightly coupled to yfinance. When yfinance fails or lacks data (mutual funds), the system has no alternatives.

### Solution: Multi-Source Data Layer
```python
# src/data/sources/base.py
class DataSource(Protocol):
    """Common interface for all data sources."""
    def fetch_prices(self, ticker: str, start: datetime, end: datetime) -> pd.Series
    def fetch_dividends(self, ticker: str, start: datetime, end: datetime) -> pd.Series
    def fetch_splits(self, ticker: str, start: datetime, end: datetime) -> pd.Series
    def get_metadata(self, ticker: str) -> Dict[str, Any]
    def is_available(self, ticker: str) -> bool

# src/data/sources/
├── yfinance_source.py      # Current primary
├── polygon_source.py        # High-quality alternative
├── alpha_vantage_source.py  # Free tier backup
├── eod_historical.py        # Comprehensive paid option
├── manual_csv_source.py     # Fallback for missing data
└── composite_source.py      # Intelligent routing
```

### Benefits
- Never blocked by single API failure
- Support for mutual funds via alternative sources
- Graceful degradation
- A/B testing of data quality

## Focus Area 2: Persistent Caching System

### Problem
Every run re-fetches all data. This is slow, wastes API calls, and risks rate limiting.

### Solution: Tiered Cache Architecture
```python
# src/data/cache/
├── base.py              # Cache interface
├── sqlite_cache.py      # Local SQLite for development
├── redis_cache.py       # Redis for production
├── filesystem_cache.py  # Simple file-based option
└── cache_manager.py     # Coordinates multiple caches

class CacheManager:
    """Manages multi-tier caching with TTL and invalidation."""
    
    def __init__(self):
        self.l1_cache = MemoryCache(ttl_seconds=300)      # 5 min hot cache
        self.l2_cache = SQLiteCache(ttl_days=1)           # Daily cache
        self.l3_cache = FileSystemCache(ttl_days=30)      # Monthly archive
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        # Try caches in order
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            if data := cache.get(key):
                # Promote to faster caches
                self._promote(key, data)
                return data
        return None
```

### Features
- TTL-based expiration
- Selective invalidation
- Compression for large datasets
- Cache warming/preloading
- Usage statistics

## Focus Area 3: Data Quality Framework

### Problem
No systematic monitoring of data quality issues (missing data, suspicious values, stale prices).

### Solution: Quality Monitoring System
```python
# src/data/quality/
├── validators.py        # Data validation rules
├── monitors.py          # Real-time quality monitoring
├── alerts.py           # Quality issue notifications
└── reports.py          # Quality dashboards

class DataQualityMonitor:
    """Monitors and reports on data quality issues."""
    
    def validate_prices(self, data: pd.DataFrame) -> QualityReport:
        issues = []
        
        # Check for gaps
        if gaps := self._find_gaps(data):
            issues.append(DataGap(gaps))
        
        # Check for suspicious returns
        if outliers := self._find_outliers(data):
            issues.append(ReturnOutlier(outliers))
        
        # Check for stale data
        if stale := self._find_stale_prices(data):
            issues.append(StaleData(stale))
        
        return QualityReport(issues)
```

### Checks
- Missing data periods
- Extreme returns (>25% daily)
- Zero volume periods
- Stale prices (unchanged for N days)
- Weekend/holiday data
- Corporate action detection

## Focus Area 4: Metadata Management

### Problem
Limited metadata about securities (asset class, inception date, expense ratios, etc.).

### Solution: Rich Metadata System
```python
# src/data/metadata/
├── security_master.py   # Core security information
├── fund_metadata.py     # Mutual fund/ETF specifics
├── mappings.py         # Ticker changes, mergers
└── enrichment.py       # Auto-enrichment from APIs

class SecurityMaster:
    """Central repository for security metadata."""
    
    def get_security_info(self, ticker: str) -> SecurityInfo:
        return SecurityInfo(
            ticker=ticker,
            name=self._get_name(ticker),
            asset_class=self._get_asset_class(ticker),
            inception_date=self._get_inception(ticker),
            is_etf=self._is_etf(ticker),
            is_mutual_fund=self._is_mutual_fund(ticker),
            expense_ratio=self._get_expense_ratio(ticker),
            aum=self._get_aum(ticker),
            issuer=self._get_issuer(ticker),
            benchmark=self._get_benchmark(ticker)
        )
```

### Features
- Automatic enrichment from multiple sources
- Historical ticker mappings
- Corporate action tracking
- Fund categorization
- Expense tracking

## Focus Area 5: Parallel Data Fetching

### Problem
Sequential fetching is slow for large universes. No batching optimization.

### Solution: Async/Parallel Framework
```python
# src/data/parallel/
├── batch_fetcher.py     # Intelligent batching
├── async_fetcher.py     # Async I/O for APIs
├── pool_manager.py      # Thread/process pools
└── rate_limiter.py      # Respect API limits

class ParallelDataFetcher:
    """Fetches data in parallel while respecting rate limits."""
    
    async def fetch_universe(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime
    ) -> Dict[str, pd.DataFrame]:
        # Group by data source
        by_source = self._group_by_optimal_source(tickers)
        
        # Fetch in parallel with rate limiting
        tasks = []
        for source, ticker_batch in by_source.items():
            for chunk in self._chunk_for_rate_limit(ticker_batch, source):
                tasks.append(self._fetch_chunk(source, chunk, start, end))
        
        results = await asyncio.gather(*tasks)
        return self._merge_results(results)
```

### Benefits
- 10x+ speed improvement for large universes
- Automatic rate limit handling
- Optimal source selection per security
- Progress reporting

## Focus Area 6: Alternative Data Integration

### Problem
Only traditional price/volume data. Missing alternative datasets that could improve analysis.

### Solution: Alternative Data Framework
```python
# src/data/alternative/
├── sentiment.py         # News/social sentiment
├── fundamentals.py      # Earnings, ratios
├── macro.py            # Economic indicators
├── flow.py             # Fund flows data
└── options.py          # Options flow/implied vol

class AlternativeDataHub:
    """Integrates alternative data sources."""
    
    def get_enhanced_data(self, ticker: str) -> EnhancedData:
        return EnhancedData(
            prices=self._get_prices(ticker),
            sentiment=self._get_sentiment(ticker),
            fundamentals=self._get_fundamentals(ticker),
            flow_data=self._get_flows(ticker),
            options_data=self._get_options_metrics(ticker)
        )
```

## Implementation Priority

### Phase 1: Foundation (1 week)
1. **Data Source Abstraction** - Prevent failures, enable fallbacks
2. **Persistent Caching** - Improve speed, reduce API usage

### Phase 2: Robustness (1 week)
3. **Data Quality Framework** - Catch issues early
4. **Parallel Fetching** - 10x speed improvement

### Phase 3: Enhancement (1 week)
5. **Metadata Management** - Rich security information
6. **Alternative Data** - Advanced analytics

## Success Metrics

### Performance
- Data fetch time: <2s for 50 securities (from <20s)
- Cache hit rate: >90% for repeated analysis
- API calls: 80% reduction

### Reliability
- Zero data failures due to API issues
- 100% data availability for supported securities
- <1% missing data points after quality fixes

### Capabilities
- Support for 10,000+ securities
- 15+ data sources integrated
- Real-time quality monitoring

## Testing Strategy

### Unit Tests
- Mock all external APIs
- Test each source independently
- Validate cache behavior
- Quality check accuracy

### Integration Tests
- Multi-source fallback scenarios
- Cache coordination
- Parallel fetching stress tests
- End-to-end workflows

### Performance Tests
- Large universe fetching
- Cache performance under load
- Memory usage optimization
- API rate limit compliance

## Next Steps

1. Review and prioritize focus areas
2. Create detailed implementation tasks
3. Set up development branch
4. Begin Phase 1 implementation

The data layer is the foundation of the entire system. These improvements will make the portfolio optimizer production-ready and significantly more robust.
