# Task: Implement Cache and Quality Layers

**Status**: TODO  
**Priority**: MEDIUM  
**Estimated Time**: 2 days  
**Dependencies**: Raw and Transformed providers implemented

## Overview

Implement the caching and quality monitoring layers that wrap any DataProvider to add persistent caching and data quality checks. These are decorator-style providers that add functionality transparently.

## Part 1: Cache Layer Implementation

### 1. `src/data/providers/cached_provider.py`

```python
class CachedDataProvider:
    """
    Adds caching to any data provider.
    
    Features:
    - Transparent caching of all data requests
    - Intelligent cache key generation
    - TTL-based expiration
    - Cache warming utilities
    - Metrics on hit/miss rates
    """
    
    def __init__(
        self,
        provider: DataProvider,
        cache_manager: CacheManager,
        cache_config: Optional[CacheConfig] = None
    ):
        self.provider = provider
        self.cache = cache_manager
        self.config = cache_config or CacheConfig()
        self.stats = CacheStatistics()
    
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
        # Generate cache key
        cache_key = self._make_cache_key(data_type, start, end, ticker, frequency, **kwargs)
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats.record_hit()
            logger.debug(f"Cache hit: {cache_key}")
            return cached
        
        # Miss - fetch from provider
        self.stats.record_miss()
        logger.debug(f"Cache miss: {cache_key}")
        
        data = self.provider.get_data(data_type, start, end, ticker, frequency, **kwargs)
        
        # Determine TTL
        ttl = self._calculate_ttl(data_type, end)
        
        # Store in cache
        self.cache.set(cache_key, data, ttl)
        
        return data
    
    def _make_cache_key(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str],
        frequency: Union[str, Frequency],
        **kwargs
    ) -> str:
        """Generate cache key with smart date rounding."""
        # Round dates to improve hit rate
        cache_start, cache_end = self._round_dates(start, end, frequency)
        
        # Build key components
        parts = [
            data_type.value,
            ticker or "none",
            cache_start.isoformat(),
            cache_end.isoformat(),
            str(frequency)
        ]
        
        # Add relevant kwargs (e.g., tenor for risk-free rate)
        for k, v in sorted(kwargs.items()):
            if k in ['tenor', 'index', 'method']:  # Whitelist cache-relevant params
                parts.extend([k, str(v)])
        
        return ":".join(parts)
    
    def _round_dates(self, start: date, end: date, frequency: Union[str, Frequency]) -> tuple[date, date]:
        """Round dates to improve cache hits."""
        freq = Frequency(frequency) if isinstance(frequency, str) else frequency
        
        if freq == Frequency.DAILY:
            # Round to week boundaries
            cache_start = start - timedelta(days=start.weekday())
            cache_end = end + timedelta(days=6 - end.weekday())
        elif freq == Frequency.MONTHLY:
            # Round to quarter boundaries
            cache_start = date(start.year, ((start.month - 1) // 3) * 3 + 1, 1)
            if end.month % 3 == 0:
                cache_end = end
            else:
                next_quarter = ((end.month - 1) // 3 + 1) * 3 + 1
                if next_quarter > 12:
                    cache_end = date(end.year + 1, 1, 31)
                else:
                    cache_end = date(end.year, next_quarter, 1) - timedelta(days=1)
        else:
            # No rounding for other frequencies
            cache_start, cache_end = start, end
        
        return cache_start, cache_end
    
    def warm_cache(
        self,
        requests: List[CacheRequest],
        progress_callback: Optional[Callable] = None
    ):
        """Pre-populate cache with common requests."""
        for i, request in enumerate(requests):
            try:
                self.get_data(
                    request.data_type,
                    request.start,
                    request.end,
                    request.ticker,
                    request.frequency,
                    **request.kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to warm cache for {request}: {e}")
            
            if progress_callback:
                progress_callback(i + 1, len(requests))
```

### 2. `src/data/cache/file_cache.py`

```python
class FileSystemCache(CacheManager):
    """
    File-based cache using Parquet format.
    
    Features:
    - Efficient Parquet storage
    - TTL tracking via metadata
    - Automatic cleanup of expired entries
    - Thread-safe operations
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._metadata_file = self.cache_dir / ".cache_metadata.json"
        self._load_metadata()
    
    def get(self, key: str) -> Optional[pd.Series]:
        """Retrieve from cache if not expired."""
        with self._lock:
            # Check metadata first
            if key not in self._metadata:
                return None
            
            entry = self._metadata[key]
            if self._is_expired(entry):
                self._delete_entry(key)
                return None
            
            # Load data
            try:
                filepath = self.cache_dir / entry['filename']
                data = pd.read_parquet(filepath)
                
                # Convert to Series if needed
                if isinstance(data, pd.DataFrame):
                    if len(data.columns) == 1:
                        data = data.iloc[:, 0]
                    else:
                        raise ValueError(f"Cached data has multiple columns: {data.columns}")
                
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                self._delete_entry(key)
                return None
    
    def set(self, key: str, data: pd.Series, ttl: Optional[int] = None):
        """Store data with optional TTL in seconds."""
        with self._lock:
            # Generate filename
            filename = f"{hashlib.md5(key.encode()).hexdigest()}.parquet"
            filepath = self.cache_dir / filename
            
            # Save data
            data.to_frame().to_parquet(filepath)
            
            # Update metadata
            self._metadata[key] = {
                'filename': filename,
                'created_at': datetime.now().isoformat(),
                'ttl': ttl,
                'expires_at': (datetime.now() + timedelta(seconds=ttl)).isoformat() if ttl else None,
                'size_bytes': filepath.stat().st_size
            }
            
            self._save_metadata()
    
    def _calculate_ttl(self, data_type: DataType, end_date: date) -> int:
        """Calculate appropriate TTL based on data type and age."""
        days_old = (date.today() - end_date).days
        
        # Recent data expires quickly
        if days_old < 7:
            return 300  # 5 minutes
        elif days_old < 30:
            return 3600  # 1 hour
        elif days_old < 365:
            return 86400  # 1 day
        else:
            return 604800  # 1 week
```

## Part 2: Quality Layer Implementation

### 3. `src/data/providers/quality_provider.py`

```python
class QualityAwareDataProvider:
    """
    Adds data quality checking to any provider.
    
    Features:
    - Automatic quality validation
    - Optional auto-fixing of issues
    - Quality reporting
    - Configurable quality thresholds
    """
    
    def __init__(
        self,
        provider: DataProvider,
        quality_monitor: QualityMonitor,
        config: Optional[QualityConfig] = None
    ):
        self.provider = provider
        self.monitor = quality_monitor
        self.config = config or QualityConfig()
        self.reports = {}  # Cache recent reports
    
    def get_data(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        auto_fix: bool = True,
        min_quality_score: float = None,
        **kwargs
    ) -> pd.Series:
        """Get data with quality validation."""
        # Get data from underlying provider
        raw_data = self.provider.get_data(data_type, start, end, ticker, frequency, **kwargs)
        
        # Check quality
        if auto_fix:
            cleaned_data, report = self.monitor.check_and_fix(raw_data, data_type, ticker)
        else:
            report = self.monitor.check_data(raw_data, data_type, ticker)
            cleaned_data = raw_data
        
        # Store report
        report_key = f"{data_type.value}:{ticker or 'none'}:{start}:{end}"
        self.reports[report_key] = report
        
        # Check quality threshold
        threshold = min_quality_score or self.config.min_quality_score
        if report.quality_score < threshold:
            if report.critical_issues > 0:
                raise DataQualityError(
                    f"Data quality score {report.quality_score:.1f} below threshold {threshold} "
                    f"with {report.critical_issues} critical issues"
                )
            else:
                logger.warning(
                    f"Data quality score {report.quality_score:.1f} below threshold {threshold}"
                )
        
        return cleaned_data
    
    def get_quality_report(
        self,
        data_type: DataType,
        ticker: Optional[str],
        start: date,
        end: date
    ) -> Optional[QualityReport]:
        """Get quality report for recent data request."""
        report_key = f"{data_type.value}:{ticker or 'none'}:{start}:{end}"
        return self.reports.get(report_key)
```

### 4. `src/data/quality/monitors.py`

```python
class DefaultQualityMonitor:
    """
    Default implementation of quality monitoring.
    
    Checks for:
    - Missing data
    - Extreme values
    - Stale prices
    - Weekend/holiday data
    - Volume anomalies
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        self.checks = self._initialize_checks()
    
    def _initialize_checks(self) -> List[QualityCheck]:
        """Initialize all quality checks."""
        return [
            MissingDataCheck(max_gap_days=self.config.max_gap_days),
            ExtremeValueCheck(max_daily_change=self.config.max_daily_return),
            StalePriceCheck(max_unchanged_days=self.config.max_unchanged_days),
            WeekendDataCheck(),
            VolumeAnomalyCheck(min_volume=self.config.min_volume)
        ]
    
    def check_data(
        self,
        data: pd.Series,
        data_type: DataType,
        ticker: Optional[str] = None
    ) -> QualityReport:
        """Run all quality checks."""
        issues = []
        
        for check in self.checks:
            if check.applies_to(data_type):
                check_issues = check.run(data, data_type, ticker)
                issues.extend(check_issues)
        
        return QualityReport(
            ticker=ticker,
            data_type=data_type,
            check_date=datetime.now(),
            total_issues=len(issues),
            critical_issues=sum(1 for i in issues if i.severity == "critical"),
            warning_issues=sum(1 for i in issues if i.severity == "warning"),
            info_issues=sum(1 for i in issues if i.severity == "info"),
            issues=issues,
            data_points_checked=len(data),
            data_points_fixed=0
        )
    
    def check_and_fix(
        self,
        data: pd.Series,
        data_type: DataType,
        ticker: Optional[str] = None
    ) -> tuple[pd.Series, QualityReport]:
        """Check and attempt to fix issues."""
        fixed_data = data.copy()
        issues = []
        total_fixes = 0
        
        for check in self.checks:
            if check.applies_to(data_type):
                check_issues = check.run(fixed_data, data_type, ticker)
                issues.extend(check_issues)
                
                # Attempt fixes
                fixable_issues = [i for i in check_issues if i.can_auto_fix]
                if fixable_issues:
                    fixed_data, fixes = check.fix(fixed_data, fixable_issues)
                    total_fixes += fixes
        
        report = QualityReport(
            ticker=ticker,
            data_type=data_type,
            check_date=datetime.now(),
            total_issues=len(issues),
            critical_issues=sum(1 for i in issues if i.severity == "critical"),
            warning_issues=sum(1 for i in issues if i.severity == "warning"),
            info_issues=sum(1 for i in issues if i.severity == "info"),
            issues=issues,
            data_points_checked=len(data),
            data_points_fixed=total_fixes
        )
        
        return fixed_data, report


class ExtremeValueCheck(QualityCheck):
    """Check for extreme values in return data."""
    
    def __init__(self, max_daily_change: float = 0.25):
        self.max_daily_change = max_daily_change
    
    def applies_to(self, data_type: DataType) -> bool:
        """Only check return types."""
        return "return" in data_type.value.lower()
    
    def run(
        self,
        data: pd.Series,
        data_type: DataType,
        ticker: Optional[str]
    ) -> List[QualityIssue]:
        """Find extreme returns."""
        issues = []
        
        # Find values exceeding threshold
        extreme = data[data.abs() > self.max_daily_change]
        
        for date, value in extreme.items():
            issues.append(QualityIssue(
                severity="critical" if abs(value) > 0.5 else "warning",
                description=f"Extreme return: {value:.1%}",
                affected_dates=[date],
                can_auto_fix=False  # Don't auto-fix extreme values
            ))
        
        return issues
    
    def fix(
        self,
        data: pd.Series,
        issues: List[QualityIssue]
    ) -> tuple[pd.Series, int]:
        """Extreme values should not be auto-fixed."""
        return data, 0
```

## Part 3: Integration Examples

### 5. `src/data/factory.py`

```python
class DataLayerFactory:
    """Factory for creating configured data layer stack."""
    
    @staticmethod
    def create_production_provider(config: DataLayerConfig) -> DataProvider:
        """Create full production data provider stack."""
        
        # Create raw provider with sources
        sources = []
        if config.enable_yfinance:
            sources.append(YFinanceSource())
        if config.enable_fred:
            sources.append(FREDSource())
        if config.enable_csv:
            sources.append(CSVSource(config.csv_dir))
        
        raw_provider = DefaultRawDataProvider(sources)
        
        # Add transformation layer
        transformed = DefaultTransformedDataProvider(raw_provider)
        
        # Add caching if enabled
        if config.enable_cache:
            cache = FileSystemCache(config.cache_dir)
            cached = CachedDataProvider(transformed, cache)
            provider = cached
        else:
            provider = transformed
        
        # Add quality monitoring if enabled
        if config.enable_quality:
            monitor = DefaultQualityMonitor(config.quality_config)
            quality = QualityAwareDataProvider(provider, monitor)
            provider = quality
        
        return provider
    
    @staticmethod
    def create_test_provider() -> DataProvider:
        """Create provider for testing (no external calls)."""
        raw = MockRawDataProvider()
        transformed = DefaultTransformedDataProvider(raw)
        return transformed
```

## Testing Requirements

### Cache Layer Tests

```python
def test_cache_hit_improves_performance():
    """Verify cache significantly improves performance."""
    slow_provider = SlowMockProvider(delay=1.0)  # 1 second delay
    cache = InMemoryCache()
    cached_provider = CachedDataProvider(slow_provider, cache)
    
    # First call - slow
    start_time = time.time()
    data1 = cached_provider.get_data(RawDataType.TREASURY_3M, start, end)
    first_time = time.time() - start_time
    assert first_time > 0.9  # Should take ~1 second
    
    # Second call - fast
    start_time = time.time()
    data2 = cached_provider.get_data(RawDataType.TREASURY_3M, start, end)
    second_time = time.time() - start_time
    assert second_time < 0.1  # Should be instant
    
    assert data1.equals(data2)

def test_cache_key_rounding():
    """Test that date rounding improves hit rate."""
    provider = MockProvider()
    cache = InMemoryCache()
    cached = CachedDataProvider(provider, cache)
    
    # Two slightly different requests
    data1 = cached.get_data(
        RawDataType.ADJUSTED_CLOSE,
        date(2023, 1, 2),  # Tuesday
        date(2023, 1, 30),
        "AAPL",
        "daily"
    )
    
    data2 = cached.get_data(
        RawDataType.ADJUSTED_CLOSE,
        date(2023, 1, 3),  # Wednesday - should hit same cache
        date(2023, 1, 29),
        "AAPL",
        "daily"
    )
    
    # Should be cache hit due to rounding
    assert cached.stats.hits == 1
    assert cached.stats.misses == 1
```

### Quality Layer Tests

```python
def test_quality_check_detects_issues():
    """Test that quality checks find problems."""
    # Create data with issues
    data = pd.Series(
        [100, 100, 100, 150, 100],  # 50% jump!
        index=pd.date_range("2023-01-01", periods=5)
    )
    
    monitor = DefaultQualityMonitor()
    report = monitor.check_data(data, LogicalDataType.SIMPLE_RETURN, "TEST")
    
    assert report.total_issues >= 1
    assert report.critical_issues >= 1
    assert any("Extreme" in issue.description for issue in report.issues)

def test_quality_auto_fix():
    """Test auto-fixing of fixable issues."""
    # Data with gap
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    data = pd.Series(range(10), index=dates)
    # Remove middle dates to create gap
    data = data.drop(dates[4:7])
    
    monitor = DefaultQualityMonitor()
    fixed_data, report = monitor.check_and_fix(data, RawDataType.ADJUSTED_CLOSE, "TEST")
    
    assert len(fixed_data) > len(data)  # Gap filled
    assert report.data_points_fixed > 0
```

## Success Criteria

- [ ] Cache layer passes all tests
- [ ] Quality layer passes all tests
- [ ] Both layers are transparent (same interface)
- [ ] Performance improvement measurable
- [ ] Quality issues detected and reported
- [ ] Configuration is flexible
- [ ] Thread-safe operations
- [ ] Clear logging of operations

## Performance Targets

- Cache hit: <10ms response time
- Cache miss: <10ms overhead
- Quality check: <50ms for 1000 data points
- File cache: <100ms for read/write

## Notes

1. These layers should be completely optional
2. Order matters: Cache should wrap Quality (cache clean data)
3. Consider adding cache warming utilities
4. Quality rules should be configurable per data type
5. Add metrics/monitoring hooks for production

This completes the core data layer implementation with caching and quality monitoring.
