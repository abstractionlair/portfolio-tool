# Task: Implement Quality Monitoring Layer

**Status**: TODO  
**Priority**: MEDIUM  
**Estimated Time**: 2 days  
**Dependencies**: Caching layer complete

## Overview

The quality monitoring layer will detect and optionally fix data quality issues, ensuring the portfolio optimizer always works with clean, reliable data. This builds on the quality interfaces already defined.

## Implementation Plan

### 1. Quality Checks (`src/data/quality/checks/`)

#### 1.1 Base Quality Check
```python
# src/data/quality/checks/base.py
from abc import ABC, abstractmethod

class QualityCheck(ABC):
    """Base class for all quality checks."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    @abstractmethod
    def check(self, data: pd.Series, context: CheckContext) -> List[QualityIssue]:
        """Run quality check and return issues found."""
        pass
    
    @abstractmethod
    def fix(self, data: pd.Series, issues: List[QualityIssue]) -> pd.Series:
        """Attempt to fix identified issues."""
        pass
    
    @abstractmethod
    def applies_to(self, data_type: DataType) -> bool:
        """Check if this quality check applies to data type."""
        pass
```

#### 1.2 Missing Data Check
```python
# src/data/quality/checks/missing_data.py
class MissingDataCheck(QualityCheck):
    """Detect and fix missing data gaps."""
    
    def __init__(self, max_gap_days: int = 5, max_gap_pct: float = 0.1):
        self.max_gap_days = max_gap_days
        self.max_gap_pct = max_gap_pct
    
    def check(self, data: pd.Series, context: CheckContext) -> List[QualityIssue]:
        issues = []
        
        # Expected vs actual dates
        expected_dates = pd.date_range(
            start=data.index[0],
            end=data.index[-1],
            freq=context.frequency
        )
        missing_dates = expected_dates.difference(data.index)
        
        if len(missing_dates) > 0:
            # Find contiguous gaps
            gaps = self._find_contiguous_gaps(missing_dates)
            
            for gap_start, gap_end, gap_days in gaps:
                severity = self._determine_severity(gap_days, len(data))
                
                issues.append(QualityIssue(
                    severity=severity,
                    description=f"Missing {gap_days} days of data",
                    affected_dates=[gap_start, gap_end],
                    can_auto_fix=gap_days <= self.max_gap_days
                ))
        
        return issues
    
    def fix(self, data: pd.Series, issues: List[QualityIssue]) -> pd.Series:
        """Fix by forward filling small gaps."""
        fixed_data = data.copy()
        
        for issue in issues:
            if issue.can_auto_fix:
                # Forward fill the gap
                start, end = issue.affected_dates
                fixed_data = fixed_data.reindex(
                    pd.date_range(start, end, freq='D')
                ).ffill(limit=self.max_gap_days)
        
        return fixed_data
```

#### 1.3 Extreme Value Check
```python
# src/data/quality/checks/extreme_values.py
class ExtremeValueCheck(QualityCheck):
    """Detect extreme/outlier values."""
    
    def __init__(self, return_threshold: float = 0.25, price_threshold: float = 0.5):
        self.return_threshold = return_threshold
        self.price_threshold = price_threshold
    
    def check(self, data: pd.Series, context: CheckContext) -> List[QualityIssue]:
        issues = []
        
        if context.data_type in RETURN_TYPES:
            # Check for extreme returns
            extreme = data[data.abs() > self.return_threshold]
            for date, value in extreme.items():
                issues.append(QualityIssue(
                    severity="critical",
                    description=f"Extreme return: {value:.1%}",
                    affected_dates=[date],
                    can_auto_fix=False  # Don't auto-fix extreme values
                ))
        
        elif context.data_type in PRICE_TYPES:
            # Check for extreme price moves
            returns = data.pct_change()
            extreme = returns[returns.abs() > self.price_threshold]
            for date, value in extreme.items():
                issues.append(QualityIssue(
                    severity="critical",
                    description=f"Extreme price move: {value:.1%}",
                    affected_dates=[date],
                    can_auto_fix=False
                ))
        
        return issues
```

#### 1.4 Stale Data Check
```python
# src/data/quality/checks/stale_data.py
class StaleDataCheck(QualityCheck):
    """Detect periods of unchanged values."""
    
    def check(self, data: pd.Series, context: CheckContext) -> List[QualityIssue]:
        issues = []
        
        # Find runs of identical values
        unchanged = (data.diff() == 0)
        runs = self._find_runs(unchanged)
        
        for start, end, length in runs:
            if length >= self.max_unchanged_days:
                issues.append(QualityIssue(
                    severity="warning",
                    description=f"Value unchanged for {length} days",
                    affected_dates=[start, end],
                    can_auto_fix=False
                ))
        
        return issues
```

### 2. Quality Monitor (`src/data/quality/monitor.py`)

```python
class DefaultQualityMonitor:
    """Comprehensive data quality monitoring."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.checks = self._initialize_checks()
        self.stats = QualityStatistics()
    
    def _initialize_checks(self) -> List[QualityCheck]:
        """Initialize all configured quality checks."""
        checks = []
        
        if self.config.enable_missing_data_check:
            checks.append(MissingDataCheck(
                max_gap_days=self.config.max_gap_days
            ))
        
        if self.config.enable_extreme_value_check:
            checks.append(ExtremeValueCheck(
                return_threshold=self.config.extreme_return_threshold
            ))
        
        if self.config.enable_stale_data_check:
            checks.append(StaleDataCheck(
                max_unchanged_days=self.config.max_unchanged_days
            ))
        
        # Add more checks as needed
        checks.extend([
            WeekendDataCheck(),
            VolumeAnomalyCheck(),
            NegativePriceCheck(),
            SplitDetectionCheck()
        ])
        
        return checks
    
    def check_data(
        self,
        data: pd.Series,
        data_type: DataType,
        ticker: Optional[str] = None
    ) -> QualityReport:
        """Run all applicable quality checks."""
        context = CheckContext(
            data_type=data_type,
            ticker=ticker,
            frequency=infer_frequency(data),
            check_date=datetime.now()
        )
        
        all_issues = []
        for check in self.checks:
            if check.applies_to(data_type):
                issues = check.check(data, context)
                all_issues.extend(issues)
        
        return self._create_report(all_issues, context, len(data))
    
    def check_and_fix(
        self,
        data: pd.Series,
        data_type: DataType,
        ticker: Optional[str] = None
    ) -> Tuple[pd.Series, QualityReport]:
        """Check data quality and attempt fixes."""
        report = self.check_data(data, data_type, ticker)
        
        if not self.config.auto_fix or report.quality_score >= self.config.min_quality_score:
            return data, report
        
        # Apply fixes
        fixed_data = data.copy()
        fixes_applied = 0
        
        for check in self.checks:
            if check.applies_to(data_type):
                relevant_issues = [i for i in report.issues if check.can_handle_issue(i)]
                if relevant_issues:
                    fixed_data = check.fix(fixed_data, relevant_issues)
                    fixes_applied += len([i for i in relevant_issues if i.can_auto_fix])
        
        # Re-run checks on fixed data
        final_report = self.check_data(fixed_data, data_type, ticker)
        final_report.data_points_fixed = fixes_applied
        
        return fixed_data, final_report
```

### 3. Quality-Aware Provider (`src/data/providers/quality_provider.py`)

```python
class QualityAwareDataProvider:
    """Adds quality monitoring to any data provider."""
    
    def __init__(
        self,
        provider: DataProvider,
        monitor: QualityMonitor,
        config: Optional[QualityConfig] = None
    ):
        self.provider = provider
        self.monitor = monitor
        self.config = config or QualityConfig()
        self.reports_cache = {}
    
    def get_data(
        self,
        data_type: DataType,
        start: date,
        end: date,
        ticker: Optional[str] = None,
        frequency: Union[str, Frequency] = "daily",
        min_quality_score: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """Get data with quality validation."""
        # Fetch raw data
        data = self.provider.get_data(data_type, start, end, ticker, frequency, **kwargs)
        
        # Check quality
        if self.config.auto_fix:
            data, report = self.monitor.check_and_fix(data, data_type, ticker)
        else:
            report = self.monitor.check_data(data, data_type, ticker)
        
        # Cache report
        report_key = f"{data_type}:{ticker}:{start}:{end}"
        self.reports_cache[report_key] = report
        
        # Validate quality threshold
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
        
        return data
```

### 4. Quality Reporting (`src/data/quality/reporting.py`)

```python
class QualityReporter:
    """Generate quality reports and visualizations."""
    
    def __init__(self, monitor: QualityMonitor):
        self.monitor = monitor
    
    def generate_universe_report(
        self,
        universe_data: Dict[str, pd.Series],
        data_type: DataType
    ) -> UniverseQualityReport:
        """Generate quality report for entire universe."""
        ticker_reports = {}
        
        for ticker, data in universe_data.items():
            report = self.monitor.check_data(data, data_type, ticker)
            ticker_reports[ticker] = report
        
        return UniverseQualityReport(
            check_date=datetime.now(),
            data_type=data_type,
            ticker_reports=ticker_reports,
            summary_statistics=self._calculate_summary_stats(ticker_reports)
        )
    
    def create_quality_dashboard(
        self,
        report: UniverseQualityReport,
        output_path: str
    ):
        """Create visual quality dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Quality score distribution
        scores = [r.quality_score for r in report.ticker_reports.values()]
        axes[0, 0].hist(scores, bins=20, edgecolor='black')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_title('Data Quality Score Distribution')
        
        # 2. Issues by type
        issue_counts = self._count_issues_by_type(report)
        axes[0, 1].bar(issue_counts.keys(), issue_counts.values())
        axes[0, 1].set_xlabel('Issue Type')
        axes[0, 1].set_title('Issues by Type')
        
        # ... more visualizations
```

### 5. Quality Configuration

```yaml
# config/quality.yaml
quality:
  # Enable/disable checks
  checks:
    missing_data:
      enabled: true
      max_gap_days: 5
      max_gap_percentage: 0.1
    
    extreme_values:
      enabled: true
      return_threshold: 0.25  # 25% daily return
      price_threshold: 0.50   # 50% price move
    
    stale_data:
      enabled: true
      max_unchanged_days: 5
    
    volume_anomalies:
      enabled: true
      min_volume: 1000
      spike_threshold: 10  # 10x average
  
  # Auto-fix settings
  auto_fix:
    enabled: true
    min_quality_score: 80.0
    
  # Reporting
  reporting:
    save_reports: true
    report_directory: ./quality_reports
    
  # Alerting
  alerts:
    enabled: true
    critical_threshold: 70.0
    email_on_critical: false
```

## Testing Strategy

### Unit Tests
- Test each quality check independently
- Test issue detection accuracy
- Test fix logic
- Test quality score calculation

### Integration Tests
- Test with real problematic data
- Test auto-fix workflows
- Test reporting generation
- Test alert triggering

### Test Data Sets
Create known problematic data:
```python
# tests/data/quality/fixtures.py
def create_data_with_gaps():
    """Create series with missing data."""
    
def create_data_with_outliers():
    """Create series with extreme values."""
    
def create_stale_data():
    """Create series with unchanged values."""
```

## Success Criteria

- [ ] All major quality checks implemented
- [ ] Auto-fix capability for safe fixes
- [ ] Quality scoring system
- [ ] Comprehensive reporting
- [ ] Visual dashboards
- [ ] Performance < 50ms overhead
- [ ] Configurable thresholds
- [ ] Alert system for critical issues
- [ ] Integration with provider stack

## Next Steps

After quality monitoring:
1. Provider factory with full configuration
2. Production deployment guide
3. Web API development
4. Portfolio optimization integration

The quality layer ensures we're always working with reliable data!
