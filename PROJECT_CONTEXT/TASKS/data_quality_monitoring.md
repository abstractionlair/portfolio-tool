# Data Quality Monitoring System

**Component**: Orthogonal Data Layer Enhancement  
**Status**: SPECIFICATION  
**Priority**: HIGH - Prevents silent failures  
**Estimated Time**: 3-4 days

## Problem Statement

The current data layer has no systematic quality monitoring. Issues like:
- Missing data periods
- Extreme/suspicious returns  
- Stale prices (unchanged for days)
- Weekend/holiday data pollution
- Corporate actions (splits not adjusted)

These can cause optimization failures or worse - bad investment decisions based on faulty data.

## Solution Overview

A comprehensive data quality monitoring system that:
1. **Detects** quality issues automatically
2. **Fixes** what can be fixed programmatically  
3. **Alerts** on issues requiring attention
4. **Reports** on overall data health
5. **Prevents** bad data from reaching optimization

## Detailed Design

### 1. Quality Check Framework

```python
# src/data/quality/checks.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class QualityIssue:
    """Base class for quality issues."""
    severity: str  # 'critical', 'warning', 'info'
    ticker: str
    description: str
    affected_dates: List[datetime]
    can_auto_fix: bool = False
    
class QualityCheck(ABC):
    """Base class for quality checks."""
    
    @abstractmethod
    def check(self, data: pd.DataFrame, ticker: str) -> List[QualityIssue]:
        """Run quality check and return issues found."""
        pass
    
    @abstractmethod
    def fix(self, data: pd.DataFrame, issues: List[QualityIssue]) -> pd.DataFrame:
        """Attempt to fix issues if possible."""
        pass

class MissingDataCheck(QualityCheck):
    """Check for missing data periods."""
    
    def __init__(self, max_gap_days: int = 5):
        self.max_gap_days = max_gap_days
    
    def check(self, data: pd.DataFrame, ticker: str) -> List[QualityIssue]:
        issues = []
        
        # Find gaps in index
        date_range = pd.date_range(data.index.min(), data.index.max(), freq='B')
        missing_dates = date_range.difference(data.index)
        
        if len(missing_dates) > 0:
            # Group consecutive missing dates
            gaps = self._find_gaps(missing_dates)
            
            for gap_start, gap_end, gap_days in gaps:
                if gap_days > self.max_gap_days:
                    issues.append(QualityIssue(
                        severity='warning',
                        ticker=ticker,
                        description=f"Missing {gap_days} days of data",
                        affected_dates=[gap_start, gap_end],
                        can_auto_fix=gap_days <= 10
                    ))
        
        return issues
    
    def fix(self, data: pd.DataFrame, issues: List[QualityIssue]) -> pd.DataFrame:
        """Forward-fill small gaps."""
        for issue in issues:
            if issue.can_auto_fix:
                # Forward fill up to 10 days
                data = data.asfreq('B').ffill(limit=10)
        return data

class ExtremeReturnCheck(QualityCheck):
    """Check for suspicious returns."""
    
    def __init__(self, max_daily_return: float = 0.25):
        self.max_daily_return = max_daily_return
    
    def check(self, data: pd.DataFrame, ticker: str) -> List[QualityIssue]:
        issues = []
        
        # Calculate returns
        returns = data['Adj Close'].pct_change()
        extreme = returns[returns.abs() > self.max_daily_return]
        
        for date, return_val in extreme.items():
            issues.append(QualityIssue(
                severity='critical',
                ticker=ticker,
                description=f"Extreme return: {return_val:.1%}",
                affected_dates=[date],
                can_auto_fix=False  # Requires manual review
            ))
        
        return issues

class StalePriceCheck(QualityCheck):
    """Check for unchanged prices over multiple days."""
    
    def __init__(self, max_unchanged_days: int = 5):
        self.max_unchanged_days = max_unchanged_days
    
    def check(self, data: pd.DataFrame, ticker: str) -> List[QualityIssue]:
        issues = []
        
        # Find consecutive unchanged prices
        price_changes = data['Adj Close'].diff()
        unchanged = (price_changes == 0)
        
        # Find runs of unchanged prices
        runs = self._find_runs(unchanged)
        
        for start, end, length in runs:
            if length >= self.max_unchanged_days:
                issues.append(QualityIssue(
                    severity='warning',
                    ticker=ticker,
                    description=f"Price unchanged for {length} days",
                    affected_dates=[start, end],
                    can_auto_fix=False
                ))
        
        return issues

class VolumeAnomalyCheck(QualityCheck):
    """Check for volume anomalies."""
    
    def check(self, data: pd.DataFrame, ticker: str) -> List[QualityIssue]:
        issues = []
        
        if 'Volume' not in data.columns:
            return issues
        
        # Check for zero volume
        zero_volume = data[data['Volume'] == 0]
        if len(zero_volume) > 0:
            issues.append(QualityIssue(
                severity='warning',
                ticker=ticker,
                description=f"Zero volume on {len(zero_volume)} days",
                affected_dates=zero_volume.index.tolist(),
                can_auto_fix=False
            ))
        
        # Check for volume spikes (10x average)
        avg_volume = data['Volume'].rolling(20).mean()
        volume_spikes = data[data['Volume'] > avg_volume * 10]
        
        for date, volume in volume_spikes['Volume'].items():
            if pd.notna(avg_volume.loc[date]):
                issues.append(QualityIssue(
                    severity='info',
                    ticker=ticker,
                    description=f"Volume spike: {volume/avg_volume.loc[date]:.1f}x average",
                    affected_dates=[date],
                    can_auto_fix=False
                ))
        
        return issues
```

### 2. Quality Monitor

```python
# src/data/quality/monitor.py
class DataQualityMonitor:
    """Comprehensive data quality monitoring."""
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        self.checks = [
            MissingDataCheck(self.config.max_gap_days),
            ExtremeReturnCheck(self.config.max_daily_return),
            StalePriceCheck(self.config.max_unchanged_days),
            VolumeAnomalyCheck(),
            WeekendDataCheck(),
            CorporateActionCheck()
        ]
        self.results = {}
    
    def validate_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        auto_fix: bool = True
    ) -> Tuple[pd.DataFrame, QualityReport]:
        """Validate and optionally fix data."""
        
        all_issues = []
        fixed_data = data.copy()
        
        # Run all checks
        for check in self.checks:
            issues = check.check(fixed_data, ticker)
            all_issues.extend(issues)
            
            # Auto-fix if enabled
            if auto_fix and any(i.can_auto_fix for i in issues):
                fixed_data = check.fix(fixed_data, issues)
        
        # Create report
        report = QualityReport(
            ticker=ticker,
            check_date=datetime.now(),
            total_issues=len(all_issues),
            critical_issues=sum(1 for i in all_issues if i.severity == 'critical'),
            warning_issues=sum(1 for i in all_issues if i.severity == 'warning'),
            info_issues=sum(1 for i in all_issues if i.severity == 'info'),
            issues=all_issues,
            data_points_checked=len(data),
            data_points_fixed=len(data) - len(fixed_data)
        )
        
        # Store results
        self.results[ticker] = report
        
        return fixed_data, report
    
    def validate_universe(
        self,
        universe_data: Dict[str, pd.DataFrame],
        parallel: bool = True
    ) -> Dict[str, QualityReport]:
        """Validate entire universe of securities."""
        
        if parallel:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self.validate_data, ticker, data): ticker
                    for ticker, data in universe_data.items()
                }
                
                results = {}
                for future in as_completed(futures):
                    ticker = futures[future]
                    _, report = future.result()
                    results[ticker] = report
                    
            return results
        else:
            return {
                ticker: self.validate_data(ticker, data)[1]
                for ticker, data in universe_data.items()
            }
```

### 3. Quality Reporting

```python
# src/data/quality/reports.py
@dataclass
class QualityReport:
    """Data quality report for a security."""
    ticker: str
    check_date: datetime
    total_issues: int
    critical_issues: int
    warning_issues: int
    info_issues: int
    issues: List[QualityIssue]
    data_points_checked: int
    data_points_fixed: int
    
    @property
    def quality_score(self) -> float:
        """Calculate quality score (0-100)."""
        if self.total_issues == 0:
            return 100.0
        
        # Weight by severity
        weighted_issues = (
            self.critical_issues * 10 +
            self.warning_issues * 3 +
            self.info_issues * 1
        )
        
        # Normalize by data points
        issue_rate = weighted_issues / self.data_points_checked
        
        # Convert to 0-100 score
        return max(0, 100 * (1 - issue_rate))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'check_date': self.check_date.isoformat(),
            'quality_score': self.quality_score,
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues,
            'warning_issues': self.warning_issues,
            'info_issues': self.info_issues,
            'data_points_checked': self.data_points_checked,
            'data_points_fixed': self.data_points_fixed,
            'issues': [
                {
                    'severity': i.severity,
                    'description': i.description,
                    'dates': [d.isoformat() for d in i.affected_dates],
                    'can_auto_fix': i.can_auto_fix
                }
                for i in self.issues[:10]  # Top 10 issues
            ]
        }

class QualityDashboard:
    """Generate quality dashboards and reports."""
    
    def __init__(self, monitor: DataQualityMonitor):
        self.monitor = monitor
    
    def generate_universe_report(self) -> UniverseQualityReport:
        """Generate report for entire universe."""
        reports = list(self.monitor.results.values())
        
        return UniverseQualityReport(
            check_date=datetime.now(),
            total_securities=len(reports),
            average_quality_score=np.mean([r.quality_score for r in reports]),
            securities_with_critical_issues=sum(1 for r in reports if r.critical_issues > 0),
            securities_with_warnings=sum(1 for r in reports if r.warning_issues > 0),
            total_issues_found=sum(r.total_issues for r in reports),
            total_issues_fixed=sum(r.data_points_fixed for r in reports),
            worst_securities=[
                r.ticker for r in sorted(reports, key=lambda x: x.quality_score)[:10]
            ],
            best_securities=[
                r.ticker for r in sorted(reports, key=lambda x: x.quality_score, reverse=True)[:10]
            ]
        )
    
    def create_visual_dashboard(self, output_path: str):
        """Create visual quality dashboard."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Quality score distribution
        scores = [r.quality_score for r in self.monitor.results.values()]
        axes[0, 0].hist(scores, bins=20, edgecolor='black')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Number of Securities')
        axes[0, 0].set_title('Data Quality Score Distribution')
        
        # 2. Issues by severity
        severities = ['critical', 'warning', 'info']
        counts = [
            sum(r.critical_issues for r in self.monitor.results.values()),
            sum(r.warning_issues for r in self.monitor.results.values()),
            sum(r.info_issues for r in self.monitor.results.values())
        ]
        axes[0, 1].bar(severities, counts, color=['red', 'orange', 'blue'])
        axes[0, 1].set_xlabel('Severity')
        axes[0, 1].set_ylabel('Total Issues')
        axes[0, 1].set_title('Issues by Severity')
        
        # 3. Top 10 worst quality
        worst = sorted(self.monitor.results.values(), key=lambda x: x.quality_score)[:10]
        axes[1, 0].barh([r.ticker for r in worst], [r.quality_score for r in worst])
        axes[1, 0].set_xlabel('Quality Score')
        axes[1, 0].set_title('Worst Quality Securities')
        
        # 4. Issue type breakdown
        issue_types = defaultdict(int)
        for report in self.monitor.results.values():
            for issue in report.issues:
                issue_type = issue.description.split(':')[0]
                issue_types[issue_type] += 1
        
        axes[1, 1].pie(issue_types.values(), labels=issue_types.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title('Issues by Type')
        
        plt.tight_layout()
        plt.savefig(output_path)
```

### 4. Integration with Data Layer

```python
# src/data/enhanced_fetcher.py
class QualityAwareDataFetcher:
    """Data fetcher with integrated quality monitoring."""
    
    def __init__(
        self,
        source_router: DataSourceRouter,
        cache_manager: CacheManager,
        quality_monitor: DataQualityMonitor
    ):
        self.router = source_router
        self.cache = cache_manager
        self.monitor = quality_monitor
    
    def fetch_validated_data(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        auto_fix: bool = True,
        require_quality_score: float = 80.0
    ) -> Tuple[pd.DataFrame, QualityReport]:
        """Fetch data with quality validation."""
        
        # Check cache first
        cache_key = f"validated:{ticker}:{start}:{end}"
        if cached := self.cache.get(cache_key):
            return cached
        
        # Fetch raw data
        raw_data = self.router.fetch_with_fallback(
            ticker, "fetch_prices", start, end
        )
        
        # Validate and fix
        validated_data, report = self.monitor.validate_data(
            ticker, raw_data, auto_fix=auto_fix
        )
        
        # Check quality threshold
        if report.quality_score < require_quality_score:
            logger.warning(
                f"{ticker} quality score {report.quality_score:.1f} "
                f"below threshold {require_quality_score}"
            )
            
            if report.critical_issues > 0:
                raise DataQualityError(
                    f"{ticker} has {report.critical_issues} critical issues"
                )
        
        # Cache validated data
        self.cache.set(cache_key, (validated_data, report), ttl=86400)
        
        return validated_data, report
```

### 5. Usage Examples

```python
# Example 1: Single security validation
monitor = DataQualityMonitor()
data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

validated_data, report = monitor.validate_data("AAPL", data)
print(f"Quality Score: {report.quality_score:.1f}")
print(f"Issues Found: {report.total_issues}")
print(f"Critical Issues: {report.critical_issues}")

# Example 2: Universe validation with reporting
universe_data = fetch_universe_data(["AAPL", "MSFT", "GOOGL", ...])
reports = monitor.validate_universe(universe_data)

dashboard = QualityDashboard(monitor)
universe_report = dashboard.generate_universe_report()
dashboard.create_visual_dashboard("quality_report.png")

# Example 3: Integration with optimization
fetcher = QualityAwareDataFetcher(router, cache, monitor)

# Only use data meeting quality threshold
validated_data, report = fetcher.fetch_validated_data(
    "AAPL",
    start_date,
    end_date,
    require_quality_score=90.0
)

# Example 4: Custom quality checks
class CustomSpreadCheck(QualityCheck):
    """Check for unrealistic bid-ask spreads."""
    
    def check(self, data: pd.DataFrame, ticker: str) -> List[QualityIssue]:
        # Implementation
        pass

monitor.checks.append(CustomSpreadCheck())
```

## Implementation Plan

### Day 1: Core Framework
- [ ] Quality check base classes
- [ ] Core quality checks (missing data, extreme returns, stale prices)
- [ ] Quality report structures
- [ ] Unit tests

### Day 2: Advanced Checks
- [ ] Volume anomaly detection
- [ ] Weekend/holiday data checks
- [ ] Corporate action detection
- [ ] Auto-fix capabilities
- [ ] Integration tests

### Day 3: Monitoring & Reporting
- [ ] Quality monitor implementation
- [ ] Parallel validation
- [ ] Dashboard generation
- [ ] Visual reports
- [ ] Performance optimization

### Day 4: Integration & Documentation
- [ ] Integration with data fetchers
- [ ] Cache integration
- [ ] API documentation
- [ ] Usage examples
- [ ] Deployment guide

## Benefits

### Immediate
1. **Catch bad data** before it affects optimization
2. **Auto-fix** common issues (gaps, alignment)
3. **Quality scores** for every security
4. **Alerts** for critical issues

### Long-term
1. **Trust** in data quality
2. **Audit trail** of data issues
3. **Continuous improvement** via monitoring
4. **Extensible** framework for custom checks

## Success Metrics

- Detect 100% of extreme returns (>25% daily)
- Identify 100% of multi-day gaps
- Flag 95%+ of stale price periods
- Quality score >90 for 80%+ of universe
- <1% false positives

## Future Enhancements

1. **ML-based anomaly detection**
2. **Peer comparison** (similar securities)
3. **Historical pattern learning**
4. **Automated reporting emails**
5. **Integration with production monitoring**

The quality monitoring system will ensure the portfolio optimizer always works with clean, validated data.
