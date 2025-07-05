# Task: Implement Portfolio Analytics

**Status**: COMPLETED  
**Assigned**: Claude Code  
**Priority**: High  
**Dependencies**: Portfolio, Position, and Exposure classes must exist

## Objective
Implement comprehensive analytics for portfolios including return calculations, risk metrics, and exposure-based performance attribution. This will provide the measurement framework needed for optimization and performance evaluation.

## Requirements

### 1. Return Calculations

Create a `PortfolioAnalytics` class that calculates various return metrics:

```python
class PortfolioAnalytics:
    def __init__(self, portfolio: Portfolio, market_data: MarketDataFetcher):
        self.portfolio = portfolio
        self.market_data = market_data
    
    def calculate_returns(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'daily'  # 'daily', 'monthly', 'annual'
    ) -> pd.Series:
        """Calculate portfolio returns over time."""
    
    def calculate_position_returns(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """Calculate returns for a specific position."""
    
    def time_weighted_return(
        self,
        start_date: datetime,
        end_date: datetime,
        cash_flows: Optional[List[CashFlow]] = None
    ) -> float:
        """Calculate time-weighted return accounting for cash flows."""
```

### 2. Risk Metrics

Implement standard risk measurements:

```python
def calculate_volatility(
    self,
    returns: pd.Series,
    annualize: bool = True
) -> float:
    """Calculate return volatility (standard deviation)."""

def calculate_sharpe_ratio(
    self,
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """Calculate Sharpe ratio."""

def calculate_max_drawdown(self, values: pd.Series) -> Dict[str, Any]:
    """Calculate maximum drawdown and recovery information."""

def calculate_var(
    self,
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'  # or 'parametric'
) -> float:
    """Calculate Value at Risk."""

def calculate_cvar(
    self,
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
```

### 3. Exposure-Based Analytics

Leverage the exposure decomposition system:

```python
def calculate_exposure_returns(
    self,
    fund_map: FundExposureMap,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Calculate returns by exposure type."""

def exposure_attribution(
    self,
    fund_map: FundExposureMap,
    start_date: datetime,
    end_date: datetime
) -> Dict[ExposureType, float]:
    """Attribute portfolio returns to each exposure type."""

def calculate_exposure_correlations(
    self,
    fund_map: FundExposureMap,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Calculate correlation matrix of exposures."""
```

### 4. Performance Metrics

Standard performance measurements:

```python
def calculate_information_ratio(
    self,
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """Calculate information ratio vs benchmark."""

def calculate_tracking_error(
    self,
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """Calculate tracking error vs benchmark."""

def calculate_beta(
    self,
    returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """Calculate portfolio beta."""

def calculate_alpha(
    self,
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """Calculate portfolio alpha (Jensen's alpha)."""
```

### 5. Reporting and Summary

Create comprehensive analytics summaries:

```python
@dataclass
class PortfolioAnalyticsSummary:
    """Summary of portfolio analytics."""
    period_start: datetime
    period_end: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    best_position: str
    worst_position: str
    exposure_returns: Dict[ExposureType, float]

def generate_analytics_summary(
    self,
    start_date: datetime,
    end_date: datetime,
    fund_map: Optional[FundExposureMap] = None
) -> PortfolioAnalyticsSummary:
    """Generate comprehensive analytics summary."""
```

### 6. Historical Analysis

Support for analyzing historical portfolio states:

```python
def analyze_historical_portfolio(
    self,
    historical_positions: List[Dict[datetime, Portfolio]],
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Analyze portfolio performance over time with changing positions."""
```

## Implementation Steps

1. Create `src/portfolio/analytics.py` with PortfolioAnalytics class
2. Implement return calculation methods
3. Add risk metric calculations
4. Integrate with exposure decomposition system
5. Create performance attribution methods
6. Build comprehensive reporting
7. Write thorough tests in `tests/test_analytics.py`
8. Create example script `examples/portfolio_analytics_example.py`

## Test Cases

1. Daily/monthly/annual return calculations
2. Volatility and Sharpe ratio calculations
3. Drawdown analysis with recovery metrics
4. VaR and CVaR calculations
5. Exposure-based return attribution
6. Correlation analysis between exposures
7. Performance vs benchmark metrics
8. Handling of cash flows and position changes

## Key Considerations

- **Handle missing data**: Prices might not be available for all dates
- **Cash flows**: Account for deposits/withdrawals in return calculations
- **Frequency conversion**: Support daily to monthly/annual conversions
- **Exposure changes**: Handle time-varying exposures properly
- **Performance**: Optimize for large portfolios with long histories
- **Numerical stability**: Use appropriate methods for calculations

## Success Criteria

- [x] All return calculations match expected values
- [x] Risk metrics align with standard definitions
- [x] Exposure attribution sums to total portfolio return
- [x] Handles edge cases (empty periods, single position, etc.)
- [x] Performance acceptable for 1000+ position portfolios
- [x] Clear documentation with examples
- [x] Comprehensive test coverage

## Example Usage

```python
# Create analytics object
analytics = PortfolioAnalytics(portfolio, market_data)

# Calculate returns
returns = analytics.calculate_returns(start_date, end_date, 'daily')

# Get risk metrics
volatility = analytics.calculate_volatility(returns)
sharpe = analytics.calculate_sharpe_ratio(returns)
max_dd = analytics.calculate_max_drawdown(returns.cumsum())

# Exposure attribution
fund_map = FundExposureMap('data/fund_universe.yaml')
attribution = analytics.exposure_attribution(fund_map, start_date, end_date)

# Generate summary
summary = analytics.generate_analytics_summary(start_date, end_date, fund_map)
print(f"Total Return: {summary.total_return:.2%}")
print(f"Sharpe Ratio: {summary.sharpe_ratio:.2f}")
```

## Notes

- Start with basic return calculations and build up
- Consider using pandas for time series handling
- Leverage numpy for efficient calculations
- Think about caching for expensive computations
- Consider integration with visualization tools later
