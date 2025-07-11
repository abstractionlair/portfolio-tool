# Task: Create Fund Exposure Mapping System

**Status**: NOT STARTED  
**Priority**: HIGH  
**Estimated Time**: 2 days  
**Dependencies**: Global forecast horizon implementation

## Overview

Build a comprehensive system for mapping funds/ETFs to their underlying exposure components. This enables the two-stage optimization process: first optimize target exposures, then find the best fund combination to achieve those exposures.

## Current State

- Exposure universe is well-defined (`config/exposure_universe.yaml`)
- Exposure risk estimation works at exposure level
- Portfolio optimizer works at ticker/fund level
- **Missing**: The mapping between funds and their exposure decompositions

## Requirements

### 1. Fund Exposure Database

Create a structured database of funds with their exposure mappings:

```yaml
# config/fund_exposures.yaml
funds:
  # Return Stacked ETFs - 200% notional exposure
  RSSB:
    name: "Return Stacked Bonds"
    description: "100% US Bonds + 100% Managed Futures via futures/swaps"
    exposures:
      broad_ust: 1.0
      trend_following: 1.0
    total_notional: 2.0
    leverage_cost_spread: 0.50  # 50 bps over cash rate
    
  RSST:
    name: "Return Stacked US Stocks"
    description: "100% US Large Cap + 100% Managed Futures"
    exposures:
      us_large_equity: 1.0
      trend_following: 1.0
    total_notional: 2.0
    leverage_cost_spread: 0.50
    
  # PIMCO StocksPLUS Funds - embedded leverage
  PSLDX:
    name: "PIMCO StocksPLUS Long Duration"
    description: "S&P 500 exposure via derivatives + long duration bonds"
    exposures:
      us_large_equity: 1.0
      long_duration_bonds: 0.3  # Approximate - needs validation
    total_notional: 1.3
    leverage_cost_spread: 0.35  # Institutional pricing
    
  # Traditional funds - 100% notional
  VOO:
    name: "Vanguard S&P 500 ETF"
    exposures:
      us_large_equity: 1.0
    total_notional: 1.0
    
  TLT:
    name: "iShares 20+ Year Treasury"
    exposures:
      long_duration_bonds: 1.0
    total_notional: 1.0
```

### 2. Fund Exposure Manager Class

```python
# src/portfolio/fund_exposure_manager.py
class FundExposureManager:
    """Manages the mapping between funds and their exposure decompositions."""
    
    def __init__(self, config_path: str = "config/fund_exposures.yaml"):
        """Load fund exposure mappings from configuration."""
        self.fund_exposures = self._load_fund_exposures(config_path)
        
    def get_fund_exposures(self, ticker: str) -> Dict[str, float]:
        """Get exposure breakdown for a fund."""
        
    def get_fund_leverage(self, ticker: str) -> float:
        """Get total notional exposure (leverage) for a fund."""
        
    def find_funds_with_exposure(self, exposure_id: str) -> List[str]:
        """Find all funds that provide a specific exposure."""
        
    def calculate_portfolio_exposures(
        self, 
        fund_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate aggregate exposures from fund holdings."""
        
    def validate_exposure_mapping(
        self,
        ticker: str,
        returns_data: pd.DataFrame,
        exposure_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Validate that fund returns can be replicated by stated exposures."""
```

### 3. Exposure Replication Validator

```python
# src/portfolio/exposure_replication_validator.py
class ExposureReplicationValidator:
    """Validates fund exposure mappings using return replication."""
    
    def validate_fund_exposures(
        self,
        fund_ticker: str,
        stated_exposures: Dict[str, float],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily"
    ) -> ValidationResult:
        """
        Validate that stated exposures can replicate fund returns.
        
        Uses regression analysis to check if the linear combination
        of exposure returns matches the fund returns.
        """
        
    def find_optimal_exposures(
        self,
        fund_ticker: str,
        candidate_exposures: List[str],
        start_date: datetime,
        end_date: datetime,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Use optimization to find best exposure combination.
        
        Useful for discovering exposures of new funds or
        validating stated exposures.
        """
```

### 4. Two-Stage Portfolio Optimizer

```python
# src/optimization/two_stage_optimizer.py
class TwoStagePortfolioOptimizer:
    """
    Implements two-stage optimization:
    1. Optimize target exposure weights
    2. Find fund combination to achieve target exposures
    """
    
    def __init__(
        self,
        exposure_risk_estimator: ExposureRiskEstimator,
        fund_exposure_manager: FundExposureManager
    ):
        self.risk_estimator = exposure_risk_estimator
        self.fund_manager = fund_exposure_manager
        
    def optimize_exposures(
        self,
        exposure_universe: List[str],
        constraints: ExposureConstraints,
        objective: ObjectiveType
    ) -> Dict[str, float]:
        """
        Stage 1: Find optimal exposure weights.
        
        Uses exposure-level risk model to find optimal combination
        of exposures based on objective (max Sharpe, min vol, etc.)
        """
        
    def find_fund_portfolio(
        self,
        target_exposures: Dict[str, float],
        available_funds: List[str],
        constraints: FundConstraints
    ) -> Dict[str, float]:
        """
        Stage 2: Find fund weights to achieve target exposures.
        
        Minimizes tracking error to target exposures while
        respecting constraints (position limits, costs, etc.)
        """
        
    def optimize_end_to_end(
        self,
        exposure_universe: List[str],
        available_funds: List[str],
        exposure_constraints: ExposureConstraints,
        fund_constraints: FundConstraints,
        objective: ObjectiveType
    ) -> OptimizationResult:
        """Full two-stage optimization with detailed results."""
```

## Implementation Steps

### Phase 1: Fund Exposure Database
1. Create initial `fund_exposures.yaml` with core funds
2. Include leveraged funds (Return Stacked, PIMCO)
3. Add traditional ETFs and mutual funds
4. Document leverage costs and constraints

### Phase 2: Core Classes
1. Implement `FundExposureManager` for loading and querying
2. Create `ExposureReplicationValidator` for validation
3. Build comprehensive test suite

### Phase 3: Two-Stage Optimization
1. Implement exposure-level optimization
2. Add fund selection optimization
3. Create end-to-end workflow
4. Add visualization of results

### Phase 4: Validation and Testing
1. Validate exposure mappings for key funds using historical data
2. Test two-stage optimization with various objectives
3. Compare results with single-stage optimization
4. Document performance and accuracy

## Validation Requirements

1. **Exposure Mapping Accuracy**:
   - R² > 0.90 for return replication
   - Tracking error < 2% annually
   - Stable exposures over time

2. **Optimization Consistency**:
   - Two-stage results should be close to single-stage when possible
   - Respect all constraints in both stages
   - Handle cases where exact exposure match isn't possible

3. **Leverage Handling**:
   - Correctly account for leverage costs
   - Track total notional exposure
   - Ensure risk calculations reflect leverage

## Success Criteria

- [ ] Fund exposure database with 20+ key funds mapped
- [ ] Exposure mappings validated with return replication
- [ ] Two-stage optimization producing sensible portfolios
- [ ] Clear documentation of exposure methodology
- [ ] Integration tests passing
- [ ] Example notebooks demonstrating the system

## Example Usage

```python
# Initialize components
risk_estimator = ExposureRiskEstimator(
    exposure_universe, 
    forecast_horizon=21
)
fund_manager = FundExposureManager()
optimizer = TwoStagePortfolioOptimizer(risk_estimator, fund_manager)

# Define target exposures
target_exposures = {
    'us_large_equity': 0.4,
    'broad_ust': 0.3,
    'trend_following': 0.2,
    'real_estate': 0.1
}

# Find funds to achieve target
fund_weights = optimizer.find_fund_portfolio(
    target_exposures,
    available_funds=['VOO', 'TLT', 'RSST', 'RSSB', 'VNQ'],
    constraints=FundConstraints(max_positions=4)
)
# Might return: {'RSST': 0.2, 'RSSB': 0.3, 'VNQ': 0.1, 'VOO': 0.2}
```

## Notes

- Start with well-known funds where exposures are documented
- Use regression analysis to validate and discover exposures
- Consider time-varying exposures for dynamic funds
- Document methodology for determining leverage costs
- Build incrementally - start with simple funds, add complex ones later

This system enables true exposure-based portfolio optimization while maintaining practical implementation through available funds.
