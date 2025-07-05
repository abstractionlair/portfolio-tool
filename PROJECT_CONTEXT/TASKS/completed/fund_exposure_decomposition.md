# Task: Implement Fund Exposure Decomposition System

**Status**: COMPLETED  
**Assigned**: Claude Code  
**Priority**: High (should be done after basic Portfolio structures)  
**Dependencies**: Portfolio and Position classes must exist first  

## Objective
Implement a system to decompose complex funds (especially internally leveraged funds) into their underlying exposures. This is critical for accurate portfolio analysis and optimization.

## Background
Many funds in the investment universe provide exposure to multiple asset classes or strategies:
- **RSSB**: 100% Stocks + 100% Bonds (200% notional)
- **RSST**: 100% Stocks + 50% Managed Futures (150% notional)
- **PSLDX**: ~100% Stocks + ~100% Long Duration Bonds
- **SPY**: 100% US Equity (simple case)

The optimizer needs to work with true exposures, not just fund positions.

## Requirements

### 1. Fund Definition Format
Create a standardized format for defining fund exposures:

```yaml
# Example: fund_exposures.yaml
RSSB:
  name: "Return Stacked US Stocks & Bonds ETF"
  exposures:
    US_EQUITY: 1.0
    US_BONDS: 1.0
  total_notional: 2.0
  
RSST:
  name: "Return Stacked US Stocks & Managed Futures ETF"
  exposures:
    US_EQUITY: 1.0
    MANAGED_FUTURES: 0.5
  total_notional: 1.5

PSLDX:
  name: "PIMCO StocksPLUS Long Duration Fund"
  exposures:
    US_EQUITY: 1.0
    LONG_DURATION_BONDS: 1.0
  total_notional: 2.0
```

### 2. Core Classes

#### ExposureType (Enum)
Define standard exposure types:
- US_EQUITY
- INTL_EQUITY
- EM_EQUITY
- US_BONDS
- INTL_BONDS
- LONG_DURATION_BONDS
- COMMODITIES
- MANAGED_FUTURES
- TREND_FOLLOWING
- CARRY
- VALUE_FACTOR
- MOMENTUM_FACTOR
- QUALITY_FACTOR
- LOW_VOL_FACTOR

#### Exposure
```python
@dataclass
class Exposure:
    exposure_type: ExposureType
    amount: float  # Can be negative for short exposures
    
    def scale(self, factor: float) -> 'Exposure':
        """Scale exposure by a factor"""
```

#### FundDefinition
```python
@dataclass
class FundDefinition:
    symbol: str
    name: str
    exposures: Dict[ExposureType, float]
    total_notional: float
    
    def get_exposures(self, position_value: float) -> List[Exposure]:
        """Convert position value to list of exposures"""
```

#### FundExposureMap
```python
class FundExposureMap:
    def __init__(self, definitions_path: str):
        """Load fund definitions from YAML/JSON file"""
    
    def get_fund_definition(self, symbol: str) -> Optional[FundDefinition]:
        """Get definition for a specific fund"""
    
    def add_fund_definition(self, definition: FundDefinition):
        """Add or update a fund definition"""
```

#### ExposureCalculator
```python
class ExposureCalculator:
    def __init__(self, fund_map: FundExposureMap):
        self.fund_map = fund_map
    
    def calculate_position_exposures(self, position: Position, current_price: float) -> List[Exposure]:
        """Calculate exposures for a single position"""
    
    def calculate_portfolio_exposures(self, portfolio: Portfolio, prices: Dict[str, float]) -> Dict[ExposureType, float]:
        """Calculate total exposures for entire portfolio"""
```

### 3. Return Replication Validator

Implement a system to validate exposure assumptions:

```python
class ReturnReplicator:
    def __init__(self, market_data_fetcher):
        self.data_fetcher = market_data_fetcher
    
    def validate_fund_exposures(
        self, 
        fund_symbol: str,
        fund_definition: FundDefinition,
        replication_symbols: Dict[ExposureType, str],
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Validate that fund returns can be replicated using stated exposures
        Returns metrics including R-squared, tracking error, etc.
        """
```

### 4. Integration Points

Update existing classes:

```python
# In Position class
def get_exposures(self, fund_map: FundExposureMap, current_price: float) -> List[Exposure]:
    """Get underlying exposures for this position"""

# In Portfolio class  
def calculate_total_exposures(self, fund_map: FundExposureMap, prices: Dict[str, float]) -> Dict[ExposureType, float]:
    """Calculate aggregate exposures across all positions"""
```

## Implementation Steps

1. Define ExposureType enum with all relevant exposure categories
2. Create Exposure and FundDefinition data classes
3. Implement FundExposureMap to load/save definitions
4. Build ExposureCalculator for position and portfolio calculations
5. Create initial fund_exposures.yaml with known funds
6. Implement ReturnReplicator for validation
7. Add integration methods to Position and Portfolio classes
8. Write comprehensive tests
9. Create example showing exposure decomposition

## Test Cases

1. Simple fund (SPY) returns single exposure
2. Return Stacked fund correctly shows multiple exposures
3. Portfolio with mixed funds shows aggregate exposures
4. Return replication achieves high R-squared for known funds
5. Handle missing fund definitions gracefully
6. Exposure calculations scale with position size
7. Short positions create negative exposures

## Success Criteria

- [x] Fund definitions can be loaded from YAML/JSON
- [x] Position exposures calculated correctly
- [x] Portfolio aggregate exposures accurate
- [x] Return replication validates known funds (framework implemented)
- [x] Clear documentation and examples
- [x] All tests passing
- [x] Integration with existing Portfolio classes

## Notes

- Start with a small set of well-understood funds
- Consider versioning fund definitions as they may change over time
- Think about how to handle funds with dynamic exposures
- May need to support time-varying exposures in the future
