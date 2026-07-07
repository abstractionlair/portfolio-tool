# Task: Add Fund Exposure Mappings to Existing System

**Status**: NOT STARTED  
**Priority**: HIGH  
**Estimated Time**: 1-2 days  
**Approach**: EXTEND EXISTING - Do not create parallel systems

## Current System Inventory

### What Already Exists:

1. **Exposure System**:
   - `src/portfolio/exposures.py` - Has ExposureType enum and Exposure class
   - `src/data/exposure_universe.py` - Defines the exposure universe
   - `config/exposure_universe.yaml` - Configuration of all exposures
   - **This is the foundation - extend it**

2. **Fund Data**:
   - `src/data/market_data.py` - Fetches fund/ETF data
   - Already handles fund returns via yfinance

3. **Return Decomposition**:
   - `src/portfolio/return_replicator.py` - ReturnReplicator class
   - `src/data/return_decomposition.py` - Return decomposition logic
   - **Already can validate if exposures replicate returns!**

4. **Portfolio Classes**:
   - `src/portfolio/portfolio.py` - Portfolio class
   - `src/portfolio/position.py` - Position class with leverage support
   - Already tracks positions and leverage

## What's Missing

The system can work with exposures and funds separately, but lacks the mapping between them. We need to add:
1. A configuration file mapping funds to exposures
2. A way to convert fund holdings to exposure holdings

## Required Changes (MINIMAL)

### 1. Create Fund Exposure Configuration

**New File**: `config/fund_exposures.yaml`

```yaml
# Fund to Exposure Mappings
# Validated using existing ReturnReplicator

funds:
  # Traditional Funds - 100% notional
  VOO:
    name: "Vanguard S&P 500 ETF"
    exposures:
      us_large_equity: 1.0
      
  # Leveraged Funds - Return Stacked
  RSSB:
    name: "Return Stacked Bonds"
    exposures:
      broad_ust: 1.0
      trend_following: 1.0
    leverage_factor: 2.0  # Matches Position.leverage_factor
    
  # Add more funds as needed...
```

### 2. Extend Existing Exposure System

**File**: `src/portfolio/exposures.py`

**Add to existing file**:
```python
class FundExposureMap:
    """Maps funds to their exposure decompositions.
    
    Uses existing ReturnReplicator to validate mappings.
    """
    
    def __init__(self, config_path: str = "config/fund_exposures.yaml"):
        self.mappings = self._load_mappings(config_path)
        self.replicator = ReturnReplicator()  # Use existing!
        
    def get_exposures(self, fund: str) -> Dict[ExposureType, float]:
        """Get exposure breakdown for a fund."""
        # Return exposure mappings from config
        
    def validate_mapping(self, fund: str, start_date, end_date) -> float:
        """Validate using existing ReturnReplicator."""
        # Use self.replicator.calculate_replication_quality()
```

### 3. Add Exposure Calculation to Portfolio

**File**: `src/portfolio/portfolio.py`

**Add method to existing Portfolio class**:
```python
def calculate_exposures(self, fund_exposure_map: FundExposureMap) -> Dict[ExposureType, float]:
    """Calculate aggregate exposures from current positions.
    
    Uses existing position weights and fund mappings.
    """
    exposures = {}
    for position in self.positions.values():
        fund_exposures = fund_exposure_map.get_exposures(position.symbol)
        for exp_type, exp_weight in fund_exposures.items():
            weight = self.get_position_weight(position.symbol)
            exposures[exp_type] = exposures.get(exp_type, 0) + weight * exp_weight
    return exposures
```

### 4. Extend Return Replicator for Validation

**File**: `src/portfolio/return_replicator.py`

The ReturnReplicator already exists and can validate! Just add a convenience method:

```python
def validate_fund_exposures(self, fund: str, exposures: Dict[str, float], 
                          start_date, end_date) -> Dict[str, float]:
    """Validate that stated exposures replicate fund returns.
    
    This mostly wraps existing functionality for the fund use case.
    """
    # Use existing calculate_replication_quality method
```

## What NOT to Do

1. **Don't create new return calculation systems** - ReturnReplicator exists
2. **Don't create new exposure classes** - Extend existing ExposureType/Exposure
3. **Don't create parallel portfolio systems** - Add to existing Portfolio class
4. **Don't reimplement return decomposition** - It already exists!

## Testing Approach

1. **Extend existing tests**:
   - `tests/test_portfolio.py` - Add exposure calculation tests
   - `tests/test_return_replicator.py` - Add fund validation tests

2. **Validate key funds**:
   - VOO → 100% us_large_equity
   - RSSB → 100% broad_ust + 100% trend_following
   - Use existing ReturnReplicator to verify

## Success Criteria

- [ ] fund_exposures.yaml created with 10+ funds mapped
- [ ] FundExposureMap class added to exposures.py
- [ ] Portfolio.calculate_exposures() method works
- [ ] Return replication validates to R² > 0.90 for simple funds
- [ ] Existing tests still pass
- [ ] New tests for exposure calculation pass

## Example Usage After Implementation

```python
# Load fund mappings
fund_map = FundExposureMap()

# Use with existing portfolio
portfolio = Portfolio()
portfolio.add_position(Position("VOO", 100, 300.0))
portfolio.add_position(Position("RSSB", 50, 100.0))

# Calculate exposures using new method
exposures = portfolio.calculate_exposures(fund_map)
# Result: {ExposureType.US_LARGE_EQUITY: 0.75, 
#          ExposureType.BROAD_UST: 0.125,
#          ExposureType.TREND_FOLLOWING: 0.125}

# Validate a fund mapping
quality = fund_map.validate_mapping("RSSB", start_date, end_date)
# Uses existing ReturnReplicator internally
```

This builds directly on the existing codebase rather than creating new systems.
