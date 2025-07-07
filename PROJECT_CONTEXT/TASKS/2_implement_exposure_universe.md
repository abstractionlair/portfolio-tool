# Task 2: Implement ExposureUniverse Data Integration

**Status**: TODO  
**Priority**: HIGH - Core infrastructure  
**Estimated Time**: 3-4 hours  
**Dependencies**: Task 1 (Rate Series Support)

## Context
We have an exposure universe configuration at `/config/exposure_universe.yaml` but no code to load and use it. The TotalReturnFetcher needs to work seamlessly with the ExposureUniverse to fetch returns for all exposures.

## Problem
- The ExposureUniverse class is referenced but not implemented
- We need a clean way to load the YAML config
- Each exposure has multiple implementation options (fallbacks)
- Some exposures are composites of multiple tickers

## Requirements

### 1. Create ExposureUniverse Class
Location: `/src/data/exposure_universe.py`

```python
class ExposureUniverse:
    """Manages the universe of exposures and their implementations."""
    
    def __init__(self):
        self.exposures = {}
        self.categories = {}
        
    def load_config(self, config_path: str):
        """Load exposure universe from YAML configuration."""
        
    def get_exposure(self, exposure_id: str) -> dict:
        """Get exposure configuration by ID."""
        
    def list_exposures(self, category: Optional[str] = None) -> List[str]:
        """List all exposure IDs, optionally filtered by category."""
        
    def get_implementation_details(self, exposure_id: str) -> dict:
        """Get implementation details for an exposure."""
```

### 2. Implementation Details

#### YAML Loading
```python
def load_config(self, config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse categories and exposures
    for category in config['categories']:
        category_id = category['id']
        self.categories[category_id] = category
        
        for exposure in category['exposures']:
            exposure['category'] = category_id
            self.exposures[exposure['id']] = exposure
```

#### Handle Different Implementation Types
- `single_ticker`: Simple ticker fetch
- `etf_average`: Average returns of multiple ETFs
- `rate_series`: FRED rate data (requires Task 1)
- `composite`: Weighted combination of exposures

### 3. Update TotalReturnFetcher Integration
Location: `/src/data/total_returns.py`

Add method:
```python
def fetch_universe_returns(
    self,
    universe: ExposureUniverse,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily"
) -> Dict[str, dict]:
    """Fetch returns for all exposures in universe.
    
    Returns:
        Dict mapping exposure_id to {
            'returns': pd.Series,
            'success': bool,
            'implementation': str (which implementation was used),
            'error': Optional[str]
        }
    """
```

### 4. Handle Composite Exposures
Some exposures might be weighted combinations:
```python
def fetch_composite_returns(
    self,
    components: List[dict],
    start_date: datetime,
    end_date: datetime,
    frequency: str = "daily"
) -> pd.Series:
    """Fetch and combine returns for composite exposures.
    
    Args:
        components: List of {'ticker': str, 'weight': float}
    """
```

### 5. Add Tests
Location: `/tests/test_exposure_universe.py`

```python
def test_load_config():
    """Test loading exposure universe configuration."""
    
def test_get_exposure():
    """Test retrieving individual exposures."""
    
def test_fetch_universe_returns():
    """Test fetching returns for entire universe."""
    
def test_composite_exposure():
    """Test composite exposure calculations."""
```

## Testing Instructions

1. Test loading the config:
```python
universe = ExposureUniverse()
universe.load_config('/Users/scottmcguire/portfolio-tool/config/exposure_universe.yaml')
print(f"Loaded {len(universe.exposures)} exposures")
print(f"Categories: {list(universe.categories.keys())}")
```

2. Test fetching all returns:
```python
fetcher = TotalReturnFetcher()
results = fetcher.fetch_universe_returns(
    universe,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    frequency="monthly"
)

# Check success rate
successful = sum(1 for r in results.values() if r['success'])
print(f"Successfully fetched {successful}/{len(results)} exposures")
```

3. Test specific exposures:
```python
# Test each implementation type
test_exposures = [
    'us_large_cap',      # single_ticker
    'cash_rate',         # rate_series
    'trend_following',   # etf_average with mutual funds
]

for exp_id in test_exposures:
    exposure = universe.get_exposure(exp_id)
    print(f"\n{exp_id}: {exposure['name']}")
    print(f"  Implementations: {[impl['type'] for impl in exposure['implementations']]}")
```

## Success Criteria
- [ ] ExposureUniverse class created and working
- [ ] YAML config loads successfully
- [ ] All 16 exposures accessible via get_exposure()
- [ ] fetch_universe_returns returns data for most exposures
- [ ] Composite exposures calculated correctly
- [ ] Clear error messages for failed fetches
- [ ] Tests cover all implementation types
- [ ] Integration with return decomposition works

## Notes
- Start simple - get single ticker working first
- Use existing mutual fund fallback lists from the YAML
- Log which implementation was used for each exposure
- Some mutual funds might not be available in yfinance - that's OK, just log it
- The config structure is already well-designed, just implement it faithfully
