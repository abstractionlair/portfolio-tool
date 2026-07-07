# Task: Implement Correct Compounding Rate Transformations

**Status**: READY  
**Priority**: MEDIUM  
**Type**: Enhancement/Bug Fix  
**Estimated Time**: 2-3 hours
**Dependencies**: Enhanced equity decomposition (COMPLETE)

## Problem Statement

The current implementation uses simple division to convert annual rates to daily rates:
```python
# Current approach (approximation)
daily_rate = annual_rate / 252
```

This should use proper compounding mathematics:
```python
# Correct approach
daily_rate = (1 + annual_rate) ** (1/252) - 1
```

While the differences are small, using correct compounding ensures:
1. Mathematical consistency
2. Accurate long-term compounding
3. Proper handling of higher rates where the approximation error grows

## Impact Analysis

### Current Approximation Errors

For typical rates:
- **3% annual inflation**:
  - Simple division: 3% / 252 = 0.01190%
  - Correct compounding: (1.03)^(1/252) - 1 = 0.01174%
  - Error: 0.00016% daily (1.4% relative error)

- **5% annual risk-free**:
  - Simple division: 5% / 252 = 0.01984%
  - Correct compounding: (1.05)^(1/252) - 1 = 0.01938%
  - Error: 0.00046% daily (2.4% relative error)

- **10% annual rate** (higher rates show larger errors):
  - Simple division: 10% / 252 = 0.03968%
  - Correct compounding: (1.10)^(1/252) - 1 = 0.03797%
  - Error: 0.00171% daily (4.5% relative error)

### Compounding Impact

Over a year, these small daily errors compound:
- Using simple division: (1 + 0.0001190 × 252) = 1.03000 ✓
- Using daily compounding: (1 + 0.0001174)^252 = 1.03000 ✓

Both methods preserve the annual rate, but the paths differ slightly, affecting:
- Volatility calculations
- Risk-adjusted returns
- Component attribution

## Implementation Design

### 1. Create Rate Conversion Utilities

**New File**: `src/data/providers/calculators/rate_converter.py`

```python
"""
Rate conversion utilities for proper compounding calculations.

This module provides mathematically correct conversions between different
rate frequencies, accounting for compounding effects.
"""

import numpy as np
from typing import Union, Literal
import pandas as pd

class RateConverter:
    """Handles conversions between different rate frequencies with proper compounding."""
    
    # Trading days per period
    PERIODS = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4,
        'semi-annual': 2,
        'annual': 1
    }
    
    @classmethod
    def annualize_rate(
        cls,
        rate: Union[float, pd.Series],
        from_frequency: str,
        compounding: bool = True
    ) -> Union[float, pd.Series]:
        """
        Convert a rate from given frequency to annual.
        
        Args:
            rate: The rate to convert (as decimal, e.g., 0.01 for 1%)
            from_frequency: Source frequency ('daily', 'monthly', etc.)
            compounding: If True, use compound interest formula
            
        Returns:
            Annualized rate
        """
        periods = cls.PERIODS.get(from_frequency.lower())
        if periods is None:
            raise ValueError(f"Unknown frequency: {from_frequency}")
        
        if periods == 1:  # Already annual
            return rate
        
        if compounding:
            # Compound: (1 + r)^n - 1
            return (1 + rate) ** periods - 1
        else:
            # Simple: r * n
            return rate * periods
    
    @classmethod
    def convert_rate(
        cls,
        rate: Union[float, pd.Series],
        from_frequency: str,
        to_frequency: str,
        compounding: bool = True
    ) -> Union[float, pd.Series]:
        """
        Convert a rate between any two frequencies.
        
        Args:
            rate: The rate to convert
            from_frequency: Source frequency
            to_frequency: Target frequency
            compounding: If True, use compound interest formula
            
        Returns:
            Rate at target frequency
        """
        from_periods = cls.PERIODS.get(from_frequency.lower())
        to_periods = cls.PERIODS.get(to_frequency.lower())
        
        if from_periods is None or to_periods is None:
            raise ValueError(f"Unknown frequency: {from_frequency} or {to_frequency}")
        
        if from_periods == to_periods:
            return rate
        
        if compounding:
            # First annualize, then convert to target
            # Annual rate = (1 + from_rate)^from_periods - 1
            # Target rate = (1 + annual_rate)^(1/to_periods) - 1
            
            # This simplifies to:
            # target_rate = (1 + from_rate)^(from_periods/to_periods) - 1
            
            exponent = from_periods / to_periods
            return (1 + rate) ** exponent - 1
        else:
            # Simple scaling
            return rate * (from_periods / to_periods)
    
    @classmethod
    def to_daily_rate(
        cls,
        annual_rate: Union[float, pd.Series],
        compounding: bool = True
    ) -> Union[float, pd.Series]:
        """
        Convert annual rate to daily rate.
        
        Args:
            annual_rate: Annual rate (as decimal)
            compounding: If True, use (1+r)^(1/252)-1, else r/252
            
        Returns:
            Daily rate
        """
        return cls.convert_rate(annual_rate, 'annual', 'daily', compounding)
    
    @classmethod
    def to_monthly_rate(
        cls,
        annual_rate: Union[float, pd.Series],
        compounding: bool = True
    ) -> Union[float, pd.Series]:
        """
        Convert annual rate to monthly rate.
        
        Args:
            annual_rate: Annual rate (as decimal)
            compounding: If True, use (1+r)^(1/12)-1, else r/12
            
        Returns:
            Monthly rate
        """
        return cls.convert_rate(annual_rate, 'annual', 'monthly', compounding)
    
    @classmethod
    def compound_returns(
        cls,
        returns: pd.Series,
        from_frequency: str,
        to_frequency: str
    ) -> pd.Series:
        """
        Compound a series of returns from one frequency to another.
        
        This is different from rate conversion - it aggregates actual returns.
        
        Args:
            returns: Series of returns
            from_frequency: Source frequency
            to_frequency: Target frequency (must be lower)
            
        Returns:
            Compounded returns at target frequency
        """
        # Map frequencies to pandas resample rules
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q',
            'annual': 'A'
        }
        
        target_freq = freq_map.get(to_frequency.lower())
        if not target_freq:
            raise ValueError(f"Unknown target frequency: {to_frequency}")
        
        # Compound returns: product of (1 + r) - 1
        return (1 + returns).resample(target_freq).prod() - 1
```

### 2. Update TransformedDataProvider Methods

Update the rate computation methods to use proper compounding:

```python
# In transformed_provider.py

def _compute_inflation_rate(self, start: date, end: date, frequency: str, **kwargs) -> pd.Series:
    """Compute inflation rate from price indices."""
    method = kwargs.get("method", self.config["inflation_method"])
    
    # ... existing code to get inflation data ...
    
    # Convert to appropriate frequency using proper compounding
    if hasattr(self, 'rate_converter'):
        rate_converter = self.rate_converter
    else:
        from .calculators.rate_converter import RateConverter
        rate_converter = RateConverter
    
    # Inflation is typically already computed as YoY (annual rate)
    # Convert to target frequency with compounding
    if frequency.lower() != "annual":
        inflation = rate_converter.convert_rate(
            inflation, 
            from_frequency='annual',
            to_frequency=frequency,
            compounding=True
        )
    
    return self._trim_and_convert(inflation, start, end, frequency, frequency, "rate")

def _compute_nominal_risk_free(self, start: date, end: date, frequency: str, **kwargs) -> pd.Series:
    """Compute nominal risk-free rate."""
    # ... existing code to get risk-free rate ...
    
    # Treasury rates are annualized - convert properly
    if hasattr(self, 'rate_converter'):
        rate_converter = self.rate_converter
    else:
        from .calculators.rate_converter import RateConverter
        rate_converter = RateConverter
    
    if frequency.lower() != "annual":
        risk_free_rate = rate_converter.convert_rate(
            risk_free_rate,
            from_frequency='annual', 
            to_frequency=frequency,
            compounding=True
        )
    
    return self._trim_and_convert(risk_free_rate, start, end, frequency, frequency, "rate")
```

### 3. Add Configuration Option

Allow users to choose between simple and compound conversions:

```python
# In TransformedDataProvider.__init__
self.config = {
    # ... existing config ...
    "use_compound_rate_conversion": True,  # New option
    "rate_conversion_method": "compound"  # 'compound' or 'simple'
}

# Add rate converter instance
from .calculators.rate_converter import RateConverter
self.rate_converter = RateConverter()
```

### 4. Update Return Calculations

Ensure return calculations properly compound when aggregating:

```python
def _aggregate_returns_to_frequency(
    self,
    daily_returns: pd.Series,
    target_frequency: str
) -> pd.Series:
    """Aggregate returns to a lower frequency with proper compounding."""
    if target_frequency.lower() == 'daily':
        return daily_returns
    
    return self.rate_converter.compound_returns(
        daily_returns,
        from_frequency='daily',
        to_frequency=target_frequency
    )
```

### 5. Testing Requirements

**New File**: `tests/data/test_rate_converter.py`

```python
import pytest
import numpy as np
import pandas as pd
from src.data.providers.calculators.rate_converter import RateConverter

class TestRateConverter:
    """Test rate conversion calculations."""
    
    def test_annual_to_daily_conversion(self):
        """Test converting annual rates to daily."""
        # Test cases with known values
        test_cases = [
            (0.03, 0.0001174),   # 3% annual
            (0.05, 0.0001938),   # 5% annual
            (0.10, 0.0003797),   # 10% annual
        ]
        
        for annual_rate, expected_daily in test_cases:
            daily_rate = RateConverter.to_daily_rate(annual_rate, compounding=True)
            assert abs(daily_rate - expected_daily) < 1e-7
            
            # Verify it compounds back correctly
            annual_check = (1 + daily_rate) ** 252 - 1
            assert abs(annual_check - annual_rate) < 1e-10
    
    def test_simple_vs_compound(self):
        """Test difference between simple and compound conversion."""
        annual_rate = 0.05
        
        # Simple division
        daily_simple = RateConverter.to_daily_rate(annual_rate, compounding=False)
        assert abs(daily_simple - 0.05/252) < 1e-10
        
        # Compound conversion
        daily_compound = RateConverter.to_daily_rate(annual_rate, compounding=True)
        assert daily_compound < daily_simple  # Compound should be slightly less
        
    def test_rate_conversion_consistency(self):
        """Test that conversions are consistent across frequencies."""
        annual_rate = 0.06
        
        # Convert annual to daily
        daily = RateConverter.convert_rate(annual_rate, 'annual', 'daily')
        
        # Convert annual to monthly  
        monthly = RateConverter.convert_rate(annual_rate, 'annual', 'monthly')
        
        # Convert monthly to daily
        daily_from_monthly = RateConverter.convert_rate(monthly, 'monthly', 'daily')
        
        # Should be the same
        assert abs(daily - daily_from_monthly) < 1e-10
    
    def test_series_conversion(self):
        """Test conversion works with pandas Series."""
        dates = pd.date_range('2024-01-01', periods=100)
        annual_rates = pd.Series(np.random.uniform(0.02, 0.08, 100), index=dates)
        
        daily_rates = RateConverter.to_daily_rate(annual_rates)
        
        # Check each rate converts correctly
        for i in range(len(annual_rates)):
            expected = (1 + annual_rates.iloc[i]) ** (1/252) - 1
            assert abs(daily_rates.iloc[i] - expected) < 1e-10
```

### 6. Integration Tests

Test the impact on equity decomposition:

```python
def test_decomposition_with_correct_rates(provider):
    """Test that decomposition uses correct rate conversions."""
    # ... setup test data ...
    
    # Configure to use compound conversions
    provider.config['use_compound_rate_conversion'] = True
    
    result = provider.decompose_equity_returns(
        ticker='TEST',
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        earnings_data=earnings,
        frequency='daily'
    )
    
    # Verify rates are properly converted
    # Daily inflation from 3% annual should be ~0.01174%
    avg_daily_inflation = result['inflation'].mean()
    expected_daily_inflation = (1.03) ** (1/252) - 1
    assert abs(avg_daily_inflation - expected_daily_inflation) < 0.0001
```

## Success Criteria

- [ ] RateConverter class implemented with all conversion methods
- [ ] Compound interest formula used for all rate conversions
- [ ] Simple division available as option for backward compatibility
- [ ] All existing tests still pass
- [ ] New tests verify correct compounding mathematics
- [ ] Documentation explains the difference and when to use each method
- [ ] Performance impact is negligible

## Documentation Updates

Add section to documentation explaining:

1. **Why Compounding Matters**
   - Mathematical correctness
   - Consistency with financial theory
   - Impact on long-term calculations

2. **When to Use Each Method**
   - Compound: Default for accuracy
   - Simple: Quick approximations or backward compatibility

3. **Impact on Results**
   - Show examples of differences
   - Explain when differences matter
   - Guide on interpreting results

## Example Usage

```python
from src.data.providers.calculators.rate_converter import RateConverter

# Convert 5% annual to daily
daily_rate = RateConverter.to_daily_rate(0.05)
print(f"5% annual = {daily_rate:.5%} daily (compound)")
print(f"5% annual = {0.05/252:.5%} daily (simple)")

# Convert between any frequencies
monthly_rate = 0.004  # 0.4% monthly
annual_rate = RateConverter.convert_rate(
    monthly_rate, 
    from_frequency='monthly',
    to_frequency='annual'
)
print(f"{monthly_rate:.2%} monthly = {annual_rate:.2%} annual")
```

## Notes

- This change improves mathematical accuracy without breaking existing functionality
- The differences are small for typical rates but grow with higher rates
- Proper compounding is essential for derivatives pricing and risk management
- This brings the implementation in line with financial industry standards
