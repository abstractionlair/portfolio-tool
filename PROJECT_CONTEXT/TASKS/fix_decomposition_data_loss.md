# Task: Fix Return Decomposition Data Loss with Modular Filling Strategy

**Status**: TODO  
**Priority**: CRITICAL BLOCKER - Must fix before optimization  
**Estimated Time**: 2-3 hours  
**Dependencies**: None - Critical fix

## Overview

The return decomposition process is losing ~42% of data due to aggressive `.dropna()`. We'll implement a modular filling strategy that:
1. Uses forward-fill as the immediate fix
2. Provides a clean interface to swap in better methods later
3. Dramatically improves data retention from 58% to >95%

## Implementation Plan

### Phase 1: Create Modular Filling Interface (30 minutes)

Create new file: `/src/data/alignment_strategies.py`

```python
"""
Modular strategies for aligning FRED data with market returns.

This module provides different strategies for handling missing FRED data
(inflation, risk-free rates) when aligning with market returns.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AlignmentStrategy(ABC):
    """Base class for data alignment strategies."""
    
    @abstractmethod
    def align(self, 
              market_data: pd.Series, 
              fred_data: pd.Series,
              max_lag: Optional[int] = None) -> pd.Series:
        """Align FRED data with market data timeline.
        
        Args:
            market_data: Market returns with desired index
            fred_data: FRED data to align
            max_lag: Maximum number of periods to look back
            
        Returns:
            FRED data aligned to market data index
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name for logging."""
        pass


class ForwardFillStrategy(AlignmentStrategy):
    """Simple forward-fill strategy - use last known value."""
    
    def __init__(self, limit: Optional[int] = None):
        """Initialize forward-fill strategy.
        
        Args:
            limit: Maximum number of periods to forward-fill (None = no limit)
        """
        self.limit = limit
    
    def align(self, 
              market_data: pd.Series, 
              fred_data: pd.Series,
              max_lag: Optional[int] = None) -> pd.Series:
        """Align using forward-fill."""
        # Use limit if specified
        limit = max_lag or self.limit
        
        # Reindex to market data timeline
        aligned = fred_data.reindex(market_data.index)
        
        # Forward fill
        aligned = aligned.fillna(method='ffill', limit=limit)
        
        # If still NaN at beginning, backfill
        aligned = aligned.fillna(method='bfill', limit=5)  # Only backfill up to 5 periods
        
        # Log statistics
        original_count = fred_data.notna().sum()
        final_count = aligned.notna().sum()
        market_count = len(market_data)
        
        logger.info(f"ForwardFillStrategy: {original_count} -> {final_count} "
                   f"data points ({final_count/market_count*100:.1f}% coverage)")
        
        return aligned
    
    def get_name(self) -> str:
        return f"ForwardFill(limit={self.limit})"


class InterpolationStrategy(AlignmentStrategy):
    """Linear interpolation between known values."""
    
    def __init__(self, method: str = 'linear', limit: Optional[int] = None):
        """Initialize interpolation strategy.
        
        Args:
            method: Interpolation method ('linear', 'time', 'cubic')
            limit: Maximum gap size to interpolate
        """
        self.method = method
        self.limit = limit
    
    def align(self, 
              market_data: pd.Series, 
              fred_data: pd.Series,
              max_lag: Optional[int] = None) -> pd.Series:
        """Align using interpolation."""
        # Reindex to market data timeline
        aligned = fred_data.reindex(market_data.index)
        
        # Interpolate
        aligned = aligned.interpolate(method=self.method, limit=self.limit)
        
        # Forward fill any remaining (especially at the end)
        aligned = aligned.fillna(method='ffill', limit=max_lag)
        
        return aligned
    
    def get_name(self) -> str:
        return f"Interpolation(method={self.method}, limit={self.limit})"


class SmartFillStrategy(AlignmentStrategy):
    """Intelligent filling based on data characteristics."""
    
    def __init__(self, 
                 inflation_target: float = 0.02,
                 rf_rate_default: float = 0.02):
        """Initialize smart fill strategy.
        
        Args:
            inflation_target: Annual inflation target for fallback
            rf_rate_default: Default risk-free rate for fallback
        """
        self.inflation_target = inflation_target
        self.rf_rate_default = rf_rate_default
    
    def align(self, 
              market_data: pd.Series, 
              fred_data: pd.Series,
              max_lag: Optional[int] = None) -> pd.Series:
        """Align using intelligent filling based on series characteristics."""
        # This is a placeholder for future enhancement
        # For now, use forward fill with reasonable defaults
        
        # Reindex to market data timeline
        aligned = fred_data.reindex(market_data.index)
        
        # Forward fill up to 45 days (reasonable for monthly FRED data)
        aligned = aligned.fillna(method='ffill', limit=max_lag or 45)
        
        # For any remaining NaN, use smart defaults
        if aligned.isna().any():
            # Detect if this is inflation or rate data based on typical values
            mean_value = fred_data.mean()
            
            if mean_value < 0.01:  # Likely inflation (monthly)
                # Use monthly inflation target
                fill_value = self.inflation_target / 12
            else:  # Likely interest rate
                fill_value = self.rf_rate_default / 12
            
            aligned = aligned.fillna(fill_value)
            
            logger.info(f"SmartFillStrategy: Used default value {fill_value:.4f} "
                       f"for {aligned.isna().sum()} missing points")
        
        return aligned
    
    def get_name(self) -> str:
        return "SmartFill"


class AlignmentStrategyFactory:
    """Factory for creating alignment strategies."""
    
    _strategies = {
        'forward_fill': ForwardFillStrategy,
        'interpolation': InterpolationStrategy,
        'smart_fill': SmartFillStrategy
    }
    
    @classmethod
    def create(cls, strategy_name: str = 'forward_fill', **kwargs) -> AlignmentStrategy:
        """Create an alignment strategy.
        
        Args:
            strategy_name: Name of strategy to create
            **kwargs: Arguments passed to strategy constructor
            
        Returns:
            AlignmentStrategy instance
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. "
                           f"Available: {list(cls._strategies.keys())}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a new strategy type."""
        cls._strategies[name] = strategy_class
```

### Phase 2: Update ReturnDecomposer (1 hour)

Modify `/src/data/return_decomposition.py`:

```python
# Add import at top
from .alignment_strategies import AlignmentStrategy, AlignmentStrategyFactory

# Update __init__ to accept strategy
def __init__(self,
             fred_fetcher: Optional[FREDDataFetcher] = None,
             total_return_fetcher: Optional[TotalReturnFetcher] = None,
             alignment_strategy: Optional[AlignmentStrategy] = None):
    """Initialize the return decomposer.
    
    Args:
        fred_fetcher: FREDDataFetcher instance
        total_return_fetcher: TotalReturnFetcher instance
        alignment_strategy: Strategy for aligning FRED data with returns
    """
    self.fred_fetcher = fred_fetcher or FREDDataFetcher()
    self.total_return_fetcher = total_return_fetcher or TotalReturnFetcher()
    
    # Use forward-fill as default strategy
    self.alignment_strategy = alignment_strategy or AlignmentStrategyFactory.create('forward_fill')
    
    # Cache for decomposition results
    self._decomposition_cache = {}

# Update decompose_returns method
def decompose_returns(self, ...) -> pd.DataFrame:
    """Decompose returns into inflation + real risk-free rate + spread."""
    # ... existing code until Step 1 ...
    
    # Step 1: Get inflation rates
    inflation_rates_raw = self.fred_fetcher.get_inflation_rates_for_returns(
        start_date, end_date, frequency, inflation_series
    )
    
    # Align inflation data with returns using strategy
    if not inflation_rates_raw.empty:
        inflation_rates = self.alignment_strategy.align(returns, inflation_rates_raw)
        logger.info(f"Aligned inflation data using {self.alignment_strategy.get_name()}")
    else:
        logger.warning("No inflation data available for decomposition")
        inflation_rates = pd.Series(0, index=returns.index, name='inflation')
    
    # Step 2: Get nominal risk-free rates
    nominal_rf_rates_raw = self.fred_fetcher.fetch_risk_free_rate(
        start_date, end_date, risk_free_maturity, frequency
    )
    
    # Align risk-free rate data using strategy
    if not nominal_rf_rates_raw.empty:
        nominal_rf_rates = self.alignment_strategy.align(returns, nominal_rf_rates_raw)
        logger.info(f"Aligned risk-free rate data using {self.alignment_strategy.get_name()}")
    else:
        logger.warning("No risk-free rate data available for decomposition")
        nominal_rf_rates = pd.Series(0.02, index=returns.index, name='nominal_rf')
    
    # ... rest of existing code ...
    
    # Step 5: Align all series
    decomposition_df = pd.DataFrame({
        'total_return': returns,
        'inflation': inflation_rates,  # Now pre-aligned
        'nominal_rf_rate': rf_returns,  # Now pre-aligned
        'real_rf_rate': real_rf_rates   # Calculated from aligned data
    })
    
    # Only drop rows where return data itself is missing
    decomposition_df = decomposition_df.dropna(subset=['total_return'])
    
    # Log data retention statistics
    initial_count = len(returns)
    final_count = len(decomposition_df)
    retention_rate = final_count / initial_count * 100
    
    logger.info(f"Data retention: {final_count}/{initial_count} ({retention_rate:.1f}%)")
    
    if retention_rate < 90:
        logger.warning(f"Low data retention rate: {retention_rate:.1f}%. "
                      "Consider adjusting alignment strategy.")
    
    # ... rest of existing code ...
```

### Phase 3: Add Configuration Support (30 minutes)

Create `/config/decomposition_config.yaml`:

```yaml
# Return Decomposition Configuration

# Alignment strategy for FRED data
alignment:
  # Strategy name: forward_fill, interpolation, smart_fill
  strategy: forward_fill
  
  # Strategy-specific parameters
  parameters:
    # For forward_fill
    limit: null  # null means no limit
    
    # For interpolation (when implemented)
    method: linear
    
    # For smart_fill (when implemented)
    inflation_target: 0.02
    rf_rate_default: 0.02

# Data quality thresholds
quality:
  min_retention_rate: 0.90  # Warn if data retention < 90%
  max_forward_fill_days: 45  # Maximum days to forward-fill
```

### Phase 4: Update Risk Premium Estimator (30 minutes)

Update initialization to use configurable alignment:

```python
def __init__(self, universe: ExposureUniverse, decomposer: Optional[ReturnDecomposer] = None):
    """Initialize with optional custom decomposer."""
    self.universe = universe
    
    if decomposer is None:
        # Load alignment configuration
        config_path = Path('config/decomposition_config.yaml')
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            strategy_name = config['alignment']['strategy']
            strategy_params = config['alignment']['parameters']
            
            alignment_strategy = AlignmentStrategyFactory.create(
                strategy_name, **strategy_params
            )
            
            decomposer = ReturnDecomposer(alignment_strategy=alignment_strategy)
        else:
            # Use default decomposer with forward-fill
            decomposer = ReturnDecomposer()
    
    self.decomposer = decomposer
```

### Phase 5: Create Tests (30 minutes)

Create `/tests/test_alignment_strategies.py`:

```python
"""Test alignment strategies for FRED data."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.alignment_strategies import (
    ForwardFillStrategy, InterpolationStrategy, 
    AlignmentStrategyFactory
)


class TestForwardFillStrategy:
    """Test forward-fill alignment strategy."""
    
    def test_basic_forward_fill(self):
        """Test basic forward-fill functionality."""
        # Create market data (daily)
        market_dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        market_data = pd.Series(
            np.random.normal(0, 0.01, len(market_dates)),
            index=market_dates
        )
        
        # Create FRED data (sparse, monthly)
        fred_dates = pd.date_range('2024-01-01', '2024-01-31', freq='MS')
        fred_data = pd.Series([0.002, 0.003], index=fred_dates[:2])
        
        # Apply strategy
        strategy = ForwardFillStrategy()
        aligned = strategy.align(market_data, fred_data)
        
        # Check results
        assert len(aligned) == len(market_data)
        assert aligned.notna().all()  # No missing values
        assert aligned.iloc[0] == 0.002  # First value
        assert aligned.iloc[15] == 0.002  # Forward-filled
    
    def test_retention_improvement(self):
        """Test that forward-fill improves data retention."""
        # This tests the actual problem we're solving
        # ... test implementation ...


def test_factory():
    """Test strategy factory."""
    strategy = AlignmentStrategyFactory.create('forward_fill', limit=30)
    assert isinstance(strategy, ForwardFillStrategy)
    assert strategy.limit == 30
```

## Verification Steps

After implementation:

1. **Test Data Retention**:
   ```python
   # Run decomposition and check retention
   decomposer = ReturnDecomposer()
   result = decomposer.decompose_returns(returns, ...)
   
   retention_rate = len(result) / len(returns)
   print(f"Data retention: {retention_rate:.1%}")  # Should be >95%
   ```

2. **Compare Strategies**:
   ```python
   # Test different strategies
   strategies = ['forward_fill', 'interpolation', 'smart_fill']
   for strategy_name in strategies:
       strategy = AlignmentStrategyFactory.create(strategy_name)
       # ... compare results
   ```

3. **Run Optimization Test**:
   - Verify optimization now has sufficient data
   - Check that results are more stable

## Success Criteria

- [ ] Data retention improves from ~58% to >95%
- [ ] Modular design allows easy strategy swapping
- [ ] Tests pass showing alignment works correctly
- [ ] Configuration file controls strategy selection
- [ ] Logging shows clear retention statistics

## Future Enhancements

The modular design allows future improvements:
- Kalman filter for missing data
- Seasonal adjustment for inflation
- Market-based proxies for risk-free rate
- Machine learning imputation methods

## Priority

**CRITICAL**: This blocks the production optimization. Implement immediately.
