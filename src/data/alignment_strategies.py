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
        aligned = aligned.ffill(limit=limit)
        
        # If still NaN at beginning, backfill
        aligned = aligned.bfill(limit=5)  # Only backfill up to 5 periods
        
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
        aligned = aligned.ffill(limit=max_lag)
        
        # Log statistics
        original_count = fred_data.notna().sum()
        final_count = aligned.notna().sum()
        market_count = len(market_data)
        
        logger.info(f"InterpolationStrategy: {original_count} -> {final_count} "
                   f"data points ({final_count/market_count*100:.1f}% coverage)")
        
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
        # Reindex to market data timeline
        aligned = fred_data.reindex(market_data.index)
        
        # Forward fill up to 45 days (reasonable for monthly FRED data)
        aligned = aligned.ffill(limit=max_lag or 45)
        
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
        
        # Log statistics
        original_count = fred_data.notna().sum()
        final_count = aligned.notna().sum()
        market_count = len(market_data)
        
        logger.info(f"SmartFillStrategy: {original_count} -> {final_count} "
                   f"data points ({final_count/market_count*100:.1f}% coverage)")
        
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