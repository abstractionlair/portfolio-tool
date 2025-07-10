"""
Concrete implementations of metric calculators for the modular validation framework.

These implementations provide individual metric calculation components that can be
composed and configured independently.
"""

import numpy as np
from typing import Optional
import logging

from .interfaces import MetricCalculator

logger = logging.getLogger(__name__)


class MSECalculator(MetricCalculator):
    """Mean Squared Error calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((forecasts - actuals) ** 2)
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'mse'


class MAECalculator(MetricCalculator):
    """Mean Absolute Error calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(forecasts - actuals))
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'mae'


class RMSECalculator(MetricCalculator):
    """Root Mean Squared Error calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        mse = np.mean((forecasts - actuals) ** 2)
        return np.sqrt(mse)
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'rmse'


class HitRateCalculator(MetricCalculator):
    """Hit rate calculator (forecast within threshold of actual)."""
    
    def __init__(self, threshold: float = 0.25):
        """Initialize with threshold for hit rate calculation.
        
        Args:
            threshold: Relative error threshold for hit rate (default 25%)
        """
        self.threshold = threshold
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate hit rate (forecast within threshold of actual)."""
        relative_errors = np.abs(forecasts - actuals) / np.maximum(actuals, 1e-8)
        return np.mean(relative_errors < self.threshold)
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'hit_rate'
    
    def validate_inputs(self, forecasts: np.ndarray, actuals: np.ndarray) -> bool:
        """Validate inputs for hit rate calculation."""
        return (super().validate_inputs(forecasts, actuals) and 
                np.all(actuals > 0))  # Hit rate requires positive actuals


class BiasCalculator(MetricCalculator):
    """Bias calculator (mean forecast error)."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate bias (mean forecast error)."""
        return np.mean(forecasts - actuals)
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'bias'


class DirectionalAccuracyCalculator(MetricCalculator):
    """Directional accuracy calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate directional accuracy."""
        if len(forecasts) < 2:
            return np.nan
        
        forecast_changes = np.diff(forecasts) > 0
        actual_changes = np.diff(actuals) > 0
        return np.mean(forecast_changes == actual_changes)
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'directional_accuracy'
    
    def validate_inputs(self, forecasts: np.ndarray, actuals: np.ndarray) -> bool:
        """Validate inputs for directional accuracy calculation."""
        return (super().validate_inputs(forecasts, actuals) and 
                len(forecasts) >= 2)  # Need at least 2 points for direction


class VolatilityForecastCorrelationCalculator(MetricCalculator):
    """Volatility forecast correlation calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate volatility forecast correlation."""
        if len(forecasts) < 2:
            return np.nan
        
        try:
            correlation_matrix = np.corrcoef(forecasts, actuals)
            return correlation_matrix[0, 1]
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return np.nan
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'volatility_forecast_correlation'
    
    def validate_inputs(self, forecasts: np.ndarray, actuals: np.ndarray) -> bool:
        """Validate inputs for correlation calculation."""
        return (super().validate_inputs(forecasts, actuals) and 
                len(forecasts) >= 2 and
                np.var(forecasts) > 0 and np.var(actuals) > 0)  # Need variance for correlation


class RelativeBiasCalculator(MetricCalculator):
    """Relative bias calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate relative bias."""
        if not np.all(actuals > 0):
            return np.nan
        
        return np.mean((forecasts - actuals) / actuals)
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'relative_bias'
    
    def validate_inputs(self, forecasts: np.ndarray, actuals: np.ndarray) -> bool:
        """Validate inputs for relative bias calculation."""
        return (super().validate_inputs(forecasts, actuals) and 
                np.all(actuals > 0))  # Relative bias requires positive actuals


class MAPECalculator(MetricCalculator):
    """Mean Absolute Percentage Error calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        if not np.all(actuals > 0):
            return np.nan
        
        return np.mean(np.abs((forecasts - actuals) / actuals)) * 100
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'mape'
    
    def validate_inputs(self, forecasts: np.ndarray, actuals: np.ndarray) -> bool:
        """Validate inputs for MAPE calculation."""
        return (super().validate_inputs(forecasts, actuals) and 
                np.all(actuals > 0))  # MAPE requires positive actuals


class TheilsUCalculator(MetricCalculator):
    """Theil's U statistic calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        if len(forecasts) < 2:
            return np.nan
        
        try:
            # Theil's U = sqrt(mean((f_t - a_t)^2)) / sqrt(mean(a_t^2))
            numerator = np.sqrt(np.mean((forecasts - actuals) ** 2))
            denominator = np.sqrt(np.mean(actuals ** 2))
            
            if denominator == 0:
                return np.nan
            
            return numerator / denominator
        except Exception as e:
            logger.warning(f"Theil's U calculation failed: {e}")
            return np.nan
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'theils_u'
    
    def validate_inputs(self, forecasts: np.ndarray, actuals: np.ndarray) -> bool:
        """Validate inputs for Theil's U calculation."""
        return (super().validate_inputs(forecasts, actuals) and 
                np.sum(actuals ** 2) > 0)  # Need non-zero sum of squares


class R2Calculator(MetricCalculator):
    """R-squared (coefficient of determination) calculator."""
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate R-squared."""
        if len(forecasts) < 2:
            return np.nan
        
        try:
            ss_res = np.sum((actuals - forecasts) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            
            if ss_tot == 0:
                return np.nan
            
            return 1 - (ss_res / ss_tot)
        except Exception as e:
            logger.warning(f"R-squared calculation failed: {e}")
            return np.nan
    
    def get_name(self) -> str:
        """Get metric name."""
        return 'r_squared'
    
    def validate_inputs(self, forecasts: np.ndarray, actuals: np.ndarray) -> bool:
        """Validate inputs for R-squared calculation."""
        return (super().validate_inputs(forecasts, actuals) and 
                len(forecasts) >= 2 and
                np.var(actuals) > 0)  # Need variance in actuals