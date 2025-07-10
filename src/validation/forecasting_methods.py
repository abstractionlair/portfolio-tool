"""
Concrete implementations of forecasting methods for the modular validation framework.

These implementations migrate the existing forecasting methods to the new
interface-based architecture, enabling dependency injection and modularity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

from .interfaces import ForecastingMethod

logger = logging.getLogger(__name__)


class HistoricalVolatilityMethod(ForecastingMethod):
    """Historical volatility forecasting method."""
    
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        """Generate historical volatility forecast."""
        window = parameters.get('window', min(len(train_data), 20))
        
        if len(train_data) < 2:
            return np.nan
            
        vol = train_data.iloc[-window:].std()
        return vol if np.isfinite(vol) and vol > 0 else np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'window': 20}
    
    def get_method_name(self) -> str:
        """Get the method name."""
        return 'historical'
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for historical method."""
        window = parameters.get('window', 20)
        return isinstance(window, (int, float)) and window > 0


class EWMAVolatilityMethod(ForecastingMethod):
    """Exponentially Weighted Moving Average forecasting method."""
    
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        """Generate EWMA volatility forecast."""
        lambda_param = parameters.get('lambda', 0.94)
        min_periods = min(parameters.get('min_periods', 5), len(train_data) // 2)
        min_periods = max(3, min_periods)
        
        if len(train_data) < min_periods:
            return train_data.std()
        
        try:
            # EWMA calculation
            data_len = min(len(train_data), 50)  # Limit for efficiency
            data_subset = train_data.iloc[-data_len:]
            
            weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(data_subset))])
            weights = weights[::-1]  # Most recent first
            weights = weights / weights.sum()
            
            squared_returns = data_subset.values ** 2
            ewma_variance = np.sum(weights * squared_returns)
            
            if ewma_variance > 0:
                ewma_vol = np.sqrt(ewma_variance)
                return ewma_vol if np.isfinite(ewma_vol) else np.nan
            else:
                return np.nan
        except Exception as e:
            logger.warning(f"EWMA calculation failed: {e}")
            return np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'lambda': 0.94, 'min_periods': 5}
    
    def get_method_name(self) -> str:
        """Get the method name."""
        return 'ewma'
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for EWMA method."""
        lambda_param = parameters.get('lambda', 0.94)
        min_periods = parameters.get('min_periods', 5)
        
        return (isinstance(lambda_param, (int, float)) and 0 < lambda_param < 1 and
                isinstance(min_periods, (int, float)) and min_periods > 0)


class ExponentialSmoothingMethod(ForecastingMethod):
    """Exponential smoothing forecasting method."""
    
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        """Generate exponential smoothing forecast."""
        alpha = parameters.get('alpha', 0.3)
        
        if len(train_data) < 2:
            return train_data.std()
        
        try:
            # Simple exponential smoothing on squared returns
            squared_returns = train_data.values ** 2
            smoothed_var = squared_returns[0]
            
            for i in range(1, len(squared_returns)):
                smoothed_var = alpha * squared_returns[i] + (1 - alpha) * smoothed_var
            
            vol = np.sqrt(smoothed_var)
            return vol if np.isfinite(vol) and vol > 0 else np.nan
        except Exception as e:
            logger.warning(f"Exponential smoothing calculation failed: {e}")
            return np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'alpha': 0.3}
    
    def get_method_name(self) -> str:
        """Get the method name."""
        return 'exponential_smoothing'
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for exponential smoothing method."""
        alpha = parameters.get('alpha', 0.3)
        return isinstance(alpha, (int, float)) and 0 < alpha <= 1


class RobustMADMethod(ForecastingMethod):
    """Robust Median Absolute Deviation forecasting method."""
    
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        """Generate robust MAD forecast."""
        try:
            # Median Absolute Deviation approach
            mad = np.median(np.abs(train_data - np.median(train_data)))
            # Convert MAD to volatility estimate (assuming normal distribution)
            vol = mad * 1.4826
            return vol if np.isfinite(vol) and vol > 0 else np.nan
        except Exception as e:
            logger.warning(f"Robust MAD calculation failed: {e}")
            return np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {}
    
    def get_method_name(self) -> str:
        """Get the method name."""
        return 'robust_mad'
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for robust MAD method."""
        # No parameters to validate for this method
        return True


class QuantileRangeMethod(ForecastingMethod):
    """Quantile range forecasting method."""
    
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        """Generate quantile range forecast."""
        try:
            # Inter-quartile range approach
            q75, q25 = np.percentile(np.abs(train_data), [75, 25])
            vol = (q75 - q25) / 1.349  # Convert to volatility estimate
            return vol if np.isfinite(vol) and vol > 0 else np.nan
        except Exception as e:
            logger.warning(f"Quantile range calculation failed: {e}")
            return np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {}
    
    def get_method_name(self) -> str:
        """Get the method name."""
        return 'quantile_range'
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for quantile range method."""
        # No parameters to validate for this method
        return True