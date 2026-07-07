"""
Concrete implementations of validation strategies for the modular validation framework.

These implementations migrate the existing validation strategies to the new
interface-based architecture, enabling dependency injection and modularity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

from .interfaces import ValidationStrategy, ForecastingService
from .parameter_validation import ValidationResult

logger = logging.getLogger(__name__)


class WalkForwardValidationStrategy(ValidationStrategy):
    """Walk-forward validation strategy for longer time series."""
    
    def __init__(self, min_train_periods: int = 40, max_test_points: int = 15):
        """Initialize walk-forward validation strategy.
        
        Args:
            min_train_periods: Minimum training periods required
            max_test_points: Maximum test points to use
        """
        self.min_train_periods = min_train_periods
        self.max_test_points = max_test_points
    
    def validate(self, data: pd.Series, combination: Dict[str, Any],
                forecasting_service: ForecastingService) -> ValidationResult:
        """Execute walk-forward validation strategy."""
        method = combination['method']
        parameters = combination['parameters']
        horizon = combination.get('horizon', 21)
        
        test_points = min(self.max_test_points, len(data) - self.min_train_periods - horizon)
        
        forecasts = []
        actuals = []
        
        for i in range(self.min_train_periods, self.min_train_periods + test_points):
            train_data = data.iloc[:i]
            
            try:
                forecast = forecasting_service.forecast(method, train_data, parameters)
            except Exception as e:
                logger.warning(f"Forecast failed at step {i}: {e}")
                continue
            
            # Actual value at horizon
            if i + horizon < len(data):
                actual = abs(data.iloc[i + horizon])
            else:
                # For insufficient data, scale by horizon to reflect the longer prediction period
                actual = abs(data.iloc[i]) * np.sqrt(horizon / 21)  # Normalize to 21-day base
            
            if np.isfinite(forecast) and np.isfinite(actual) and forecast > 0 and actual > 0:
                forecasts.append(forecast)
                actuals.append(actual)
        
        if len(forecasts) < 3:
            return ValidationResult(
                success=False,
                validation_method="walk_forward",
                num_forecasts=len(forecasts),
                mse=np.nan, mae=np.nan, rmse=np.nan, hit_rate=np.nan,
                bias=np.nan, directional_accuracy=np.nan,
                volatility_forecast_correlation=np.nan,
                forecast_mean=np.nan, forecast_std=np.nan,
                actual_mean=np.nan, actual_std=np.nan,
                forecast_actual_correlation=np.nan, relative_bias=np.nan,
                error_message=f'Insufficient valid forecasts in walk-forward: {len(forecasts)}'
            )
        
        return self._calculate_all_metrics(np.array(forecasts), np.array(actuals), 'walk_forward')
    
    def get_required_data_length(self, horizon: int) -> int:
        """Get minimum data length required for this strategy."""
        return self.min_train_periods + 3  # Need at least min_train + some test points
    
    def get_strategy_name(self) -> str:
        """Get strategy name for reporting."""
        return 'walk_forward'
    
    def _calculate_all_metrics(self, forecasts: np.ndarray, actuals: np.ndarray, validation_type: str) -> ValidationResult:
        """Calculate all validation metrics."""
        return ValidationResult(
            success=True,
            validation_method=validation_type,
            num_forecasts=len(forecasts),
            mse=np.mean((forecasts - actuals) ** 2),
            mae=np.mean(np.abs(forecasts - actuals)),
            rmse=np.sqrt(np.mean((forecasts - actuals) ** 2)),
            hit_rate=np.mean(np.abs(forecasts - actuals) / np.maximum(actuals, 1e-8) < 0.25),
            bias=np.mean(forecasts - actuals),
            directional_accuracy=np.mean((np.diff(forecasts) > 0) == (np.diff(actuals) > 0)) if len(forecasts) > 1 else np.nan,
            volatility_forecast_correlation=np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan,
            forecast_mean=np.mean(forecasts),
            forecast_std=np.std(forecasts),
            actual_mean=np.mean(actuals),
            actual_std=np.std(actuals),
            forecast_actual_correlation=np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan,
            relative_bias=np.mean((forecasts - actuals) / actuals) if np.all(actuals > 0) else np.nan
        )


class ReducedWalkForwardValidationStrategy(ValidationStrategy):
    """Reduced walk-forward validation strategy for medium-length time series."""
    
    def __init__(self, min_train_periods: int = 25, max_test_points: int = 8):
        """Initialize reduced walk-forward validation strategy.
        
        Args:
            min_train_periods: Minimum training periods required
            max_test_points: Maximum test points to use
        """
        self.min_train_periods = min_train_periods
        self.max_test_points = max_test_points
    
    def validate(self, data: pd.Series, combination: Dict[str, Any],
                forecasting_service: ForecastingService) -> ValidationResult:
        """Execute reduced walk-forward validation strategy."""
        method = combination['method']
        parameters = combination['parameters']
        horizon = combination.get('horizon', 21)
        
        test_points = min(self.max_test_points, len(data) - self.min_train_periods)
        
        forecasts = []
        actuals = []
        
        for i in range(self.min_train_periods, self.min_train_periods + test_points):
            train_data = data.iloc[:i]
            
            try:
                forecast = forecasting_service.forecast(method, train_data, parameters)
            except Exception as e:
                logger.warning(f"Forecast failed at step {i}: {e}")
                continue
            
            # Actual value at horizon
            if i + horizon < len(data):
                actual = abs(data.iloc[i + horizon])
            else:
                # For insufficient data, scale by horizon to reflect the longer prediction period
                actual = abs(data.iloc[i]) * np.sqrt(horizon / 21)  # Normalize to 21-day base
            
            if np.isfinite(forecast) and np.isfinite(actual) and forecast > 0 and actual > 0:
                forecasts.append(forecast)
                actuals.append(actual)
        
        if len(forecasts) < 3:
            return ValidationResult(
                success=False,
                validation_method="reduced_walk_forward",
                num_forecasts=len(forecasts),
                mse=np.nan, mae=np.nan, rmse=np.nan, hit_rate=np.nan,
                bias=np.nan, directional_accuracy=np.nan,
                volatility_forecast_correlation=np.nan,
                forecast_mean=np.nan, forecast_std=np.nan,
                actual_mean=np.nan, actual_std=np.nan,
                forecast_actual_correlation=np.nan, relative_bias=np.nan,
                error_message=f'Insufficient valid forecasts in reduced walk-forward: {len(forecasts)}'
            )
        
        return self._calculate_all_metrics(np.array(forecasts), np.array(actuals), 'reduced_walk_forward')
    
    def get_required_data_length(self, horizon: int) -> int:
        """Get minimum data length required for this strategy."""
        return self.min_train_periods + 3  # Need at least min_train + some test points
    
    def get_strategy_name(self) -> str:
        """Get strategy name for reporting."""
        return 'reduced_walk_forward'
    
    def _calculate_all_metrics(self, forecasts: np.ndarray, actuals: np.ndarray, validation_type: str) -> ValidationResult:
        """Calculate all validation metrics."""
        return ValidationResult(
            success=True,
            validation_method=validation_type,
            num_forecasts=len(forecasts),
            mse=np.mean((forecasts - actuals) ** 2),
            mae=np.mean(np.abs(forecasts - actuals)),
            rmse=np.sqrt(np.mean((forecasts - actuals) ** 2)),
            hit_rate=np.mean(np.abs(forecasts - actuals) / np.maximum(actuals, 1e-8) < 0.25),
            bias=np.mean(forecasts - actuals),
            directional_accuracy=np.mean((np.diff(forecasts) > 0) == (np.diff(actuals) > 0)) if len(forecasts) > 1 else np.nan,
            volatility_forecast_correlation=np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan,
            forecast_mean=np.mean(forecasts),
            forecast_std=np.std(forecasts),
            actual_mean=np.mean(actuals),
            actual_std=np.std(actuals),
            forecast_actual_correlation=np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan,
            relative_bias=np.mean((forecasts - actuals) / actuals) if np.all(actuals > 0) else np.nan
        )


class SimpleHoldoutValidationStrategy(ValidationStrategy):
    """Simple holdout validation strategy for shorter time series."""
    
    def __init__(self, test_ratio: float = 0.2, min_test_size: int = 5):
        """Initialize simple holdout validation strategy.
        
        Args:
            test_ratio: Ratio of data to use for testing
            min_test_size: Minimum test size in periods
        """
        self.test_ratio = test_ratio
        self.min_test_size = min_test_size
    
    def validate(self, data: pd.Series, combination: Dict[str, Any],
                forecasting_service: ForecastingService) -> ValidationResult:
        """Execute simple holdout validation strategy."""
        method = combination['method']
        parameters = combination['parameters']
        horizon = combination.get('horizon', 21)
        
        # Use last 20% for testing, minimum 5 periods
        test_size = max(self.min_test_size, int(len(data) * self.test_ratio))
        train_size = len(data) - test_size
        
        forecasts = []
        actuals = []
        
        for i in range(test_size):
            if train_size + i >= len(data):
                break
                
            train_data = data.iloc[:train_size + i] if i > 0 else data.iloc[:train_size]
            
            try:
                forecast = forecasting_service.forecast(method, train_data, parameters)
            except Exception as e:
                logger.warning(f"Forecast failed at step {i}: {e}")
                continue
            
            # Actual volatility over horizon
            if train_size + i + horizon <= len(data):
                actual_returns = data.iloc[train_size + i:train_size + i + horizon]
                if len(actual_returns) > 1:
                    # Compute realized volatility over the horizon period
                    # Scale by the square root of the horizon for proper horizon-dependent validation
                    actual = actual_returns.std() * np.sqrt(len(actual_returns))
                else:
                    actual = abs(data.iloc[train_size + i])
            else:
                # For insufficient data, use available data up to series end but scale by horizon
                available_end = len(data)
                actual_returns = data.iloc[train_size + i:available_end]
                if len(actual_returns) > 1:
                    # Scale by the expected horizon, not just the available data
                    actual = actual_returns.std() * np.sqrt(horizon)
                else:
                    # If only one data point, use absolute value scaled by horizon
                    actual = abs(data.iloc[train_size + i]) * np.sqrt(horizon / 21)  # Normalize to 21-day base
            
            if np.isfinite(forecast) and np.isfinite(actual) and forecast > 0 and actual > 0:
                forecasts.append(forecast)
                actuals.append(actual)
        
        if len(forecasts) < 2:
            return ValidationResult(
                success=False,
                validation_method="simple_holdout",
                num_forecasts=len(forecasts),
                mse=np.nan, mae=np.nan, rmse=np.nan, hit_rate=np.nan,
                bias=np.nan, directional_accuracy=np.nan,
                volatility_forecast_correlation=np.nan,
                forecast_mean=np.nan, forecast_std=np.nan,
                actual_mean=np.nan, actual_std=np.nan,
                forecast_actual_correlation=np.nan, relative_bias=np.nan,
                error_message=f'Insufficient valid forecasts in holdout: {len(forecasts)}'
            )
        
        return self._calculate_all_metrics(np.array(forecasts), np.array(actuals), 'simple_holdout')
    
    def get_required_data_length(self, horizon: int) -> int:
        """Get minimum data length required for this strategy."""
        return int(self.min_test_size / self.test_ratio)  # Just need enough for train/test split
    
    def get_strategy_name(self) -> str:
        """Get strategy name for reporting."""
        return 'simple_holdout'
    
    def _calculate_all_metrics(self, forecasts: np.ndarray, actuals: np.ndarray, validation_type: str) -> ValidationResult:
        """Calculate all validation metrics."""
        return ValidationResult(
            success=True,
            validation_method=validation_type,
            num_forecasts=len(forecasts),
            mse=np.mean((forecasts - actuals) ** 2),
            mae=np.mean(np.abs(forecasts - actuals)),
            rmse=np.sqrt(np.mean((forecasts - actuals) ** 2)),
            hit_rate=np.mean(np.abs(forecasts - actuals) / np.maximum(actuals, 1e-8) < 0.25),
            bias=np.mean(forecasts - actuals),
            directional_accuracy=np.mean((np.diff(forecasts) > 0) == (np.diff(actuals) > 0)) if len(forecasts) > 1 else np.nan,
            volatility_forecast_correlation=np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan,
            forecast_mean=np.mean(forecasts),
            forecast_std=np.std(forecasts),
            actual_mean=np.mean(actuals),
            actual_std=np.std(actuals),
            forecast_actual_correlation=np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan,
            relative_bias=np.mean((forecasts - actuals) / actuals) if np.all(actuals > 0) else np.nan
        )