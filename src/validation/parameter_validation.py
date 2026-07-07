"""
Parameter Validation Framework for Risk Premium Forecasting.

This module provides adaptive validation that adjusts to data availability while
maintaining statistical rigor. It supports different validation approaches based 
on data constraints and provides comprehensive metrics for forecasting accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Enumeration of available validation methods."""
    WALK_FORWARD = "walk_forward"
    REDUCED_WALK_FORWARD = "reduced_walk_forward"
    SIMPLE_HOLDOUT = "simple_holdout"
    ADAPTIVE = "adaptive"


@dataclass
class ValidationResult:
    """Container for validation results."""
    success: bool
    validation_method: str
    num_forecasts: int
    mse: float
    mae: float
    rmse: float
    hit_rate: float
    bias: float
    directional_accuracy: float
    volatility_forecast_correlation: float
    forecast_mean: float
    forecast_std: float
    actual_mean: float
    actual_std: float
    forecast_actual_correlation: float
    relative_bias: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'validation_method': self.validation_method,
            'num_forecasts': self.num_forecasts,
            'mse': self.mse,
            'mae': self.mae,
            'rmse': self.rmse,
            'hit_rate': self.hit_rate,
            'bias': self.bias,
            'directional_accuracy': self.directional_accuracy,
            'volatility_forecast_correlation': self.volatility_forecast_correlation,
            'forecast_mean': self.forecast_mean,
            'forecast_std': self.forecast_std,
            'actual_mean': self.actual_mean,
            'actual_std': self.actual_std,
            'forecast_actual_correlation': self.forecast_actual_correlation,
            'relative_bias': self.relative_bias,
            'error_message': self.error_message
        }


class ForecastingMethod(ABC):
    """Abstract base class for forecasting methods."""
    
    @abstractmethod
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        """Generate a volatility forecast from training data."""
        pass
    
    @abstractmethod
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters for this method."""
        pass


class HistoricalMethod(ForecastingMethod):
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


class EWMAMethod(ForecastingMethod):
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
        except Exception:
            return np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'lambda': 0.94, 'min_periods': 5}


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
        except Exception:
            return np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'alpha': 0.3}


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
        except Exception:
            return np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {}


class QuantileRangeMethod(ForecastingMethod):
    """Quantile range forecasting method."""
    
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        """Generate quantile range forecast."""
        try:
            # Inter-quartile range approach
            q75, q25 = np.percentile(np.abs(train_data), [75, 25])
            vol = (q75 - q25) / 1.349  # Convert to volatility estimate
            return vol if np.isfinite(vol) and vol > 0 else np.nan
        except Exception:
            return np.nan
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {}


class ParameterValidationFramework:
    """
    Adaptive parameter validation framework that adjusts validation approach
    based on data availability while maintaining statistical rigor.
    """
    
    def __init__(self, adaptive_mode: bool = True):
        """Initialize the validation framework."""
        self.adaptive_mode = adaptive_mode
        
        # Register forecasting methods
        self.forecasting_methods = {
            'historical': HistoricalMethod(),
            'ewma': EWMAMethod(),
            'exponential_smoothing': ExponentialSmoothingMethod(),
            'robust_mad': RobustMADMethod(),
            'quantile_range': QuantileRangeMethod()
        }
        
        # Metric calculation functions
        self.metrics = {
            'mse': self._calculate_mse,
            'mae': self._calculate_mae,
            'rmse': self._calculate_rmse,
            'hit_rate': self._calculate_hit_rate,
            'bias': self._calculate_bias,
            'directional_accuracy': self._calculate_directional_accuracy,
            'volatility_forecast_correlation': self._calculate_vol_forecast_corr
        }
        
        logger.info(f"Initialized ParameterValidationFramework with adaptive_mode={adaptive_mode}")
    
    def validate_parameter_combination(
        self, 
        risk_premium_series: pd.Series,
        combination: Dict[str, Any],
        validation_method: ValidationMethod = ValidationMethod.ADAPTIVE
    ) -> ValidationResult:
        """
        Validate a parameter combination using the specified validation method.
        
        Args:
            risk_premium_series: Time series of risk premium returns
            combination: Dictionary containing method, parameters, and horizon
            validation_method: Validation method to use
            
        Returns:
            ValidationResult containing all validation metrics
        """
        clean_series = risk_premium_series.dropna()
        total_periods = len(clean_series)
        
        logger.debug(f"Validating combination: {combination['method']} with {total_periods} periods")
        
        # Determine validation approach
        if validation_method == ValidationMethod.ADAPTIVE:
            if total_periods >= 60:
                method = ValidationMethod.WALK_FORWARD
            elif total_periods >= 35:
                method = ValidationMethod.REDUCED_WALK_FORWARD
            elif total_periods >= 25:
                method = ValidationMethod.SIMPLE_HOLDOUT
            else:
                return ValidationResult(
                    success=False,
                    validation_method="insufficient_data",
                    num_forecasts=0,
                    mse=np.nan, mae=np.nan, rmse=np.nan, hit_rate=np.nan,
                    bias=np.nan, directional_accuracy=np.nan,
                    volatility_forecast_correlation=np.nan,
                    forecast_mean=np.nan, forecast_std=np.nan,
                    actual_mean=np.nan, actual_std=np.nan,
                    forecast_actual_correlation=np.nan, relative_bias=np.nan,
                    error_message=f'Insufficient data: {total_periods} periods (minimum 25 required)'
                )
        else:
            method = validation_method
        
        # Execute validation
        try:
            if method == ValidationMethod.WALK_FORWARD:
                return self._walk_forward_validation(clean_series, combination)
            elif method == ValidationMethod.REDUCED_WALK_FORWARD:
                return self._reduced_walk_forward_validation(clean_series, combination)
            elif method == ValidationMethod.SIMPLE_HOLDOUT:
                return self._simple_holdout_validation(clean_series, combination)
            else:
                raise ValueError(f"Unknown validation method: {method}")
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return ValidationResult(
                success=False,
                validation_method=method.value,
                num_forecasts=0,
                mse=np.nan, mae=np.nan, rmse=np.nan, hit_rate=np.nan,
                bias=np.nan, directional_accuracy=np.nan,
                volatility_forecast_correlation=np.nan,
                forecast_mean=np.nan, forecast_std=np.nan,
                actual_mean=np.nan, actual_std=np.nan,
                forecast_actual_correlation=np.nan, relative_bias=np.nan,
                error_message=str(e)
            )
    
    def _walk_forward_validation(self, series: pd.Series, combination: Dict) -> ValidationResult:
        """Full walk-forward validation for longer series."""
        method = combination['method']
        parameters = combination['parameters']
        horizon = combination.get('horizon', 21)
        
        min_train = 40
        test_points = min(15, len(series) - min_train - horizon)
        
        forecasts = []
        actuals = []
        
        for i in range(min_train, min_train + test_points):
            train_data = series.iloc[:i]
            forecast = self._generate_forecast(train_data, method, parameters)
            
            # Actual value at horizon
            if i + horizon < len(series):
                actual = abs(series.iloc[i + horizon])
            else:
                # For insufficient data, scale by horizon to reflect the longer prediction period
                actual = abs(series.iloc[i]) * np.sqrt(horizon / 21)  # Normalize to 21-day base
            
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
    
    def _reduced_walk_forward_validation(self, series: pd.Series, combination: Dict) -> ValidationResult:
        """Reduced walk-forward for medium-length series."""
        method = combination['method']
        parameters = combination['parameters']
        horizon = combination.get('horizon', 21)
        
        min_train = 25
        test_points = min(8, len(series) - min_train)
        
        forecasts = []
        actuals = []
        
        for i in range(min_train, min_train + test_points):
            train_data = series.iloc[:i]
            forecast = self._generate_forecast(train_data, method, parameters)
            
            # Actual value at horizon
            if i + horizon < len(series):
                actual = abs(series.iloc[i + horizon])
            else:
                # For insufficient data, scale by horizon to reflect the longer prediction period
                actual = abs(series.iloc[i]) * np.sqrt(horizon / 21)  # Normalize to 21-day base
            
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
    
    def _simple_holdout_validation(self, series: pd.Series, combination: Dict) -> ValidationResult:
        """Simple holdout for shorter series."""
        method = combination['method']
        parameters = combination['parameters']
        horizon = combination.get('horizon', 21)
        
        # Use last 20% for testing, minimum 5 periods
        test_size = max(5, int(len(series) * 0.2))
        train_size = len(series) - test_size
        
        forecasts = []
        actuals = []
        
        for i in range(test_size):
            if train_size + i >= len(series):
                break
                
            train_data = series.iloc[:train_size + i] if i > 0 else series.iloc[:train_size]
            forecast = self._generate_forecast(train_data, method, parameters)
            
            # Actual volatility over horizon
            if train_size + i + horizon <= len(series):
                actual_returns = series.iloc[train_size + i:train_size + i + horizon]
                if len(actual_returns) > 1:
                    # Compute realized volatility over the horizon period
                    # Scale by the square root of the horizon for proper horizon-dependent validation
                    actual = actual_returns.std() * np.sqrt(len(actual_returns))
                else:
                    actual = abs(series.iloc[train_size + i])
            else:
                # For insufficient data, use available data up to series end but scale by horizon
                available_end = len(series)
                actual_returns = series.iloc[train_size + i:available_end]
                if len(actual_returns) > 1:
                    # Scale by the expected horizon, not just the available data
                    actual = actual_returns.std() * np.sqrt(horizon)
                else:
                    # If only one data point, use absolute value scaled by horizon
                    actual = abs(series.iloc[train_size + i]) * np.sqrt(horizon / 21)  # Normalize to 21-day base
            
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
    
    def _generate_forecast(self, train_data: pd.Series, method: str, parameters: Dict) -> float:
        """Generate volatility forecast using specified method."""
        if method not in self.forecasting_methods:
            logger.warning(f"Unknown forecasting method: {method}, using historical")
            method = 'historical'
        
        forecasting_method = self.forecasting_methods[method]
        return forecasting_method.forecast(train_data, parameters)
    
    def _calculate_all_metrics(self, forecasts: np.ndarray, actuals: np.ndarray, validation_type: str) -> ValidationResult:
        """Calculate all validation metrics."""
        results = {'validation_type': validation_type}
        
        # Calculate each metric
        for metric_name, metric_func in self.metrics.items():
            try:
                value = metric_func(forecasts, actuals)
                results[metric_name] = value if np.isfinite(value) else np.nan
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_name}: {str(e)}")
                results[metric_name] = np.nan
        
        # Calculate additional statistics
        results.update({
            'num_forecasts': len(forecasts),
            'forecast_mean': np.mean(forecasts),
            'forecast_std': np.std(forecasts),
            'actual_mean': np.mean(actuals),
            'actual_std': np.std(actuals),
            'forecast_actual_correlation': np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan,
            'relative_bias': np.mean((forecasts - actuals) / actuals) if np.all(actuals > 0) else np.nan
        })
        
        return ValidationResult(
            success=True,
            validation_method=validation_type,
            num_forecasts=results['num_forecasts'],
            mse=results['mse'],
            mae=results['mae'],
            rmse=results['rmse'],
            hit_rate=results['hit_rate'],
            bias=results['bias'],
            directional_accuracy=results['directional_accuracy'],
            volatility_forecast_correlation=results['volatility_forecast_correlation'],
            forecast_mean=results['forecast_mean'],
            forecast_std=results['forecast_std'],
            actual_mean=results['actual_mean'],
            actual_std=results['actual_std'],
            forecast_actual_correlation=results['forecast_actual_correlation'],
            relative_bias=results['relative_bias']
        )
    
    def _calculate_mse(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((forecasts - actuals) ** 2)
    
    def _calculate_mae(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(forecasts - actuals))
    
    def _calculate_rmse(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(self._calculate_mse(forecasts, actuals))
    
    def _calculate_hit_rate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate hit rate (forecast within 25% of actual)."""
        relative_errors = np.abs(forecasts - actuals) / np.maximum(actuals, 1e-8)
        return np.mean(relative_errors < 0.25)
    
    def _calculate_bias(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate bias (mean forecast error)."""
        return np.mean(forecasts - actuals)
    
    def _calculate_directional_accuracy(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate directional accuracy."""
        if len(forecasts) < 2:
            return np.nan
        
        forecast_changes = np.diff(forecasts) > 0
        actual_changes = np.diff(actuals) > 0
        return np.mean(forecast_changes == actual_changes)
    
    def _calculate_vol_forecast_corr(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate volatility forecast correlation."""
        return np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan
    
    def get_supported_methods(self) -> List[str]:
        """Get list of supported forecasting methods."""
        return list(self.forecasting_methods.keys())
    
    def get_method_defaults(self, method: str) -> Dict[str, Any]:
        """Get default parameters for a forecasting method."""
        if method not in self.forecasting_methods:
            raise ValueError(f"Unknown forecasting method: {method}")
        return self.forecasting_methods[method].get_parameter_defaults()