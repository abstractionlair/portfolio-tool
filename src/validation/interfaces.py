"""
Abstract interfaces for the modular parameter validation framework.

This module defines the core abstractions that enable dependency injection
and modular design in the parameter validation system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .parameter_validation import ValidationResult


@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    adaptive_thresholds: Dict[str, int] = field(default_factory=lambda: {
        'walk_forward': 60,
        'reduced_walk_forward': 35, 
        'simple_holdout': 25
    })
    default_forecasting_method: str = 'historical'
    enabled_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'rmse', 'hit_rate', 'bias', 'directional_accuracy',
        'volatility_forecast_correlation'
    ])
    validation_timeout_seconds: Optional[float] = None
    max_forecasts_per_validation: int = 50


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
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this forecasting method."""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate that parameters are acceptable for this method."""
        return True  # Default implementation accepts all parameters


class ValidationStrategy(ABC):
    """Abstract base class for validation strategies."""
    
    @abstractmethod
    def validate(self, data: pd.Series, combination: Dict[str, Any],
                forecasting_service: 'ForecastingService') -> ValidationResult:
        """Execute validation strategy."""
        pass
        
    @abstractmethod
    def get_required_data_length(self, horizon: int) -> int:
        """Get minimum data length required for this strategy."""
        pass
        
    @abstractmethod  
    def get_strategy_name(self) -> str:
        """Get strategy name for reporting."""
        pass
    
    def can_handle_data_length(self, data_length: int, horizon: int) -> bool:
        """Check if this strategy can handle the given data length."""
        return data_length >= self.get_required_data_length(horizon)


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""
    
    @abstractmethod
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate metric value."""
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass
    
    def validate_inputs(self, forecasts: np.ndarray, actuals: np.ndarray) -> bool:
        """Validate that inputs are suitable for this metric."""
        return (len(forecasts) == len(actuals) and 
                len(forecasts) > 0 and
                np.all(np.isfinite(forecasts)) and 
                np.all(np.isfinite(actuals)))


class ForecastingService:
    """Service for managing and executing forecasting methods."""
    
    def __init__(self, methods: Dict[str, ForecastingMethod]):
        """Initialize with a dictionary of forecasting methods."""
        self.methods = methods
    
    def forecast(self, method_name: str, data: pd.Series, 
                parameters: Dict[str, Any]) -> float:
        """Execute a forecasting method."""
        if method_name not in self.methods:
            raise ValueError(f"Unknown forecasting method: {method_name}")
        
        method = self.methods[method_name]
        
        # Validate parameters
        if not method.validate_parameters(parameters):
            raise ValueError(f"Invalid parameters for method {method_name}: {parameters}")
        
        return method.forecast(data, parameters)
    
    def get_available_methods(self) -> List[str]:
        """Get list of available forecasting methods."""
        return list(self.methods.keys())
        
    def get_method_defaults(self, method_name: str) -> Dict[str, Any]:
        """Get default parameters for a method."""
        if method_name not in self.methods:
            raise ValueError(f"Unknown forecasting method: {method_name}")
        
        return self.methods[method_name].get_parameter_defaults()
    
    def has_method(self, method_name: str) -> bool:
        """Check if a method is available."""
        return method_name in self.methods


class MetricsService:
    """Service for calculating validation metrics."""
    
    def __init__(self, calculators: Dict[str, MetricCalculator]):
        """Initialize with a dictionary of metric calculators."""
        self.calculators = calculators
    
    def calculate_all_metrics(self, forecasts: np.ndarray, 
                            actuals: np.ndarray,
                            enabled_metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate all configured metrics."""
        if enabled_metrics is None:
            enabled_metrics = list(self.calculators.keys())
        
        results = {}
        for metric_name in enabled_metrics:
            if metric_name in self.calculators:
                try:
                    value = self.calculate_metric(metric_name, forecasts, actuals)
                    results[metric_name] = value
                except Exception as e:
                    # Log error but continue with other metrics
                    results[metric_name] = np.nan
        
        return results
        
    def calculate_metric(self, metric_name: str, forecasts: np.ndarray,
                        actuals: np.ndarray) -> float:
        """Calculate a specific metric."""
        if metric_name not in self.calculators:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        calculator = self.calculators[metric_name]
        
        if not calculator.validate_inputs(forecasts, actuals):
            raise ValueError(f"Invalid inputs for metric {metric_name}")
        
        return calculator.calculate(forecasts, actuals)
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        return list(self.calculators.keys())


class AdaptiveStrategySelector:
    """Selects appropriate validation strategy based on data characteristics."""
    
    def __init__(self, config: ValidationConfig):
        """Initialize with configuration."""
        self.config = config
    
    def select_strategy(self, data_length: int, 
                       available_strategies: Dict[str, ValidationStrategy],
                       horizon: int = 21) -> ValidationStrategy:
        """Select best strategy for given data length."""
        thresholds = self.config.adaptive_thresholds
        
        # Check strategies in order of preference (most data first)
        if data_length >= thresholds.get('walk_forward', 60):
            if 'walk_forward' in available_strategies:
                strategy = available_strategies['walk_forward']
                if strategy.can_handle_data_length(data_length, horizon):
                    return strategy
        
        if data_length >= thresholds.get('reduced_walk_forward', 35):
            if 'reduced_walk_forward' in available_strategies:
                strategy = available_strategies['reduced_walk_forward']
                if strategy.can_handle_data_length(data_length, horizon):
                    return strategy
        
        if data_length >= thresholds.get('simple_holdout', 25):
            if 'simple_holdout' in available_strategies:
                strategy = available_strategies['simple_holdout']
                if strategy.can_handle_data_length(data_length, horizon):
                    return strategy
        
        # If no strategy can handle the data, raise an error
        raise ValueError(f"No validation strategy can handle data length {data_length} with horizon {horizon}")
    
    def get_strategy_recommendation(self, data_length: int, horizon: int = 21) -> str:
        """Get recommended strategy name for given data length."""
        thresholds = self.config.adaptive_thresholds
        
        if data_length >= thresholds.get('walk_forward', 60):
            return 'walk_forward'
        elif data_length >= thresholds.get('reduced_walk_forward', 35):
            return 'reduced_walk_forward'
        elif data_length >= thresholds.get('simple_holdout', 25):
            return 'simple_holdout'
        else:
            return 'insufficient_data'