"""
Main orchestrator for the modular parameter validation framework.

This module provides the main coordination logic that brings together all the
modular components (forecasting, validation strategies, metrics) to provide
a unified validation interface.
"""

import pandas as pd
from typing import Dict, Any, Optional
import logging

from .interfaces import (
    ValidationConfig, 
    ForecastingService, 
    MetricsService, 
    AdaptiveStrategySelector,
    ValidationStrategy
)
from .parameter_validation import ValidationResult, ValidationMethod

logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """Main orchestrator that coordinates all validation services."""
    
    def __init__(self, 
                 forecasting_service: ForecastingService,
                 validation_strategies: Dict[str, ValidationStrategy],
                 metrics_service: MetricsService,
                 adaptive_selector: AdaptiveStrategySelector,
                 config: ValidationConfig):
        """Initialize the validation orchestrator.
        
        Args:
            forecasting_service: Service for forecasting methods
            validation_strategies: Dictionary of validation strategies
            metrics_service: Service for calculating metrics
            adaptive_selector: Selector for adaptive strategy selection
            config: Configuration for the framework
        """
        self.forecasting_service = forecasting_service
        self.validation_strategies = validation_strategies
        self.metrics_service = metrics_service
        self.adaptive_selector = adaptive_selector
        self.config = config
        
        logger.info("Initialized ValidationOrchestrator with modular architecture")
    
    def validate_parameter_combination(self, 
                                     data: pd.Series,
                                     combination: Dict[str, Any],
                                     validation_method: Optional[str] = None) -> ValidationResult:
        """Main validation entry point.
        
        Args:
            data: Time series data for validation
            combination: Parameter combination to validate (method, parameters, horizon)
            validation_method: Specific validation method to use, or None for adaptive
            
        Returns:
            ValidationResult containing all validation metrics
        """
        clean_data = data.dropna()
        total_periods = len(clean_data)
        horizon = combination.get('horizon', 21)
        
        logger.debug(f"Validating combination: {combination['method']} with {total_periods} periods")
        
        try:
            # Select validation strategy
            if validation_method is None or validation_method == 'adaptive':
                strategy = self.adaptive_selector.select_strategy(
                    total_periods, self.validation_strategies, horizon
                )
            else:
                if validation_method not in self.validation_strategies:
                    raise ValueError(f"Unknown validation method: {validation_method}")
                strategy = self.validation_strategies[validation_method]
            
            # Check if strategy can handle the data
            if not strategy.can_handle_data_length(total_periods, horizon):
                return ValidationResult(
                    success=False,
                    validation_method=strategy.get_strategy_name(),
                    num_forecasts=0,
                    mse=float('nan'), mae=float('nan'), rmse=float('nan'), hit_rate=float('nan'),
                    bias=float('nan'), directional_accuracy=float('nan'),
                    volatility_forecast_correlation=float('nan'),
                    forecast_mean=float('nan'), forecast_std=float('nan'),
                    actual_mean=float('nan'), actual_std=float('nan'),
                    forecast_actual_correlation=float('nan'), relative_bias=float('nan'),
                    error_message=f'Insufficient data: {total_periods} periods, need {strategy.get_required_data_length(horizon)}'
                )
            
            # Execute validation
            result = strategy.validate(clean_data, combination, self.forecasting_service)
            
            # Enhance result with additional metrics if needed
            if result.success and hasattr(result, 'num_forecasts') and result.num_forecasts > 0:
                # The validation strategies already calculate metrics, but we could enhance here if needed
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return ValidationResult(
                success=False,
                validation_method=validation_method or 'unknown',
                num_forecasts=0,
                mse=float('nan'), mae=float('nan'), rmse=float('nan'), hit_rate=float('nan'),
                bias=float('nan'), directional_accuracy=float('nan'),
                volatility_forecast_correlation=float('nan'),
                forecast_mean=float('nan'), forecast_std=float('nan'),
                actual_mean=float('nan'), actual_std=float('nan'),
                forecast_actual_correlation=float('nan'), relative_bias=float('nan'),
                error_message=str(e)
            )
    
    def get_supported_methods(self) -> Dict[str, Any]:
        """Get information about supported methods and strategies.
        
        Returns:
            Dictionary containing supported forecasting methods and validation strategies
        """
        return {
            'forecasting_methods': self.forecasting_service.get_available_methods(),
            'validation_strategies': list(self.validation_strategies.keys()),
            'metrics': self.metrics_service.get_available_metrics(),
            'config': {
                'adaptive_thresholds': self.config.adaptive_thresholds,
                'default_forecasting_method': self.config.default_forecasting_method,
                'enabled_metrics': self.config.enabled_metrics
            }
        }
    
    def get_method_defaults(self, method: str) -> Dict[str, Any]:
        """Get default parameters for a forecasting method.
        
        Args:
            method: Name of the forecasting method
            
        Returns:
            Dictionary of default parameters
        """
        return self.forecasting_service.get_method_defaults(method)
    
    def get_strategy_recommendation(self, data_length: int, horizon: int = 21) -> str:
        """Get recommended validation strategy for given data characteristics.
        
        Args:
            data_length: Length of available data
            horizon: Forecasting horizon
            
        Returns:
            Recommended strategy name
        """
        return self.adaptive_selector.get_strategy_recommendation(data_length, horizon)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current configuration and return diagnostics.
        
        Returns:
            Dictionary containing configuration validation results
        """
        diagnostics = {
            'forecasting_service': {
                'available_methods': len(self.forecasting_service.get_available_methods()),
                'methods': self.forecasting_service.get_available_methods()
            },
            'validation_strategies': {
                'available_strategies': len(self.validation_strategies),
                'strategies': list(self.validation_strategies.keys())
            },
            'metrics_service': {
                'available_metrics': len(self.metrics_service.get_available_metrics()),
                'metrics': self.metrics_service.get_available_metrics()
            },
            'config': {
                'adaptive_thresholds': self.config.adaptive_thresholds,
                'default_method': self.config.default_forecasting_method,
                'enabled_metrics': self.config.enabled_metrics
            },
            'validation_status': 'valid'
        }
        
        # Check for potential issues
        issues = []
        
        if len(self.forecasting_service.get_available_methods()) == 0:
            issues.append("No forecasting methods available")
        
        if len(self.validation_strategies) == 0:
            issues.append("No validation strategies available")
        
        if self.config.default_forecasting_method not in self.forecasting_service.get_available_methods():
            issues.append(f"Default forecasting method '{self.config.default_forecasting_method}' not available")
        
        if issues:
            diagnostics['validation_status'] = 'issues_found'
            diagnostics['issues'] = issues
        
        return diagnostics