"""
Compatibility adapter for the modular validation framework.

This adapter provides a drop-in replacement for the existing ParameterValidationFramework
while using the new modular architecture under the hood. This enables smooth migration
without breaking existing code.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import logging

from .factory import ValidationFrameworkFactory
from .orchestrator import ValidationOrchestrator
from .interfaces import ValidationConfig
from .parameter_validation import ValidationResult, ValidationMethod

logger = logging.getLogger(__name__)


class CompatibilityValidationFramework:
    """
    Compatibility adapter that provides the same interface as ParameterValidationFramework
    but uses the new modular architecture internally.
    
    This allows existing code to use the new framework without modifications.
    """
    
    def __init__(self, adaptive_mode: bool = True):
        """Initialize the compatibility framework.
        
        Args:
            adaptive_mode: Whether to use adaptive validation (same as original)
        """
        self.adaptive_mode = adaptive_mode
        
        # Create the underlying modular framework
        self._framework = ValidationFrameworkFactory.create_default_framework()
        
        logger.info(f"Initialized CompatibilityValidationFramework with adaptive_mode={adaptive_mode}")
    
    def validate_parameter_combination(
        self, 
        risk_premium_series: pd.Series,
        combination: Dict[str, Any],
        validation_method: ValidationMethod = ValidationMethod.ADAPTIVE
    ) -> ValidationResult:
        """
        Validate a parameter combination using the specified validation method.
        
        This method provides the exact same interface as the original framework.
        
        Args:
            risk_premium_series: Time series of risk premium returns
            combination: Dictionary containing method, parameters, and horizon
            validation_method: Validation method to use
            
        Returns:
            ValidationResult containing all validation metrics
        """
        # Convert ValidationMethod enum to string for the new framework
        if validation_method == ValidationMethod.ADAPTIVE:
            method_str = None  # Use adaptive selection
        else:
            method_str = validation_method.value
        
        # Use the new modular framework
        return self._framework.validate_parameter_combination(
            data=risk_premium_series,
            combination=combination,
            validation_method=method_str
        )
    
    def get_supported_methods(self) -> List[str]:
        """Get list of supported forecasting methods.
        
        Returns:
            List of method names (same interface as original)
        """
        return self._framework.forecasting_service.get_available_methods()
    
    def get_method_defaults(self, method: str) -> Dict[str, Any]:
        """Get default parameters for a forecasting method.
        
        Args:
            method: Name of the forecasting method
            
        Returns:
            Dictionary of default parameters (same interface as original)
        """
        return self._framework.get_method_defaults(method)
    
    @property
    def forecasting_methods(self) -> Dict[str, Any]:
        """Property to access forecasting methods (for compatibility).
        
        This provides access to the underlying forecasting methods in a way
        that's compatible with the original framework's interface.
        """
        # Create a compatibility dict that maps method names to method objects
        methods = {}
        for method_name in self._framework.forecasting_service.get_available_methods():
            # Create a wrapper that provides the original interface
            methods[method_name] = MethodWrapper(
                self._framework.forecasting_service, method_name
            )
        return methods
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Property to access metrics (for compatibility).
        
        This provides access to the underlying metrics in a way that's
        compatible with the original framework's interface.
        """
        # Return metric names mapped to callable wrappers
        return {
            name: MetricWrapper(self._framework.metrics_service, name)
            for name in self._framework.metrics_service.get_available_metrics()
        }


class MethodWrapper:
    """Wrapper to make new forecasting methods compatible with original interface."""
    
    def __init__(self, forecasting_service, method_name: str):
        """Initialize method wrapper.
        
        Args:
            forecasting_service: The new forecasting service
            method_name: Name of the method to wrap
        """
        self.service = forecasting_service
        self.method_name = method_name
    
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        """Generate forecast using the new service."""
        return self.service.forecast(self.method_name, train_data, parameters)
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return self.service.get_method_defaults(self.method_name)


class MetricWrapper:
    """Wrapper to make new metric calculators compatible with original interface."""
    
    def __init__(self, metrics_service, metric_name: str):
        """Initialize metric wrapper.
        
        Args:
            metrics_service: The new metrics service
            metric_name: Name of the metric to wrap
        """
        self.service = metrics_service
        self.metric_name = metric_name
    
    def __call__(self, forecasts, actuals):
        """Calculate metric using the new service."""
        return self.service.calculate_metric(self.metric_name, forecasts, actuals)


def create_legacy_framework(adaptive_mode: bool = True) -> CompatibilityValidationFramework:
    """Factory function to create a compatibility framework.
    
    This function provides a convenient way to create the compatibility framework
    with the same interface as the original framework.
    
    Args:
        adaptive_mode: Whether to use adaptive validation
        
    Returns:
        CompatibilityValidationFramework instance
    """
    return CompatibilityValidationFramework(adaptive_mode=adaptive_mode)


def migrate_to_modular_framework(
    existing_framework: Any,
    preserve_config: bool = True
) -> ValidationOrchestrator:
    """Migrate from existing framework to new modular framework.
    
    This function helps migrate existing code from the old framework to the new one.
    
    Args:
        existing_framework: Instance of the old ParameterValidationFramework
        preserve_config: Whether to try to preserve existing configuration
        
    Returns:
        New ValidationOrchestrator instance
    """
    if preserve_config and hasattr(existing_framework, 'adaptive_mode'):
        # Create configuration based on existing framework
        config = ValidationConfig()
        # Could extract more configuration here if needed
        
        logger.info("Migrating with preserved configuration")
        return ValidationFrameworkFactory.create_default_framework(config)
    else:
        logger.info("Migrating with default configuration")
        return ValidationFrameworkFactory.create_default_framework()


class ModularFrameworkProxy:
    """
    Proxy that automatically redirects calls to the new modular framework.
    
    This can be used to gradually migrate code by replacing imports without
    changing the rest of the code.
    """
    
    def __init__(self):
        """Initialize the proxy."""
        self._framework = None
    
    def __getattr__(self, name):
        """Proxy attribute access to the underlying framework."""
        if self._framework is None:
            self._framework = CompatibilityValidationFramework()
        
        return getattr(self._framework, name)
    
    def __call__(self, *args, **kwargs):
        """Allow the proxy to be called like a constructor."""
        return CompatibilityValidationFramework(*args, **kwargs)