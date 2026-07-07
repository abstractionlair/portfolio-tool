"""
Modular parameter validation framework.

This package provides a modular, extensible framework for validating risk premium
forecasting parameters. The framework uses dependency injection and clean interfaces
to enable focused development and testing of individual components.

Main Components:
- ValidationOrchestrator: Main entry point for validation
- ValidationFrameworkFactory: Factory for easy framework construction
- Interfaces: Abstract base classes for all components
- Forecasting methods: Concrete implementations of forecasting algorithms
- Validation strategies: Concrete implementations of validation approaches
- Metric calculators: Concrete implementations of performance metrics

Quick Start:
    >>> from src.validation import ValidationFrameworkFactory
    >>> framework = ValidationFrameworkFactory.create_default_framework()
    >>> result = framework.validate_parameter_combination(data, combination)
"""

# Core framework components
from .orchestrator import ValidationOrchestrator
from .factory import ValidationFrameworkFactory
from .compatibility_adapter import CompatibilityValidationFramework, create_legacy_framework

# Interface definitions
from .interfaces import (
    ValidationConfig,
    ForecastingMethod,
    ValidationStrategy,
    MetricCalculator,
    ForecastingService,
    MetricsService,
    AdaptiveStrategySelector
)

# Concrete implementations
from .forecasting_methods import (
    HistoricalVolatilityMethod,
    EWMAVolatilityMethod,
    ExponentialSmoothingMethod,
    RobustMADMethod,
    QuantileRangeMethod
)

from .validation_strategies import (
    WalkForwardValidationStrategy,
    ReducedWalkForwardValidationStrategy,
    SimpleHoldoutValidationStrategy
)

from .metric_calculators import (
    MSECalculator,
    MAECalculator,
    RMSECalculator,
    HitRateCalculator,
    BiasCalculator,
    DirectionalAccuracyCalculator,
    VolatilityForecastCorrelationCalculator,
    RelativeBiasCalculator,
    MAPECalculator,
    TheilsUCalculator,
    R2Calculator
)

# Legacy compatibility
from .parameter_validation import (
    ParameterValidationFramework,
    ValidationResult,
    ValidationMethod,
    ForecastingMethod as LegacyForecastingMethod,
    HistoricalMethod,
    EWMAMethod,
    ExponentialSmoothingMethod as LegacyExponentialSmoothingMethod,
    RobustMADMethod as LegacyRobustMADMethod,
    QuantileRangeMethod as LegacyQuantileRangeMethod
)

__all__ = [
    # Core framework
    'ValidationOrchestrator',
    'ValidationFrameworkFactory',
    'CompatibilityValidationFramework',
    'create_legacy_framework',
    
    # Interfaces
    'ValidationConfig',
    'ForecastingMethod',
    'ValidationStrategy', 
    'MetricCalculator',
    'ForecastingService',
    'MetricsService',
    'AdaptiveStrategySelector',
    
    # Forecasting methods
    'HistoricalVolatilityMethod',
    'EWMAVolatilityMethod', 
    'ExponentialSmoothingMethod',
    'RobustMADMethod',
    'QuantileRangeMethod',
    
    # Validation strategies
    'WalkForwardValidationStrategy',
    'ReducedWalkForwardValidationStrategy',
    'SimpleHoldoutValidationStrategy',
    
    # Metric calculators
    'MSECalculator',
    'MAECalculator',
    'RMSECalculator',
    'HitRateCalculator',
    'BiasCalculator',
    'DirectionalAccuracyCalculator',
    'VolatilityForecastCorrelationCalculator',
    'RelativeBiasCalculator',
    'MAPECalculator',
    'TheilsUCalculator',
    'R2Calculator',
    
    # Legacy compatibility
    'ParameterValidationFramework',
    'ValidationResult',
    'ValidationMethod',
    'LegacyForecastingMethod',
    'HistoricalMethod',
    'EWMAMethod',
    'LegacyExponentialSmoothingMethod',
    'LegacyRobustMADMethod',
    'LegacyQuantileRangeMethod'
]