"""
Factory for creating configured validation framework instances.

This module provides factory methods to easily construct the validation framework
with default or custom configurations, enabling easy setup and dependency injection.
"""

from typing import Dict, Optional, List
import logging

from .interfaces import (
    ValidationConfig, 
    ForecastingService, 
    MetricsService, 
    AdaptiveStrategySelector,
    ForecastingMethod,
    ValidationStrategy,
    MetricCalculator
)
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
from .orchestrator import ValidationOrchestrator

logger = logging.getLogger(__name__)


class ValidationFrameworkFactory:
    """Factory for creating configured validation framework instances."""
    
    @staticmethod
    def create_default_framework(config: Optional[ValidationConfig] = None) -> ValidationOrchestrator:
        """Create framework with default components.
        
        Args:
            config: Optional configuration, uses defaults if not provided
            
        Returns:
            Fully configured ValidationOrchestrator
        """
        if config is None:
            config = ValidationConfig()
        
        logger.info("Creating default validation framework")
        
        # Create forecasting methods
        forecasting_methods = ValidationFrameworkFactory._create_default_forecasting_methods()
        forecasting_service = ForecastingService(forecasting_methods)
        
        # Create validation strategies
        validation_strategies = ValidationFrameworkFactory._create_default_validation_strategies()
        
        # Create metrics
        metric_calculators = ValidationFrameworkFactory._create_default_metric_calculators()
        metrics_service = MetricsService(metric_calculators)
        
        # Create adaptive selector
        adaptive_selector = AdaptiveStrategySelector(config)
        
        return ValidationOrchestrator(
            forecasting_service=forecasting_service,
            validation_strategies=validation_strategies,
            metrics_service=metrics_service,
            adaptive_selector=adaptive_selector,
            config=config
        )
    
    @staticmethod
    def create_custom_framework(
        forecasting_methods: Dict[str, ForecastingMethod],
        validation_strategies: Dict[str, ValidationStrategy],
        metric_calculators: Dict[str, MetricCalculator],
        config: ValidationConfig
    ) -> ValidationOrchestrator:
        """Create framework with custom components.
        
        Args:
            forecasting_methods: Dictionary of forecasting methods
            validation_strategies: Dictionary of validation strategies
            metric_calculators: Dictionary of metric calculators
            config: Configuration object
            
        Returns:
            Fully configured ValidationOrchestrator
        """
        logger.info("Creating custom validation framework")
        
        forecasting_service = ForecastingService(forecasting_methods)
        metrics_service = MetricsService(metric_calculators)
        adaptive_selector = AdaptiveStrategySelector(config)
        
        return ValidationOrchestrator(
            forecasting_service=forecasting_service,
            validation_strategies=validation_strategies,
            metrics_service=metrics_service,
            adaptive_selector=adaptive_selector,
            config=config
        )
    
    @staticmethod
    def create_minimal_framework(config: Optional[ValidationConfig] = None) -> ValidationOrchestrator:
        """Create framework with minimal components for testing/development.
        
        Args:
            config: Optional configuration, uses defaults if not provided
            
        Returns:
            Minimally configured ValidationOrchestrator
        """
        if config is None:
            config = ValidationConfig()
        
        logger.info("Creating minimal validation framework")
        
        # Minimal forecasting methods
        forecasting_methods = {
            'historical': HistoricalVolatilityMethod(),
            'ewma': EWMAVolatilityMethod()
        }
        forecasting_service = ForecastingService(forecasting_methods)
        
        # Minimal validation strategies
        validation_strategies = {
            'simple_holdout': SimpleHoldoutValidationStrategy(),
            'walk_forward': WalkForwardValidationStrategy()
        }
        
        # Core metrics only
        metric_calculators = {
            'mse': MSECalculator(),
            'mae': MAECalculator(),
            'bias': BiasCalculator()
        }
        metrics_service = MetricsService(metric_calculators)
        
        # Create adaptive selector
        adaptive_selector = AdaptiveStrategySelector(config)
        
        return ValidationOrchestrator(
            forecasting_service=forecasting_service,
            validation_strategies=validation_strategies,
            metrics_service=metrics_service,
            adaptive_selector=adaptive_selector,
            config=config
        )
    
    @staticmethod
    def _create_default_forecasting_methods() -> Dict[str, ForecastingMethod]:
        """Create default set of forecasting methods."""
        return {
            'historical': HistoricalVolatilityMethod(),
            'ewma': EWMAVolatilityMethod(),
            'exponential_smoothing': ExponentialSmoothingMethod(),
            'robust_mad': RobustMADMethod(),
            'quantile_range': QuantileRangeMethod()
        }
    
    @staticmethod
    def _create_default_validation_strategies() -> Dict[str, ValidationStrategy]:
        """Create default set of validation strategies."""
        return {
            'walk_forward': WalkForwardValidationStrategy(),
            'reduced_walk_forward': ReducedWalkForwardValidationStrategy(),
            'simple_holdout': SimpleHoldoutValidationStrategy()
        }
    
    @staticmethod
    def _create_default_metric_calculators() -> Dict[str, MetricCalculator]:
        """Create default set of metric calculators."""
        return {
            'mse': MSECalculator(),
            'mae': MAECalculator(),
            'rmse': RMSECalculator(),
            'hit_rate': HitRateCalculator(),
            'bias': BiasCalculator(),
            'directional_accuracy': DirectionalAccuracyCalculator(),
            'volatility_forecast_correlation': VolatilityForecastCorrelationCalculator(),
            'relative_bias': RelativeBiasCalculator(),
            'mape': MAPECalculator(),
            'theils_u': TheilsUCalculator(),
            'r_squared': R2Calculator()
        }
    
    @staticmethod
    def create_framework_for_testing() -> ValidationOrchestrator:
        """Create framework optimized for testing scenarios.
        
        Returns:
            ValidationOrchestrator configured for testing
        """
        # Configuration with lower thresholds for testing
        config = ValidationConfig(
            adaptive_thresholds={
                'walk_forward': 30,
                'reduced_walk_forward': 20,
                'simple_holdout': 10
            },
            default_forecasting_method='historical',
            enabled_metrics=['mse', 'mae', 'bias'],
            max_forecasts_per_validation=10  # Limit for faster testing
        )
        
        return ValidationFrameworkFactory.create_minimal_framework(config)
    
    @staticmethod
    def get_available_components() -> Dict[str, List[str]]:
        """Get information about available components.
        
        Returns:
            Dictionary listing all available components by type
        """
        return {
            'forecasting_methods': [
                'historical', 'ewma', 'exponential_smoothing', 
                'robust_mad', 'quantile_range'
            ],
            'validation_strategies': [
                'walk_forward', 'reduced_walk_forward', 'simple_holdout'
            ],
            'metric_calculators': [
                'mse', 'mae', 'rmse', 'hit_rate', 'bias', 
                'directional_accuracy', 'volatility_forecast_correlation',
                'relative_bias', 'mape', 'theils_u', 'r_squared'
            ]
        }