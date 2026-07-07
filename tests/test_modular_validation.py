"""
Tests for the modular validation framework.

These tests verify that the new modular architecture works correctly
and provides the same functionality as the original framework while
being more extensible and maintainable.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.validation import (
    ValidationFrameworkFactory,
    ValidationConfig,
    ValidationOrchestrator,
    HistoricalVolatilityMethod,
    EWMAVolatilityMethod,
    WalkForwardValidationStrategy,
    SimpleHoldoutValidationStrategy,
    MSECalculator,
    MAECalculator,
    ForecastingService,
    MetricsService,
    AdaptiveStrategySelector
)


class TestModularValidationFramework:
    """Test the complete modular validation framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test data
        np.random.seed(42)
        self.test_data = pd.Series(np.random.normal(0, 0.02, 100))
        
        # Create default framework
        self.framework = ValidationFrameworkFactory.create_default_framework()
        
        # Test combination
        self.test_combination = {
            'method': 'historical',
            'parameters': {'window': 20},
            'horizon': 21
        }
    
    def test_framework_creation(self):
        """Test that framework can be created successfully."""
        assert isinstance(self.framework, ValidationOrchestrator)
        
        # Test framework components
        assert self.framework.forecasting_service is not None
        assert self.framework.validation_strategies is not None
        assert self.framework.metrics_service is not None
        assert self.framework.adaptive_selector is not None
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        assert config.adaptive_thresholds['walk_forward'] == 60
        assert config.adaptive_thresholds['reduced_walk_forward'] == 35
        assert config.adaptive_thresholds['simple_holdout'] == 25
        assert config.default_forecasting_method == 'historical'
        assert 'mse' in config.enabled_metrics
        assert 'mae' in config.enabled_metrics
    
    def test_basic_validation(self):
        """Test basic validation functionality."""
        result = self.framework.validate_parameter_combination(
            self.test_data, self.test_combination
        )
        
        assert result.success
        assert result.validation_method in ['walk_forward', 'reduced_walk_forward', 'simple_holdout']
        assert result.num_forecasts > 0
        assert np.isfinite(result.mse)
        assert np.isfinite(result.mae)
    
    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection."""
        # Test with different data lengths
        short_data = self.test_data[:30]
        medium_data = self.test_data[:50]
        long_data = self.test_data[:80]
        
        # Short data should use simple_holdout
        result_short = self.framework.validate_parameter_combination(
            short_data, self.test_combination
        )
        assert result_short.validation_method == 'simple_holdout'
        
        # Medium data should use reduced_walk_forward
        result_medium = self.framework.validate_parameter_combination(
            medium_data, self.test_combination
        )
        assert result_medium.validation_method == 'reduced_walk_forward'
        
        # Long data should use walk_forward
        result_long = self.framework.validate_parameter_combination(
            long_data, self.test_combination
        )
        assert result_long.validation_method == 'walk_forward'
    
    def test_specific_validation_method(self):
        """Test specifying a specific validation method."""
        result = self.framework.validate_parameter_combination(
            self.test_data, self.test_combination, validation_method='simple_holdout'
        )
        
        assert result.success
        assert result.validation_method == 'simple_holdout'
    
    def test_different_forecasting_methods(self):
        """Test different forecasting methods."""
        methods_to_test = ['historical', 'ewma', 'exponential_smoothing', 'robust_mad']
        
        for method in methods_to_test:
            if method in self.framework.forecasting_service.get_available_methods():
                combination = {
                    'method': method,
                    'parameters': self.framework.get_method_defaults(method),
                    'horizon': 21
                }
                
                result = self.framework.validate_parameter_combination(
                    self.test_data, combination
                )
                
                assert result.success, f"Method {method} failed validation"
                assert result.num_forecasts > 0
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        very_short_data = self.test_data[:10]  # Too short for any strategy
        
        result = self.framework.validate_parameter_combination(
            very_short_data, self.test_combination
        )
        
        assert not result.success
        assert 'No validation strategy can handle' in result.error_message
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid method
        invalid_combination = {
            'method': 'nonexistent_method',
            'parameters': {},
            'horizon': 21
        }
        
        result = self.framework.validate_parameter_combination(
            self.test_data, invalid_combination
        )
        
        assert not result.success
        assert result.error_message is not None
    
    def test_framework_information_methods(self):
        """Test methods that provide framework information."""
        # Test get_supported_methods
        supported = self.framework.get_supported_methods()
        
        assert 'forecasting_methods' in supported
        assert 'validation_strategies' in supported
        assert 'metrics' in supported
        assert 'config' in supported
        
        # Test get_method_defaults
        defaults = self.framework.get_method_defaults('historical')
        assert 'window' in defaults
        
        # Test get_strategy_recommendation
        recommendation = self.framework.get_strategy_recommendation(100)
        assert recommendation == 'walk_forward'
        
        recommendation = self.framework.get_strategy_recommendation(30)
        assert recommendation == 'simple_holdout'
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        diagnostics = self.framework.validate_configuration()
        
        assert diagnostics['validation_status'] == 'valid'
        assert diagnostics['forecasting_service']['available_methods'] > 0
        assert diagnostics['validation_strategies']['available_strategies'] > 0
        assert diagnostics['metrics_service']['available_metrics'] > 0


class TestFactoryMethods:
    """Test the factory methods for creating frameworks."""
    
    def test_default_framework_creation(self):
        """Test creating default framework."""
        framework = ValidationFrameworkFactory.create_default_framework()
        
        assert isinstance(framework, ValidationOrchestrator)
        assert len(framework.forecasting_service.get_available_methods()) >= 5
        assert len(framework.validation_strategies) >= 3
    
    def test_minimal_framework_creation(self):
        """Test creating minimal framework."""
        framework = ValidationFrameworkFactory.create_minimal_framework()
        
        assert isinstance(framework, ValidationOrchestrator)
        assert len(framework.forecasting_service.get_available_methods()) >= 2
        assert len(framework.validation_strategies) >= 2
    
    def test_testing_framework_creation(self):
        """Test creating framework for testing."""
        framework = ValidationFrameworkFactory.create_framework_for_testing()
        
        assert isinstance(framework, ValidationOrchestrator)
        # Should have lower thresholds for testing
        assert framework.config.adaptive_thresholds['walk_forward'] == 30
        assert framework.config.adaptive_thresholds['simple_holdout'] == 10
    
    def test_custom_framework_creation(self):
        """Test creating custom framework."""
        # Create custom components
        forecasting_methods = {
            'test_method': HistoricalVolatilityMethod()
        }
        
        validation_strategies = {
            'test_strategy': SimpleHoldoutValidationStrategy()
        }
        
        metric_calculators = {
            'test_metric': MSECalculator()
        }
        
        config = ValidationConfig()
        
        framework = ValidationFrameworkFactory.create_custom_framework(
            forecasting_methods=forecasting_methods,
            validation_strategies=validation_strategies,
            metric_calculators=metric_calculators,
            config=config
        )
        
        assert isinstance(framework, ValidationOrchestrator)
        assert 'test_method' in framework.forecasting_service.get_available_methods()
        assert 'test_strategy' in framework.validation_strategies
    
    def test_available_components(self):
        """Test getting available components information."""
        components = ValidationFrameworkFactory.get_available_components()
        
        assert 'forecasting_methods' in components
        assert 'validation_strategies' in components
        assert 'metric_calculators' in components
        
        assert len(components['forecasting_methods']) >= 5
        assert len(components['validation_strategies']) >= 3
        assert len(components['metric_calculators']) >= 7


class TestModularComponents:
    """Test individual modular components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = pd.Series(np.random.normal(0, 0.02, 100))
    
    def test_forecasting_service(self):
        """Test forecasting service functionality."""
        methods = {
            'historical': HistoricalVolatilityMethod(),
            'ewma': EWMAVolatilityMethod()
        }
        
        service = ForecastingService(methods)
        
        # Test basic forecasting
        forecast = service.forecast('historical', self.test_data, {'window': 20})
        assert np.isfinite(forecast)
        assert forecast > 0
        
        # Test method information
        assert 'historical' in service.get_available_methods()
        assert 'ewma' in service.get_available_methods()
        
        defaults = service.get_method_defaults('historical')
        assert 'window' in defaults
        
        assert service.has_method('historical')
        assert not service.has_method('nonexistent')
    
    def test_metrics_service(self):
        """Test metrics service functionality."""
        calculators = {
            'mse': MSECalculator(),
            'mae': MAECalculator()
        }
        
        service = MetricsService(calculators)
        
        # Test metric calculation
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 1.9, 3.2])
        
        mse = service.calculate_metric('mse', forecasts, actuals)
        assert np.isfinite(mse)
        assert mse > 0
        
        all_metrics = service.calculate_all_metrics(forecasts, actuals)
        assert 'mse' in all_metrics
        assert 'mae' in all_metrics
        
        # Test subset calculation
        subset_metrics = service.calculate_all_metrics(
            forecasts, actuals, enabled_metrics=['mse']
        )
        assert 'mse' in subset_metrics
        assert 'mae' not in subset_metrics
    
    def test_adaptive_strategy_selector(self):
        """Test adaptive strategy selector."""
        config = ValidationConfig()
        selector = AdaptiveStrategySelector(config)
        
        strategies = {
            'walk_forward': WalkForwardValidationStrategy(),
            'simple_holdout': SimpleHoldoutValidationStrategy()
        }
        
        # Test strategy selection
        strategy = selector.select_strategy(100, strategies)
        assert strategy.get_strategy_name() == 'walk_forward'
        
        strategy = selector.select_strategy(30, strategies)
        assert strategy.get_strategy_name() == 'simple_holdout'
        
        # Test recommendations
        assert selector.get_strategy_recommendation(100) == 'walk_forward'
        assert selector.get_strategy_recommendation(30) == 'simple_holdout'
        assert selector.get_strategy_recommendation(10) == 'insufficient_data'


class TestBackwardCompatibility:
    """Test that new framework provides backward compatibility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = pd.Series(np.random.normal(0, 0.02, 100))
        
        self.test_combination = {
            'method': 'historical',
            'parameters': {'window': 20},
            'horizon': 21
        }
    
    def test_result_structure_compatibility(self):
        """Test that results have the same structure as legacy framework."""
        framework = ValidationFrameworkFactory.create_default_framework()
        
        result = framework.validate_parameter_combination(
            self.test_data, self.test_combination
        )
        
        # Check that all expected fields are present
        expected_fields = [
            'success', 'validation_method', 'num_forecasts',
            'mse', 'mae', 'rmse', 'hit_rate', 'bias',
            'directional_accuracy', 'volatility_forecast_correlation',
            'forecast_mean', 'forecast_std', 'actual_mean', 'actual_std',
            'forecast_actual_correlation', 'relative_bias'
        ]
        
        for field in expected_fields:
            assert hasattr(result, field), f"Missing field: {field}"
        
        # Check that result has to_dict method
        assert hasattr(result, 'to_dict')
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
    
    def test_method_compatibility(self):
        """Test that forecasting methods produce similar results."""
        framework = ValidationFrameworkFactory.create_default_framework()
        
        # Test that the same methods are available
        available_methods = framework.forecasting_service.get_available_methods()
        expected_methods = ['historical', 'ewma', 'exponential_smoothing', 'robust_mad', 'quantile_range']
        
        for method in expected_methods:
            assert method in available_methods, f"Missing method: {method}"