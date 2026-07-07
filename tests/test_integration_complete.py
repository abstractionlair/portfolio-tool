"""
Comprehensive integration tests for the CompatibilityValidationFramework.

These tests ensure that the integrated modular framework works correctly
with all existing components and produces expected results.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.validation import CompatibilityValidationFramework
from src.validation.parameter_validation import ValidationMethod, ParameterValidationFramework
from src.search.parameter_search import ParameterSearchEngine, SearchConfiguration
from data.multi_frequency import Frequency


class TestCompleteIntegration:
    """Test complete integration of CompatibilityValidationFramework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create test data
        self.test_data = pd.Series(np.random.normal(0, 0.02, 150))
        
        # Create validation frameworks
        self.compat_framework = CompatibilityValidationFramework()
        self.original_framework = ParameterValidationFramework()
        
        # Test combination
        self.test_combination = {
            'method': 'historical',
            'parameters': {'window': 20},
            'horizon': 21
        }
    
    def test_basic_compatibility(self):
        """Test basic compatibility between frameworks."""
        # Test with compatibility framework
        compat_result = self.compat_framework.validate_parameter_combination(
            self.test_data, self.test_combination, ValidationMethod.ADAPTIVE
        )
        
        # Test with original framework
        original_result = self.original_framework.validate_parameter_combination(
            self.test_data, self.test_combination, ValidationMethod.ADAPTIVE
        )
        
        # Both should succeed
        assert compat_result.success
        assert original_result.success
        
        # Should use same validation method
        assert compat_result.validation_method == original_result.validation_method
        
        # Should have similar results
        assert abs(compat_result.num_forecasts - original_result.num_forecasts) <= 1
        
        # MSE should be reasonably close (within 10%)
        if np.isfinite(compat_result.mse) and np.isfinite(original_result.mse):
            relative_diff = abs(compat_result.mse - original_result.mse) / max(original_result.mse, 1e-10)
            assert relative_diff < 0.1
    
    def test_different_validation_methods(self):
        """Test different validation methods work correctly."""
        methods = [ValidationMethod.ADAPTIVE, ValidationMethod.SIMPLE_HOLDOUT]
        
        for method in methods:
            result = self.compat_framework.validate_parameter_combination(
                self.test_data, self.test_combination, method
            )
            
            # Should either succeed or fail gracefully
            assert hasattr(result, 'success')
            if result.success:
                assert result.num_forecasts > 0
                assert np.isfinite(result.mse)
                assert np.isfinite(result.mae)
    
    def test_different_forecasting_methods(self):
        """Test different forecasting methods."""
        methods = ['historical', 'ewma', 'exponential_smoothing']
        
        for method in methods:
            if method in self.compat_framework.get_supported_methods():
                combination = {
                    'method': method,
                    'parameters': self.compat_framework.get_method_defaults(method),
                    'horizon': 21
                }
                
                result = self.compat_framework.validate_parameter_combination(
                    self.test_data, combination
                )
                
                assert result.success or result.error_message is not None
    
    def test_interface_compatibility(self):
        """Test that interfaces are compatible."""
        # Test supported methods
        supported = self.compat_framework.get_supported_methods()
        original_supported = self.original_framework.get_supported_methods()
        assert set(supported) == set(original_supported)
        
        # Test method defaults
        for method in supported:
            compat_defaults = self.compat_framework.get_method_defaults(method)
            original_defaults = self.original_framework.get_method_defaults(method)
            assert compat_defaults == original_defaults
        
        # Test forecasting_methods property
        compat_methods = self.compat_framework.forecasting_methods
        assert isinstance(compat_methods, dict)
        assert 'historical' in compat_methods
        
        # Test that method wrapper works
        historical = compat_methods['historical']
        forecast = historical.forecast(self.test_data, {'window': 20})
        assert np.isfinite(forecast)
        assert forecast > 0
    
    def test_parameter_search_integration(self):
        """Test integration with parameter search engine."""
        # Create sample exposure data
        exposure_data = {
            'test_exposure': {'spread': self.test_data}
        }
        
        # Create search configuration
        search_config = SearchConfiguration(
            history_lengths=[100],
            frequencies=[Frequency.WEEKLY],
            horizons=[21],
            methods={
                'historical': {
                    'description': 'Historical Volatility',
                    'parameters': [{'window': 20}]
                }
            }
        )
        
        # Create search engine with compatibility framework
        search_engine = ParameterSearchEngine(
            exposure_data=exposure_data,
            available_exposures=['test_exposure'],
            validation_framework=self.compat_framework,
            search_config=search_config
        )
        
        # Run search
        results = search_engine.run_search(
            estimation_date=pd.Timestamp('2021-01-01'),
            save_results=False
        )
        
        # Verify results
        assert results['summary']['total_combinations'] == 1
        assert results['summary']['completed'] == 1
        assert len(results['results']) > 0 or len(results['failed_combinations']) > 0
        
        # If successful, check result structure
        if results['results']:
            result = results['results'][0]
            assert 'combination' in result
            assert 'success' in result
            assert 'aggregate_metrics' in result
    
    def test_horizon_differentiation(self):
        """Test that different horizons produce different results."""
        horizons = [5, 10, 15, 21]
        results = []
        
        for horizon in horizons:
            combination = self.test_combination.copy()
            combination['horizon'] = horizon
            
            result = self.compat_framework.validate_parameter_combination(
                self.test_data, combination
            )
            
            if result.success:
                results.append(result.mse)
        
        # Should have at least some successful results
        assert len(results) >= 2
        
        # Results should be different (horizon bug fix verification)
        unique_results = set(f"{mse:.8f}" for mse in results)
        assert len(unique_results) == len(results), "Some horizons produced identical MSE values"
    
    def test_result_structure_compatibility(self):
        """Test that result structure is fully compatible."""
        result = self.compat_framework.validate_parameter_combination(
            self.test_data, self.test_combination
        )
        
        # Check all expected attributes exist
        expected_attrs = [
            'success', 'validation_method', 'num_forecasts',
            'mse', 'mae', 'rmse', 'hit_rate', 'bias',
            'directional_accuracy', 'volatility_forecast_correlation',
            'forecast_mean', 'forecast_std', 'actual_mean', 'actual_std',
            'forecast_actual_correlation', 'relative_bias'
        ]
        
        for attr in expected_attrs:
            assert hasattr(result, attr), f"Missing attribute: {attr}"
        
        # Test to_dict method
        assert hasattr(result, 'to_dict')
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        
        for attr in expected_attrs:
            assert attr in result_dict
    
    def test_error_handling(self):
        """Test error handling compatibility."""
        # Test with invalid method
        invalid_combination = {
            'method': 'nonexistent_method',
            'parameters': {},
            'horizon': 21
        }
        
        result = self.compat_framework.validate_parameter_combination(
            self.test_data, invalid_combination
        )
        
        assert not result.success
        assert result.error_message is not None
        
        # Test with insufficient data
        short_data = self.test_data[:5]
        result = self.compat_framework.validate_parameter_combination(
            short_data, self.test_combination
        )
        
        assert not result.success
        assert result.error_message is not None


class TestModularArchitectureBenefits:
    """Test that modular architecture provides expected benefits."""
    
    def test_configuration_flexibility(self):
        """Test that new architecture allows configuration flexibility."""
        from src.validation import ValidationFrameworkFactory, ValidationConfig
        
        # Create custom configuration
        custom_config = ValidationConfig(
            adaptive_thresholds={
                'walk_forward': 50,
                'reduced_walk_forward': 30,
                'simple_holdout': 15
            },
            default_forecasting_method='ewma'
        )
        
        # Create framework with custom config
        framework = ValidationFrameworkFactory.create_default_framework(custom_config)
        
        # Test that custom config is used
        assert framework.config.adaptive_thresholds['walk_forward'] == 50
        assert framework.config.default_forecasting_method == 'ewma'
        
        # Test that custom thresholds affect strategy selection
        recommendation = framework.get_strategy_recommendation(40)
        assert recommendation == 'reduced_walk_forward'
    
    def test_extensibility(self):
        """Test that new architecture is extensible."""
        from src.validation import (
            ValidationFrameworkFactory,
            HistoricalVolatilityMethod,
            SimpleHoldoutValidationStrategy,
            MSECalculator
        )
        
        # Create custom components
        custom_methods = {
            'custom_historical': HistoricalVolatilityMethod()
        }
        
        custom_strategies = {
            'custom_holdout': SimpleHoldoutValidationStrategy()
        }
        
        custom_metrics = {
            'custom_mse': MSECalculator()
        }
        
        # Create custom framework
        from src.validation import ValidationConfig
        config = ValidationConfig()
        framework = ValidationFrameworkFactory.create_custom_framework(
            forecasting_methods=custom_methods,
            validation_strategies=custom_strategies,
            metric_calculators=custom_metrics,
            config=config
        )
        
        # Test that custom components are available
        assert 'custom_historical' in framework.forecasting_service.get_available_methods()
        assert 'custom_holdout' in framework.validation_strategies
        assert 'custom_mse' in framework.metrics_service.get_available_metrics()
    
    def test_ai_friendly_development(self):
        """Test benefits for AI-friendly development."""
        from src.validation import ValidationFrameworkFactory
        
        framework = ValidationFrameworkFactory.create_default_framework()
        
        # Test focused component access
        forecasting_service = framework.forecasting_service
        metrics_service = framework.metrics_service
        
        # Each service should have clear, focused interface
        assert hasattr(forecasting_service, 'forecast')
        assert hasattr(forecasting_service, 'get_available_methods')
        assert hasattr(metrics_service, 'calculate_metric')
        assert hasattr(metrics_service, 'get_available_metrics')
        
        # Test diagnostic capabilities
        diagnostics = framework.validate_configuration()
        assert diagnostics['validation_status'] == 'valid'
        assert 'forecasting_service' in diagnostics
        assert 'validation_strategies' in diagnostics
        assert 'metrics_service' in diagnostics