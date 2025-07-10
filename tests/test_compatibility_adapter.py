"""
Tests for the compatibility adapter.

These tests ensure that the compatibility adapter provides the same interface
as the original ParameterValidationFramework while using the new modular
architecture under the hood.
"""

import pytest
import numpy as np
import pandas as pd

from src.validation.compatibility_adapter import (
    CompatibilityValidationFramework,
    create_legacy_framework,
    migrate_to_modular_framework
)
from src.validation.parameter_validation import ValidationMethod, ParameterValidationFramework


class TestCompatibilityAdapter:
    """Test the compatibility adapter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test data
        np.random.seed(42)
        self.test_data = pd.Series(np.random.normal(0, 0.02, 100))
        
        # Create compatibility framework
        self.compat_framework = CompatibilityValidationFramework()
        
        # Create original framework for comparison
        self.original_framework = ParameterValidationFramework()
        
        # Test combination
        self.test_combination = {
            'method': 'historical',
            'parameters': {'window': 20},
            'horizon': 21
        }
    
    def test_initialization(self):
        """Test that compatibility framework initializes correctly."""
        framework = CompatibilityValidationFramework(adaptive_mode=True)
        assert framework.adaptive_mode is True
        
        framework = CompatibilityValidationFramework(adaptive_mode=False)
        assert framework.adaptive_mode is False
    
    def test_validate_parameter_combination_interface(self):
        """Test that validation method has the same interface as original."""
        # Test with ValidationMethod enum
        result = self.compat_framework.validate_parameter_combination(
            self.test_data, self.test_combination, ValidationMethod.ADAPTIVE
        )
        
        assert result.success
        assert hasattr(result, 'validation_method')
        assert hasattr(result, 'num_forecasts')
        assert hasattr(result, 'mse')
        assert hasattr(result, 'mae')
    
    def test_different_validation_methods(self):
        """Test different validation methods work."""
        methods_to_test = [
            ValidationMethod.ADAPTIVE,
            ValidationMethod.SIMPLE_HOLDOUT,
            ValidationMethod.WALK_FORWARD
        ]
        
        for method in methods_to_test:
            result = self.compat_framework.validate_parameter_combination(
                self.test_data, self.test_combination, method
            )
            
            # Should succeed or fail gracefully
            assert hasattr(result, 'success')
            if result.success:
                assert result.num_forecasts > 0
    
    def test_get_supported_methods(self):
        """Test get_supported_methods compatibility."""
        methods = self.compat_framework.get_supported_methods()
        
        assert isinstance(methods, list)
        assert 'historical' in methods
        assert len(methods) > 0
        
        # Compare with original framework
        original_methods = self.original_framework.get_supported_methods()
        assert set(methods) == set(original_methods)
    
    def test_get_method_defaults(self):
        """Test get_method_defaults compatibility."""
        defaults = self.compat_framework.get_method_defaults('historical')
        
        assert isinstance(defaults, dict)
        assert 'window' in defaults
        
        # Compare with original framework
        original_defaults = self.original_framework.get_method_defaults('historical')
        assert defaults == original_defaults
    
    def test_forecasting_methods_property(self):
        """Test forecasting_methods property compatibility."""
        methods = self.compat_framework.forecasting_methods
        
        assert isinstance(methods, dict)
        assert 'historical' in methods
        
        # Test that method wrapper works
        historical_method = methods['historical']
        assert hasattr(historical_method, 'forecast')
        assert hasattr(historical_method, 'get_parameter_defaults')
        
        # Test forecast functionality
        forecast = historical_method.forecast(self.test_data, {'window': 20})
        assert np.isfinite(forecast)
        assert forecast > 0
        
        # Test defaults
        defaults = historical_method.get_parameter_defaults()
        assert isinstance(defaults, dict)
    
    def test_metrics_property(self):
        """Test metrics property compatibility."""
        metrics = self.compat_framework.metrics
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'mae' in metrics
        
        # Test that metric wrapper works
        mse_calc = metrics['mse']
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 1.9, 3.2])
        
        mse_value = mse_calc(forecasts, actuals)
        assert np.isfinite(mse_value)
        assert mse_value > 0
    
    def test_result_compatibility(self):
        """Test that results are compatible with original framework."""
        result = self.compat_framework.validate_parameter_combination(
            self.test_data, self.test_combination
        )
        
        original_result = self.original_framework.validate_parameter_combination(
            self.test_data, self.test_combination
        )
        
        # Check that both have the same structure
        assert type(result) == type(original_result)
        assert hasattr(result, 'to_dict')
        
        result_dict = result.to_dict()
        original_dict = original_result.to_dict()
        
        # Should have the same keys
        assert set(result_dict.keys()) == set(original_dict.keys())


class TestFactoryFunctions:
    """Test factory functions for compatibility."""
    
    def test_create_legacy_framework(self):
        """Test create_legacy_framework function."""
        framework = create_legacy_framework(adaptive_mode=True)
        
        assert isinstance(framework, CompatibilityValidationFramework)
        assert framework.adaptive_mode is True
        
        framework = create_legacy_framework(adaptive_mode=False)
        assert framework.adaptive_mode is False
    
    def test_migrate_to_modular_framework(self):
        """Test migration function."""
        original_framework = ParameterValidationFramework()
        
        # Migrate with config preservation
        new_framework = migrate_to_modular_framework(
            original_framework, preserve_config=True
        )
        
        assert new_framework is not None
        
        # Test that new framework works
        test_data = pd.Series(np.random.normal(0, 0.02, 100))
        combination = {
            'method': 'historical',
            'parameters': {'window': 20},
            'horizon': 21
        }
        
        result = new_framework.validate_parameter_combination(
            test_data, combination
        )
        
        assert result.success
    
    def test_migrate_without_config_preservation(self):
        """Test migration without config preservation."""
        original_framework = ParameterValidationFramework()
        
        new_framework = migrate_to_modular_framework(
            original_framework, preserve_config=False
        )
        
        assert new_framework is not None


class TestDropInReplacement:
    """Test that compatibility adapter can be used as drop-in replacement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = pd.Series(np.random.normal(0, 0.02, 100))
        self.test_combination = {
            'method': 'historical',
            'parameters': {'window': 20},
            'horizon': 21
        }
    
    def test_same_results_as_original(self):
        """Test that compatibility adapter produces similar results to original."""
        original_framework = ParameterValidationFramework()
        compat_framework = CompatibilityValidationFramework()
        
        # Run validation with both frameworks
        original_result = original_framework.validate_parameter_combination(
            self.test_data, self.test_combination
        )
        
        compat_result = compat_framework.validate_parameter_combination(
            self.test_data, self.test_combination
        )
        
        # Both should succeed
        assert original_result.success
        assert compat_result.success
        
        # Both should use the same validation method
        assert original_result.validation_method == compat_result.validation_method
        
        # Both should have similar number of forecasts
        assert abs(original_result.num_forecasts - compat_result.num_forecasts) <= 1
        
        # Metrics should be reasonably similar (within 10% due to potential minor differences)
        if np.isfinite(original_result.mse) and np.isfinite(compat_result.mse):
            relative_diff = abs(original_result.mse - compat_result.mse) / max(original_result.mse, 1e-10)
            assert relative_diff < 0.1
    
    def test_code_replacement_example(self):
        """Test example of replacing original framework with compatibility adapter."""
        # This simulates existing code that uses the original framework
        def validation_function(framework, data, combination):
            """Example function that uses validation framework."""
            result = framework.validate_parameter_combination(data, combination)
            
            if result.success:
                return {
                    'mse': result.mse,
                    'mae': result.mae,
                    'num_forecasts': result.num_forecasts,
                    'method': result.validation_method
                }
            else:
                return None
        
        # Test with original framework
        original_framework = ParameterValidationFramework()
        original_output = validation_function(
            original_framework, self.test_data, self.test_combination
        )
        
        # Test with compatibility framework (drop-in replacement)
        compat_framework = CompatibilityValidationFramework()
        compat_output = validation_function(
            compat_framework, self.test_data, self.test_combination
        )
        
        # Both should work and produce similar results
        assert original_output is not None
        assert compat_output is not None
        
        assert original_output['method'] == compat_output['method']
        assert abs(original_output['num_forecasts'] - compat_output['num_forecasts']) <= 1