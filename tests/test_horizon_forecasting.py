"""
Tests for horizon forecasting edge cases and cross-horizon consistency.

This module addresses the previous bugs where identical errors appeared
across different horizons, indicating problems in the validation logic.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, assume
from datetime import datetime, timedelta

from src.validation.parameter_validation import (
    ParameterValidationFramework,
    ValidationMethod,
    ValidationResult
)


class TestHorizonForecasting:
    """Test horizon-specific forecasting behavior."""
    
    def setup_method(self):
        """Set up test framework."""
        self.framework = ParameterValidationFramework(adaptive_mode=True)
        np.random.seed(42)
    
    def test_horizon_specific_validation_results(self):
        """Test that different horizons produce different validation results."""
        # Create a longer series to ensure sufficient data
        series = pd.Series(np.random.normal(0, 0.02, 200))
        
        horizons = [21, 42, 63]
        results = {}
        
        for horizon in horizons:
            combination = {
                'method': 'ewma',
                'parameters': {'lambda': 0.94, 'min_periods': 30},
                'horizon': horizon
            }
            
            result = self.framework.validate_parameter_combination(
                series, combination, ValidationMethod.WALK_FORWARD
            )
            
            assert result.success, f"Validation failed for horizon {horizon}"
            results[horizon] = result
        
        # Check that results are different across horizons
        # MSE should generally differ across horizons
        mse_values = [results[h].mse for h in horizons]
        assert not all(abs(mse_values[0] - mse) < 1e-10 for mse in mse_values), \
            f"MSE values suspiciously identical across horizons: {mse_values}"
        
        # Hit rates should generally differ
        hit_rates = [results[h].hit_rate for h in horizons]
        assert not all(abs(hit_rates[0] - hr) < 1e-10 for hr in hit_rates), \
            f"Hit rates suspiciously identical across horizons: {hit_rates}"
    
    def test_horizon_scaling_behavior(self):
        """Test that horizon scaling behaves correctly."""
        series = pd.Series(np.random.normal(0, 0.02, 150))
        
        # Test different validation methods with horizon scaling
        for method in ['walk_forward', 'reduced_walk_forward', 'simple_holdout']:
            if method == 'walk_forward' and len(series) < 60:
                continue
            if method == 'reduced_walk_forward' and len(series) < 35:
                continue
            if method == 'simple_holdout' and len(series) < 25:
                continue
            
            validation_method = ValidationMethod(method)
            
            short_horizon_result = self.framework.validate_parameter_combination(
                series, 
                {'method': 'historical', 'parameters': {'window': 20}, 'horizon': 5},
                validation_method
            )
            
            long_horizon_result = self.framework.validate_parameter_combination(
                series,
                {'method': 'historical', 'parameters': {'window': 20}, 'horizon': 60},
                validation_method
            )
            
            if short_horizon_result.success and long_horizon_result.success:
                # Longer horizons should generally have higher uncertainty
                # (though this is not guaranteed, we check it's not identical)
                assert short_horizon_result.mse != long_horizon_result.mse, \
                    f"Identical MSE for different horizons in {method}"
    
    def test_insufficient_data_horizon_handling(self):
        """Test behavior when insufficient data for requested horizon."""
        # Create a short series
        series = pd.Series(np.random.normal(0, 0.02, 30))
        
        # Request a horizon that's too large relative to data
        combination = {
            'method': 'historical',
            'parameters': {'window': 10},
            'horizon': 50  # Larger than available data
        }
        
        result = self.framework.validate_parameter_combination(
            series, combination, ValidationMethod.SIMPLE_HOLDOUT
        )
        
        # Should handle gracefully - either succeed with scaling or fail gracefully
        if result.success:
            assert result.num_forecasts > 0
            assert np.isfinite(result.mse)
        else:
            assert result.error_message is not None
    
    @given(
        series_length=st.integers(min_value=60, max_value=150),
        horizons=st.lists(st.integers(min_value=5, max_value=30), min_size=2, max_size=3),
        volatility=st.floats(min_value=0.01, max_value=0.05)
    )
    def test_horizon_consistency_property(self, series_length, horizons, volatility):
        """Property-based test for horizon consistency."""
        # Remove restrictive assumptions that cause filtering
        horizons = list(set(horizons))  # Remove duplicates
        if len(horizons) < 2:
            horizons = [5, 15]  # Fallback to reasonable defaults
        
        # Generate test series
        series = pd.Series(np.random.normal(0, volatility, series_length))
        
        results = {}
        for horizon in horizons:
            combination = {
                'method': 'historical',
                'parameters': {'window': min(20, series_length // 3)},
                'horizon': horizon
            }
            
            result = self.framework.validate_parameter_combination(
                series, combination, ValidationMethod.ADAPTIVE
            )
            
            if result.success:
                results[horizon] = result
        
        # Property: If we have results for multiple horizons, they should be distinct
        if len(results) > 1:
            mse_values = [result.mse for result in results.values()]
            # Not all MSE values should be identical (allowing for some numerical precision)
            distinct_mse = len(set(round(mse, 8) for mse in mse_values))
            assert distinct_mse > 1 or len(results) == 1, \
                f"All MSE values identical across horizons: {mse_values}"
    
    def test_horizon_edge_case_validation(self):
        """Test edge cases in horizon validation."""
        series = pd.Series(np.random.normal(0, 0.02, 100))
        
        # Test horizon = 0 (should be handled gracefully)
        combination = {
            'method': 'historical',
            'parameters': {'window': 10},
            'horizon': 0
        }
        
        result = self.framework.validate_parameter_combination(
            series, combination, ValidationMethod.ADAPTIVE
        )
        
        # Should either succeed with reasonable behavior or fail gracefully
        if result.success:
            assert result.num_forecasts >= 0
        else:
            assert result.error_message is not None
        
        # Test horizon = 1 (minimum reasonable horizon)
        combination['horizon'] = 1
        result = self.framework.validate_parameter_combination(
            series, combination, ValidationMethod.ADAPTIVE
        )
        
        assert result.success, "Should handle horizon=1 successfully"
        assert result.num_forecasts > 0
    
    def test_cross_method_horizon_consistency(self):
        """Test that different forecasting methods handle horizons consistently."""
        series = pd.Series(np.random.normal(0, 0.02, 120))
        horizon = 21
        
        methods = ['historical', 'ewma', 'exponential_smoothing']
        results = {}
        
        for method in methods:
            if method == 'historical':
                params = {'window': 20}
            elif method == 'ewma':
                params = {'lambda': 0.94, 'min_periods': 10}
            else:  # exponential_smoothing
                params = {'alpha': 0.3}
            
            combination = {
                'method': method,
                'parameters': params,
                'horizon': horizon
            }
            
            result = self.framework.validate_parameter_combination(
                series, combination, ValidationMethod.WALK_FORWARD
            )
            
            if result.success:
                results[method] = result
        
        # All methods should produce some forecasts
        assert len(results) > 0, "No methods succeeded"
        
        # Check that different methods produce different results
        # (they should, given different forecasting approaches)
        if len(results) > 1:
            mse_values = [result.mse for result in results.values()]
            # Allow some tolerance for numerical precision
            unique_mse = len(set(round(mse, 6) for mse in mse_values))
            assert unique_mse > 1 or len(results) == 1, \
                f"All methods produced identical MSE: {mse_values}"
    
    def test_validation_method_horizon_interaction(self):
        """Test interaction between validation methods and horizons."""
        series = pd.Series(np.random.normal(0, 0.02, 150))
        horizon = 30
        
        validation_methods = [
            ValidationMethod.WALK_FORWARD,
            ValidationMethod.REDUCED_WALK_FORWARD,
            ValidationMethod.SIMPLE_HOLDOUT
        ]
        
        results = {}
        
        for val_method in validation_methods:
            combination = {
                'method': 'historical',
                'parameters': {'window': 20},
                'horizon': horizon
            }
            
            result = self.framework.validate_parameter_combination(
                series, combination, val_method
            )
            
            if result.success:
                results[val_method.value] = result
        
        # Different validation methods should generally produce different results
        assert len(results) > 0, "No validation methods succeeded"
        
        if len(results) > 1:
            num_forecasts = [result.num_forecasts for result in results.values()]
            # Different validation methods should produce different numbers of forecasts
            assert len(set(num_forecasts)) > 1, \
                f"All validation methods produced same number of forecasts: {num_forecasts}"


class TestHorizonRegressionPrevention:
    """Tests specifically designed to prevent regression of previous horizon bugs."""
    
    def setup_method(self):
        """Set up test framework."""
        self.framework = ParameterValidationFramework(adaptive_mode=True)
        np.random.seed(42)
    
    def test_prevent_identical_horizon_errors(self):
        """Regression test: Prevent identical errors across horizons."""
        series = pd.Series(np.random.normal(0, 0.02, 200))
        
        # Test the exact scenario that previously caused identical errors
        horizons = [21, 42, 63]
        
        # Use EWMA method which was prone to the bug
        results = []
        for horizon in horizons:
            combination = {
                'method': 'ewma',
                'parameters': {'lambda': 0.94, 'min_periods': 30},
                'horizon': horizon
            }
            
            result = self.framework.validate_parameter_combination(
                series, combination, ValidationMethod.WALK_FORWARD
            )
            
            assert result.success, f"Validation failed for horizon {horizon}"
            results.append(result)
        
        # Check that the specific bug doesn't occur
        mse_values = [r.mse for r in results]
        mae_values = [r.mae for r in results]
        
        # MSE values should not be identical (allowing for minimal numerical precision)
        mse_rounded = [round(mse, 10) for mse in mse_values]
        assert len(set(mse_rounded)) > 1, \
            f"REGRESSION: Identical MSE values across horizons {horizons}: {mse_values}"
        
        # MAE values should not be identical
        mae_rounded = [round(mae, 10) for mae in mae_values]
        assert len(set(mae_rounded)) > 1, \
            f"REGRESSION: Identical MAE values across horizons {horizons}: {mae_values}"
    
    def test_prevent_same_horizon_highest_lowest_error(self):
        """Regression test: Prevent same horizon being both highest and lowest error."""
        series = pd.Series(np.random.normal(0, 0.02, 200))
        
        horizons = [7, 14, 21, 42, 63]
        results = {}
        
        for horizon in horizons:
            combination = {
                'method': 'ewma',
                'parameters': {'lambda': 0.94, 'min_periods': 20},
                'horizon': horizon
            }
            
            result = self.framework.validate_parameter_combination(
                series, combination, ValidationMethod.WALK_FORWARD
            )
            
            if result.success:
                results[horizon] = result
        
        assert len(results) >= 3, "Need at least 3 successful results for this test"
        
        # Find horizon with highest and lowest MSE
        mse_by_horizon = {h: r.mse for h, r in results.items()}
        highest_mse_horizon = max(mse_by_horizon.keys(), key=lambda h: mse_by_horizon[h])
        lowest_mse_horizon = min(mse_by_horizon.keys(), key=lambda h: mse_by_horizon[h])
        
        assert highest_mse_horizon != lowest_mse_horizon, \
            f"REGRESSION: Same horizon {highest_mse_horizon} has both highest and lowest MSE"
        
        # Same check for MAE
        mae_by_horizon = {h: r.mae for h, r in results.items()}
        highest_mae_horizon = max(mae_by_horizon.keys(), key=lambda h: mae_by_horizon[h])
        lowest_mae_horizon = min(mae_by_horizon.keys(), key=lambda h: mae_by_horizon[h])
        
        assert highest_mae_horizon != lowest_mae_horizon, \
            f"REGRESSION: Same horizon {highest_mae_horizon} has both highest and lowest MAE"
    
    def test_horizon_scaling_prevents_fallback_bug(self):
        """Test that horizon scaling prevents the previous fallback bug."""
        series = pd.Series(np.random.normal(0, 0.02, 100))
        
        # Test scenario where insufficient data for full horizon
        # should scale properly, not fall back to same data point
        
        short_series = series[:50]  # Limited data
        long_horizon = 40  # Horizon close to series length
        
        combination = {
            'method': 'historical',
            'parameters': {'window': 15},
            'horizon': long_horizon
        }
        
        result = self.framework.validate_parameter_combination(
            short_series, combination, ValidationMethod.SIMPLE_HOLDOUT
        )
        
        if result.success:
            # Should have some forecasts
            assert result.num_forecasts > 0
            
            # Compare with a shorter horizon on same data
            short_combination = {
                'method': 'historical',
                'parameters': {'window': 15},
                'horizon': 10
            }
            
            short_result = self.framework.validate_parameter_combination(
                short_series, short_combination, ValidationMethod.SIMPLE_HOLDOUT
            )
            
            if short_result.success:
                # Results should be different (proper scaling vs fallback)
                assert abs(result.mse - short_result.mse) > 1e-10, \
                    "REGRESSION: Horizon scaling not working, identical results"