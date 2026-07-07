"""
Tests for the modular validation framework interfaces.

These tests ensure the new architecture components work correctly
and provide the foundation for TDD refactoring.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.validation.interfaces import (
    ValidationConfig,
    ForecastingMethod,
    ValidationStrategy,
    MetricCalculator,
    ForecastingService,
    MetricsService,
    AdaptiveStrategySelector
)
from src.validation.parameter_validation import ValidationResult


# Test Implementations for Testing

class MockForecastingMethod(ForecastingMethod):
    """Mock forecasting method for testing."""
    
    def __init__(self, name: str, default_params: Dict[str, Any]):
        self._name = name
        self._default_params = default_params
        self.forecast_calls = []
    
    def forecast(self, train_data: pd.Series, parameters: Dict[str, Any]) -> float:
        self.forecast_calls.append((train_data.copy(), parameters.copy()))
        # Return a simple mock forecast
        return train_data.std() * (1 + parameters.get('multiplier', 0))
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        return self._default_params.copy()
    
    def get_method_name(self) -> str:
        return self._name


class MockValidationStrategy(ValidationStrategy):
    """Mock validation strategy for testing."""
    
    def __init__(self, name: str, required_length: int):
        self._name = name
        self._required_length = required_length
        self.validate_calls = []
    
    def validate(self, data: pd.Series, combination: Dict[str, Any],
                forecasting_service: ForecastingService) -> ValidationResult:
        self.validate_calls.append((data.copy(), combination.copy()))
        
        # Mock validation logic
        method = combination['method']
        parameters = combination.get('parameters', {})
        
        forecasts = []
        actuals = []
        
        # Generate some mock forecasts and actuals
        for i in range(min(5, len(data) // 3)):
            forecast = forecasting_service.forecast(method, data[:10+i*5], parameters)
            actual = abs(data.iloc[10+i*5]) if 10+i*5 < len(data) else forecast
            forecasts.append(forecast)
            actuals.append(actual)
        
        if len(forecasts) == 0:
            return ValidationResult(
                success=False,
                validation_method=self._name,
                num_forecasts=0,
                mse=np.nan, mae=np.nan, rmse=np.nan, hit_rate=np.nan,
                bias=np.nan, directional_accuracy=np.nan,
                volatility_forecast_correlation=np.nan,
                forecast_mean=np.nan, forecast_std=np.nan,
                actual_mean=np.nan, actual_std=np.nan,
                forecast_actual_correlation=np.nan, relative_bias=np.nan,
                error_message="No forecasts generated"
            )
        
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        
        return ValidationResult(
            success=True,
            validation_method=self._name,
            num_forecasts=len(forecasts),
            mse=np.mean((forecasts - actuals) ** 2),
            mae=np.mean(np.abs(forecasts - actuals)),
            rmse=np.sqrt(np.mean((forecasts - actuals) ** 2)),
            hit_rate=0.8,  # Mock value
            bias=np.mean(forecasts - actuals),
            directional_accuracy=0.7,  # Mock value
            volatility_forecast_correlation=0.5,  # Mock value
            forecast_mean=np.mean(forecasts),
            forecast_std=np.std(forecasts),
            actual_mean=np.mean(actuals),
            actual_std=np.std(actuals),
            forecast_actual_correlation=np.corrcoef(forecasts, actuals)[0, 1] if len(forecasts) > 1 else np.nan,
            relative_bias=np.mean((forecasts - actuals) / actuals) if np.all(actuals > 0) else np.nan
        )
    
    def get_required_data_length(self, horizon: int) -> int:
        return self._required_length + horizon
    
    def get_strategy_name(self) -> str:
        return self._name


class MockMetricCalculator(MetricCalculator):
    """Mock metric calculator for testing."""
    
    def __init__(self, name: str):
        self._name = name
        self.calculation_calls = []
    
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        self.calculation_calls.append((forecasts.copy(), actuals.copy()))
        
        if self._name == 'mse':
            return np.mean((forecasts - actuals) ** 2)
        elif self._name == 'mae':
            return np.mean(np.abs(forecasts - actuals))
        else:
            return np.random.random()  # Mock value
    
    def get_name(self) -> str:
        return self._name


# Tests

class TestValidationConfig:
    """Test ValidationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        assert config.adaptive_thresholds['walk_forward'] == 60
        assert config.adaptive_thresholds['reduced_walk_forward'] == 35
        assert config.adaptive_thresholds['simple_holdout'] == 25
        assert config.default_forecasting_method == 'historical'
        assert 'mse' in config.enabled_metrics
        assert 'mae' in config.enabled_metrics
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_thresholds = {'walk_forward': 100, 'simple_holdout': 50}
        config = ValidationConfig(
            adaptive_thresholds=custom_thresholds,
            default_forecasting_method='ewma',
            enabled_metrics=['mse', 'rmse']
        )
        
        assert config.adaptive_thresholds == custom_thresholds
        assert config.default_forecasting_method == 'ewma'
        assert config.enabled_metrics == ['mse', 'rmse']


class TestForecastingService:
    """Test ForecastingService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.method1 = MockForecastingMethod('method1', {'param1': 1.0})
        self.method2 = MockForecastingMethod('method2', {'param2': 2.0})
        
        self.service = ForecastingService({
            'method1': self.method1,
            'method2': self.method2
        })
        
        self.test_data = pd.Series(np.random.normal(0, 0.02, 50))
    
    def test_forecast_execution(self):
        """Test basic forecasting execution."""
        parameters = {'param1': 1.5, 'multiplier': 0.1}
        
        result = self.service.forecast('method1', self.test_data, parameters)
        
        assert isinstance(result, float)
        assert len(self.method1.forecast_calls) == 1
        assert self.method1.forecast_calls[0][1] == parameters
    
    def test_get_available_methods(self):
        """Test getting available methods."""
        methods = self.service.get_available_methods()
        
        assert 'method1' in methods
        assert 'method2' in methods
        assert len(methods) == 2
    
    def test_get_method_defaults(self):
        """Test getting method defaults."""
        defaults = self.service.get_method_defaults('method1')
        
        assert defaults == {'param1': 1.0}
    
    def test_unknown_method_error(self):
        """Test error handling for unknown method."""
        with pytest.raises(ValueError, match="Unknown forecasting method"):
            self.service.forecast('unknown', self.test_data, {})
    
    def test_has_method(self):
        """Test method existence checking."""
        assert self.service.has_method('method1')
        assert self.service.has_method('method2')
        assert not self.service.has_method('unknown')


class TestMetricsService:
    """Test MetricsService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mse_calc = MockMetricCalculator('mse')
        self.mae_calc = MockMetricCalculator('mae')
        
        self.service = MetricsService({
            'mse': self.mse_calc,
            'mae': self.mae_calc
        })
        
        self.forecasts = np.array([1.0, 2.0, 3.0])
        self.actuals = np.array([1.1, 1.9, 3.2])
    
    def test_calculate_metric(self):
        """Test calculating individual metric."""
        result = self.service.calculate_metric('mse', self.forecasts, self.actuals)
        
        assert isinstance(result, float)
        assert len(self.mse_calc.calculation_calls) == 1
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics."""
        results = self.service.calculate_all_metrics(self.forecasts, self.actuals)
        
        assert 'mse' in results
        assert 'mae' in results
        assert len(self.mse_calc.calculation_calls) == 1
        assert len(self.mae_calc.calculation_calls) == 1
    
    def test_calculate_subset_metrics(self):
        """Test calculating subset of metrics."""
        results = self.service.calculate_all_metrics(
            self.forecasts, self.actuals, enabled_metrics=['mse']
        )
        
        assert 'mse' in results
        assert 'mae' not in results
        assert len(self.mse_calc.calculation_calls) == 1
        assert len(self.mae_calc.calculation_calls) == 0
    
    def test_unknown_metric_error(self):
        """Test error handling for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            self.service.calculate_metric('unknown', self.forecasts, self.actuals)
    
    def test_get_available_metrics(self):
        """Test getting available metrics."""
        metrics = self.service.get_available_metrics()
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert len(metrics) == 2


class TestAdaptiveStrategySelector:
    """Test AdaptiveStrategySelector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ValidationConfig()
        self.selector = AdaptiveStrategySelector(self.config)
        
        self.strategies = {
            'walk_forward': MockValidationStrategy('walk_forward', 35),  # 35 + 21 = 56 < 60 threshold
            'reduced_walk_forward': MockValidationStrategy('reduced_walk_forward', 10),  # 10 + 21 = 31 < 35 threshold
            'simple_holdout': MockValidationStrategy('simple_holdout', 1)  # 1 + 21 = 22 < 25 threshold
        }
    
    def test_select_walk_forward(self):
        """Test selection of walk forward strategy."""
        strategy = self.selector.select_strategy(100, self.strategies)
        
        assert strategy.get_strategy_name() == 'walk_forward'
    
    def test_select_reduced_walk_forward(self):
        """Test selection of reduced walk forward strategy."""
        strategy = self.selector.select_strategy(50, self.strategies)
        
        assert strategy.get_strategy_name() == 'reduced_walk_forward'
    
    def test_select_simple_holdout(self):
        """Test selection of simple holdout strategy."""
        strategy = self.selector.select_strategy(30, self.strategies)
        
        assert strategy.get_strategy_name() == 'simple_holdout'
    
    def test_insufficient_data_error(self):
        """Test error when no strategy can handle data."""
        with pytest.raises(ValueError, match="No validation strategy can handle"):
            self.selector.select_strategy(10, self.strategies)
    
    def test_get_strategy_recommendation(self):
        """Test getting strategy recommendations."""
        assert self.selector.get_strategy_recommendation(100) == 'walk_forward'
        assert self.selector.get_strategy_recommendation(50) == 'reduced_walk_forward'
        assert self.selector.get_strategy_recommendation(30) == 'simple_holdout'
        assert self.selector.get_strategy_recommendation(10) == 'insufficient_data'
    
    def test_custom_thresholds(self):
        """Test with custom threshold configuration."""
        custom_config = ValidationConfig(adaptive_thresholds={
            'walk_forward': 80,
            'reduced_walk_forward': 40,
            'simple_holdout': 20
        })
        selector = AdaptiveStrategySelector(custom_config)
        
        assert selector.get_strategy_recommendation(70) == 'reduced_walk_forward'
        assert selector.get_strategy_recommendation(90) == 'walk_forward'


class TestIntegration:
    """Integration tests for interface components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        # Create services
        methods = {
            'simple': MockForecastingMethod('simple', {'window': 10})
        }
        self.forecasting_service = ForecastingService(methods)
        
        calculators = {
            'mse': MockMetricCalculator('mse'),
            'mae': MockMetricCalculator('mae')
        }
        self.metrics_service = MetricsService(calculators)
        
        # Create test data
        self.test_data = pd.Series(np.random.normal(0, 0.02, 100))
    
    def test_strategy_integration(self):
        """Test that validation strategy integrates with services."""
        strategy = MockValidationStrategy('test_strategy', 30)
        
        combination = {
            'method': 'simple',
            'parameters': {'window': 15},
            'horizon': 21
        }
        
        result = strategy.validate(self.test_data, combination, self.forecasting_service)
        
        assert result.success
        assert result.validation_method == 'test_strategy'
        assert result.num_forecasts > 0
        assert len(strategy.validate_calls) == 1
    
    def test_end_to_end_workflow(self):
        """Test complete workflow using all components."""
        config = ValidationConfig()
        selector = AdaptiveStrategySelector(config)
        
        strategies = {
            'simple_holdout': MockValidationStrategy('simple_holdout', 15)
        }
        
        # Select strategy
        strategy = selector.select_strategy(50, strategies)
        assert strategy.get_strategy_name() == 'simple_holdout'
        
        # Execute validation
        combination = {
            'method': 'simple',
            'parameters': {'window': 10},
            'horizon': 5
        }
        
        result = strategy.validate(self.test_data, combination, self.forecasting_service)
        
        assert result.success
        assert result.num_forecasts > 0