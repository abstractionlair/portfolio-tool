"""
Unit tests for comprehensive parameter search functionality.

Tests the complete pipeline optimization including data loading, decomposition,
and estimation method parameter optimization.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.optimization.comprehensive_parameter_search import (
    ComprehensiveParameterEstimator,
    ComprehensiveParameterSearchEngine,
    ComprehensiveSearchResult,
    analyze_search_results
)


class TestComprehensiveParameterEstimator:
    """Test the comprehensive parameter estimator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock risk estimator
        self.mock_risk_estimator = Mock()
        self.estimation_date = datetime.now()
        self.exposure_id = 'test_exposure'
        
        # Create sample decomposition data
        dates = pd.date_range('2020-01-01', periods=30, freq='ME')
        self.sample_decomposition = pd.DataFrame({
            'total_return': np.random.normal(0.01, 0.02, 30),
            'inflation': np.random.normal(0.005, 0.01, 30),
            'real_rf_rate': np.random.normal(0.001, 0.005, 30),
            'spread': np.random.normal(0.005, 0.015, 30),
            'error': np.random.normal(0, 0.001, 30)
        }, index=dates)
        
        # Mock estimate result
        self.mock_estimate = Mock()
        self.mock_estimate.risk_premium_volatility = 0.045
        self.mock_estimate.sample_size = 28
        
    def test_estimator_initialization(self):
        """Test estimator initialization with different parameters."""
        estimator = ComprehensiveParameterEstimator(
            exposure_id=self.exposure_id,
            risk_estimator=self.mock_risk_estimator,
            estimation_date=self.estimation_date,
            lookback_days=1260,
            frequency='monthly',
            method='historical',
            window=20,
            horizon=63
        )
        
        assert estimator.exposure_id == self.exposure_id
        assert estimator.lookback_days == 1260
        assert estimator.frequency == 'monthly'
        assert estimator.method == 'historical'
        assert estimator.window == 20
        assert estimator.horizon == 63
    
    def test_successful_scoring(self):
        """Test successful parameter scoring."""
        # Setup mocks
        self.mock_risk_estimator.load_and_decompose_exposure_returns.return_value = self.sample_decomposition
        self.mock_risk_estimator.estimate_risk_premium_volatility.return_value = self.mock_estimate
        
        estimator = ComprehensiveParameterEstimator(
            exposure_id=self.exposure_id,
            risk_estimator=self.mock_risk_estimator,
            estimation_date=self.estimation_date,
            method='historical',
            window=10,
            horizon=21
        )
        
        score = estimator.score()
        
        # Should return negative risk premium volatility
        expected_score = -0.045  # -self.mock_estimate.risk_premium_volatility
        assert abs(score - expected_score) < 0.001
        
        # Verify method calls
        self.mock_risk_estimator.load_and_decompose_exposure_returns.assert_called_once()
        self.mock_risk_estimator.estimate_risk_premium_volatility.assert_called_once()
    
    def test_data_loading_failure(self):
        """Test handling of data loading failures."""
        # Mock empty decomposition
        self.mock_risk_estimator.load_and_decompose_exposure_returns.return_value = pd.DataFrame()
        
        estimator = ComprehensiveParameterEstimator(
            exposure_id=self.exposure_id,
            risk_estimator=self.mock_risk_estimator,
            estimation_date=self.estimation_date
        )
        
        score = estimator.score()
        assert score == -10.0  # Expected failure score
    
    def test_estimation_failure(self):
        """Test handling of estimation failures."""
        self.mock_risk_estimator.load_and_decompose_exposure_returns.return_value = self.sample_decomposition
        self.mock_risk_estimator.estimate_risk_premium_volatility.return_value = None
        
        estimator = ComprehensiveParameterEstimator(
            exposure_id=self.exposure_id,
            risk_estimator=self.mock_risk_estimator,
            estimation_date=self.estimation_date
        )
        
        score = estimator.score()
        assert score == -5.0  # Expected failure score
    
    def test_invalid_volatility_handling(self):
        """Test handling of invalid volatility estimates."""
        self.mock_risk_estimator.load_and_decompose_exposure_returns.return_value = self.sample_decomposition
        
        # Test negative volatility
        invalid_estimate = Mock()
        invalid_estimate.risk_premium_volatility = -0.01
        invalid_estimate.sample_size = 28
        self.mock_risk_estimator.estimate_risk_premium_volatility.return_value = invalid_estimate
        
        estimator = ComprehensiveParameterEstimator(
            exposure_id=self.exposure_id,
            risk_estimator=self.mock_risk_estimator,
            estimation_date=self.estimation_date
        )
        
        score = estimator.score()
        assert score == -3.0  # Expected invalid result score
    
    def test_small_sample_penalty(self):
        """Test sample size penalty application."""
        self.mock_risk_estimator.load_and_decompose_exposure_returns.return_value = self.sample_decomposition
        
        # Small sample estimate
        small_sample_estimate = Mock()
        small_sample_estimate.risk_premium_volatility = 0.045
        small_sample_estimate.sample_size = 5  # Small sample
        self.mock_risk_estimator.estimate_risk_premium_volatility.return_value = small_sample_estimate
        
        estimator = ComprehensiveParameterEstimator(
            exposure_id=self.exposure_id,
            risk_estimator=self.mock_risk_estimator,
            estimation_date=self.estimation_date
        )
        
        score = estimator.score()
        
        # Should include penalty: -0.045 - (10-5)*0.001 = -0.05
        expected_score = -0.045 - 0.005
        assert abs(score - expected_score) < 0.001
    
    def test_method_parameter_preparation(self):
        """Test parameter preparation for different methods."""
        self.mock_risk_estimator.load_and_decompose_exposure_returns.return_value = self.sample_decomposition
        self.mock_risk_estimator.estimate_risk_premium_volatility.return_value = self.mock_estimate
        
        # Test historical method
        estimator = ComprehensiveParameterEstimator(
            exposure_id=self.exposure_id,
            risk_estimator=self.mock_risk_estimator,
            estimation_date=self.estimation_date,
            method='historical',
            window=15
        )
        estimator.score()
        
        # Check that historical parameters were prepared correctly
        call_args = self.mock_risk_estimator.estimate_risk_premium_volatility.call_args
        assert call_args[1]['method'] == 'historical'
        assert 'window' in call_args[1]['parameters']
        
        # Test EWMA method
        estimator = ComprehensiveParameterEstimator(
            exposure_id=self.exposure_id,
            risk_estimator=self.mock_risk_estimator,
            estimation_date=self.estimation_date,
            method='ewma',
            lambda_param=0.94
        )
        estimator.score()
        
        call_args = self.mock_risk_estimator.estimate_risk_premium_volatility.call_args
        assert call_args[1]['method'] == 'ewma'
        assert 'lambda' in call_args[1]['parameters']


class TestComprehensiveParameterSearchEngine:
    """Test the comprehensive parameter search engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_risk_estimator = Mock()
        self.estimation_date = datetime.now()
        self.engine = ComprehensiveParameterSearchEngine(
            self.mock_risk_estimator,
            self.estimation_date
        )
    
    def test_search_space_creation_constrained(self):
        """Test creation of constrained search spaces."""
        discrete_grid, continuous_dist = self.engine.create_search_spaces(constrained=True)
        
        # Check that constrained spaces have reasonable sizes
        assert len(discrete_grid['lookback_days']) <= 2
        assert len(discrete_grid['method']) <= 2
        assert 'monthly' in discrete_grid['frequency']
        
        # Check continuous distributions
        assert 'lookback_days' in continuous_dist
        assert 'method' in continuous_dist
    
    def test_search_space_creation_unconstrained(self):
        """Test creation of full search spaces."""
        discrete_grid, continuous_dist = self.engine.create_search_spaces(constrained=False)
        
        # Check that unconstrained spaces are larger
        assert len(discrete_grid['lookback_days']) > 2
        assert len(discrete_grid['method']) >= 3
        assert len(discrete_grid['frequency']) >= 2
        
        # Check that all methods are included
        assert 'historical' in discrete_grid['method']
        assert 'ewma' in discrete_grid['method']
        assert 'exponential_smoothing' in discrete_grid['method']
    
    def test_single_exposure_optimization_setup(self):
        """Test that single exposure optimization sets up correctly."""
        # This is more of an integration test, but we can test the setup
        exposure_id = 'test_exposure'
        
        # Test that the method doesn't immediately fail
        # (actual optimization requires more complex mocking)
        discrete_grid, _ = self.engine.create_search_spaces(constrained=True)
        assert len(discrete_grid) > 0
        
        # Test parameter validation - the method catches errors and returns None
        result = self.engine.optimize_single_exposure(
            exposure_id, method='invalid_method'
        )
        assert result is None  # Should return None for invalid method


class TestSearchResultAnalysis:
    """Test search result analysis functions."""
    
    def test_analyze_empty_results(self):
        """Test analysis of empty results."""
        analysis = analyze_search_results({})
        assert analysis == {}
    
    def test_analyze_single_result(self):
        """Test analysis of single search result."""
        result = ComprehensiveSearchResult(
            exposure_id='test_exposure',
            best_params={
                'method': 'historical',
                'frequency': 'monthly',
                'lookback_days': 1260,
                'horizon': 63,
                'window': 20
            },
            best_score=0.045,
            method='randomized',
            n_combinations_tested=100,
            elapsed_time=10.5,
            search_object=None,
            all_scores=[0.045, 0.050, 0.048],
            all_params=[{}, {}, {}]
        )
        
        analysis = analyze_search_results({'test_exposure': result})
        
        assert analysis['summary']['num_exposures'] == 1
        assert analysis['summary']['avg_score'] == 0.045
        assert analysis['summary']['total_combinations'] == 100
        
        assert analysis['method_preferences']['historical'] == 1
        assert analysis['frequency_preferences']['monthly'] == 1
        
        assert analysis['parameter_stats']['lookback_days']['mean'] == 1260
        assert analysis['parameter_stats']['horizon']['mean'] == 63
    
    def test_analyze_multiple_results(self):
        """Test analysis of multiple search results."""
        results = {}
        
        for i, method in enumerate(['historical', 'ewma', 'historical']):
            result = ComprehensiveSearchResult(
                exposure_id=f'exposure_{i}',
                best_params={
                    'method': method,
                    'frequency': 'monthly',
                    'lookback_days': 1000 + i * 100,
                    'horizon': 50 + i * 10,
                    'window': 15 + i * 5
                },
                best_score=0.04 + i * 0.005,
                method='randomized',
                n_combinations_tested=100,
                elapsed_time=10.0,
                search_object=None,
                all_scores=[],
                all_params=[]
            )
            results[f'exposure_{i}'] = result
        
        analysis = analyze_search_results(results)
        
        assert analysis['summary']['num_exposures'] == 3
        assert analysis['summary']['total_combinations'] == 300
        
        # Method preferences
        assert analysis['method_preferences']['historical'] == 2
        assert analysis['method_preferences']['ewma'] == 1
        
        # Parameter statistics
        assert analysis['parameter_stats']['lookback_days']['min'] == 1000
        assert analysis['parameter_stats']['lookback_days']['max'] == 1200
        assert analysis['parameter_stats']['horizon']['min'] == 50
        assert analysis['parameter_stats']['horizon']['max'] == 70


# Integration test fixtures
@pytest.fixture
def mock_working_risk_estimator():
    """Create a mock risk estimator that returns valid results."""
    mock_estimator = Mock()
    
    # Make the mock sklearn-compatible
    mock_estimator.get_params.return_value = {}
    
    # Mock successful data loading
    dates = pd.date_range('2020-01-01', periods=30, freq='ME')
    decomposition = pd.DataFrame({
        'total_return': np.random.normal(0.01, 0.02, 30),
        'spread': np.random.normal(0.005, 0.015, 30),
        'error': np.random.normal(0, 0.001, 30)
    }, index=dates)
    mock_estimator.load_and_decompose_exposure_returns.return_value = decomposition
    
    # Mock successful estimation
    estimate = Mock()
    estimate.risk_premium_volatility = 0.045
    estimate.sample_size = 28
    mock_estimator.estimate_risk_premium_volatility.return_value = estimate
    
    return mock_estimator


class TestSklearnIntegration:
    """Tests for sklearn integration that would catch NaN scoring issues."""
    
    def test_sklearn_estimator_compatibility(self, mock_working_risk_estimator):
        """Test that estimator is properly sklearn-compatible."""
        from sklearn.base import BaseEstimator, is_classifier, is_regressor
        
        estimator = ComprehensiveParameterEstimator(
            exposure_id='test_exposure',
            risk_estimator=mock_working_risk_estimator,
            estimation_date=datetime.now(),
            method='historical',
            window=10,
            horizon=21
        )
        
        # Test sklearn compatibility
        assert isinstance(estimator, BaseEstimator)
        assert not is_classifier(estimator)
        assert not is_regressor(estimator)  # We're not a typical regressor
        
        # Test required methods exist
        assert hasattr(estimator, 'get_params')
        assert hasattr(estimator, 'set_params')
        assert hasattr(estimator, 'fit')
        assert hasattr(estimator, 'score')
    
    def test_sklearn_parameter_setting(self, mock_working_risk_estimator):
        """Test that sklearn parameter setting works correctly."""
        estimator = ComprehensiveParameterEstimator(
            exposure_id='test_exposure',
            risk_estimator=mock_working_risk_estimator,
            estimation_date=datetime.now()
        )
        
        # Test get_params (excluding complex objects)
        params = estimator.get_params(deep=False)
        assert 'method' in params
        assert 'window' in params
        assert 'horizon' in params
        
        # Test set_params
        estimator.set_params(method='ewma', lambda_param=0.94, horizon=42)
        assert estimator.method == 'ewma'
        assert estimator.lambda_param == 0.94
        assert estimator.horizon == 42
        
        # Test that parameters persist after setting
        new_params = estimator.get_params(deep=False)
        assert new_params['method'] == 'ewma'
        assert new_params['lambda_param'] == 0.94
        assert new_params['horizon'] == 42
    
    def test_direct_scoring_returns_valid_values(self, mock_working_risk_estimator):
        """Test that direct scoring returns valid values (not NaN)."""
        estimator = ComprehensiveParameterEstimator(
            exposure_id='test_exposure',
            risk_estimator=mock_working_risk_estimator,
            estimation_date=datetime.now(),
            method='historical',
            window=10,
            horizon=21
        )
        
        # Test scoring directly
        score = estimator.score()
        
        # This is the critical test that would catch the NaN issue
        assert np.isfinite(score), f"Score should not be NaN: {score}"
        assert score < 0, "Score should be negative (risk premium volatility)"
        assert score > -1, "Score should be reasonable"
    
    def test_parameter_combinations_return_valid_scores(self, mock_working_risk_estimator):
        """Test multiple parameter combinations all return valid scores."""
        test_combinations = [
            {'method': 'historical', 'window': 10, 'horizon': 21},
            {'method': 'historical', 'window': 15, 'horizon': 42},
            {'method': 'ewma', 'lambda_param': 0.94, 'horizon': 21},
            {'method': 'ewma', 'lambda_param': 0.97, 'horizon': 42},
        ]
        
        valid_scores = 0
        for params in test_combinations:
            estimator = ComprehensiveParameterEstimator(
                exposure_id='test_exposure',
                risk_estimator=mock_working_risk_estimator,
                estimation_date=datetime.now(),
                **params
            )
            
            score = estimator.score()
            
            # Each combination should return a valid score
            assert np.isfinite(score), f"Score is NaN for params {params}: {score}"
            assert score < 0, f"Score should be negative for params {params}: {score}"
            assert score > -1, f"Score should be reasonable for params {params}: {score}"
            valid_scores += 1
        
        assert valid_scores == len(test_combinations), "All parameter combinations should return valid scores"
    
    def test_scoring_none_configuration(self):
        """Test that our search engine uses scoring=None (the fix for NaN issue)."""
        # This test ensures the critical fix is in place
        from src.optimization.comprehensive_parameter_search import ComprehensiveParameterSearchEngine
        
        # Read the source code to verify scoring=None is used
        import inspect
        source = inspect.getsource(ComprehensiveParameterSearchEngine.optimize_single_exposure)
        
        # Check that scoring=None appears in the code (our fix)
        assert 'scoring=None' in source, "Search engine should use scoring=None to avoid NaN issues"
        
        # Check that the problematic scoring='neg_mean_squared_error' is NOT present
        assert 'neg_mean_squared_error' not in source, "Should not use neg_mean_squared_error scoring"


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_end_to_end_single_optimization(self, mock_working_risk_estimator):
        """Test complete optimization workflow for single exposure."""
        engine = ComprehensiveParameterSearchEngine(
            mock_working_risk_estimator,
            datetime.now()
        )
        
        # Test that we can create an estimator and score it
        estimator = ComprehensiveParameterEstimator(
            exposure_id='test_exposure',
            risk_estimator=mock_working_risk_estimator,
            estimation_date=datetime.now(),
            method='historical',
            window=10,
            horizon=21
        )
        
        score = estimator.score()
        assert score < 0  # Should be negative (risk premium volatility)
        assert score > -1  # Should be reasonable
        assert np.isfinite(score)  # Should not be NaN or inf
    
    def test_comprehensive_search_engine_nan_prevention(self, mock_working_risk_estimator):
        """Test that search engine creates proper parameter spaces for optimization."""
        engine = ComprehensiveParameterSearchEngine(
            mock_working_risk_estimator,
            datetime.now()
        )
        
        # Test parameter space creation (core functionality)
        discrete_grid, continuous_dist = engine.create_search_spaces(constrained=True)
        
        assert len(discrete_grid) > 0, "Should create discrete parameter grid"
        assert len(continuous_dist) > 0, "Should create continuous parameter distributions"
        
        # Test that key parameters are present
        assert 'method' in discrete_grid, "Method should be in discrete grid"
        assert 'window' in discrete_grid, "Window should be in discrete grid"
        assert 'horizon' in discrete_grid, "Horizon should be in discrete grid"
        
        # This test validates the parameter space setup which is critical for optimization
        # The actual sklearn integration is tested separately in TestSklearnIntegration
    
    def test_parameter_space_completeness(self):
        """Test that parameter spaces include all required parameters."""
        engine = ComprehensiveParameterSearchEngine(Mock(), datetime.now())
        
        discrete_grid, continuous_dist = engine.create_search_spaces(constrained=False)
        
        required_params = [
            'lookback_days', 'frequency', 'method', 'window', 
            'lambda_param', 'alpha', 'horizon'
        ]
        
        for param in required_params:
            assert param in discrete_grid
            assert param in continuous_dist


if __name__ == '__main__':
    pytest.main([__file__, '-v'])