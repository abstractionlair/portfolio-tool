"""
Comprehensive Parameter Search for Risk Premium Estimation

This module provides sklearn-compatible parameter optimization that tests the complete
pipeline: data loading + decomposition + estimation for each parameter combination.

Unlike traditional approaches that only optimize estimation method parameters on fixed data,
this tests the full parameter space including data configuration parameters.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class ComprehensiveSearchResult:
    """Results from comprehensive parameter search."""
    exposure_id: str
    best_params: Dict[str, Any]
    best_score: float
    method: str
    n_combinations_tested: int
    elapsed_time: float
    search_object: Any
    all_scores: List[float]
    all_params: List[Dict[str, Any]]


class ComprehensiveParameterEstimator(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible estimator for comprehensive parameter validation.
    
    Tests the complete pipeline for each parameter combination:
    1. Data loading with specified lookback_days and frequency
    2. Return decomposition 
    3. Risk premium estimation with specified method/parameters
    4. Validation and scoring
    """
    
    def __init__(self, 
                 exposure_id: str = None,
                 risk_estimator = None,
                 estimation_date = None,
                 lookback_days: int = 1260,
                 frequency: str = 'monthly',
                 method: str = 'historical', 
                 window: int = 20,
                 lambda_param: float = 0.94,
                 alpha: float = 0.3,
                 horizon: int = 63):
        """
        Initialize comprehensive parameter estimator.
        
        Args:
            exposure_id: ID of exposure to optimize
            risk_estimator: Risk premium estimator instance
            estimation_date: Date for estimation
            lookback_days: Days of historical data to load
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            method: Estimation method ('historical', 'ewma', 'exponential_smoothing')
            window: Window size for historical method
            lambda_param: Decay parameter for EWMA
            alpha: Smoothing parameter for exponential smoothing
            horizon: Forecast horizon in days
        """
        self.exposure_id = exposure_id
        self.risk_estimator = risk_estimator
        self.estimation_date = estimation_date
        self.lookback_days = lookback_days
        self.frequency = frequency
        self.method = method
        self.window = window
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.horizon = horizon
        
    def fit(self, X=None, y=None):
        """Sklearn requires fit method."""
        return self
        
    def score(self, X=None, y=None) -> float:
        """
        Score the complete parameter combination.
        
        Returns:
            Negative risk premium volatility (sklearn maximizes scores)
        """
        if self.risk_estimator is None or self.exposure_id is None:
            return -1000.0
            
        try:
            # Step 1: Load and decompose data
            decomposition = self.risk_estimator.load_and_decompose_exposure_returns(
                exposure_id=self.exposure_id,
                estimation_date=self.estimation_date,
                lookback_days=self.lookback_days,
                frequency=self.frequency
            )
            
            if decomposition.empty or 'spread' not in decomposition.columns:
                return -10.0
            
            # Step 2: Prepare method parameters
            data_periods = len(decomposition)
            parameters = {}
            
            if self.method == 'historical':
                effective_window = min(int(self.window), max(3, data_periods - 1))
                parameters = {'window': effective_window}
            elif self.method == 'ewma':
                min_periods = min(5, max(3, data_periods // 3))
                parameters = {'lambda': self.lambda_param, 'min_periods': min_periods}
            elif self.method == 'exponential_smoothing':
                parameters = {'alpha': self.alpha}
            
            # Step 3: Estimate risk premium
            effective_horizon = min(self.horizon, max(1, data_periods // 4))
            
            estimate = self.risk_estimator.estimate_risk_premium_volatility(
                exposure_id=self.exposure_id,
                estimation_date=self.estimation_date,
                forecast_horizon=effective_horizon,
                method=self.method,
                parameters=parameters,
                lookback_days=self.lookback_days,
                frequency=self.frequency
            )
            
            if estimate is None:
                return -5.0
                
            # Step 4: Validate and score
            risk_premium_vol = estimate.risk_premium_volatility
            sample_size = estimate.sample_size
            
            if (risk_premium_vol <= 0 or risk_premium_vol > 1.0 or 
                not np.isfinite(risk_premium_vol) or sample_size < 3):
                return -3.0
                
            # Score based on risk premium volatility (lower is better)
            base_score = -risk_premium_vol
            
            # Small penalty for very small sample sizes
            if sample_size < 10:
                sample_penalty = (10 - sample_size) * 0.001
                base_score -= sample_penalty
            
            return base_score
            
        except Exception:
            return -1.0


class ComprehensiveParameterSearchEngine:
    """
    Engine for comprehensive parameter search across complete pipeline.
    """
    
    def __init__(self, risk_estimator, estimation_date):
        """
        Initialize search engine.
        
        Args:
            risk_estimator: Risk premium estimator instance
            estimation_date: Date for estimation
        """
        self.risk_estimator = risk_estimator
        self.estimation_date = estimation_date
        
    def create_search_spaces(self, 
                           constrained: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create parameter search spaces.
        
        Args:
            constrained: Whether to use constrained (stable) or full parameter ranges
            
        Returns:
            Tuple of (discrete_grid, continuous_distributions)
        """
        if constrained:
            # Constrained search space for stability
            discrete_grid = {
                'lookback_days': [1008, 1260],
                'frequency': ['monthly'],
                'method': ['historical', 'ewma'],
                'window': [10, 15, 20],
                'lambda_param': [0.94, 0.97],
                'alpha': [0.3],
                'horizon': [21, 42]
            }
            
            continuous_distributions = {
                'lookback_days': randint(1008, 1260),
                'frequency': ['monthly'],
                'method': ['historical', 'ewma'],
                'window': randint(8, 25),
                'lambda_param': uniform(0.92, 0.07),
                'alpha': [0.3],
                'horizon': randint(21, 63)
            }
        else:
            # Full search space for comprehensive optimization
            discrete_grid = {
                'lookback_days': [504, 756, 1008, 1260],
                'frequency': ['weekly', 'monthly'],
                'method': ['historical', 'ewma', 'exponential_smoothing'],
                'window': [10, 20, 40, 60],
                'lambda_param': [0.90, 0.94, 0.97, 0.99],
                'alpha': [0.1, 0.3, 0.5, 0.7],
                'horizon': [21, 42, 63, 126]
            }
            
            continuous_distributions = {
                'lookback_days': randint(252, 1260),
                'frequency': ['weekly', 'monthly'],
                'method': ['historical', 'ewma', 'exponential_smoothing'],
                'window': randint(5, 100),
                'lambda_param': uniform(0.85, 0.14),
                'alpha': uniform(0.05, 0.95),
                'horizon': randint(21, 252)
            }
        
        return discrete_grid, continuous_distributions
    
    def optimize_single_exposure(self, 
                               exposure_id: str,
                               method: str = 'randomized',
                               n_iter: int = 100,
                               constrained: bool = True,
                               n_jobs: int = -1,
                               random_state: int = 42) -> Optional[ComprehensiveSearchResult]:
        """
        Optimize parameters for a single exposure.
        
        Args:
            exposure_id: Exposure to optimize
            method: 'grid' or 'randomized'
            n_iter: Number of iterations for randomized search
            constrained: Whether to use constrained parameter space
            n_jobs: Number of parallel jobs
            random_state: Random seed
            
        Returns:
            ComprehensiveSearchResult or None if failed
        """
        discrete_grid, continuous_distributions = self.create_search_spaces(constrained)
        
        estimator = ComprehensiveParameterEstimator(
            exposure_id=exposure_id,
            risk_estimator=self.risk_estimator,
            estimation_date=self.estimation_date
        )
        
        start_time = time.time()
        
        try:
            if method == 'grid':
                search = GridSearchCV(
                    estimator,
                    discrete_grid,
                    cv=None,  # No cross-validation - we do internal validation
                    n_jobs=min(n_jobs, 4),  # Limit parallelism for grid search
                    scoring=None  # Use estimator's score method
                )
                # Create dummy data for sklearn
                dummy_X = [[0] for _ in range(5)]
                search.fit(dummy_X)
                n_tested = len(search.cv_results_['params'])
                
            elif method == 'randomized':
                search = RandomizedSearchCV(
                    estimator,
                    continuous_distributions,
                    n_iter=n_iter,
                    cv=None,
                    n_jobs=n_jobs,
                    scoring=None,  # Use estimator's score method
                    random_state=random_state
                )
                dummy_X = [[0] for _ in range(5)]
                search.fit(dummy_X)
                n_tested = n_iter
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            elapsed_time = time.time() - start_time
            
            # Extract all scores and parameters for analysis
            all_scores = [-score for score in search.cv_results_['mean_test_score']]
            all_params = search.cv_results_['params']
            
            return ComprehensiveSearchResult(
                exposure_id=exposure_id,
                best_params=search.best_params_,
                best_score=-search.best_score_,
                method=method,
                n_combinations_tested=n_tested,
                elapsed_time=elapsed_time,
                search_object=search,
                all_scores=all_scores,
                all_params=all_params
            )
            
        except Exception as e:
            print(f"Optimization failed for {exposure_id}: {e}")
            return None
    
    def optimize_multiple_exposures(self, 
                                  exposure_ids: List[str],
                                  method: str = 'randomized',
                                  n_iter: int = 100,
                                  constrained: bool = True,
                                  n_jobs: int = -1) -> Dict[str, ComprehensiveSearchResult]:
        """
        Optimize parameters for multiple exposures.
        
        Args:
            exposure_ids: List of exposures to optimize
            method: 'grid' or 'randomized'
            n_iter: Number of iterations for randomized search
            constrained: Whether to use constrained parameter space
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary mapping exposure_id to ComprehensiveSearchResult
        """
        results = {}
        
        for i, exposure_id in enumerate(exposure_ids, 1):
            print(f"Optimizing {i}/{len(exposure_ids)}: {exposure_id}")
            
            result = self.optimize_single_exposure(
                exposure_id=exposure_id,
                method=method,
                n_iter=n_iter,
                constrained=constrained,
                n_jobs=n_jobs
            )
            
            if result:
                results[exposure_id] = result
                print(f"  ✅ Best score: {result.best_score:.6f}, "
                      f"Time: {result.elapsed_time:.1f}s, "
                      f"Tests: {result.n_combinations_tested}")
            else:
                print(f"  ❌ Failed")
        
        return results
    
    def compare_search_methods(self, 
                             exposure_id: str,
                             n_iter: int = 50,
                             constrained: bool = True) -> Dict[str, ComprehensiveSearchResult]:
        """
        Compare different search methods on the same exposure.
        
        Args:
            exposure_id: Exposure to test
            n_iter: Number of iterations for randomized search
            constrained: Whether to use constrained parameter space
            
        Returns:
            Dictionary mapping method name to ComprehensiveSearchResult
        """
        results = {}
        
        # Test grid search
        print(f"Testing Grid Search on {exposure_id}...")
        grid_result = self.optimize_single_exposure(
            exposure_id, method='grid', constrained=constrained, n_jobs=1
        )
        if grid_result:
            results['grid'] = grid_result
        
        # Test randomized search
        print(f"Testing Randomized Search on {exposure_id}...")
        random_result = self.optimize_single_exposure(
            exposure_id, method='randomized', n_iter=n_iter, constrained=constrained
        )
        if random_result:
            results['randomized'] = random_result
        
        return results


def analyze_search_results(results: Dict[str, ComprehensiveSearchResult]) -> Dict[str, Any]:
    """
    Analyze comprehensive search results across multiple exposures.
    
    Args:
        results: Dictionary mapping exposure_id to ComprehensiveSearchResult
        
    Returns:
        Dictionary with analysis results
    """
    if not results:
        return {}
    
    # Aggregate statistics
    all_scores = [result.best_score for result in results.values()]
    all_times = [result.elapsed_time for result in results.values()]
    all_tests = [result.n_combinations_tested for result in results.values()]
    
    # Parameter frequency analysis
    method_counts = {}
    frequency_counts = {}
    lookback_stats = []
    horizon_stats = []
    
    for result in results.values():
        params = result.best_params
        
        # Method preferences
        method = params.get('method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
        
        # Frequency preferences
        freq = params.get('frequency', 'unknown')
        frequency_counts[freq] = frequency_counts.get(freq, 0) + 1
        
        # Parameter distributions
        if 'lookback_days' in params:
            lookback_stats.append(params['lookback_days'])
        if 'horizon' in params:
            horizon_stats.append(params['horizon'])
    
    return {
        'summary': {
            'num_exposures': len(results),
            'avg_score': np.mean(all_scores),
            'score_std': np.std(all_scores),
            'avg_time': np.mean(all_times),
            'total_combinations': sum(all_tests)
        },
        'method_preferences': method_counts,
        'frequency_preferences': frequency_counts,
        'parameter_stats': {
            'lookback_days': {
                'min': min(lookback_stats) if lookback_stats else None,
                'max': max(lookback_stats) if lookback_stats else None,
                'mean': np.mean(lookback_stats) if lookback_stats else None
            },
            'horizon': {
                'min': min(horizon_stats) if horizon_stats else None,
                'max': max(horizon_stats) if horizon_stats else None,
                'mean': np.mean(horizon_stats) if horizon_stats else None
            }
        }
    }