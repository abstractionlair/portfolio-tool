"""Expected return-specific parameter optimization."""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from itertools import product
from sklearn.model_selection import ParameterGrid

from .base import ComponentOptimizer, ComponentOptimalParameters

logger = logging.getLogger(__name__)


class ExpectedReturnOptimizer(ComponentOptimizer):
    """
    Optimize parameters for expected return estimation.
    
    Objectives:
    - Maximize directional accuracy
    - Maximize information ratio
    - Minimize bias
    """
    
    def __init__(self, risk_estimator, logger=None):
        """Initialize expected return optimizer.
        
        Args:
            risk_estimator: RiskPremiumEstimator instance
            logger: Optional logger instance
        """
        super().__init__(risk_estimator, logger)
        self.optimization_objectives = ['directional_accuracy', 'information_ratio', 'bias']
        self.return_models = ['historical', 'ewma', 'momentum', 'mean_reversion']
        
    def get_optimization_objectives(self) -> List[str]:
        """Get list of optimization objectives for expected returns."""
        return self.optimization_objectives
    
    def optimize_parameters(self, 
                          exposure_ids: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          n_splits: int = 5) -> Dict[str, ComponentOptimalParameters]:
        """Optimize expected return parameters for each exposure.
        
        Args:
            exposure_ids: List of exposure IDs to optimize
            start_date: Start of optimization period
            end_date: End of optimization period
            n_splits: Number of time series cross-validation splits
            
        Returns:
            Dictionary mapping exposure_id to optimal parameters
        """
        self._validate_time_period(start_date, end_date, min_days=504)  # Need 2+ years for returns
        
        results = {}
        
        for exposure_id in exposure_ids:
            self._log_info(f"Optimizing expected return parameters for {exposure_id}")
            
            try:
                optimal_params = self._optimize_single_exposure(
                    exposure_id, start_date, end_date, n_splits
                )
                results[exposure_id] = optimal_params
                
            except Exception as e:
                self._log_warning(f"Failed to optimize {exposure_id}: {e}")
                # Create fallback parameters
                results[exposure_id] = self._create_fallback_parameters(exposure_id)
        
        return results
    
    def _optimize_single_exposure(self,
                                exposure_id: str,
                                start_date: datetime,
                                end_date: datetime,
                                n_splits: int) -> ComponentOptimalParameters:
        """Optimize parameters for a single exposure."""
        
        # Get parameter grid
        param_grid = self._get_return_parameter_grid()
        
        # Create time splits for cross-validation
        time_splits = self._create_time_splits(start_date, end_date, n_splits)
        
        best_score = -np.inf  # We maximize directional accuracy
        best_params = None
        best_validation_metrics = {}
        
        # Grid search over parameters
        for params in ParameterGrid(param_grid):
            directional_scores = []
            info_ratio_scores = []
            bias_scores = []
            
            # Cross-validation over time splits
            for train_end, test_end in time_splits:
                try:
                    # Load data for this split
                    train_start = train_end - timedelta(days=params['lookback_days'])
                    
                    decomposition = self.risk_estimator.load_and_decompose_exposure_returns(
                        exposure_id=exposure_id,
                        estimation_date=train_end,
                        lookback_days=params['lookback_days'],
                        frequency=params['frequency']
                    )
                    
                    if decomposition is None or decomposition.empty or 'spread' not in decomposition.columns:
                        continue
                    
                    # Test data for validation
                    test_decomposition = self.risk_estimator.load_and_decompose_exposure_returns(
                        exposure_id=exposure_id,
                        estimation_date=test_end,
                        lookback_days=params['lookback_days'],
                        frequency=params['frequency']
                    )
                    
                    if test_decomposition is None or test_decomposition.empty or 'spread' not in test_decomposition.columns:
                        continue
                    
                    # Score parameters on this split
                    split_scores = self.score_parameters(
                        exposure_id=exposure_id,
                        parameters=params,
                        train_data=decomposition,
                        test_data=test_decomposition
                    )
                    
                    if (split_scores.get('directional_accuracy') is not None and 
                        not np.isnan(split_scores['directional_accuracy'])):
                        directional_scores.append(split_scores['directional_accuracy'])
                        
                    if (split_scores.get('information_ratio') is not None and
                        not np.isnan(split_scores['information_ratio'])):
                        info_ratio_scores.append(split_scores['information_ratio'])
                        
                    if (split_scores.get('bias') is not None and
                        not np.isnan(split_scores['bias'])):
                        bias_scores.append(split_scores['bias'])
                        
                except Exception as e:
                    self._log_warning(f"Error in split for {exposure_id}: {e}")
                    continue
            
            # Combine scores (higher directional accuracy is better)
            if directional_scores:
                # Primary objective: directional accuracy
                avg_directional = np.mean(directional_scores)
                
                # Secondary objectives
                avg_info_ratio = np.mean(info_ratio_scores) if info_ratio_scores else 0.0
                avg_bias = np.mean(np.abs(bias_scores)) if bias_scores else 0.0
                
                # Combined score (weighted combination)
                combined_score = (0.6 * avg_directional + 
                                0.3 * min(avg_info_ratio, 2.0) -  # Cap info ratio contribution
                                0.1 * min(avg_bias, 1.0))  # Penalize bias
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_params = params.copy()
                    best_validation_metrics = {
                        'directional_accuracy': avg_directional,
                        'information_ratio': avg_info_ratio,
                        'bias': avg_bias,
                        'combined_score': combined_score,
                        'n_splits': len(directional_scores)
                    }
        
        if best_params is None:
            return self._create_fallback_parameters(exposure_id)
        
        return ComponentOptimalParameters(
            component_type='expected_returns',
            exposure_id=exposure_id,
            method=best_params['method'],
            parameters={k: v for k, v in best_params.items() if k not in ['method', 'lookback_days', 'frequency']},
            lookback_days=best_params['lookback_days'],
            frequency=best_params['frequency'],
            score=best_score,
            validation_metrics=best_validation_metrics
        )
    
    def score_parameters(self,
                        exposure_id: str,
                        parameters: Dict[str, Any],
                        train_data: Any,
                        test_data: Any) -> Dict[str, float]:
        """Score expected return prediction accuracy.
        
        Args:
            exposure_id: Exposure being evaluated
            parameters: Parameter dictionary to score
            train_data: Training dataset (ExposureDecomposition)
            test_data: Test dataset (ExposureDecomposition)
            
        Returns:
            Dictionary of score_name -> score_value
        """
        try:
            # Generate return prediction based on training data
            expected_return = self._estimate_expected_return(
                train_data.exposure_returns, parameters
            )
            
            if np.isnan(expected_return):
                return {'directional_accuracy': np.nan, 'information_ratio': np.nan, 'bias': np.nan}
            
            # Get actual returns from test period
            if 'spread' not in test_data.columns:
                return {'directional_accuracy': np.nan, 'information_ratio': np.nan, 'bias': np.nan}
            
            test_returns = test_data['spread']
            if test_returns.empty:
                return {'directional_accuracy': np.nan, 'information_ratio': np.nan, 'bias': np.nan}
            
            actual_return = test_returns.mean()  # Average return over test period
            
            # Calculate directional accuracy
            directional_accuracy = self._calculate_directional_accuracy(expected_return, actual_return)
            
            # Calculate information ratio (prediction accuracy relative to volatility)
            information_ratio = self._calculate_information_ratio(
                expected_return, actual_return, test_returns.std()
            )
            
            # Calculate bias (systematic over/under prediction)
            bias = expected_return - actual_return
            
            return {
                'directional_accuracy': directional_accuracy,
                'information_ratio': information_ratio,
                'bias': bias,
                'expected_return': expected_return,
                'actual_return': actual_return
            }
            
        except Exception as e:
            self._log_warning(f"Error scoring return parameters for {exposure_id}: {e}")
            return {'directional_accuracy': np.nan, 'information_ratio': np.nan, 'bias': np.nan}
    
    def _estimate_expected_return(self, returns: pd.Series, parameters: Dict[str, Any]) -> float:
        """Estimate expected return using specified method and parameters.
        
        Args:
            returns: Historical return series
            parameters: Method parameters
            
        Returns:
            Expected return estimate
        """
        if returns.empty or len(returns) < 5:
            return np.nan
        
        method = parameters['method']
        
        try:
            if method == 'historical':
                # Simple historical average
                return returns.mean()
                
            elif method == 'ewma':
                # Exponentially weighted moving average
                decay_factor = parameters.get('decay_factor', 0.97)
                weights = np.array([decay_factor ** i for i in range(len(returns))][::-1])
                weights = weights / weights.sum()
                return np.dot(returns.values, weights)
                
            elif method == 'momentum':
                # Momentum-based prediction
                momentum_period = parameters.get('momentum_period', 12)
                if len(returns) < momentum_period:
                    return returns.mean()
                
                recent_returns = returns.iloc[-momentum_period:]
                momentum_signal = recent_returns.mean()
                
                # Scale momentum signal
                momentum_strength = parameters.get('momentum_strength', 0.5)
                base_return = returns.mean()
                return base_return + momentum_strength * momentum_signal
                
            elif method == 'mean_reversion':
                # Mean reversion model
                long_term_mean = returns.mean()
                recent_period = parameters.get('recent_period', 6)
                
                if len(returns) < recent_period:
                    return long_term_mean
                
                recent_mean = returns.iloc[-recent_period:].mean()
                reversion_strength = parameters.get('reversion_strength', 0.3)
                
                # Expect reversion toward long-term mean
                return long_term_mean + reversion_strength * (long_term_mean - recent_mean)
                
            else:
                # Default to historical mean
                return returns.mean()
                
        except Exception:
            return np.nan
    
    def _calculate_directional_accuracy(self, prediction: float, actual: float) -> float:
        """Calculate directional accuracy of prediction.
        
        Args:
            prediction: Predicted return
            actual: Actual return
            
        Returns:
            Directional accuracy (1.0 if correct direction, 0.0 if wrong, 0.5 if either is zero)
        """
        if np.isnan(prediction) or np.isnan(actual):
            return np.nan
        
        # Handle case where prediction or actual is near zero
        threshold = 1e-6
        if abs(prediction) < threshold or abs(actual) < threshold:
            return 0.5
        
        # Check if signs match
        return 1.0 if (prediction > 0) == (actual > 0) else 0.0
    
    def _calculate_information_ratio(self, prediction: float, actual: float, volatility: float) -> float:
        """Calculate information ratio (risk-adjusted prediction accuracy).
        
        Args:
            prediction: Predicted return
            actual: Actual return
            volatility: Return volatility
            
        Returns:
            Information ratio
        """
        if np.isnan(prediction) or np.isnan(actual) or np.isnan(volatility) or volatility <= 0:
            return np.nan
        
        # Calculate prediction error
        prediction_error = abs(prediction - actual)
        
        # Information ratio: inverse of normalized prediction error
        normalized_error = prediction_error / volatility
        
        # Return inverse (higher is better, capped at reasonable values)
        if normalized_error < 1e-6:
            return 10.0  # Very good prediction
        else:
            return min(1.0 / normalized_error, 10.0)
    
    def _get_return_parameter_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid for expected return optimization."""
        return {
            'method': ['historical', 'ewma', 'momentum', 'mean_reversion'],
            'lookback_days': [252, 504, 756],  # 1-3 years (shorter for returns)
            'frequency': ['monthly', 'weekly'],  # Daily often too noisy
            'decay_factor': [0.9, 0.95, 0.99],  # For EWMA
            'momentum_period': [3, 6, 12],  # For momentum model
            'momentum_strength': [0.2, 0.5, 0.8],  # Momentum signal strength
            'recent_period': [3, 6, 12],  # For mean reversion
            'reversion_strength': [0.1, 0.3, 0.5],  # Mean reversion strength
            'min_periods': [20, 40, 60]  # Minimum periods for estimation
        }
    
    def _create_fallback_parameters(self, exposure_id: str) -> ComponentOptimalParameters:
        """Create fallback parameters when optimization fails."""
        return ComponentOptimalParameters(
            component_type='expected_returns',
            exposure_id=exposure_id,
            method='historical',
            parameters={'min_periods': 40},
            lookback_days=504,  # 2 years
            frequency='monthly',
            score=np.nan,
            validation_metrics={'error': 'optimization_failed'}
        )