"""Volatility-specific parameter optimization."""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from itertools import product
from sklearn.model_selection import ParameterGrid

from .base import ComponentOptimizer, ComponentOptimalParameters

logger = logging.getLogger(__name__)


class VolatilityOptimizer(ComponentOptimizer):
    """
    Optimize parameters for volatility forecasting.
    
    Objectives:
    - Minimize MSE of volatility forecasts
    - Maximize realized volatility correlation
    - Minimize QLIKE score
    """
    
    def __init__(self, risk_estimator, logger=None):
        """Initialize volatility optimizer.
        
        Args:
            risk_estimator: RiskPremiumEstimator instance
            logger: Optional logger instance
        """
        super().__init__(risk_estimator, logger)
        self.optimization_objectives = ['mse', 'qlike', 'realized_correlation']
        
    def get_optimization_objectives(self) -> List[str]:
        """Get list of optimization objectives for volatility."""
        return self.optimization_objectives
    
    def optimize_parameters(self, 
                          exposure_ids: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          n_splits: int = 5) -> Dict[str, ComponentOptimalParameters]:
        """Optimize volatility parameters for each exposure.
        
        Args:
            exposure_ids: List of exposure IDs to optimize
            start_date: Start of optimization period
            end_date: End of optimization period
            n_splits: Number of time series cross-validation splits
            
        Returns:
            Dictionary mapping exposure_id to optimal parameters
        """
        self._validate_time_period(start_date, end_date, min_days=756)  # Need 2+ years for volatility
        
        results = {}
        
        for exposure_id in exposure_ids:
            self._log_info(f"Optimizing volatility parameters for {exposure_id}")
            
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
        param_grid = self._get_volatility_parameter_grid()
        
        # Create time splits for cross-validation
        time_splits = self._create_time_splits(start_date, end_date, n_splits)
        
        best_score = float('inf')  # We minimize MSE
        best_params = None
        best_validation_metrics = {}
        
        # Grid search over parameters
        for params in ParameterGrid(param_grid):
            scores = []
            
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
                    
                    if split_scores.get('mse') is not None and not np.isnan(split_scores['mse']):
                        scores.append(split_scores['mse'])
                        
                except Exception as e:
                    self._log_warning(f"Error in split for {exposure_id}: {e}")
                    continue
            
            # Average score across splits
            if scores:
                avg_score = np.mean(scores)
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params.copy()
                    best_validation_metrics = {
                        'mse': avg_score,
                        'n_splits': len(scores),
                        'score_std': np.std(scores)
                    }
        
        if best_params is None:
            return self._create_fallback_parameters(exposure_id)
        
        return ComponentOptimalParameters(
            component_type='volatility',
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
        """Score volatility forecasting accuracy.
        
        Args:
            exposure_id: Exposure being evaluated
            parameters: Parameter dictionary to score
            train_data: Training dataset (ExposureDecomposition)
            test_data: Test dataset (ExposureDecomposition)
            
        Returns:
            Dictionary of score_name -> score_value
        """
        try:
            # Estimate volatility using training data
            vol_estimate = self.risk_estimator.estimate_risk_premium_volatility(
                exposure_id=exposure_id,
                estimation_date=datetime.now(),
                method=parameters['method'],
                parameters={k: v for k, v in parameters.items() if k not in ['method']},
                lookback_days=parameters.get('lookback_days', 756),
                frequency=parameters.get('frequency', 'daily')
            )
            
            if vol_estimate is None or np.isnan(vol_estimate.risk_premium_volatility):
                return {'mse': np.nan, 'qlike': np.nan, 'realized_correlation': np.nan}
            
            # Extract the risk premium volatility value
            vol_value = vol_estimate.risk_premium_volatility
            
            # Calculate realized volatility from test data
            if 'spread' not in test_data.columns:
                return {'mse': np.nan, 'qlike': np.nan, 'realized_correlation': np.nan}
            
            test_returns = test_data['spread']
            if test_returns.empty:
                return {'mse': np.nan, 'qlike': np.nan, 'realized_correlation': np.nan}
            
            realized_vol = self._calculate_realized_volatility(test_returns, horizon=21)
            
            if np.isnan(realized_vol):
                return {'mse': np.nan, 'qlike': np.nan, 'realized_correlation': np.nan}
            
            # Calculate MSE (primary objective)
            mse = (vol_value - realized_vol) ** 2
            
            # Calculate QLIKE score
            qlike = self._calculate_qlike(vol_value, realized_vol)
            
            return {
                'mse': mse,
                'qlike': qlike,
                'realized_correlation': 1.0,  # Placeholder - would need multiple periods
                'forecast': vol_value,
                'realized': realized_vol
            }
            
        except Exception as e:
            self._log_warning(f"Error scoring parameters for {exposure_id}: {e}")
            return {'mse': np.nan, 'qlike': np.nan, 'realized_correlation': np.nan}
    
    def _calculate_realized_volatility(self, returns: pd.Series, horizon: int = 21) -> float:
        """Calculate realized volatility from returns.
        
        Args:
            returns: Time series of returns
            horizon: Forecast horizon in business days
            
        Returns:
            Annualized realized volatility
        """
        if returns.empty or len(returns) < 5:
            return np.nan
        
        # Calculate sample volatility
        volatility = returns.std()
        
        # Annualize based on frequency
        if 'daily' in str(returns.index.freq).lower():
            # Daily data
            annualized_vol = volatility * np.sqrt(252)
        elif 'monthly' in str(returns.index.freq).lower() or returns.index.freq == 'ME':
            # Monthly data
            annualized_vol = volatility * np.sqrt(12)
        elif 'weekly' in str(returns.index.freq).lower():
            # Weekly data
            annualized_vol = volatility * np.sqrt(52)
        else:
            # Default to daily assumption
            annualized_vol = volatility * np.sqrt(252)
        
        return annualized_vol
    
    def _calculate_qlike(self, forecast: float, realized: float) -> float:
        """Calculate QLIKE score (quasi-likelihood).
        
        Args:
            forecast: Forecasted volatility
            realized: Realized volatility
            
        Returns:
            QLIKE score (lower is better)
        """
        if forecast <= 0 or realized <= 0:
            return np.nan
        
        # QLIKE = log(forecast) + realized/forecast
        return np.log(forecast) + realized / forecast
    
    def _get_volatility_parameter_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid for volatility optimization."""
        return {
            'method': ['historical', 'ewma', 'garch'],
            'lookback_days': [504, 756, 1008, 1260],  # 2-5 years
            'frequency': ['daily', 'weekly', 'monthly'],
            'decay_factor': [0.94, 0.97, 0.99],  # For EWMA
            'min_periods': [30, 60, 120]  # Minimum periods for estimation
        }
    
    def _create_fallback_parameters(self, exposure_id: str) -> ComponentOptimalParameters:
        """Create fallback parameters when optimization fails."""
        return ComponentOptimalParameters(
            component_type='volatility',
            exposure_id=exposure_id,
            method='historical',
            parameters={'min_periods': 60},
            lookback_days=756,  # 3 years
            frequency='monthly',
            score=np.nan,
            validation_metrics={'error': 'optimization_failed'}
        )