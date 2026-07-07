"""
Portfolio-level parameter optimization.
Finds optimal forecast horizon and parameters for accurate portfolio risk prediction.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')

try:
    from ..data.total_returns import TotalReturnFetcher
    from ..data.exposure_universe import ExposureUniverse
except ImportError:
    from data.total_returns import TotalReturnFetcher
    from data.exposure_universe import ExposureUniverse

logger = logging.getLogger(__name__)

@dataclass
class HorizonOptimizationResult:
    """Results for a specific horizon."""
    horizon: int
    volatility_params: Dict[str, Dict]  # Per-exposure vol params
    correlation_params: Dict  # Global correlation params
    return_params: Dict[str, Dict]  # Per-exposure return params
    goodness_score: float
    validation_metrics: Dict


class PortfolioLevelOptimizer:
    """
    Two-level optimization:
    1. For each horizon, optimize all parameters
    2. Select horizon with best portfolio-level accuracy
    """
    
    def __init__(self, exposure_universe: ExposureUniverse, component_optimizers: Dict = None):
        self.universe = exposure_universe
        self.fetcher = TotalReturnFetcher()
        
        # For now, we'll implement optimization logic directly
        # In future, we can use component_optimizers if needed
        self.component_optimizers = component_optimizers or {}
        
    def optimize_all_horizons(
        self,
        candidate_horizons: List[int] = None,
        test_portfolios: Optional[List[Dict[str, float]]] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict:
        """
        Main entry point: Find optimal horizon and all parameters.
        
        Args:
            candidate_horizons: Horizons to test (days)
            test_portfolios: Portfolio weights to test goodness
            start_date: Start of historical data
            end_date: End of historical data
            
        Returns:
            Complete optimization results with selected horizon
        """
        
        if candidate_horizons is None:
            candidate_horizons = [5, 21, 63]
            
        if start_date is None:
            start_date = datetime(2018, 1, 1)
        if end_date is None:
            end_date = datetime(2024, 12, 31)
        
        if test_portfolios is None:
            test_portfolios = self._get_default_test_portfolios()
        
        logger.info(f"Testing horizons: {candidate_horizons}")
        logger.info(f"Using {len(test_portfolios)} test portfolios")
        
        horizon_results = {}
        
        # Level 1: Optimize for each horizon
        for horizon in candidate_horizons:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing for {horizon}-day horizon")
            logger.info(f"{'='*60}")
            
            result = self._optimize_for_horizon(
                horizon, test_portfolios, start_date, end_date
            )
            horizon_results[horizon] = result
            
            logger.info(f"Horizon {horizon} goodness score: {result.goodness_score:.6f}")
        
        # Level 2: Select best horizon
        best_horizon = max(
            horizon_results.keys(),
            key=lambda h: horizon_results[h].goodness_score
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMAL HORIZON: {best_horizon} days")
        logger.info(f"Best goodness score: {horizon_results[best_horizon].goodness_score:.6f}")
        
        return {
            'optimal_horizon': best_horizon,
            'optimal_parameters': horizon_results[best_horizon],
            'all_horizon_results': horizon_results,
            'test_portfolios': test_portfolios
        }
    
    def _optimize_for_horizon(
        self,
        horizon: int,
        test_portfolios: List[Dict[str, float]],
        start_date: datetime,
        end_date: datetime
    ) -> HorizonOptimizationResult:
        """Optimize all parameters for a specific horizon."""
        
        # Step 1: Optimize volatility for each exposure
        logger.info(f"Optimizing volatility parameters for {horizon}-day horizon...")
        volatility_params = self._optimize_volatilities_for_horizon(
            horizon, start_date, end_date
        )
        
        # Step 2: Optimize correlation parameters
        logger.info(f"Optimizing correlation parameters for {horizon}-day horizon...")
        correlation_params = self._optimize_correlation_for_horizon(
            horizon, volatility_params, start_date, end_date
        )
        
        # Step 3: Optimize return prediction parameters
        logger.info(f"Optimizing return prediction parameters for {horizon}-day horizon...")
        return_params = self._optimize_returns_for_horizon(
            horizon, volatility_params, start_date, end_date
        )
        
        # Step 4: Compute goodness score using test portfolios
        logger.info(f"Computing goodness score using {len(test_portfolios)} portfolios...")
        goodness_score, validation_metrics = self._compute_goodness_score(
            horizon, volatility_params, correlation_params, return_params,
            test_portfolios, start_date, end_date
        )
        
        return HorizonOptimizationResult(
            horizon=horizon,
            volatility_params=volatility_params,
            correlation_params=correlation_params,
            return_params=return_params,
            goodness_score=goodness_score,
            validation_metrics=validation_metrics
        )
    
    def _optimize_volatilities_for_horizon(
        self,
        horizon: int,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict]:
        """
        Optimize volatility parameters for each exposure independently.
        Key: Parameters are optimized to predict h-day ahead volatility.
        """
        
        volatility_params = {}
        exposures = list(self.universe.exposures.keys())
        
        for exp_id in exposures:
            logger.debug(f"Optimizing {exp_id} for {horizon}-day volatility...")
            
            # Get exposure data
            try:
                returns, _ = self.fetcher.fetch_returns_for_exposure(
                    self.universe.exposures[exp_id],
                    start_date=start_date,
                    end_date=end_date,
                    frequency='daily'
                )
                
                if returns is None or len(returns) < 100:
                    logger.warning(f"Insufficient data for {exp_id}, using fallback parameters")
                    volatility_params[exp_id] = self._get_fallback_volatility_params(horizon)
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to fetch data for {exp_id}: {e}")
                volatility_params[exp_id] = self._get_fallback_volatility_params(horizon)
                continue
            
            # Test different methods and parameters
            best_method = None
            best_params = None
            best_score = float('inf')
            
            # Test each method
            for method in ['historical', 'ewma', 'garch']:
                try:
                    if method == 'historical':
                        param_grid = {
                            'lookback_days': [252, 504, 756],
                            'min_periods': [30, 60, 120]
                        }
                    elif method == 'ewma':
                        param_grid = {
                            'lambda': [0.90, 0.94, 0.97, 0.99],
                            'lookback_days': [252, 504, 756],
                            'min_periods': [30, 60]
                        }
                    elif method == 'garch':
                        # Simplified GARCH for now
                        param_grid = {
                            'p': [1], 'q': [1],
                            'lookback_days': [504, 756]
                        }
                    
                    # Grid search for this method
                    method_score, method_params = self._grid_search_volatility(
                        exp_id, method, param_grid, horizon, returns
                    )
                    
                    if method_score < best_score:
                        best_score = method_score
                        best_method = method
                        best_params = method_params
                        
                except Exception as e:
                    logger.debug(f"Method {method} failed for {exp_id}: {e}")
                    continue
            
            if best_method is None:
                logger.warning(f"No method worked for {exp_id}, using fallback")
                volatility_params[exp_id] = self._get_fallback_volatility_params(horizon)
            else:
                volatility_params[exp_id] = {
                    'method': best_method,
                    'parameters': best_params,
                    'horizon': horizon,
                    'validation_score': best_score
                }
                
                logger.debug(f"  Best for {exp_id}: {best_method} (score: {best_score:.6f})")
        
        return volatility_params
    
    def _grid_search_volatility(
        self,
        exp_id: str,
        method: str,
        param_grid: Dict,
        horizon: int,
        returns: pd.Series
    ) -> Tuple[float, Dict]:
        """Grid search for best parameters for a specific method."""
        
        best_score = float('inf')
        best_params = None
        
        # Generate all parameter combinations
        param_combinations = []
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)
        
        # Test each combination with walk-forward validation
        for params in param_combinations:
            try:
                score = self._validate_volatility_params(
                    method, params, horizon, returns
                )
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.debug(f"Parameter combination failed for {exp_id}: {params}, error: {e}")
                continue
        
        return best_score, best_params
    
    def _validate_volatility_params(
        self,
        method: str,
        params: Dict,
        horizon: int,
        returns: pd.Series
    ) -> float:
        """Validate volatility parameters using walk-forward validation."""
        
        min_train_periods = max(params.get('lookback_days', 252), 100)
        test_periods = 30
        
        if len(returns) < min_train_periods + horizon + test_periods:
            return float('inf')
        
        scores = []
        
        # Walk-forward validation
        for i in range(min_train_periods, len(returns) - horizon - test_periods, horizon):
            train_data = returns.iloc[i-min_train_periods:i]
            
            # Predict volatility using the method
            if method == 'historical':
                predicted_vol = train_data.std() * np.sqrt(252 / horizon)
                
            elif method == 'ewma':
                lambda_param = params['lambda']
                ewma_vol = train_data.ewm(alpha=1-lambda_param, min_periods=params['min_periods']).std()
                if len(ewma_vol) > 0 and not pd.isna(ewma_vol.iloc[-1]):
                    predicted_vol = ewma_vol.iloc[-1] * np.sqrt(252 / horizon)
                else:
                    continue
                    
            elif method == 'garch':
                # Simplified GARCH - use EWMA as proxy for now
                predicted_vol = train_data.ewm(alpha=0.06, min_periods=30).std().iloc[-1] * np.sqrt(252 / horizon)
            
            # Calculate realized volatility over horizon
            future_data = returns.iloc[i:i+horizon]
            if len(future_data) == horizon:
                realized_vol = future_data.std() * np.sqrt(252 / horizon)
                
                # MSE
                if not pd.isna(predicted_vol) and not pd.isna(realized_vol):
                    error = (predicted_vol - realized_vol) ** 2
                    scores.append(error)
        
        if len(scores) == 0:
            return float('inf')
        
        return np.mean(scores)
    
    def _optimize_correlation_for_horizon(
        self,
        horizon: int,
        volatility_params: Dict,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Optimize correlation parameters for the given horizon.
        Uses standardized returns (returns / predicted volatility).
        """
        
        # Test different correlation estimation methods
        param_combinations = [
            {'method': 'historical', 'lookback_days': 252, 'min_periods': 60},
            {'method': 'historical', 'lookback_days': 504, 'min_periods': 120},
            {'method': 'ewma', 'lambda': 0.94, 'lookback_days': 504, 'min_periods': 120},
            {'method': 'ewma', 'lambda': 0.97, 'lookback_days': 504, 'min_periods': 120},
            {'method': 'ewma', 'lambda': 0.99, 'lookback_days': 756, 'min_periods': 120},
        ]
        
        best_score = float('inf')
        best_params = None
        
        # Get exposure data for correlation estimation
        exposure_returns = {}
        exposures = list(volatility_params.keys())
        
        for exp_id in exposures:
            try:
                returns, _ = self.fetcher.fetch_returns_for_exposure(
                    self.universe.exposures[exp_id],
                    start_date=start_date,
                    end_date=end_date,
                    frequency='daily'
                )
                if returns is not None and len(returns) > 100:
                    exposure_returns[exp_id] = returns
            except Exception as e:
                logger.debug(f"Failed to fetch {exp_id} for correlation: {e}")
        
        if len(exposure_returns) < 2:
            logger.warning("Insufficient data for correlation optimization, using fallback")
            return {
                'parameters': {'method': 'ewma', 'lambda': 0.97, 'lookback_days': 504, 'min_periods': 120},
                'horizon': horizon,
                'validation_score': 0.1
            }
        
        # Test each parameter combination
        for params in param_combinations:
            try:
                score = self._validate_correlation_params(
                    params, horizon, exposure_returns
                )
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.debug(f"Correlation parameter validation failed: {params}, error: {e}")
                continue
        
        if best_params is None:
            best_params = {'method': 'ewma', 'lambda': 0.97, 'lookback_days': 504, 'min_periods': 120}
            best_score = 0.1
        
        logger.debug(f"Best correlation params: {best_params} (score: {best_score:.6f})")
        
        return {
            'parameters': best_params,
            'horizon': horizon,
            'validation_score': best_score
        }
    
    def _validate_correlation_params(
        self,
        params: Dict,
        horizon: int,
        exposure_returns: Dict[str, pd.Series]
    ) -> float:
        """Validate correlation parameters using walk-forward validation."""
        
        # Align all return series
        returns_df = pd.DataFrame(exposure_returns).dropna()
        
        if len(returns_df) < 200:
            return float('inf')
        
        min_train_periods = params.get('lookback_days', 504)
        scores = []
        
        # Walk-forward validation
        for i in range(min_train_periods, len(returns_df) - horizon, horizon):
            train_data = returns_df.iloc[i-min_train_periods:i]
            
            # Estimate correlation matrix
            if params['method'] == 'historical':
                corr_matrix = train_data.corr()
            elif params['method'] == 'ewma':
                lambda_param = params['lambda']
                corr_matrix = train_data.ewm(alpha=1-lambda_param, min_periods=params['min_periods']).corr().iloc[-len(train_data.columns):, :]
            
            # Calculate realized correlation over horizon
            future_data = returns_df.iloc[i:i+horizon]
            if len(future_data) == horizon:
                realized_corr = future_data.corr()
                
                # Frobenius norm of difference
                if not corr_matrix.isna().any().any() and not realized_corr.isna().any().any():
                    diff_matrix = corr_matrix - realized_corr
                    frobenius_norm = np.sqrt((diff_matrix ** 2).sum().sum())
                    scores.append(frobenius_norm)
        
        if len(scores) == 0:
            return float('inf')
        
        return np.mean(scores)
    
    def _compute_goodness_score(
        self,
        horizon: int,
        volatility_params: Dict,
        correlation_params: Dict,
        return_params: Dict,
        test_portfolios: List[Dict[str, float]],
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[float, Dict]:
        """
        Compute portfolio-level goodness score including return prediction accuracy.
        This is the key innovation: we measure how well the parameters
        predict portfolio-level risk, not just individual asset risk.
        """
        
        # Get all exposure returns for portfolio analysis
        exposure_returns = {}
        exposures = list(volatility_params.keys())
        
        for exp_id in exposures:
            try:
                returns, _ = self.fetcher.fetch_returns_for_exposure(
                    self.universe.exposures[exp_id],
                    start_date=start_date,
                    end_date=end_date,
                    frequency='daily'
                )
                if returns is not None and len(returns) > 100:
                    exposure_returns[exp_id] = returns
            except Exception as e:
                logger.debug(f"Failed to fetch {exp_id} for goodness score: {e}")
        
        if len(exposure_returns) < 2:
            logger.warning("Insufficient data for goodness score computation")
            return -float('inf'), {'error': 'Insufficient data'}
        
        # Align return series
        returns_df = pd.DataFrame(exposure_returns).dropna()
        
        # Walk-forward validation
        validation_window = 252  # 1 year of data for estimation
        portfolio_scores = []
        detailed_metrics = []
        
        for i in range(validation_window + horizon, len(returns_df), horizon):
            train_start_idx = i - validation_window - horizon
            train_end_idx = i - horizon
            test_start_idx = i - horizon
            test_end_idx = i
            
            train_data = returns_df.iloc[train_start_idx:train_end_idx]
            test_data = returns_df.iloc[test_start_idx:test_end_idx]
            
            # For each test portfolio
            for portfolio_weights in test_portfolios:
                # Filter portfolio to available exposures
                filtered_weights = {}
                for exp_id, weight in portfolio_weights.items():
                    if exp_id in returns_df.columns:
                        filtered_weights[exp_id] = weight
                
                if len(filtered_weights) < 2:
                    continue
                
                # Normalize weights
                total_weight = sum(filtered_weights.values())
                if total_weight > 0:
                    filtered_weights = {k: v/total_weight for k, v in filtered_weights.items()}
                else:
                    continue
                
                # Predict portfolio volatility
                predicted_vol = self._predict_portfolio_volatility(
                    filtered_weights,
                    volatility_params,
                    correlation_params,
                    train_data,
                    horizon
                )
                
                # Calculate realized portfolio volatility
                realized_vol = self._calculate_realized_portfolio_volatility(
                    filtered_weights,
                    test_data
                )
                
                # Predict portfolio return
                predicted_return = self._predict_portfolio_return(
                    filtered_weights,
                    return_params,
                    train_data,
                    horizon
                )
                
                # Calculate realized portfolio return
                realized_return = self._calculate_realized_portfolio_return(
                    filtered_weights,
                    test_data
                )
                
                if predicted_vol is not None and realized_vol is not None and predicted_return is not None and realized_return is not None:
                    # Volatility prediction error (MSE)
                    vol_error = (predicted_vol - realized_vol) ** 2
                    
                    # Return prediction accuracy (directional accuracy)
                    return_accuracy = self._calculate_directional_accuracy(
                        predicted_return, realized_return
                    )
                    
                    # Combined score: 70% volatility accuracy, 30% return accuracy
                    # (lower vol_error is better, higher return_accuracy is better)
                    combined_score = -0.7 * vol_error + 0.3 * return_accuracy
                    portfolio_scores.append(combined_score)
                    
                    detailed_metrics.append({
                        'test_period': i,
                        'portfolio': filtered_weights,
                        'predicted_vol': predicted_vol,
                        'realized_vol': realized_vol,
                        'predicted_return': predicted_return,
                        'realized_return': realized_return,
                        'vol_error': vol_error,
                        'return_accuracy': return_accuracy,
                        'combined_score': combined_score
                    })
        
        # Goodness = average combined score (higher is better)
        if portfolio_scores:
            goodness = np.mean(portfolio_scores)
            
            # Calculate separate metrics for volatility and return prediction
            vol_errors = [m['vol_error'] for m in detailed_metrics if 'vol_error' in m]
            return_accuracies = [m['return_accuracy'] for m in detailed_metrics if 'return_accuracy' in m]
            
            validation_metrics = {
                'combined_score': goodness,
                'vol_rmse': np.sqrt(np.mean(vol_errors)) if vol_errors else None,
                'return_accuracy': np.mean(return_accuracies) if return_accuracies else None,
                'n_tests': len(portfolio_scores),
                'detailed_metrics': detailed_metrics[:10]  # Save first 10 for analysis
            }
        else:
            goodness = -float('inf')
            validation_metrics = {'error': 'No valid test periods'}
        
        return goodness, validation_metrics
    
    def _predict_portfolio_volatility(
        self,
        weights: Dict[str, float],
        volatility_params: Dict,
        correlation_params: Dict,
        train_data: pd.DataFrame,
        horizon: int
    ) -> Optional[float]:
        """
        Predict portfolio volatility using individual vol predictions and correlation.
        
        Portfolio variance = w' * Σ * w
        where Σ = D * C * D (D = diag(vols), C = correlation matrix)
        """
        
        exposures = list(weights.keys())
        vol_predictions = {}
        
        # Get individual volatility predictions
        for exp_id in exposures:
            if exp_id in volatility_params and exp_id in train_data.columns:
                vol_pred = self._predict_single_volatility(
                    exp_id,
                    volatility_params[exp_id],
                    train_data[exp_id],
                    horizon
                )
                if vol_pred is not None:
                    vol_predictions[exp_id] = vol_pred
        
        if len(vol_predictions) < len(exposures):
            return None
        
        # Get correlation prediction
        corr_matrix = self._predict_correlation_matrix(
            exposures,
            correlation_params,
            train_data,
            horizon
        )
        
        if corr_matrix is None:
            return None
        
        # Calculate portfolio variance
        weight_vector = np.array([weights[exp] for exp in exposures])
        vol_vector = np.array([vol_predictions[exp] for exp in exposures])
        
        # Covariance = D * C * D
        D = np.diag(vol_vector)
        cov_matrix = D @ corr_matrix @ D
        
        # Portfolio variance
        portfolio_var = weight_vector @ cov_matrix @ weight_vector
        portfolio_vol = np.sqrt(max(portfolio_var, 0))  # Ensure non-negative
        
        return portfolio_vol
    
    def _predict_single_volatility(
        self,
        exp_id: str,
        params: Dict,
        returns: pd.Series,
        horizon: int
    ) -> Optional[float]:
        """Predict single exposure volatility using optimized parameters."""
        
        method = params['method']
        param_dict = params['parameters']
        
        try:
            if method == 'historical':
                lookback = param_dict.get('lookback_days', 252)
                recent_data = returns.tail(lookback)
                if len(recent_data) >= param_dict.get('min_periods', 30):
                    vol = recent_data.std() * np.sqrt(252 / horizon)
                    return vol
                    
            elif method == 'ewma':
                lambda_param = param_dict.get('lambda', 0.94)
                min_periods = param_dict.get('min_periods', 30)
                ewma_vol = returns.ewm(alpha=1-lambda_param, min_periods=min_periods).std()
                if len(ewma_vol) > 0 and not pd.isna(ewma_vol.iloc[-1]):
                    vol = ewma_vol.iloc[-1] * np.sqrt(252 / horizon)
                    return vol
                    
            elif method == 'garch':
                # Simplified GARCH using EWMA as proxy
                ewma_vol = returns.ewm(alpha=0.06, min_periods=30).std()
                if len(ewma_vol) > 0 and not pd.isna(ewma_vol.iloc[-1]):
                    vol = ewma_vol.iloc[-1] * np.sqrt(252 / horizon)
                    return vol
                    
        except Exception as e:
            logger.debug(f"Failed to predict volatility for {exp_id}: {e}")
        
        return None
    
    def _predict_correlation_matrix(
        self,
        exposures: List[str],
        correlation_params: Dict,
        train_data: pd.DataFrame,
        horizon: int
    ) -> Optional[np.ndarray]:
        """Predict correlation matrix using optimized parameters."""
        
        params = correlation_params['parameters']
        method = params['method']
        
        try:
            # Filter to available exposures
            available_data = train_data[exposures].dropna()
            
            if len(available_data) < params.get('min_periods', 60):
                return None
            
            if method == 'historical':
                corr_matrix = available_data.corr().values
            elif method == 'ewma':
                lambda_param = params.get('lambda', 0.97)
                corr_matrix = available_data.ewm(alpha=1-lambda_param, min_periods=params['min_periods']).corr().iloc[-len(exposures):, :].values
            
            # Ensure valid correlation matrix
            if np.isnan(corr_matrix).any():
                return None
            
            # Ensure positive definite
            eigenvals = np.linalg.eigvals(corr_matrix)
            if np.min(eigenvals) < 1e-8:
                # Add small regularization to diagonal
                corr_matrix += np.eye(len(corr_matrix)) * 1e-6
            
            return corr_matrix
            
        except Exception as e:
            logger.debug(f"Failed to predict correlation matrix: {e}")
            return None
    
    def _calculate_realized_portfolio_volatility(
        self,
        weights: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Optional[float]:
        """Calculate realized portfolio volatility over the test period."""
        
        try:
            # Calculate portfolio returns
            exposures = list(weights.keys())
            available_data = returns_data[exposures].dropna()
            
            if len(available_data) == 0:
                return None
            
            weight_vector = np.array([weights[exp] for exp in exposures])
            portfolio_returns = available_data.values @ weight_vector
            
            # Calculate volatility
            portfolio_vol = np.std(portfolio_returns) * np.sqrt(252 / len(available_data))
            
            return portfolio_vol
            
        except Exception as e:
            logger.debug(f"Failed to calculate realized portfolio volatility: {e}")
            return None
    
    def _predict_portfolio_return(
        self,
        weights: Dict[str, float],
        return_params: Dict,
        train_data: pd.DataFrame,
        horizon: int
    ) -> Optional[float]:
        """Predict portfolio return using individual return predictions."""
        
        try:
            exposures = list(weights.keys())
            return_predictions = {}
            
            # Get individual return predictions
            for exp_id in exposures:
                if exp_id in return_params and exp_id in train_data.columns:
                    params = return_params[exp_id]
                    predicted_return = self._estimate_expected_return(
                        train_data[exp_id],
                        params['method'],
                        params['parameters']
                    )
                    if not pd.isna(predicted_return):
                        return_predictions[exp_id] = predicted_return
            
            if len(return_predictions) < len(exposures):
                return None
            
            # Calculate portfolio return as weighted average
            portfolio_return = sum(
                weights[exp] * return_predictions[exp]
                for exp in exposures
            )
            
            return portfolio_return
            
        except Exception as e:
            logger.debug(f"Failed to predict portfolio return: {e}")
            return None
    
    def _calculate_realized_portfolio_return(
        self,
        weights: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Optional[float]:
        """Calculate realized portfolio return over the test period."""
        
        try:
            # Calculate portfolio returns
            exposures = list(weights.keys())
            available_data = returns_data[exposures].dropna()
            
            if len(available_data) == 0:
                return None
            
            weight_vector = np.array([weights[exp] for exp in exposures])
            portfolio_returns = available_data.values @ weight_vector
            
            # Calculate average return
            portfolio_return = np.mean(portfolio_returns)
            
            return portfolio_return
            
        except Exception as e:
            logger.debug(f"Failed to calculate realized portfolio return: {e}")
            return None
    
    def _get_default_test_portfolios(self) -> List[Dict[str, float]]:
        """Generate test portfolios for goodness evaluation."""
        
        exposures = list(self.universe.exposures.keys())
        test_portfolios = []
        
        # 1. Equal-weighted
        equal_weight = 1.0 / len(exposures)
        test_portfolios.append({exp: equal_weight for exp in exposures})
        
        # 2. Traditional 60/40 (simplified)
        equity_exposures = [e for e in exposures if 'equity' in e]
        bond_exposures = [e for e in exposures if 'ust' in e or 'bond' in e]
        
        if equity_exposures and bond_exposures:
            trad_portfolio = {exp: 0.0 for exp in exposures}
            for exp in equity_exposures:
                trad_portfolio[exp] = 0.6 / len(equity_exposures)
            for exp in bond_exposures:
                trad_portfolio[exp] = 0.4 / len(bond_exposures)
            test_portfolios.append(trad_portfolio)
        
        # 3. Alternatives-heavy
        alt_exposures = [e for e in exposures if e in 
                        ['trend_following', 'factor_style_equity', 'factor_style_other']]
        if alt_exposures:
            alt_portfolio = {exp: 0.0 for exp in exposures}
            remaining_exposures = [e for e in exposures if e not in alt_exposures]
            
            for exp in alt_exposures:
                alt_portfolio[exp] = 0.5 / len(alt_exposures)
            for exp in remaining_exposures:
                alt_portfolio[exp] = 0.5 / len(remaining_exposures)
            test_portfolios.append(alt_portfolio)
        
        logger.info(f"Created {len(test_portfolios)} test portfolios")
        return test_portfolios
    
    def _optimize_returns_for_horizon(
        self,
        horizon: int,
        volatility_params: Dict,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict]:
        """
        Optimize return prediction parameters for each exposure.
        """
        
        exposures = list(volatility_params.keys())
        return_params = {}
        
        logger.info(f"Optimizing returns for {len(exposures)} exposures")
        
        # Return prediction methods to test
        methods = ['historical', 'ewma', 'momentum', 'mean_reversion']
        
        for exp_id in exposures:
            logger.debug(f"Optimizing returns for {exp_id}")
            
            try:
                # Get exposure data
                returns, _ = self.fetcher.fetch_returns_for_exposure(
                    self.universe.exposures[exp_id],
                    start_date=start_date,
                    end_date=end_date,
                    frequency='daily'
                )
                
                if returns is None or len(returns) < 252:
                    logger.warning(f"Insufficient data for {exp_id}, using fallback")
                    return_params[exp_id] = self._get_fallback_return_params(horizon)
                    continue
                
                # Test different return prediction methods
                best_score = -float('inf')  # Higher directional accuracy is better
                best_method = None
                best_params = None
                
                for method in methods:
                    try:
                        if method == 'historical':
                            param_grid = {
                                'lookback_days': [126, 252, 504],
                                'min_periods': [30, 60]
                            }
                        elif method == 'ewma':
                            param_grid = {
                                'decay_factor': [0.90, 0.95, 0.99],
                                'lookback_days': [252, 504],
                                'min_periods': [30, 60]
                            }
                        elif method == 'momentum':
                            param_grid = {
                                'momentum_period': [3, 6, 12],
                                'momentum_strength': [0.2, 0.5, 0.8],
                                'lookback_days': [252, 504],
                                'min_periods': [30, 60]
                            }
                        elif method == 'mean_reversion':
                            param_grid = {
                                'recent_period': [3, 6, 12],
                                'reversion_strength': [0.1, 0.3, 0.5],
                                'lookback_days': [252, 504],
                                'min_periods': [30, 60]
                            }
                        
                        # Grid search for this method
                        method_score, method_params = self._grid_search_returns(
                            exp_id, method, param_grid, horizon, returns
                        )
                        
                        if method_score > best_score:
                            best_score = method_score
                            best_method = method
                            best_params = method_params
                            
                    except Exception as e:
                        logger.debug(f"Method {method} failed for {exp_id}: {e}")
                        continue
                
                if best_method is None:
                    logger.warning(f"No method worked for {exp_id}, using fallback")
                    return_params[exp_id] = self._get_fallback_return_params(horizon)
                else:
                    return_params[exp_id] = {
                        'method': best_method,
                        'parameters': best_params,
                        'horizon': horizon,
                        'validation_score': best_score
                    }
                    
                    logger.debug(f"  Best for {exp_id}: {best_method} (score: {best_score:.3f})")
                
            except Exception as e:
                logger.warning(f"Failed to optimize returns for {exp_id}: {e}")
                return_params[exp_id] = self._get_fallback_return_params(horizon)
        
        return return_params
    
    def _grid_search_returns(
        self,
        exp_id: str,
        method: str,
        param_grid: Dict,
        horizon: int,
        returns: pd.Series
    ) -> Tuple[float, Dict]:
        """Grid search for best return prediction parameters."""
        
        best_score = -float('inf')
        best_params = None
        
        # Generate all parameter combinations
        param_combinations = []
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)
        
        # Test each combination with walk-forward validation
        for params in param_combinations:
            try:
                score = self._validate_return_params(
                    method, params, horizon, returns
                )
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.debug(f"Parameter combination failed for {exp_id}: {params}, error: {e}")
                continue
        
        return best_score, best_params
    
    def _validate_return_params(
        self,
        method: str,
        params: Dict,
        horizon: int,
        returns: pd.Series
    ) -> float:
        """Validate return prediction parameters using walk-forward validation."""
        
        min_train_periods = max(params.get('lookback_days', 252), 100)
        test_periods = 30
        
        if len(returns) < min_train_periods + horizon + test_periods:
            return -float('inf')
        
        directional_accuracies = []
        
        # Walk-forward validation
        for i in range(min_train_periods, len(returns) - horizon - test_periods, horizon):
            train_data = returns.iloc[i-min_train_periods:i]
            
            # Predict expected return using the method
            predicted_return = self._estimate_expected_return(train_data, method, params)
            
            if pd.isna(predicted_return):
                continue
            
            # Calculate actual return over horizon
            future_data = returns.iloc[i:i+horizon]
            if len(future_data) == horizon:
                actual_return = future_data.mean()
                
                # Calculate directional accuracy
                if not pd.isna(actual_return):
                    directional_accuracy = self._calculate_directional_accuracy(
                        predicted_return, actual_return
                    )
                    directional_accuracies.append(directional_accuracy)
        
        if len(directional_accuracies) == 0:
            return -float('inf')
        
        return np.mean(directional_accuracies)
    
    def _estimate_expected_return(
        self,
        returns: pd.Series,
        method: str,
        params: Dict
    ) -> float:
        """Estimate expected return using specified method and parameters."""
        
        if returns.empty or len(returns) < 5:
            return np.nan
        
        try:
            if method == 'historical':
                # Simple historical average
                return returns.mean()
                
            elif method == 'ewma':
                # Exponentially weighted moving average
                decay_factor = params.get('decay_factor', 0.97)
                weights = np.array([decay_factor ** i for i in range(len(returns))][::-1])
                weights = weights / weights.sum()
                return np.dot(returns.values, weights)
                
            elif method == 'momentum':
                # Momentum-based prediction
                momentum_period = params.get('momentum_period', 12)
                if len(returns) < momentum_period:
                    return returns.mean()
                
                recent_returns = returns.iloc[-momentum_period:]
                momentum_signal = recent_returns.mean()
                
                # Scale momentum signal
                momentum_strength = params.get('momentum_strength', 0.5)
                base_return = returns.mean()
                return base_return + momentum_strength * momentum_signal
                
            elif method == 'mean_reversion':
                # Mean reversion model
                long_term_mean = returns.mean()
                recent_period = params.get('recent_period', 6)
                
                if len(returns) < recent_period:
                    return long_term_mean
                
                recent_mean = returns.iloc[-recent_period:].mean()
                reversion_strength = params.get('reversion_strength', 0.3)
                
                # Expect reversion toward long-term mean
                return long_term_mean + reversion_strength * (long_term_mean - recent_mean)
                
            else:
                # Default to historical mean
                return returns.mean()
                
        except Exception:
            return np.nan
    
    def _calculate_directional_accuracy(self, prediction: float, actual: float) -> float:
        """Calculate directional accuracy of prediction."""
        
        if np.isnan(prediction) or np.isnan(actual):
            return np.nan
        
        # Handle case where prediction or actual is near zero
        threshold = 1e-6
        if abs(prediction) < threshold or abs(actual) < threshold:
            return 0.5
        
        # Check if signs match
        return 1.0 if (prediction > 0) == (actual > 0) else 0.0
    
    def _get_fallback_return_params(self, horizon: int) -> Dict:
        """Get fallback parameters when return optimization fails."""
        return {
            'method': 'historical',
            'parameters': {
                'lookback_days': 252,
                'min_periods': 60
            },
            'horizon': horizon,
            'validation_score': 0.5  # 50% directional accuracy
        }
    
    def _get_fallback_volatility_params(self, horizon: int) -> Dict:
        """Get fallback parameters when optimization fails."""
        return {
            'method': 'historical',
            'parameters': {
                'lookback_days': 252,
                'min_periods': 60
            },
            'horizon': horizon,
            'validation_score': 0.1
        }