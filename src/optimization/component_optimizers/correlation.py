"""Correlation-specific parameter optimization."""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from itertools import product
from sklearn.model_selection import ParameterGrid

from .base import ComponentOptimizer, ComponentOptimalParameters

logger = logging.getLogger(__name__)


class CorrelationOptimizer(ComponentOptimizer):
    """
    Optimize parameters for correlation matrix estimation.
    
    Objectives:
    - Maximize temporal stability
    - Minimize condition number
    - Preserve eigenvalue structure
    """
    
    def __init__(self, risk_estimator, logger=None):
        """Initialize correlation optimizer.
        
        Args:
            risk_estimator: RiskPremiumEstimator instance
            logger: Optional logger instance
        """
        super().__init__(risk_estimator, logger)
        self.optimization_objectives = ['stability', 'condition_number', 'eigenvalue_stability']
        
    def get_optimization_objectives(self) -> List[str]:
        """Get list of optimization objectives for correlation."""
        return self.optimization_objectives
    
    def optimize_parameters(self, 
                          exposure_ids: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          n_splits: int = 5) -> Dict[str, ComponentOptimalParameters]:
        """Optimize correlation parameters.
        
        Note: Correlation optimization returns a single set of parameters
        that applies to all exposures, since correlation is estimated jointly.
        
        Args:
            exposure_ids: List of exposure IDs to optimize
            start_date: Start of optimization period
            end_date: End of optimization period
            n_splits: Number of time series cross-validation splits
            
        Returns:
            Dictionary with single entry for 'correlation' key
        """
        self._validate_time_period(start_date, end_date, min_days=1008)  # Need 3+ years for correlation
        
        self._log_info(f"Optimizing correlation parameters for {len(exposure_ids)} exposures")
        
        try:
            optimal_params = self._optimize_correlation_matrix(
                exposure_ids, start_date, end_date, n_splits
            )
            return {'correlation': optimal_params}
            
        except Exception as e:
            self._log_warning(f"Failed to optimize correlation parameters: {e}")
            return {'correlation': self._create_fallback_parameters()}
    
    def _optimize_correlation_matrix(self,
                                   exposure_ids: List[str],
                                   start_date: datetime,
                                   end_date: datetime,
                                   n_splits: int) -> ComponentOptimalParameters:
        """Optimize correlation matrix parameters."""
        
        # Get parameter grid
        param_grid = self._get_correlation_parameter_grid()
        
        # Create time splits for cross-validation
        time_splits = self._create_time_splits(start_date, end_date, n_splits)
        
        best_score = float('inf')  # We minimize instability
        best_params = None
        best_validation_metrics = {}
        
        # Grid search over parameters
        for params in ParameterGrid(param_grid):
            stability_scores = []
            condition_scores = []
            
            # Cross-validation over time splits
            for train_end, test_end in time_splits:
                try:
                    # Calculate correlation matrices for this split
                    train_start = train_end - timedelta(days=params['lookback_days'])
                    
                    # Get correlation matrix for each period
                    train_corr = self._estimate_correlation_matrix(
                        exposure_ids, train_start, train_end, params
                    )
                    
                    test_start = train_end
                    test_corr = self._estimate_correlation_matrix(
                        exposure_ids, test_start, test_end, params
                    )
                    
                    if train_corr is None or test_corr is None:
                        continue
                    
                    # Score parameters on this split
                    split_scores = self.score_parameters(
                        exposure_id='correlation',  # Not used for correlation
                        parameters=params,
                        train_data=train_corr,
                        test_data=test_corr
                    )
                    
                    if (split_scores.get('stability') is not None and 
                        not np.isnan(split_scores['stability'])):
                        stability_scores.append(split_scores['stability'])
                        
                    if (split_scores.get('condition_number') is not None and
                        not np.isnan(split_scores['condition_number'])):
                        condition_scores.append(split_scores['condition_number'])
                        
                except Exception as e:
                    self._log_warning(f"Error in correlation split: {e}")
                    continue
            
            # Combine scores (lower is better for both)
            if stability_scores and condition_scores:
                # Normalize and combine scores
                stability_score = np.mean(stability_scores)
                condition_score = np.mean(condition_scores)
                
                # Combined score (weighted average)
                combined_score = 0.7 * stability_score + 0.3 * np.log10(condition_score)
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_params = params.copy()
                    best_validation_metrics = {
                        'stability': stability_score,
                        'condition_number': condition_score,
                        'combined_score': combined_score,
                        'n_splits': len(stability_scores)
                    }
        
        if best_params is None:
            return self._create_fallback_parameters()
        
        return ComponentOptimalParameters(
            component_type='correlation',
            exposure_id='correlation',  # Single set of parameters for all exposures
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
        """Score correlation matrix stability and conditioning.
        
        Args:
            exposure_id: Not used for correlation (single matrix for all exposures)
            parameters: Parameter dictionary to score
            train_data: Training correlation matrix
            test_data: Test correlation matrix
            
        Returns:
            Dictionary of score_name -> score_value
        """
        try:
            if train_data is None or test_data is None:
                return {'stability': np.nan, 'condition_number': np.nan, 'eigenvalue_stability': np.nan}
            
            # Calculate temporal stability (Frobenius norm of difference)
            stability_score = self._calculate_matrix_stability(train_data, test_data)
            
            # Calculate condition number (numerical stability)
            condition_number = self._calculate_condition_number(test_data)
            
            # Calculate eigenvalue stability
            eigenvalue_stability = self._calculate_eigenvalue_stability(train_data, test_data)
            
            return {
                'stability': stability_score,
                'condition_number': condition_number,
                'eigenvalue_stability': eigenvalue_stability
            }
            
        except Exception as e:
            self._log_warning(f"Error scoring correlation parameters: {e}")
            return {'stability': np.nan, 'condition_number': np.nan, 'eigenvalue_stability': np.nan}
    
    def _estimate_correlation_matrix(self,
                                   exposure_ids: List[str],
                                   start_date: datetime,
                                   end_date: datetime,
                                   parameters: Dict[str, Any]) -> Optional[np.ndarray]:
        """Estimate correlation matrix for given parameters."""
        try:
            # Load data for all exposures
            exposure_returns = {}
            
            for exposure_id in exposure_ids:
                decomposition = self.risk_estimator.load_and_decompose_exposure_returns(
                    exposure_id=exposure_id,
                    estimation_date=end_date,
                    lookback_days=parameters['lookback_days'],
                    frequency=parameters['frequency']
                )
                
                if decomposition is not None and not decomposition.empty and 'spread' in decomposition.columns:
                    exposure_returns[exposure_id] = decomposition['spread']
            
            if len(exposure_returns) < 2:
                return None
            
            # Combine into DataFrame
            returns_df = pd.DataFrame(exposure_returns)
            
            # Handle missing data
            if parameters.get('min_periods'):
                returns_df = returns_df.dropna(thresh=parameters['min_periods'])
            else:
                returns_df = returns_df.dropna()
            
            if returns_df.empty or len(returns_df) < 10:
                return None
            
            # Calculate correlation matrix based on method
            if parameters['method'] == 'historical':
                corr_matrix = returns_df.corr().values
            elif parameters['method'] == 'ewma':
                # Exponentially weighted correlation
                decay_factor = parameters.get('decay_factor', 0.97)
                corr_matrix = returns_df.ewm(alpha=1-decay_factor).corr().iloc[-len(exposure_ids):].values
            elif parameters['method'] == 'robust':
                # Robust correlation (using median absolute deviation)
                from scipy.stats import spearmanr
                corr_matrix, _ = spearmanr(returns_df.values)
            else:
                # Default to sample correlation
                corr_matrix = returns_df.corr().values
            
            # Ensure positive definite
            corr_matrix = self._ensure_positive_definite(corr_matrix)
            
            return corr_matrix
            
        except Exception as e:
            self._log_warning(f"Error estimating correlation matrix: {e}")
            return None
    
    def _calculate_matrix_stability(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Calculate stability between two correlation matrices.
        
        Args:
            matrix1: First correlation matrix
            matrix2: Second correlation matrix
            
        Returns:
            Stability score (Frobenius norm of difference, lower is better)
        """
        if matrix1.shape != matrix2.shape:
            return np.nan
        
        return np.linalg.norm(matrix1 - matrix2, 'fro')
    
    def _calculate_condition_number(self, correlation_matrix: np.ndarray) -> float:
        """Calculate condition number for numerical stability.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Condition number (lower is better, ideally < 1000)
        """
        try:
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Remove near-zero eigenvalues
            
            if len(eigenvalues) == 0:
                return np.inf
            
            return np.max(eigenvalues) / np.min(eigenvalues)
            
        except Exception:
            return np.inf
    
    def _calculate_eigenvalue_stability(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Calculate eigenvalue stability between matrices.
        
        Args:
            matrix1: First correlation matrix
            matrix2: Second correlation matrix
            
        Returns:
            Eigenvalue stability score (lower is better)
        """
        try:
            eigs1 = np.sort(np.linalg.eigvals(matrix1))[::-1]  # Sort descending
            eigs2 = np.sort(np.linalg.eigvals(matrix2))[::-1]
            
            # Calculate normalized difference in eigenvalues
            return np.mean(np.abs(eigs1 - eigs2) / (np.abs(eigs1) + 1e-6))
            
        except Exception:
            return np.nan
    
    def _ensure_positive_definite(self, matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
        """Ensure matrix is positive definite by adjusting eigenvalues.
        
        Args:
            matrix: Input matrix
            min_eigenvalue: Minimum eigenvalue to enforce
            
        Returns:
            Positive definite matrix
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            
            # Clip eigenvalues to minimum value
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            
            # Reconstruct matrix
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
        except Exception:
            # Fallback: add small value to diagonal
            return matrix + min_eigenvalue * np.eye(matrix.shape[0])
    
    def _get_correlation_parameter_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid for correlation optimization."""
        return {
            'method': ['historical', 'ewma', 'robust'],
            'lookback_days': [756, 1008, 1260, 1512],  # 3-6 years (longer for correlation)
            'frequency': ['weekly', 'monthly'],  # Daily often too noisy for correlation
            'decay_factor': [0.94, 0.97, 0.99],  # For EWMA
            'min_periods': [60, 120, 180]  # Minimum periods for estimation
        }
    
    def _create_fallback_parameters(self) -> ComponentOptimalParameters:
        """Create fallback parameters when optimization fails."""
        return ComponentOptimalParameters(
            component_type='correlation',
            exposure_id='correlation',
            method='historical',
            parameters={'min_periods': 120},
            lookback_days=1008,  # 4 years
            frequency='monthly',
            score=np.nan,
            validation_metrics={'error': 'optimization_failed'}
        )