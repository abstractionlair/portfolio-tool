"""
Production interface for risk estimation using pre-optimized parameters.

This is the main entry point for downstream code that needs risk estimates.
It automatically loads and uses the optimal parameters discovered through
component-specific optimization.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

from .component_optimizers import UnifiedOptimalParameters, ComponentOptimalParameters
from .risk_premium_estimator import RiskPremiumEstimator
from ..data.exposure_universe import ExposureUniverse

logger = logging.getLogger(__name__)


class OptimizedRiskEstimator:
    """
    Main production interface using pre-optimized parameters.
    
    This class provides a simple, clean API for getting risk estimates
    using the optimal parameters discovered through component-specific
    optimization.
    
    Example:
        # Just works - loads optimal parameters automatically
        estimator = OptimizedRiskEstimator()
        
        # Get covariance matrix for portfolio optimization
        cov_matrix = estimator.get_covariance_matrix(
            ['us_large_equity', 'bonds', 'commodities'],
            datetime.now()
        )
    """
    
    def __init__(self, 
                 parameter_file: str = "config/optimal_parameters.yaml",
                 risk_estimator: Optional[RiskPremiumEstimator] = None,
                 auto_optimize: bool = False,
                 auto_optimize_if_stale_days: int = 180):
        """Initialize with optimal parameters.
        
        Args:
            parameter_file: Path to optimized parameters YAML file
            risk_estimator: Optional pre-configured RiskPremiumEstimator
            auto_optimize: Whether to run optimization if parameters missing
            auto_optimize_if_stale_days: Re-optimize if params older than this
        """
        self.parameter_file = Path(parameter_file)
        self.risk_estimator = risk_estimator or self._create_default_estimator()
        self.auto_optimize = auto_optimize
        self.auto_optimize_if_stale_days = auto_optimize_if_stale_days
        
        # Load optimal parameters
        self.optimal_params = self._load_or_optimize_parameters()
        
    def _create_default_estimator(self) -> RiskPremiumEstimator:
        """Create default risk estimator."""
        try:
            from ..data.exposure_universe import ExposureUniverse
            from ..data.return_decomposition import ReturnDecomposer
            
            # Try multiple paths for the config file
            config_paths = [
                'config/exposure_universe.yaml',
                '../config/exposure_universe.yaml',
                './config/exposure_universe.yaml'
            ]
            
            universe = None
            last_error = None
            
            for config_path in config_paths:
                try:
                    universe = ExposureUniverse.from_yaml(config_path)
                    logger.info(f"Loaded exposure universe from {config_path}")
                    break
                except Exception as path_error:
                    last_error = path_error
                    continue
            
            if universe is None:
                raise Exception(f"Could not find exposure_universe.yaml in any expected location. Last error: {last_error}")
            
            decomposer = ReturnDecomposer()
            return RiskPremiumEstimator(universe, decomposer)
            
        except Exception as e:
            logger.error(f"Failed to create default estimator: {e}")
            raise ValueError(
                "Could not create default risk estimator. "
                "Please provide a pre-configured risk_estimator."
            )
    
    def _load_or_optimize_parameters(self) -> UnifiedOptimalParameters:
        """Load parameters from file or run optimization if needed."""
        # Try to load existing parameters
        if self.parameter_file.exists():
            try:
                with open(self.parameter_file, 'r') as f:
                    params = UnifiedOptimalParameters.from_yaml(f.read())
                
                # Check if parameters are stale
                days_old = (datetime.now() - params.optimization_date).days
                if days_old > self.auto_optimize_if_stale_days:
                    logger.warning(
                        f"Optimal parameters are {days_old} days old "
                        f"(threshold: {self.auto_optimize_if_stale_days} days)"
                    )
                    if self.auto_optimize:
                        return self._run_optimization()
                
                logger.info(f"Loaded optimal parameters from {self.parameter_file}")
                return params
                
            except Exception as e:
                logger.error(f"Failed to load parameters: {e}")
                if self.auto_optimize:
                    return self._run_optimization()
                else:
                    raise ValueError(
                        f"Could not load parameters from {self.parameter_file}. "
                        "Run optimization first or set auto_optimize=True"
                    )
        
        # No parameters found
        if self.auto_optimize:
            logger.info("No parameters found, running optimization...")
            return self._run_optimization()
        else:
            raise FileNotFoundError(
                f"No optimal parameters found at {self.parameter_file}. "
                "Run component optimization first or set auto_optimize=True"
            )
    
    def _run_optimization(self) -> UnifiedOptimalParameters:
        """Run component optimization to generate parameters."""
        from .component_optimizers import ComponentOptimizationOrchestrator
        
        # Get all available exposures
        exposure_ids = list(self.risk_estimator.universe.exposures.keys())
        
        # Set optimization period (last 5 years)
        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(years=5)
        
        # Run optimization
        logger.info("Running component optimization...")
        orchestrator = ComponentOptimizationOrchestrator(self.risk_estimator)
        
        optimal_params = orchestrator.optimize_all_components(
            exposure_ids=exposure_ids,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save parameters
        self.parameter_file.parent.mkdir(parents=True, exist_ok=True)
        orchestrator.save_optimal_parameters(optimal_params, str(self.parameter_file))
        logger.info(f"Saved optimal parameters to {self.parameter_file}")
        
        return optimal_params
    
    def get_volatility_estimate(self, 
                               exposure_id: str,
                               estimation_date: datetime,
                               horizon: int = 21) -> Dict[str, float]:
        """Get volatility estimate using optimal parameters.
        
        Args:
            exposure_id: ID of exposure to estimate
            estimation_date: Date for estimation
            horizon: Forecast horizon in days
            
        Returns:
            Dictionary with 'volatility' (annualized) and 'variance'
        """
        if exposure_id not in self.optimal_params.volatility_params:
            raise ValueError(f"No optimal parameters for exposure: {exposure_id}")
        
        params = self.optimal_params.volatility_params[exposure_id]
        
        # Use optimal parameters for estimation
        estimate = self.risk_estimator.estimate_risk_premium_volatility(
            exposure_id=exposure_id,
            estimation_date=estimation_date,
            forecast_horizon=horizon,
            method=params.method,
            parameters=params.parameters,
            lookback_days=params.lookback_days,
            frequency=params.frequency
        )
        
        return {
            'volatility': estimate.risk_premium_volatility,
            'variance': estimate.risk_premium_variance,
            'sample_size': estimate.sample_size
        }
    
    def get_correlation_matrix(self, 
                              exposure_ids: List[str],
                              estimation_date: datetime) -> pd.DataFrame:
        """Get correlation matrix using optimal parameters.
        
        Args:
            exposure_ids: List of exposure IDs
            estimation_date: Date for estimation
            
        Returns:
            DataFrame with correlation matrix
        """
        params = self.optimal_params.correlation_params
        
        # Use optimal parameters for correlation estimation
        corr_matrix = self.risk_estimator.estimate_risk_premium_correlation_matrix(
            exposures=exposure_ids,
            estimation_date=estimation_date,
            method=params.method,
            parameters=params.parameters,
            lookback_days=params.lookback_days,
            frequency=params.frequency
        )
        
        return corr_matrix
    
    def get_expected_returns(self, 
                           exposure_ids: List[str],
                           estimation_date: datetime,
                           horizon: int = 21) -> pd.Series:
        """Get expected returns using optimal parameters.
        
        Args:
            exposure_ids: List of exposure IDs
            estimation_date: Date for estimation
            horizon: Forecast horizon in days
            
        Returns:
            Series with annualized expected returns
        """
        returns = {}
        
        for exp_id in exposure_ids:
            if exp_id not in self.optimal_params.expected_return_params:
                logger.warning(f"No return parameters for {exp_id}, using zero")
                returns[exp_id] = 0.0
                continue
            
            params = self.optimal_params.expected_return_params[exp_id]
            
            # Estimate expected return using optimal method
            expected_return = self._estimate_expected_return(
                exp_id, estimation_date, params, horizon
            )
            returns[exp_id] = expected_return
        
        return pd.Series(returns, name='expected_returns')
    
    def _estimate_expected_return(self,
                                 exposure_id: str,
                                 estimation_date: datetime,
                                 params: ComponentOptimalParameters,
                                 horizon: int) -> float:
        """Estimate expected return for a single exposure."""
        try:
            # Get historical data using optimal parameters
            decomposition = self.risk_estimator.load_and_decompose_exposure_returns(
                exposure_id=exposure_id,
                estimation_date=estimation_date,
                lookback_days=params.lookback_days,
                frequency=params.frequency
            )
            
            if decomposition is None or decomposition.empty or 'spread' not in decomposition.columns:
                logger.warning(f"No data for {exposure_id}, using zero return")
                return 0.0
            
            returns = decomposition['spread']
            
            if params.method == 'historical':
                # Simple historical average
                return returns.mean() * 252  # Annualize
            
            elif params.method == 'momentum':
                # Simple momentum signal
                momentum_period = params.parameters.get('momentum_period', 12)
                if len(returns) < momentum_period:
                    return returns.mean() * 252
                
                recent_return = returns.tail(momentum_period).mean()
                momentum_strength = params.parameters.get('momentum_strength', 0.5)
                base_return = returns.mean()
                
                return (base_return + momentum_strength * recent_return) * 252
            
            elif params.method == 'ewma':
                # EWMA of returns
                decay_factor = params.parameters.get('decay_factor', 0.97)
                ewma_return = returns.ewm(alpha=1-decay_factor).mean().iloc[-1]
                return ewma_return * 252
            
            elif params.method == 'mean_reversion':
                # Mean reversion model
                long_term_mean = returns.mean()
                recent_period = params.parameters.get('recent_period', 6)
                
                if len(returns) < recent_period:
                    return long_term_mean * 252
                
                recent_mean = returns.tail(recent_period).mean()
                reversion_strength = params.parameters.get('reversion_strength', 0.3)
                
                # Expect reversion toward long-term mean
                expected = long_term_mean + reversion_strength * (long_term_mean - recent_mean)
                return expected * 252
            
            else:
                # Default to historical
                return returns.mean() * 252
                
        except Exception as e:
            logger.warning(f"Failed to estimate return for {exposure_id}: {e}")
            return 0.0
    
    def get_covariance_matrix(self, 
                             exposure_ids: List[str],
                             estimation_date: datetime) -> pd.DataFrame:
        """Get full covariance matrix using optimal parameters.
        
        This is the main method for portfolio optimization.
        Combines optimally estimated volatilities and correlations.
        
        Args:
            exposure_ids: List of exposure IDs
            estimation_date: Date for estimation
            
        Returns:
            DataFrame with covariance matrix
        """
        # Get volatilities using optimal parameters
        volatilities = {}
        for exp_id in exposure_ids:
            try:
                vol_est = self.get_volatility_estimate(exp_id, estimation_date)
                volatilities[exp_id] = vol_est['volatility']
            except Exception as e:
                logger.warning(f"Failed to get volatility for {exp_id}: {e}")
                volatilities[exp_id] = 0.15  # Default 15% volatility
        
        # Get correlation matrix using optimal parameters
        correlations = self.get_correlation_matrix(exposure_ids, estimation_date)
        
        # Ensure alignment
        correlations = correlations.loc[exposure_ids, exposure_ids]
        
        # Combine into covariance matrix
        vol_array = np.array([volatilities[exp_id] for exp_id in exposure_ids])
        cov_matrix = correlations.values * np.outer(vol_array, vol_array)
        
        return pd.DataFrame(
            cov_matrix, 
            index=exposure_ids, 
            columns=exposure_ids
        )
    
    def get_optimization_ready_inputs(self, 
                                    exposure_ids: List[str],
                                    estimation_date: datetime,
                                    risk_free_rate: float = 0.02) -> Dict:
        """Get all inputs needed for portfolio optimization.
        
        This is the main convenience method that provides everything
        needed to run portfolio optimization.
        
        Args:
            exposure_ids: List of exposure IDs
            estimation_date: Date for estimation
            risk_free_rate: Risk-free rate for Sharpe calculations
            
        Returns:
            Dictionary with:
                - 'expected_returns': Series of expected returns
                - 'covariance_matrix': Covariance matrix DataFrame
                - 'risk_free_rate': Risk-free rate
                - 'estimation_date': Date of estimation
        """
        return {
            'expected_returns': self.get_expected_returns(exposure_ids, estimation_date),
            'covariance_matrix': self.get_covariance_matrix(exposure_ids, estimation_date),
            'risk_free_rate': risk_free_rate,
            'estimation_date': estimation_date
        }
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """Get summary of optimal parameters for all exposures.
        
        Returns:
            DataFrame with parameter summary
        """
        rows = []
        
        # Volatility parameters
        for exp_id, params in self.optimal_params.volatility_params.items():
            row = {
                'exposure_id': exp_id,
                'component': 'volatility',
                'method': params.method,
                'lookback_days': params.lookback_days,
                'frequency': params.frequency,
                'score': params.score
            }
            row.update(params.parameters)
            rows.append(row)
        
        # Correlation parameters (single set for all)
        params = self.optimal_params.correlation_params
        row = {
            'exposure_id': 'ALL',
            'component': 'correlation',
            'method': params.method,
            'lookback_days': params.lookback_days,
            'frequency': params.frequency,
            'score': params.score
        }
        row.update(params.parameters)
        rows.append(row)
        
        # Expected return parameters
        for exp_id, params in self.optimal_params.expected_return_params.items():
            row = {
                'exposure_id': exp_id,
                'component': 'expected_returns',
                'method': params.method,
                'lookback_days': params.lookback_days,
                'frequency': params.frequency,
                'score': params.score
            }
            row.update(params.parameters)
            rows.append(row)
        
        return pd.DataFrame(rows)


# Convenience functions for simple access
def get_best_risk_estimates(exposure_ids: List[str], 
                          estimation_date: datetime) -> pd.DataFrame:
    """Simple entry point - just get the best covariance matrix.
    
    Args:
        exposure_ids: List of exposure IDs
        estimation_date: Date for estimation
        
    Returns:
        Covariance matrix DataFrame
    """
    estimator = OptimizedRiskEstimator()
    return estimator.get_covariance_matrix(exposure_ids, estimation_date)


def get_optimization_inputs(exposure_ids: List[str],
                          estimation_date: datetime,
                          risk_free_rate: float = 0.02) -> Dict:
    """Get everything needed for portfolio optimization.
    
    Args:
        exposure_ids: List of exposure IDs
        estimation_date: Date for estimation
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary with expected_returns, covariance_matrix, and risk_free_rate
    """
    estimator = OptimizedRiskEstimator()
    return estimator.get_optimization_ready_inputs(
        exposure_ids, estimation_date, risk_free_rate
    )