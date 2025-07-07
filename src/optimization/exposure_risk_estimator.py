"""
Exposure-Level Risk Estimation Framework.

This module provides forward-looking volatility and correlation estimates at the
exposure level using optimized parameters from the parameter optimization framework.
This bridges parameter validation to portfolio optimization.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

try:
    from ..data.exposure_universe import ExposureUniverse
    from ..data.multi_frequency import Frequency, MultiFrequencyDataFetcher, ReturnCompounding
    from .parameter_optimization import ParameterOptimizer
    from .ewma import EWMAEstimator, EWMAParameters, GARCHEstimator
except ImportError:
    # Fallback for direct execution
    from data.exposure_universe import ExposureUniverse
    from data.multi_frequency import Frequency, MultiFrequencyDataFetcher, ReturnCompounding
    from optimization.parameter_optimization import ParameterOptimizer
    from optimization.ewma import EWMAEstimator, EWMAParameters, GARCHEstimator

logger = logging.getLogger(__name__)


@dataclass
class ExposureRiskEstimate:
    """Container for exposure-level risk estimates."""
    exposure_id: str
    volatility: float  # Annualized
    forecast_horizon: int
    estimation_date: datetime
    method: str
    frequency: str
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExposureRiskMatrix:
    """Container for complete exposure risk matrix."""
    volatilities: pd.Series
    correlation_matrix: pd.DataFrame
    covariance_matrix: pd.DataFrame
    estimation_date: datetime
    forecast_horizon: int
    method: str
    exposures: List[str]


class ExposureRiskEstimator:
    """Estimate forward-looking volatilities and correlations for exposures."""
    
    def __init__(
        self,
        exposure_universe: ExposureUniverse,
        parameter_optimizer: Optional[ParameterOptimizer] = None,
        data_fetcher: Optional[MultiFrequencyDataFetcher] = None
    ):
        """Initialize exposure risk estimator.
        
        Args:
            exposure_universe: Universe of exposures to analyze
            parameter_optimizer: Optimizer with validated parameters (optional)
            data_fetcher: Data fetcher for historical returns (optional)
        """
        self.exposure_universe = exposure_universe
        self.parameter_optimizer = parameter_optimizer
        self.data_fetcher = data_fetcher or MultiFrequencyDataFetcher()
        
        # Cache for exposure returns and risk estimates
        self._exposure_returns_cache = {}
        self._risk_estimates_cache = {}
        
        # Default parameters if no optimizer provided
        self._default_params = {
            'ewma_lambda': 0.94,
            'min_periods': 30,
            'frequency': Frequency.DAILY
        }
        
        logger.info(f"Initialized ExposureRiskEstimator with {len(exposure_universe)} exposures")
    
    def estimate_exposure_risks(
        self,
        exposures: List[str],
        estimation_date: datetime,
        lookback_days: int = 756,  # 3 years default
        forecast_horizon: int = 21,  # 1 month default
        method: str = 'optimal'  # 'optimal', 'ewma', 'garch', 'historical'
    ) -> Dict[str, ExposureRiskEstimate]:
        """
        Estimate volatilities for multiple exposures.
        
        Args:
            exposures: List of exposure IDs
            estimation_date: Date for estimation
            lookback_days: Historical data to use
            forecast_horizon: Days ahead to forecast
            method: Estimation method
            
        Returns:
            Dictionary of exposure_id -> risk estimate
        """
        logger.info(f"Estimating risks for {len(exposures)} exposures using {method} method")
        
        risk_estimates = {}
        
        for exposure_id in exposures:
            try:
                estimate = self._estimate_single_exposure_risk(
                    exposure_id, estimation_date, lookback_days, forecast_horizon, method
                )
                
                if estimate:
                    risk_estimates[exposure_id] = estimate
                    logger.debug(f"Estimated {exposure_id}: {estimate.volatility:.1%} volatility")
                else:
                    logger.warning(f"Failed to estimate risk for exposure {exposure_id}")
                    
            except Exception as e:
                logger.error(f"Error estimating risk for {exposure_id}: {e}")
                continue
        
        logger.info(f"Successfully estimated risks for {len(risk_estimates)}/{len(exposures)} exposures")
        return risk_estimates
    
    def estimate_exposure_correlation_matrix(
        self,
        exposures: List[str],
        estimation_date: datetime,
        lookback_days: int = 756,
        method: str = 'optimal'
    ) -> pd.DataFrame:
        """
        Estimate correlation matrix between exposures.
        
        Args:
            exposures: List of exposure IDs
            estimation_date: Date for estimation
            lookback_days: Historical data to use
            method: Estimation method
            
        Returns:
            DataFrame with exposure correlations
        """
        logger.info(f"Estimating correlation matrix for {len(exposures)} exposures")
        
        # Load returns for all exposures
        exposure_returns = {}
        for exposure_id in exposures:
            returns = self._load_exposure_returns(
                exposure_id, estimation_date, lookback_days
            )
            if returns is not None and len(returns) > 50:  # Minimum data requirement
                exposure_returns[exposure_id] = returns
        
        if len(exposure_returns) < 2:
            logger.warning("Insufficient exposures with data for correlation estimation")
            return pd.DataFrame()
        
        # Align all return series
        returns_df = pd.DataFrame(exposure_returns)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 50:
            logger.warning("Insufficient aligned data for correlation estimation")
            return pd.DataFrame()
        
        # Estimate correlation matrix based on method
        if method == 'optimal' and self.parameter_optimizer:
            corr_matrix = self._estimate_correlation_with_optimal_params(
                returns_df, estimation_date
            )
        elif method in ['ewma', 'optimal']:
            corr_matrix = self._estimate_correlation_ewma(returns_df)
        elif method == 'historical':
            corr_matrix = returns_df.corr()
        else:
            logger.warning(f"Unknown correlation method {method}, using historical")
            corr_matrix = returns_df.corr()
        
        # Ensure positive semi-definite
        corr_matrix = self._ensure_psd_correlation_matrix(corr_matrix)
        
        logger.info(f"Estimated correlation matrix: {corr_matrix.shape}")
        return corr_matrix
    
    def get_risk_matrix(
        self,
        exposures: List[str],
        estimation_date: datetime,
        forecast_horizon: int = 21,
        method: str = 'optimal'
    ) -> ExposureRiskMatrix:
        """
        Get both volatilities and correlation matrix.
        
        Args:
            exposures: List of exposure IDs
            estimation_date: Date for estimation
            forecast_horizon: Days ahead to forecast
            method: Estimation method
            
        Returns:
            ExposureRiskMatrix with all risk components
        """
        logger.info(f"Building complete risk matrix for {len(exposures)} exposures")
        
        # Get volatility estimates
        risk_estimates = self.estimate_exposure_risks(
            exposures, estimation_date, forecast_horizon=forecast_horizon, method=method
        )
        
        # Extract volatilities
        volatilities = pd.Series({
            exp_id: estimate.volatility 
            for exp_id, estimate in risk_estimates.items()
        })
        
        # Get correlation matrix
        correlation_matrix = self.estimate_exposure_correlation_matrix(
            list(volatilities.index), estimation_date, method=method
        )
        
        # Align volatilities and correlation matrix
        common_exposures = list(set(volatilities.index) & set(correlation_matrix.index))
        volatilities = volatilities.loc[common_exposures]
        correlation_matrix = correlation_matrix.loc[common_exposures, common_exposures]
        
        # Calculate covariance matrix
        vol_array = volatilities.values
        covariance_matrix = correlation_matrix * np.outer(vol_array, vol_array)
        covariance_matrix = pd.DataFrame(
            covariance_matrix, 
            index=correlation_matrix.index, 
            columns=correlation_matrix.columns
        )
        
        risk_matrix = ExposureRiskMatrix(
            volatilities=volatilities,
            correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            estimation_date=estimation_date,
            forecast_horizon=forecast_horizon,
            method=method,
            exposures=common_exposures
        )
        
        logger.info(f"Built risk matrix for {len(common_exposures)} exposures")
        return risk_matrix
    
    def _estimate_single_exposure_risk(
        self,
        exposure_id: str,
        estimation_date: datetime,
        lookback_days: int,
        forecast_horizon: int,
        method: str
    ) -> Optional[ExposureRiskEstimate]:
        """Estimate risk for a single exposure."""
        
        # Load historical returns
        returns = self._load_exposure_returns(exposure_id, estimation_date, lookback_days)
        
        if returns is None or len(returns) < 30:
            logger.warning(f"Insufficient data for {exposure_id}: {len(returns) if returns is not None else 0} periods")
            return None
        
        # Get optimal parameters for this method and horizon
        params = self._get_estimation_parameters(method, forecast_horizon)
        
        try:
            # Estimate volatility based on method
            if method in ['optimal', 'ewma']:
                volatility = self._estimate_volatility_ewma(
                    returns, params, forecast_horizon
                )
            elif method == 'garch':
                volatility = self._estimate_volatility_garch(
                    returns, forecast_horizon
                )
            elif method == 'historical':
                volatility = self._estimate_volatility_historical(
                    returns, forecast_horizon
                )
            else:
                logger.warning(f"Unknown method {method}, using historical")
                volatility = self._estimate_volatility_historical(
                    returns, forecast_horizon
                )
            
            if volatility is None or not np.isfinite(volatility):
                logger.warning(f"Invalid volatility estimate for {exposure_id}: {volatility}")
                return None
            
            return ExposureRiskEstimate(
                exposure_id=exposure_id,
                volatility=volatility,
                forecast_horizon=forecast_horizon,
                estimation_date=estimation_date,
                method=method,
                frequency=params.get('frequency', 'daily'),
                sample_size=len(returns),
                parameters=params
            )
            
        except Exception as e:
            logger.error(f"Error estimating volatility for {exposure_id}: {e}")
            return None
    
    def _load_exposure_returns(
        self,
        exposure_id: str,
        estimation_date: datetime,
        lookback_days: int
    ) -> Optional[pd.Series]:
        """Load historical returns for an exposure."""
        
        # Check cache first
        cache_key = f"{exposure_id}_{estimation_date.date()}_{lookback_days}"
        if cache_key in self._exposure_returns_cache:
            return self._exposure_returns_cache[cache_key]
        
        # Get exposure definition
        exposure = self.exposure_universe.get_exposure(exposure_id)
        if not exposure:
            logger.warning(f"Exposure {exposure_id} not found in universe")
            return None
        
        # Get preferred implementation
        impl = exposure.get_preferred_implementation()
        if not impl:
            logger.warning(f"No implementation found for exposure {exposure_id}")
            return None
        
        # Calculate date range
        start_date = estimation_date - timedelta(days=lookback_days)
        
        try:
            # Fetch returns based on implementation type
            if impl.type == 'fund' and impl.ticker:
                returns = self.data_fetcher._fetch_single_ticker_returns(
                    impl.ticker, start_date, estimation_date, Frequency.DAILY, validate=True
                )
            elif impl.type == 'etf_average' and impl.tickers:
                # Average returns across multiple tickers
                ticker_returns = []
                for ticker in impl.tickers:
                    try:
                        ticker_ret = self.data_fetcher._fetch_single_ticker_returns(
                            ticker, start_date, estimation_date, Frequency.DAILY, validate=True
                        )
                        if len(ticker_ret) > 0:
                            ticker_returns.append(ticker_ret)
                    except Exception as e:
                        logger.debug(f"Failed to fetch {ticker}: {e}")
                        continue
                
                if ticker_returns:
                    # Align and average
                    returns_df = pd.DataFrame(ticker_returns).T
                    returns = returns_df.mean(axis=1, skipna=True).dropna()
                else:
                    logger.warning(f"No valid tickers for exposure {exposure_id}")
                    return None
            else:
                logger.warning(f"Unsupported implementation type {impl.type} for {exposure_id}")
                return None
            
            if len(returns) > 0:
                # Cache the result
                self._exposure_returns_cache[cache_key] = returns
                logger.debug(f"Loaded {len(returns)} returns for {exposure_id}")
                return returns
            else:
                logger.warning(f"No returns loaded for exposure {exposure_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading returns for {exposure_id}: {e}")
            return None
    
    def _get_estimation_parameters(
        self, 
        method: str, 
        forecast_horizon: int
    ) -> Dict[str, Any]:
        """Get estimation parameters for method and horizon."""
        
        if method == 'optimal' and self.parameter_optimizer:
            # Get optimal parameters for this horizon
            optimal_params = self.parameter_optimizer.get_optimal_parameters_for_horizon(
                forecast_horizon
            )
            
            if optimal_params:
                return {
                    'ewma_lambda': optimal_params.get('lambda', self._default_params['ewma_lambda']),
                    'min_periods': optimal_params.get('min_periods', self._default_params['min_periods']),
                    'frequency': optimal_params.get('frequency', 'daily')
                }
        
        # Default parameters based on method
        if method in ['ewma', 'optimal']:
            return {
                'ewma_lambda': 0.94,
                'min_periods': 30,
                'frequency': 'daily'
            }
        else:
            return self._default_params.copy()
    
    def _estimate_volatility_ewma(
        self,
        returns: pd.Series,
        params: Dict[str, Any],
        forecast_horizon: int
    ) -> Optional[float]:
        """Estimate volatility using EWMA."""
        
        try:
            ewma_params = EWMAParameters(
                lambda_=params.get('ewma_lambda', 0.94),
                min_periods=params.get('min_periods', 30)
            )
            
            estimator = EWMAEstimator(ewma_params)
            
            # Forecast volatility
            forecast_vol = estimator.forecast_volatility(
                returns,
                horizon=forecast_horizon,
                method='simple',
                annualize=True,
                frequency='daily'
            )
            
            return float(forecast_vol) if np.isfinite(forecast_vol) else None
            
        except Exception as e:
            logger.error(f"EWMA volatility estimation failed: {e}")
            return None
    
    def _estimate_volatility_garch(
        self,
        returns: pd.Series,
        forecast_horizon: int
    ) -> Optional[float]:
        """Estimate volatility using GARCH."""
        
        try:
            estimator = GARCHEstimator()
            
            # Forecast volatility
            forecast_var = estimator.forecast_variance(
                returns,
                horizon=forecast_horizon,
                annualize=True,
                frequency='daily'
            )
            
            # Return volatility (standard deviation)
            if len(forecast_var) > 0:
                return float(np.sqrt(forecast_var[0]))
            else:
                return None
                
        except Exception as e:
            logger.error(f"GARCH volatility estimation failed: {e}")
            return None
    
    def _estimate_volatility_historical(
        self,
        returns: pd.Series,
        forecast_horizon: int
    ) -> Optional[float]:
        """Estimate volatility using historical standard deviation."""
        
        try:
            # Simple historical volatility, annualized
            vol = returns.std() * np.sqrt(252)
            return float(vol) if np.isfinite(vol) else None
            
        except Exception as e:
            logger.error(f"Historical volatility estimation failed: {e}")
            return None
    
    def _estimate_correlation_with_optimal_params(
        self,
        returns_df: pd.DataFrame,
        estimation_date: datetime
    ) -> pd.DataFrame:
        """Estimate correlation using optimal parameters."""
        
        try:
            # Get optimal parameters for correlation estimation
            optimal_params = self.parameter_optimizer.get_optimal_parameters_for_horizon(21)
            
            if optimal_params:
                ewma_params = EWMAParameters(
                    lambda_=optimal_params.get('lambda', 0.94),
                    min_periods=optimal_params.get('min_periods', 30)
                )
            else:
                ewma_params = EWMAParameters(lambda_=0.94, min_periods=30)
            
            estimator = EWMAEstimator(ewma_params)
            correlation_matrix = estimator.estimate_correlation_matrix(
                returns_df, frequency='daily'
            )
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Optimal correlation estimation failed: {e}")
            # Fallback to simple EWMA
            return self._estimate_correlation_ewma(returns_df)
    
    def _estimate_correlation_ewma(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate correlation using EWMA with default parameters."""
        
        try:
            ewma_params = EWMAParameters(lambda_=0.94, min_periods=30)
            estimator = EWMAEstimator(ewma_params)
            
            correlation_matrix = estimator.estimate_correlation_matrix(
                returns_df, frequency='daily'
            )
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"EWMA correlation estimation failed: {e}")
            # Fallback to historical correlation
            return returns_df.corr()
    
    def _ensure_psd_correlation_matrix(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Ensure correlation matrix is positive semi-definite."""
        
        try:
            # Check if already PSD
            eigenvals = np.linalg.eigvals(corr_matrix.values)
            
            if np.all(eigenvals >= -1e-8):  # Small tolerance for numerical errors
                return corr_matrix
            
            # Fix by eigenvalue clipping
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix.values)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Clip negative eigenvalues
            
            # Reconstruct matrix
            fixed_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Rescale to ensure unit diagonal
            diag_sqrt = np.sqrt(np.diag(fixed_matrix))
            fixed_matrix = fixed_matrix / np.outer(diag_sqrt, diag_sqrt)
            
            # Ensure exactly 1.0 on diagonal
            np.fill_diagonal(fixed_matrix, 1.0)
            
            result = pd.DataFrame(
                fixed_matrix, 
                index=corr_matrix.index, 
                columns=corr_matrix.columns
            )
            
            logger.debug("Fixed correlation matrix to be positive semi-definite")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fix correlation matrix: {e}")
            # Return identity matrix as fallback
            n = len(corr_matrix)
            identity = pd.DataFrame(
                np.eye(n), 
                index=corr_matrix.index, 
                columns=corr_matrix.columns
            )
            return identity
    
    def clear_cache(self):
        """Clear internal caches."""
        self._exposure_returns_cache.clear()
        self._risk_estimates_cache.clear()
        logger.debug("Cleared risk estimator caches")


def build_portfolio_risk_matrix(
    exposure_weights: Dict[str, float],
    risk_estimator: ExposureRiskEstimator,
    estimation_date: datetime,
    forecast_horizon: int = 21,
    method: str = 'optimal'
) -> Tuple[np.ndarray, List[str]]:
    """
    Build covariance matrix for portfolio optimization.
    
    Args:
        exposure_weights: Dictionary of exposure_id -> target_weight
        risk_estimator: Configured risk estimator
        estimation_date: Date for estimation
        forecast_horizon: Days ahead to forecast
        method: Estimation method
        
    Returns:
        Tuple of (covariance_matrix, exposure_list)
    """
    exposures = list(exposure_weights.keys())
    
    # Get risk matrix
    risk_matrix = risk_estimator.get_risk_matrix(
        exposures, estimation_date, forecast_horizon, method
    )
    
    # Return covariance matrix and exposure order
    return risk_matrix.covariance_matrix.values, risk_matrix.exposures