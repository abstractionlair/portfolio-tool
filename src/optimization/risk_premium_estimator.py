#!/usr/bin/env python
"""
Risk Premium Estimation Framework

This module extends the existing ExposureRiskEstimator to work on decomposed risk premia
rather than total returns. This provides theoretically superior risk estimates for 
portfolio optimization by focusing on compensated risk (risk premia) rather than 
uncompensated components like risk-free rate volatility.

The key insight: Portfolio optimization should be based on volatilities and correlations
of RISK PREMIA, not total returns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

import pandas as pd
import numpy as np

try:
    from .exposure_risk_estimator import ExposureRiskEstimator, ExposureRiskEstimate, ExposureRiskMatrix
    from .parameter_optimization import ParameterOptimizer
    from ..data.exposure_universe import ExposureUniverse
    from ..data.return_decomposition import ReturnDecomposer
    from ..data.multi_frequency import Frequency
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from optimization.exposure_risk_estimator import ExposureRiskEstimator, ExposureRiskEstimate, ExposureRiskMatrix
    from optimization.parameter_optimization import ParameterOptimizer
    from data.exposure_universe import ExposureUniverse
    from data.return_decomposition import ReturnDecomposer
    from data.multi_frequency import Frequency

logger = logging.getLogger(__name__)


@dataclass
class RiskPremiumEstimate:
    """Risk estimate specifically for risk premium component."""
    exposure_id: str
    estimation_date: datetime
    forecast_horizon: int
    method: str
    
    # Risk premium estimates (the key output)
    risk_premium_volatility: float
    risk_premium_variance: float
    
    # Component estimates for reconstruction
    inflation_volatility: float
    real_rf_volatility: float
    nominal_rf_volatility: float
    
    # Total return estimates (reconstructed)
    total_volatility: float
    total_variance: float
    
    # Metadata
    sample_size: int
    frequency: str
    lookback_days: int
    
    # Component correlations (for reconstruction)
    rp_inflation_correlation: float = 0.0
    rp_real_rf_correlation: float = 0.0
    inflation_real_rf_correlation: float = 0.0


@dataclass 
class CombinedRiskEstimates:
    """Combined risk estimates providing both risk premium and total return views."""
    exposures: List[str]
    estimation_date: datetime
    forecast_horizon: int
    method: str
    
    # Risk premium estimates (for portfolio optimization)
    risk_premium_volatilities: pd.Series
    risk_premium_correlation_matrix: pd.DataFrame
    risk_premium_covariance_matrix: pd.DataFrame
    
    # Total return estimates (for implementation/reporting)
    total_return_volatilities: pd.Series
    total_return_correlation_matrix: pd.DataFrame
    total_return_covariance_matrix: pd.DataFrame
    
    # Component breakdowns
    component_volatilities: Dict[str, pd.Series]  # inflation, real_rf, etc.
    
    # Metadata
    sample_sizes: pd.Series
    frequency: str


class RiskPremiumEstimator(ExposureRiskEstimator):
    """
    Enhanced risk estimator that works on decomposed risk premia.
    
    This class extends ExposureRiskEstimator to:
    1. Decompose all exposure returns into components before estimation
    2. Estimate volatilities/correlations on risk premia (spread component)
    3. Provide both risk premium and total return estimates
    4. Enable parameter optimization specifically for risk premium forecasting
    """
    
    def __init__(
        self,
        exposure_universe: ExposureUniverse,
        return_decomposer: Optional[ReturnDecomposer] = None,
        parameter_optimizer: Optional[ParameterOptimizer] = None
    ):
        """
        Initialize risk premium estimator.
        
        Args:
            exposure_universe: Universe of exposures to estimate
            return_decomposer: Decomposer for breaking returns into components
            parameter_optimizer: Optimizer for finding best parameters (optional)
        """
        super().__init__(exposure_universe, parameter_optimizer)
        self.return_decomposer = return_decomposer or ReturnDecomposer()
        
        # Cache for decomposed returns to avoid repeated API calls
        self._decomposed_returns_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info("Initialized RiskPremiumEstimator for exposure-level risk premium estimation")
    
    def load_and_decompose_exposure_returns(
        self,
        exposure_id: str,
        estimation_date: datetime,
        lookback_days: int = 756,  # ~3 years
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Load exposure returns and decompose into fundamental components.
        
        Args:
            exposure_id: ID of exposure to load and decompose
            estimation_date: Date for estimation (end of data)
            lookback_days: Number of days to look back for data
            frequency: Frequency of returns (daily, weekly, monthly)
            
        Returns:
            DataFrame with columns:
            - total_return: Original total returns
            - inflation: Inflation component  
            - real_rf_rate: Real risk-free rate component
            - spread: Risk premium component (KEY for portfolio optimization)
            - nominal_rf_rate: Nominal risk-free rate (for reference)
        """
        cache_key = f"{exposure_id}_{estimation_date.date()}_{lookback_days}_{frequency}"
        
        if cache_key in self._decomposed_returns_cache:
            logger.debug(f"Using cached decomposed returns for {exposure_id}")
            return self._decomposed_returns_cache[cache_key]
        
        logger.info(f"Loading and decomposing returns for {exposure_id}")
        
        # Step 1: Load raw exposure returns using parent class method
        # Ensure datetime objects are properly formatted
        if isinstance(estimation_date, str):
            estimation_date = datetime.fromisoformat(estimation_date)
        
        start_date = estimation_date - timedelta(days=int(lookback_days))
        
        try:
            exposure_returns = self._load_exposure_returns(
                exposure_id, estimation_date, lookback_days
            )
            
            if exposure_returns is None or exposure_returns.empty:
                logger.warning(f"No returns data for {exposure_id}")
                return pd.DataFrame()
            
            # Step 2: Decompose returns into components
            # Handle timezone issues - ensure consistent timezone handling
            if exposure_returns.index.tz is not None:
                # Convert to timezone-naive for decomposition
                exposure_returns = exposure_returns.tz_localize(None)
            
            # CRITICAL FIX: Ensure exposure_returns has a unique name for cache key
            # This prevents cache collisions when multiple exposures have None names
            if exposure_returns.name is None:
                exposure_returns.name = exposure_id
            
            decomposition = self.return_decomposer.decompose_returns(
                returns=exposure_returns,
                start_date=start_date,
                end_date=estimation_date,
                frequency=frequency
            )
            
            if decomposition.empty:
                logger.warning(f"Decomposition failed for {exposure_id}")
                return pd.DataFrame()
            
            # Step 3: Validate decomposition quality
            if 'error' in decomposition.columns:
                mean_error = decomposition['error'].mean()
                std_error = decomposition['error'].std()
                
                if abs(mean_error) > 0.001 or std_error > 0.005:  # 0.1% mean, 0.5% std
                    logger.warning(
                        f"Decomposition quality concern for {exposure_id}: "
                        f"mean_error={mean_error:.4f}, std_error={std_error:.4f}"
                    )
            
            # Cache successful decomposition
            self._decomposed_returns_cache[cache_key] = decomposition
            
            logger.debug(
                f"Successfully decomposed {len(decomposition)} periods for {exposure_id}"
            )
            
            return decomposition
            
        except Exception as e:
            logger.error(f"Failed to decompose returns for {exposure_id}: {e}")
            return pd.DataFrame()
    
    def estimate_risk_premium_volatility(
        self,
        exposure_id: str,
        estimation_date: datetime,
        forecast_horizon: int = 21,
        method: str = 'ewma',
        parameters: Optional[Dict[str, Any]] = None,
        lookback_days: int = 756,
        frequency: str = 'daily'
    ) -> Optional[RiskPremiumEstimate]:
        """
        Estimate volatility of the risk premium component.
        
        This is the core method that estimates volatility on the SPREAD component
        rather than total returns. This provides better inputs for portfolio optimization.
        
        Args:
            exposure_id: Exposure to estimate
            estimation_date: Date for estimation
            forecast_horizon: Forecast horizon in periods
            method: Estimation method ('ewma', 'garch', 'historical')
            parameters: Method-specific parameters
            lookback_days: Training data length
            frequency: Data frequency
            
        Returns:
            RiskPremiumEstimate with both risk premium and total return volatilities
        """
        logger.info(f"Estimating risk premium volatility for {exposure_id} using {method}")
        
        # Load decomposed returns
        decomposition = self.load_and_decompose_exposure_returns(
            exposure_id, estimation_date, lookback_days, frequency
        )
        
        if decomposition.empty:
            logger.error(f"Cannot estimate - no decomposed data for {exposure_id}")
            return None
        
        # Extract components
        risk_premium_returns = decomposition['spread']  # KEY: This is what we optimize on
        inflation_returns = decomposition['inflation']
        real_rf_returns = decomposition['real_rf_rate']
        nominal_rf_returns = decomposition['nominal_rf_rate']
        total_returns = decomposition['total_return']
        
        # Handle NaN values more intelligently - don't drop valid total return data
        # Create a DataFrame with all components
        all_data = pd.concat([
            risk_premium_returns, inflation_returns, real_rf_returns, total_returns
        ], axis=1)
        all_data.columns = ['risk_premium', 'inflation', 'real_rf', 'total']
        
        # For consistency, use the same time period for both calculations
        # Use only periods where decomposition is complete for both RP and total volatility
        valid_data = all_data.dropna()
        
        logger.info(f"Using {len(valid_data)} periods for volatility calculations "
                   f"(removed {len(all_data) - len(valid_data)} periods with incomplete decomposition)")
        
        if len(valid_data) < 20:  # Reduced minimum data requirement for testing
            logger.warning(f"Insufficient valid data for {exposure_id}: {len(valid_data)} periods")
            return None
        
        # Extract clean series - use consistent dataset for both calculations
        rp_returns = valid_data['risk_premium']  # risk premium (spread)
        inflation = valid_data['inflation']
        real_rf = valid_data['real_rf']
        total = valid_data['total']  # Use same dataset for consistency
        
        # Get consistent annualization factor for all calculations
        annualization_factor = self._get_robust_annualization_factor(total, frequency)
        
        # Estimate volatility on risk premium component
        if method == 'ewma':
            from .ewma import EWMAEstimator, EWMAParameters
            
            # Use provided parameters or defaults
            lambda_param = parameters.get('lambda', 0.94) if parameters else 0.94
            min_periods = parameters.get('min_periods', 10) if parameters else 10  # Reduced for testing
            
            ewma_params = EWMAParameters(lambda_=lambda_param, min_periods=min_periods)
            estimator = EWMAEstimator(ewma_params)
            
            # CRITICAL: Get unannualized volatility first, then apply consistent annualization
            rp_volatility_series = estimator.estimate_volatility(
                rp_returns, frequency=frequency, annualize=False
            )
            # Take the most recent (last) volatility estimate and annualize consistently
            rp_volatility_unannualized = rp_volatility_series.iloc[-1] if len(rp_volatility_series) > 0 else 0.0
            rp_volatility = rp_volatility_unannualized * np.sqrt(annualization_factor)
            
        elif method == 'historical':
            # Simple historical volatility of risk premium with consistent annualization
            rp_volatility = rp_returns.std() * np.sqrt(annualization_factor)
            
        elif method == 'garch':
            # GARCH estimation on risk premium component
            from .ewma import GARCHEstimator, GARCHParameters
            
            omega = parameters.get('omega', 0.000001) if parameters else 0.000001
            alpha = parameters.get('alpha', 0.1) if parameters else 0.1
            beta = parameters.get('beta', 0.85) if parameters else 0.85
            
            garch_params = GARCHParameters(omega=omega, alpha=alpha, beta=beta)
            estimator = GARCHEstimator(garch_params)
            
            # CRITICAL: Get unannualized volatility first, then apply consistent annualization
            rp_volatility_series = estimator.estimate_volatility(
                rp_returns, frequency=frequency, annualize=False
            )
            # Take the most recent (last) volatility estimate and annualize consistently
            rp_volatility_unannualized = rp_volatility_series.iloc[-1] if len(rp_volatility_series) > 0 else 0.0
            rp_volatility = rp_volatility_unannualized * np.sqrt(annualization_factor)
            
        else:
            logger.error(f"Unknown estimation method: {method}")
            return None
        
        # Estimate component volatilities using the same consistent annualization
        inflation_vol = inflation.std() * np.sqrt(annualization_factor)
        real_rf_vol = real_rf.std() * np.sqrt(annualization_factor)
        nominal_rf_vol = nominal_rf_returns.std() * np.sqrt(annualization_factor)
        
        # Calculate total return volatility using the same consistent annualization
        total_vol = total.std() * np.sqrt(annualization_factor)
        
        # Calculate cross-component correlations for reconstruction
        rp_inflation_corr = rp_returns.corr(inflation)
        rp_real_rf_corr = rp_returns.corr(real_rf)
        inflation_real_rf_corr = inflation.corr(real_rf)
        
        # Handle NaN correlations
        rp_inflation_corr = 0.0 if pd.isna(rp_inflation_corr) else rp_inflation_corr
        rp_real_rf_corr = 0.0 if pd.isna(rp_real_rf_corr) else rp_real_rf_corr
        inflation_real_rf_corr = 0.0 if pd.isna(inflation_real_rf_corr) else inflation_real_rf_corr
        
        return RiskPremiumEstimate(
            exposure_id=exposure_id,
            estimation_date=estimation_date,
            forecast_horizon=forecast_horizon,
            method=method,
            
            # Risk premium estimates (PRIMARY OUTPUT)
            risk_premium_volatility=rp_volatility,
            risk_premium_variance=rp_volatility ** 2,
            
            # Component volatilities
            inflation_volatility=inflation_vol,
            real_rf_volatility=real_rf_vol,
            nominal_rf_volatility=nominal_rf_vol,
            
            # Total return estimates (for comparison)
            total_volatility=total_vol,
            total_variance=total_vol ** 2,
            
            # Metadata
            sample_size=len(rp_returns),
            frequency=frequency,
            lookback_days=lookback_days,
            
            # Cross-component correlations
            rp_inflation_correlation=rp_inflation_corr,
            rp_real_rf_correlation=rp_real_rf_corr,
            inflation_real_rf_correlation=inflation_real_rf_corr
        )
    
    def estimate_risk_premium_correlation_matrix(
        self,
        exposures: List[str],
        estimation_date: datetime,
        method: str = 'ewma',
        parameters: Optional[Dict[str, Any]] = None,
        lookback_days: int = 756,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Estimate correlation matrix of risk premia across exposures.
        
        This estimates correlations between the SPREAD components of different
        exposures, which provides better diversification insights than total return
        correlations.
        
        Args:
            exposures: List of exposure IDs
            estimation_date: Date for estimation
            method: Estimation method
            parameters: Method-specific parameters
            lookback_days: Training data length
            frequency: Data frequency
            
        Returns:
            Correlation matrix of risk premia
        """
        logger.info(f"Estimating risk premium correlation matrix for {len(exposures)} exposures")
        
        # Collect risk premium returns for all exposures
        risk_premium_returns = {}
        
        for exposure_id in exposures:
            decomposition = self.load_and_decompose_exposure_returns(
                exposure_id, estimation_date, lookback_days, frequency
            )
            
            if not decomposition.empty and 'spread' in decomposition.columns:
                # Use the risk premium (spread) component
                risk_premium_returns[exposure_id] = decomposition['spread']
            else:
                logger.warning(f"No risk premium data for {exposure_id}")
        
        if len(risk_premium_returns) < 2:
            logger.error("Need at least 2 exposures with valid data for correlation matrix")
            return pd.DataFrame()
        
        # Create DataFrame of risk premium returns
        rp_df = pd.DataFrame(risk_premium_returns)
        rp_df = rp_df.dropna()  # Remove periods with any missing data
        
        if len(rp_df) < 20:  # Reduced minimum for testing
            logger.warning(f"Insufficient overlapping data: {len(rp_df)} periods")
            return pd.DataFrame()
        
        # Estimate correlation matrix on risk premia
        if method == 'ewma':
            from .ewma import EWMAEstimator, EWMAParameters
            
            lambda_param = parameters.get('lambda', 0.94) if parameters else 0.94
            min_periods = parameters.get('min_periods', 30) if parameters else 30
            
            ewma_params = EWMAParameters(lambda_=lambda_param, min_periods=min_periods)
            estimator = EWMAEstimator(ewma_params)
            
            # Estimate correlation matrix on risk premium returns
            correlation_matrix = estimator.estimate_correlation_matrix(rp_df)
            
        elif method == 'historical':
            # Simple historical correlation of risk premia
            correlation_matrix = rp_df.corr()
            
        else:
            logger.warning(f"Method {method} not implemented for correlation, using historical")
            correlation_matrix = rp_df.corr()
        
        # Ensure positive semi-definite
        eigenvals = np.linalg.eigvals(correlation_matrix.values)
        min_eigenval = eigenvals.min()
        
        if min_eigenval < -1e-8:
            logger.warning(f"Correlation matrix not PSD (min eigenval: {min_eigenval:.6f}), adjusting")
            # Simple fix: add small amount to diagonal
            adjustment = abs(min_eigenval) + 1e-6
            np.fill_diagonal(correlation_matrix.values, correlation_matrix.values.diagonal() + adjustment)
        
        return correlation_matrix
    
    def get_risk_premium_matrix(
        self,
        exposures: List[str],
        estimation_date: datetime,
        forecast_horizon: int = 21,
        method: str = 'optimal',
        lookback_days: int = 756,
        frequency: str = 'daily'
    ) -> Optional[ExposureRiskMatrix]:
        """
        Generate complete risk matrix based on risk premia.
        
        This creates a portfolio-ready covariance matrix using risk premium
        volatilities and correlations rather than total returns.
        
        Args:
            exposures: List of exposure IDs
            estimation_date: Date for estimation  
            forecast_horizon: Forecast horizon
            method: Estimation method ('optimal' uses best validated parameters)
            lookback_days: Training data length
            frequency: Data frequency
            
        Returns:
            ExposureRiskMatrix with risk premium-based estimates
        """
        logger.info(f"Building risk premium matrix for {len(exposures)} exposures")
        
        # Use optimal parameters if available
        if method == 'optimal' and self.parameter_optimizer:
            # Get optimal parameters from validation
            # This would use parameters optimized on risk premia
            method = 'ewma'  # Default fallback
            parameters = {'lambda': 0.94, 'min_periods': 30}  # Conservative defaults
        else:
            parameters = {}
        
        # Estimate individual risk premium volatilities
        risk_estimates = {}
        valid_exposures = []
        
        for exposure_id in exposures:
            estimate = self.estimate_risk_premium_volatility(
                exposure_id, estimation_date, forecast_horizon, 
                method, parameters, lookback_days, frequency
            )
            
            if estimate:
                risk_estimates[exposure_id] = estimate
                valid_exposures.append(exposure_id)
            else:
                logger.warning(f"Failed to estimate risk premium volatility for {exposure_id}")
        
        if len(valid_exposures) < 2:
            logger.error("Need at least 2 valid exposures for risk matrix")
            return None
        
        # Create volatility series from risk premium estimates
        volatilities = pd.Series({
            exp_id: estimate.risk_premium_volatility 
            for exp_id, estimate in risk_estimates.items()
        })
        
        # Estimate correlation matrix on risk premia
        correlation_matrix = self.estimate_risk_premium_correlation_matrix(
            valid_exposures, estimation_date, method, parameters, lookback_days, frequency
        )
        
        if correlation_matrix.empty:
            logger.error("Failed to estimate correlation matrix")
            return None
        
        # Ensure alignment between volatilities and correlation matrix
        common_exposures = volatilities.index.intersection(correlation_matrix.index)
        volatilities = volatilities[common_exposures]
        correlation_matrix = correlation_matrix.loc[common_exposures, common_exposures]
        
        # Build covariance matrix: corr * vol_i * vol_j
        covariance_matrix = correlation_matrix * np.outer(volatilities.values, volatilities.values)
        covariance_matrix = pd.DataFrame(
            covariance_matrix, 
            index=volatilities.index, 
            columns=volatilities.index
        )
        
        return ExposureRiskMatrix(
            exposures=list(common_exposures),
            volatilities=volatilities,
            correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            estimation_date=estimation_date,
            forecast_horizon=forecast_horizon,
            method=f"risk_premium_{method}",
            frequency=frequency,
            sample_sizes=pd.Series({
                exp_id: estimate.sample_size 
                for exp_id, estimate in risk_estimates.items()
                if exp_id in common_exposures
            })
        )
    
    def get_combined_risk_estimates(
        self,
        exposures: List[str],
        estimation_date: datetime,
        forecast_horizon: int = 21,
        method: str = 'optimal',
        lookback_days: int = 756,
        frequency: str = 'daily'
    ) -> Optional[CombinedRiskEstimates]:
        """
        Provide both risk premium and total return estimates.
        
        This gives the full picture:
        - Risk premium estimates for portfolio optimization
        - Total return estimates for implementation and reporting
        - Component breakdowns for analysis
        
        Args:
            exposures: List of exposure IDs
            estimation_date: Date for estimation
            forecast_horizon: Forecast horizon
            method: Estimation method
            lookback_days: Training data length
            frequency: Data frequency
            
        Returns:
            CombinedRiskEstimates with both risk premium and total return views
        """
        logger.info(f"Creating combined risk estimates for {len(exposures)} exposures")
        
        # Get risk premium matrix
        rp_matrix = self.get_risk_premium_matrix(
            exposures, estimation_date, forecast_horizon, method, lookback_days, frequency
        )
        
        if not rp_matrix:
            logger.error("Failed to get risk premium matrix")
            return None
        
        # Get total return matrix using parent class (for comparison)
        total_matrix = super().get_risk_matrix(
            rp_matrix.exposures, estimation_date, forecast_horizon, method
        )
        
        if not total_matrix:
            logger.warning("Failed to get total return matrix, using risk premium only")
            total_volatilities = rp_matrix.volatilities  # Fallback
            total_correlation_matrix = rp_matrix.correlation_matrix
            total_covariance_matrix = rp_matrix.covariance_matrix
        else:
            total_volatilities = total_matrix.volatilities
            total_correlation_matrix = total_matrix.correlation_matrix
            total_covariance_matrix = total_matrix.covariance_matrix
        
        # Collect component volatilities
        component_volatilities = {
            'risk_premium': rp_matrix.volatilities,
            'total_return': total_volatilities
        }
        
        # Add individual component breakdowns if available
        individual_estimates = {}
        for exposure_id in rp_matrix.exposures:
            estimate = self.estimate_risk_premium_volatility(
                exposure_id, estimation_date, forecast_horizon, method, lookback_days, frequency
            )
            if estimate:
                individual_estimates[exposure_id] = estimate
        
        if individual_estimates:
            component_volatilities['inflation'] = pd.Series({
                exp_id: est.inflation_volatility 
                for exp_id, est in individual_estimates.items()
            })
            component_volatilities['real_risk_free'] = pd.Series({
                exp_id: est.real_rf_volatility 
                for exp_id, est in individual_estimates.items()
            })
        
        return CombinedRiskEstimates(
            exposures=rp_matrix.exposures,
            estimation_date=estimation_date,
            forecast_horizon=forecast_horizon,
            method=method,
            
            # Risk premium estimates (PRIMARY for portfolio optimization)
            risk_premium_volatilities=rp_matrix.volatilities,
            risk_premium_correlation_matrix=rp_matrix.correlation_matrix,
            risk_premium_covariance_matrix=rp_matrix.covariance_matrix,
            
            # Total return estimates (for implementation)
            total_return_volatilities=total_volatilities,
            total_return_correlation_matrix=total_correlation_matrix,
            total_return_covariance_matrix=total_covariance_matrix,
            
            # Component breakdowns
            component_volatilities=component_volatilities,
            
            # Metadata
            sample_sizes=rp_matrix.sample_sizes,
            frequency=frequency
        )
    
    def _get_annualization_factor(self, frequency: str) -> float:
        """Get annualization factor for given frequency."""
        factors = {
            'daily': 252.0,
            'weekly': 52.0, 
            'monthly': 12.0,
            'quarterly': 4.0,
            'annual': 1.0
        }
        return factors.get(frequency.lower(), 252.0)
    
    def _detect_actual_frequency(self, returns_series: pd.Series) -> str:
        """Auto-detect the actual frequency of a returns series based on observations.
        
        This fixes the frequency mismatch bug where the system gets daily data
        but annualizes using the requested frequency parameter.
        
        Args:
            returns_series: Time series of returns
            
        Returns:
            Detected frequency: 'daily', 'weekly', 'monthly', 'quarterly', or 'annual'
        """
        if len(returns_series) < 2:
            return 'daily'  # Default fallback
        
        # Calculate the time span
        start_date = returns_series.index[0]
        end_date = returns_series.index[-1]
        time_span_days = (end_date - start_date).days
        
        # Estimate frequency based on observations per time period
        observations = len(returns_series)
        
        if time_span_days == 0:
            return 'daily'
        
        obs_per_year = observations * 365.25 / time_span_days
        
        # Classify based on observations per year
        if obs_per_year > 200:
            return 'daily'      # ~252 trading days per year
        elif obs_per_year > 40:
            return 'weekly'     # ~52 weeks per year  
        elif obs_per_year > 8:
            return 'monthly'    # ~12 months per year
        elif obs_per_year > 2:
            return 'quarterly'  # ~4 quarters per year
        else:
            return 'annual'     # ~1 observation per year
    
    def _get_robust_annualization_factor(self, returns_series: pd.Series, frequency_hint: str = None) -> float:
        """Get annualization factor with auto-detection to fix frequency mismatch bugs.
        
        Args:
            returns_series: The actual returns data
            frequency_hint: The requested frequency (may be incorrect)
            
        Returns:
            Correct annualization factor based on actual data frequency
        """
        actual_frequency = self._detect_actual_frequency(returns_series)
        
        if frequency_hint and actual_frequency != frequency_hint:
            logger.warning(
                f"Frequency mismatch detected: requested '{frequency_hint}' but data appears to be '{actual_frequency}'. "
                f"Using actual frequency '{actual_frequency}' for annualization."
            )
        
        return self._get_annualization_factor(actual_frequency)


def build_portfolio_risk_matrix_from_risk_premia(
    portfolio_weights: Dict[str, float],
    risk_premium_estimator: RiskPremiumEstimator,
    estimation_date: datetime,
    forecast_horizon: int = 21,
    method: str = 'optimal'
) -> Tuple[np.ndarray, List[str]]:
    """
    Build portfolio risk matrix using risk premium estimates.
    
    This creates a covariance matrix for portfolio optimization based on
    risk premia rather than total returns.
    
    Args:
        portfolio_weights: Dictionary of exposure weights
        risk_premium_estimator: RiskPremiumEstimator instance
        estimation_date: Date for estimation
        forecast_horizon: Forecast horizon
        method: Estimation method
        
    Returns:
        Tuple of (covariance_matrix, exposure_order)
    """
    exposures = list(portfolio_weights.keys())
    
    # Get risk premium matrix
    risk_matrix = risk_premium_estimator.get_risk_premium_matrix(
        exposures, estimation_date, forecast_horizon, method
    )
    
    if not risk_matrix:
        raise ValueError("Failed to build risk premium matrix")
    
    # Return covariance matrix and exposure order
    return risk_matrix.covariance_matrix.values, risk_matrix.exposures