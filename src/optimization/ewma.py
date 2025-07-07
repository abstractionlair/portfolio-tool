"""Exponentially Weighted Moving Average (EWMA) models for risk estimation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

try:
    from ..data.multi_frequency import Frequency
except ImportError:
    try:
        from data.multi_frequency import Frequency
    except ImportError:
        # Fallback enum for standalone usage
        from enum import Enum
        class Frequency(Enum):
            DAILY = "daily"
            WEEKLY = "weekly"
            MONTHLY = "monthly"
            QUARTERLY = "quarterly"
            ANNUAL = "annual"
            
            @property
            def annualization_factor(self) -> float:
                mapping = {
                    self.DAILY: 252.0,
                    self.WEEKLY: 52.0,
                    self.MONTHLY: 12.0,
                    self.QUARTERLY: 4.0,
                    self.ANNUAL: 1.0
                }
                return mapping[self]

logger = logging.getLogger(__name__)


@dataclass
class EWMAParameters:
    """Parameters for EWMA estimation."""
    lambda_: float = 0.94  # Decay parameter (RiskMetrics default)
    min_periods: int = 30  # Minimum observations required
    center: bool = False   # Whether to center returns
    adjust: bool = True    # Whether to adjust for bias in early periods
    
    def __post_init__(self):
        """Validate parameters."""
        if not 0 < self.lambda_ < 1:
            raise ValueError("Lambda must be between 0 and 1")
        if self.min_periods < 1:
            raise ValueError("min_periods must be positive")


class EWMAEstimator:
    """Exponentially Weighted Moving Average estimator for returns and covariance."""
    
    def __init__(self, params: Optional[EWMAParameters] = None):
        """Initialize EWMA estimator.
        
        Args:
            params: EWMA parameters. If None, uses defaults.
        """
        self.params = params or EWMAParameters()
        self._variance_cache = {}
        self._covariance_cache = {}
        
        logger.debug(f"Initialized EWMAEstimator with lambda={self.params.lambda_}")
    
    def estimate_variance(
        self,
        returns: pd.Series,
        annualize: bool = True,
        frequency: Union[str, Frequency] = 'daily'
    ) -> pd.Series:
        """Estimate EWMA variance for a single return series.
        
        Args:
            returns: Return series (should be stationary)
            annualize: Whether to annualize the variance
            frequency: Data frequency for annualization (str or Frequency enum)
            
        Returns:
            Series of EWMA variance estimates
        """
        if len(returns) < self.params.min_periods:
            raise ValueError(f"Need at least {self.params.min_periods} observations")
        
        # Create cache key
        cache_key = (
            str(returns.index[0]), str(returns.index[-1]), 
            self.params.lambda_, annualize, frequency
        )
        
        if cache_key in self._variance_cache:
            return self._variance_cache[cache_key]
        
        # Center returns if requested
        if self.params.center:
            returns = returns - returns.mean()
        
        # Calculate EWMA variance using pandas ewm
        ewma_var = returns.ewm(
            alpha=1-self.params.lambda_,
            adjust=self.params.adjust,
            min_periods=self.params.min_periods
        ).var()
        
        # Annualize if requested
        if annualize:
            annualization_factor = self._get_annualization_factor(frequency)
            ewma_var = ewma_var * annualization_factor
        
        # Cache result
        self._variance_cache[cache_key] = ewma_var
        
        logger.debug(f"Estimated EWMA variance for {len(returns)} periods")
        return ewma_var
    
    def estimate_covariance_matrix(
        self,
        returns_df: pd.DataFrame,
        annualize: bool = True,
        frequency: Union[str, Frequency] = 'daily'
    ) -> pd.DataFrame:
        """Estimate EWMA covariance matrix for multiple return series.
        
        Args:
            returns_df: DataFrame of return series
            annualize: Whether to annualize the covariance
            frequency: Data frequency for annualization
            
        Returns:
            DataFrame containing the EWMA covariance matrix
        """
        if len(returns_df) < self.params.min_periods:
            raise ValueError(f"Need at least {self.params.min_periods} observations")
        
        # Create cache key
        cache_key = (
            str(returns_df.index[0]), str(returns_df.index[-1]),
            tuple(returns_df.columns), self.params.lambda_, annualize, frequency
        )
        
        if cache_key in self._covariance_cache:
            return self._covariance_cache[cache_key]
        
        # Center returns if requested
        if self.params.center:
            returns_df = returns_df - returns_df.mean()
        
        # Calculate EWMA covariance using pandas ewm
        ewma_cov = returns_df.ewm(
            alpha=1-self.params.lambda_,
            adjust=self.params.adjust,
            min_periods=self.params.min_periods
        ).cov()
        
        # Get the latest covariance matrix
        latest_date = ewma_cov.index.get_level_values(0)[-1]
        cov_matrix = ewma_cov.loc[latest_date]
        
        # Annualize if requested
        if annualize:
            annualization_factor = self._get_annualization_factor(frequency)
            cov_matrix = cov_matrix * annualization_factor
        
        # Cache result
        self._covariance_cache[cache_key] = cov_matrix
        
        logger.debug(f"Estimated EWMA covariance matrix for {len(returns_df.columns)} assets")
        return cov_matrix
    
    def estimate_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        frequency: Union[str, Frequency] = 'daily'
    ) -> pd.DataFrame:
        """Estimate EWMA correlation matrix.
        
        Args:
            returns_df: DataFrame of return series
            frequency: Data frequency (for consistency)
            
        Returns:
            DataFrame containing the EWMA correlation matrix
        """
        # Get EWMA covariance matrix (not annualized for correlation)
        cov_matrix = self.estimate_covariance_matrix(
            returns_df, annualize=False, frequency=frequency
        )
        
        # Convert to correlation matrix
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        
        # Ensure correlation matrix properties
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        return pd.DataFrame(
            corr_matrix, 
            index=cov_matrix.index, 
            columns=cov_matrix.columns
        )
    
    def estimate_volatility(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        annualize: bool = True,
        frequency: str = 'daily'
    ) -> Union[pd.Series, pd.DataFrame]:
        """Estimate EWMA volatility (standard deviation).
        
        Args:
            returns: Return series or DataFrame
            annualize: Whether to annualize the volatility
            frequency: Data frequency for annualization
            
        Returns:
            EWMA volatility estimates
        """
        if isinstance(returns, pd.Series):
            variance = self.estimate_variance(returns, annualize, frequency)
            return np.sqrt(variance)
        else:
            # For DataFrame, calculate volatility for each column
            volatilities = {}
            for column in returns.columns:
                var_series = self.estimate_variance(returns[column], annualize, frequency)
                volatilities[column] = np.sqrt(var_series)
            
            return pd.DataFrame(volatilities, index=returns.index)
    
    def rolling_estimate(
        self,
        returns_df: pd.DataFrame,
        window: int,
        min_periods: Optional[int] = None,
        annualize: bool = True,
        frequency: str = 'daily'
    ) -> Dict[str, pd.DataFrame]:
        """Calculate rolling EWMA estimates over time.
        
        Args:
            returns_df: DataFrame of return series
            window: Rolling window size
            min_periods: Minimum periods for calculation
            annualize: Whether to annualize estimates
            frequency: Data frequency
            
        Returns:
            Dictionary with 'variance', 'volatility', 'correlation' DataFrames
        """
        if min_periods is None:
            min_periods = max(self.params.min_periods, window // 2)
        
        n_assets = len(returns_df.columns)
        n_periods = len(returns_df)
        
        # Initialize result containers
        variance_results = pd.DataFrame(
            index=returns_df.index, 
            columns=returns_df.columns
        )
        correlation_results = []
        
        # Calculate rolling estimates
        for i in range(min_periods, n_periods + 1):
            end_idx = i
            start_idx = max(0, end_idx - window)
            
            window_returns = returns_df.iloc[start_idx:end_idx]
            date = returns_df.index[end_idx - 1]
            
            try:
                # Estimate variance for each asset
                for asset in returns_df.columns:
                    asset_returns = window_returns[asset].dropna()
                    if len(asset_returns) >= min_periods:
                        var_estimate = self.estimate_variance(
                            asset_returns, annualize, frequency
                        ).iloc[-1]
                        variance_results.loc[date, asset] = var_estimate
                
                # Estimate correlation matrix
                if len(window_returns) >= min_periods:
                    corr_matrix = self.estimate_correlation_matrix(
                        window_returns, frequency
                    )
                    
                    # Store with date index
                    corr_with_date = corr_matrix.copy()
                    corr_with_date.index = pd.MultiIndex.from_product(
                        [[date], corr_matrix.index]
                    )
                    correlation_results.append(corr_with_date)
                    
            except Exception as e:
                logger.warning(f"Failed to estimate for period ending {date}: {e}")
                continue
        
        # Combine correlation results
        if correlation_results:
            correlation_df = pd.concat(correlation_results)
        else:
            correlation_df = pd.DataFrame()
        
        # Calculate volatility from variance
        volatility_results = variance_results.apply(pd.to_numeric, errors='coerce')
        volatility_results = np.sqrt(volatility_results)
        
        return {
            'variance': variance_results,
            'volatility': volatility_results,
            'correlation': correlation_df
        }
    
    def forecast_volatility(
        self,
        returns: pd.Series,
        horizon: int = 1,
        method: str = 'simple',
        annualize: bool = True,
        frequency: str = 'daily'
    ) -> float:
        """Forecast volatility using EWMA model.
        
        Args:
            returns: Historical return series
            horizon: Forecast horizon (periods ahead)
            method: Forecasting method ('simple', 'mean_reverting')
            annualize: Whether to annualize the forecast
            frequency: Data frequency
            
        Returns:
            Forecasted volatility
        """
        # Get current EWMA variance
        current_variance = self.estimate_variance(
            returns, annualize=False, frequency=frequency
        ).iloc[-1]
        
        if method == 'simple':
            # Simple EWMA forecast: variance is constant
            forecast_variance = current_variance
            
        elif method == 'mean_reverting':
            # Mean-reverting forecast using long-term variance
            long_term_var = returns.var()
            # Simple mean reversion with half-life of 30 days
            mean_reversion_speed = np.log(2) / 30
            
            forecast_variance = (
                long_term_var + 
                (current_variance - long_term_var) * 
                np.exp(-mean_reversion_speed * horizon)
            )
        else:
            raise ValueError(f"Unknown forecasting method: {method}")
        
        # Annualize if requested
        if annualize:
            annualization_factor = self._get_annualization_factor(frequency)
            forecast_variance *= annualization_factor
        
        return np.sqrt(forecast_variance)
    
    def _get_annualization_factor(self, frequency: Union[str, Frequency]) -> float:
        """Get annualization factor for different frequencies."""
        if isinstance(frequency, Frequency):
            return frequency.annualization_factor
        
        # String-based lookup for backward compatibility
        factors = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'annual': 1
        }
        
        if frequency not in factors:
            logger.warning(f"Unknown frequency {frequency}, using daily")
            return factors['daily']
        
        return factors[frequency]
    
    def clear_cache(self):
        """Clear estimation cache."""
        self._variance_cache.clear()
        self._covariance_cache.clear()
        logger.debug("Cleared EWMA estimation cache")


class GARCHEstimator:
    """Simple GARCH(1,1) model for volatility estimation."""
    
    def __init__(
        self,
        omega: float = 0.000001,  # Long-term variance component
        alpha: float = 0.1,       # ARCH coefficient
        beta: float = 0.85        # GARCH coefficient
    ):
        """Initialize GARCH estimator.
        
        Args:
            omega: Long-term variance component
            alpha: ARCH coefficient (impact of recent shocks)
            beta: GARCH coefficient (persistence of volatility)
        """
        if alpha + beta >= 1:
            raise ValueError("GARCH model is not stationary (alpha + beta >= 1)")
        
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        
        logger.debug(f"Initialized GARCH(1,1) with omega={omega}, alpha={alpha}, beta={beta}")
    
    def estimate_variance(
        self,
        returns: pd.Series,
        annualize: bool = True,
        frequency: str = 'daily'
    ) -> pd.Series:
        """Estimate GARCH variance for a return series.
        
        Args:
            returns: Return series
            annualize: Whether to annualize variance
            frequency: Data frequency
            
        Returns:
            Series of GARCH variance estimates
        """
        if len(returns) < 30:
            raise ValueError("Need at least 30 observations for GARCH")
        
        returns_array = returns.values
        n = len(returns_array)
        
        # Initialize variance series
        variance_series = np.zeros(n)
        
        # Initial variance (sample variance of first 30 observations)
        variance_series[0] = np.var(returns_array[:30])
        
        # GARCH recursion
        for t in range(1, n):
            variance_series[t] = (
                self.omega + 
                self.alpha * returns_array[t-1]**2 + 
                self.beta * variance_series[t-1]
            )
        
        # Create pandas Series
        garch_variance = pd.Series(variance_series, index=returns.index)
        
        # Annualize if requested
        if annualize:
            if frequency == 'daily':
                garch_variance *= 252
            elif frequency == 'monthly':
                garch_variance *= 12
        
        return garch_variance
    
    def forecast_variance(
        self,
        returns: pd.Series,
        horizon: int = 1,
        annualize: bool = True,
        frequency: str = 'daily'
    ) -> np.ndarray:
        """Forecast GARCH variance.
        
        Args:
            returns: Historical return series
            horizon: Forecast horizon
            annualize: Whether to annualize
            frequency: Data frequency
            
        Returns:
            Array of variance forecasts
        """
        # Get current variance estimate
        current_variance = self.estimate_variance(
            returns, annualize=False, frequency=frequency
        ).iloc[-1]
        
        # Calculate long-term variance
        long_term_variance = self.omega / (1 - self.alpha - self.beta)
        
        # Multi-step forecast
        forecasts = np.zeros(horizon)
        
        for h in range(horizon):
            if h == 0:
                forecasts[h] = (
                    self.omega + 
                    (self.alpha + self.beta) * current_variance
                )
            else:
                forecasts[h] = (
                    long_term_variance + 
                    (self.alpha + self.beta)**h * (current_variance - long_term_variance)
                )
        
        # Annualize if requested
        if annualize:
            if frequency == 'daily':
                forecasts *= 252
            elif frequency == 'monthly':
                forecasts *= 12
        
        return forecasts