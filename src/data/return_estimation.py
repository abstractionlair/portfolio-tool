"""
Return Estimation Framework for Portfolio Optimization.

This module provides comprehensive return and risk estimation capabilities
including real returns, geometric returns, and various estimation methods.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf, EmpiricalCovariance

from .exposure_universe import ExposureUniverse, Exposure
from .total_returns import TotalReturnFetcher
from .fred_data import FREDDataFetcher

logger = logging.getLogger(__name__)


class ReturnEstimationFramework:
    """Comprehensive framework for estimating returns and covariances."""
    
    def __init__(self, 
                 total_return_fetcher: Optional[TotalReturnFetcher] = None,
                 fred_fetcher: Optional[FREDDataFetcher] = None):
        """Initialize the return estimation framework.
        
        Args:
            total_return_fetcher: TotalReturnFetcher instance
            fred_fetcher: FREDDataFetcher instance
        """
        self.total_return_fetcher = total_return_fetcher or TotalReturnFetcher()
        self.fred_fetcher = fred_fetcher or FREDDataFetcher()
        
        # Cache for computed results
        self._return_cache = {}
        self._covariance_cache = {}
    
    def estimate_real_returns(
        self,
        universe: ExposureUniverse,
        start_date: datetime,
        end_date: datetime,
        method: str = "historical",
        lookback_years: Optional[int] = None,
        frequency: str = "monthly",
        inflation_series: str = "cpi_all",
        risk_free_maturity: str = "3m"
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Estimate real returns for all exposures in the universe.
        
        Args:
            universe: ExposureUniverse object
            start_date: Start date for data
            end_date: End date for data
            method: 'historical', 'shrinkage', 'capm_adjusted'
            lookback_years: Override date range with lookback period
            frequency: 'daily', 'weekly', 'monthly'
            inflation_series: Inflation series to use
            risk_free_maturity: Risk-free rate maturity
            
        Returns:
            Tuple of (returns_dataframe, implementation_dict)
        """
        if lookback_years:
            start_date = end_date - timedelta(days=lookback_years * 365)
        
        cache_key = f"real_returns_{hash(str(universe.exposures.keys()))}_{start_date}_{end_date}_{method}_{frequency}"
        if cache_key in self._return_cache:
            logger.debug("Using cached real returns")
            return self._return_cache[cache_key]
        
        logger.info(f"Estimating real returns using {method} method from {start_date.date()} to {end_date.date()}")
        
        # Step 1: Fetch nominal returns for all exposures
        logger.info("Fetching nominal returns for all exposures...")
        universe_returns = self.total_return_fetcher.fetch_universe_returns(
            universe, start_date, end_date, frequency
        )
        
        # Step 2: Fetch inflation data
        logger.info(f"Fetching {inflation_series} inflation data...")
        inflation_index = self.fred_fetcher.fetch_inflation_data(
            start_date, end_date, inflation_series, frequency
        )
        
        if inflation_index.empty:
            logger.warning("No inflation data available, using nominal returns")
            inflation_rates = pd.Series(dtype=float)
        else:
            # Match inflation frequency to return frequency
            # If returns are monthly, inflation should be monthly too
            should_annualize = frequency == "annual"
            inflation_rates = self.fred_fetcher.calculate_inflation_rate(
                inflation_index, periods=1, annualize=should_annualize
            )
        
        # Step 3: Convert to real returns
        real_returns_data = {}
        implementation_info = {}
        
        for exposure_id, return_data in universe_returns.items():
            if not return_data['success'] or return_data['returns'].empty:
                logger.warning(f"Skipping {exposure_id} - no data available")
                continue
            
            nominal_returns = return_data['returns']
            implementation_info[exposure_id] = return_data['implementation']
            
            if not inflation_rates.empty:
                # Convert to real returns
                real_returns = self.fred_fetcher.convert_to_real_returns(
                    nominal_returns, inflation_rates, method="exact"
                )
            else:
                # Use nominal returns if no inflation data
                real_returns = nominal_returns
                logger.warning(f"Using nominal returns for {exposure_id} - no inflation data")
            
            if not real_returns.empty:
                real_returns_data[exposure_id] = real_returns
        
        if not real_returns_data:
            logger.error("No valid return data for any exposure")
            return pd.DataFrame(), {}
        
        # Step 4: Create aligned DataFrame
        returns_df = pd.DataFrame(real_returns_data)
        returns_df = returns_df.dropna()  # Remove dates with missing data
        
        if returns_df.empty:
            logger.error("No overlapping return data after alignment")
            return pd.DataFrame(), implementation_info
        
        logger.info(f"Aligned return data: {len(returns_df)} observations for {len(returns_df.columns)} exposures")
        
        # Step 5: Apply estimation method
        if method == "historical":
            estimated_returns = self._estimate_historical_real_returns(returns_df)
        elif method == "shrinkage":
            estimated_returns = self._estimate_shrinkage_real_returns(returns_df)
        elif method == "capm_adjusted":
            # For CAPM, we need risk-free rate
            risk_free_rates = self.fred_fetcher.fetch_risk_free_rate(
                start_date, end_date, risk_free_maturity, frequency
            )
            estimated_returns = self._estimate_capm_adjusted_returns(returns_df, risk_free_rates)
        else:
            raise ValueError(f"Unsupported estimation method: {method}")
        
        # Cache results
        result = (estimated_returns, implementation_info)
        self._return_cache[cache_key] = result
        
        logger.info(f"Successfully estimated real returns for {len(estimated_returns)} exposures")
        return result
    
    def estimate_covariance_matrix(
        self,
        returns_df: pd.DataFrame,
        method: str = "sample",
        shrinkage_target: str = "diagonal",
        frequency: str = "monthly"
    ) -> np.ndarray:
        """Estimate covariance matrix with various methods.
        
        Args:
            returns_df: DataFrame of returns
            method: 'sample', 'shrinkage', 'ledoit_wolf'
            shrinkage_target: 'diagonal', 'identity', 'constant_correlation'
            frequency: Data frequency for annualization
            
        Returns:
            Annualized covariance matrix
        """
        if returns_df.empty:
            return np.array([])
        
        cache_key = f"covariance_{hash(str(returns_df.columns.tolist()))}_{method}_{shrinkage_target}_{len(returns_df)}"
        if cache_key in self._covariance_cache:
            logger.debug("Using cached covariance matrix")
            return self._covariance_cache[cache_key]
        
        logger.info(f"Estimating covariance matrix using {method} method")
        
        # Remove any remaining NaN values
        clean_returns = returns_df.dropna()
        
        if clean_returns.empty:
            logger.error("No valid data for covariance estimation")
            return np.eye(len(returns_df.columns)) * 0.04  # Default 20% vol
        
        if method == "sample":
            cov_matrix = self._estimate_sample_covariance(clean_returns)
        elif method == "shrinkage":
            cov_matrix = self._estimate_shrinkage_covariance(clean_returns, shrinkage_target)
        elif method == "ledoit_wolf":
            cov_matrix = self._estimate_ledoit_wolf_covariance(clean_returns)
        else:
            raise ValueError(f"Unsupported covariance estimation method: {method}")
        
        # Annualize based on frequency
        annualization_factor = self._get_annualization_factor(frequency)
        cov_matrix_annualized = cov_matrix * annualization_factor
        
        # Cache result
        self._covariance_cache[cache_key] = cov_matrix_annualized
        
        logger.info(f"Estimated {cov_matrix_annualized.shape[0]}x{cov_matrix_annualized.shape[1]} covariance matrix")
        return cov_matrix_annualized
    
    def _estimate_historical_real_returns(self, returns_df: pd.DataFrame) -> pd.Series:
        """Estimate historical mean real returns."""
        return returns_df.mean() * self._get_annualization_factor("monthly")
    
    def _estimate_shrinkage_real_returns(self, returns_df: pd.DataFrame, shrinkage_intensity: float = 0.2) -> pd.Series:
        """Estimate real returns with shrinkage toward grand mean."""
        historical_means = returns_df.mean()
        grand_mean = historical_means.mean()
        
        # Shrink toward grand mean
        shrunk_returns = (1 - shrinkage_intensity) * historical_means + shrinkage_intensity * grand_mean
        
        return shrunk_returns * self._get_annualization_factor("monthly")
    
    def _estimate_capm_adjusted_returns(self, returns_df: pd.DataFrame, risk_free_rates: pd.Series) -> pd.Series:
        """Estimate returns using CAPM adjustments."""
        if risk_free_rates.empty:
            logger.warning("No risk-free rate data, falling back to historical means")
            return self._estimate_historical_real_returns(returns_df)
        
        # Align data
        aligned_data = pd.concat([returns_df, risk_free_rates.rename('rf_rate')], axis=1).dropna()
        
        if aligned_data.empty:
            logger.warning("No aligned data for CAPM estimation")
            return self._estimate_historical_real_returns(returns_df)
        
        rf_rate = aligned_data['rf_rate']
        asset_returns = aligned_data[returns_df.columns]
        
        # For now, use simple historical means adjusted by current risk-free environment
        # More sophisticated CAPM implementation could be added later
        historical_means = asset_returns.mean()
        current_rf = rf_rate.mean() if not rf_rate.empty else 0.02
        
        # Simple adjustment: add current risk-free rate
        adjusted_returns = historical_means + current_rf
        
        return adjusted_returns * self._get_annualization_factor("monthly")
    
    def _estimate_sample_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Estimate sample covariance matrix."""
        return returns_df.cov().values
    
    def _estimate_shrinkage_covariance(self, returns_df: pd.DataFrame, target: str) -> np.ndarray:
        """Estimate covariance with shrinkage."""
        sample_cov = returns_df.cov().values
        
        if target == "diagonal":
            # Shrink toward diagonal matrix
            shrinkage_target = np.diag(np.diag(sample_cov))
        elif target == "identity":
            # Shrink toward identity scaled by average variance
            avg_var = np.mean(np.diag(sample_cov))
            shrinkage_target = np.eye(sample_cov.shape[0]) * avg_var
        elif target == "constant_correlation":
            # Shrink toward constant correlation matrix
            vols = np.sqrt(np.diag(sample_cov))
            avg_corr = np.mean(sample_cov / np.outer(vols, vols) - np.eye(len(vols)))
            shrinkage_target = avg_corr * np.outer(vols, vols) + np.diag(vols ** 2)
        else:
            raise ValueError(f"Unsupported shrinkage target: {target}")
        
        # Apply shrinkage (simple 20% shrinkage)
        shrinkage_intensity = 0.2
        shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * shrinkage_target
        
        return shrunk_cov
    
    def _estimate_ledoit_wolf_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Estimate covariance using Ledoit-Wolf shrinkage."""
        try:
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_df.values).covariance_
            return cov_matrix
        except Exception as e:
            logger.warning(f"Ledoit-Wolf estimation failed: {e}, falling back to sample covariance")
            return self._estimate_sample_covariance(returns_df)
    
    def _get_annualization_factor(self, frequency: str) -> float:
        """Get annualization factor for given frequency."""
        factors = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'annual': 1
        }
        return factors.get(frequency, 12)  # Default to monthly
    
    def calculate_geometric_returns(self, arithmetic_returns: pd.Series, frequency: str = "daily") -> pd.Series:
        """Convert arithmetic to geometric returns.
        
        Args:
            arithmetic_returns: Series of arithmetic returns
            frequency: Return frequency
            
        Returns:
            Series of geometric returns
        """
        if arithmetic_returns.empty:
            return pd.Series(dtype=float)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + arithmetic_returns).cumprod()
        
        # Calculate geometric mean
        n_periods = len(arithmetic_returns)
        if n_periods == 0:
            return pd.Series(dtype=float)
        
        # Geometric return = (final_value / initial_value)^(1/n) - 1
        geometric_return = cumulative_returns.iloc[-1] ** (1/n_periods) - 1
        
        # Annualize
        annualization_factor = self._get_annualization_factor(frequency)
        geometric_return_annualized = (1 + geometric_return) ** annualization_factor - 1
        
        return pd.Series([geometric_return_annualized], index=[arithmetic_returns.name])
    
    def get_estimation_summary(self, 
                             universe: ExposureUniverse,
                             start_date: datetime,
                             end_date: datetime) -> pd.DataFrame:
        """Get comprehensive summary of data availability and quality.
        
        Args:
            universe: ExposureUniverse object
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with estimation summary
        """
        summary_data = []
        
        # Fetch data for all exposures
        universe_returns = self.total_return_fetcher.fetch_universe_returns(
            universe, start_date, end_date, "monthly"
        )
        
        for exposure_id, return_data in universe_returns.items():
            exposure = universe.get_exposure(exposure_id)
            
            summary = {
                'exposure_id': exposure_id,
                'name': exposure.name if exposure else 'Unknown',
                'category': exposure.category if exposure else 'Unknown',
                'implementation': return_data['implementation'],
                'success': return_data['success'],
                'observations': len(return_data['returns']) if return_data['success'] else 0,
                'start_date': return_data['returns'].index[0] if return_data['success'] and not return_data['returns'].empty else None,
                'end_date': return_data['returns'].index[-1] if return_data['success'] and not return_data['returns'].empty else None,
                'mean_return': return_data['returns'].mean() if return_data['success'] and not return_data['returns'].empty else None,
                'volatility': return_data['returns'].std() if return_data['success'] and not return_data['returns'].empty else None,
                'years_data': (return_data['returns'].index[-1] - return_data['returns'].index[0]).days / 365.25 if return_data['success'] and not return_data['returns'].empty else 0
            }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def validate_estimation_inputs(self, 
                                 universe: ExposureUniverse,
                                 start_date: datetime,
                                 end_date: datetime) -> Dict[str, Union[bool, List[str]]]:
        """Validate inputs for return estimation.
        
        Args:
            universe: ExposureUniverse object
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict with validation results
        """
        issues = []
        
        # Check date range
        if end_date <= start_date:
            issues.append("End date must be after start date")
        
        date_range_years = (end_date - start_date).days / 365.25
        if date_range_years < 2:
            issues.append(f"Date range too short: {date_range_years:.1f} years (minimum 2 years recommended)")
        
        # Check universe
        if len(universe) == 0:
            issues.append("Empty exposure universe")
        
        # Check data availability
        try:
            summary = self.get_estimation_summary(universe, start_date, end_date)
            successful_exposures = summary['success'].sum()
            total_exposures = len(summary)
            
            if successful_exposures == 0:
                issues.append("No exposures have available data")
            elif successful_exposures < total_exposures * 0.5:
                issues.append(f"Limited data availability: only {successful_exposures}/{total_exposures} exposures have data")
            
            # Check for sufficient history
            insufficient_history = summary[summary['years_data'] < 3]
            if len(insufficient_history) > 0:
                issues.append(f"{len(insufficient_history)} exposures have less than 3 years of data")
        
        except Exception as e:
            issues.append(f"Error checking data availability: {str(e)}")
        
        # Check FRED connectivity
        try:
            self.fred_fetcher.get_latest_rates()
        except Exception as e:
            issues.append(f"FRED data access issue: {str(e)}")
        
        valid = len(issues) == 0
        return {'valid': valid, 'issues': issues}