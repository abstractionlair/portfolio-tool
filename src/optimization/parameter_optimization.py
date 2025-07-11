"""
Multi-Frequency Parameter Optimization Framework.

This module implements comprehensive parameter optimization for:
- Frequency selection (daily, weekly, monthly, quarterly)
- EWMA parameters (lambda, min_periods)
- Volatility forecasting validation
- Correlation forecasting optimization
- Expected return estimation parameters

Uses real exposure universe data for proper backtesting and validation.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from itertools import product
import warnings
warnings.filterwarnings('ignore')

try:
    from ..data.multi_frequency import Frequency, MultiFrequencyDataFetcher, FrequencyConverter
    from ..optimization.ewma import EWMAEstimator, EWMAParameters
    from ..data.exposure_universe import ExposureUniverse
    from ..data.total_returns import TotalReturnFetcher
except ImportError:
    # Fallback for direct execution
    from data.multi_frequency import Frequency, MultiFrequencyDataFetcher, FrequencyConverter
    from optimization.ewma import EWMAEstimator, EWMAParameters
    from data.exposure_universe import ExposureUniverse
    from data.total_returns import TotalReturnFetcher

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    # Frequency optimization
    test_frequencies: List[Frequency] = field(default_factory=lambda: [
        Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY, Frequency.QUARTERLY
    ])
    
    # EWMA parameter grid
    lambda_values: List[float] = field(default_factory=lambda: [0.90, 0.92, 0.94, 0.96, 0.98])
    min_periods_values: List[int] = field(default_factory=lambda: [20, 30, 50, 100])
    
    # Backtesting configuration
    min_train_periods: int = 100  # Reduced minimum training periods for better validation
    test_periods: int = 30        # Reduced test periods for more data efficiency
    rolling_window: bool = True   # Use rolling window validation
    
    # Forecasting horizons
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 5, 21, 63])  # 1d, 1w, 1m, 1q
    
    # Validation metrics
    volatility_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'qlike', 'hit_rate'
    ])
    correlation_metrics: List[str] = field(default_factory=lambda: [
        'frobenius_norm', 'correlation_distance', 'eigenvalue_distance'
    ])


@dataclass
class ValidationResult:
    """Results from parameter validation."""
    frequency: Frequency
    lambda_param: float
    min_periods: int
    horizon: int
    
    # Volatility forecasting metrics
    volatility_mse: float = 0.0
    volatility_mae: float = 0.0
    volatility_qlike: float = 0.0
    volatility_hit_rate: float = 0.0
    
    # Correlation forecasting metrics
    correlation_frobenius: float = 0.0
    correlation_distance: float = 0.0
    eigenvalue_distance: float = 0.0
    
    # Summary metrics
    combined_score: float = 0.0
    sample_size: int = 0


class ParameterOptimizer:
    """Comprehensive parameter optimizer for multi-frequency models."""
    
    def __init__(
        self,
        exposure_universe: ExposureUniverse,
        config: Optional[OptimizationConfig] = None,
        data_fetcher: Optional[TotalReturnFetcher] = None
    ):
        """Initialize parameter optimizer.
        
        Args:
            exposure_universe: Exposure universe for testing
            config: Optimization configuration
            data_fetcher: Data fetcher for historical data
        """
        self.exposure_universe = exposure_universe
        self.config = config or OptimizationConfig()
        self.data_fetcher = data_fetcher or TotalReturnFetcher()
        self.multi_freq_fetcher = MultiFrequencyDataFetcher()
        
        # Results storage
        self.validation_results: List[ValidationResult] = []
        self.optimal_parameters: Dict[str, Any] = {}
        
        logger.info(f"Initialized ParameterOptimizer with {len(exposure_universe)} exposures")
    
    def optimize_for_horizon(
        self,
        exposures: List[str],
        target_horizon: int,
        start_date: datetime,
        end_date: datetime,
        validation_method: str = "walk_forward"
    ) -> Dict[str, Any]:
        """
        Optimize parameters for all exposures for a specific forecast horizon.
        
        Args:
            exposures: List of exposure IDs to optimize
            target_horizon: Forecast horizon in days (must be same for all)
            start_date: Start of historical data
            end_date: End of historical data
            validation_method: Method for validation
            
        Returns:
            Dictionary of exposure_id -> optimal parameters for the horizon
        """
        logger.info(f"Starting horizon-specific parameter optimization for {target_horizon}-day horizon...")
        
        # Store target horizon for use in validation
        self.target_horizon = target_horizon
        
        # Get exposures to test
        if exposures:
            exposure_objects = [self.exposure_universe.get_exposure(exp_id) 
                              for exp_id in exposures if self.exposure_universe.get_exposure(exp_id)]
        else:
            exposure_objects = list(self.exposure_universe.exposures.values())
        
        # Filter for implementable exposures
        implementable_exposures = []
        for exposure in exposure_objects:
            impl = exposure.get_preferred_implementation()
            if impl and impl.get_primary_tickers():
                implementable_exposures.append(exposure)
        
        logger.info(f"Testing {len(implementable_exposures)} implementable exposures for {target_horizon}-day horizon")
        
        # Load historical data for all exposures
        exposure_data = self._load_exposure_data(implementable_exposures, start_date, end_date)
        
        if not exposure_data:
            raise ValueError("No exposure data could be loaded for optimization")
        
        # Run parameter grid search with horizon-specific validation
        self._run_horizon_specific_grid_search(exposure_data, start_date, end_date, target_horizon)
        
        # Analyze results and find optimal parameters for this horizon
        self.optimal_parameters = self._analyze_horizon_optimization_results(target_horizon)
        
        logger.info(f"Parameter optimization completed for {target_horizon}-day horizon")
        return self.optimal_parameters

    def optimize_all_parameters(
        self,
        start_date: datetime,
        end_date: datetime,
        exposure_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Optimize all parameters using comprehensive backtesting.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            exposure_ids: Specific exposures to test (None for all)
            
        Returns:
            Dictionary with optimal parameters and results
        """
        logger.info("Starting comprehensive parameter optimization...")
        
        # Get exposures to test
        if exposure_ids:
            exposures = [self.exposure_universe.get_exposure(exp_id) 
                        for exp_id in exposure_ids if self.exposure_universe.get_exposure(exp_id)]
        else:
            exposures = list(self.exposure_universe.exposures.values())
        
        # Filter for implementable exposures
        implementable_exposures = []
        for exposure in exposures:
            impl = exposure.get_preferred_implementation()
            if impl and impl.get_primary_tickers():
                implementable_exposures.append(exposure)
        
        logger.info(f"Testing {len(implementable_exposures)} implementable exposures")
        
        # Load historical data for all exposures
        exposure_data = self._load_exposure_data(implementable_exposures, start_date, end_date)
        
        if not exposure_data:
            raise ValueError("No exposure data could be loaded for optimization")
        
        # Run parameter grid search
        self._run_parameter_grid_search(exposure_data, start_date, end_date)
        
        # Analyze results and find optimal parameters
        self.optimal_parameters = self._analyze_optimization_results()
        
        logger.info("Parameter optimization completed")
        return self.optimal_parameters
    
    def _load_exposure_data(
        self,
        exposures: List,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.Series]:
        """Load historical return data for exposures."""
        exposure_data = {}
        
        for exposure in exposures:
            try:
                impl = exposure.get_preferred_implementation()
                if not impl:
                    continue
                
                tickers = impl.get_primary_tickers()
                if not tickers:
                    continue
                
                # For single ticker implementations
                if len(tickers) == 1:
                    ticker = tickers[0]
                    try:
                        # Use multi-frequency fetcher for consistent data
                        returns = self.multi_freq_fetcher._fetch_single_ticker_returns(
                            ticker, start_date, end_date, Frequency.DAILY, validate=True
                        )
                        
                        if len(returns) > self.config.min_train_periods:
                            exposure_data[exposure.id] = returns
                            logger.debug(f"Loaded {len(returns)} periods for {exposure.id} ({ticker})")
                        else:
                            logger.warning(f"Insufficient data for {exposure.id}: {len(returns)} periods")
                            
                    except Exception as e:
                        logger.warning(f"Failed to load data for {exposure.id} ({ticker}): {e}")
                        continue
                
                # For multi-ticker implementations (ETF averages)
                elif impl.type == 'etf_average':
                    try:
                        returns_list = []
                        for ticker in tickers:
                            ticker_returns = self.multi_freq_fetcher._fetch_single_ticker_returns(
                                ticker, start_date, end_date, Frequency.DAILY, validate=True
                            )
                            if len(ticker_returns) > 0:
                                returns_list.append(ticker_returns)
                        
                        if returns_list:
                            # Average the returns across tickers
                            combined_df = pd.DataFrame(returns_list).T
                            avg_returns = combined_df.mean(axis=1, skipna=True).dropna()
                            
                            if len(avg_returns) > self.config.min_train_periods:
                                exposure_data[exposure.id] = avg_returns
                                logger.debug(f"Loaded averaged data for {exposure.id}: {len(avg_returns)} periods")
                            
                    except Exception as e:
                        logger.warning(f"Failed to load ETF average data for {exposure.id}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing exposure {exposure.id}: {e}")
                continue
        
        logger.info(f"Successfully loaded data for {len(exposure_data)} exposures")
        return exposure_data
    
    def _run_parameter_grid_search(
        self,
        exposure_data: Dict[str, pd.Series],
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Run comprehensive grid search over all parameters."""
        logger.info("Running parameter grid search...")
        
        # Create parameter combinations
        param_combinations = list(product(
            self.config.test_frequencies,
            self.config.lambda_values,
            self.config.min_periods_values,
            self.config.forecast_horizons
        ))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Test each combination
        for i, (frequency, lambda_val, min_periods, horizon) in enumerate(param_combinations):
            if i % 20 == 0:
                logger.info(f"Progress: {i+1}/{len(param_combinations)} combinations")
            
            try:
                result = self._validate_parameter_combination(
                    exposure_data, frequency, lambda_val, min_periods, horizon
                )
                
                if result and result.sample_size > 0:
                    self.validation_results.append(result)
                    
            except Exception as e:
                logger.debug(f"Failed parameter combination {frequency.value}, λ={lambda_val}, "
                           f"min_periods={min_periods}, horizon={horizon}: {e}")
                continue
        
        logger.info(f"Completed grid search: {len(self.validation_results)} valid results")
    
    def _validate_parameter_combination(
        self,
        exposure_data: Dict[str, pd.Series],
        frequency: Frequency,
        lambda_val: float,
        min_periods: int,
        horizon: int
    ) -> Optional[ValidationResult]:
        """Validate a specific parameter combination."""
        
        # Initialize EWMA estimator
        ewma_params = EWMAParameters(lambda_=lambda_val, min_periods=min_periods)
        ewma_estimator = EWMAEstimator(ewma_params)
        
        # Collect metrics across all exposures
        volatility_metrics = []
        correlation_metrics = []
        
        for exposure_id, returns in exposure_data.items():
            try:
                # Convert to target frequency
                freq_returns = self._convert_to_frequency(returns, frequency)
                
                if len(freq_returns) < min_periods + horizon + self.config.test_periods:
                    continue
                
                # Run time series validation
                vol_metrics = self._validate_volatility_forecasting(
                    freq_returns, ewma_estimator, horizon, frequency
                )
                
                if vol_metrics:
                    volatility_metrics.append(vol_metrics)
                    
            except Exception as e:
                logger.debug(f"Failed validation for {exposure_id}: {e}")
                continue
        
        # Run correlation validation if we have multiple exposures
        if len(exposure_data) >= 2:
            try:
                corr_metrics = self._validate_correlation_forecasting(
                    exposure_data, frequency, ewma_estimator, horizon
                )
                if corr_metrics:
                    correlation_metrics.append(corr_metrics)
            except Exception as e:
                logger.debug(f"Failed correlation validation: {e}")
        
        # Aggregate results
        if not volatility_metrics:
            return None
        
        result = ValidationResult(
            frequency=frequency,
            lambda_param=lambda_val,
            min_periods=min_periods,
            horizon=horizon,
            sample_size=len(volatility_metrics)
        )
        
        # Aggregate volatility metrics
        vol_df = pd.DataFrame(volatility_metrics)
        result.volatility_mse = vol_df['mse'].mean()
        result.volatility_mae = vol_df['mae'].mean()
        result.volatility_qlike = vol_df['qlike'].mean()
        result.volatility_hit_rate = vol_df['hit_rate'].mean()
        
        # Aggregate correlation metrics if available
        if correlation_metrics:
            corr_df = pd.DataFrame(correlation_metrics)
            result.correlation_frobenius = corr_df['frobenius_norm'].mean()
            result.correlation_distance = corr_df['correlation_distance'].mean()
            result.eigenvalue_distance = corr_df['eigenvalue_distance'].mean()
        
        # Calculate combined score (lower is better for most metrics)
        result.combined_score = self._calculate_combined_score(result)
        
        return result
    
    def _convert_to_frequency(self, returns: pd.Series, frequency: Frequency) -> pd.Series:
        """Convert returns to target frequency."""
        try:
            from ..data.multi_frequency import ReturnCompounding
        except ImportError:
            from data.multi_frequency import ReturnCompounding
        return ReturnCompounding.resample_returns(returns, frequency)
    
    def _validate_volatility_forecasting(
        self,
        returns: pd.Series,
        ewma_estimator: EWMAEstimator,
        horizon: int,
        frequency: Frequency
    ) -> Optional[Dict[str, float]]:
        """Validate volatility forecasting for a single return series."""
        
        min_train = ewma_estimator.params.min_periods
        test_periods = self.config.test_periods
        
        if len(returns) < min_train + horizon + test_periods:
            return None
        
        forecasts = []
        realized_vols = []
        
        # Rolling window validation
        for i in range(min_train, len(returns) - horizon - test_periods, 5):  # Step by 5 for better coverage
            try:
                # Training data
                train_returns = returns.iloc[:i]
                
                # Forecast volatility
                forecast_vol = ewma_estimator.forecast_volatility(
                    train_returns, horizon=horizon, method='simple',
                    annualize=False, frequency=frequency
                )
                
                # Realized volatility over forecast horizon
                future_returns = returns.iloc[i:i+horizon]
                if len(future_returns) == horizon:
                    # Fix: For single period, use absolute return instead of std
                    if horizon == 1:
                        realized_vol = abs(future_returns.iloc[0])
                    else:
                        realized_vol = future_returns.std()
                    
                    # Only add if both values are valid and finite
                    if np.isfinite(forecast_vol) and np.isfinite(realized_vol) and realized_vol > 0:
                        forecasts.append(forecast_vol)
                        realized_vols.append(realized_vol)
                    
            except Exception:
                continue
        
        if len(forecasts) < 5:  # Reduced minimum threshold
            return None
        
        forecasts = np.array(forecasts)
        realized_vols = np.array(realized_vols)
        
        # Calculate metrics with validation
        valid_mask = np.isfinite(forecasts) & np.isfinite(realized_vols) & (forecasts > 0) & (realized_vols > 0)
        
        if valid_mask.sum() == 0:
            return None
            
        valid_forecasts = forecasts[valid_mask]
        valid_realized = realized_vols[valid_mask]
        
        mse = np.mean((valid_forecasts - valid_realized) ** 2)
        mae = np.mean(np.abs(valid_forecasts - valid_realized))
        
        # QLIKE loss (Quasi-likelihood) with protection against extreme values
        try:
            qlike = np.mean(valid_realized ** 2 / valid_forecasts ** 2 + np.log(valid_forecasts ** 2))
        except (ValueError, RuntimeWarning):
            qlike = np.inf
        
        # Hit rate (percentage of times forecast captures realized vol within 20%)
        # Add validation to prevent division by zero or invalid values
        valid_mask = (realized_vols > 0) & np.isfinite(realized_vols) & np.isfinite(forecasts)
        if valid_mask.sum() > 0:
            relative_errors = np.abs(forecasts[valid_mask] - realized_vols[valid_mask]) / realized_vols[valid_mask]
            hit_rate = np.mean(relative_errors < 0.20)
        else:
            hit_rate = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'qlike': qlike,
            'hit_rate': hit_rate
        }
    
    def _validate_correlation_forecasting(
        self,
        exposure_data: Dict[str, pd.Series],
        frequency: Frequency,
        ewma_estimator: EWMAEstimator,
        horizon: int
    ) -> Optional[Dict[str, float]]:
        """Validate correlation forecasting across multiple exposures."""
        
        # Select a subset of exposures for correlation testing
        exposure_ids = list(exposure_data.keys())[:10]  # Limit to 10 for computational efficiency
        
        if len(exposure_ids) < 2:
            return None
        
        # Align data and convert to frequency
        aligned_data = {}
        for exp_id in exposure_ids:
            freq_returns = self._convert_to_frequency(exposure_data[exp_id], frequency)
            aligned_data[exp_id] = freq_returns
        
        # Create DataFrame with common index
        returns_df = pd.DataFrame(aligned_data).dropna()
        
        if len(returns_df) < ewma_estimator.params.min_periods + horizon + 20:
            return None
        
        min_train = ewma_estimator.params.min_periods
        
        forecast_corrs = []
        realized_corrs = []
        
        # Rolling validation
        for i in range(min_train, len(returns_df) - horizon, 20):  # Step by 20 for speed
            try:
                # Training data
                train_data = returns_df.iloc[:i]
                
                # Forecast correlation matrix
                forecast_corr = ewma_estimator.estimate_correlation_matrix(
                    train_data, frequency=frequency
                )
                
                # Realized correlation over forecast horizon
                future_data = returns_df.iloc[i:i+horizon]
                if len(future_data) >= horizon:
                    realized_corr = future_data.corr()
                    
                    forecast_corrs.append(forecast_corr.values)
                    realized_corrs.append(realized_corr.values)
                    
            except Exception:
                continue
        
        if len(forecast_corrs) < 5:  # Need at least 5 correlation forecasts
            return None
        
        forecast_corrs = np.array(forecast_corrs)
        realized_corrs = np.array(realized_corrs)
        
        # Calculate correlation metrics
        frobenius_norms = []
        correlation_distances = []
        eigenvalue_distances = []
        
        for fc, rc in zip(forecast_corrs, realized_corrs):
            try:
                # Check for NaN or infinite values
                if not (np.isfinite(fc).all() and np.isfinite(rc).all()):
                    continue
                
                # Frobenius norm of difference
                frobenius_norms.append(np.linalg.norm(fc - rc, 'fro'))
                
                # Correlation distance
                corr_coef = np.corrcoef(fc.flatten(), rc.flatten())
                if np.isfinite(corr_coef).all():
                    correlation_distances.append(1 - corr_coef[0, 1])
                
                # Eigenvalue distance
                fc_eigs = np.linalg.eigvals(fc)
                rc_eigs = np.linalg.eigvals(rc)
                if np.isfinite(fc_eigs).all() and np.isfinite(rc_eigs).all():
                    fc_eigs.sort()
                    rc_eigs.sort()
                    eigenvalue_distances.append(np.mean(np.abs(fc_eigs - rc_eigs)))
                    
            except (np.linalg.LinAlgError, ValueError):
                # Skip problematic correlation matrices
                continue
        
        # Return metrics only if we have valid data
        if not frobenius_norms:
            return None
        
        return {
            'frobenius_norm': np.mean(frobenius_norms) if frobenius_norms else 0.0,
            'correlation_distance': np.mean(correlation_distances) if correlation_distances else 0.0,
            'eigenvalue_distance': np.mean(eigenvalue_distances) if eigenvalue_distances else 0.0
        }
    
    def _calculate_combined_score(self, result: ValidationResult) -> float:
        """Calculate combined score for parameter ranking."""
        
        # Normalize and weight different metrics
        vol_score = (
            0.3 * result.volatility_mse +  # MSE (lower is better)
            0.2 * result.volatility_mae +  # MAE (lower is better)
            0.3 * result.volatility_qlike +  # QLIKE (lower is better)
            0.2 * (1 - result.volatility_hit_rate)  # Hit rate (higher is better)
        )
        
        # Correlation score (if available)
        if result.correlation_frobenius > 0:
            corr_score = (
                0.4 * result.correlation_frobenius +
                0.3 * result.correlation_distance +
                0.3 * result.eigenvalue_distance
            )
            combined_score = 0.7 * vol_score + 0.3 * corr_score
        else:
            combined_score = vol_score
        
        return combined_score
    
    def _run_horizon_specific_grid_search(
        self,
        exposure_data: Dict[str, pd.Series],
        start_date: datetime,
        end_date: datetime,
        target_horizon: int
    ) -> None:
        """Run grid search optimized for specific forecast horizon."""
        logger.info(f"Running horizon-specific grid search for {target_horizon}-day forecasts...")
        
        # Create parameter combinations (exclude horizon since it's fixed)
        param_combinations = list(product(
            [Frequency.DAILY],  # Focus on daily frequency for horizon-specific optimization
            self.config.lambda_values,
            self.config.min_periods_values
        ))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations for {target_horizon}-day horizon")
        
        # Test each combination with the fixed horizon
        for i, (frequency, lambda_val, min_periods) in enumerate(param_combinations):
            if i % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(param_combinations)} combinations")
            
            try:
                result = self._validate_parameter_combination(
                    exposure_data, frequency, lambda_val, min_periods, target_horizon
                )
                
                if result and result.sample_size > 0:
                    self.validation_results.append(result)
                    
            except Exception as e:
                logger.debug(f"Failed parameter combination λ={lambda_val}, "
                           f"min_periods={min_periods}, horizon={target_horizon}: {e}")
                continue
        
        logger.info(f"Completed horizon-specific grid search: {len(self.validation_results)} valid results")
    
    def _analyze_horizon_optimization_results(self, target_horizon: int) -> Dict[str, Any]:
        """Analyze optimization results for specific horizon."""
        if not self.validation_results:
            logger.warning("No validation results to analyze")
            return {}
        
        logger.info(f"Analyzing {len(self.validation_results)} validation results for {target_horizon}-day horizon")
        
        # Convert results to DataFrame for analysis
        results_data = []
        for result in self.validation_results:
            if result.horizon == target_horizon:  # Only include results for target horizon
                results_data.append({
                    'frequency': result.frequency.value,
                    'lambda': result.lambda_param,
                    'min_periods': result.min_periods,
                    'horizon': result.horizon,
                    'volatility_mse': result.volatility_mse,
                    'volatility_mae': result.volatility_mae,
                    'volatility_qlike': result.volatility_qlike,
                    'correlation_frobenius': result.correlation_frobenius,
                    'combined_score': result.combined_score,
                    'sample_size': result.sample_size
                })
        
        if not results_data:
            logger.warning(f"No results found for horizon {target_horizon}")
            return {}
        
        results_df = pd.DataFrame(results_data)
        
        # Find optimal parameters for different components
        optimal_params = {
            'global_settings': {
                'forecast_horizon': target_horizon,
                'optimization_date': datetime.now().strftime('%Y-%m-%d'),
                'total_combinations_tested': len(results_data),
                'best_combined_score': results_df['combined_score'].min()
            },
            f'horizon_{target_horizon}_parameters': {
                'volatility': {},
                'correlation': {}
            }
        }
        
        # Find best volatility parameters (lowest MSE)
        best_vol_idx = results_df['volatility_mse'].idxmin()
        best_vol_result = results_df.loc[best_vol_idx]
        
        optimal_params[f'horizon_{target_horizon}_parameters']['volatility'] = {
            'method': 'ewma',
            'lambda': float(best_vol_result['lambda']),
            'min_periods': int(best_vol_result['min_periods']),
            'validation_score': float(best_vol_result['volatility_mse']),
            'sample_size': int(best_vol_result['sample_size'])
        }
        
        # Find best correlation parameters (lowest Frobenius norm)
        best_corr_idx = results_df['correlation_frobenius'].idxmin()
        best_corr_result = results_df.loc[best_corr_idx]
        
        optimal_params[f'horizon_{target_horizon}_parameters']['correlation'] = {
            'method': 'ewma',
            'lambda': float(best_corr_result['lambda']),
            'min_periods': int(best_corr_result['min_periods']),
            'validation_score': float(best_corr_result['correlation_frobenius']),
            'sample_size': int(best_corr_result['sample_size'])
        }
        
        # Add summary statistics
        optimal_params['validation_summary'] = {
            'volatility_mse_range': [float(results_df['volatility_mse'].min()), float(results_df['volatility_mse'].max())],
            'correlation_frobenius_range': [float(results_df['correlation_frobenius'].min()), float(results_df['correlation_frobenius'].max())],
            'lambda_range_tested': [float(results_df['lambda'].min()), float(results_df['lambda'].max())],
            'min_periods_range_tested': [int(results_df['min_periods'].min()), int(results_df['min_periods'].max())]
        }
        
        logger.info(f"Found optimal parameters for {target_horizon}-day horizon:")
        logger.info(f"  Volatility: λ={optimal_params[f'horizon_{target_horizon}_parameters']['volatility']['lambda']:.3f}, "
                   f"min_periods={optimal_params[f'horizon_{target_horizon}_parameters']['volatility']['min_periods']}")
        logger.info(f"  Correlation: λ={optimal_params[f'horizon_{target_horizon}_parameters']['correlation']['lambda']:.3f}, "
                   f"min_periods={optimal_params[f'horizon_{target_horizon}_parameters']['correlation']['min_periods']}")
        
        return optimal_params
    
    def _analyze_optimization_results(self) -> Dict[str, Any]:
        """Analyze optimization results and determine optimal parameters."""
        
        if not self.validation_results:
            raise ValueError("No validation results available")
        
        results_df = pd.DataFrame([
            {
                'frequency': r.frequency.value,
                'lambda': r.lambda_param,
                'min_periods': r.min_periods,
                'horizon': r.horizon,
                'combined_score': r.combined_score,
                'vol_mse': r.volatility_mse,
                'vol_hit_rate': r.volatility_hit_rate,
                'sample_size': r.sample_size
            }
            for r in self.validation_results
        ])
        
        # Find optimal parameters by horizon
        optimal_by_horizon = {}
        
        for horizon in self.config.forecast_horizons:
            horizon_results = results_df[results_df['horizon'] == horizon]
            # Filter out NaN values
            horizon_results = horizon_results.dropna(subset=['combined_score'])
            
            if len(horizon_results) > 0:
                # Find best combination (lowest combined score)
                best_idx = horizon_results['combined_score'].idxmin()
                
                # Check if the index is valid (not NaN)
                if pd.notna(best_idx):
                    best_result = horizon_results.loc[best_idx]
                else:
                    continue
                
                optimal_by_horizon[f'{horizon}_period'] = {
                    'frequency': best_result['frequency'],
                    'lambda': best_result['lambda'],
                    'min_periods': int(best_result['min_periods']),
                    'combined_score': best_result['combined_score'],
                    'volatility_mse': best_result['vol_mse'],
                    'hit_rate': best_result['vol_hit_rate']
                }
        
        # Find overall optimal parameters (lowest average score across horizons)
        # Filter out NaN values first
        valid_results = results_df.dropna(subset=['combined_score'])
        
        if len(valid_results) == 0:
            raise ValueError("No valid optimization results found")
        
        param_scores = valid_results.groupby(['frequency', 'lambda', 'min_periods'])['combined_score'].mean()
        overall_best = param_scores.idxmin()
        
        overall_optimal = {
            'frequency': overall_best[0],
            'lambda': overall_best[1],
            'min_periods': int(overall_best[2]),
            'average_score': param_scores[overall_best]
        }
        
        # Frequency analysis (use valid results)
        freq_analysis = valid_results.groupby('frequency').agg({
            'combined_score': ['mean', 'std', 'count'],
            'vol_hit_rate': 'mean'
        }).round(4)
        
        # Lambda analysis (use valid results)
        lambda_analysis = valid_results.groupby('lambda').agg({
            'combined_score': ['mean', 'std'],
            'vol_hit_rate': 'mean'
        }).round(4)
        
        return {
            'overall_optimal': overall_optimal,
            'optimal_by_horizon': optimal_by_horizon,
            'frequency_analysis': freq_analysis.to_dict(),
            'lambda_analysis': lambda_analysis.to_dict(),
            'total_combinations_tested': len(self.validation_results),
            'summary_statistics': {
                'best_frequency': valid_results.groupby('frequency')['combined_score'].mean().idxmin(),
                'best_lambda': valid_results.groupby('lambda')['combined_score'].mean().idxmin(),
                'best_min_periods': valid_results.groupby('min_periods')['combined_score'].mean().idxmin(),
                'mean_hit_rate': valid_results['vol_hit_rate'].mean(),
                'mean_score': valid_results['combined_score'].mean()
            }
        }
    
    def get_optimal_parameters_for_horizon(self, horizon: int) -> Dict[str, Any]:
        """Get optimal parameters for a specific forecasting horizon."""
        key = f'{horizon}_period'
        if key in self.optimal_parameters.get('optimal_by_horizon', {}):
            return self.optimal_parameters['optimal_by_horizon'][key]
        return self.optimal_parameters.get('overall_optimal', {})
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of all validation results."""
        if not self.validation_results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.validation_results:
            summary_data.append({
                'frequency': result.frequency.value,
                'lambda': result.lambda_param,
                'min_periods': result.min_periods,
                'horizon': result.horizon,
                'combined_score': result.combined_score,
                'volatility_mse': result.volatility_mse,
                'volatility_hit_rate': result.volatility_hit_rate,
                'sample_size': result.sample_size
            })
        
        return pd.DataFrame(summary_data)


def run_exposure_universe_optimization(
    exposure_universe_path: str,
    start_date: datetime,
    end_date: datetime,
    exposure_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run comprehensive parameter optimization on exposure universe.
    
    Args:
        exposure_universe_path: Path to exposure universe YAML
        start_date: Start date for analysis
        end_date: End date for analysis
        exposure_ids: Specific exposures to test
        output_path: Path to save results
        
    Returns:
        Optimization results dictionary
    """
    # Load exposure universe
    universe = ExposureUniverse.from_yaml(exposure_universe_path)
    
    # Create optimizer
    optimizer = ParameterOptimizer(universe)
    
    # Run optimization
    results = optimizer.optimize_all_parameters(start_date, end_date, exposure_ids)
    
    # Save results if requested
    if output_path:
        import json
        with open(output_path, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict()
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
    
    return results