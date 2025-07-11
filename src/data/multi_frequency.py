"""Multi-Frequency Data Support Module.

This module provides comprehensive support for handling financial data at different
frequencies (daily, weekly, monthly, quarterly) with proper return compounding,
frequency-aware aggregation, and robust data alignment.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from .interfaces import Frequency  # Import canonical Frequency enum

logger = logging.getLogger(__name__)


class ReturnCompounding:
    """Utilities for proper return compounding across frequencies."""
    
    @staticmethod
    def compound_returns(returns: pd.Series, periods: int = 1) -> pd.Series:
        """Compound returns over specified periods.
        
        Args:
            returns: Series of periodic returns
            periods: Number of periods to compound over
            
        Returns:
            Series of compounded returns
        """
        if periods <= 1:
            return returns
        
        # Use rolling window to compound returns
        return (1 + returns).rolling(window=periods, min_periods=periods).apply(
            lambda x: x.prod() - 1, raw=True
        )
    
    @staticmethod
    def resample_returns(
        returns: pd.Series, 
        target_freq: Frequency,
        source_freq: Optional[Frequency] = None
    ) -> pd.Series:
        """Resample returns to target frequency with proper compounding.
        
        Args:
            returns: Source return series
            target_freq: Target frequency
            source_freq: Source frequency (auto-detected if None)
            
        Returns:
            Resampled return series
        """
        if source_freq is None:
            source_freq = ReturnCompounding._detect_frequency(returns)
        
        if source_freq == target_freq:
            return returns
        
        # Convert to price series for proper resampling
        prices = (1 + returns).cumprod()
        
        # Resample prices
        resampled_prices = prices.resample(target_freq.pandas_freq).last()
        
        # Convert back to returns
        resampled_returns = resampled_prices.pct_change().dropna()
        
        return resampled_returns
    
    @staticmethod
    def _detect_frequency(data: pd.Series) -> Frequency:
        """Auto-detect the frequency of a time series."""
        if len(data) < 2:
            return Frequency.DAILY
        
        # Calculate median time difference
        time_diffs = pd.Series(data.index).diff().dropna()
        median_diff = time_diffs.median()
        
        # Map to frequencies
        if median_diff <= timedelta(days=1):
            return Frequency.DAILY
        elif median_diff <= timedelta(days=7):
            return Frequency.WEEKLY
        elif median_diff <= timedelta(days=32):
            return Frequency.MONTHLY
        elif median_diff <= timedelta(days=95):
            return Frequency.QUARTERLY
        else:
            return Frequency.ANNUAL
    
    @staticmethod
    def align_frequencies(
        data_dict: Dict[str, pd.Series],
        target_freq: Frequency
    ) -> Dict[str, pd.Series]:
        """Align multiple return series to the same frequency.
        
        Args:
            data_dict: Dictionary of return series
            target_freq: Target frequency for alignment
            
        Returns:
            Dictionary of aligned return series
        """
        aligned_data = {}
        
        for name, series in data_dict.items():
            try:
                aligned_data[name] = ReturnCompounding.resample_returns(
                    series, target_freq
                )
            except Exception as e:
                logger.warning(f"Failed to align {name} to {target_freq.value}: {e}")
                continue
        
        return aligned_data


class MultiFrequencyDataFetcher:
    """Enhanced data fetcher with multi-frequency support."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the multi-frequency data fetcher.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = cache_dir
        self._cache = {}
    
    def fetch_returns(
        self,
        tickers: Union[str, List[str]],
        start_date: datetime,
        end_date: datetime,
        frequency: Frequency = Frequency.DAILY,
        validate: bool = True
    ) -> Union[pd.Series, pd.DataFrame]:
        """Fetch returns for tickers at specified frequency.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date
            end_date: End date
            frequency: Target frequency
            validate: Whether to validate data quality
            
        Returns:
            Returns series (single ticker) or DataFrame (multiple tickers)
        """
        if isinstance(tickers, str):
            return self._fetch_single_ticker_returns(
                tickers, start_date, end_date, frequency, validate
            )
        else:
            return self._fetch_multiple_tickers_returns(
                tickers, start_date, end_date, frequency, validate
            )
    
    def _fetch_single_ticker_returns(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        frequency: Frequency,
        validate: bool
    ) -> pd.Series:
        """Fetch returns for a single ticker."""
        # Create cache key
        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}_{frequency.value}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            import yfinance as yf
            
            # Fetch data with appropriate interval
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=frequency.yfinance_interval,
                auto_adjust=True  # Use adjusted close for total returns
            )
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Calculate returns from adjusted close
            prices = data['Close']
            returns = prices.pct_change().dropna()
            
            # Validate data quality if requested
            if validate:
                returns = self._validate_returns(returns, ticker)
            
            # Cache and return
            self._cache[cache_key] = returns
            return returns
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            # Return empty series with proper index
            return pd.Series([], dtype=float, name=ticker)
    
    def _fetch_multiple_tickers_returns(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: Frequency,
        validate: bool
    ) -> pd.DataFrame:
        """Fetch returns for multiple tickers."""
        returns_dict = {}
        
        for ticker in tickers:
            returns = self._fetch_single_ticker_returns(
                ticker, start_date, end_date, frequency, validate
            )
            if not returns.empty:
                returns_dict[ticker] = returns
        
        if not returns_dict:
            return pd.DataFrame()
        
        # Align all series to common index
        returns_df = pd.DataFrame(returns_dict)
        
        # Forward fill small gaps (up to 5 periods)
        returns_df = returns_df.fillna(method='ffill', limit=5)
        
        return returns_df
    
    def _validate_returns(self, returns: pd.Series, ticker: str) -> pd.Series:
        """Validate and clean return data."""
        # Remove extreme outliers (beyond 5 standard deviations)
        std_threshold = 5
        mean_return = returns.mean()
        std_return = returns.std()
        
        outlier_mask = np.abs(returns - mean_return) > (std_threshold * std_return)
        
        if outlier_mask.sum() > 0:
            logger.warning(f"Removing {outlier_mask.sum()} outliers from {ticker}")
            returns = returns[~outlier_mask]
        
        # Remove returns that are exactly zero (likely data errors)
        zero_mask = returns == 0
        if zero_mask.sum() > len(returns) * 0.1:  # More than 10% zeros
            logger.warning(f"High number of zero returns in {ticker}: {zero_mask.sum()}")
        
        return returns
    
    def get_frequency_statistics(
        self,
        returns: pd.Series,
        frequency: Frequency
    ) -> Dict[str, float]:
        """Calculate frequency-specific statistics for returns.
        
        Args:
            returns: Return series
            frequency: Data frequency
            
        Returns:
            Dictionary of statistics
        """
        if returns.empty:
            return {}
        
        # Basic statistics
        stats = {
            'mean_return': returns.mean(),
            'volatility': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'periods': len(returns)
        }
        
        # Annualized statistics
        annualization_factor = frequency.annualization_factor
        annualized_return = stats['mean_return'] * annualization_factor
        annualized_volatility = stats['volatility'] * np.sqrt(annualization_factor)
        
        stats.update({
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': (annualized_return - 0.02) / annualized_volatility if annualized_volatility > 0 else 0
        })
        
        # Frequency-specific metrics
        if frequency in [Frequency.DAILY, Frequency.WEEKLY]:
            # Calculate maximum drawdown for high-frequency data
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            stats['max_drawdown'] = drawdown.min()
        
        return stats


class FrequencyConverter:
    """Utilities for converting data between frequencies."""
    
    @staticmethod
    def convert_covariance_matrix(
        cov_matrix: np.ndarray,
        source_freq: Frequency,
        target_freq: Frequency
    ) -> np.ndarray:
        """Convert covariance matrix between frequencies.
        
        Args:
            cov_matrix: Source covariance matrix
            source_freq: Source frequency
            target_freq: Target frequency
            
        Returns:
            Converted covariance matrix
        """
        if source_freq == target_freq:
            return cov_matrix
        
        # Calculate conversion factor for variance scaling
        # For variance: multiply by (periods_per_year_target / periods_per_year_source)
        # For daily to annual: multiply by (1 / 252) * 252 = 1 * 252 = 252
        source_periods_per_year = source_freq.annualization_factor
        target_periods_per_year = target_freq.annualization_factor
        
        # Conversion factor for variance scaling
        conversion_factor = source_periods_per_year / target_periods_per_year
        
        return cov_matrix * conversion_factor
    
    @staticmethod
    def convert_volatility(
        volatility: float,
        source_freq: Frequency,
        target_freq: Frequency
    ) -> float:
        """Convert volatility between frequencies.
        
        Args:
            volatility: Source volatility
            source_freq: Source frequency
            target_freq: Target frequency
            
        Returns:
            Converted volatility
        """
        if source_freq == target_freq:
            return volatility
        
        # Convert to annual volatility first
        annual_vol = volatility * np.sqrt(source_freq.annualization_factor)
        
        # Convert to target frequency
        target_vol = annual_vol / np.sqrt(target_freq.annualization_factor)
        
        return target_vol
    
    @staticmethod
    def get_optimal_frequency(
        data_length: int,
        analysis_horizon: timedelta
    ) -> Frequency:
        """Suggest optimal frequency based on data length and analysis horizon.
        
        Args:
            data_length: Number of data points available
            analysis_horizon: Desired analysis horizon
            
        Returns:
            Suggested frequency
        """
        # For short horizons, use higher frequency
        if analysis_horizon <= timedelta(days=30):
            return Frequency.DAILY if data_length >= 100 else Frequency.WEEKLY
        elif analysis_horizon <= timedelta(days=90):
            return Frequency.WEEKLY if data_length >= 50 else Frequency.MONTHLY
        elif analysis_horizon <= timedelta(days=365):
            return Frequency.MONTHLY if data_length >= 24 else Frequency.QUARTERLY
        else:
            return Frequency.QUARTERLY if data_length >= 20 else Frequency.ANNUAL


class MultiFrequencyAnalyzer:
    """Analyzer for multi-frequency portfolio analysis."""
    
    def __init__(self, data_fetcher: Optional[MultiFrequencyDataFetcher] = None):
        """Initialize the analyzer.
        
        Args:
            data_fetcher: Data fetcher instance
        """
        self.data_fetcher = data_fetcher or MultiFrequencyDataFetcher()
    
    def analyze_frequency_impact(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        frequencies: List[Frequency] = None
    ) -> pd.DataFrame:
        """Analyze how different frequencies affect risk/return metrics.
        
        Args:
            tickers: List of tickers to analyze
            start_date: Analysis start date
            end_date: Analysis end date
            frequencies: List of frequencies to test
            
        Returns:
            DataFrame with results by frequency and ticker
        """
        if frequencies is None:
            frequencies = [Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY]
        
        results = []
        
        for freq in frequencies:
            try:
                returns_df = self.data_fetcher.fetch_returns(
                    tickers, start_date, end_date, freq
                )
                
                for ticker in tickers:
                    if ticker in returns_df.columns:
                        returns = returns_df[ticker].dropna()
                        if len(returns) > 0:
                            stats = self.data_fetcher.get_frequency_statistics(returns, freq)
                            result = {
                                'ticker': ticker,
                                'frequency': freq.value,
                                **stats
                            }
                            results.append(result)
            
            except Exception as e:
                logger.warning(f"Failed to analyze frequency {freq.value}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def find_optimal_rebalancing_frequency(
        self,
        portfolio_returns: pd.Series,
        frequencies: List[Frequency] = None
    ) -> Tuple[Frequency, Dict[str, float]]:
        """Find optimal rebalancing frequency based on transaction costs and returns.
        
        Args:
            portfolio_returns: Daily portfolio returns
            frequencies: Frequencies to test
            
        Returns:
            Tuple of (optimal_frequency, metrics_dict)
        """
        if frequencies is None:
            frequencies = [Frequency.WEEKLY, Frequency.MONTHLY, Frequency.QUARTERLY]
        
        results = {}
        
        for freq in frequencies:
            # Resample returns to frequency
            rebalanced_returns = ReturnCompounding.resample_returns(portfolio_returns, freq)
            
            if len(rebalanced_returns) > 0:
                # Calculate metrics
                annual_return = rebalanced_returns.mean() * freq.annualization_factor
                annual_vol = rebalanced_returns.std() * np.sqrt(freq.annualization_factor)
                sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
                
                # Estimate transaction costs (simplified)
                rebalancing_periods = len(rebalanced_returns)
                estimated_turnover = 0.20  # 20% turnover per rebalance
                transaction_cost = 0.001   # 10 bps per transaction
                total_costs = rebalancing_periods * estimated_turnover * transaction_cost
                
                # Net return after costs
                net_annual_return = annual_return - total_costs
                net_sharpe = (net_annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
                
                results[freq] = {
                    'gross_return': annual_return,
                    'net_return': net_annual_return,
                    'volatility': annual_vol,
                    'gross_sharpe': sharpe,
                    'net_sharpe': net_sharpe,
                    'rebalances': rebalancing_periods,
                    'total_costs': total_costs
                }
        
        # Find frequency with highest net Sharpe ratio
        if results:
            optimal_freq = max(results.keys(), key=lambda f: results[f]['net_sharpe'])
            return optimal_freq, results[optimal_freq]
        else:
            return Frequency.MONTHLY, {}


def calculate_returns(
    prices: pd.Series,
    method: str = 'simple',
    periods: int = 1
) -> pd.Series:
    """Calculate returns from price series with various methods.
    
    Args:
        prices: Price series
        method: Return calculation method ('simple', 'log', 'compound')
        periods: Number of periods for return calculation
        
    Returns:
        Return series
    """
    if method == 'simple':
        returns = prices.pct_change(periods=periods)
    elif method == 'log':
        returns = np.log(prices / prices.shift(periods))
    elif method == 'compound':
        returns = (prices / prices.shift(periods)) - 1
    else:
        raise ValueError(f"Unknown return method: {method}")
    
    return returns.dropna()