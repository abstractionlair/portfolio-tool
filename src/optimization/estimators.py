"""Return and risk estimation utilities."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

try:
    from data import MarketDataFetcher, calculate_returns
except ImportError:
    try:
        from ..data import MarketDataFetcher, calculate_returns
    except ImportError:
        # Define stubs for standalone usage
        class MarketDataFetcher:
            pass
        def calculate_returns(*args, **kwargs):
            return None

try:
    from .ewma import EWMAEstimator, EWMAParameters, GARCHEstimator
except ImportError:
    try:
        from ewma import EWMAEstimator, EWMAParameters, GARCHEstimator
    except ImportError:
        # Define stubs for standalone usage
        class EWMAEstimator:
            pass
        class EWMAParameters:
            pass
        class GARCHEstimator:
            pass

logger = logging.getLogger(__name__)


@dataclass
class MarketView:
    """Represents a view on asset returns for Black-Litterman."""
    assets: List[str]
    view_type: str  # 'absolute' or 'relative'
    expected_return: float  # Expected return or expected outperformance
    confidence: float  # Confidence level (0-1)
    description: str = ""


class ReturnEstimator:
    """Estimates expected returns and covariance matrices."""
    
    def __init__(self, market_data: MarketDataFetcher, ewma_params: Optional[EWMAParameters] = None):
        """Initialize with market data source.
        
        Args:
            market_data: MarketDataFetcher instance for historical data
            ewma_params: EWMA parameters for exponential weighting. If None, uses defaults.
        """
        self.market_data = market_data
        self._cache = {}  # Cache for historical data
        self.ewma_estimator = EWMAEstimator(ewma_params)
        self.garch_estimator = GARCHEstimator()
        
        logger.debug("Initialized ReturnEstimator with EWMA support")
    
    def estimate_expected_returns(
        self,
        symbols: List[str],
        method: str = 'historical',
        lookback_years: int = 5,
        frequency: str = 'daily'
    ) -> np.ndarray:
        """Estimate expected returns using various methods.
        
        Args:
            symbols: List of symbols to estimate returns for
            method: Estimation method ('historical', 'capm', 'shrinkage')
            lookback_years: Years of historical data to use
            frequency: Data frequency ('daily', 'monthly')
            
        Returns:
            Array of expected annual returns
        """
        if method == 'historical':
            return self._estimate_historical_returns(symbols, lookback_years, frequency)
        elif method == 'capm':
            return self._estimate_capm_returns(symbols, lookback_years, frequency)
        elif method == 'shrinkage':
            return self._estimate_shrinkage_returns(symbols, lookback_years, frequency)
        else:
            raise ValueError(f"Unsupported estimation method: {method}")
    
    def estimate_covariance_matrix(
        self,
        symbols: List[str],
        method: str = 'sample',
        lookback_years: int = 5,
        frequency: str = 'daily',
        shrinkage_target: str = 'diagonal'
    ) -> np.ndarray:
        """Estimate covariance matrix with shrinkage options.
        
        Args:
            symbols: List of symbols
            method: Estimation method ('sample', 'shrinkage', 'exponential', 'ewma', 'garch')
            lookback_years: Years of historical data
            frequency: Data frequency
            shrinkage_target: Shrinkage target for covariance ('diagonal', 'identity')
            
        Returns:
            Covariance matrix (annualized)
        """
        if method == 'sample':
            return self._estimate_sample_covariance(symbols, lookback_years, frequency)
        elif method == 'shrinkage':
            return self._estimate_shrinkage_covariance(symbols, lookback_years, frequency, shrinkage_target)
        elif method == 'exponential':
            return self._estimate_exponential_covariance(symbols, lookback_years, frequency)
        elif method == 'ewma':
            return self._estimate_ewma_covariance(symbols, lookback_years, frequency)
        elif method == 'garch':
            return self._estimate_garch_covariance(symbols, lookback_years, frequency)
        else:
            raise ValueError(f"Unsupported covariance estimation method: {method}")
    
    def _get_return_data(self, symbols: List[str], lookback_years: int, frequency: str) -> pd.DataFrame:
        """Get historical return data with caching."""
        cache_key = f"{'-'.join(sorted(symbols))}_{lookback_years}y_{frequency}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)
        
        try:
            # Fetch price data
            price_data = self.market_data.fetch_price_history(symbols, start_date, end_date)
            
            # Calculate returns
            returns_data = {}
            for symbol, prices in price_data.items():
                if not prices.empty:
                    returns = calculate_returns(prices, period='daily')
                    if not returns.empty:
                        returns_data[symbol] = returns
            
            if not returns_data:
                raise ValueError("No valid return data available")
            
            # Align return series
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if frequency == 'monthly':
                # Convert to monthly returns
                monthly_returns = {}
                for symbol in returns_df.columns:
                    daily_rets = returns_df[symbol]
                    # Compound daily returns to monthly
                    monthly_values = (1 + daily_rets).resample('ME').prod() - 1
                    monthly_returns[symbol] = monthly_values
                
                returns_df = pd.DataFrame(monthly_returns).dropna()
            
            self._cache[cache_key] = returns_df
            return returns_df
            
        except Exception as e:
            logger.error(f"Error fetching return data: {e}")
            # Return dummy data for testing
            dates = pd.date_range(start_date, end_date, freq='D' if frequency == 'daily' else 'ME')
            dummy_data = {}
            for symbol in symbols:
                # Generate random returns with some correlation structure
                np.random.seed(hash(symbol) % 2**32)  # Deterministic per symbol
                dummy_data[symbol] = np.random.normal(0.0005, 0.015, len(dates))
            
            returns_df = pd.DataFrame(dummy_data, index=dates)
            self._cache[cache_key] = returns_df
            return returns_df
    
    def _estimate_historical_returns(self, symbols: List[str], lookback_years: int, frequency: str) -> np.ndarray:
        """Estimate returns using historical mean."""
        returns_df = self._get_return_data(symbols, lookback_years, frequency)
        
        # Calculate mean returns
        mean_returns = returns_df.mean()
        
        # Annualize based on frequency
        if frequency == 'daily':
            annual_returns = mean_returns * 252
        elif frequency == 'monthly':
            annual_returns = mean_returns * 12
        else:
            annual_returns = mean_returns
        
        # Return as array in symbol order
        return np.array([annual_returns.get(symbol, 0.0) for symbol in symbols])
    
    def _estimate_capm_returns(self, symbols: List[str], lookback_years: int, frequency: str) -> np.ndarray:
        """Estimate returns using CAPM model."""
        returns_df = self._get_return_data(symbols + ['SPY'], lookback_years, frequency)
        
        # Use SPY as market proxy
        if 'SPY' not in returns_df.columns:
            logger.warning("SPY not available for CAPM estimation, falling back to historical")
            return self._estimate_historical_returns(symbols, lookback_years, frequency)
        
        market_returns = returns_df['SPY']
        risk_free_rate = 0.02 / (252 if frequency == 'daily' else 12)  # Assume 2% annual risk-free rate
        
        capm_returns = []
        
        for symbol in symbols:
            if symbol in returns_df.columns and symbol != 'SPY':
                asset_returns = returns_df[symbol]
                
                # Calculate beta
                excess_market = market_returns - risk_free_rate
                excess_asset = asset_returns - risk_free_rate
                
                # Simple linear regression for beta
                covariance = np.cov(excess_asset, excess_market)[0, 1]
                market_variance = np.var(excess_market)
                
                if market_variance > 0:
                    beta = covariance / market_variance
                else:
                    beta = 1.0
                
                # CAPM expected return
                market_premium = market_returns.mean() - risk_free_rate
                expected_return = risk_free_rate + beta * market_premium
                
                # Annualize
                if frequency == 'daily':
                    expected_return *= 252
                elif frequency == 'monthly':
                    expected_return *= 12
                
                capm_returns.append(expected_return)
            else:
                # Fallback to historical mean
                if symbol in returns_df.columns:
                    hist_return = returns_df[symbol].mean()
                    if frequency == 'daily':
                        hist_return *= 252
                    elif frequency == 'monthly':
                        hist_return *= 12
                    capm_returns.append(hist_return)
                else:
                    capm_returns.append(0.0)
        
        return np.array(capm_returns)
    
    def _estimate_shrinkage_returns(self, symbols: List[str], lookback_years: int, frequency: str) -> np.ndarray:
        """Estimate returns with shrinkage toward grand mean."""
        historical_returns = self._estimate_historical_returns(symbols, lookback_years, frequency)
        
        # Shrink toward grand mean
        grand_mean = np.mean(historical_returns)
        shrinkage_intensity = 0.2  # 20% shrinkage
        
        shrunk_returns = (1 - shrinkage_intensity) * historical_returns + shrinkage_intensity * grand_mean
        
        return shrunk_returns
    
    def _estimate_sample_covariance(self, symbols: List[str], lookback_years: int, frequency: str) -> np.ndarray:
        """Estimate sample covariance matrix."""
        returns_df = self._get_return_data(symbols, lookback_years, frequency)
        
        # Select only the symbols we need
        available_symbols = [s for s in symbols if s in returns_df.columns]
        
        if not available_symbols:
            # Return identity matrix as fallback
            return np.eye(len(symbols)) * 0.04  # 20% annual volatility
        
        # Calculate sample covariance
        cov_matrix = returns_df[available_symbols].cov().values
        
        # Annualize
        if frequency == 'daily':
            cov_matrix *= 252
        elif frequency == 'monthly':
            cov_matrix *= 12
        
        # If we're missing some symbols, pad with identity
        if len(available_symbols) < len(symbols):
            full_cov = np.eye(len(symbols)) * 0.04  # Default variance
            
            # Map available symbols to their positions
            symbol_map = {symbol: i for i, symbol in enumerate(symbols)}
            available_indices = [symbol_map[s] for s in available_symbols]
            
            # Fill in the available covariances
            for i, idx_i in enumerate(available_indices):
                for j, idx_j in enumerate(available_indices):
                    full_cov[idx_i, idx_j] = cov_matrix[i, j]
            
            return full_cov
        
        return cov_matrix
    
    def _estimate_shrinkage_covariance(
        self,
        symbols: List[str],
        lookback_years: int,
        frequency: str,
        shrinkage_target: str
    ) -> np.ndarray:
        """Estimate covariance matrix with shrinkage."""
        sample_cov = self._estimate_sample_covariance(symbols, lookback_years, frequency)
        
        # Define shrinkage target
        if shrinkage_target == 'diagonal':
            # Shrink toward diagonal matrix (remove correlations)
            target = np.diag(np.diag(sample_cov))
        elif shrinkage_target == 'identity':
            # Shrink toward identity matrix
            avg_variance = np.mean(np.diag(sample_cov))
            target = np.eye(len(symbols)) * avg_variance
        else:
            raise ValueError(f"Unsupported shrinkage target: {shrinkage_target}")
        
        # Apply shrinkage (Ledoit-Wolf type)
        shrinkage_intensity = 0.1  # 10% shrinkage
        shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target
        
        return shrunk_cov
    
    def _estimate_exponential_covariance(self, symbols: List[str], lookback_years: int, frequency: str) -> np.ndarray:
        """Estimate covariance using exponentially weighted returns."""
        returns_df = self._get_return_data(symbols, lookback_years, frequency)
        
        # Select available symbols
        available_symbols = [s for s in symbols if s in returns_df.columns]
        
        if not available_symbols:
            return np.eye(len(symbols)) * 0.04
        
        returns_matrix = returns_df[available_symbols].values
        T = len(returns_matrix)
        
        # Exponential decay parameter
        decay = 0.94  # Daily decay factor
        
        # Calculate exponentially weighted covariance
        weights = np.array([decay ** i for i in range(T)])[::-1]
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted means
        weighted_means = np.average(returns_matrix, axis=0, weights=weights)
        
        # Weighted covariance
        centered_returns = returns_matrix - weighted_means
        weighted_cov = np.zeros((len(available_symbols), len(available_symbols)))
        
        for i in range(T):
            outer_product = np.outer(centered_returns[i], centered_returns[i])
            weighted_cov += weights[i] * outer_product
        
        # Annualize
        if frequency == 'daily':
            weighted_cov *= 252
        elif frequency == 'monthly':
            weighted_cov *= 12
        
        # Handle missing symbols
        if len(available_symbols) < len(symbols):
            full_cov = np.eye(len(symbols)) * 0.04
            symbol_map = {symbol: i for i, symbol in enumerate(symbols)}
            available_indices = [symbol_map[s] for s in available_symbols]
            
            for i, idx_i in enumerate(available_indices):
                for j, idx_j in enumerate(available_indices):
                    full_cov[idx_i, idx_j] = weighted_cov[i, j]
            
            return full_cov
        
        return weighted_cov
    
    def _estimate_ewma_covariance(self, symbols: List[str], lookback_years: int, frequency: str) -> np.ndarray:
        """Estimate covariance using EWMA estimator."""
        returns_df = self._get_return_data(symbols, lookback_years, frequency)
        
        # Select available symbols
        available_symbols = [s for s in symbols if s in returns_df.columns]
        
        if not available_symbols:
            return np.eye(len(symbols)) * 0.04
        
        try:
            # Use EWMA estimator
            ewma_cov = self.ewma_estimator.estimate_covariance_matrix(
                returns_df[available_symbols], 
                annualize=True, 
                frequency=frequency
            )
            
            # Convert to numpy array
            if isinstance(ewma_cov, pd.DataFrame):
                cov_matrix = ewma_cov.values
            else:
                cov_matrix = ewma_cov
            
            # Handle missing symbols
            if len(available_symbols) < len(symbols):
                full_cov = np.eye(len(symbols)) * 0.04
                symbol_map = {symbol: i for i, symbol in enumerate(symbols)}
                available_indices = [symbol_map[s] for s in available_symbols]
                
                for i, idx_i in enumerate(available_indices):
                    for j, idx_j in enumerate(available_indices):
                        full_cov[idx_i, idx_j] = cov_matrix[i, j]
                
                return full_cov
            
            return cov_matrix
            
        except Exception as e:
            logger.warning(f"EWMA covariance estimation failed: {e}")
            # Fallback to sample covariance
            return self._estimate_sample_covariance(symbols, lookback_years, frequency)
    
    def _estimate_garch_covariance(self, symbols: List[str], lookback_years: int, frequency: str) -> np.ndarray:
        """Estimate covariance using GARCH models for diagonal and sample correlation."""
        returns_df = self._get_return_data(symbols, lookback_years, frequency)
        
        # Select available symbols
        available_symbols = [s for s in symbols if s in returns_df.columns]
        
        if not available_symbols:
            return np.eye(len(symbols)) * 0.04
        
        try:
            n_assets = len(available_symbols)
            cov_matrix = np.zeros((n_assets, n_assets))
            
            # Estimate GARCH variance for each asset
            variances = []
            for symbol in available_symbols:
                returns_series = returns_df[symbol].dropna()
                if len(returns_series) >= 30:
                    garch_var = self.garch_estimator.estimate_variance(
                        returns_series, annualize=True, frequency=frequency
                    ).iloc[-1]  # Get latest variance estimate
                    variances.append(garch_var)
                else:
                    # Fallback to sample variance
                    sample_var = returns_series.var()
                    if frequency == 'daily':
                        sample_var *= 252
                    elif frequency == 'monthly':
                        sample_var *= 12
                    variances.append(sample_var)
            
            variances = np.array(variances)
            
            # Estimate correlation matrix using sample correlation
            correlation_matrix = returns_df[available_symbols].corr().values
            
            # Combine GARCH variances with sample correlations
            std_devs = np.sqrt(variances)
            cov_matrix = correlation_matrix * np.outer(std_devs, std_devs)
            
            # Handle missing symbols
            if len(available_symbols) < len(symbols):
                full_cov = np.eye(len(symbols)) * 0.04
                symbol_map = {symbol: i for i, symbol in enumerate(symbols)}
                available_indices = [symbol_map[s] for s in available_symbols]
                
                for i, idx_i in enumerate(available_indices):
                    for j, idx_j in enumerate(available_indices):
                        full_cov[idx_i, idx_j] = cov_matrix[i, j]
                
                return full_cov
            
            return cov_matrix
            
        except Exception as e:
            logger.warning(f"GARCH covariance estimation failed: {e}")
            # Fallback to sample covariance
            return self._estimate_sample_covariance(symbols, lookback_years, frequency)
    
    def estimate_rolling_volatility(
        self,
        symbols: List[str],
        window: int = 252,
        method: str = 'ewma',
        lookback_years: int = 5,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """Estimate rolling volatility using various methods.
        
        Args:
            symbols: List of symbols
            window: Rolling window size
            method: Estimation method ('sample', 'ewma', 'garch')
            lookback_years: Years of historical data
            frequency: Data frequency
            
        Returns:
            DataFrame of rolling volatility estimates
        """
        returns_df = self._get_return_data(symbols, lookback_years, frequency)
        available_symbols = [s for s in symbols if s in returns_df.columns]
        
        if not available_symbols:
            return pd.DataFrame()
        
        if method == 'ewma':
            # Use EWMA estimator
            volatility_results = self.ewma_estimator.estimate_volatility(
                returns_df[available_symbols], 
                annualize=True, 
                frequency=frequency
            )
            return volatility_results
            
        elif method == 'garch':
            # Use GARCH for each asset
            volatility_data = {}
            for symbol in available_symbols:
                try:
                    garch_var = self.garch_estimator.estimate_variance(
                        returns_df[symbol], annualize=True, frequency=frequency
                    )
                    volatility_data[symbol] = np.sqrt(garch_var)
                except:
                    # Fallback to rolling sample volatility
                    rolling_vol = returns_df[symbol].rolling(window).std()
                    if frequency == 'daily':
                        rolling_vol *= np.sqrt(252)
                    elif frequency == 'monthly':
                        rolling_vol *= np.sqrt(12)
                    volatility_data[symbol] = rolling_vol
            
            return pd.DataFrame(volatility_data, index=returns_df.index)
            
        else:  # sample method
            # Rolling sample volatility
            rolling_vol = returns_df[available_symbols].rolling(window).std()
            if frequency == 'daily':
                rolling_vol *= np.sqrt(252)
            elif frequency == 'monthly':
                rolling_vol *= np.sqrt(12)
            
            return rolling_vol
    
    def forecast_volatility(
        self,
        symbols: List[str],
        horizon: int = 22,  # ~1 month for daily data
        method: str = 'ewma',
        lookback_years: int = 2,
        frequency: str = 'daily'
    ) -> Dict[str, float]:
        """Forecast volatility for multiple assets.
        
        Args:
            symbols: List of symbols
            horizon: Forecast horizon in periods
            method: Forecasting method ('ewma', 'garch')
            lookback_years: Years of historical data
            frequency: Data frequency
            
        Returns:
            Dictionary of volatility forecasts by symbol
        """
        returns_df = self._get_return_data(symbols, lookback_years, frequency)
        available_symbols = [s for s in symbols if s in returns_df.columns]
        
        forecasts = {}
        
        for symbol in available_symbols:
            returns_series = returns_df[symbol].dropna()
            
            try:
                if method == 'ewma':
                    forecast = self.ewma_estimator.forecast_volatility(
                        returns_series, 
                        horizon=horizon, 
                        method='simple',
                        annualize=True, 
                        frequency=frequency
                    )
                    forecasts[symbol] = forecast
                    
                elif method == 'garch':
                    garch_forecasts = self.garch_estimator.forecast_variance(
                        returns_series, 
                        horizon=horizon, 
                        annualize=True, 
                        frequency=frequency
                    )
                    # Take the mean forecast and convert to volatility
                    forecasts[symbol] = np.sqrt(np.mean(garch_forecasts))
                    
                else:
                    # Simple historical volatility
                    hist_vol = returns_series.std()
                    if frequency == 'daily':
                        hist_vol *= np.sqrt(252)
                    elif frequency == 'monthly':
                        hist_vol *= np.sqrt(12)
                    forecasts[symbol] = hist_vol
                    
            except Exception as e:
                logger.warning(f"Volatility forecast failed for {symbol}: {e}")
                # Fallback to historical volatility
                hist_vol = returns_series.std()
                if frequency == 'daily':
                    hist_vol *= np.sqrt(252)
                elif frequency == 'monthly':
                    hist_vol *= np.sqrt(12)
                forecasts[symbol] = hist_vol
        
        return forecasts
    
    def estimate_market_weights(self, symbols: List[str]) -> np.ndarray:
        """Estimate market capitalization weights (simplified).
        
        Args:
            symbols: List of symbols
            
        Returns:
            Array of market cap weights (normalized to sum to 1)
        """
        # This is a simplified implementation
        # In practice, you'd fetch actual market cap data
        
        # Default weights based on common knowledge
        default_weights = {
            'SPY': 0.4,   # Large US equity weight
            'QQQ': 0.15,  # Tech weight
            'TLT': 0.2,   # Bond weight
            'IWM': 0.1,   # Small cap weight
            'VTI': 0.35,  # Total market weight
            'AGG': 0.2,   # Aggregate bonds
            'GLD': 0.05,  # Gold weight
            'VNQ': 0.05,  # REIT weight
        }
        
        weights = []
        for symbol in symbols:
            if symbol in default_weights:
                weights.append(default_weights[symbol])
            else:
                weights.append(1.0 / len(symbols))  # Equal weight fallback
        
        # Normalize to sum to 1
        weights = np.array(weights)
        return weights / np.sum(weights)