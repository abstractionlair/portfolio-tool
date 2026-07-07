"""
Total Return Data Fetching and Processing.

This module provides enhanced data fetching capabilities specifically focused on
total returns including dividends and distributions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import warnings

import pandas as pd
import yfinance as yf
import numpy as np
from pandas import DataFrame, Series

from .exposure_universe import ExposureUniverse, Exposure, Implementation

logger = logging.getLogger(__name__)


class TotalReturnFetcher:
    """Enhanced data fetcher focused on total returns."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the total return fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self._cache = {}  # Simple in-memory cache for now
    
    def fetch_total_returns(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily",
        validate: bool = True
    ) -> pd.Series:
        """Fetch total returns for a single ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            frequency: 'daily', 'weekly', 'monthly'
            validate: Whether to validate data quality
            
        Returns:
            Series of total returns
        """
        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}_{frequency}"
        
        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"Using cached data for {ticker}")
            return self._cache[cache_key]
        
        try:
            # Fetch data with proper total return handling
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Use auto_adjust=True to get total returns automatically
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True  # This gives us total returns
                )
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.Series(dtype=float)
            
            # Extract close prices (which are now total return adjusted)
            if isinstance(data.columns, pd.MultiIndex):
                # Handle case where yfinance returns MultiIndex columns
                prices = data[('Close', ticker)] if ('Close', ticker) in data.columns else data['Close'].iloc[:, 0]
            else:
                prices = data['Close']
            
            # Calculate returns
            returns = self._calculate_returns(prices, frequency)
            
            # Validate if requested
            if validate:
                validation_result = self._validate_returns(returns, ticker)
                if not validation_result['valid']:
                    logger.warning(f"Data validation failed for {ticker}: {validation_result['issues']}")
            
            # Cache result
            self._cache[cache_key] = returns
            
            logger.debug(f"Fetched {len(returns)} {frequency} returns for {ticker}")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching total returns for {ticker}: {e}")
            return pd.Series(dtype=float)
    
    def fetch_composite_returns(
        self,
        components: List[Dict[str, Union[str, float]]],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily"
    ) -> pd.Series:
        """Fetch returns for a weighted composite of securities.
        
        Args:
            components: List of dicts with 'ticker' and 'weight' keys
            start_date: Start date
            end_date: End date
            frequency: Return frequency
            
        Returns:
            Series of composite returns
        """
        # Fetch returns for each component
        component_returns = {}
        total_weight = 0.0
        
        for component in components:
            ticker = component['ticker']
            weight = component.get('weight', 1.0)
            
            returns = self.fetch_total_returns(ticker, start_date, end_date, frequency)
            if not returns.empty:
                component_returns[ticker] = returns * weight
                total_weight += weight
        
        if not component_returns:
            logger.warning("No valid components found for composite")
            return pd.Series(dtype=float)
        
        # Combine returns
        all_returns = pd.DataFrame(component_returns)
        
        # Handle missing data by forward filling
        all_returns = all_returns.ffill().fillna(0)
        
        # Calculate weighted composite returns
        if total_weight > 0:
            composite_returns = all_returns.sum(axis=1) / total_weight
        else:
            composite_returns = all_returns.mean(axis=1)
        
        logger.info(f"Created composite from {len(component_returns)} components")
        return composite_returns
    
    def fetch_etf_average_returns(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily",
        weighting: str = "equal"
    ) -> pd.Series:
        """Fetch returns for an average of ETFs.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            frequency: Return frequency
            weighting: 'equal' or 'market_cap'
            
        Returns:
            Series of average returns
        """
        if weighting != "equal":
            logger.warning(f"Weighting '{weighting}' not implemented, using equal weights")
        
        # Create components list with equal weights
        components = [{'ticker': ticker, 'weight': 1.0/len(tickers)} for ticker in tickers]
        
        return self.fetch_composite_returns(components, start_date, end_date, frequency)
    
    def fetch_rate_series_returns(
        self,
        series_code: str,
        source: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily"
    ) -> pd.Series:
        """Fetch returns from a rate series (e.g., FRED Treasury rates).
        
        Args:
            series_code: FRED series code (e.g., 'DGS3MO')
            source: Data source (currently only 'FRED' supported)
            start_date: Start date
            end_date: End date
            frequency: Return frequency ('daily', 'monthly', 'annual')
            
        Returns:
            Series of returns at the specified frequency
        """
        if source != "FRED":
            raise ValueError(f"Unsupported rate series source: {source}")
        
        # Import here to avoid circular imports
        from .fred_data import FREDDataFetcher
        
        # Use existing FRED fetcher
        fred_fetcher = FREDDataFetcher()
        
        # Fetch the rate data (comes as annualized decimal, e.g., 0.0525 for 5.25%)
        rates = fred_fetcher.fetch_series(
            series_code=series_code,
            start_date=start_date,
            end_date=end_date,
            frequency='D'  # Always fetch daily, then convert
        )
        
        if rates.empty:
            logger.warning(f"No FRED data returned for {series_code}")
            return pd.Series(dtype=float)
        
        # FRED returns percentages, convert to decimal if needed
        # Check if rates are already in decimal form (< 1) or percentage form (> 1)
        if rates.max() > 1:
            rates = rates / 100.0
        
        # Convert annualized rates to period returns
        if frequency == "daily":
            # Daily return from annualized rate
            # Using 252 trading days per year
            period_returns = rates / 252
        elif frequency == "weekly":
            # Weekly return from annualized rate
            period_returns = (1 + rates) ** (1/52) - 1
        elif frequency == "monthly":
            # Monthly return from annualized rate
            # Using geometric conversion: (1 + r)^(1/12) - 1
            period_returns = (1 + rates) ** (1/12) - 1
        elif frequency == "annual":
            # Already annualized
            period_returns = rates
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Forward fill missing values (weekends, holidays)
        period_returns = period_returns.ffill()
        
        # Resample to target frequency if needed
        if frequency == "weekly":
            period_returns = period_returns.resample('W').last()
        elif frequency == "monthly":
            period_returns = period_returns.resample('ME').last()
        
        logger.info(f"Fetched {len(period_returns)} {frequency} returns from FRED {series_code}")
        return period_returns
    
    def fetch_returns_for_exposure(
        self,
        exposure: Exposure,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily",
        available_tickers: Optional[List[str]] = None
    ) -> Tuple[pd.Series, str]:
        """Fetch returns for an exposure using the best available implementation.
        
        Args:
            exposure: Exposure object
            start_date: Start date
            end_date: End date
            frequency: Return frequency
            available_tickers: List of available tickers for validation
            
        Returns:
            Tuple of (returns_series, implementation_description)
        """
        # Get the preferred implementation
        preferred_impl = exposure.get_preferred_implementation(available_tickers)
        
        if not preferred_impl:
            logger.error(f"No suitable implementation found for exposure {exposure.id}")
            return pd.Series(dtype=float), "No implementation available"
        
        # Handle different implementation types
        if preferred_impl.type == "etf_average":
            returns = self.fetch_etf_average_returns(
                preferred_impl.tickers,
                start_date,
                end_date,
                frequency
            )
            description = f"ETF average of {preferred_impl.tickers}"
            
        elif preferred_impl.type == "fund_average":
            # Handle mutual fund averages the same way as ETF averages
            returns = self.fetch_etf_average_returns(
                preferred_impl.tickers,
                start_date,
                end_date,
                frequency
            )
            description = f"Fund average of {preferred_impl.tickers}"
            
        elif preferred_impl.type == "fund":
            returns = self.fetch_total_returns(
                preferred_impl.ticker,
                start_date,
                end_date,
                frequency
            )
            description = f"Fund {preferred_impl.ticker}"
            
        elif preferred_impl.type == "composite":
            returns = self.fetch_composite_returns(
                preferred_impl.components,
                start_date,
                end_date,
                frequency
            )
            component_tickers = [c.get('ticker') for c in preferred_impl.components]
            description = f"Composite of {component_tickers}"
            
        elif preferred_impl.type == "rate_series":
            # Handle FRED rate series
            returns = self.fetch_rate_series_returns(
                preferred_impl.series,
                preferred_impl.source,
                start_date,
                end_date,
                frequency
            )
            description = f"Rate series {preferred_impl.series} from {preferred_impl.source}"
            
        else:
            logger.warning(f"Implementation type '{preferred_impl.type}' not supported for {exposure.id}")
            returns = pd.Series(dtype=float)
            description = f"Unsupported implementation type: {preferred_impl.type}"
        
        logger.info(f"Fetched returns for {exposure.id} using {description}")
        return returns, description
    
    def fetch_universe_returns(
        self,
        universe: ExposureUniverse,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily",
        available_tickers: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Union[pd.Series, str]]]:
        """Fetch returns for all exposures in a universe.
        
        Args:
            universe: ExposureUniverse object
            start_date: Start date
            end_date: End date
            frequency: Return frequency
            available_tickers: List of available tickers
            
        Returns:
            Dict mapping exposure_id to dict with 'returns' and 'implementation'
        """
        results = {}
        
        for exposure in universe:
            try:
                returns, description = self.fetch_returns_for_exposure(
                    exposure,
                    start_date,
                    end_date,
                    frequency,
                    available_tickers
                )
                
                results[exposure.id] = {
                    'returns': returns,
                    'implementation': description,
                    'success': not returns.empty
                }
                
                if returns.empty:
                    logger.warning(f"Failed to fetch returns for exposure {exposure.id}")
                else:
                    logger.info(f"Successfully fetched {len(returns)} returns for {exposure.id}")
                    
            except Exception as e:
                logger.error(f"Error fetching returns for exposure {exposure.id}: {e}")
                results[exposure.id] = {
                    'returns': pd.Series(dtype=float),
                    'implementation': f"Error: {str(e)}",
                    'success': False
                }
        
        # Summary
        successful = sum(1 for r in results.values() if r['success'])
        logger.info(f"Successfully fetched returns for {successful}/{len(results)} exposures")
        
        return results
    
    def _calculate_returns(self, prices: pd.Series, frequency: str) -> pd.Series:
        """Calculate returns from price series.
        
        Args:
            prices: Price series
            frequency: 'daily', 'weekly', 'monthly'
            
        Returns:
            Returns series
        """
        if frequency == "daily":
            return prices.pct_change().dropna()
        elif frequency == "weekly":
            weekly_prices = prices.resample('W').last()
            return weekly_prices.pct_change().dropna()
        elif frequency == "monthly":
            monthly_prices = prices.resample('ME').last()
            return monthly_prices.pct_change().dropna()
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
    
    def _validate_returns(self, returns: pd.Series, ticker: str) -> Dict[str, Union[bool, List[str]]]:
        """Validate return data quality.
        
        Args:
            returns: Returns series
            ticker: Ticker symbol for logging
            
        Returns:
            Dict with validation results
        """
        issues = []
        
        if returns.empty:
            issues.append("No data")
            return {'valid': False, 'issues': issues}
        
        # Check for excessive missing data
        missing_pct = returns.isna().sum() / len(returns)
        if missing_pct > 0.1:  # More than 10% missing
            issues.append(f"Too much missing data: {missing_pct:.1%}")
        
        # Check for extreme outliers (>50% daily moves)
        extreme_returns = returns[abs(returns) > 0.5]
        if len(extreme_returns) > 0:
            issues.append(f"Extreme returns detected: {len(extreme_returns)} days >50%")
        
        # Check for long gaps in data (allow for weekends/holidays)
        date_diff = pd.Series(returns.index).diff()
        max_gap = date_diff.max()
        if max_gap and max_gap.days > 40:  # More lenient for monthly data
            issues.append(f"Long data gap detected: {max_gap.days} days")
        
        # Check for unrealistic volatility
        annualized_vol = returns.std() * np.sqrt(252)
        if annualized_vol > 2.0:  # >200% annual volatility
            issues.append(f"Unrealistic volatility: {annualized_vol:.1%}")
        
        valid = len(issues) == 0
        return {'valid': valid, 'issues': issues}
    
    def get_data_summary(self, returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Get summary statistics for multiple return series.
        
        Args:
            returns_dict: Dict mapping names to return series
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for name, returns in returns_dict.items():
            if returns.empty:
                continue
                
            summary = {
                'exposure': name,
                'start_date': returns.index[0],
                'end_date': returns.index[-1],
                'observations': len(returns),
                'years': (returns.index[-1] - returns.index[0]).days / 365.25,
                'missing_pct': returns.isna().sum() / len(returns),
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'annualized_vol': returns.std() * np.sqrt(252),
                'min_return': returns.min(),
                'max_return': returns.max(),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            }
            summary_data.append(summary)
        
        if not summary_data:
            return pd.DataFrame()
        
        return pd.DataFrame(summary_data).set_index('exposure')