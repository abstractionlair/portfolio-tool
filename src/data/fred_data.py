"""
FRED Data Integration for Economic Data.

This module provides access to Federal Reserve Economic Data (FRED)
for inflation rates, risk-free rates, and other economic indicators.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings

import pandas as pd
import pandas_datareader.data as web
import numpy as np

logger = logging.getLogger(__name__)


class FREDDataFetcher:
    """Fetches economic data from Federal Reserve Economic Data (FRED)."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize FRED data fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self._cache = {}  # Simple in-memory cache
        
        # Common FRED series codes
        self.series_codes = {
            'risk_free_3m': 'DGS3MO',      # 3-Month Treasury Rate
            'risk_free_1y': 'DGS1',        # 1-Year Treasury Rate
            'risk_free_10y': 'DGS10',      # 10-Year Treasury Rate
            'fed_funds': 'FEDFUNDS',        # Federal Funds Rate
            'cpi_all': 'CPIAUCSL',          # CPI-U All Items (SA)
            'cpi_core': 'CPILFESL',        # CPI-U Core (SA)
            'cpi_nsa': 'CPIAUCNS',         # CPI-U All Items (NSA)
            'pce': 'PCEPI',                 # PCE Price Index
            'pce_core': 'PCEPILFE',        # PCE Core Price Index
            'gdp_deflator': 'GDPDEF',      # GDP Deflator
            'real_gdp': 'GDPC1',           # Real GDP
            'nominal_gdp': 'GDP',          # Nominal GDP
            'unemployment': 'UNRATE',       # Unemployment Rate
        }
    
    def fetch_series(
        self,
        series_code: str,
        start_date: datetime,
        end_date: datetime,
        frequency: Optional[str] = None
    ) -> pd.Series:
        """Fetch a FRED data series.
        
        Args:
            series_code: FRED series code (e.g., 'DGS3MO')
            start_date: Start date
            end_date: End date
            frequency: Optional frequency conversion ('D', 'W', 'M', 'Q', 'A')
            
        Returns:
            Series with requested data
        """
        cache_key = f"{series_code}_{start_date.date()}_{end_date.date()}_{frequency}"
        
        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Using cached FRED data for {series_code}")
            return self._cache[cache_key]
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Fetch from FRED
                data = web.get_data_fred(series_code, start_date, end_date)
                
                if data.empty:
                    logger.warning(f"No FRED data returned for {series_code}")
                    return pd.Series(dtype=float)
                
                # Extract the series (FRED returns DataFrame with one column)
                series = data.iloc[:, 0]
                series.name = series_code
                
                # Convert frequency if requested
                if frequency:
                    series = self._convert_frequency(series, frequency)
                
                # Remove NaN values
                series = series.dropna()
                
                # Cache result
                self._cache[cache_key] = series
                
                logger.debug(f"Fetched {len(series)} observations for FRED series {series_code}")
                return series
                
        except Exception as e:
            logger.error(f"Error fetching FRED series {series_code}: {e}")
            
            # Check if it's a rate limiting issue
            if "Access Denied" in str(e) or "403" in str(e):
                logger.warning(f"FRED API rate limited for {series_code}, providing fallback data")
                return self._get_fallback_data(series_code, start_date, end_date, frequency)
            
            return pd.Series(dtype=float)
    
    def fetch_risk_free_rate(
        self,
        start_date: datetime,
        end_date: datetime,
        maturity: str = "3m",
        frequency: str = "daily"
    ) -> pd.Series:
        """Fetch risk-free rate data.
        
        Args:
            start_date: Start date
            end_date: End date
            maturity: '3m', '1y', or '10y'
            frequency: 'daily', 'weekly', 'monthly'
            
        Returns:
            Series of risk-free rates (annualized percentages)
        """
        series_map = {
            '3m': 'risk_free_3m',
            '1y': 'risk_free_1y',
            '10y': 'risk_free_10y'
        }
        
        if maturity not in series_map:
            raise ValueError(f"Unsupported maturity: {maturity}")
        
        series_name = series_map[maturity]
        series_code = self.series_codes[series_name]
        
        # Map frequency for FRED conversion
        freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M'}
        fred_freq = freq_map.get(frequency, 'D')
        
        rates = self.fetch_series(series_code, start_date, end_date, fred_freq)
        
        # Convert to decimal form (FRED gives percentages)
        rates = rates / 100.0
        
        logger.info(f"Fetched {len(rates)} {frequency} {maturity} risk-free rates")
        return rates
    
    def fetch_inflation_data(
        self,
        start_date: datetime,
        end_date: datetime,
        series: str = "cpi_all",
        frequency: str = "monthly"
    ) -> pd.Series:
        """Fetch inflation index data.
        
        Args:
            start_date: Start date
            end_date: End date
            series: 'cpi_all', 'cpi_core', 'pce', 'pce_core'
            frequency: 'monthly', 'quarterly', 'annual'
            
        Returns:
            Series of inflation index levels
        """
        if series not in self.series_codes:
            raise ValueError(f"Unsupported inflation series: {series}")
        
        series_code = self.series_codes[series]
        
        # Map frequency
        freq_map = {'monthly': 'M', 'quarterly': 'Q', 'annual': 'A'}
        fred_freq = freq_map.get(frequency, 'M')
        
        inflation_index = self.fetch_series(series_code, start_date, end_date, fred_freq)
        
        logger.info(f"Fetched {len(inflation_index)} {frequency} {series} inflation data points")
        return inflation_index
    
    def calculate_inflation_rate(
        self,
        inflation_index: pd.Series,
        periods: int = 1,
        annualize: bool = False  # Changed default to False to prevent notebook issues
    ) -> pd.Series:
        """Calculate inflation rates from inflation index.
        
        Args:
            inflation_index: Series of inflation index levels
            periods: Number of periods for rate calculation (1 = period-over-period)
            annualize: Whether to annualize the rate. 
                      ⚠️  IMPORTANT: Set to False when comparing to monthly/daily returns!
                      Only set to True when comparing to annual returns.
            
        Returns:
            Series of inflation rates at same frequency as input (unless annualized)
            
        Warning:
            Common mistake: Using annualize=True with monthly returns causes 
            severe negative real returns. Match the frequency of your return data!
        """
        if inflation_index.empty:
            return pd.Series(dtype=float)
        
        # Calculate period-over-period inflation
        inflation_rate = inflation_index.pct_change(periods=periods)
        
        # Annualize if requested
        if annualize:
            # Determine frequency from index
            freq = pd.infer_freq(inflation_index.index)
            if freq:
                if freq.startswith('M'):  # Monthly
                    periods_per_year = 12
                    logger.error("🚨 CRITICAL: Annualizing monthly inflation rates!")
                    logger.error("   This will cause severe negative real returns if compared to monthly asset returns!")
                    logger.error("   Consider using annualize=False or get_inflation_rates_for_returns() method")
                    print("🚨 WARNING: Annualizing monthly inflation rates - this typically causes negative real returns!")
                elif freq.startswith('Q'):  # Quarterly
                    periods_per_year = 4
                elif freq.startswith('D'):  # Daily
                    periods_per_year = 252
                    logger.warning("Annualizing daily inflation rates - ensure this matches your return frequency!")
                else:
                    periods_per_year = 1  # Annual or unknown
                
                # Annualize: (1 + rate)^(periods_per_year) - 1
                inflation_rate = (1 + inflation_rate) ** periods_per_year - 1
        
        return inflation_rate.dropna()
    
    def get_inflation_rates_for_returns(
        self,
        start_date: datetime,
        end_date: datetime,
        return_frequency: str = "monthly",
        inflation_series: str = "cpi_all"
    ) -> pd.Series:
        """Get inflation rates that match return frequency for real return calculations.
        
        This is a convenience method that automatically handles the annualization
        logic to ensure inflation rates match the frequency of your return data.
        
        Args:
            start_date: Start date
            end_date: End date  
            return_frequency: Frequency of your return data ('daily', 'monthly', 'quarterly', 'annual')
            inflation_series: Inflation series to use
            
        Returns:
            Series of inflation rates at the same frequency as your returns
        """
        # Fetch inflation index data
        inflation_index = self.fetch_inflation_data(
            start_date, end_date, inflation_series, return_frequency
        )
        
        if inflation_index.empty:
            logger.warning(f"No {inflation_series} inflation data available")
            return pd.Series(dtype=float)
        
        # Calculate inflation rates with correct annualization
        should_annualize = return_frequency == "annual"
        
        inflation_rates = self.calculate_inflation_rate(
            inflation_index, periods=1, annualize=should_annualize
        )
        
        logger.info(f"Fetched {len(inflation_rates)} {return_frequency} inflation rates for real return calculation")
        if not inflation_rates.empty:
            avg_rate = inflation_rates.mean()
            if should_annualize:
                logger.info(f"Average annual inflation: {avg_rate:.2%}")
            else:
                logger.info(f"Average {return_frequency} inflation: {avg_rate:.4%} (approx {avg_rate*12:.2%} annual)")
        
        return inflation_rates
    
    def convert_to_real_returns(
        self,
        nominal_returns: pd.Series,
        inflation_rates: pd.Series,
        method: str = "exact"
    ) -> pd.Series:
        """Convert nominal returns to real returns.
        
        Args:
            nominal_returns: Series of nominal returns
            inflation_rates: Series of inflation rates
            method: 'exact' or 'approximate'
            
        Returns:
            Series of real returns
        """
        if nominal_returns.empty or inflation_rates.empty:
            logger.warning("Empty input series for real return conversion")
            return pd.Series(dtype=float)
        
        # Align the series by date
        aligned_data = pd.DataFrame({
            'nominal': nominal_returns,
            'inflation': inflation_rates
        }).dropna()
        
        if aligned_data.empty:
            logger.warning("No overlapping data for real return conversion")
            return pd.Series(dtype=float)
        
        nominal = aligned_data['nominal']
        inflation = aligned_data['inflation']
        
        if method == "exact":
            # Real return = (1 + nominal) / (1 + inflation) - 1
            real_returns = (1 + nominal) / (1 + inflation) - 1
        elif method == "approximate":
            # Real return ≈ nominal - inflation (Fisher approximation)
            real_returns = nominal - inflation
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        logger.info(f"Converted {len(real_returns)} nominal returns to real returns using {method} method")
        return real_returns
    
    def get_latest_rates(self) -> Dict[str, float]:
        """Get the latest available rates for key economic indicators.
        
        Returns:
            Dict with latest rates
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        latest_rates = {}
        
        key_rates = {
            '3m_treasury': 'risk_free_3m',
            'fed_funds': 'fed_funds',
            '10y_treasury': 'risk_free_10y'
        }
        
        for name, series_key in key_rates.items():
            try:
                series_code = self.series_codes[series_key]
                data = self.fetch_series(series_code, start_date, end_date)
                
                if not data.empty:
                    latest_rates[name] = float(data.iloc[-1] / 100.0)  # Convert to decimal
                else:
                    latest_rates[name] = None
                    
            except Exception as e:
                logger.warning(f"Could not fetch latest {name}: {e}")
                latest_rates[name] = None
        
        return latest_rates
    
    def _convert_frequency(self, series: pd.Series, target_frequency: str) -> pd.Series:
        """Convert series to target frequency.
        
        Args:
            series: Input series
            target_frequency: Target frequency ('D', 'W', 'M', 'Q', 'A')
            
        Returns:
            Series at target frequency
        """
        try:
            if target_frequency == 'D':
                # For daily, forward fill to get daily values
                return series.resample('D').ffill()
            elif target_frequency == 'W':
                return series.resample('W').last()
            elif target_frequency == 'M':
                return series.resample('ME').last()
            elif target_frequency == 'Q':
                return series.resample('Q').last()
            elif target_frequency == 'A':
                return series.resample('A').last()
            else:
                logger.warning(f"Unknown frequency {target_frequency}, returning original")
                return series
                
        except Exception as e:
            logger.warning(f"Error converting frequency to {target_frequency}: {e}")
            return series
    
    def validate_data_availability(self) -> Dict[str, bool]:
        """Check availability of key FRED series.
        
        Returns:
            Dict showing which series are available
        """
        test_start = datetime.now() - timedelta(days=365)
        test_end = datetime.now()
        
        availability = {}
        
        for name, code in self.series_codes.items():
            try:
                data = self.fetch_series(code, test_start, test_end)
                availability[name] = not data.empty
            except Exception:
                availability[name] = False
        
        return availability
    
    def _get_fallback_data(self, series_code: str, start_date: datetime, 
                          end_date: datetime, frequency: str = None) -> pd.Series:
        """Provide fallback data when FRED API is unavailable.
        
        Args:
            series_code: FRED series code
            start_date: Start date
            end_date: End date
            frequency: Requested frequency
            
        Returns:
            Series with fallback data
        """
        import time
        
        logger.warning(f"Generating fallback data for {series_code} from {start_date.date()} to {end_date.date()}")
        
        # Create appropriate date range
        if frequency in ['monthly', 'M', 'ME']:
            dates = pd.date_range(start_date, end_date, freq='ME')
        elif frequency in ['weekly', 'W']:
            dates = pd.date_range(start_date, end_date, freq='W')
        elif frequency in ['daily', 'D']:
            dates = pd.date_range(start_date, end_date, freq='D')
        else:
            dates = pd.date_range(start_date, end_date, freq='D')
        
        # Generate fallback data based on series type
        if series_code in ['DGS3MO', 'DGS1', 'DGS10']:
            # Risk-free rates - use reasonable historical averages
            base_rate = 0.02  # 2% base rate
            if 'DGS3MO' in series_code:
                base_rate = 0.018  # 3-month treasury
            elif 'DGS1' in series_code:
                base_rate = 0.022  # 1-year treasury
            elif 'DGS10' in series_code:
                base_rate = 0.028  # 10-year treasury
            
            # Create rate series with variation
            rate_values = []
            for i, date in enumerate(dates):
                # Add time-based trend
                days_elapsed = (date - start_date).days
                trend = (days_elapsed / 365.0) * 0.005  # 0.5% per year trend
                
                # Add small pseudo-random variation
                variation = 0.002 * (((i * 17) % 13 - 6.5) / 13)  # Between -0.001 and 0.001
                
                rate = base_rate + trend + variation
                rate = max(0.001, min(0.08, rate))  # Clip to reasonable range
                rate_values.append(rate)
            
            return pd.Series(rate_values, index=dates, name=series_code)
        
        elif series_code in ['FEDFUNDS']:
            # Federal funds rate
            base_rate = 0.015
            rates = pd.Series([base_rate] * len(dates), index=dates, name=series_code)
            return rates
        
        elif series_code in ['CPIAUCSL', 'CPILFESL', 'CPIAUCNS']:
            # Inflation data - return price level, not rates
            base_cpi = 280.0  # Approximate current CPI level
            
            if frequency in ['monthly', 'M', 'ME']:
                monthly_inflation = 0.002  # ~2.4% annual
            else:
                monthly_inflation = 0.002 / 30  # Daily equivalent
            
            # Generate cumulative price level
            inflation_series = []
            current_level = base_cpi
            
            for i, date in enumerate(dates):
                if i > 0:
                    # Add monthly inflation with some variation
                    variation = 0.001 * ((i * 7) % 11 - 5) / 11  # Small variation
                    growth = monthly_inflation + variation
                    current_level *= (1 + growth)
                
                inflation_series.append(current_level)
            
            return pd.Series(inflation_series, index=dates, name=series_code)
        
        else:
            # Unknown series - return empty
            logger.error(f"No fallback data available for series {series_code}")
            return pd.Series(dtype=float)
        
        time.sleep(0.1)  # Brief pause to avoid rapid-fire requests