"""
Return calculation utilities.

This module provides calculations for various types of returns from price and dividend data.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ReturnCalculator:
    """Calculates various types of returns from price and dividend data."""
    
    def calculate_simple_returns(
        self, 
        prices: pd.Series, 
        frequency: str = "daily"
    ) -> pd.Series:
        """
        Calculate simple returns: (P_t - P_{t-1}) / P_{t-1}
        
        Args:
            prices: Price series with datetime index
            frequency: Frequency hint (not used for calculation but for validation)
            
        Returns:
            Series of simple returns with same index as prices (first value is NaN)
        """
        if len(prices) < 2:
            logger.warning("Need at least 2 price observations for return calculation")
            return pd.Series(dtype=float, index=prices.index, name=f"{prices.name}_returns")
        
        # Calculate simple returns: (P_t / P_{t-1}) - 1
        returns = prices.pct_change()
        returns.name = f"{prices.name}_returns" if prices.name else "returns"
        
        return returns
    
    def calculate_total_returns(
        self, 
        prices: pd.Series, 
        dividends: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate total returns including dividends: (P_t + D_t) / P_{t-1} - 1
        
        Args:
            prices: Price series (typically adjusted close)
            dividends: Dividend series (optional). If None, uses price-only returns
            
        Returns:
            Series of total returns including dividend reinvestment
        """
        if len(prices) < 2:
            logger.warning("Need at least 2 price observations for return calculation")
            return pd.Series(dtype=float, index=prices.index, name=f"{prices.name}_total_returns")
        
        if dividends is None:
            logger.debug("No dividends provided, calculating price-only returns")
            returns = self.calculate_simple_returns(prices)
            returns.name = f"{prices.name}_total_returns" if prices.name else "total_returns"
            return returns
        
        # Handle timezone mismatches by converting to timezone-naive
        prices_clean = prices.copy()
        if hasattr(prices.index, 'tz') and prices.index.tz is not None:
            prices_clean.index = prices.index.tz_localize(None)
        
        dividends_clean = dividends.copy() if dividends is not None else None
        if dividends_clean is not None and hasattr(dividends.index, 'tz') and dividends.index.tz is not None:
            dividends_clean.index = dividends.index.tz_localize(None)
        
        # Remove any NaN prices first
        prices_clean = prices_clean.dropna()
        
        if len(prices_clean) < 2:
            logger.warning("Insufficient non-NaN price data for return calculation")
            return pd.Series(dtype=float, index=prices.index, name=f"{prices.name}_total_returns")
        
        # For dividend data, only keep actual dividend payments (filter out zeros and NaNs)
        if not dividends_clean.empty:
            dividends_clean = dividends_clean.dropna()
            dividends_clean = dividends_clean[dividends_clean > 0]  # Only actual dividends
        
        # Use the price index as the primary index and align dividends to it
        # This ensures we have returns for all price dates
        aligned_dividends = pd.Series(0.0, index=prices_clean.index, name='dividends')
        
        # Add actual dividend payments on their respective dates
        # Handle timezone alignment for dividend date matching
        for div_date, div_amount in dividends_clean.items():
            # Convert dividend date to match price index timezone if needed
            if hasattr(prices_clean.index, 'tz') and prices_clean.index.tz is not None:
                if div_date.tz is None:
                    div_date = div_date.tz_localize(prices_clean.index.tz)
                elif div_date.tz != prices_clean.index.tz:
                    div_date = div_date.tz_convert(prices_clean.index.tz)
            
            if div_date in aligned_dividends.index:
                aligned_dividends.loc[div_date] = div_amount
        
        logger.debug(f"Aligned data: {len(prices_clean)} price points, "
                    f"{len(aligned_dividends[aligned_dividends > 0])} dividend payments")
        
        # Calculate returns using vectorized operations
        # Total return = (Price_t + Dividend_t) / Price_{t-1} - 1
        prev_prices = prices_clean.shift(1)
        total_returns = (prices_clean + aligned_dividends) / prev_prices - 1
        
        # The first return is always NaN (no previous price), which is correct
        total_returns.name = f"{prices.name}_total_returns" if prices.name else "total_returns"
        
        # Check if we need to reindex back to original timezone-aware index
        original_tz_naive_index = prices.index.tz_localize(None) if hasattr(prices.index, 'tz') and prices.index.tz is not None else prices.index
        
        if len(prices_clean) == len(prices) and prices_clean.index.equals(original_tz_naive_index):
            # Convert result back to original timezone if needed
            if hasattr(prices.index, 'tz') and prices.index.tz is not None:
                result = total_returns.copy()
                result.index = result.index.tz_localize(prices.index.tz)
                return result
            else:
                return total_returns
        else:
            # Need to reindex - convert target index to timezone-naive first
            result = total_returns.reindex(original_tz_naive_index)
            # Convert back to original timezone if needed
            if hasattr(prices.index, 'tz') and prices.index.tz is not None:
                result.index = result.index.tz_localize(prices.index.tz)
            return result
    
    def calculate_comprehensive_total_returns(
        self,
        prices: pd.Series,
        dividends: Optional[pd.Series] = None,
        splits: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate comprehensive total returns including all corporate actions.
        
        This method accounts for:
        - Price appreciation
        - Dividend payments
        - Stock splits and stock dividends
        - Other corporate actions affecting share count
        
        Args:
            prices: Price series (should be unadjusted close for accuracy)
            dividends: Dividend series (per share payments)
            splits: Split series (split ratios, e.g., 2.0 for 2:1 split)
            
        Returns:
            Series of comprehensive total returns
        """
        if len(prices) < 2:
            logger.warning("Need at least 2 price observations for return calculation")
            return pd.Series(dtype=float, index=prices.index, name=f"{prices.name}_comprehensive_total_returns")
        
        # If using adjusted close prices (which already account for splits),
        # and we have split data, we need to be careful not to double-adjust
        logger.info("Calculating comprehensive total returns with corporate actions")
        
        # Align all data on common index
        data_dict = {'prices': prices}
        
        if dividends is not None:
            # Handle timezone alignment
            if hasattr(dividends.index, 'tz') and dividends.index.tz is not None:
                dividends = dividends.copy()
                dividends.index = dividends.index.tz_localize(None)
            data_dict['dividends'] = dividends
            
        if splits is not None:
            # Handle timezone alignment
            if hasattr(splits.index, 'tz') and splits.index.tz is not None:
                splits = splits.copy()
                splits.index = splits.index.tz_localize(None)
            data_dict['splits'] = splits
        
        # Handle timezone for prices
        if hasattr(prices.index, 'tz') and prices.index.tz is not None:
            data_dict['prices'] = prices.copy()
            data_dict['prices'].index = prices.index.tz_localize(None)
        
        # Create aligned DataFrame
        aligned_data = pd.DataFrame(data_dict)
        aligned_data['dividends'] = aligned_data.get('dividends', pd.Series(index=aligned_data.index)).fillna(0.0)
        aligned_data['splits'] = aligned_data.get('splits', pd.Series(index=aligned_data.index)).fillna(1.0)
        
        # Calculate comprehensive total returns
        prices_adj = aligned_data['prices'].copy()
        dividends_adj = aligned_data['dividends'].copy()
        splits_adj = aligned_data['splits'].copy()
        
        # Adjust for stock splits if using unadjusted prices
        # Note: If prices are already adjusted, splits should be 1.0 or not provided
        cumulative_split_factor = splits_adj.cumprod()
        
        # For each period, calculate total return including corporate actions
        total_returns = pd.Series(index=prices_adj.index, dtype=float)
        
        for i in range(1, len(prices_adj)):
            curr_date = prices_adj.index[i]
            prev_date = prices_adj.index[i-1]
            
            # Current and previous prices
            p_curr = prices_adj.iloc[i]
            p_prev = prices_adj.iloc[i-1]
            
            # Dividend on current date (if any)
            div_curr = dividends_adj.iloc[i]
            
            # Split ratio for current period
            split_curr = splits_adj.iloc[i]
            
            if pd.isna(p_curr) or pd.isna(p_prev) or p_prev <= 0:
                total_returns.iloc[i] = np.nan
                continue
            
            # Calculate total return including all effects
            # Formula: ((P_t + D_t) * Split_t) / P_{t-1} - 1
            # Where Split_t adjusts for stock splits
            total_return = ((p_curr + div_curr) * split_curr) / p_prev - 1
            total_returns.iloc[i] = total_return
        
        total_returns.name = f"{prices.name}_comprehensive_total_returns" if prices.name else "comprehensive_total_returns"
        
        # Log summary of corporate actions included
        split_events = len(splits_adj[splits_adj != 1.0]) if splits is not None else 0
        dividend_events = len(dividends_adj[dividends_adj > 0]) if dividends is not None else 0
        
        logger.info(f"Comprehensive total returns calculated: {dividend_events} dividend events, {split_events} split events")
        
        return total_returns
    
    def calculate_log_returns(
        self, 
        prices: pd.Series
    ) -> pd.Series:
        """
        Calculate log returns: ln(P_t / P_{t-1})
        
        Args:
            prices: Price series with datetime index
            
        Returns:
            Series of log returns
        """
        if len(prices) < 2:
            logger.warning("Need at least 2 price observations for return calculation")
            return pd.Series(dtype=float, index=prices.index, name=f"{prices.name}_log_returns")
        
        # Calculate log returns: ln(P_t / P_{t-1})
        log_returns = np.log(prices / prices.shift(1))
        log_returns.name = f"{prices.name}_log_returns" if prices.name else "log_returns"
        
        return log_returns
    
    def calculate_excess_returns(
        self, 
        returns: pd.Series, 
        risk_free_rate: pd.Series
    ) -> pd.Series:
        """
        Calculate excess returns: R_t - RF_t
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate series (same frequency as returns)
            
        Returns:
            Series of excess returns
        """
        # Align the two series on their date index
        aligned_data = pd.DataFrame({'returns': returns, 'risk_free': risk_free_rate})
        
        # Calculate excess returns
        excess_returns = aligned_data['returns'] - aligned_data['risk_free']
        excess_returns.name = f"{returns.name}_excess" if returns.name else "excess_returns"
        
        return excess_returns
    
    def calculate_cumulative_returns(
        self, 
        returns: pd.Series
    ) -> pd.Series:
        """
        Calculate cumulative returns from a return series.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Series of cumulative returns (starting at 1.0)
        """
        if len(returns) == 0:
            return pd.Series(dtype=float, index=returns.index, name=f"{returns.name}_cumulative")
        
        # Calculate cumulative returns: (1 + r1) * (1 + r2) * ... - 1
        # Using (1 + returns).cumprod() - 1 for numerical stability
        cumulative = (1 + returns.fillna(0)).cumprod() - 1
        cumulative.name = f"{returns.name}_cumulative" if returns.name else "cumulative_returns"
        
        return cumulative
    
    def annualize_returns(
        self, 
        returns: pd.Series, 
        frequency: str = "daily"
    ) -> pd.Series:
        """
        Annualize a return series based on its frequency.
        
        Args:
            returns: Return series
            frequency: Frequency of the returns ("daily", "weekly", "monthly", etc.)
            
        Returns:
            Annualized return series
        """
        # Define periods per year for different frequencies
        periods_per_year = {
            "daily": 252,    # Trading days
            "weekly": 52,
            "monthly": 12,
            "quarterly": 4,
            "annual": 1
        }
        
        periods = periods_per_year.get(frequency.lower(), 252)
        
        if periods == 1:
            # Already annual
            return returns.copy()
        
        # Annualize: (1 + avg_return)^periods - 1
        # For a series, we'll annualize each period's return
        annualized = (1 + returns) ** periods - 1
        annualized.name = f"{returns.name}_annualized" if returns.name else "annualized_returns"
        
        return annualized
    
    def compound_returns(
        self, 
        returns: pd.Series, 
        from_frequency: str, 
        to_frequency: str
    ) -> pd.Series:
        """
        Compound returns from one frequency to another.
        
        Args:
            returns: Return series at original frequency
            from_frequency: Original frequency ("daily", "weekly", "monthly")
            to_frequency: Target frequency (must be lower frequency)
            
        Returns:
            Returns compounded to target frequency
        """
        frequency_map = {
            "daily": "D",
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE", 
            "annual": "YE"
        }
        
        if to_frequency not in frequency_map:
            raise ValueError(f"Unsupported target frequency: {to_frequency}")
        
        pandas_freq = frequency_map[to_frequency]
        
        # Compound returns by period
        # (1 + r1) * (1 + r2) * ... - 1 for each period
        period_returns = (1 + returns.fillna(0)).resample(pandas_freq).prod() - 1
        period_returns.name = f"{returns.name}_{to_frequency}" if returns.name else f"{to_frequency}_returns"
        
        return period_returns