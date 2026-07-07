"""
Economic calculation utilities.

This module provides calculations for economic indicators from raw economic data.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class EconomicCalculator:
    """Calculates economic indicators from raw economic data."""
    
    def calculate_inflation_rate(
        self, 
        price_index: pd.Series, 
        method: str = "yoy"
    ) -> pd.Series:
        """
        Calculate inflation rate from price indices (CPI/PCE).
        
        Args:
            price_index: Price index series (CPI, PCE, etc.)
            method: Calculation method:
                - "yoy": Year-over-year percentage change
                - "mom_annualized": Month-over-month annualized
                - "mom": Month-over-month (not annualized)
                
        Returns:
            Inflation rate series (as decimal, e.g., 0.02 for 2%)
        """
        if len(price_index) < 2:
            logger.warning("Need at least 2 observations for inflation calculation")
            return pd.Series(dtype=float, index=price_index.index, name="inflation_rate")
        
        if method == "yoy":
            # Year-over-year: (CPI_t / CPI_{t-12}) - 1
            # Need at least 12 months of data
            if len(price_index) < 12:
                logger.warning("Need at least 12 observations for YoY inflation")
                return pd.Series(dtype=float, index=price_index.index, name="inflation_rate")
            
            inflation = price_index.pct_change(periods=12)
            
        elif method == "mom_annualized":
            # Month-over-month annualized: ((CPI_t / CPI_{t-1})^12) - 1
            monthly_change = price_index.pct_change()
            inflation = (1 + monthly_change) ** 12 - 1
            
        elif method == "mom":
            # Month-over-month: (CPI_t / CPI_{t-1}) - 1
            inflation = price_index.pct_change()
            
        else:
            raise ValueError(f"Unsupported inflation calculation method: {method}")
        
        inflation.name = f"inflation_rate_{method}"
        return inflation
    
    def calculate_real_rate(
        self, 
        nominal_rate: pd.Series, 
        inflation_rate: pd.Series
    ) -> pd.Series:
        """
        Calculate real interest rate using Fisher equation.
        
        Fisher equation: (1 + nominal) / (1 + inflation) - 1
        Approximation: nominal - inflation (for small rates)
        
        Args:
            nominal_rate: Nominal interest rate series (as decimal)
            inflation_rate: Inflation rate series (as decimal)
            
        Returns:
            Real interest rate series (as decimal)
        """
        # Align the two series
        aligned_data = pd.DataFrame({
            'nominal': nominal_rate, 
            'inflation': inflation_rate
        })
        
        # Use exact Fisher equation for accuracy
        real_rate = (1 + aligned_data['nominal']) / (1 + aligned_data['inflation']) - 1
        real_rate.name = "real_rate"
        
        return real_rate
    
    def calculate_term_premium(
        self, 
        long_rate: pd.Series, 
        short_rate: pd.Series
    ) -> pd.Series:
        """
        Calculate term premium: long_rate - short_rate
        
        Args:
            long_rate: Long-term interest rate (e.g., 10Y Treasury)
            short_rate: Short-term interest rate (e.g., 3M Treasury)
            
        Returns:
            Term premium series (as decimal)
        """
        # Align the two series
        aligned_data = pd.DataFrame({
            'long': long_rate,
            'short': short_rate
        })
        
        term_premium = aligned_data['long'] - aligned_data['short']
        term_premium.name = "term_premium"
        
        return term_premium
    
    def select_risk_free_rate(
        self, 
        available_rates: Dict[str, pd.Series], 
        tenor: str = "3m"
    ) -> pd.Series:
        """
        Select appropriate risk-free rate based on availability and tenor preference.
        
        Args:
            available_rates: Dictionary of rate series keyed by tenor
                           (e.g., {"3m": series, "6m": series, "1y": series})
            tenor: Preferred tenor ("3m", "6m", "1y", "2y", etc.)
            
        Returns:
            Selected risk-free rate series
        """
        if not available_rates:
            raise ValueError("No rates available for selection")
        
        # Define fallback order for different tenors
        fallback_chains = {
            "3m": ["3m", "6m", "1y", "2y", "fed_funds"],
            "6m": ["6m", "3m", "1y", "2y", "fed_funds"],
            "1y": ["1y", "6m", "2y", "3m", "fed_funds"],
            "2y": ["2y", "1y", "5y", "6m", "3m", "fed_funds"],
            "5y": ["5y", "2y", "10y", "1y", "6m", "3m"],
            "10y": ["10y", "5y", "2y", "1y", "6m", "3m"]
        }
        
        fallback_order = fallback_chains.get(tenor, ["3m", "6m", "1y", "2y", "fed_funds"])
        
        # Try each fallback option
        for fallback_tenor in fallback_order:
            if fallback_tenor in available_rates:
                selected_rate = available_rates[fallback_tenor].copy()
                selected_rate.name = f"risk_free_rate_{fallback_tenor}"
                logger.debug(f"Selected {fallback_tenor} rate as risk-free proxy (requested: {tenor})")
                return selected_rate
        
        # If no exact matches, try partial matches
        for key in available_rates.keys():
            if any(partial in key.lower() for partial in fallback_order):
                selected_rate = available_rates[key].copy()
                selected_rate.name = f"risk_free_rate_{key}"
                logger.debug(f"Selected {key} rate as risk-free proxy (requested: {tenor})")
                return selected_rate
        
        # Last resort: use any available rate
        first_key = list(available_rates.keys())[0]
        selected_rate = available_rates[first_key].copy()
        selected_rate.name = f"risk_free_rate_{first_key}"
        logger.warning(f"Using {first_key} rate as risk-free proxy (requested: {tenor}, limited options)")
        
        return selected_rate
    
    def calculate_real_yield(
        self, 
        nominal_yield: pd.Series, 
        breakeven_inflation: pd.Series
    ) -> pd.Series:
        """
        Calculate real yield from nominal yield and breakeven inflation.
        
        For TIPS: Real Yield ≈ Nominal Yield - Breakeven Inflation
        
        Args:
            nominal_yield: Nominal yield series
            breakeven_inflation: Breakeven inflation expectations
            
        Returns:
            Real yield series
        """
        aligned_data = pd.DataFrame({
            'nominal': nominal_yield,
            'breakeven': breakeven_inflation
        })
        
        real_yield = aligned_data['nominal'] - aligned_data['breakeven']
        real_yield.name = "real_yield"
        
        return real_yield
    
    def calculate_inflation_expectations(
        self, 
        nominal_rate: pd.Series, 
        real_rate: pd.Series
    ) -> pd.Series:
        """
        Calculate implied inflation expectations from nominal and real rates.
        
        From Fisher equation: Expected Inflation ≈ Nominal Rate - Real Rate
        
        Args:
            nominal_rate: Nominal interest rate
            real_rate: Real interest rate (e.g., TIPS yield)
            
        Returns:
            Implied inflation expectations
        """
        aligned_data = pd.DataFrame({
            'nominal': nominal_rate,
            'real': real_rate
        })
        
        inflation_expectations = aligned_data['nominal'] - aligned_data['real']
        inflation_expectations.name = "inflation_expectations"
        
        return inflation_expectations
    
    def smooth_economic_series(
        self, 
        series: pd.Series, 
        window: int = 3,
        method: str = "rolling_mean"
    ) -> pd.Series:
        """
        Smooth economic series to reduce noise.
        
        Args:
            series: Economic time series
            window: Smoothing window size
            method: Smoothing method ("rolling_mean", "ewma")
            
        Returns:
            Smoothed series
        """
        if method == "rolling_mean":
            smoothed = series.rolling(window=window, center=True).mean()
        elif method == "ewma":
            # Exponentially weighted moving average
            smoothed = series.ewm(span=window).mean()
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")
        
        smoothed.name = f"{series.name}_smoothed" if series.name else "smoothed_series"
        return smoothed