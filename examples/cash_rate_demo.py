#!/usr/bin/env python3
"""
Example demonstrating how to fetch cash/risk-free rate returns.
Shows the solution for the DGS3MO rate series issue.
"""

from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fred_data import FREDDataFetcher
from src.data.market_data import MarketDataFetcher


def fetch_cash_rate_returns(start_date, end_date, frequency="daily"):
    """
    Fetch cash/risk-free rate returns using FRED data.
    
    This demonstrates the solution for the rate series issue:
    - FRED provides annualized rates (e.g., 5.25%)
    - We need to convert to period returns
    """
    
    print("Fetching Cash/Risk-Free Rate Returns")
    print("=" * 50)
    
    # Initialize FRED fetcher
    fred_fetcher = FREDDataFetcher()
    
    # Method 1: Using FRED DGS3MO (preferred)
    print("\n1. Fetching from FRED (DGS3MO)...")
    try:
        # Fetch risk-free rate (comes as annualized percentage)
        rf_rates = fred_fetcher.fetch_risk_free_rate(
            start_date=start_date,
            end_date=end_date,
            maturity="3m",
            frequency=frequency
        )
        
        print(f"   ✓ Fetched {len(rf_rates)} observations")
        print(f"   First rate: {rf_rates.iloc[0]*100:.2f}% (annualized)")
        print(f"   Last rate: {rf_rates.iloc[-1]*100:.2f}% (annualized)")
        
        # Convert to period returns based on frequency
        if frequency == "daily":
            # Daily return = annual rate / 252 trading days
            period_returns = rf_rates / 252
            print(f"   Daily return: {period_returns.iloc[-1]*100:.4f}%")
        elif frequency == "monthly":
            # Monthly return = (1 + annual_rate)^(1/12) - 1
            period_returns = (1 + rf_rates) ** (1/12) - 1
            print(f"   Monthly return: {period_returns.iloc[-1]*100:.2f}%")
        else:
            period_returns = rf_rates
            
        return period_returns
        
    except Exception as e:
        print(f"   ✗ Error fetching FRED data: {e}")
        
    # Method 2: Fallback to T-bill ETFs
    print("\n2. Fallback: Fetching T-bill ETF (BIL)...")
    try:
        fetcher = MarketDataFetcher()
        bil_data = fetcher.fetch_price_history(
            "BIL",
            start_date=start_date,
            end_date=end_date
        )
        
        if "BIL" in bil_data:
            prices = bil_data["BIL"]["Adj Close"]
            returns = prices.pct_change().dropna()
            
            print(f"   ✓ Fetched {len(returns)} daily returns")
            print(f"   Average daily return: {returns.mean()*100:.4f}%")
            print(f"   Annualized return: {((1 + returns.mean())**252 - 1)*100:.2f}%")
            
            return returns
    except Exception as e:
        print(f"   ✗ Error fetching ETF data: {e}")
        
    return None


def compare_methods():
    """Compare FRED rates vs ETF returns."""
    
    print("\n\nComparing FRED vs ETF Methods")
    print("=" * 50)
    
    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Get both
    fred_fetcher = FREDDataFetcher()
    market_fetcher = MarketDataFetcher()
    
    # FRED data
    fred_rates = fred_fetcher.fetch_risk_free_rate(
        start_date=start_date,
        end_date=end_date,
        maturity="3m",
        frequency="daily"
    )
    fred_daily_returns = fred_rates / 252  # Convert to daily
    
    # ETF data
    bil_data = market_fetcher.fetch_price_history(
        "BIL",
        start_date=start_date,
        end_date=end_date
    )
    
    if "BIL" in bil_data:
        bil_returns = bil_data["BIL"]["Adj Close"].pct_change().dropna()
        
        # Align dates
        common_dates = fred_daily_returns.index.intersection(bil_returns.index)
        
        if len(common_dates) > 0:
            fred_aligned = fred_daily_returns.loc[common_dates]
            bil_aligned = bil_returns.loc[common_dates]
            
            # Statistics
            print(f"\nComparison over {len(common_dates)} days:")
            print(f"FRED average daily return: {fred_aligned.mean()*100:.4f}%")
            print(f"BIL average daily return: {bil_aligned.mean()*100:.4f}%")
            print(f"Correlation: {fred_aligned.corr(bil_aligned):.4f}")
            
            # Annualized
            fred_annual = (1 + fred_aligned.mean())**252 - 1
            bil_annual = (1 + bil_aligned.mean())**252 - 1
            print(f"\nAnnualized returns:")
            print(f"FRED: {fred_annual*100:.2f}%")
            print(f"BIL: {bil_annual*100:.2f}%")


def demonstrate_rate_conversion():
    """Show the key insight: how to convert rate series to returns."""
    
    print("\n\nRate Series to Returns Conversion")
    print("=" * 50)
    
    # Example rate
    annual_rate = 0.0525  # 5.25% annual
    
    print(f"Annual rate: {annual_rate*100:.2f}%")
    print("\nConversions:")
    
    # Daily (simple)
    daily_simple = annual_rate / 252
    print(f"Daily (simple): {daily_simple*100:.4f}%")
    
    # Daily (compound) - more accurate but usually overkill
    daily_compound = (1 + annual_rate)**(1/252) - 1
    print(f"Daily (compound): {daily_compound*100:.4f}%")
    
    # Monthly
    monthly = (1 + annual_rate)**(1/12) - 1
    print(f"Monthly: {monthly*100:.3f}%")
    
    # Quarterly
    quarterly = (1 + annual_rate)**(1/4) - 1
    print(f"Quarterly: {quarterly*100:.2f}%")
    
    print("\nKey insight: FRED provides annualized rates,")
    print("we must convert to match our return frequency!")


if __name__ == "__main__":
    # Demonstrate fetching cash rate returns
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months
    
    returns = fetch_cash_rate_returns(start_date, end_date, "daily")
    
    # Compare methods
    compare_methods()
    
    # Show conversion logic
    demonstrate_rate_conversion()
