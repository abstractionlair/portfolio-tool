#!/usr/bin/env python
"""
Example of decomposing returns into inflation, real risk-free rate, and spread components.

This demonstrates how to:
1. Fetch total returns for assets
2. Decompose returns into fundamental components
3. Analyze the contribution of each component
4. Compare decompositions across different assets
"""

import logging
from datetime import datetime
import pandas as pd

from src.data import (
    ReturnDecomposer,
    TotalReturnFetcher,
    FREDDataFetcher,
    ExposureUniverse
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run return decomposition example."""
    # Initialize components
    decomposer = ReturnDecomposer()
    total_return_fetcher = TotalReturnFetcher()
    
    # Define time period
    end_date = datetime.now()
    start_date = datetime(2019, 1, 1)  # 5 years of data
    
    # Example 1: Decompose returns for a single asset (S&P 500)
    print("\n" + "="*80)
    print("Example 1: S&P 500 Return Decomposition")
    print("="*80 + "\n")
    
    # Fetch S&P 500 returns
    spy_data = total_return_fetcher.fetch_total_returns(
        'SPY', start_date, end_date, frequency='monthly'
    )
    
    if spy_data and spy_data['success']:
        spy_returns = spy_data['returns']
        
        # Decompose returns
        spy_decomposition = decomposer.decompose_returns(
            spy_returns,
            start_date,
            end_date,
            frequency='monthly',
            inflation_series='cpi_all',
            risk_free_maturity='3m'
        )
        
        # Get summary
        spy_summary = decomposer.get_decomposition_summary(spy_decomposition, annualize=True)
        
        print("S&P 500 (SPY) Return Decomposition (Annualized):")
        print(f"  Total Return: {spy_summary['total_return']:.2%}")
        print(f"  - Inflation: {spy_summary['inflation']:.2%}")
        print(f"  - Real Risk-Free Rate: {spy_summary['real_rf_rate']:.2%}")
        print(f"  - Equity Risk Premium: {spy_summary['spread']:.2%}")
        print(f"\nThe equity risk premium represents {spy_summary.get('spread_pct', 0):.1f}% of total return")
        
        # Show first few rows of decomposition
        print("\nFirst 5 months of decomposition:")
        print(spy_decomposition[['total_return', 'inflation', 'real_rf_rate', 'spread']].head())
    
    # Example 2: Compare decompositions across asset classes
    print("\n" + "="*80)
    print("Example 2: Multi-Asset Return Decomposition Comparison")
    print("="*80 + "\n")
    
    # Define assets to analyze
    assets = {
        'SPY': 'S&P 500 (Equities)',
        'TLT': '20+ Year Treasuries',
        'GLD': 'Gold',
        'VNQ': 'Real Estate'
    }
    
    # Fetch returns for all assets
    asset_returns = {}
    for ticker, name in assets.items():
        data = total_return_fetcher.fetch_total_returns(
            ticker, start_date, end_date, frequency='monthly'
        )
        if data and data['success']:
            asset_returns[ticker] = data['returns']
    
    # Create DataFrame of returns
    returns_df = pd.DataFrame(asset_returns)
    
    # Decompose all assets
    decompositions = decomposer.decompose_portfolio_returns(
        returns_df,
        start_date,
        end_date,
        frequency='monthly'
    )
    
    # Create comparison table
    comparison_data = []
    for ticker, decomp_df in decompositions.items():
        summary = decomposer.get_decomposition_summary(decomp_df, annualize=True)
        comparison_data.append({
            'Asset': f"{assets[ticker]} ({ticker})",
            'Total Return': f"{summary['total_return']:.2%}",
            'Inflation': f"{summary['inflation']:.2%}",
            'Real RF Rate': f"{summary['real_rf_rate']:.2%}",
            'Risk Premium': f"{summary['spread']:.2%}",
            'Premium %': f"{summary.get('spread_pct', 0):.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("Return Decomposition Comparison (Annualized):")
    print(comparison_df.to_string(index=False))
    
    # Example 3: Decompose entire exposure universe
    print("\n" + "="*80)
    print("Example 3: Exposure Universe Return Decomposition")
    print("="*80 + "\n")
    
    # Load exposure universe
    universe = ExposureUniverse()
    universe.load_config('/Users/scottmcguire/portfolio-tool/config/exposure_universe.yaml')
    
    # Decompose returns for all exposures
    universe_decompositions = decomposer.decompose_universe_returns(
        universe,
        start_date,
        end_date,
        frequency='monthly'
    )
    
    # Create report
    report = decomposer.create_decomposition_report(universe_decompositions)
    
    print("Exposure Universe Decomposition Summary:")
    print(report.to_string(index=False))
    
    # Example 4: Analyze decomposition over different time periods
    print("\n" + "="*80)
    print("Example 4: Time Period Analysis - SPY")
    print("="*80 + "\n")
    
    periods = [
        ("Last 1 Year", datetime(2024, 1, 1)),
        ("Last 3 Years", datetime(2022, 1, 1)),
        ("Last 5 Years", datetime(2020, 1, 1)),
        ("Since COVID", datetime(2020, 3, 1))
    ]
    
    period_results = []
    for period_name, period_start in periods:
        # Fetch returns
        data = total_return_fetcher.fetch_total_returns(
            'SPY', period_start, end_date, frequency='monthly'
        )
        
        if data and data['success']:
            # Decompose
            decomp = decomposer.decompose_returns(
                data['returns'],
                period_start,
                end_date,
                frequency='monthly'
            )
            
            # Get summary
            summary = decomposer.get_decomposition_summary(decomp, annualize=True)
            
            period_results.append({
                'Period': period_name,
                'Total Return': f"{summary['total_return']:.2%}",
                'Inflation': f"{summary['inflation']:.2%}",
                'Real RF': f"{summary['real_rf_rate']:.2%}",
                'Risk Premium': f"{summary['spread']:.2%}"
            })
    
    period_df = pd.DataFrame(period_results)
    print("S&P 500 Decomposition Across Different Time Periods:")
    print(period_df.to_string(index=False))
    
    # Example 5: Different inflation measures
    print("\n" + "="*80)
    print("Example 5: Impact of Different Inflation Measures")
    print("="*80 + "\n")
    
    inflation_measures = {
        'cpi_all': 'CPI-U All Items',
        'cpi_core': 'CPI-U Core',
        'pce': 'PCE Price Index',
        'pce_core': 'PCE Core'
    }
    
    # Use TLT (bonds) as they're more sensitive to inflation measures
    tlt_data = total_return_fetcher.fetch_total_returns(
        'TLT', start_date, end_date, frequency='monthly'
    )
    
    if tlt_data and tlt_data['success']:
        inflation_comparison = []
        
        for measure_code, measure_name in inflation_measures.items():
            try:
                decomp = decomposer.decompose_returns(
                    tlt_data['returns'],
                    start_date,
                    end_date,
                    frequency='monthly',
                    inflation_series=measure_code
                )
                
                summary = decomposer.get_decomposition_summary(decomp, annualize=True)
                
                inflation_comparison.append({
                    'Inflation Measure': measure_name,
                    'Avg Inflation': f"{summary['inflation']:.2%}",
                    'Real Return': f"{summary['total_return'] - summary['inflation']:.2%}",
                    'Real RF Rate': f"{summary['real_rf_rate']:.2%}"
                })
            except Exception as e:
                logger.warning(f"Could not use {measure_name}: {e}")
        
        inflation_df = pd.DataFrame(inflation_comparison)
        print("TLT Return Decomposition with Different Inflation Measures:")
        print(inflation_df.to_string(index=False))


if __name__ == "__main__":
    main()
