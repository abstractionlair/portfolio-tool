#!/usr/bin/env python
"""
EWMA Standalone Demo

Focuses on demonstrating the core EWMA and GARCH functionality
without complex integration dependencies.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import EWMA classes directly
from optimization.ewma import EWMAEstimator, EWMAParameters, GARCHEstimator


def create_realistic_data():
    """Create realistic return data with volatility clustering."""
    print("Creating realistic market data with volatility clustering...")
    
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
    
    # Create market factor with time-varying volatility (volatility clustering)
    market_vol = np.zeros(n_days)
    market_returns = np.zeros(n_days)
    
    # GARCH-like process for market volatility
    omega, alpha, beta = 0.000001, 0.1, 0.85
    market_vol[0] = 0.01
    
    for t in range(1, n_days):
        market_vol[t] = np.sqrt(omega + alpha * market_returns[t-1]**2 + beta * market_vol[t-1]**2)
        market_returns[t] = np.random.normal(0.0005, market_vol[t])
    
    # Create correlated asset returns
    assets_data = {}
    correlations = {'SPY': 1.0, 'TLT': -0.3, 'GLD': -0.1, 'VTI': 0.95}
    idiosync_vols = {'SPY': 0.005, 'TLT': 0.008, 'GLD': 0.012, 'VTI': 0.004}
    
    for asset, corr in correlations.items():
        idiosyncratic = np.random.normal(0, idiosync_vols[asset], n_days)
        assets_data[asset] = corr * market_returns + idiosyncratic
    
    returns_df = pd.DataFrame(assets_data, index=dates)
    
    print(f"Generated {n_days} days of data for {len(assets_data)} assets")
    print("Assets:", list(assets_data.keys()))
    return returns_df


def demonstrate_ewma_vs_sample():
    """Compare EWMA vs sample covariance estimation."""
    print("\n" + "="*60)
    print("EWMA vs SAMPLE COVARIANCE COMPARISON")
    print("="*60)
    
    returns_df = create_realistic_data()
    
    # EWMA estimation
    ewma_estimator = EWMAEstimator(EWMAParameters(lambda_=0.94))
    ewma_cov = ewma_estimator.estimate_covariance_matrix(returns_df)
    
    # Sample covariance
    sample_cov = returns_df.cov() * 252  # Annualized
    
    print("\nLatest EWMA Volatilities:")
    ewma_vols = np.sqrt(np.diag(ewma_cov))
    for i, asset in enumerate(returns_df.columns):
        print(f"  {asset}: {ewma_vols[i]:.2%}")
    
    print("\nSample Volatilities:")
    sample_vols = np.sqrt(np.diag(sample_cov))
    for i, asset in enumerate(returns_df.columns):
        print(f"  {asset}: {sample_vols[i]:.2%}")
    
    print("\nVolatility Differences (EWMA - Sample):")
    for i, asset in enumerate(returns_df.columns):
        diff = ewma_vols[i] - sample_vols[i]
        print(f"  {asset}: {diff:+.2%}")
    
    return returns_df, ewma_cov, sample_cov


def demonstrate_parameter_sensitivity():
    """Show how different EWMA parameters affect estimates."""
    print("\n" + "="*60)
    print("EWMA PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    returns_df = create_realistic_data()
    spy_returns = returns_df['SPY']
    
    # Test different lambda values
    lambdas = [0.90, 0.94, 0.97, 0.99]
    
    print("Effect of Lambda Parameter on SPY Volatility:")
    print("Lambda | Current Vol | 1-Month Ago | Responsiveness")
    print("-" * 55)
    
    for lambda_val in lambdas:
        estimator = EWMAEstimator(EWMAParameters(lambda_=lambda_val))
        vol_series = estimator.estimate_volatility(spy_returns)
        
        current_vol = vol_series.iloc[-1]
        month_ago_vol = vol_series.iloc[-22] if len(vol_series) > 22 else vol_series.iloc[0]
        responsiveness = abs(current_vol - month_ago_vol) / month_ago_vol
        
        print(f"{lambda_val:5.2f} | {current_vol:10.2%} | {month_ago_vol:11.2%} | {responsiveness:12.2%}")


def demonstrate_forecasting():
    """Show volatility forecasting capabilities."""
    print("\n" + "="*60)
    print("VOLATILITY FORECASTING")
    print("="*60)
    
    returns_df = create_realistic_data()
    spy_returns = returns_df['SPY']
    
    # EWMA forecasting
    ewma_estimator = EWMAEstimator()
    
    print("EWMA Volatility Forecasts for SPY:")
    horizons = [1, 5, 22, 66, 252]  # 1 day, 1 week, 1 month, 3 months, 1 year
    
    for horizon in horizons:
        forecast = ewma_estimator.forecast_volatility(
            spy_returns, 
            horizon=horizon, 
            method='simple'
        )
        
        # Convert horizon to descriptive text
        if horizon == 1:
            period = "1 day"
        elif horizon == 5:
            period = "1 week"
        elif horizon == 22:
            period = "1 month"
        elif horizon == 66:
            period = "3 months"
        else:
            period = "1 year"
        
        print(f"  {period:>8}: {forecast:.2%}")
    
    # GARCH forecasting
    print("\nGARCH Volatility Forecasts for SPY:")
    garch_estimator = GARCHEstimator()
    
    try:
        variance_forecasts = garch_estimator.forecast_variance(spy_returns, horizon=22)
        vol_forecasts = np.sqrt(variance_forecasts)
        
        print(f"  Day 1:   {vol_forecasts[0]:.2%}")
        print(f"  Day 5:   {vol_forecasts[4]:.2%}")
        print(f"  Day 22:  {vol_forecasts[21]:.2%}")
        
    except Exception as e:
        print(f"  GARCH forecasting failed: {e}")


def create_volatility_comparison_plot():
    """Create a plot comparing different volatility estimation methods."""
    print("\n" + "="*60)
    print("CREATING VOLATILITY COMPARISON PLOT")
    print("="*60)
    
    returns_df = create_realistic_data()
    spy_returns = returns_df['SPY']
    
    # Calculate different volatility estimates
    # 1. Rolling sample volatility (30-day)
    rolling_vol_30 = spy_returns.rolling(30).std() * np.sqrt(252)
    
    # 2. Rolling sample volatility (60-day)
    rolling_vol_60 = spy_returns.rolling(60).std() * np.sqrt(252)
    
    # 3. EWMA volatilities with different lambdas
    ewma_94 = EWMAEstimator(EWMAParameters(lambda_=0.94))
    ewma_vol_94 = ewma_94.estimate_volatility(spy_returns)
    
    ewma_97 = EWMAEstimator(EWMAParameters(lambda_=0.97))
    ewma_vol_97 = ewma_97.estimate_volatility(spy_returns)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Returns
    plt.subplot(3, 1, 1)
    plt.plot(returns_df.index, spy_returns * 100, alpha=0.7, color='black', linewidth=0.5)
    plt.title('SPY Daily Returns (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Volatility comparison
    plt.subplot(3, 1, 2)
    plt.plot(returns_df.index, rolling_vol_30 * 100, label='30-Day Rolling', alpha=0.8, linewidth=1.5)
    plt.plot(returns_df.index, rolling_vol_60 * 100, label='60-Day Rolling', alpha=0.8, linewidth=1.5)
    plt.plot(returns_df.index, ewma_vol_94 * 100, label='EWMA (λ=0.94)', alpha=0.9, linewidth=2)
    plt.plot(returns_df.index, ewma_vol_97 * 100, label='EWMA (λ=0.97)', alpha=0.9, linewidth=2)
    
    plt.title('Volatility Estimates Comparison (Annualized %)', fontsize=12, fontweight='bold')
    plt.ylabel('Volatility (%)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Volatility differences
    plt.subplot(3, 1, 3)
    diff_94_vs_60 = (ewma_vol_94 - rolling_vol_60) * 100
    diff_97_vs_60 = (ewma_vol_97 - rolling_vol_60) * 100
    
    plt.plot(returns_df.index, diff_94_vs_60, label='EWMA(0.94) - 60Day', alpha=0.8)
    plt.plot(returns_df.index, diff_97_vs_60, label='EWMA(0.97) - 60Day', alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.title('Volatility Differences from 60-Day Rolling (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Difference (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ewma_volatility_analysis.png', dpi=300, bbox_inches='tight')
    print("Volatility analysis chart saved as 'ewma_volatility_analysis.png'")
    
    # Print summary statistics
    print(f"\nVolatility Summary (Last Value):")
    print(f"30-Day Rolling:   {rolling_vol_30.iloc[-1]:.2%}")
    print(f"60-Day Rolling:   {rolling_vol_60.iloc[-1]:.2%}")
    print(f"EWMA (λ=0.94):    {ewma_vol_94.iloc[-1]:.2%}")
    print(f"EWMA (λ=0.97):    {ewma_vol_97.iloc[-1]:.2%}")


def demonstrate_correlation_dynamics():
    """Show how EWMA captures changing correlations."""
    print("\n" + "="*60)
    print("CORRELATION DYNAMICS WITH EWMA")
    print("="*60)
    
    returns_df = create_realistic_data()
    
    # Calculate EWMA and sample correlations
    ewma_estimator = EWMAEstimator()
    ewma_corr = ewma_estimator.estimate_correlation_matrix(returns_df)
    sample_corr = returns_df.corr()
    
    print("EWMA Correlation Matrix (Latest):")
    print(ewma_corr.round(3))
    
    print("\nSample Correlation Matrix (Full Period):")
    print(sample_corr.round(3))
    
    print("\nCorrelation Differences (EWMA - Sample):")
    corr_diff = ewma_corr - sample_corr
    print(corr_diff.round(3))
    
    # Focus on SPY-TLT correlation (stocks vs bonds)
    print(f"\nSPY-TLT Correlation Analysis:")
    print(f"  EWMA (latest):     {ewma_corr.loc['SPY', 'TLT']:.3f}")
    print(f"  Sample (average):  {sample_corr.loc['SPY', 'TLT']:.3f}")
    print(f"  Difference:        {ewma_corr.loc['SPY', 'TLT'] - sample_corr.loc['SPY', 'TLT']:+.3f}")


def main():
    """Run the complete EWMA standalone demonstration."""
    print("EWMA Standalone Demo")
    print("="*60)
    print("Advanced risk estimation using EWMA and GARCH models")
    print("Focused on core functionality without complex dependencies")
    print()
    
    try:
        # Run core demonstrations
        demonstrate_ewma_vs_sample()
        demonstrate_parameter_sensitivity()
        demonstrate_forecasting()
        demonstrate_correlation_dynamics()
        create_volatility_comparison_plot()
        
        print("\n" + "="*60)
        print("🎉 EWMA STANDALONE DEMO COMPLETED!")
        print("="*60)
        print("\n✅ Key EWMA Features Demonstrated:")
        print("  • EWMA variance and covariance estimation")
        print("  • Parameter sensitivity analysis")
        print("  • Volatility forecasting (EWMA & GARCH)")
        print("  • Correlation dynamics")
        print("  • Comparison with traditional methods")
        print("\n📊 Charts Generated:")
        print("  • ewma_volatility_analysis.png")
        print("\n🔧 EWMA provides more responsive risk estimates")
        print("   that adapt quickly to changing market conditions!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()