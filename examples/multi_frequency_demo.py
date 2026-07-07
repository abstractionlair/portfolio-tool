#!/usr/bin/env python
"""
Multi-Frequency Data Support Demo

Demonstrates the new multi-frequency data capabilities including:
- Proper return compounding across different frequencies
- Frequency-aware risk estimation  
- Optimal frequency selection for different analysis horizons
- Integration with EWMA risk models
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

# Import multi-frequency classes
from data.multi_frequency import (
    Frequency, ReturnCompounding, MultiFrequencyDataFetcher,
    FrequencyConverter, MultiFrequencyAnalyzer
)

# Import EWMA classes for integration demo
from optimization.ewma import EWMAEstimator, EWMAParameters


def create_realistic_multi_frequency_data():
    """Create realistic return data across different frequencies."""
    print("Creating realistic multi-frequency market data...")
    
    np.random.seed(42)
    
    # Start with high-frequency daily data
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    daily_dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create realistic market data with time-varying volatility
    n_days = len(daily_dates)
    
    # Market factor with volatility clustering
    market_vol = np.zeros(n_days)
    market_returns = np.zeros(n_days)
    
    # Simple GARCH-like process
    omega, alpha, beta = 0.000001, 0.08, 0.90
    market_vol[0] = 0.015
    
    for t in range(1, n_days):
        market_vol[t] = np.sqrt(
            omega + alpha * market_returns[t-1]**2 + beta * market_vol[t-1]**2
        )
        market_returns[t] = np.random.normal(0.0005, market_vol[t])
    
    # Create correlated assets
    assets = {
        'SPY': {'beta': 1.0, 'alpha': 0.0001, 'vol': 0.003},
        'TLT': {'beta': -0.25, 'alpha': 0.0002, 'vol': 0.008},
        'GLD': {'beta': -0.05, 'alpha': 0.0003, 'vol': 0.015},
        'VTI': {'beta': 0.98, 'alpha': 0.00005, 'vol': 0.0025}
    }
    
    daily_data = {}
    for asset, params in assets.items():
        idiosyncratic = np.random.normal(0, params['vol'], n_days)
        daily_data[asset] = (
            params['alpha'] + 
            params['beta'] * market_returns + 
            idiosyncratic
        )
    
    daily_returns = pd.DataFrame(daily_data, index=daily_dates)
    
    print(f"Generated {len(daily_returns)} days of data for {len(assets)} assets")
    return daily_returns


def demonstrate_frequency_conversion():
    """Demonstrate frequency conversion and return compounding."""
    print("\n" + "="*60)
    print("FREQUENCY CONVERSION DEMONSTRATION")
    print("="*60)
    
    daily_returns = create_realistic_multi_frequency_data()
    spy_returns = daily_returns['SPY']
    
    # Convert to different frequencies
    frequencies = [Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY, Frequency.QUARTERLY]
    converted_data = {}
    
    for freq in frequencies:
        converted = ReturnCompounding.resample_returns(spy_returns, freq)
        converted_data[freq.value] = converted
        
        print(f"\n{freq.value.upper()} FREQUENCY:")
        print(f"  Periods: {len(converted)}")
        print(f"  Mean Return: {converted.mean():.4f}")
        print(f"  Volatility: {converted.std():.4f}")
        print(f"  Annualized Return: {converted.mean() * freq.annualization_factor:.2%}")
        print(f"  Annualized Volatility: {converted.std() * np.sqrt(freq.annualization_factor):.2%}")
    
    return converted_data


def demonstrate_frequency_statistics():
    """Demonstrate frequency-aware statistics calculation."""
    print("\n" + "="*60)
    print("FREQUENCY-AWARE STATISTICS")
    print("="*60)
    
    daily_returns = create_realistic_multi_frequency_data()
    fetcher = MultiFrequencyDataFetcher()
    
    # Calculate statistics for different frequencies
    spy_returns = daily_returns['SPY']
    
    print("SPY Statistics by Frequency:")
    print("-" * 40)
    
    frequencies = [Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY, Frequency.QUARTERLY]
    
    for freq in frequencies:
        # Convert returns to frequency
        freq_returns = ReturnCompounding.resample_returns(spy_returns, freq)
        
        # Get frequency-specific statistics
        stats = fetcher.get_frequency_statistics(freq_returns, freq)
        
        print(f"\n{freq.value.upper()}:")
        print(f"  Periods: {stats.get('periods', 0)}")
        print(f"  Mean Return: {stats.get('mean_return', 0):.4f}")
        print(f"  Volatility: {stats.get('volatility', 0):.4f}")
        print(f"  Annualized Return: {stats.get('annualized_return', 0):.2%}")
        print(f"  Annualized Volatility: {stats.get('annualized_volatility', 0):.2%}")
        print(f"  Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}")
        
        if 'max_drawdown' in stats:
            print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")


def demonstrate_ewma_multi_frequency():
    """Demonstrate EWMA estimation with different frequencies."""
    print("\n" + "="*60)
    print("EWMA WITH MULTI-FREQUENCY DATA")
    print("="*60)
    
    daily_returns = create_realistic_multi_frequency_data()
    
    # Test EWMA with different frequencies
    frequencies = [Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY]
    
    ewma_estimator = EWMAEstimator(EWMAParameters(lambda_=0.94))
    
    print("EWMA Volatility Estimates by Frequency:")
    print("-" * 45)
    
    for freq in frequencies:
        print(f"\n{freq.value.upper()} FREQUENCY:")
        
        for asset in daily_returns.columns:
            # Convert to target frequency
            asset_returns = ReturnCompounding.resample_returns(
                daily_returns[asset], freq
            )
            
            if len(asset_returns) > 30:  # Ensure enough data
                try:
                    # Estimate EWMA volatility
                    ewma_vol = ewma_estimator.estimate_volatility(
                        asset_returns, annualize=True, frequency=freq
                    )
                    
                    current_vol = ewma_vol.iloc[-1]
                    print(f"  {asset}: {current_vol:.2%}")
                    
                except Exception as e:
                    print(f"  {asset}: Failed ({str(e)[:30]}...)")
            else:
                print(f"  {asset}: Insufficient data")


def demonstrate_optimal_frequency_selection():
    """Demonstrate optimal frequency selection for different analysis horizons."""
    print("\n" + "="*60)
    print("OPTIMAL FREQUENCY SELECTION")
    print("="*60)
    
    # Test different scenarios
    scenarios = [
        {"data_length": 1000, "horizon": timedelta(days=30), "description": "Short-term analysis, lots of data"},
        {"data_length": 100, "horizon": timedelta(days=30), "description": "Short-term analysis, limited data"},
        {"data_length": 500, "horizon": timedelta(days=180), "description": "Medium-term analysis, medium data"},
        {"data_length": 50, "horizon": timedelta(days=365), "description": "Long-term analysis, limited data"},
        {"data_length": 200, "horizon": timedelta(days=730), "description": "Very long-term analysis"},
    ]
    
    print("Frequency Recommendations:")
    print("-" * 40)
    
    for scenario in scenarios:
        optimal_freq = FrequencyConverter.get_optimal_frequency(
            scenario["data_length"], scenario["horizon"]
        )
        
        print(f"\nScenario: {scenario['description']}")
        print(f"  Data Length: {scenario['data_length']} periods")
        print(f"  Analysis Horizon: {scenario['horizon'].days} days")
        print(f"  Recommended Frequency: {optimal_freq.value.upper()}")
        print(f"  Annualization Factor: {optimal_freq.annualization_factor}")


def demonstrate_frequency_impact_analysis():
    """Demonstrate analysis of how frequency affects portfolio metrics."""
    print("\n" + "="*60)
    print("FREQUENCY IMPACT ANALYSIS")  
    print("="*60)
    
    daily_returns = create_realistic_multi_frequency_data()
    
    # Create mock analyzer for demonstration
    class MockFetcher:
        def fetch_returns(self, tickers, start_date, end_date, frequency):
            freq_returns = {}
            for ticker in tickers:
                if ticker in daily_returns.columns:
                    converted = ReturnCompounding.resample_returns(
                        daily_returns[ticker], frequency
                    )
                    freq_returns[ticker] = converted
            return pd.DataFrame(freq_returns)
        
        def get_frequency_statistics(self, returns, frequency):
            if returns.empty:
                return {}
            return {
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'annualized_return': returns.mean() * frequency.annualization_factor,
                'annualized_volatility': returns.std() * np.sqrt(frequency.annualization_factor),
                'sharpe_ratio': (returns.mean() * frequency.annualization_factor - 0.02) / 
                               (returns.std() * np.sqrt(frequency.annualization_factor)),
                'periods': len(returns)
            }
    
    analyzer = MultiFrequencyAnalyzer(MockFetcher())
    
    # Analyze impact across frequencies
    tickers = ['SPY', 'TLT', 'GLD']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    results = analyzer.analyze_frequency_impact(
        tickers, start_date, end_date,
        frequencies=[Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY, Frequency.QUARTERLY]
    )
    
    print("Frequency Impact Analysis Results:")
    print("-" * 40)
    
    # Display results in a formatted way
    for ticker in tickers:
        print(f"\n{ticker}:")
        ticker_results = results[results['ticker'] == ticker]
        
        for _, row in ticker_results.iterrows():
            freq = row['frequency']
            ann_ret = row.get('annualized_return', 0)
            ann_vol = row.get('annualized_volatility', 0)
            sharpe = row.get('sharpe_ratio', 0)
            periods = row.get('periods', 0)
            
            print(f"  {freq.upper():>9}: Return={ann_ret:>6.1%} Vol={ann_vol:>6.1%} Sharpe={sharpe:>5.2f} N={periods:>3}")


def demonstrate_rebalancing_frequency_optimization():
    """Demonstrate optimal rebalancing frequency selection."""
    print("\n" + "="*60)
    print("OPTIMAL REBALANCING FREQUENCY")
    print("="*60)
    
    daily_returns = create_realistic_multi_frequency_data()
    
    # Create a simple equal-weight portfolio
    portfolio_weights = np.array([0.25, 0.25, 0.25, 0.25])
    portfolio_returns = (daily_returns * portfolio_weights).sum(axis=1)
    
    # Mock analyzer for rebalancing analysis
    class MockAnalyzer:
        def find_optimal_rebalancing_frequency(self, returns, frequencies):
            results = {}
            
            for freq in frequencies:
                # Convert returns to frequency
                freq_returns = ReturnCompounding.resample_returns(returns, freq)
                
                if len(freq_returns) > 0:
                    ann_return = freq_returns.mean() * freq.annualization_factor
                    ann_vol = freq_returns.std() * np.sqrt(freq.annualization_factor)
                    
                    # Estimate transaction costs
                    rebalances = len(freq_returns)
                    turnover_per_rebalance = 0.15  # 15% turnover
                    transaction_cost_bps = 5      # 5 bps cost
                    
                    total_costs = rebalances * turnover_per_rebalance * (transaction_cost_bps / 10000)
                    net_return = ann_return - total_costs
                    
                    net_sharpe = (net_return - 0.02) / ann_vol if ann_vol > 0 else 0
                    
                    results[freq] = {
                        'gross_return': ann_return,
                        'net_return': net_return,
                        'volatility': ann_vol,
                        'net_sharpe': net_sharpe,
                        'rebalances': rebalances,
                        'total_costs': total_costs
                    }
            
            # Find optimal frequency
            if results:
                optimal = max(results.keys(), key=lambda f: results[f]['net_sharpe'])
                return optimal, results[optimal]
            return Frequency.MONTHLY, {}
    
    analyzer = MockAnalyzer()
    
    optimal_freq, metrics = analyzer.find_optimal_rebalancing_frequency(
        portfolio_returns,
        frequencies=[Frequency.WEEKLY, Frequency.MONTHLY, Frequency.QUARTERLY]
    )
    
    print(f"Optimal Rebalancing Frequency: {optimal_freq.value.upper()}")
    print("\nOptimal Frequency Metrics:")
    for key, value in metrics.items():
        if 'return' in key or 'sharpe' in key:
            print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
        elif 'costs' in key:
            print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")


def create_frequency_comparison_chart():
    """Create visualization comparing different frequencies."""
    print("\n" + "="*60)
    print("CREATING FREQUENCY COMPARISON CHART")
    print("="*60)
    
    daily_returns = create_realistic_multi_frequency_data()
    spy_returns = daily_returns['SPY']
    
    # Convert to different frequencies
    weekly_returns = ReturnCompounding.resample_returns(spy_returns, Frequency.WEEKLY)
    monthly_returns = ReturnCompounding.resample_returns(spy_returns, Frequency.MONTHLY)
    quarterly_returns = ReturnCompounding.resample_returns(spy_returns, Frequency.QUARTERLY)
    
    # Calculate cumulative returns for each frequency
    daily_cumulative = (1 + spy_returns).cumprod()
    weekly_cumulative = (1 + weekly_returns).cumprod()
    monthly_cumulative = (1 + monthly_returns).cumprod()
    quarterly_cumulative = (1 + quarterly_returns).cumprod()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Cumulative returns comparison
    ax1 = axes[0, 0]
    ax1.plot(daily_cumulative.index, daily_cumulative.values, 
             label='Daily', alpha=0.8, linewidth=1)
    ax1.plot(weekly_cumulative.index, weekly_cumulative.values, 
             label='Weekly', alpha=0.9, linewidth=1.5)
    ax1.plot(monthly_cumulative.index, monthly_cumulative.values, 
             label='Monthly', alpha=0.9, linewidth=2)
    ax1.plot(quarterly_cumulative.index, quarterly_cumulative.values, 
             label='Quarterly', alpha=0.9, linewidth=2.5)
    
    ax1.set_title('Cumulative Returns by Frequency')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Return distributions
    ax2 = axes[0, 1]
    ax2.hist(spy_returns.dropna(), bins=50, alpha=0.7, density=True, label='Daily')
    ax2.hist(weekly_returns.dropna(), bins=30, alpha=0.7, density=True, label='Weekly')
    ax2.hist(monthly_returns.dropna(), bins=20, alpha=0.7, density=True, label='Monthly')
    
    ax2.set_title('Return Distributions by Frequency')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling volatility comparison
    ax3 = axes[1, 0]
    
    # Calculate rolling volatilities (annualized)
    daily_vol = spy_returns.rolling(60).std() * np.sqrt(252)
    weekly_vol = weekly_returns.rolling(12).std() * np.sqrt(52)
    monthly_vol = monthly_returns.rolling(6).std() * np.sqrt(12)
    
    ax3.plot(daily_vol.index, daily_vol.values, label='Daily (60d window)', alpha=0.8)
    ax3.plot(weekly_vol.index, weekly_vol.values, label='Weekly (12w window)', alpha=0.9)
    ax3.plot(monthly_vol.index, monthly_vol.values, label='Monthly (6m window)', alpha=0.9)
    
    ax3.set_title('Rolling Volatility by Frequency (Annualized)')
    ax3.set_ylabel('Volatility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Frequency statistics comparison
    ax4 = axes[1, 1]
    
    frequencies = ['Daily', 'Weekly', 'Monthly', 'Quarterly']
    returns_data = [spy_returns, weekly_returns, monthly_returns, quarterly_returns]
    
    ann_returns = []
    ann_vols = []
    sharpe_ratios = []
    
    for i, (freq_name, freq_returns) in enumerate(zip(frequencies, returns_data)):
        freq_enum = [Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY, Frequency.QUARTERLY][i]
        
        ann_ret = freq_returns.mean() * freq_enum.annualization_factor
        ann_vol = freq_returns.std() * np.sqrt(freq_enum.annualization_factor)
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
        
        ann_returns.append(ann_ret)
        ann_vols.append(ann_vol)
        sharpe_ratios.append(sharpe)
    
    x = np.arange(len(frequencies))
    width = 0.25
    
    ax4.bar(x - width, ann_returns, width, label='Ann. Return', alpha=0.8)
    ax4.bar(x, ann_vols, width, label='Ann. Volatility', alpha=0.8)
    ax4.bar(x + width, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.8)
    
    ax4.set_title('Summary Statistics by Frequency')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(frequencies)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_frequency_analysis.png', dpi=300, bbox_inches='tight')
    print("Multi-frequency analysis chart saved as 'multi_frequency_analysis.png'")
    
    # Print summary statistics
    print("\nSummary Statistics by Frequency:")
    print("-" * 50)
    print(f"{'Frequency':<12} {'Ann. Return':<12} {'Ann. Vol':<12} {'Sharpe':<8} {'Periods':<8}")
    print("-" * 50)
    
    for i, freq_name in enumerate(frequencies):
        print(f"{freq_name:<12} {ann_returns[i]:<12.2%} {ann_vols[i]:<12.2%} "
              f"{sharpe_ratios[i]:<8.3f} {len(returns_data[i]):<8}")


def main():
    """Run the complete multi-frequency demonstration."""
    print("Multi-Frequency Data Support Demo")
    print("="*60)
    print("Comprehensive demonstration of multi-frequency data capabilities")
    print("for portfolio optimization and risk analysis.")
    print()
    
    try:
        # Run all demonstrations
        demonstrate_frequency_conversion()
        demonstrate_frequency_statistics()
        demonstrate_ewma_multi_frequency()
        demonstrate_optimal_frequency_selection()
        demonstrate_frequency_impact_analysis()
        demonstrate_rebalancing_frequency_optimization()
        create_frequency_comparison_chart()
        
        print("\n" + "="*60)
        print("🎉 MULTI-FREQUENCY DEMO COMPLETED!")
        print("="*60)
        print("\n✅ Key Multi-Frequency Features Demonstrated:")
        print("  • Proper return compounding across frequencies")
        print("  • Frequency-aware risk estimation with EWMA")
        print("  • Optimal frequency selection for different horizons")
        print("  • Impact analysis of frequency on portfolio metrics")
        print("  • Optimal rebalancing frequency determination")
        print("  • Visual comparison of frequency effects")
        print("\n📊 Charts Generated:")
        print("  • multi_frequency_analysis.png")
        print("\n🔧 Multi-frequency support enables:")
        print("   - More appropriate analysis for different time horizons")
        print("   - Better risk estimation for various rebalancing schedules")
        print("   - Optimal frequency selection based on data and objectives")
        print("   - Professional-grade multi-timeframe portfolio analysis")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()