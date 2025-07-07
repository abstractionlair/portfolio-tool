#!/usr/bin/env python
"""
Simple Portfolio Visualization Demo

Demonstrates core visualization capabilities without complex dependencies.
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

# Import visualization classes
from visualization import (
    PerformanceVisualizer, 
    AllocationVisualizer,
    OptimizationVisualizer, 
    DecompositionVisualizer
)

# Simple OptimizationResult for demo
class OptimizationResult:
    def __init__(self, weights, expected_return, volatility, sharpe_ratio, metadata=None):
        self.weights = weights
        self.expected_return = expected_return
        self.volatility = volatility
        self.sharpe_ratio = sharpe_ratio
        self.metadata = metadata or {}


def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data...")
    
    # Sample portfolio returns
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate different asset returns with realistic correlations
    n_days = len(dates)
    
    # Market factor
    market_factor = np.random.normal(0.0008, 0.015, n_days)
    
    # Asset-specific returns
    portfolio_returns = market_factor + np.random.normal(0.0002, 0.008, n_days)
    sp500_returns = market_factor + np.random.normal(0.0001, 0.003, n_days)
    bond_returns = np.random.normal(0.0003, 0.006, n_days) * 0.3 + market_factor * 0.1
    gold_returns = np.random.normal(0.0005, 0.018, n_days) - market_factor * 0.2
    
    returns_df = pd.DataFrame({
        'Portfolio': portfolio_returns,
        'S&P 500': sp500_returns,
        'Bonds': bond_returns,
        'Gold': gold_returns
    }, index=dates)
    
    # Sample portfolio weights
    weights = pd.Series({
        'SPY': 0.40,
        'AGG': 0.30,
        'GLD': 0.15,
        'VNQ': 0.10,
        'TLT': 0.05
    })
    
    # Sample optimization results
    opt_results = {
        'Equal Weight': OptimizationResult(
            weights=pd.Series({'SPY': 0.25, 'AGG': 0.25, 'GLD': 0.25, 'VNQ': 0.25}),
            expected_return=0.08,
            volatility=0.12,
            sharpe_ratio=0.67
        ),
        'Risk Parity': OptimizationResult(
            weights=pd.Series({'SPY': 0.35, 'AGG': 0.45, 'GLD': 0.15, 'VNQ': 0.05}),
            expected_return=0.075,
            volatility=0.10,
            sharpe_ratio=0.75
        ),
        'Max Sharpe': OptimizationResult(
            weights=pd.Series({'SPY': 0.55, 'AGG': 0.20, 'GLD': 0.10, 'VNQ': 0.15}),
            expected_return=0.095,
            volatility=0.14,
            sharpe_ratio=0.68
        )
    }
    
    # Sample decomposition data
    inflation = np.random.normal(0.00015, 0.003, len(dates))
    real_rf = np.random.normal(0.00008, 0.001, len(dates))
    
    decomposition = pd.DataFrame({
        'total_return': portfolio_returns,
        'inflation': inflation,
        'real_rf_rate': real_rf,
        'spread': portfolio_returns - inflation - real_rf
    }, index=dates)
    
    return returns_df, weights, opt_results, decomposition


def demo_performance_visualization():
    """Demonstrate performance visualization capabilities."""
    print("\n" + "="*60)
    print("PERFORMANCE VISUALIZATION DEMO")
    print("="*60)
    
    # Create sample data
    returns_df, _, _, _ = create_sample_data()
    
    # Initialize visualizer
    perf_viz = PerformanceVisualizer()
    
    # 1. Cumulative returns chart
    print("\n1. Creating cumulative returns chart...")
    fig1 = perf_viz.plot_cumulative_returns(
        returns_df,
        title="Portfolio Performance Comparison",
        benchmark=returns_df['S&P 500']
    )
    fig1.savefig('demo_cumulative_returns.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_cumulative_returns.png")
    
    # 2. Rolling metrics
    print("\n2. Creating rolling metrics chart...")
    fig2 = perf_viz.plot_rolling_metrics(
        returns_df['Portfolio'],
        metrics=['volatility', 'sharpe', 'drawdown'],
        window=252
    )
    fig2.savefig('demo_rolling_metrics.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_rolling_metrics.png")
    
    # 3. Drawdown analysis
    print("\n3. Creating drawdown analysis...")
    fig3 = perf_viz.plot_drawdown(
        returns_df['Portfolio'],
        title="Portfolio Drawdown Analysis"
    )
    fig3.savefig('demo_drawdown_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_drawdown_analysis.png")
    
    # 4. Performance dashboard
    print("\n4. Creating performance dashboard...")
    fig4 = perf_viz.create_performance_dashboard(
        returns_df['Portfolio'],
        benchmark=returns_df['S&P 500'],
        title="Comprehensive Portfolio Dashboard"
    )
    fig4.savefig('demo_performance_dashboard.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_performance_dashboard.png")
    
    plt.close('all')  # Clean up


def demo_allocation_visualization():
    """Demonstrate allocation visualization capabilities."""
    print("\n" + "="*60)
    print("ALLOCATION VISUALIZATION DEMO")
    print("="*60)
    
    # Create sample data
    _, weights, _, _ = create_sample_data()
    
    # Initialize visualizer
    alloc_viz = AllocationVisualizer()
    
    # 1. Pie chart
    print("\n1. Creating allocation pie chart...")
    fig1 = alloc_viz.plot_allocation_pie(
        weights,
        title="Current Portfolio Allocation"
    )
    fig1.savefig('demo_allocation_pie.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_allocation_pie.png")
    
    # 2. Bar chart with long/short positions
    print("\n2. Creating allocation bar chart...")
    # Add some leverage for demo
    weights_with_leverage = weights.copy()
    weights_with_leverage['CASH'] = -0.05  # Borrowed cash (leverage)
    
    fig2 = alloc_viz.plot_allocation_bar(
        weights_with_leverage,
        title="Portfolio Weights (Including Leverage)",
        show_net=True
    )
    fig2.savefig('demo_allocation_bar.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_allocation_bar.png")
    
    # 3. Allocation comparison
    print("\n3. Creating allocation comparison...")
    allocations = {
        'Current': weights,
        'Target': pd.Series({'SPY': 0.45, 'AGG': 0.25, 'GLD': 0.20, 'VNQ': 0.10}),
        'Conservative': pd.Series({'SPY': 0.30, 'AGG': 0.50, 'GLD': 0.10, 'VNQ': 0.10})
    }
    
    fig3 = alloc_viz.plot_allocation_comparison(
        allocations,
        title="Portfolio Allocation Scenarios"
    )
    fig3.savefig('demo_allocation_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_allocation_comparison.png")
    
    # 4. Waterfall chart
    print("\n4. Creating rebalancing waterfall...")
    fig4 = alloc_viz.plot_allocation_waterfall(
        allocations['Current'],
        allocations['Target'],
        title="Portfolio Rebalancing Analysis"
    )
    fig4.savefig('demo_allocation_waterfall.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_allocation_waterfall.png")
    
    plt.close('all')  # Clean up


def demo_optimization_visualization():
    """Demonstrate optimization visualization capabilities."""
    print("\n" + "="*60)
    print("OPTIMIZATION VISUALIZATION DEMO")
    print("="*60)
    
    # Create sample data
    returns_df, _, opt_results, _ = create_sample_data()
    
    # Initialize visualizer
    opt_viz = OptimizationVisualizer()
    
    # 1. Efficient frontier
    print("\n1. Creating efficient frontier...")
    fig1 = opt_viz.plot_efficient_frontier(
        returns_df[['S&P 500', 'Bonds', 'Gold']],  # Use subset for clarity
        optimal_points=list(opt_results.values()),
        risk_free_rate=0.02,
        n_portfolios=1000  # Reasonable number for demo
    )
    fig1.savefig('demo_efficient_frontier.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_efficient_frontier.png")
    
    # 2. Optimization comparison
    print("\n2. Creating optimization comparison...")
    fig2 = opt_viz.plot_optimization_comparison(
        opt_results,
        metrics=['expected_return', 'volatility', 'sharpe_ratio']
    )
    fig2.savefig('demo_optimization_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_optimization_comparison.png")
    
    # 3. Weight comparison
    print("\n3. Creating weight comparison...")
    fig3 = opt_viz.plot_weight_comparison(
        opt_results,
        title="Strategy Weight Comparison"
    )
    fig3.savefig('demo_weight_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_weight_comparison.png")
    
    # 4. Optimization report (without returns data to avoid alignment issues)
    print("\n4. Creating optimization report...")
    fig4 = opt_viz.create_optimization_report(
        opt_results['Max Sharpe'],
        title="Max Sharpe Portfolio Analysis"
    )
    fig4.savefig('demo_optimization_report.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_optimization_report.png")
    
    plt.close('all')  # Clean up


def demo_decomposition_visualization():
    """Demonstrate return decomposition visualization capabilities."""
    print("\n" + "="*60)
    print("DECOMPOSITION VISUALIZATION DEMO")
    print("="*60)
    
    # Create sample data
    _, _, _, decomposition = create_sample_data()
    
    # Initialize visualizer
    decomp_viz = DecompositionVisualizer()
    
    # 1. Return components over time
    print("\n1. Creating return components chart...")
    fig1 = decomp_viz.plot_return_components(
        decomposition,
        title="Return Decomposition Over Time"
    )
    fig1.savefig('demo_return_components.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_return_components.png")
    
    # 2. Component comparison across assets
    print("\n2. Creating component comparison...")
    # Create decompositions for multiple assets
    decompositions = {
        'Stocks': decomposition.copy(),
        'Bonds': decomposition.copy() * 0.6,  # Lower volatility
        'Gold': decomposition.copy() * 1.2    # Higher volatility
    }
    
    # Adjust spreads to be more realistic
    decompositions['Bonds']['spread'] *= 0.3  # Lower risk premium for bonds
    decompositions['Gold']['spread'] *= 0.8   # Moderate risk premium for gold
    
    fig2 = decomp_viz.plot_component_comparison(
        decompositions,
        title="Return Components Across Asset Classes"
    )
    fig2.savefig('demo_component_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_component_comparison.png")
    
    # 3. Inflation impact
    print("\n3. Creating inflation impact analysis...")
    nominal_returns = decomposition['total_return']
    real_returns = decomposition['total_return'] - decomposition['inflation']
    inflation = decomposition['inflation']
    
    fig3 = decomp_viz.plot_inflation_impact(
        nominal_returns,
        real_returns,
        inflation,
        title="Impact of Inflation on Investment Returns"
    )
    fig3.savefig('demo_inflation_impact.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_inflation_impact.png")
    
    # 4. Waterfall chart
    print("\n4. Creating waterfall chart...")
    summary = decomposition.mean() * 252  # Annualized
    
    fig4 = decomp_viz.create_waterfall_chart(
        summary,
        title="Annual Return Decomposition"
    )
    fig4.savefig('demo_waterfall_decomposition.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_waterfall_decomposition.png")
    
    plt.close('all')  # Clean up


def create_summary_dashboard():
    """Create a comprehensive summary dashboard."""
    print("\n" + "="*60)
    print("CREATING SUMMARY DASHBOARD")
    print("="*60)
    
    # Create sample data
    returns_df, weights, opt_results, decomposition = create_sample_data()
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Layout: 4 rows, 3 columns
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    print("\nBuilding comprehensive dashboard...")
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Row 1: Performance overview
    ax1 = fig.add_subplot(gs[0, :2])
    cum_returns = (1 + returns_df).cumprod()
    for i, col in enumerate(cum_returns.columns):
        ax1.plot(cum_returns.index, cum_returns[col], 
                linewidth=2, color=colors[i], label=col)
    ax1.set_title('Cumulative Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    wedges, texts, autotexts = ax2.pie(
        weights.values, labels=weights.index, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax2.set_title('Current Allocation', fontsize=14, fontweight='bold')
    
    # Row 2: Optimization comparison
    ax3 = fig.add_subplot(gs[1, :])
    strategies = list(opt_results.keys())
    metrics = ['expected_return', 'volatility', 'sharpe_ratio']
    x = np.arange(len(strategies))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [getattr(opt_results[s], metric) for s in strategies]
        bars = ax3.bar(x + i*width, values, width, 
                      label=metric.replace('_', ' ').title(),
                      color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            if metric in ['expected_return', 'volatility']:
                label = f'{value:.1%}'
            else:
                label = f'{value:.2f}'
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    label, ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Value')
    ax3.set_title('Optimization Strategy Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(strategies)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Return decomposition
    ax4 = fig.add_subplot(gs[2, :2])
    
    # Waterfall-style decomposition
    components = ['inflation', 'real_rf_rate', 'spread']
    component_labels = ['Inflation', 'Real Risk-Free', 'Risk Premium']
    component_colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    # Calculate annual averages
    annual_components = decomposition.mean() * 252
    values = [annual_components[comp] for comp in components]
    total_value = annual_components['total_return']
    
    # Create waterfall
    cumulative = np.cumsum([0] + values)
    for i, (label, value, color) in enumerate(zip(component_labels, values, component_colors)):
        ax4.bar(i, value, bottom=cumulative[i], color=color, alpha=0.8,
               edgecolor='black', linewidth=1)
        ax4.text(i, cumulative[i] + value/2, f'{value:.1%}',
                ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    # Total bar
    ax4.bar(len(values), total_value, color='darkblue', alpha=0.8,
           edgecolor='black', linewidth=2)
    ax4.text(len(values), total_value/2, f'{total_value:.1%}',
            ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    labels = component_labels + ['Total Return']
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_title('Return Decomposition (Annual)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Return Contribution')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Risk metrics
    ax5 = fig.add_subplot(gs[2, 2])
    rolling_vol = returns_df['Portfolio'].rolling(63).std() * np.sqrt(252)
    rolling_sharpe = (returns_df['Portfolio'].rolling(63).mean() / 
                     returns_df['Portfolio'].rolling(63).std()) * np.sqrt(252)
    
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(rolling_vol.index, rolling_vol.values, 
                    color='red', linewidth=1, label='Volatility')
    line2 = ax5_twin.plot(rolling_sharpe.index, rolling_sharpe.values,
                         color='blue', linewidth=1, label='Sharpe Ratio')
    
    ax5.set_ylabel('Volatility', color='red')
    ax5_twin.set_ylabel('Sharpe Ratio', color='blue')
    ax5.set_title('Rolling Risk Metrics', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    
    # Row 4: Statistics table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Calculate comprehensive statistics
    portfolio_returns = returns_df['Portfolio']
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    cum_ret = (1 + portfolio_returns).cumprod()
    max_dd = (cum_ret / cum_ret.expanding().max() - 1).min()
    
    # Best strategy
    best_strategy = max(opt_results.keys(), key=lambda x: opt_results[x].sharpe_ratio)
    
    stats_data = [
        ['Annual Return', f'{annual_return:.2%}', 'Current Positions', f'{len(weights)}'],
        ['Volatility', f'{annual_vol:.2%}', 'Best Strategy', best_strategy],
        ['Sharpe Ratio', f'{sharpe:.2f}', 'Max Sharpe Ratio', f'{opt_results[best_strategy].sharpe_ratio:.2f}'],
        ['Max Drawdown', f'{max_dd:.2%}', 'Avg Inflation', f'{decomposition["inflation"].mean() * 252:.1%}'],
        ['Total Period', f'{len(portfolio_returns)} days', 'Risk Premium', f'{decomposition["spread"].mean() * 252:.1%}']
    ]
    
    # Create two-column table
    table = ax6.table(cellText=stats_data,
                     colLabels=['Portfolio Metric', 'Value', 'Analysis Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data)):
        for j in range(4):
            if j % 2 == 0:  # Metric columns
                table[(i+1, j)].set_facecolor('#f0f0f0')
            else:  # Value columns
                table[(i+1, j)].set_facecolor('#ffffff')
    
    # Header styling
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Portfolio Analysis Dashboard', fontsize=20, fontweight='bold')
    fig.savefig('demo_summary_dashboard.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_summary_dashboard.png")
    
    plt.close('all')


def main():
    """Run all visualization demonstrations."""
    print("Portfolio Visualization Demo")
    print("=" * 60)
    print("This demo showcases all visualization capabilities")
    print("Charts will be saved as PNG files in the current directory")
    
    try:
        # Run individual demos
        demo_performance_visualization()
        demo_allocation_visualization()
        demo_optimization_visualization()
        demo_decomposition_visualization()
        
        # Create summary dashboard
        create_summary_dashboard()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- demo_cumulative_returns.png")
        print("- demo_rolling_metrics.png")
        print("- demo_drawdown_analysis.png")
        print("- demo_performance_dashboard.png")
        print("- demo_allocation_pie.png")
        print("- demo_allocation_bar.png")
        print("- demo_allocation_comparison.png")
        print("- demo_allocation_waterfall.png")
        print("- demo_efficient_frontier.png")
        print("- demo_optimization_comparison.png")
        print("- demo_weight_comparison.png")
        print("- demo_optimization_report.png")
        print("- demo_return_components.png")
        print("- demo_component_comparison.png")
        print("- demo_inflation_impact.png")
        print("- demo_waterfall_decomposition.png")
        print("- demo_summary_dashboard.png")
        print("\n✨ All visualization tools are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()