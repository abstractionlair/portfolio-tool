#!/usr/bin/env python
"""
Portfolio Visualization Demo

Demonstrates all visualization capabilities including:
- Performance analysis
- Allocation breakdowns  
- Optimization results
- Return decomposition
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
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

# Import data classes
from data.total_returns import TotalReturnFetcher
from data.fred_data import FREDDataFetcher
from data.return_decomposition import ReturnDecomposer

# Import portfolio classes
from portfolio.portfolio import Portfolio, Position
from optimization.engine import OptimizationResult


def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data...")
    
    # Sample portfolio returns
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate different asset returns
    returns_data = {
        'Portfolio': np.random.normal(0.0008, 0.012, len(dates)),
        'S&P 500': np.random.normal(0.0009, 0.015, len(dates)),
        'Bonds': np.random.normal(0.0003, 0.006, len(dates)),
        'Gold': np.random.normal(0.0005, 0.018, len(dates))
    }
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Sample portfolio weights
    weights = pd.Series({
        'SPY': 0.4,
        'AGG': 0.3,
        'GLD': 0.15,
        'VNQ': 0.1,
        'TLT': 0.05
    })
    
    # Sample optimization results using the correct OptimizationResult structure
    opt_results = {
        'Equal Weight': OptimizationResult(
            weights=pd.Series({'SPY': 0.25, 'AGG': 0.25, 'GLD': 0.25, 'VNQ': 0.25}),
            objective_value=0.67,
            expected_return=0.08,
            expected_volatility=0.12,
            sharpe_ratio=0.67,
            exposures={},
            total_notional=1.0,
            success=True,
            message="Optimization successful"
        ),
        'Risk Parity': OptimizationResult(
            weights=pd.Series({'SPY': 0.35, 'AGG': 0.45, 'GLD': 0.15, 'VNQ': 0.05}),
            objective_value=0.75,
            expected_return=0.075,
            expected_volatility=0.10,
            sharpe_ratio=0.75,
            exposures={},
            total_notional=1.0,
            success=True,
            message="Optimization successful"
        ),
        'Max Sharpe': OptimizationResult(
            weights=pd.Series({'SPY': 0.55, 'AGG': 0.20, 'GLD': 0.10, 'VNQ': 0.15}),
            objective_value=0.68,
            expected_return=0.095,
            expected_volatility=0.14,
            sharpe_ratio=0.68,
            exposures={},
            total_notional=1.0,
            success=True,
            message="Optimization successful"
        )
    }
    
    # Sample decomposition data
    inflation = np.random.normal(0.00015, 0.003, len(dates))
    real_rf = np.random.normal(0.00008, 0.001, len(dates))
    
    decomposition = pd.DataFrame({
        'total_return': returns_data['Portfolio'],
        'inflation': inflation,
        'real_rf_rate': real_rf,
        'spread': returns_data['Portfolio'] - inflation - real_rf
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
    print("   Saved: demo_cumulative_returns.png")
    
    # 2. Rolling metrics
    print("\n2. Creating rolling metrics chart...")
    fig2 = perf_viz.plot_rolling_metrics(
        returns_df['Portfolio'],
        metrics=['volatility', 'sharpe', 'drawdown'],
        window=252
    )
    fig2.savefig('demo_rolling_metrics.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_rolling_metrics.png")
    
    # 3. Drawdown analysis
    print("\n3. Creating drawdown analysis...")
    fig3 = perf_viz.plot_drawdown(
        returns_df['Portfolio'],
        title="Portfolio Drawdown Analysis"
    )
    fig3.savefig('demo_drawdown_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_drawdown_analysis.png")
    
    # 4. Performance dashboard
    print("\n4. Creating performance dashboard...")
    fig4 = perf_viz.create_performance_dashboard(
        returns_df['Portfolio'],
        benchmark=returns_df['S&P 500'],
        title="Comprehensive Portfolio Dashboard"
    )
    fig4.savefig('demo_performance_dashboard.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_performance_dashboard.png")
    
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
    print("   Saved: demo_allocation_pie.png")
    
    # 2. Bar chart with long/short positions
    print("\n2. Creating allocation bar chart...")
    # Add some short positions for demo
    weights_with_shorts = weights.copy()
    weights_with_shorts['CASH'] = -0.05  # Short cash (leverage)
    
    fig2 = alloc_viz.plot_allocation_bar(
        weights_with_shorts,
        title="Portfolio Weights (Long/Short)",
        show_net=True
    )
    fig2.savefig('demo_allocation_bar.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_allocation_bar.png")
    
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
    print("   Saved: demo_allocation_comparison.png")
    
    # 4. Waterfall chart
    print("\n4. Creating rebalancing waterfall...")
    fig4 = alloc_viz.plot_allocation_waterfall(
        allocations['Current'],
        allocations['Target'],
        title="Portfolio Rebalancing Analysis"
    )
    fig4.savefig('demo_allocation_waterfall.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_allocation_waterfall.png")
    
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
        risk_free_rate=0.02
    )
    fig1.savefig('demo_efficient_frontier.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_efficient_frontier.png")
    
    # 2. Optimization comparison
    print("\n2. Creating optimization comparison...")
    fig2 = opt_viz.plot_optimization_comparison(
        opt_results,
        metrics=['expected_return', 'volatility', 'sharpe_ratio']
    )
    fig2.savefig('demo_optimization_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_optimization_comparison.png")
    
    # 3. Weight comparison
    print("\n3. Creating weight comparison...")
    fig3 = opt_viz.plot_weight_comparison(
        opt_results,
        title="Strategy Weight Comparison"
    )
    fig3.savefig('demo_weight_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_weight_comparison.png")
    
    # 4. Optimization report (without returns data to avoid alignment issues)
    print("\n4. Creating optimization report...")
    fig4 = opt_viz.create_optimization_report(
        opt_results['Max Sharpe'],
        title="Max Sharpe Portfolio Analysis"
    )
    fig4.savefig('demo_optimization_report.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_optimization_report.png")
    
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
    print("   Saved: demo_return_components.png")
    
    # 2. Component comparison across assets
    print("\n2. Creating component comparison...")
    # Create decompositions for multiple assets
    decompositions = {
        'Stocks': decomposition.copy(),
        'Bonds': decomposition.copy() * 0.6,  # Lower volatility
        'Gold': decomposition.copy() * 1.2    # Higher volatility
    }
    
    fig2 = decomp_viz.plot_component_comparison(
        decompositions,
        title="Return Components Across Asset Classes"
    )
    fig2.savefig('demo_component_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_component_comparison.png")
    
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
    print("   Saved: demo_inflation_impact.png")
    
    # 4. Waterfall chart
    print("\n4. Creating waterfall chart...")
    summary = decomposition.mean() * 252  # Annualized
    
    fig4 = decomp_viz.create_waterfall_chart(
        summary,
        title="Annual Return Decomposition"
    )
    fig4.savefig('demo_waterfall_decomposition.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_waterfall_decomposition.png")
    
    plt.close('all')  # Clean up


def create_combined_dashboard():
    """Create a comprehensive dashboard combining all visualization types."""
    print("\n" + "="*60)
    print("COMBINED DASHBOARD DEMO")
    print("="*60)
    
    # Create sample data
    returns_df, weights, opt_results, decomposition = create_sample_data()
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Layout: 4 rows, 3 columns
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Initialize visualizers
    perf_viz = PerformanceVisualizer()
    alloc_viz = AllocationVisualizer()
    opt_viz = OptimizationVisualizer()
    decomp_viz = DecompositionVisualizer()
    
    print("\nBuilding comprehensive dashboard...")
    
    # Row 1: Performance metrics
    ax1 = fig.add_subplot(gs[0, :2])
    cum_returns = (1 + returns_df['Portfolio']).cumprod()
    ax1.plot(cum_returns.index, cum_returns.values, linewidth=2, color='blue')
    ax1.set_title('Portfolio Cumulative Performance', fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    wedges, texts, autotexts = ax2.pie(
        weights.values, labels=weights.index, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    ax2.set_title('Current Allocation', fontweight='bold')
    
    # Row 2: Optimization results
    ax3 = fig.add_subplot(gs[1, :])
    strategies = list(opt_results.keys())
    metrics = ['expected_return', 'expected_volatility', 'sharpe_ratio']
    x = np.arange(len(strategies))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [getattr(opt_results[s], metric) for s in strategies]
        ax3.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(),
               color=colors[i])
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Value')
    ax3.set_title('Optimization Strategy Comparison', fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(strategies)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Return decomposition
    ax4 = fig.add_subplot(gs[2, :2])
    components = ['inflation', 'real_rf_rate', 'spread']
    component_colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    # Calculate cumulative components
    cum_inflation = (1 + decomposition['inflation']).cumprod()
    cum_real_rf = (1 + decomposition['real_rf_rate']).cumprod()
    cum_total = (1 + decomposition['total_return']).cumprod()
    
    ax4.fill_between(decomposition.index, 0, cum_inflation - 1,
                    color=component_colors[0], alpha=0.7, label='Inflation')
    ax4.plot(cum_total.index, cum_total - 1, color='black', linewidth=2, label='Total')
    ax4.set_title('Return Decomposition', fontweight='bold')
    ax4.set_ylabel('Cumulative Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Row 3: Risk metrics
    ax5 = fig.add_subplot(gs[2, 2])
    rolling_vol = returns_df['Portfolio'].rolling(63).std() * np.sqrt(252)
    ax5.plot(rolling_vol.index, rolling_vol.values, color='red', linewidth=1)
    ax5.set_title('Rolling Volatility', fontweight='bold')
    ax5.set_ylabel('Volatility')
    ax5.grid(True, alpha=0.3)
    
    # Row 4: Statistics table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Calculate key statistics
    portfolio_returns = returns_df['Portfolio']
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    max_dd = ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().expanding().max() - 1).min()
    
    stats_data = [
        ['Annual Return', f'{annual_return:.2%}'],
        ['Volatility', f'{annual_vol:.2%}'],
        ['Sharpe Ratio', f'{sharpe:.2f}'],
        ['Max Drawdown', f'{max_dd:.2%}'],
        ['Current Allocation', f'{len(weights)} positions'],
        ['Best Strategy', max(opt_results.keys(), key=lambda x: opt_results[x].sharpe_ratio)]
    ]
    
    table = ax6.table(cellText=stats_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.2, 0.1, 0.6, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data)):
        table[(i+1, 0)].set_facecolor('#f0f0f0')
        table[(i+1, 1)].set_facecolor('#ffffff')
    
    table[(0, 0)].set_facecolor('#4472C4')
    table[(0, 1)].set_facecolor('#4472C4')
    table[(0, 0)].set_text_props(weight='bold', color='white')
    table[(0, 1)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Portfolio Analysis Dashboard', fontsize=20, fontweight='bold')
    fig.savefig('demo_combined_dashboard.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_combined_dashboard.png")
    
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
        
        # Create combined dashboard
        create_combined_dashboard()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
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
        print("- demo_combined_dashboard.png")
        print("\nAll visualization tools are working correctly!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()