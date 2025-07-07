#!/usr/bin/env python
"""
Interactive Portfolio Visualization Demo

This version displays charts on screen instead of just saving files.
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

# Import OptimizationResult
from optimization.engine import OptimizationResult

def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data...")
    
    # Sample portfolio returns
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate different asset returns with realistic correlations
    n_days = len(dates)
    market_factor = np.random.normal(0.0008, 0.015, n_days)
    
    returns_df = pd.DataFrame({
        'Portfolio': market_factor + np.random.normal(0.0002, 0.008, n_days),
        'S&P 500': market_factor + np.random.normal(0.0001, 0.003, n_days),
        'Bonds': np.random.normal(0.0003, 0.006, n_days) * 0.3 + market_factor * 0.1,
        'Gold': np.random.normal(0.0005, 0.018, n_days) - market_factor * 0.2
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
        'total_return': returns_df['Portfolio'].values,
        'inflation': inflation,
        'real_rf_rate': real_rf,
        'spread': returns_df['Portfolio'].values - inflation - real_rf
    }, index=dates)
    
    return returns_df, weights, opt_results, decomposition


def show_interactive_demos():
    """Display charts interactively."""
    returns_df, weights, opt_results, decomposition = create_sample_data()
    
    print("Interactive Portfolio Visualization Demo")
    print("=" * 50)
    print("Charts will display in separate windows.")
    print("Close each window to proceed to the next chart.")
    print()
    
    # 1. Performance Dashboard
    print("1. Showing Performance Dashboard...")
    perf_viz = PerformanceVisualizer()
    fig = perf_viz.create_performance_dashboard(
        returns_df['Portfolio'],
        benchmark=returns_df['S&P 500'],
        title="Portfolio Performance Dashboard"
    )
    plt.show()
    
    # 2. Allocation Pie Chart
    print("2. Showing Portfolio Allocation...")
    alloc_viz = AllocationVisualizer()
    fig = alloc_viz.plot_allocation_pie(weights, title="Portfolio Allocation")
    plt.show()
    
    # 3. Optimization Comparison
    print("3. Showing Optimization Strategy Comparison...")
    opt_viz = OptimizationVisualizer()
    fig = opt_viz.plot_optimization_comparison(
        opt_results,
        metrics=['expected_return', 'expected_volatility', 'sharpe_ratio']
    )
    plt.show()
    
    # 4. Efficient Frontier
    print("4. Showing Efficient Frontier...")
    fig = opt_viz.plot_efficient_frontier(
        returns_df[['Portfolio', 'S&P 500', 'Bonds']],
        optimal_points=list(opt_results.values()),
        n_portfolios=1000
    )
    plt.show()
    
    # 5. Return Decomposition
    print("5. Showing Return Decomposition...")
    decomp_viz = DecompositionVisualizer()
    fig = decomp_viz.plot_return_components(
        decomposition,
        title="Portfolio Return Decomposition"
    )
    plt.show()
    
    # 6. Allocation Comparison
    print("6. Showing Allocation Scenarios...")
    allocations = {
        'Current': weights,
        'Conservative': pd.Series({'SPY': 0.30, 'AGG': 0.50, 'GLD': 0.10, 'VNQ': 0.10}),
        'Aggressive': pd.Series({'SPY': 0.60, 'AGG': 0.15, 'GLD': 0.15, 'VNQ': 0.10})
    }
    fig = alloc_viz.plot_allocation_comparison(
        allocations,
        title="Portfolio Allocation Scenarios"
    )
    plt.show()
    
    print("\n" + "=" * 50)
    print("🎉 Interactive demo completed!")
    print("All charts have been displayed.")


if __name__ == "__main__":
    try:
        show_interactive_demos()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()