#!/usr/bin/env python
"""
Working Portfolio Visualization Demo

This version works with the corrected imports and demonstrates the visualization tools.
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
        'total_return': portfolio_returns,
        'inflation': inflation,
        'real_rf_rate': real_rf,
        'spread': portfolio_returns - inflation - real_rf
    }, index=dates)
    
    return returns_df, weights, opt_results, decomposition


def quick_demo():
    """Run a quick demonstration of all visualization tools."""
    print("Portfolio Visualization Working Demo")
    print("=" * 50)
    
    # Create sample data
    returns_df, weights, opt_results, decomposition = create_sample_data()
    
    print("\n1. Testing Performance Visualization...")
    try:
        perf_viz = PerformanceVisualizer()
        fig = perf_viz.plot_cumulative_returns(
            returns_df,
            title="Portfolio Performance Comparison"
        )
        fig.savefig('working_demo_performance.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   ✅ Performance chart saved")
    except Exception as e:
        print(f"   ❌ Performance visualization failed: {e}")
    
    print("\n2. Testing Allocation Visualization...")
    try:
        alloc_viz = AllocationVisualizer()
        fig = alloc_viz.plot_allocation_pie(weights)
        fig.savefig('working_demo_allocation.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   ✅ Allocation chart saved")
    except Exception as e:
        print(f"   ❌ Allocation visualization failed: {e}")
    
    print("\n3. Testing Optimization Visualization...")
    try:
        opt_viz = OptimizationVisualizer()
        fig = opt_viz.plot_optimization_comparison(opt_results)
        fig.savefig('working_demo_optimization.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   ✅ Optimization chart saved")
    except Exception as e:
        print(f"   ❌ Optimization visualization failed: {e}")
    
    print("\n4. Testing Decomposition Visualization...")
    try:
        decomp_viz = DecompositionVisualizer()
        fig = decomp_viz.plot_return_components(decomposition)
        fig.savefig('working_demo_decomposition.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   ✅ Decomposition chart saved")
    except Exception as e:
        print(f"   ❌ Decomposition visualization failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("Generated files:")
    print("  - working_demo_performance.png")
    print("  - working_demo_allocation.png") 
    print("  - working_demo_optimization.png")
    print("  - working_demo_decomposition.png")


if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()