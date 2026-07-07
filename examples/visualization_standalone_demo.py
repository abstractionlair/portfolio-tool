#!/usr/bin/env python
"""
Completely standalone portfolio visualization demo.
No relative imports - works from any directory.
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path - works from any location
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Now import visualization classes
try:
    from visualization.performance import PerformanceVisualizer
    from visualization.allocation import AllocationVisualizer
    from visualization.optimization import OptimizationVisualizer
    from visualization.decomposition import DecompositionVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Src directory: {src_dir}")
    print(f"Src directory exists: {src_dir.exists()}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

# Simple OptimizationResult for demo
class OptimizationResult:
    def __init__(self, weights, expected_return, volatility, sharpe_ratio, metadata=None):
        self.weights = weights
        self.expected_return = expected_return
        self.volatility = volatility
        self.sharpe_ratio = sharpe_ratio
        self.metadata = metadata or {}


def main():
    """Run a simple visualization demo."""
    print("Standalone Portfolio Visualization Demo")
    print("=" * 50)
    
    # Change to script directory for output files
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Create sample data
    print("Creating sample data...")
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Sample returns
    returns_df = pd.DataFrame({
        'Portfolio': np.random.normal(0.0008, 0.012, len(dates)),
        'Benchmark': np.random.normal(0.0006, 0.010, len(dates))
    }, index=dates)
    
    # Sample weights
    weights = pd.Series({
        'SPY': 0.40,
        'AGG': 0.30,
        'GLD': 0.20,
        'VNQ': 0.10
    })
    
    # Test each visualizer
    print("\n1. Testing PerformanceVisualizer...")
    try:
        perf_viz = PerformanceVisualizer()
        fig = perf_viz.plot_cumulative_returns(returns_df['Portfolio'])
        fig.savefig('test_performance.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   ✓ Performance visualization working")
    except Exception as e:
        print(f"   ✗ Performance visualization failed: {e}")
    
    print("\n2. Testing AllocationVisualizer...")
    try:
        alloc_viz = AllocationVisualizer()
        fig = alloc_viz.plot_allocation_pie(weights)
        fig.savefig('test_allocation.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   ✓ Allocation visualization working")
    except Exception as e:
        print(f"   ✗ Allocation visualization failed: {e}")
    
    print("\n3. Testing OptimizationVisualizer...")
    try:
        opt_viz = OptimizationVisualizer()
        
        # Sample optimization results
        opt_results = {
            'Strategy A': OptimizationResult(
                weights=pd.Series({'SPY': 0.6, 'AGG': 0.4}),
                expected_return=0.08,
                volatility=0.12,
                sharpe_ratio=0.67
            ),
            'Strategy B': OptimizationResult(
                weights=pd.Series({'SPY': 0.4, 'AGG': 0.6}),
                expected_return=0.06,
                volatility=0.08,
                sharpe_ratio=0.75
            )
        }
        
        fig = opt_viz.plot_optimization_comparison(opt_results)
        fig.savefig('test_optimization.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   ✓ Optimization visualization working")
    except Exception as e:
        print(f"   ✗ Optimization visualization failed: {e}")
    
    print("\n4. Testing DecompositionVisualizer...")
    try:
        decomp_viz = DecompositionVisualizer()
        
        # Sample decomposition data
        decomposition = pd.DataFrame({
            'total_return': returns_df['Portfolio'].values,
            'inflation': np.random.normal(0.00015, 0.003, len(dates)),
            'real_rf_rate': np.random.normal(0.00008, 0.001, len(dates)),
            'spread': np.random.normal(0.0005, 0.010, len(dates))
        }, index=dates)
        
        fig = decomp_viz.plot_return_components(decomposition)
        fig.savefig('test_decomposition.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   ✓ Decomposition visualization working")
    except Exception as e:
        print(f"   ✗ Decomposition visualization failed: {e}")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("Generated test files:")
    for file in ['test_performance.png', 'test_allocation.png', 
                'test_optimization.png', 'test_decomposition.png']:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not created)")


if __name__ == "__main__":
    main()