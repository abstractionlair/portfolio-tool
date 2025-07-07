#!/usr/bin/env python
"""
Parameter Optimization Demo for Real Exposure Universe Data

This script demonstrates comprehensive parameter optimization for:
- Optimal frequency selection for different forecasting horizons
- EWMA lambda parameter tuning
- Volatility forecasting validation
- Correlation forecasting optimization

Uses real exposure universe data for proper validation.
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
import seaborn as sns
from datetime import datetime, timedelta

# Import optimization classes
from optimization.parameter_optimization import (
    ParameterOptimizer, OptimizationConfig, run_exposure_universe_optimization
)
from data.exposure_universe import ExposureUniverse
from data.multi_frequency import Frequency


def load_exposure_universe():
    """Load the exposure universe configuration."""
    print("Loading exposure universe...")
    
    universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
    
    if not universe_path.exists():
        print(f"❌ Exposure universe file not found: {universe_path}")
        print("Please ensure the exposure universe configuration exists.")
        return None
    
    try:
        universe = ExposureUniverse.from_yaml(str(universe_path))
        print(f"✅ Loaded {len(universe)} exposures from {universe_path}")
        return universe
    except Exception as e:
        print(f"❌ Failed to load exposure universe: {e}")
        return None


def create_focused_optimization_config():
    """Create a focused optimization configuration for demonstration."""
    return OptimizationConfig(
        # Test key frequencies
        test_frequencies=[Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY],
        
        # Focused parameter grid for speed
        lambda_values=[0.92, 0.94, 0.96],
        min_periods_values=[20, 30],
        
        # Key forecasting horizons
        forecast_horizons=[1, 5, 21],  # 1d, 1w, 1m
        
        # Improved backtesting settings for available data
        min_train_periods=100,  # Reduced from 200
        test_periods=20,        # Reduced from 50
        rolling_window=True
    )


def run_parameter_optimization_demo():
    """Run comprehensive parameter optimization demo."""
    
    # Load exposure universe
    universe = load_exposure_universe()
    if not universe:
        return
    
    # Create optimization configuration
    config = create_focused_optimization_config()
    
    # Select a subset of exposures for demonstration (using correct IDs from config)
    test_exposures = [
        'us_large_equity',
        'us_small_equity', 
        'intl_developed_large_equity',
        'short_ust',
        'broad_ust',
        'commodities',
        'real_estate'
    ]
    
    # Filter to existing exposures
    available_exposures = []
    for exp_id in test_exposures:
        if universe.get_exposure(exp_id):
            available_exposures.append(exp_id)
    
    print(f"Testing {len(available_exposures)} exposures")
    
    # Set analysis period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2000)  # ~5.5 years of data
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(universe, config)
    
    try:
        print("Running parameter optimization...")
        
        results = optimizer.optimize_all_parameters(
            start_date=start_date,
            end_date=end_date,
            exposure_ids=available_exposures
        )
        
        # Display results
        display_optimization_results(results, optimizer)
        
        # Create visualizations
        create_optimization_visualizations(optimizer)
        
        return results
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_optimization_results(results: dict, optimizer: ParameterOptimizer):
    """Display optimization results in a formatted way."""
    print("\nOptimization Results:")
    
    # Overall optimal parameters
    if 'overall_optimal' in results:
        optimal = results['overall_optimal']
        print(f"Overall optimal: {optimal['frequency'].upper()}, λ={optimal['lambda']:.3f}, min_periods={optimal['min_periods']}")
        print(f"Average score: {optimal['average_score']:.4f}")
    
    # Optimal by horizon
    if 'optimal_by_horizon' in results:
        print("By horizon:")
        for horizon_key, params in results['optimal_by_horizon'].items():
            horizon = horizon_key.replace('_period', '')
            print(f"  {horizon}d: {params['frequency']}, λ={params['lambda']:.3f}, hit_rate={params['hit_rate']:.1%}")
    
    # Summary statistics
    if 'summary_statistics' in results:
        stats = results['summary_statistics']
        print(f"Average hit rate: {stats['mean_hit_rate']:.1%}, combinations tested: {results['total_combinations_tested']}")


def create_optimization_visualizations(optimizer: ParameterOptimizer):
    """Create visualizations of optimization results."""
    
    # Get validation summary
    summary_df = optimizer.get_validation_summary()
    
    if summary_df.empty:
        print("No validation data for visualization")
        return
    
    try:
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap of performance by frequency and lambda
        ax1 = axes[0, 0]
        try:
            pivot_score = summary_df.pivot_table(
                values='combined_score', 
                index='frequency', 
                columns='lambda', 
                aggfunc='mean'
            )
            
            if not pivot_score.empty:
                sns.heatmap(pivot_score, annot=True, fmt='.3f', cmap='viridis_r', ax=ax1)
                ax1.set_title('Combined Score by Frequency and Lambda\n(Lower is Better)')
                ax1.set_ylabel('Frequency')
                ax1.set_xlabel('Lambda (EWMA Decay)')
            else:
                ax1.text(0.5, 0.5, 'No score data available', ha='center', va='center')
                ax1.set_title('Combined Score (No Data)')
        except Exception as e:
            ax1.text(0.5, 0.5, f'Score plot failed: {str(e)[:30]}', ha='center', va='center')
            ax1.set_title('Combined Score (Error)')
    
        # 2. Hit rate by frequency and horizon
        ax2 = axes[0, 1]
        try:
            pivot_hit = summary_df.pivot_table(
                values='volatility_hit_rate',
                index='frequency',
                columns='horizon',
                aggfunc='mean'
            )
            
            if not pivot_hit.empty:
                sns.heatmap(pivot_hit, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2)
                ax2.set_title('Volatility Hit Rate by Frequency and Horizon\n(Higher is Better)')
                ax2.set_ylabel('Frequency')
                ax2.set_xlabel('Forecast Horizon (periods)')
            else:
                ax2.text(0.5, 0.5, 'No hit rate data available', ha='center', va='center')
                ax2.set_title('Hit Rate (No Data)')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Hit rate plot failed: {str(e)[:30]}', ha='center', va='center')
            ax2.set_title('Hit Rate (Error)')
        
        # 3. Score distribution by frequency
        ax3 = axes[1, 0]
        try:
            frequency_order = ['daily', 'weekly', 'monthly', 'quarterly']
            available_freqs = [f for f in frequency_order if f in summary_df['frequency'].values]
            
            if available_freqs:
                summary_df_filtered = summary_df[summary_df['frequency'].isin(available_freqs)]
                
                sns.boxplot(data=summary_df_filtered, x='frequency', y='combined_score', 
                            order=available_freqs, ax=ax3)
                ax3.set_title('Score Distribution by Frequency')
                ax3.set_ylabel('Combined Score (Lower is Better)')
                ax3.set_xlabel('Frequency')
                ax3.tick_params(axis='x', rotation=45)
            else:
                ax3.text(0.5, 0.5, 'No frequency data available', ha='center', va='center')
                ax3.set_title('Score Distribution (No Data)')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Frequency plot failed: {str(e)[:30]}', ha='center', va='center')
            ax3.set_title('Score Distribution (Error)')
        
        # 4. Lambda performance analysis
        ax4 = axes[1, 1]
        try:
            lambda_stats = summary_df.groupby('lambda').agg({
                'combined_score': 'mean',
                'volatility_hit_rate': 'mean'
            }).reset_index()
            
            if not lambda_stats.empty:
                ax4_twin = ax4.twinx()
                
                line1 = ax4.plot(lambda_stats['lambda'], lambda_stats['combined_score'], 
                                 'b-o', label='Combined Score')
                ax4.set_ylabel('Combined Score (Lower is Better)', color='b')
                ax4.tick_params(axis='y', labelcolor='b')
                
                line2 = ax4_twin.plot(lambda_stats['lambda'], lambda_stats['volatility_hit_rate'], 
                                      'r-s', label='Hit Rate')
                ax4_twin.set_ylabel('Hit Rate (Higher is Better)', color='r')
                ax4_twin.tick_params(axis='y', labelcolor='r')
                
                ax4.set_xlabel('Lambda (EWMA Decay Parameter)')
                ax4.set_title('Performance vs Lambda Parameter')
                
                # Add legend
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_twin.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
            else:
                ax4.text(0.5, 0.5, 'No lambda data available', ha='center', va='center')
                ax4.set_title('Lambda Performance (No Data)')
        except Exception as e:
            ax4.text(0.5, 0.5, f'Lambda plot failed: {str(e)[:30]}', ha='center', va='center')
            ax4.set_title('Lambda Performance (Error)')
        
        plt.tight_layout()
        plt.savefig('parameter_optimization_results.png', dpi=300, bbox_inches='tight')
        print("Saved parameter_optimization_results.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Create summary table
    create_results_summary_table(summary_df)


def create_results_summary_table(summary_df: pd.DataFrame):
    """Create a summary table of the best parameters."""
    
    if summary_df.empty:
        return
    
    # Clean data first - remove NaN values
    clean_df = summary_df.dropna(subset=['combined_score'])
    
    if clean_df.empty:
        return




def main():
    """Run the complete parameter optimization demonstration."""
    print("Parameter Optimization Demo")
    
    try:
        # Run optimization
        results = run_parameter_optimization_demo()
        
        if results:
            print("\n✅ Parameter optimization completed successfully")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()