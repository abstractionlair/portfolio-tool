#!/usr/bin/env python
"""
Exposure-Level Risk Estimation Demo

This demonstrates the complete pipeline from parameter optimization to 
forward-looking risk estimation for portfolio optimization:

1. Load optimal parameters from parameter optimization
2. Estimate forward-looking volatilities for exposures  
3. Build correlation matrices between exposures
4. Compare different estimation methods
5. Generate portfolio-ready covariance matrices

This bridges the gap between parameter validation and portfolio construction.
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

# Import risk estimation classes
from optimization.exposure_risk_estimator import (
    ExposureRiskEstimator, build_portfolio_risk_matrix
)
from optimization.parameter_optimization import ParameterOptimizer, OptimizationConfig
from data.exposure_universe import ExposureUniverse
from data.multi_frequency import Frequency


def load_exposure_universe():
    """Load the exposure universe configuration."""
    universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
    
    if not universe_path.exists():
        print(f"Exposure universe file not found: {universe_path}")
        return None
    
    try:
        universe = ExposureUniverse.from_yaml(str(universe_path))
        print(f"Loaded {len(universe)} exposures")
        return universe
    except Exception as e:
        print(f"Failed to load exposure universe: {e}")
        return None


def demonstrate_exposure_risk_estimation():
    """Demonstrate complete exposure risk estimation workflow."""
    # Load exposure universe
    universe = load_exposure_universe()
    if not universe:
        return
    
    # Select exposures for demonstration
    demo_exposures = [
        'us_large_equity',
        'us_treasury_long', 
        'intl_developed_large_equity',
        'commodities_broad'
    ]
    
    # Filter to available exposures
    available_exposures = [exp_id for exp_id in demo_exposures 
                          if universe.get_exposure(exp_id)]
    
    if not available_exposures:
        print("No demo exposures found in universe")
        return
    
    print(f"Testing {len(available_exposures)} exposures")
    
    # Create risk estimator
    risk_estimator = ExposureRiskEstimator(universe)
    
    # Set analysis parameters
    estimation_date = datetime.now()
    forecast_horizon = 21  # 1 month
    lookback_days = 756    # 3 years
    
    return risk_estimator, available_exposures, estimation_date, forecast_horizon


def test_individual_risk_estimation(risk_estimator, exposures, estimation_date, forecast_horizon):
    """Test individual exposure risk estimation."""
    methods = ['historical', 'ewma', 'garch']
    
    results = {}
    
    for method in methods:
        try:
            risk_estimates = risk_estimator.estimate_exposure_risks(
                exposures, 
                estimation_date, 
                forecast_horizon=forecast_horizon,
                method=method
            )
            
            results[method] = risk_estimates
            print(f"{method}: {len(risk_estimates)}/{len(exposures)} exposures")
                
        except Exception as e:
            print(f"{method} estimation failed: {e}")
            results[method] = {}
    
    return results


def test_correlation_matrix_estimation(risk_estimator, exposures, estimation_date):
    """Test correlation matrix estimation."""
    methods = ['historical', 'ewma']
    correlation_results = {}
    
    for method in methods:
        try:
            correlation_matrix = risk_estimator.estimate_exposure_correlation_matrix(
                exposures, estimation_date, method=method
            )
            
            correlation_results[method] = correlation_matrix
            
            if not correlation_matrix.empty:
                print(f"{method} correlation matrix: {correlation_matrix.shape}")
            else:
                print(f"{method} correlation: empty matrix")
                
        except Exception as e:
            print(f"{method} correlation failed: {e}")
            correlation_results[method] = pd.DataFrame()
    
    return correlation_results


def test_complete_risk_matrix(risk_estimator, exposures, estimation_date, forecast_horizon):
    """Test complete risk matrix generation."""
    methods = ['historical', 'ewma']
    
    for method in methods:
        try:
            risk_matrix = risk_estimator.get_risk_matrix(
                exposures, estimation_date, forecast_horizon, method
            )
            
            print(f"{method} risk matrix: {len(risk_matrix.exposures)} exposures")
            
        except Exception as e:
            print(f"{method} risk matrix failed: {e}")


def test_portfolio_risk_matrix_building(risk_estimator, exposures, estimation_date):
    """Test building portfolio risk matrix for optimization."""
    # Define example portfolio weights
    if len(exposures) >= 2:
        portfolio_weights = {}
        weight = 1.0 / len(exposures)
        for exp_id in exposures:
            portfolio_weights[exp_id] = weight
    else:
        portfolio_weights = {exposures[0]: 1.0}
    
    try:
        covariance_matrix, exposure_order = build_portfolio_risk_matrix(
            portfolio_weights, risk_estimator, estimation_date
        )
        
        # Calculate portfolio variance
        weights_array = np.array([portfolio_weights[exp_id] for exp_id in exposure_order])
        portfolio_variance = weights_array.T @ covariance_matrix @ weights_array
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        print(f"Portfolio volatility: {portfolio_volatility:.1%}")
        
        return covariance_matrix, exposure_order
        
    except Exception as e:
        print(f"Portfolio risk matrix failed: {e}")
        return None, None


def create_risk_comparison_visualization(individual_results, correlation_results):
    """Create visualizations comparing different methods."""
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Volatility comparison across methods
        ax1 = axes[0, 0]
        methods = list(individual_results.keys())
        exposures = list(individual_results[methods[0]].keys()) if methods else []
        
        if methods and exposures:
            volatility_data = []
            for method in methods:
                for exp_id in exposures:
                    if exp_id in individual_results[method]:
                        volatility_data.append({
                            'method': method.upper(),
                            'exposure': exp_id,
                            'volatility': individual_results[method][exp_id].volatility
                        })
            
            if volatility_data:
                vol_df = pd.DataFrame(volatility_data)
                vol_pivot = vol_df.pivot(index='exposure', columns='method', values='volatility')
                
                vol_pivot.plot(kind='bar', ax=ax1)
                ax1.set_title('Volatility Estimates by Method')
                ax1.set_ylabel('Annualized Volatility')
                ax1.tick_params(axis='x', rotation=45)
                ax1.legend()
        
        # 2. Correlation heatmap
        ax2 = axes[0, 1]
        if 'ewma' in correlation_results and not correlation_results['ewma'].empty:
            corr_matrix = correlation_results['ewma']
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax2, cbar_kws={'shrink': 0.5})
            ax2.set_title('EWMA Correlation Matrix')
        else:
            ax2.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
            ax2.set_title('Correlation Matrix (No Data)')
        
        # 3. Method comparison
        ax3 = axes[1, 0]
        if volatility_data:
            method_stats = vol_df.groupby('method')['volatility'].agg(['mean', 'std']).reset_index()
            
            x_pos = np.arange(len(method_stats))
            ax3.bar(x_pos, method_stats['mean'], yerr=method_stats['std'], 
                   capsize=5, alpha=0.7)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(method_stats['method'])
            ax3.set_title('Average Volatility by Method')
            ax3.set_ylabel('Mean Volatility ± Std')
        
        # 4. Risk distribution
        ax4 = axes[1, 1]
        if volatility_data:
            all_vols = [item['volatility'] for item in volatility_data]
            ax4.hist(all_vols, bins=15, alpha=0.7, density=True)
            ax4.set_title('Distribution of Volatility Estimates')
            ax4.set_xlabel('Annualized Volatility')
            ax4.set_ylabel('Density')
            
            # Add summary statistics
            mean_vol = np.mean(all_vols)
            std_vol = np.std(all_vols)
            ax4.axvline(mean_vol, color='red', linestyle='--', 
                       label=f'Mean: {mean_vol:.1%}')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('exposure_risk_estimation_results.png', dpi=300, bbox_inches='tight')
        print("Saved exposure_risk_estimation_results.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")




def main():
    """Run the complete exposure risk estimation demonstration."""
    print("Exposure Risk Estimation Demo")
    
    try:
        # Setup and basic demonstration
        risk_estimator, exposures, estimation_date, forecast_horizon = demonstrate_exposure_risk_estimation()
        
        if not risk_estimator or not exposures:
            print("Setup failed")
            return
        
        # Test individual risk estimation
        individual_results = test_individual_risk_estimation(
            risk_estimator, exposures, estimation_date, forecast_horizon
        )
        
        # Test correlation matrix estimation  
        correlation_results = test_correlation_matrix_estimation(
            risk_estimator, exposures, estimation_date
        )
        
        # Test complete risk matrix
        test_complete_risk_matrix(
            risk_estimator, exposures, estimation_date, forecast_horizon
        )
        
        # Test portfolio risk matrix building
        test_portfolio_risk_matrix_building(
            risk_estimator, exposures, estimation_date
        )
        
        # Create visualizations
        create_risk_comparison_visualization(individual_results, correlation_results)
        
        print("\n✅ Exposure risk estimation completed successfully")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()