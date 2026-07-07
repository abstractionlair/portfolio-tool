#!/usr/bin/env python
"""
Risk Premium Prediction Demo

This demonstrates the theoretically superior approach of predicting volatilities 
and correlations for RISK PREMIA rather than total returns.

Key insight: Portfolio optimization should focus on compensated risk (risk premium)
not uncompensated components like risk-free rate volatility.

This demo shows:
1. Loading and decomposing exposure returns into components
2. Predicting risk premium volatilities and correlations
3. Comparing risk premium vs total return estimates
4. Generating portfolio-ready covariance matrices based on risk premia
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

# Import our new risk premium framework
from optimization.risk_premium_estimator import RiskPremiumEstimator, build_portfolio_risk_matrix_from_risk_premia
from data.exposure_universe import ExposureUniverse
from data.return_decomposition import ReturnDecomposer


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


def demonstrate_risk_premium_predictions():
    """Demonstrate risk premium prediction for exposure universe."""
    print("Risk Premium Prediction Demo")
    print("=" * 50)
    
    # Load exposure universe
    universe = load_exposure_universe()
    if not universe:
        return
    
    # Initialize return decomposer and risk premium estimator
    return_decomposer = ReturnDecomposer()
    risk_estimator = RiskPremiumEstimator(universe, return_decomposer)
    
    # Select exposures for demonstration (using available IDs)
    demo_exposures = [
        'us_large_equity',
        'intl_developed_large_equity', 
        'short_ust',
        'broad_ust',
        'commodities',
        'real_estate'
    ]
    
    # Filter to available exposures
    available_exposures = []
    for exp_id in demo_exposures:
        if universe.get_exposure(exp_id):
            available_exposures.append(exp_id)
    
    if len(available_exposures) < 2:
        print("Need at least 2 available exposures for demonstration")
        return
    
    print(f"Predicting risk premia for {len(available_exposures)} exposures:")
    for exp_id in available_exposures:
        exposure = universe.get_exposure(exp_id)
        print(f"  • {exp_id}: {exposure.name}")
    
    # Set prediction parameters
    estimation_date = datetime.now()
    forecast_horizon = 252  # 1 year
    
    print(f"\nForecast horizon: {forecast_horizon} days (1 year)")
    print(f"Estimation date: {estimation_date.date()}")
    
    return risk_estimator, available_exposures, estimation_date, forecast_horizon


def test_individual_risk_premium_estimation(risk_estimator, exposures, estimation_date, forecast_horizon):
    """Test risk premium estimation for individual exposures."""
    print(f"\nIndividual Risk Premium Estimation:")
    print("-" * 40)
    
    methods = ['historical', 'ewma']
    results = {}
    
    for method in methods:
        print(f"\n{method.upper()} method:")
        method_results = {}
        
        for exposure_id in exposures:
            try:
                estimate = risk_estimator.estimate_risk_premium_volatility(
                    exposure_id=exposure_id,
                    estimation_date=estimation_date,
                    forecast_horizon=forecast_horizon,
                    method=method,
                    lookback_days=1260,  # 5 years for more data
                    frequency='monthly'  # Monthly frequency for more stable decomposition
                )
                
                if estimate:
                    method_results[exposure_id] = estimate
                    print(f"  {exposure_id}:")
                    print(f"    Risk Premium Vol: {estimate.risk_premium_volatility:.1%}")
                    print(f"    Total Return Vol:  {estimate.total_volatility:.1%}")
                    print(f"    Difference:        {(estimate.total_volatility - estimate.risk_premium_volatility):.1%}")
                    print(f"    Sample size:       {estimate.sample_size} periods")
                else:
                    print(f"  {exposure_id}: Failed to estimate")
                    
            except Exception as e:
                print(f"  {exposure_id}: Error - {e}")
        
        results[method] = method_results
    
    return results


def test_risk_premium_correlation_matrix(risk_estimator, exposures, estimation_date):
    """Test risk premium correlation matrix estimation."""
    print(f"\nRisk Premium Correlation Matrix:")
    print("-" * 40)
    
    try:
        correlation_matrix = risk_estimator.estimate_risk_premium_correlation_matrix(
            exposures=exposures,
            estimation_date=estimation_date,
            method='ewma'
        )
        
        if not correlation_matrix.empty:
            print(f"Matrix shape: {correlation_matrix.shape}")
            print("\nRisk Premium Correlations:")
            print(correlation_matrix.round(3))
            
            # Check matrix properties
            eigenvals = np.linalg.eigvals(correlation_matrix.values)
            is_psd = np.all(eigenvals >= -1e-8)
            print(f"\nMatrix is positive semi-definite: {is_psd}")
            print(f"Minimum eigenvalue: {eigenvals.min():.6f}")
            
            return correlation_matrix
        else:
            print("Failed to estimate correlation matrix")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error estimating correlations: {e}")
        return pd.DataFrame()


def test_combined_risk_estimates(risk_estimator, exposures, estimation_date, forecast_horizon):
    """Test combined risk premium and total return estimates."""
    print(f"\nCombined Risk Estimates (Risk Premium + Total Return):")
    print("-" * 60)
    
    try:
        combined = risk_estimator.get_combined_risk_estimates(
            exposures=exposures,
            estimation_date=estimation_date,
            forecast_horizon=forecast_horizon,
            method='ewma'
        )
        
        if combined:
            print(f"Successfully estimated {len(combined.exposures)} exposures")
            
            print(f"\nRisk Premium Volatilities (for portfolio optimization):")
            for exp_id, vol in combined.risk_premium_volatilities.items():
                print(f"  {exp_id}: {vol:.1%}")
            
            print(f"\nTotal Return Volatilities (for implementation):")
            for exp_id, vol in combined.total_return_volatilities.items():
                print(f"  {exp_id}: {vol:.1%}")
            
            # Show the difference
            print(f"\nVolatility Difference (Total - Risk Premium):")
            for exp_id in combined.exposures:
                rp_vol = combined.risk_premium_volatilities[exp_id]
                total_vol = combined.total_return_volatilities[exp_id]
                diff = total_vol - rp_vol
                print(f"  {exp_id}: {diff:.1%} ({diff/total_vol:.0%} of total)")
            
            return combined
        else:
            print("Failed to get combined estimates")
            return None
            
    except Exception as e:
        print(f"Error getting combined estimates: {e}")
        return None


def test_portfolio_risk_matrix_from_risk_premia(risk_estimator, exposures, estimation_date, forecast_horizon):
    """Test building portfolio risk matrix from risk premia."""
    print(f"\nPortfolio Risk Matrix from Risk Premia:")
    print("-" * 45)
    
    # Create equal-weight portfolio
    portfolio_weights = {exp_id: 1.0 / len(exposures) for exp_id in exposures}
    
    print(f"Equal-weight portfolio:")
    for exp_id, weight in portfolio_weights.items():
        print(f"  {exp_id}: {weight:.1%}")
    
    try:
        covariance_matrix, exposure_order = build_portfolio_risk_matrix_from_risk_premia(
            portfolio_weights=portfolio_weights,
            risk_premium_estimator=risk_estimator,
            estimation_date=estimation_date,
            forecast_horizon=forecast_horizon,
            method='ewma'
        )
        
        print(f"\nRisk matrix shape: {covariance_matrix.shape}")
        print(f"Exposure order: {exposure_order}")
        
        # Calculate portfolio volatility
        weights_array = np.array([portfolio_weights[exp_id] for exp_id in exposure_order])
        portfolio_variance = weights_array.T @ covariance_matrix @ weights_array
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        print(f"\nPortfolio statistics (based on risk premia):")
        print(f"  Portfolio volatility: {portfolio_volatility:.1%} (annualized)")
        print(f"  Portfolio variance:   {portfolio_variance:.6f}")
        
        # Calculate risk contributions
        risk_contributions = (covariance_matrix @ weights_array) * weights_array / portfolio_variance
        
        print(f"\nRisk contributions:")
        for i, exp_id in enumerate(exposure_order):
            print(f"  {exp_id}: {risk_contributions[i]:.1%}")
        
        return covariance_matrix, exposure_order
        
    except Exception as e:
        print(f"Error building portfolio risk matrix: {e}")
        return None, None


def create_comparison_visualization(individual_results, correlation_matrix, combined_estimates):
    """Create visualizations comparing risk premium vs total return estimates."""
    print(f"\nCreating risk premium comparison visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Risk Premium vs Total Return Volatilities
        ax1 = axes[0, 0]
        if 'ewma' in individual_results and individual_results['ewma']:
            exposures = list(individual_results['ewma'].keys())
            rp_vols = [individual_results['ewma'][exp].risk_premium_volatility for exp in exposures]
            total_vols = [individual_results['ewma'][exp].total_volatility for exp in exposures]
            
            x_pos = np.arange(len(exposures))
            width = 0.35
            
            ax1.bar(x_pos - width/2, rp_vols, width, label='Risk Premium', alpha=0.8)
            ax1.bar(x_pos + width/2, total_vols, width, label='Total Return', alpha=0.8)
            
            ax1.set_xlabel('Exposures')
            ax1.set_ylabel('Annualized Volatility')
            ax1.set_title('Risk Premium vs Total Return Volatilities')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(exposures, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Risk Premium Correlation Heatmap
        ax2 = axes[0, 1]
        if not correlation_matrix.empty:
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax2, cbar_kws={'shrink': 0.5})
            ax2.set_title('Risk Premium Correlation Matrix')
        else:
            ax2.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
            ax2.set_title('Risk Premium Correlations (No Data)')
        
        # 3. Volatility Difference Analysis
        ax3 = axes[1, 0]
        if 'ewma' in individual_results and individual_results['ewma']:
            exposures = list(individual_results['ewma'].keys())
            differences = []
            
            for exp in exposures:
                estimate = individual_results['ewma'][exp]
                diff = estimate.total_volatility - estimate.risk_premium_volatility
                differences.append(diff)
            
            ax3.bar(range(len(exposures)), differences, alpha=0.7)
            ax3.set_xlabel('Exposures')
            ax3.set_ylabel('Volatility Difference (Total - Risk Premium)')
            ax3.set_title('Uncompensated Risk Component')
            ax3.set_xticks(range(len(exposures)))
            ax3.set_xticklabels(exposures, rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. Component Breakdown
        ax4 = axes[1, 1]
        if combined_estimates and combined_estimates.component_volatilities:
            components = ['risk_premium', 'inflation', 'real_risk_free']
            available_components = [c for c in components if c in combined_estimates.component_volatilities]
            
            if len(available_components) >= 2:
                exposure_sample = combined_estimates.exposures[0]  # Take first exposure as example
                
                component_vols = []
                component_labels = []
                
                for comp in available_components:
                    vol_series = combined_estimates.component_volatilities[comp]
                    if exposure_sample in vol_series.index:
                        component_vols.append(vol_series[exposure_sample])
                        component_labels.append(comp.replace('_', ' ').title())
                
                if component_vols:
                    ax4.pie(component_vols, labels=component_labels, autopct='%1.1f%%')
                    ax4.set_title(f'Risk Component Breakdown\n({exposure_sample})')
                else:
                    ax4.text(0.5, 0.5, 'No component data', ha='center', va='center')
            else:
                ax4.text(0.5, 0.5, 'Insufficient components', ha='center', va='center')
        else:
            ax4.text(0.5, 0.5, 'No component data', ha='center', va='center')
            ax4.set_title('Component Breakdown (No Data)')
        
        plt.tight_layout()
        plt.savefig('risk_premium_prediction_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved risk_premium_prediction_analysis.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    """Run the complete risk premium prediction demonstration."""
    try:
        # Setup and basic demonstration
        risk_estimator, exposures, estimation_date, forecast_horizon = demonstrate_risk_premium_predictions()
        
        if not risk_estimator or not exposures:
            print("Setup failed")
            return
        
        # Test individual risk premium estimation
        individual_results = test_individual_risk_premium_estimation(
            risk_estimator, exposures, estimation_date, forecast_horizon
        )
        
        # Test correlation matrix estimation  
        correlation_matrix = test_risk_premium_correlation_matrix(
            risk_estimator, exposures, estimation_date
        )
        
        # Test combined estimates
        combined_estimates = test_combined_risk_estimates(
            risk_estimator, exposures, estimation_date, forecast_horizon
        )
        
        # Test portfolio risk matrix building
        test_portfolio_risk_matrix_from_risk_premia(
            risk_estimator, exposures, estimation_date, forecast_horizon
        )
        
        # Create visualizations
        create_comparison_visualization(individual_results, correlation_matrix, combined_estimates)
        
        print(f"\n✅ Risk premium prediction demo completed successfully")
        
        print(f"\nKey Insights:")
        print(f"• Risk premium volatilities are typically LOWER than total return volatilities")
        print(f"• The difference represents uncompensated risk (rate/inflation volatility)")
        print(f"• Portfolio optimization should use risk premium estimates for better results")
        print(f"• This approach aligns with academic asset pricing theory")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()