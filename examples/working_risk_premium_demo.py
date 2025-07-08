#!/usr/bin/env python
"""
Working Risk Premium Prediction Demo

This demonstrates the complete working risk premium prediction framework
for exposure-level risk estimation using decomposed returns.

Focus: Get actual risk premium predictions working for the exposure universe.
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

# Import framework
from optimization.risk_premium_estimator import RiskPremiumEstimator
from data.exposure_universe import ExposureUniverse
from data.return_decomposition import ReturnDecomposer


def load_exposure_universe():
    """Load the exposure universe."""
    universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
    
    try:
        universe = ExposureUniverse.from_yaml(str(universe_path))
        print(f"Loaded {len(universe)} exposures")
        return universe
    except Exception as e:
        print(f"Failed to load universe: {e}")
        return None


def demonstrate_working_risk_premium_predictions():
    """Demonstrate working risk premium predictions."""
    print("Working Risk Premium Prediction for Exposure Universe")
    print("=" * 60)
    
    # Setup
    universe = load_exposure_universe()
    if not universe:
        return None, None, None, None
    
    risk_estimator = RiskPremiumEstimator(universe, ReturnDecomposer())
    
    # Test exposures (focus on liquid, established markets)
    test_exposures = [
        'us_large_equity',
        'intl_developed_large_equity',
        'commodities',
        'real_estate'
    ]
    
    # Filter to available
    available_exposures = [exp_id for exp_id in test_exposures 
                          if universe.get_exposure(exp_id)]
    
    print(f"Testing {len(available_exposures)} exposures:")
    for exp_id in available_exposures:
        exposure = universe.get_exposure(exp_id)
        print(f"  • {exp_id}: {exposure.name}")
    
    # Prediction parameters  
    estimation_date = datetime.now()
    forecast_horizon = 252  # 1 year
    
    print(f"\nPrediction setup:")
    print(f"  Forecast horizon: {forecast_horizon} days (1 year)")
    print(f"  Estimation date: {estimation_date.date()}")
    
    return risk_estimator, available_exposures, estimation_date, forecast_horizon


def predict_individual_risk_premia(risk_estimator, exposures, estimation_date, forecast_horizon):
    """Predict risk premia for individual exposures."""
    print(f"\n1. Individual Risk Premium Predictions:")
    print("-" * 45)
    
    predictions = {}
    
    for exposure_id in exposures:
        print(f"\n{exposure_id}:")
        
        try:
            # Use historical method (most robust)
            rp_estimate = risk_estimator.estimate_risk_premium_volatility(
                exposure_id=exposure_id,
                estimation_date=estimation_date,
                forecast_horizon=forecast_horizon,
                method='historical',
                lookback_days=1260,  # 5 years
                frequency='monthly'  # More stable than daily
            )
            
            if rp_estimate:
                predictions[exposure_id] = rp_estimate
                
                print(f"  ✅ Risk Premium Volatility: {rp_estimate.risk_premium_volatility:.1%}")
                print(f"     Total Return Volatility:  {rp_estimate.total_volatility:.1%}")
                print(f"     Uncompensated Risk:       {(rp_estimate.total_volatility - rp_estimate.risk_premium_volatility):.1%}")
                print(f"     Sample Size:              {rp_estimate.sample_size} periods")
                print(f"     Inflation Volatility:     {rp_estimate.inflation_volatility:.1%}")
                print(f"     Real RF Volatility:       {rp_estimate.real_rf_volatility:.1%}")
                
                # Show key insight
                rp_percentage = rp_estimate.risk_premium_volatility / rp_estimate.total_volatility
                print(f"     Risk Premium % of Total:  {rp_percentage:.0%}")
                
            else:
                print(f"  ❌ Failed to estimate")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return predictions


def build_risk_premium_correlation_matrix(risk_estimator, exposures, estimation_date):
    """Build correlation matrix of risk premia."""
    print(f"\n2. Risk Premium Correlation Matrix:")
    print("-" * 40)
    
    try:
        # Use monthly frequency for stability
        correlation_matrix = risk_estimator.estimate_risk_premium_correlation_matrix(
            exposures=exposures,
            estimation_date=estimation_date,
            method='historical',  # Most robust
            lookback_days=1260,   # 5 years
            frequency='monthly'
        )
        
        if not correlation_matrix.empty:
            print(f"✅ Risk Premium Correlation Matrix ({correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}):")
            print(correlation_matrix.round(3))
            
            # Check matrix properties
            eigenvals = np.linalg.eigvals(correlation_matrix.values)
            min_eigenval = eigenvals.min()
            is_psd = min_eigenval >= -1e-8
            
            print(f"\nMatrix Properties:")
            print(f"  Positive Semi-Definite: {is_psd}")
            print(f"  Minimum Eigenvalue: {min_eigenval:.6f}")
            
            return correlation_matrix
        else:
            print("❌ Failed to build correlation matrix")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error building correlation matrix: {e}")
        return pd.DataFrame()


def demonstrate_portfolio_construction(predictions, correlation_matrix):
    """Show how to use risk premium estimates for portfolio construction."""
    print(f"\n3. Portfolio Construction with Risk Premia:")
    print("-" * 50)
    
    if not predictions or correlation_matrix.empty:
        print("❌ Insufficient data for portfolio construction")
        return
    
    # Extract risk premium volatilities
    exposures = list(predictions.keys())
    rp_volatilities = pd.Series({
        exp_id: pred.risk_premium_volatility 
        for exp_id, pred in predictions.items()
    })
    
    total_volatilities = pd.Series({
        exp_id: pred.total_volatility 
        for exp_id, pred in predictions.items()
    })
    
    print(f"Risk Premium Volatilities (use for portfolio optimization):")
    for exp_id, vol in rp_volatilities.items():
        print(f"  {exp_id}: {vol:.1%}")
    
    # Build covariance matrix
    common_exposures = rp_volatilities.index.intersection(correlation_matrix.index)
    if len(common_exposures) >= 2:
        aligned_vols = rp_volatilities[common_exposures]
        aligned_corr = correlation_matrix.loc[common_exposures, common_exposures]
        
        # Covariance = correlation * vol_i * vol_j
        covariance_matrix = aligned_corr * np.outer(aligned_vols.values, aligned_vols.values)
        
        print(f"\nRisk Premium Covariance Matrix:")
        print(f"Shape: {covariance_matrix.shape}")
        print(f"Exposures: {list(common_exposures)}")
        
        # Simple equal-weight portfolio example
        n_assets = len(common_exposures)
        weights = np.ones(n_assets) / n_assets
        
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        print(f"\nEqual-Weight Portfolio (based on risk premia):")
        print(f"  Portfolio Volatility: {portfolio_volatility:.1%}")
        print(f"  Number of Assets: {n_assets}")
        
        # Compare to total return approach
        total_vols_aligned = total_volatilities[common_exposures]
        total_cov = aligned_corr * np.outer(total_vols_aligned.values, total_vols_aligned.values)
        total_portfolio_vol = np.sqrt(weights @ total_cov @ weights)
        
        print(f"\nComparison:")
        print(f"  Risk Premium Portfolio Vol: {portfolio_volatility:.1%}")
        print(f"  Total Return Portfolio Vol: {total_portfolio_vol:.1%}")
        print(f"  Difference: {(total_portfolio_vol - portfolio_volatility):.1%}")
        
        return covariance_matrix, common_exposures
    else:
        print("❌ Need at least 2 common exposures for portfolio construction")
        return None, None


def analyze_risk_components(predictions):
    """Analyze the different risk components."""
    print(f"\n4. Risk Component Analysis:")
    print("-" * 35)
    
    if not predictions:
        print("❌ No predictions available")
        return
    
    print(f"{'Exposure':<25} {'Total Vol':<10} {'RP Vol':<10} {'Uncompensated':<15} {'RP %':<8}")
    print("-" * 70)
    
    for exp_id, pred in predictions.items():
        total_vol = pred.total_volatility
        rp_vol = pred.risk_premium_volatility
        uncompensated = total_vol - rp_vol
        rp_percentage = rp_vol / total_vol
        
        print(f"{exp_id:<25} {total_vol:<10.1%} {rp_vol:<10.1%} {uncompensated:<15.1%} {rp_percentage:<8.0%}")
    
    # Summary insights
    avg_rp_percentage = np.mean([pred.risk_premium_volatility / pred.total_volatility 
                                for pred in predictions.values()])
    
    print(f"\nKey Insights:")
    print(f"  Average Risk Premium as % of Total: {avg_rp_percentage:.0%}")
    print(f"  Range varies by asset class (equity ~100%, bonds much lower)")
    print(f"  Uncompensated risk should NOT drive portfolio decisions")


def create_risk_premium_visualization(predictions, correlation_matrix):
    """Create visualization of risk premium analysis."""
    print(f"\n5. Creating Risk Premium Analysis Visualization...")
    
    if not predictions:
        print("❌ No data for visualization")
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Risk Premium vs Total Return Volatilities
        ax1 = axes[0, 0]
        exposures = list(predictions.keys())
        rp_vols = [predictions[exp].risk_premium_volatility for exp in exposures]
        total_vols = [predictions[exp].total_volatility for exp in exposures]
        
        x_pos = np.arange(len(exposures))
        width = 0.35
        
        ax1.bar(x_pos - width/2, rp_vols, width, label='Risk Premium', alpha=0.8, color='blue')
        ax1.bar(x_pos + width/2, total_vols, width, label='Total Return', alpha=0.8, color='orange')
        
        ax1.set_xlabel('Exposures')
        ax1.set_ylabel('Annualized Volatility')
        ax1.set_title('Risk Premium vs Total Return Volatilities')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([exp.replace('_', ' ').title() for exp in exposures], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk Premium Correlation Matrix
        ax2 = axes[0, 1]
        if not correlation_matrix.empty:
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax2, cbar_kws={'shrink': 0.8})
            ax2.set_title('Risk Premium Correlations')
        else:
            ax2.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
            ax2.set_title('Risk Premium Correlations (No Data)')
        
        # 3. Uncompensated Risk Component
        ax3 = axes[1, 0]
        uncompensated = [predictions[exp].total_volatility - predictions[exp].risk_premium_volatility 
                        for exp in exposures]
        
        bars = ax3.bar(range(len(exposures)), uncompensated, alpha=0.7, color='red')
        ax3.set_xlabel('Exposures')
        ax3.set_ylabel('Uncompensated Volatility')
        ax3.set_title('Uncompensated Risk Component')
        ax3.set_xticks(range(len(exposures)))
        ax3.set_xticklabels([exp.replace('_', ' ').title() for exp in exposures], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.1%}', ha='center', va='bottom')
        
        # 4. Component Breakdown for first exposure
        ax4 = axes[1, 1]
        if exposures:
            sample_exp = exposures[0]
            sample_pred = predictions[sample_exp]
            
            components = ['Risk Premium', 'Inflation', 'Real RF']
            values = [
                sample_pred.risk_premium_volatility,
                sample_pred.inflation_volatility,
                sample_pred.real_rf_volatility
            ]
            
            ax4.pie(values, labels=components, autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'Risk Components\n({sample_exp.replace("_", " ").title()})')
        
        plt.tight_layout()
        plt.savefig('working_risk_premium_predictions.png', dpi=300, bbox_inches='tight')
        print("Saved working_risk_premium_predictions.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    """Run the complete working risk premium prediction demo."""
    try:
        # Setup
        risk_estimator, exposures, estimation_date, forecast_horizon = demonstrate_working_risk_premium_predictions()
        
        if not risk_estimator:
            print("❌ Setup failed")
            return
        
        # 1. Individual predictions
        predictions = predict_individual_risk_premia(
            risk_estimator, exposures, estimation_date, forecast_horizon
        )
        
        # 2. Correlation matrix
        correlation_matrix = build_risk_premium_correlation_matrix(
            risk_estimator, exposures, estimation_date
        )
        
        # 3. Portfolio construction
        covariance_matrix, common_exposures = demonstrate_portfolio_construction(
            predictions, correlation_matrix
        )
        
        # 4. Component analysis
        analyze_risk_components(predictions)
        
        # 5. Visualization
        create_risk_premium_visualization(predictions, correlation_matrix)
        
        print(f"\n" + "=" * 60)
        print("🎉 WORKING RISK PREMIUM PREDICTIONS COMPLETE!")
        print("=" * 60)
        
        if predictions:
            print(f"\n✅ Successfully predicted risk premia for {len(predictions)} exposures")
            print(f"✅ Generated correlation matrix: {correlation_matrix.shape if not correlation_matrix.empty else 'Failed'}")
            print(f"✅ Portfolio construction: {'Success' if covariance_matrix is not None else 'Failed'}")
            
            print(f"\n🎯 KEY ACHIEVEMENTS:")
            print(f"• Risk premium decomposition working on real exposure data")
            print(f"• Forward-looking 1-year risk premium volatility predictions")
            print(f"• Risk premium correlation matrix for diversification analysis")
            print(f"• Portfolio-ready covariance matrices based on compensated risk")
            
            print(f"\n📊 NEXT STEPS:")
            print(f"1. Integrate with parameter optimization for optimal EWMA/GARCH parameters")
            print(f"2. Test with full exposure universe (all 16 exposures)")
            print(f"3. Compare portfolio optimization results vs traditional approach")
            print(f"4. Build automated prediction pipeline")
        else:
            print(f"\n❌ No successful predictions - need to debug data issues")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()