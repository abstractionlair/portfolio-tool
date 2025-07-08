#!/usr/bin/env python
"""
Full Universe Risk Premium Analysis

This script tests the risk premium prediction framework on the complete 
exposure universe (all 16 exposures), providing comprehensive analysis
of risk premia decomposition and portfolio construction capabilities.

Key features:
1. Tests all 16 exposures in the universe
2. Generates risk premium volatilities and correlations
3. Compares multiple estimation methods (historical, EWMA, GARCH)
4. Creates comprehensive visualizations
5. Exports results for portfolio optimization
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
import json

# Import framework
from optimization.risk_premium_estimator import RiskPremiumEstimator
from data.exposure_universe import ExposureUniverse
from data.return_decomposition import ReturnDecomposer


def load_full_exposure_universe():
    """Load the complete exposure universe."""
    universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
    
    try:
        universe = ExposureUniverse.from_yaml(str(universe_path))
        print(f"✅ Loaded complete exposure universe: {len(universe)} exposures")
        
        # Print all exposures
        print("\nFull Exposure Universe:")
        for i, exposure_id in enumerate(universe.exposures.keys(), 1):
            exposure = universe.get_exposure(exposure_id)
            print(f"  {i:2d}. {exposure_id}: {exposure.name}")
        
        return universe
    except Exception as e:
        print(f"❌ Failed to load universe: {e}")
        return None


def test_full_universe_risk_premium_estimation(risk_estimator, universe, estimation_date, forecast_horizon):
    """Test risk premium estimation on the full universe."""
    print(f"\n" + "="*60)
    print("FULL UNIVERSE RISK PREMIUM ESTIMATION")
    print("="*60)
    
    all_exposures = list(universe.exposures.keys())
    print(f"Testing {len(all_exposures)} exposures for risk premium estimation")
    
    # Test different methods
    methods = {
        'historical': {'parameters': {}},
        'ewma': {'parameters': {'lambda': 0.94, 'min_periods': 20}}
    }
    
    results = {}
    
    for method_name, method_config in methods.items():
        print(f"\n{method_name.upper()} Method:")
        print("-" * 40)
        
        method_results = {}
        successful_estimates = 0
        
        for exposure_id in all_exposures:
            try:
                # Use robust parameters
                estimate = risk_estimator.estimate_risk_premium_volatility(
                    exposure_id=exposure_id,
                    estimation_date=estimation_date,
                    forecast_horizon=forecast_horizon,
                    method=method_name,
                    lookback_days=1260,  # 5 years
                    frequency='monthly',  # Monthly for stability
                    **method_config['parameters']
                )
                
                if estimate and estimate.risk_premium_volatility > 0:
                    method_results[exposure_id] = estimate
                    successful_estimates += 1
                    
                    # Print key metrics
                    rp_pct = estimate.risk_premium_volatility / estimate.total_volatility * 100
                    print(f"  ✅ {exposure_id:<25} RP: {estimate.risk_premium_volatility:.1%} "
                          f"Total: {estimate.total_volatility:.1%} ({rp_pct:.0f}%)")
                else:
                    print(f"  ❌ {exposure_id:<25} Failed to estimate")
                    
            except Exception as e:
                print(f"  ❌ {exposure_id:<25} Error: {str(e)[:50]}...")
        
        results[method_name] = method_results
        print(f"\n{method_name.upper()} Summary: {successful_estimates}/{len(all_exposures)} successful estimates")
    
    return results


def build_full_universe_correlation_matrix(risk_estimator, universe, estimation_date):
    """Build correlation matrix for the full universe."""
    print(f"\n" + "="*60)
    print("FULL UNIVERSE CORRELATION MATRIX")
    print("="*60)
    
    all_exposures = list(universe.exposures.keys())
    
    try:
        # Use monthly frequency for stability
        correlation_matrix = risk_estimator.estimate_risk_premium_correlation_matrix(
            exposures=all_exposures,
            estimation_date=estimation_date,
            method='historical',  # Most robust for correlations
            lookback_days=1260,   # 5 years
            frequency='monthly'
        )
        
        if not correlation_matrix.empty:
            print(f"✅ Successfully built correlation matrix: {correlation_matrix.shape}")
            
            # Matrix validation
            eigenvals = np.linalg.eigvals(correlation_matrix.values)
            min_eigenval = eigenvals.min()
            is_psd = min_eigenval >= -1e-8
            
            print(f"Matrix Properties:")
            print(f"  Shape: {correlation_matrix.shape}")
            print(f"  Positive Semi-Definite: {is_psd}")
            print(f"  Minimum Eigenvalue: {min_eigenval:.6f}")
            print(f"  Max Correlation: {correlation_matrix.values[~np.eye(correlation_matrix.shape[0], dtype=bool)].max():.3f}")
            print(f"  Min Correlation: {correlation_matrix.values[~np.eye(correlation_matrix.shape[0], dtype=bool)].min():.3f}")
            
            # Show some key correlations
            print(f"\nSample Risk Premium Correlations:")
            equity_exposures = [exp for exp in correlation_matrix.index if 'equity' in exp.lower()]
            if len(equity_exposures) >= 2:
                for i, exp1 in enumerate(equity_exposures[:3]):
                    for exp2 in equity_exposures[i+1:3]:
                        if exp1 != exp2:
                            corr = correlation_matrix.loc[exp1, exp2]
                            print(f"  {exp1} vs {exp2}: {corr:.3f}")
            
            return correlation_matrix
        else:
            print("❌ Failed to build correlation matrix")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error building correlation matrix: {e}")
        return pd.DataFrame()


def analyze_risk_premium_components(results):
    """Analyze risk premium components across the universe."""
    print(f"\n" + "="*60)
    print("RISK PREMIUM COMPONENT ANALYSIS")
    print("="*60)
    
    if not results or 'historical' not in results:
        print("❌ No results available for analysis")
        return
    
    historical_results = results['historical']
    if not historical_results:
        print("❌ No historical results available")
        return
    
    # Create comprehensive analysis
    analysis_data = []
    
    for exposure_id, estimate in historical_results.items():
        data = {
            'exposure_id': exposure_id,
            'total_volatility': estimate.total_volatility,
            'risk_premium_volatility': estimate.risk_premium_volatility,
            'inflation_volatility': estimate.inflation_volatility,
            'real_rf_volatility': estimate.real_rf_volatility,
            'uncompensated_risk': estimate.total_volatility - estimate.risk_premium_volatility,
            'risk_premium_percentage': estimate.risk_premium_volatility / estimate.total_volatility * 100,
            'sample_size': estimate.sample_size
        }
        analysis_data.append(data)
    
    df = pd.DataFrame(analysis_data)
    df = df.sort_values('risk_premium_percentage', ascending=False)
    
    print(f"Risk Premium Analysis ({len(df)} exposures):")
    print("-" * 90)
    print(f"{'Exposure':<25} {'Total Vol':<10} {'RP Vol':<10} {'RP %':<8} {'Uncomp':<10} {'Samples':<8}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        print(f"{row['exposure_id']:<25} {row['total_volatility']:<10.1%} "
              f"{row['risk_premium_volatility']:<10.1%} {row['risk_premium_percentage']:<8.0f}% "
              f"{row['uncompensated_risk']:<10.1%} {row['sample_size']:<8.0f}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Universe Statistics:")
    print(f"  Total Exposures Analyzed: {len(df)}")
    print(f"  Average Risk Premium %: {df['risk_premium_percentage'].mean():.1f}%")
    print(f"  Median Risk Premium %: {df['risk_premium_percentage'].median():.1f}%")
    print(f"  Risk Premium % Range: {df['risk_premium_percentage'].min():.1f}% - {df['risk_premium_percentage'].max():.1f}%")
    print(f"  Average Total Volatility: {df['total_volatility'].mean():.1%}")
    print(f"  Average Risk Premium Volatility: {df['risk_premium_volatility'].mean():.1%}")
    print(f"  Average Uncompensated Risk: {df['uncompensated_risk'].mean():.1%}")
    
    # Categorize by asset class
    print(f"\nAsset Class Breakdown:")
    equity_exposures = df[df['exposure_id'].str.contains('equity', case=False)]
    bond_exposures = df[df['exposure_id'].str.contains('ust|bond', case=False)]
    alt_exposures = df[~df['exposure_id'].str.contains('equity|ust|bond', case=False)]
    
    for category, subset in [('Equity', equity_exposures), ('Fixed Income', bond_exposures), ('Alternatives', alt_exposures)]:
        if len(subset) > 0:
            print(f"  {category} ({len(subset)} exposures):")
            print(f"    Average Risk Premium %: {subset['risk_premium_percentage'].mean():.1f}%")
            print(f"    Average Total Volatility: {subset['total_volatility'].mean():.1%}")
    
    return df


def create_full_universe_visualization(results, correlation_matrix, analysis_df):
    """Create comprehensive visualization for the full universe."""
    print(f"\n" + "="*60)
    print("CREATING FULL UNIVERSE VISUALIZATION")
    print("="*60)
    
    try:
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Risk Premium vs Total Volatility (Top Left)
        ax1 = plt.subplot(3, 3, 1)
        if 'historical' in results and results['historical']:
            exposures = list(results['historical'].keys())
            rp_vols = [results['historical'][exp].risk_premium_volatility for exp in exposures]
            total_vols = [results['historical'][exp].total_volatility for exp in exposures]
            
            x_pos = np.arange(len(exposures))
            width = 0.35
            
            ax1.bar(x_pos - width/2, rp_vols, width, label='Risk Premium', alpha=0.8, color='steelblue')
            ax1.bar(x_pos + width/2, total_vols, width, label='Total Return', alpha=0.8, color='orange')
            
            ax1.set_xlabel('Exposures')
            ax1.set_ylabel('Annualized Volatility')
            ax1.set_title('Risk Premium vs Total Return Volatilities')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([exp.replace('_', ' ')[:10] for exp in exposures], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Risk Premium Percentage Distribution (Top Center)
        ax2 = plt.subplot(3, 3, 2)
        if analysis_df is not None and not analysis_df.empty:
            ax2.hist(analysis_df['risk_premium_percentage'], bins=10, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.set_xlabel('Risk Premium % of Total Volatility')
            ax2.set_ylabel('Number of Exposures')
            ax2.set_title('Risk Premium Percentage Distribution')
            ax2.grid(True, alpha=0.3)
        
        # 3. Correlation Matrix Heatmap (Top Right)
        ax3 = plt.subplot(3, 3, 3)
        if not correlation_matrix.empty:
            # Show subset if too large
            if correlation_matrix.shape[0] > 10:
                subset_indices = correlation_matrix.index[:10]
                subset_corr = correlation_matrix.loc[subset_indices, subset_indices]
            else:
                subset_corr = correlation_matrix
            
            sns.heatmap(subset_corr, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax3, cbar_kws={'shrink': 0.8}, fmt='.2f')
            ax3.set_title('Risk Premium Correlations\n(Top 10 Exposures)')
        
        # 4. Uncompensated Risk Analysis (Middle Left)
        ax4 = plt.subplot(3, 3, 4)
        if analysis_df is not None and not analysis_df.empty:
            sorted_df = analysis_df.sort_values('uncompensated_risk', ascending=True)
            y_pos = np.arange(len(sorted_df))
            
            ax4.barh(y_pos, sorted_df['uncompensated_risk'], alpha=0.7, color='red')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([exp.replace('_', ' ')[:15] for exp in sorted_df['exposure_id']])
            ax4.set_xlabel('Uncompensated Risk (Annual)')
            ax4.set_title('Uncompensated Risk by Exposure')
            ax4.grid(True, alpha=0.3)
        
        # 5. Risk Premium vs Uncompensated Risk Scatter (Middle Center)
        ax5 = plt.subplot(3, 3, 5)
        if analysis_df is not None and not analysis_df.empty:
            ax5.scatter(analysis_df['risk_premium_volatility'], analysis_df['uncompensated_risk'], 
                       alpha=0.7, s=60, color='steelblue')
            ax5.set_xlabel('Risk Premium Volatility')
            ax5.set_ylabel('Uncompensated Risk')
            ax5.set_title('Risk Premium vs Uncompensated Risk')
            ax5.grid(True, alpha=0.3)
            
            # Add trend line
            if len(analysis_df) > 1:
                z = np.polyfit(analysis_df['risk_premium_volatility'], analysis_df['uncompensated_risk'], 1)
                p = np.poly1d(z)
                ax5.plot(analysis_df['risk_premium_volatility'], p(analysis_df['risk_premium_volatility']), 
                        "r--", alpha=0.8)
        
        # 6. Asset Class Breakdown (Middle Right)
        ax6 = plt.subplot(3, 3, 6)
        if analysis_df is not None and not analysis_df.empty:
            # Categorize exposures
            equity_mask = analysis_df['exposure_id'].str.contains('equity', case=False)
            bond_mask = analysis_df['exposure_id'].str.contains('ust|bond', case=False)
            alt_mask = ~(equity_mask | bond_mask)
            
            categories = ['Equity', 'Fixed Income', 'Alternatives']
            rp_averages = [
                analysis_df[equity_mask]['risk_premium_percentage'].mean() if equity_mask.any() else 0,
                analysis_df[bond_mask]['risk_premium_percentage'].mean() if bond_mask.any() else 0,
                analysis_df[alt_mask]['risk_premium_percentage'].mean() if alt_mask.any() else 0
            ]
            
            ax6.bar(categories, rp_averages, alpha=0.7, color=['steelblue', 'orange', 'green'])
            ax6.set_ylabel('Average Risk Premium %')
            ax6.set_title('Risk Premium % by Asset Class')
            ax6.grid(True, alpha=0.3)
        
        # 7. Sample Size Distribution (Bottom Left)
        ax7 = plt.subplot(3, 3, 7)
        if analysis_df is not None and not analysis_df.empty:
            ax7.hist(analysis_df['sample_size'], bins=10, alpha=0.7, color='green', edgecolor='black')
            ax7.set_xlabel('Sample Size (Months)')
            ax7.set_ylabel('Number of Exposures')
            ax7.set_title('Sample Size Distribution')
            ax7.grid(True, alpha=0.3)
        
        # 8. Volatility Comparison (Bottom Center)
        ax8 = plt.subplot(3, 3, 8)
        if 'historical' in results and 'ewma' in results:
            hist_exposures = set(results['historical'].keys())
            ewma_exposures = set(results['ewma'].keys())
            common_exposures = hist_exposures.intersection(ewma_exposures)
            
            if len(common_exposures) > 0:
                hist_vols = [results['historical'][exp].risk_premium_volatility for exp in common_exposures]
                ewma_vols = [results['ewma'][exp].risk_premium_volatility for exp in common_exposures]
                
                ax8.scatter(hist_vols, ewma_vols, alpha=0.7, s=60, color='purple')
                ax8.set_xlabel('Historical Risk Premium Vol')
                ax8.set_ylabel('EWMA Risk Premium Vol')
                ax8.set_title('Historical vs EWMA Estimates')
                ax8.grid(True, alpha=0.3)
                
                # Add diagonal line
                min_vol = min(min(hist_vols), min(ewma_vols))
                max_vol = max(max(hist_vols), max(ewma_vols))
                ax8.plot([min_vol, max_vol], [min_vol, max_vol], 'r--', alpha=0.8)
        
        # 9. Risk Premium Efficiency (Bottom Right)
        ax9 = plt.subplot(3, 3, 9)
        if analysis_df is not None and not analysis_df.empty:
            # Risk Premium Efficiency = Risk Premium Vol / Total Vol
            efficiency = analysis_df['risk_premium_percentage'] / 100
            sorted_efficiency = efficiency.sort_values(ascending=False)
            
            y_pos = np.arange(len(sorted_efficiency))
            ax9.barh(y_pos, sorted_efficiency, alpha=0.7, color='gold')
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels([analysis_df.loc[idx, 'exposure_id'].replace('_', ' ')[:15] 
                                for idx in sorted_efficiency.index])
            ax9.set_xlabel('Risk Premium Efficiency')
            ax9.set_title('Risk Premium Efficiency by Exposure')
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('full_universe_risk_premium_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved full_universe_risk_premium_analysis.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False


def export_results(results, correlation_matrix, analysis_df):
    """Export results for further analysis and portfolio optimization."""
    print(f"\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    
    try:
        # Export risk premium estimates
        if 'historical' in results and results['historical']:
            risk_premium_data = {}
            for exposure_id, estimate in results['historical'].items():
                risk_premium_data[exposure_id] = {
                    'risk_premium_volatility': float(estimate.risk_premium_volatility),
                    'total_volatility': float(estimate.total_volatility),
                    'inflation_volatility': float(estimate.inflation_volatility),
                    'real_rf_volatility': float(estimate.real_rf_volatility),
                    'sample_size': int(estimate.sample_size)
                }
            
            with open('full_universe_risk_premium_estimates.json', 'w') as f:
                json.dump(risk_premium_data, f, indent=2)
            print("✅ Exported risk premium estimates to full_universe_risk_premium_estimates.json")
        
        # Export correlation matrix
        if not correlation_matrix.empty:
            correlation_matrix.to_csv('full_universe_risk_premium_correlations.csv')
            print("✅ Exported correlation matrix to full_universe_risk_premium_correlations.csv")
        
        # Export analysis dataframe
        if analysis_df is not None and not analysis_df.empty:
            analysis_df.to_csv('full_universe_risk_premium_analysis.csv', index=False)
            print("✅ Exported analysis to full_universe_risk_premium_analysis.csv")
        
        print(f"\n📊 Export Summary:")
        print(f"  Risk Premium Estimates: {len(results.get('historical', {}))} exposures")
        print(f"  Correlation Matrix: {correlation_matrix.shape if not correlation_matrix.empty else 'None'}")
        print(f"  Analysis Data: {len(analysis_df) if analysis_df is not None else 0} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def main():
    """Run the complete full universe risk premium analysis."""
    print("FULL UNIVERSE RISK PREMIUM ANALYSIS")
    print("=" * 60)
    
    try:
        # Setup
        universe = load_full_exposure_universe()
        if not universe:
            print("❌ Failed to load universe")
            return
        
        risk_estimator = RiskPremiumEstimator(universe, ReturnDecomposer())
        
        # Parameters
        estimation_date = datetime.now()
        forecast_horizon = 252  # 1 year
        
        print(f"\nAnalysis Parameters:")
        print(f"  Estimation Date: {estimation_date.date()}")
        print(f"  Forecast Horizon: {forecast_horizon} days (1 year)")
        print(f"  Lookback Period: 1260 days (5 years)")
        print(f"  Frequency: Monthly")
        
        # 1. Full universe risk premium estimation
        results = test_full_universe_risk_premium_estimation(
            risk_estimator, universe, estimation_date, forecast_horizon
        )
        
        # 2. Build correlation matrix
        correlation_matrix = build_full_universe_correlation_matrix(
            risk_estimator, universe, estimation_date
        )
        
        # 3. Component analysis
        analysis_df = analyze_risk_premium_components(results)
        
        # 4. Create visualization
        viz_success = create_full_universe_visualization(results, correlation_matrix, analysis_df)
        
        # 5. Export results
        export_success = export_results(results, correlation_matrix, analysis_df)
        
        # Final summary
        print(f"\n" + "="*60)
        print("🎉 FULL UNIVERSE ANALYSIS COMPLETE!")
        print("="*60)
        
        successful_estimates = len(results.get('historical', {}))
        total_exposures = len(universe.exposures)
        
        print(f"\n✅ RESULTS SUMMARY:")
        print(f"  Total Exposures in Universe: {total_exposures}")
        print(f"  Successful Risk Premium Estimates: {successful_estimates}")
        print(f"  Success Rate: {successful_estimates/total_exposures*100:.1f}%")
        print(f"  Correlation Matrix: {correlation_matrix.shape if not correlation_matrix.empty else 'Failed'}")
        print(f"  Visualization: {'Success' if viz_success else 'Failed'}")
        print(f"  Data Export: {'Success' if export_success else 'Failed'}")
        
        if analysis_df is not None and not analysis_df.empty:
            print(f"\n📊 KEY INSIGHTS:")
            print(f"  Average Risk Premium %: {analysis_df['risk_premium_percentage'].mean():.1f}%")
            print(f"  Highest Risk Premium: {analysis_df.loc[analysis_df['risk_premium_percentage'].idxmax(), 'exposure_id']}")
            print(f"  Lowest Risk Premium: {analysis_df.loc[analysis_df['risk_premium_percentage'].idxmin(), 'exposure_id']}")
            print(f"  Average Total Volatility: {analysis_df['total_volatility'].mean():.1%}")
            print(f"  Average Uncompensated Risk: {analysis_df['uncompensated_risk'].mean():.1%}")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"  1. Review exported data files for portfolio optimization")
        print(f"  2. Implement parameter optimization for EWMA/GARCH methods")
        print(f"  3. Build portfolio optimization using risk premium estimates")
        print(f"  4. Compare portfolio performance vs traditional approaches")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()