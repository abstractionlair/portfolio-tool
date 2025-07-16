#!/usr/bin/env python3
"""
Portfolio-Level Optimization Analysis Demo

This script demonstrates how to use the portfolio-level analysis tools
to analyze sophisticated two-stage optimization results.

The analysis includes:
- Parameter quality assessment
- Validation performance analysis
- Method selection effectiveness
- Comprehensive visualizations
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis import PortfolioLevelAnalyzer, PortfolioOptimizationVisualizer

def main():
    """Main demonstration function"""
    
    print("Portfolio-Level Optimization Analysis Demo")
    print("==========================================")
    print()
    
    # Initialize analyzer and visualizer
    analyzer = PortfolioLevelAnalyzer()
    visualizer = PortfolioOptimizationVisualizer()
    
    try:
        # Load results
        print("1. Loading optimization results...")
        results = analyzer.load_results()
        print(f"   ✅ Loaded results for {len(results.volatility_parameters)} exposures")
        print(f"   ✅ Optimal horizon: {results.optimal_horizon} days")
        print(f"   ✅ Portfolio RMSE: {results.portfolio_rmse:.4f}")
        print()
        
        # Basic analysis
        print("2. Basic analysis...")
        exposure_summary = analyzer.get_exposure_summary()
        method_distribution = analyzer.get_method_distribution()
        validation_metrics = analyzer.get_validation_metrics()
        
        print("   Method Distribution:")
        for method, count in method_distribution.items():
            print(f"     {method}: {count} exposures")
        print()
        
        # Horizon comparison
        print("3. Horizon comparison...")
        horizon_df = analyzer.get_horizon_comparison()
        print("   Horizon Performance:")
        for _, row in horizon_df.iterrows():
            marker = "🏆" if row['horizon'] == f"{results.optimal_horizon}_day" else "  "
            print(f"   {marker} {row['horizon_days']:2d} days: {row['portfolio_rmse']:.4f} RMSE")
        print()
        
        # Validation quality
        print("4. Validation quality...")
        test_portfolios_df = analyzer.get_test_portfolios_performance()
        mean_error = test_portfolios_df['relative_error'].mean()
        print(f"   Mean relative error: {mean_error:.1%}")
        print(f"   Number of test portfolios: {len(test_portfolios_df)}")
        print()
        
        # Parameter analysis
        print("5. Parameter analysis...")
        lambda_analysis = analyzer.get_lambda_distribution()
        if not lambda_analysis.empty:
            print(f"   EWMA lambda range: {lambda_analysis['lambda'].min():.3f} - {lambda_analysis['lambda'].max():.3f}")
            print(f"   EWMA lambda mean: {lambda_analysis['lambda'].mean():.3f}")
        print()
        
        # Generate visualizations
        print("6. Generating visualizations...")
        
        # Horizon comparison plot
        fig1 = visualizer.plot_horizon_comparison(horizon_df)
        plt.savefig('horizon_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Method distribution plot
        fig2 = visualizer.plot_method_distribution(method_distribution)
        plt.savefig('method_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Validation quality plot
        fig3 = visualizer.plot_validation_quality(test_portfolios_df)
        plt.savefig('validation_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter analysis plot
        fig4 = visualizer.plot_parameter_analysis(exposure_summary)
        plt.savefig('parameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Static plots saved as PNG files")
        print()
        
        # Generate interactive dashboard
        print("7. Generating interactive dashboard...")
        dashboard = visualizer.plot_comprehensive_dashboard(analyzer)
        dashboard.write_html('portfolio_optimization_dashboard.html')
        print("   ✅ Interactive dashboard saved as HTML file")
        print()
        
        # Export summary report
        print("8. Exporting summary report...")
        report_file = analyzer.export_summary_report('portfolio_optimization_summary.txt')
        print(f"   ✅ Summary report saved to: {report_file}")
        print()
        
        # Key insights
        print("9. Key insights...")
        improvement_metrics = analyzer.calculate_improvement_metrics()
        
        print(f"   🎯 Optimal horizon: {results.optimal_horizon} days")
        print(f"   📊 Portfolio RMSE: {results.portfolio_rmse:.4f} ({results.portfolio_rmse*100:.2f}%)")
        print(f"   🚀 RMSE improvement: {improvement_metrics['rmse_improvement']:.1%}")
        print(f"   ⚡ Method diversity: {len(method_distribution)} methods used")
        print(f"   ✅ Validation tests: {validation_metrics['n_tests']}")
        print()
        
        print("✅ Portfolio-level optimization analysis complete!")
        print("   All results, visualizations, and insights have been generated.")
        print("   The system is production-ready with sophisticated parameter selection.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please ensure portfolio-level optimization has been run first.")
        print("   Run: python examples/portfolio_level_optimization_demo.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()