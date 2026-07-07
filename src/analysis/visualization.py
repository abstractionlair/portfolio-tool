"""
Visualization functions for portfolio-level optimization results

This module provides comprehensive visualization capabilities for analyzing
parameter quality, validation performance, and method selection effectiveness.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

class PortfolioOptimizationVisualizer:
    """
    Visualization class for portfolio-level optimization results
    
    Provides static (matplotlib/seaborn) and interactive (plotly) visualizations
    for analyzing parameter quality and validation performance.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = sns.color_palette('husl', n_colors=10)
        
    def plot_horizon_comparison(self, horizon_df: pd.DataFrame, 
                              metric: str = 'portfolio_rmse',
                              interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot comparison of different forecast horizons
        
        Args:
            horizon_df: DataFrame with horizon comparison data
            metric: 'portfolio_rmse' or 'goodness_score'
            interactive: Whether to create interactive plotly chart
        """
        
        if interactive:
            fig = px.bar(
                horizon_df, 
                x='horizon_days', 
                y=metric,
                title=f'Horizon Comparison - {metric.replace("_", " ").title()}',
                labels={'horizon_days': 'Forecast Horizon (Days)', 
                       metric: metric.replace('_', ' ').title()}
            )
            
            # Highlight optimal horizon
            optimal_idx = horizon_df[metric].idxmax() if metric == 'goodness_score' else horizon_df[metric].idxmin()
            optimal_horizon = horizon_df.loc[optimal_idx, 'horizon_days']
            
            fig.add_vline(x=optimal_horizon, line_dash="dash", line_color="red",
                         annotation_text=f"Optimal: {optimal_horizon} days")
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            bars = ax.bar(horizon_df['horizon_days'], horizon_df[metric], 
                         color=self.colors[0], alpha=0.7)
            
            # Highlight optimal horizon
            optimal_idx = horizon_df[metric].idxmax() if metric == 'goodness_score' else horizon_df[metric].idxmin()
            bars[optimal_idx].set_color(self.colors[1])
            bars[optimal_idx].set_alpha(1.0)
            
            ax.set_xlabel('Forecast Horizon (Days)')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Horizon Comparison - {metric.replace("_", " ").title()}')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            return fig
            
    def plot_method_distribution(self, method_counts: pd.Series, 
                               interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot distribution of optimization methods across exposures
        
        Args:
            method_counts: Series with method counts
            interactive: Whether to create interactive plotly chart
        """
        
        if interactive:
            fig = px.pie(
                values=method_counts.values,
                names=method_counts.index,
                title='Method Distribution Across Exposures'
            )
            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # Pie chart
            wedges, texts, autotexts = ax1.pie(method_counts.values, 
                                              labels=method_counts.index,
                                              autopct='%1.1f%%',
                                              colors=self.colors[:len(method_counts)])
            ax1.set_title('Method Distribution')
            
            # Bar chart
            bars = ax2.bar(method_counts.index, method_counts.values, 
                          color=self.colors[:len(method_counts)])
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Number of Exposures')
            ax2.set_title('Method Count')
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            plt.tight_layout()
            return fig
            
    def plot_validation_quality(self, test_portfolios_df: pd.DataFrame,
                               interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot validation quality showing predicted vs realized volatility
        
        Args:
            test_portfolios_df: DataFrame with test portfolio results
            interactive: Whether to create interactive plotly chart
        """
        
        if interactive:
            fig = px.scatter(
                test_portfolios_df,
                x='realized_vol',
                y='predicted_vol',
                color='portfolio_type',
                title='Validation Quality: Predicted vs Realized Volatility',
                labels={'realized_vol': 'Realized Volatility',
                       'predicted_vol': 'Predicted Volatility'}
            )
            
            # Add perfect prediction line
            min_val = min(test_portfolios_df['realized_vol'].min(), 
                         test_portfolios_df['predicted_vol'].min())
            max_val = max(test_portfolios_df['realized_vol'].max(), 
                         test_portfolios_df['predicted_vol'].max())
            
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash"),
            )
            
            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # Scatter plot with perfect prediction line
            scatter = ax1.scatter(test_portfolios_df['realized_vol'], 
                                test_portfolios_df['predicted_vol'],
                                c=[self.colors[i] for i in range(len(test_portfolios_df))],
                                alpha=0.7, s=100)
            
            # Perfect prediction line
            min_val = min(test_portfolios_df['realized_vol'].min(), 
                         test_portfolios_df['predicted_vol'].min())
            max_val = max(test_portfolios_df['realized_vol'].max(), 
                         test_portfolios_df['predicted_vol'].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
            
            ax1.set_xlabel('Realized Volatility')
            ax1.set_ylabel('Predicted Volatility')
            ax1.set_title('Predicted vs Realized Volatility')
            ax1.legend()
            
            # Error distribution
            ax2.hist(test_portfolios_df['relative_error'], bins=10, 
                    color=self.colors[0], alpha=0.7, edgecolor='black')
            ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Relative Error')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Prediction Error Distribution')
            
            plt.tight_layout()
            return fig
            
    def plot_parameter_analysis(self, exposure_summary: pd.DataFrame,
                              interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot parameter analysis for different methods
        
        Args:
            exposure_summary: DataFrame with exposure parameters
            interactive: Whether to create interactive plotly chart
        """
        
        if interactive:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Lambda Distribution (EWMA)', 'Lookback Days by Method',
                               'Validation Scores by Method', 'Parameter Correlation')
            )
            
            # Lambda distribution for EWMA
            ewma_data = exposure_summary[exposure_summary['method'] == 'ewma']
            if not ewma_data.empty:
                fig.add_trace(
                    go.Histogram(x=ewma_data['lambda'], name='Lambda'),
                    row=1, col=1
                )
            
            # Lookback days by method
            for method in exposure_summary['method'].unique():
                method_data = exposure_summary[exposure_summary['method'] == method]
                fig.add_trace(
                    go.Box(y=method_data['lookback_days'], name=method),
                    row=1, col=2
                )
            
            # Validation scores by method
            for method in exposure_summary['method'].unique():
                method_data = exposure_summary[exposure_summary['method'] == method]
                fig.add_trace(
                    go.Box(y=method_data['validation_score'], name=method),
                    row=2, col=1
                )
            
            # Lambda vs Lookback correlation for EWMA
            if not ewma_data.empty:
                fig.add_trace(
                    go.Scatter(x=ewma_data['lambda'], y=ewma_data['lookback_days'],
                              mode='markers', name='EWMA Parameters'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text="Parameter Analysis Dashboard")
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Lambda distribution for EWMA
            ewma_data = exposure_summary[exposure_summary['method'] == 'ewma']
            if not ewma_data.empty:
                axes[0, 0].hist(ewma_data['lambda'], bins=10, color=self.colors[0], 
                               alpha=0.7, edgecolor='black')
                axes[0, 0].set_xlabel('Lambda')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Lambda Distribution (EWMA)')
            
            # Lookback days by method
            methods = exposure_summary['method'].unique()
            lookback_data = [exposure_summary[exposure_summary['method'] == method]['lookback_days'] 
                           for method in methods]
            axes[0, 1].boxplot(lookback_data, labels=methods)
            axes[0, 1].set_ylabel('Lookback Days')
            axes[0, 1].set_title('Lookback Days by Method')
            
            # Validation scores by method
            validation_data = [exposure_summary[exposure_summary['method'] == method]['validation_score'].dropna() 
                             for method in methods]
            axes[1, 0].boxplot(validation_data, labels=methods)
            axes[1, 0].set_ylabel('Validation Score')
            axes[1, 0].set_title('Validation Scores by Method')
            
            # Lambda vs Lookback correlation for EWMA
            if not ewma_data.empty:
                axes[1, 1].scatter(ewma_data['lambda'], ewma_data['lookback_days'],
                                  color=self.colors[0], alpha=0.7, s=100)
                axes[1, 1].set_xlabel('Lambda')
                axes[1, 1].set_ylabel('Lookback Days')
                axes[1, 1].set_title('EWMA Parameters Correlation')
            
            plt.tight_layout()
            return fig
            
    def plot_exposure_heatmap(self, exposure_summary: pd.DataFrame,
                            interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot heatmap of exposure parameters
        
        Args:
            exposure_summary: DataFrame with exposure parameters
            interactive: Whether to create interactive plotly chart
        """
        
        # Prepare data for heatmap
        heatmap_data = exposure_summary.pivot_table(
            index='exposure', 
            columns='method', 
            values='lambda', 
            fill_value=0
        )
        
        if interactive:
            fig = px.imshow(
                heatmap_data,
                title='Exposure-Method Parameter Heatmap',
                labels={'x': 'Method', 'y': 'Exposure', 'color': 'Lambda'},
                aspect='auto'
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', 
                       cmap='viridis', ax=ax, cbar_kws={'label': 'Lambda'})
            ax.set_title('Exposure-Method Parameter Heatmap')
            ax.set_xlabel('Method')
            ax.set_ylabel('Exposure')
            
            plt.tight_layout()
            return fig
            
    def plot_optimization_performance_curve(self, analyzer, 
                                           interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot the optimization performance curve across all horizons
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        horizon_df = analyzer.get_horizon_efficiency_analysis()
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio RMSE by Horizon', 'Efficiency Score by Horizon',
                               'Relative Performance', 'Rebalancing Frequency vs Performance'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # RMSE curve
            fig.add_trace(
                go.Scatter(x=horizon_df['horizon_days'], y=horizon_df['portfolio_rmse'],
                          mode='lines+markers', name='Portfolio RMSE',
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )
            
            # Highlight optimal point
            optimal_idx = horizon_df['portfolio_rmse'].idxmin()
            optimal_point = horizon_df.loc[optimal_idx]
            fig.add_trace(
                go.Scatter(x=[optimal_point['horizon_days']], y=[optimal_point['portfolio_rmse']],
                          mode='markers', name='Optimal Horizon',
                          marker=dict(color='red', size=12, symbol='star')),
                row=1, col=1
            )
            
            # Efficiency score
            fig.add_trace(
                go.Scatter(x=horizon_df['horizon_days'], y=horizon_df['efficiency_score'],
                          mode='lines+markers', name='Efficiency Score',
                          line=dict(color='green', width=3)),
                row=1, col=2
            )
            
            # Relative performance
            fig.add_trace(
                go.Bar(x=horizon_df['horizon_days'], y=horizon_df['relative_performance'],
                       name='Relative Performance'),
                row=2, col=1
            )
            
            # Rebalancing frequency vs performance
            fig.add_trace(
                go.Scatter(x=horizon_df['rebalancing_frequency'], y=horizon_df['portfolio_rmse'],
                          mode='markers', name='Rebalancing vs RMSE',
                          marker=dict(size=10)),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text="Portfolio Optimization Performance Analysis")
            
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # RMSE curve
            axes[0, 0].plot(horizon_df['horizon_days'], horizon_df['portfolio_rmse'], 
                           'b-o', linewidth=2, markersize=6)
            optimal_idx = horizon_df['portfolio_rmse'].idxmin()
            optimal_point = horizon_df.loc[optimal_idx]
            axes[0, 0].scatter(optimal_point['horizon_days'], optimal_point['portfolio_rmse'],
                              color='red', s=100, marker='*', zorder=5)
            axes[0, 0].set_xlabel('Horizon (Days)')
            axes[0, 0].set_ylabel('Portfolio RMSE')
            axes[0, 0].set_title('Portfolio RMSE by Horizon')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Efficiency score
            axes[0, 1].plot(horizon_df['horizon_days'], horizon_df['efficiency_score'], 
                           'g-o', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('Horizon (Days)')
            axes[0, 1].set_ylabel('Efficiency Score (1/RMSE)')
            axes[0, 1].set_title('Efficiency Score by Horizon')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Relative performance
            axes[1, 0].bar(horizon_df['horizon_days'], horizon_df['relative_performance'],
                          color=self.colors[0], alpha=0.7)
            axes[1, 0].set_xlabel('Horizon (Days)')
            axes[1, 0].set_ylabel('Relative Performance')
            axes[1, 0].set_title('Relative Performance (vs Optimal)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Rebalancing frequency vs performance
            axes[1, 1].scatter(horizon_df['rebalancing_frequency'], horizon_df['portfolio_rmse'],
                              color=self.colors[1], s=100, alpha=0.7)
            axes[1, 1].set_xlabel('Rebalancing Frequency (Times/Year)')
            axes[1, 1].set_ylabel('Portfolio RMSE')
            axes[1, 1].set_title('Rebalancing Frequency vs Performance')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
    def plot_parameter_effectiveness(self, analyzer, 
                                   interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot parameter effectiveness analysis
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        exposure_summary = analyzer.get_exposure_summary()
        method_analysis = analyzer.get_parameter_effectiveness_analysis()
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Lambda Distribution by Exposure', 'Lookback Days by Method',
                               'Parameter Ranges by Method', 'Method Usage'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "pie"}]]
            )
            
            # Lambda distribution by exposure (EWMA only)
            ewma_data = exposure_summary[exposure_summary['method'] == 'ewma']
            if not ewma_data.empty:
                fig.add_trace(
                    go.Bar(x=ewma_data['exposure'], y=ewma_data['lambda'],
                           name='Lambda Values'),
                    row=1, col=1
                )
            
            # Lookback days by method
            for method in exposure_summary['method'].unique():
                method_data = exposure_summary[exposure_summary['method'] == method]
                fig.add_trace(
                    go.Box(y=method_data['lookback_days'], name=f'{method} Lookback'),
                    row=1, col=2
                )
            
            # Parameter ranges visualization
            methods = list(method_analysis.keys())
            avg_lookbacks = [method_analysis[m]['avg_lookback'] for m in methods]
            fig.add_trace(
                go.Bar(x=methods, y=avg_lookbacks, name='Avg Lookback'),
                row=2, col=1
            )
            
            # Method usage pie chart
            method_counts = exposure_summary['method'].value_counts()
            fig.add_trace(
                go.Pie(labels=method_counts.index, values=method_counts.values,
                       name='Method Usage'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text="Parameter Effectiveness Analysis")
            
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Lambda distribution by exposure (EWMA only)
            ewma_data = exposure_summary[exposure_summary['method'] == 'ewma']
            if not ewma_data.empty:
                axes[0, 0].bar(range(len(ewma_data)), ewma_data['lambda'], 
                              color=self.colors[0], alpha=0.7)
                axes[0, 0].set_xticks(range(len(ewma_data)))
                axes[0, 0].set_xticklabels(ewma_data['exposure'], rotation=45, ha='right')
                axes[0, 0].set_ylabel('Lambda Value')
                axes[0, 0].set_title('Lambda Distribution by Exposure (EWMA)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Lookback days by method
            methods = exposure_summary['method'].unique()
            lookback_data = [exposure_summary[exposure_summary['method'] == method]['lookback_days'] 
                           for method in methods]
            axes[0, 1].boxplot(lookback_data, labels=methods)
            axes[0, 1].set_ylabel('Lookback Days')
            axes[0, 1].set_title('Lookback Days by Method')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Parameter ranges
            methods = list(method_analysis.keys())
            avg_lookbacks = [method_analysis[m]['avg_lookback'] for m in methods]
            axes[1, 0].bar(methods, avg_lookbacks, color=self.colors[1], alpha=0.7)
            axes[1, 0].set_ylabel('Average Lookback Days')
            axes[1, 0].set_title('Average Lookback by Method')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Method usage
            method_counts = exposure_summary['method'].value_counts()
            axes[1, 1].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Method Usage Distribution')
            
            plt.tight_layout()
            return fig
            
    def plot_exposure_analysis(self, analyzer, 
                              interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot detailed exposure analysis
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        exposure_summary = analyzer.get_exposure_summary()
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Exposure Methods', 'Lambda vs Lookback (EWMA)',
                               'Lookback Distribution', 'Method by Exposure Type'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Exposure methods
            fig.add_trace(
                go.Bar(x=exposure_summary['exposure'], y=[1]*len(exposure_summary),
                       customdata=exposure_summary['method'],
                       name='Exposure Methods'),
                row=1, col=1
            )
            
            # Lambda vs Lookback for EWMA
            ewma_data = exposure_summary[exposure_summary['method'] == 'ewma']
            if not ewma_data.empty:
                fig.add_trace(
                    go.Scatter(x=ewma_data['lambda'], y=ewma_data['lookback_days'],
                              mode='markers', name='EWMA Parameters',
                              text=ewma_data['exposure'],
                              marker=dict(size=10)),
                    row=1, col=2
                )
            
            # Lookback distribution
            fig.add_trace(
                go.Histogram(x=exposure_summary['lookback_days'], name='Lookback Days'),
                row=2, col=1
            )
            
            # Method by exposure type (simplified categorization)
            exposure_summary['asset_type'] = exposure_summary['exposure'].apply(
                lambda x: 'Equity' if 'equity' in x else 
                         'Bond' if any(bond in x for bond in ['ust', 'bonds', 'tips']) else
                         'Alternative'
            )
            
            asset_method_counts = exposure_summary.groupby(['asset_type', 'method']).size().reset_index(name='count')
            for asset_type in asset_method_counts['asset_type'].unique():
                asset_data = asset_method_counts[asset_method_counts['asset_type'] == asset_type]
                fig.add_trace(
                    go.Bar(x=asset_data['method'], y=asset_data['count'],
                           name=f'{asset_type} Methods'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text="Detailed Exposure Analysis")
            
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Exposure methods
            method_colors = {method: self.colors[i] for i, method in enumerate(exposure_summary['method'].unique())}
            colors = [method_colors[method] for method in exposure_summary['method']]
            axes[0, 0].bar(range(len(exposure_summary)), [1]*len(exposure_summary), color=colors)
            axes[0, 0].set_xticks(range(len(exposure_summary)))
            axes[0, 0].set_xticklabels(exposure_summary['exposure'], rotation=45, ha='right')
            axes[0, 0].set_ylabel('Method')
            axes[0, 0].set_title('Method by Exposure')
            
            # Lambda vs Lookback for EWMA
            ewma_data = exposure_summary[exposure_summary['method'] == 'ewma']
            if not ewma_data.empty:
                axes[0, 1].scatter(ewma_data['lambda'], ewma_data['lookback_days'],
                                  color=self.colors[0], s=100, alpha=0.7)
                axes[0, 1].set_xlabel('Lambda')
                axes[0, 1].set_ylabel('Lookback Days')
                axes[0, 1].set_title('Lambda vs Lookback (EWMA)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Lookback distribution
            axes[1, 0].hist(exposure_summary['lookback_days'], bins=10, 
                           color=self.colors[1], alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Lookback Days')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Lookback Days Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Method by exposure type
            exposure_summary['asset_type'] = exposure_summary['exposure'].apply(
                lambda x: 'Equity' if 'equity' in x else 
                         'Bond' if any(bond in x for bond in ['ust', 'bonds', 'tips']) else
                         'Alternative'
            )
            
            asset_method_counts = exposure_summary.groupby(['asset_type', 'method']).size().unstack(fill_value=0)
            asset_method_counts.plot(kind='bar', ax=axes[1, 1], color=self.colors[:len(asset_method_counts.columns)])
            axes[1, 1].set_xlabel('Asset Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Method Usage by Asset Type')
            axes[1, 1].legend(title='Method')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
            
    def plot_risk_estimates_results(self, analyzer, 
                                   interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot the final risk estimates (volatilities and correlations)
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        exposure_vols = analyzer.get_exposure_volatilities()
        correlation_matrix = analyzer.get_correlation_matrix_results()
        risk_breakdown = analyzer.compute_portfolio_risk_breakdown()
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Exposure Volatilities by Method', 'Correlation Matrix Heatmap',
                               'Portfolio Risk Breakdown', 'Prediction vs Realized Volatility'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Exposure volatilities (placeholder - would need actual computed values)
            if 'annualized_volatility' in exposure_vols.columns and exposure_vols['annualized_volatility'].notna().any():
                for method in exposure_vols['method'].unique():
                    method_data = exposure_vols[exposure_vols['method'] == method]
                    fig.add_trace(
                        go.Bar(x=method_data['exposure'], y=method_data['annualized_volatility'],
                               name=f'{method} Volatility'),
                        row=1, col=1
                    )
            else:
                # Show placeholder message
                fig.add_annotation(
                    text="Volatilities not computed in current results<br>Run portfolio optimization to generate",
                    xref="x", yref="y", x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    row=1, col=1, showarrow=False
                )
            
            # Correlation matrix heatmap
            if not correlation_matrix.empty and not correlation_matrix.equals(pd.DataFrame(np.eye(len(correlation_matrix)))):
                fig.add_trace(
                    go.Heatmap(z=correlation_matrix.values,
                              x=correlation_matrix.columns,
                              y=correlation_matrix.index,
                              colorscale='RdBu',
                              zmid=0,
                              name='Correlations'),
                    row=1, col=2
                )
            else:
                fig.add_annotation(
                    text="Correlation matrix not available<br>in current results",
                    xref="x", yref="y", x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    row=1, col=2, showarrow=False
                )
            
            # Portfolio risk breakdown
            risk_metrics = [
                risk_breakdown['average_predicted_vol'],
                risk_breakdown['average_realized_vol'],
                risk_breakdown['prediction_accuracy']['rmse']
            ]
            metric_names = ['Avg Predicted Vol', 'Avg Realized Vol', 'RMSE']
            
            fig.add_trace(
                go.Bar(x=metric_names, y=risk_metrics, name='Risk Metrics'),
                row=2, col=1
            )
            
            # Prediction accuracy scatter
            test_portfolios = analyzer.get_test_portfolios_performance()
            fig.add_trace(
                go.Scatter(x=test_portfolios['realized_vol'], y=test_portfolios['predicted_vol'],
                          mode='markers', name='Portfolio Tests',
                          marker=dict(size=8)),
                row=2, col=2
            )
            
            # Add perfect prediction line
            min_vol = min(test_portfolios['realized_vol'].min(), test_portfolios['predicted_vol'].min())
            max_vol = max(test_portfolios['realized_vol'].max(), test_portfolios['predicted_vol'].max())
            fig.add_trace(
                go.Scatter(x=[min_vol, max_vol], y=[min_vol, max_vol],
                          mode='lines', name='Perfect Prediction',
                          line=dict(dash='dash', color='red')),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text="Risk Estimates and Results Analysis")
            
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Exposure volatilities
            if 'annualized_volatility' in exposure_vols.columns and exposure_vols['annualized_volatility'].notna().any():
                methods = exposure_vols['method'].unique()
                for i, method in enumerate(methods):
                    method_data = exposure_vols[exposure_vols['method'] == method]
                    axes[0, 0].bar(method_data['exposure'], method_data['annualized_volatility'],
                                  alpha=0.7, label=method, color=self.colors[i])
                axes[0, 0].set_ylabel('Annualized Volatility')
                axes[0, 0].set_title('Exposure Volatilities by Method')
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis='x', rotation=45)
            else:
                axes[0, 0].text(0.5, 0.5, 'Volatilities not computed\nin current results\n\nRun portfolio optimization\nto generate',
                               ha='center', va='center', transform=axes[0, 0].transAxes,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[0, 0].set_title('Exposure Volatilities')
            
            # Correlation matrix heatmap
            if not correlation_matrix.empty and not correlation_matrix.equals(pd.DataFrame(np.eye(len(correlation_matrix)))):
                im = axes[0, 1].imshow(correlation_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
                axes[0, 1].set_xticks(range(len(correlation_matrix.columns)))
                axes[0, 1].set_yticks(range(len(correlation_matrix.index)))
                axes[0, 1].set_xticklabels(correlation_matrix.columns, rotation=45)
                axes[0, 1].set_yticklabels(correlation_matrix.index)
                axes[0, 1].set_title('Correlation Matrix')
                fig.colorbar(im, ax=axes[0, 1])
            else:
                axes[0, 1].text(0.5, 0.5, 'Correlation matrix not\navailable in current results',
                               ha='center', va='center', transform=axes[0, 1].transAxes,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[0, 1].set_title('Correlation Matrix')
            
            # Portfolio risk breakdown
            risk_metrics = [
                risk_breakdown['average_predicted_vol'],
                risk_breakdown['average_realized_vol'],
                risk_breakdown['prediction_accuracy']['rmse']
            ]
            metric_names = ['Avg Predicted\nVol', 'Avg Realized\nVol', 'RMSE']
            
            bars = axes[1, 0].bar(metric_names, risk_metrics, color=self.colors[:3], alpha=0.7)
            axes[1, 0].set_ylabel('Volatility')
            axes[1, 0].set_title('Portfolio Risk Breakdown')
            
            # Add value labels on bars
            for bar, value in zip(bars, risk_metrics):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                               f'{value:.4f}', ha='center', va='bottom')
            
            # Prediction accuracy scatter
            test_portfolios = analyzer.get_test_portfolios_performance()
            axes[1, 1].scatter(test_portfolios['realized_vol'], test_portfolios['predicted_vol'],
                              alpha=0.7, s=100, color=self.colors[0])
            
            # Add perfect prediction line
            min_vol = min(test_portfolios['realized_vol'].min(), test_portfolios['predicted_vol'].min())
            max_vol = max(test_portfolios['realized_vol'].max(), test_portfolios['predicted_vol'].max())
            axes[1, 1].plot([min_vol, max_vol], [min_vol, max_vol], 'r--', alpha=0.7, label='Perfect Prediction')
            
            axes[1, 1].set_xlabel('Realized Volatility')
            axes[1, 1].set_ylabel('Predicted Volatility')
            axes[1, 1].set_title('Prediction vs Realized Volatility')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
    def plot_portfolio_composition_analysis(self, analyzer, 
                                          interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot portfolio composition analysis from test portfolios
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        detailed_results = analyzer.get_detailed_validation_results()
        
        if not detailed_results:
            return None
            
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio Weights Distribution', 'Top Holdings by Portfolio',
                               'Diversification Analysis', 'Risk Contribution Analysis'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Portfolio weights distribution
            all_weights = []
            for i, result in enumerate(detailed_results):
                weights = result['portfolio']
                for exposure, weight in weights.items():
                    if weight > 0.001:  # Only show meaningful weights
                        all_weights.append({'portfolio': i, 'exposure': exposure, 'weight': weight})
            
            weights_df = pd.DataFrame(all_weights)
            
            if not weights_df.empty:
                fig.add_trace(
                    go.Histogram(x=weights_df['weight'], nbinsx=20, name='Weight Distribution'),
                    row=1, col=1
                )
                
                # Top holdings analysis
                top_holdings = weights_df.groupby('exposure')['weight'].agg(['mean', 'max', 'count']).reset_index()
                top_holdings = top_holdings.sort_values('mean', ascending=False).head(10)
                
                fig.add_trace(
                    go.Bar(x=top_holdings['exposure'], y=top_holdings['mean'], name='Avg Weight'),
                    row=1, col=2
                )
                
                # Diversification analysis (effective number of assets)
                diversification_data = []
                for i, result in enumerate(detailed_results):
                    weights = list(result['portfolio'].values())
                    effective_n = 1 / sum(w**2 for w in weights if w > 0)
                    diversification_data.append(effective_n)
                
                fig.add_trace(
                    go.Histogram(x=diversification_data, nbinsx=10, name='Effective N Assets'),
                    row=2, col=1
                )
                
                # Risk contribution (weights vs volatility prediction errors)
                test_portfolios = analyzer.get_test_portfolios_performance()
                fig.add_trace(
                    go.Scatter(x=test_portfolios['predicted_vol'], y=test_portfolios['abs_error'],
                              mode='markers', name='Vol vs Error',
                              marker=dict(size=8)),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text="Portfolio Composition Analysis")
            
            return fig
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract portfolio weights
            all_weights = []
            for i, result in enumerate(detailed_results):
                weights = result['portfolio']
                for exposure, weight in weights.items():
                    if weight > 0.001:  # Only show meaningful weights
                        all_weights.append({'portfolio': i, 'exposure': exposure, 'weight': weight})
            
            weights_df = pd.DataFrame(all_weights)
            
            if not weights_df.empty:
                # Portfolio weights distribution
                axes[0, 0].hist(weights_df['weight'], bins=20, alpha=0.7, color=self.colors[0])
                axes[0, 0].set_xlabel('Portfolio Weight')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Portfolio Weights Distribution')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Top holdings analysis
                top_holdings = weights_df.groupby('exposure')['weight'].agg(['mean', 'max', 'count']).reset_index()
                top_holdings = top_holdings.sort_values('mean', ascending=False).head(10)
                
                axes[0, 1].bar(range(len(top_holdings)), top_holdings['mean'], color=self.colors[1], alpha=0.7)
                axes[0, 1].set_xticks(range(len(top_holdings)))
                axes[0, 1].set_xticklabels(top_holdings['exposure'], rotation=45, ha='right')
                axes[0, 1].set_ylabel('Average Weight')
                axes[0, 1].set_title('Top Holdings by Average Weight')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Diversification analysis
                diversification_data = []
                for i, result in enumerate(detailed_results):
                    weights = list(result['portfolio'].values())
                    effective_n = 1 / sum(w**2 for w in weights if w > 0)
                    diversification_data.append(effective_n)
                
                axes[1, 0].hist(diversification_data, bins=10, alpha=0.7, color=self.colors[2])
                axes[1, 0].set_xlabel('Effective Number of Assets')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Diversification Analysis')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Risk contribution
                test_portfolios = analyzer.get_test_portfolios_performance()
                axes[1, 1].scatter(test_portfolios['predicted_vol'], test_portfolios['abs_error'],
                                  alpha=0.7, s=100, color=self.colors[3])
                axes[1, 1].set_xlabel('Predicted Volatility')
                axes[1, 1].set_ylabel('Absolute Error')
                axes[1, 1].set_title('Volatility vs Prediction Error')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
    def plot_comprehensive_dashboard(self, analyzer) -> go.Figure:
        """
        Create comprehensive interactive dashboard
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
        """
        
        # Get data
        horizon_df = analyzer.get_horizon_comparison()
        method_counts = analyzer.get_method_distribution()
        test_portfolios_df = analyzer.get_test_portfolios_performance()
        exposure_summary = analyzer.get_exposure_summary()
        
        # Create subplots with proper pie chart spec
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Horizon Comparison', 'Method Distribution',
                           'Validation Quality', 'Parameter Analysis',
                           'Performance by Portfolio Type', 'Exposure Overview'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Horizon comparison
        fig.add_trace(
            go.Bar(x=horizon_df['horizon_days'], y=horizon_df['portfolio_rmse'],
                   name='Portfolio RMSE'),
            row=1, col=1
        )
        
        # Method distribution
        fig.add_trace(
            go.Pie(labels=method_counts.index, values=method_counts.values,
                   name='Methods'),
            row=1, col=2
        )
        
        # Validation quality
        fig.add_trace(
            go.Scatter(x=test_portfolios_df['realized_vol'], 
                      y=test_portfolios_df['predicted_vol'],
                      mode='markers', name='Validation'),
            row=2, col=1
        )
        
        # Parameter analysis - Lambda distribution
        ewma_data = exposure_summary[exposure_summary['method'] == 'ewma']
        if not ewma_data.empty:
            fig.add_trace(
                go.Histogram(x=ewma_data['lambda'], name='Lambda'),
                row=2, col=2
            )
        
        # Performance by portfolio type
        portfolio_stats = test_portfolios_df.groupby('portfolio_type')['abs_error'].mean()
        fig.add_trace(
            go.Bar(x=portfolio_stats.index, y=portfolio_stats.values,
                   name='Avg Abs Error'),
            row=3, col=1
        )
        
        # Exposure overview - validation scores
        fig.add_trace(
            go.Bar(x=exposure_summary['exposure'], y=exposure_summary['validation_score'],
                   name='Validation Score'),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, showlegend=True, 
                         title_text="Portfolio-Level Optimization Dashboard")
        
        return fig
        
    def save_all_plots(self, analyzer, output_dir: str = "output/portfolio_level_optimization/plots"):
        """
        Save all visualization plots to files
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            output_dir: Directory to save plots
        """
        
        from pathlib import Path
        output_path = Path(output_dir)
        
        # Handle relative paths from different working directories
        if not output_path.exists():
            # Try from parent directory (for notebooks)
            parent_output_path = Path("..") / output_dir
            if parent_output_path.parent.exists():
                output_path = parent_output_path
                
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get data
        horizon_df = analyzer.get_horizon_comparison()
        method_counts = analyzer.get_method_distribution()
        test_portfolios_df = analyzer.get_test_portfolios_performance()
        exposure_summary = analyzer.get_exposure_summary()
        
        # Save plots
        plots_to_save = [
            (self.plot_horizon_comparison, (horizon_df,), "horizon_comparison.png"),
            (self.plot_method_distribution, (method_counts,), "method_distribution.png"),
            (self.plot_validation_quality, (test_portfolios_df,), "validation_quality.png"),
            (self.plot_parameter_analysis, (exposure_summary,), "parameter_analysis.png"),
            (self.plot_exposure_heatmap, (exposure_summary,), "exposure_heatmap.png"),
            (self.plot_optimization_performance_curve, (analyzer,), "optimization_performance_curve.png"),
            (self.plot_parameter_effectiveness, (analyzer,), "parameter_effectiveness.png"),
            (self.plot_exposure_analysis, (analyzer,), "exposure_analysis.png"),
            (self.plot_risk_estimates_results, (analyzer,), "risk_estimates_results.png"),
            (self.plot_portfolio_composition_analysis, (analyzer,), "portfolio_composition_analysis.png")
        ]
        
        for plot_func, args, filename in plots_to_save:
            try:
                fig = plot_func(*args)
                if fig is not None:
                    fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
                
        print(f"All plots saved to: {output_path}")
    
    def plot_return_prediction_errors_by_exposure(self, analyzer, 
                                                 interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot return prediction errors by exposure
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        errors_df = analyzer.get_return_prediction_errors_by_exposure()
        
        if errors_df.empty:
            print("No return prediction data available")
            return None
            
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Directional Accuracy by Exposure', 'Error Rate by Method',
                               'Method Distribution', 'Parameter Analysis'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Directional accuracy by exposure
            fig.add_trace(
                go.Bar(x=errors_df['exposure'], y=errors_df['directional_accuracy'],
                       name='Directional Accuracy', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Error rate by method
            method_errors = errors_df.groupby('method')['error_rate'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=method_errors['method'], y=method_errors['error_rate'],
                       name='Average Error Rate', marker_color='lightcoral'),
                row=1, col=2
            )
            
            # Method distribution
            method_counts = errors_df['method'].value_counts()
            fig.add_trace(
                go.Pie(labels=method_counts.index, values=method_counts.values,
                       name='Method Distribution'),
                row=2, col=1
            )
            
            # Parameter analysis - show horizon distribution
            horizon_dist = errors_df['horizon'].value_counts()
            fig.add_trace(
                go.Bar(x=horizon_dist.index, y=horizon_dist.values,
                       name='Horizon Distribution', marker_color='lightgreen'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Return Prediction Error Analysis by Exposure",
                height=800,
                showlegend=False
            )
            
            return fig
            
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Directional accuracy by exposure
            axes[0, 0].bar(range(len(errors_df)), errors_df['directional_accuracy'], 
                          color=self.colors[0], alpha=0.7)
            axes[0, 0].set_xlabel('Exposure Index')
            axes[0, 0].set_ylabel('Directional Accuracy')
            axes[0, 0].set_title('Directional Accuracy by Exposure')
            axes[0, 0].set_xticks(range(len(errors_df)))
            axes[0, 0].set_xticklabels(errors_df['exposure'], rotation=45, ha='right')
            axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Error rate by method
            method_errors = errors_df.groupby('method')['error_rate'].mean()
            axes[0, 1].bar(method_errors.index, method_errors.values, 
                          color=self.colors[1], alpha=0.7)
            axes[0, 1].set_xlabel('Method')
            axes[0, 1].set_ylabel('Average Error Rate')
            axes[0, 1].set_title('Error Rate by Method')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Method distribution
            method_counts = errors_df['method'].value_counts()
            axes[1, 0].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%',
                          colors=self.colors[:len(method_counts)])
            axes[1, 0].set_title('Method Distribution')
            
            # Accuracy vs Error Rate scatter
            axes[1, 1].scatter(errors_df['directional_accuracy'], errors_df['error_rate'],
                              c=[self.colors[i % len(self.colors)] for i in range(len(errors_df))],
                              alpha=0.7, s=100)
            axes[1, 1].set_xlabel('Directional Accuracy')
            axes[1, 1].set_ylabel('Error Rate')
            axes[1, 1].set_title('Accuracy vs Error Rate')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
    
    def plot_return_prediction_errors_by_horizon(self, analyzer, 
                                               interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot return prediction errors by horizon
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        horizon_df = analyzer.get_return_prediction_errors_by_horizon()
        
        if horizon_df.empty:
            print("No horizon return prediction data available")
            return None
            
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Directional Accuracy by Horizon', 'Error Rate by Horizon',
                               'Method Diversity by Horizon', 'Accuracy Range by Horizon'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Directional accuracy by horizon
            fig.add_trace(
                go.Scatter(x=horizon_df['horizon'], y=horizon_df['avg_directional_accuracy'],
                          mode='lines+markers', name='Avg Directional Accuracy'),
                row=1, col=1
            )
            
            # Error rate by horizon
            fig.add_trace(
                go.Scatter(x=horizon_df['horizon'], y=horizon_df['avg_error_rate'],
                          mode='lines+markers', name='Avg Error Rate', line=dict(color='red')),
                row=1, col=2
            )
            
            # Method diversity by horizon
            fig.add_trace(
                go.Bar(x=horizon_df['horizon'], y=horizon_df['method_diversity'],
                       name='Method Diversity', marker_color='lightgreen'),
                row=2, col=1
            )
            
            # Accuracy range (min/max) by horizon
            fig.add_trace(
                go.Scatter(x=horizon_df['horizon'], y=horizon_df['max_directional_accuracy'],
                          mode='lines+markers', name='Max Accuracy', line=dict(color='green')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=horizon_df['horizon'], y=horizon_df['min_directional_accuracy'],
                          mode='lines+markers', name='Min Accuracy', line=dict(color='red')),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Return Prediction Error Analysis by Horizon",
                height=800,
                showlegend=True
            )
            
            return fig
            
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Directional accuracy by horizon
            axes[0, 0].plot(horizon_df['horizon'], horizon_df['avg_directional_accuracy'], 
                           'o-', color=self.colors[0], linewidth=2, markersize=8)
            axes[0, 0].fill_between(horizon_df['horizon'], 
                                   horizon_df['avg_directional_accuracy'] - horizon_df['std_directional_accuracy'],
                                   horizon_df['avg_directional_accuracy'] + horizon_df['std_directional_accuracy'],
                                   alpha=0.3, color=self.colors[0])
            axes[0, 0].set_xlabel('Horizon (days)')
            axes[0, 0].set_ylabel('Directional Accuracy')
            axes[0, 0].set_title('Directional Accuracy by Horizon')
            axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Error rate by horizon
            axes[0, 1].plot(horizon_df['horizon'], horizon_df['avg_error_rate'], 
                           'o-', color=self.colors[1], linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Horizon (days)')
            axes[0, 1].set_ylabel('Error Rate')
            axes[0, 1].set_title('Error Rate by Horizon')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Method diversity by horizon
            axes[1, 0].bar(horizon_df['horizon'], horizon_df['method_diversity'], 
                          color=self.colors[2], alpha=0.7)
            axes[1, 0].set_xlabel('Horizon (days)')
            axes[1, 0].set_ylabel('Number of Methods Used')
            axes[1, 0].set_title('Method Diversity by Horizon')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Accuracy range by horizon
            axes[1, 1].fill_between(horizon_df['horizon'], 
                                   horizon_df['min_directional_accuracy'],
                                   horizon_df['max_directional_accuracy'],
                                   alpha=0.4, color=self.colors[3], label='Accuracy Range')
            axes[1, 1].plot(horizon_df['horizon'], horizon_df['avg_directional_accuracy'], 
                           'o-', color=self.colors[0], linewidth=2, markersize=6, label='Average')
            axes[1, 1].set_xlabel('Horizon (days)')
            axes[1, 1].set_ylabel('Directional Accuracy')
            axes[1, 1].set_title('Accuracy Range by Horizon')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
    
    def plot_return_prediction_method_analysis(self, analyzer, 
                                             interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot return prediction method analysis
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        method_df = analyzer.get_return_prediction_errors_by_method()
        
        if method_df.empty:
            print("No method return prediction data available")
            return None
            
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Average Directional Accuracy', 'Error Rate Distribution',
                               'Method Usage Count', 'Accuracy Consistency'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Average directional accuracy
            fig.add_trace(
                go.Bar(x=method_df['method'], y=method_df['directional_accuracy_mean'],
                       name='Avg Directional Accuracy', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Error rate distribution
            fig.add_trace(
                go.Bar(x=method_df['method'], y=method_df['error_rate_mean'],
                       name='Avg Error Rate', marker_color='lightcoral'),
                row=1, col=2
            )
            
            # Method usage count
            fig.add_trace(
                go.Bar(x=method_df['method'], y=method_df['directional_accuracy_count'],
                       name='Usage Count', marker_color='lightgreen'),
                row=2, col=1
            )
            
            # Accuracy consistency (standard deviation)
            fig.add_trace(
                go.Bar(x=method_df['method'], y=method_df['directional_accuracy_std'],
                       name='Accuracy Std Dev', marker_color='orange'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Return Prediction Method Analysis",
                height=800,
                showlegend=False
            )
            
            return fig
            
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Average directional accuracy
            axes[0, 0].bar(method_df['method'], method_df['directional_accuracy_mean'], 
                          color=self.colors[0], alpha=0.7)
            axes[0, 0].set_xlabel('Method')
            axes[0, 0].set_ylabel('Average Directional Accuracy')
            axes[0, 0].set_title('Average Directional Accuracy by Method')
            axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Error rate distribution
            axes[0, 1].bar(method_df['method'], method_df['error_rate_mean'], 
                          color=self.colors[1], alpha=0.7)
            axes[0, 1].set_xlabel('Method')
            axes[0, 1].set_ylabel('Average Error Rate')
            axes[0, 1].set_title('Average Error Rate by Method')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Method usage count
            axes[1, 0].bar(method_df['method'], method_df['directional_accuracy_count'], 
                          color=self.colors[2], alpha=0.7)
            axes[1, 0].set_xlabel('Method')
            axes[1, 0].set_ylabel('Usage Count')
            axes[1, 0].set_title('Method Usage Count')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Accuracy consistency (with error bars)
            axes[1, 1].bar(method_df['method'], method_df['directional_accuracy_mean'], 
                          yerr=method_df['directional_accuracy_std'],
                          color=self.colors[3], alpha=0.7, capsize=5)
            axes[1, 1].set_xlabel('Method')
            axes[1, 1].set_ylabel('Directional Accuracy')
            axes[1, 1].set_title('Accuracy Consistency (Mean ± Std)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
    
    def plot_return_prediction_parameter_analysis(self, analyzer, 
                                                interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot return prediction parameter analysis
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            interactive: Whether to create interactive plotly chart
        """
        
        param_analysis = analyzer.get_return_prediction_parameter_analysis()
        
        if 'note' in param_analysis:
            print(param_analysis['note'])
            return None
            
        if interactive:
            # Create subplots for parameter analysis
            methods = [m for m in param_analysis.keys() if param_analysis[m]['count'] > 0]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Method Performance', 'Parameter Distribution', 
                               'Lookback Analysis', 'Score Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Method performance
            method_scores = [param_analysis[m]['avg_score'] for m in methods]
            fig.add_trace(
                go.Bar(x=methods, y=method_scores, name='Avg Score', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Parameter distribution (focus on methods with specific parameters)
            if 'ewma' in param_analysis and param_analysis['ewma']['count'] > 0:
                decay_factors = [p['decay_factor'] for p in param_analysis['ewma']['parameters']]
                fig.add_trace(
                    go.Histogram(x=decay_factors, name='EWMA Decay Factors', 
                               marker_color='lightgreen'),
                    row=1, col=2
                )
            
            # Lookback analysis
            if 'historical' in param_analysis and param_analysis['historical']['count'] > 0:
                lookbacks = [p['lookback_days'] for p in param_analysis['historical']['parameters']]
                fig.add_trace(
                    go.Histogram(x=lookbacks, name='Historical Lookbacks', 
                               marker_color='lightcoral'),
                    row=2, col=1
                )
            
            # Score distribution
            all_scores = []
            for method in methods:
                all_scores.extend(param_analysis[method]['scores'])
            fig.add_trace(
                go.Histogram(x=all_scores, name='All Scores', marker_color='orange'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Return Prediction Parameter Analysis",
                height=800,
                showlegend=False
            )
            
            return fig
            
        else:
            methods = [m for m in param_analysis.keys() if param_analysis[m]['count'] > 0]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Method performance
            method_scores = [param_analysis[m]['avg_score'] for m in methods]
            axes[0, 0].bar(methods, method_scores, color=self.colors[0], alpha=0.7)
            axes[0, 0].set_xlabel('Method')
            axes[0, 0].set_ylabel('Average Score')
            axes[0, 0].set_title('Method Performance')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Parameter distribution for EWMA
            if 'ewma' in param_analysis and param_analysis['ewma']['count'] > 0:
                decay_factors = [p['decay_factor'] for p in param_analysis['ewma']['parameters']]
                axes[0, 1].hist(decay_factors, bins=10, color=self.colors[1], alpha=0.7)
                axes[0, 1].set_xlabel('Decay Factor')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('EWMA Decay Factor Distribution')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No EWMA data', ha='center', va='center')
                axes[0, 1].set_title('EWMA Parameters')
            
            # Lookback analysis for Historical
            if 'historical' in param_analysis and param_analysis['historical']['count'] > 0:
                lookbacks = [p['lookback_days'] for p in param_analysis['historical']['parameters']]
                axes[1, 0].hist(lookbacks, bins=10, color=self.colors[2], alpha=0.7)
                axes[1, 0].set_xlabel('Lookback Days')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Historical Lookback Distribution')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Historical data', ha='center', va='center')
                axes[1, 0].set_title('Historical Parameters')
            
            # Score distribution
            all_scores = []
            for method in methods:
                all_scores.extend(param_analysis[method]['scores'])
            axes[1, 1].hist(all_scores, bins=15, color=self.colors[3], alpha=0.7)
            axes[1, 1].set_xlabel('Directional Accuracy Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Score Distribution')
            axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
    
    def save_all_plots_with_returns(self, analyzer, output_path: str = None):
        """
        Save all plots including return prediction visualizations
        
        Args:
            analyzer: PortfolioLevelAnalyzer instance
            output_path: Directory to save plots (default: results_dir/plots)
        """
        
        if output_path is None:
            output_path = analyzer.results_dir / "plots"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get data for standard plots
        horizon_df = analyzer.get_horizon_comparison()
        exposure_summary = analyzer.get_exposure_summary()
        method_counts = analyzer.get_method_distribution()
        test_portfolios_df = analyzer.get_test_portfolios_summary()
        
        # Save all plots including return prediction plots
        plots_to_save = [
            # Standard plots
            (self.plot_horizon_comparison, (horizon_df,), "horizon_comparison.png"),
            (self.plot_method_distribution, (method_counts,), "method_distribution.png"),
            (self.plot_validation_quality, (test_portfolios_df,), "validation_quality.png"),
            (self.plot_parameter_analysis, (exposure_summary,), "parameter_analysis.png"),
            (self.plot_exposure_heatmap, (exposure_summary,), "exposure_heatmap.png"),
            (self.plot_optimization_performance_curve, (analyzer,), "optimization_performance_curve.png"),
            (self.plot_parameter_effectiveness, (analyzer,), "parameter_effectiveness.png"),
            (self.plot_exposure_analysis, (analyzer,), "exposure_analysis.png"),
            (self.plot_risk_estimates_results, (analyzer,), "risk_estimates_results.png"),
            (self.plot_portfolio_composition_analysis, (analyzer,), "portfolio_composition_analysis.png"),
            
            # Return prediction plots
            (self.plot_return_prediction_errors_by_exposure, (analyzer,), "return_prediction_errors_by_exposure.png"),
            (self.plot_return_prediction_errors_by_horizon, (analyzer,), "return_prediction_errors_by_horizon.png"),
            (self.plot_return_prediction_method_analysis, (analyzer,), "return_prediction_method_analysis.png"),
            (self.plot_return_prediction_parameter_analysis, (analyzer,), "return_prediction_parameter_analysis.png"),
        ]
        
        for plot_func, args, filename in plots_to_save:
            try:
                fig = plot_func(*args)
                if fig is not None:
                    fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
                
        print(f"All plots (including return prediction) saved to: {output_path}")