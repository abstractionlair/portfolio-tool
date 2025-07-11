"""
Visualization utilities for optimization analysis.

This module provides professional visualization tools for analyzing
component optimization results, parameter distributions, and performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from ..analysis.parameter_analysis import ParameterAnalyzer, ParameterComparator

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class OptimizationVisualizer:
    """Create professional visualizations for optimization analysis."""
    
    def __init__(self, analyzer: ParameterAnalyzer):
        """Initialize with parameter analyzer.
        
        Args:
            analyzer: ParameterAnalyzer instance
        """
        self.analyzer = analyzer
        self.summary_df = analyzer.create_parameter_summary()
    
    def plot_method_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Create method distribution plot across components.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        method_dist = self.analyzer.get_method_distribution()
        
        fig, axes = plt.subplots(1, len(method_dist), figsize=figsize)
        if len(method_dist) == 1:
            axes = [axes]
        
        for i, (component, methods) in enumerate(method_dist.items()):
            if not methods:
                continue
                
            ax = axes[i]
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                methods.values(),
                labels=methods.keys(),
                autopct='%1.1f%%',
                startangle=90
            )
            
            ax.set_title(f'{component.replace("_", " ").title()} Methods')
            
            # Style the text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        return fig
    
    def plot_lookback_distribution(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create lookback period distribution plots.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        components = self.summary_df['component'].unique()
        
        for i, component in enumerate(components):
            if i >= len(axes):
                break
                
            ax = axes[i]
            component_data = self.summary_df[self.summary_df['component'] == component]
            lookback_days = component_data['lookback_days'].dropna()
            
            if len(lookback_days) > 0:
                # Histogram
                ax.hist(lookback_days, bins=min(10, len(lookback_days)), 
                       alpha=0.7, edgecolor='black')
                ax.set_title(f'{component.replace("_", " ").title()} Lookback Days')
                ax.set_xlabel('Lookback Days')
                ax.set_ylabel('Frequency')
                
                # Add mean line
                mean_lookback = lookback_days.mean()
                ax.axvline(mean_lookback, color='red', linestyle='--', 
                          label=f'Mean: {mean_lookback:.0f}')
                ax.legend()
        
        # Hide unused subplots
        for i in range(len(components), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_score_analysis(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create score analysis plots.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Box plot of scores by component
        ax1 = axes[0, 0]
        score_data = []
        component_labels = []
        
        for component in self.summary_df['component'].unique():
            component_scores = self.summary_df[
                self.summary_df['component'] == component
            ]['score'].dropna()
            
            if len(component_scores) > 0:
                score_data.append(component_scores)
                component_labels.append(component.replace('_', ' ').title())
        
        if score_data:
            ax1.boxplot(score_data, labels=component_labels)
            ax1.set_title('Score Distribution by Component')
            ax1.set_ylabel('Score')
            ax1.tick_params(axis='x', rotation=45)
        
        # Score vs lookback scatter plot
        ax2 = axes[0, 1]
        for component in self.summary_df['component'].unique():
            component_data = self.summary_df[self.summary_df['component'] == component]
            if len(component_data) > 0:
                ax2.scatter(component_data['lookback_days'], component_data['score'],
                           label=component.replace('_', ' ').title(), alpha=0.7)
        
        ax2.set_xlabel('Lookback Days')
        ax2.set_ylabel('Score')
        ax2.set_title('Score vs Lookback Days')
        ax2.legend()
        
        # Score histogram
        ax3 = axes[1, 0]
        all_scores = self.summary_df['score'].dropna()
        if len(all_scores) > 0:
            ax3.hist(all_scores, bins=min(15, len(all_scores)), alpha=0.7, edgecolor='black')
            ax3.set_title('Overall Score Distribution')
            ax3.set_xlabel('Score')
            ax3.set_ylabel('Frequency')
        
        # Component comparison
        ax4 = axes[1, 1]
        component_means = self.summary_df.groupby('component')['score'].mean().sort_values()
        if len(component_means) > 0:
            bars = ax4.bar(range(len(component_means)), component_means.values)
            ax4.set_xticks(range(len(component_means)))
            ax4.set_xticklabels([c.replace('_', ' ').title() for c in component_means.index], 
                               rotation=45)
            ax4.set_title('Mean Score by Component')
            ax4.set_ylabel('Mean Score')
            
            # Add value labels on bars
            for bar, value in zip(bars, component_means.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_heatmap(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """Create parameter heatmap for exposure comparison.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        # Prepare data for heatmap
        numeric_cols = ['lookback_days', 'score']
        heatmap_data = []
        
        for component in self.summary_df['component'].unique():
            component_data = self.summary_df[self.summary_df['component'] == component]
            
            for col in numeric_cols:
                if col in component_data.columns:
                    for _, row in component_data.iterrows():
                        heatmap_data.append({
                            'exposure_id': row['exposure_id'],
                            'parameter': f"{component}_{col}",
                            'value': row[col]
                        })
        
        if not heatmap_data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No numeric parameters to display', 
                   ha='center', va='center', fontsize=16)
            return fig
        
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_df = heatmap_df.pivot(index='exposure_id', columns='parameter', values='value')
        
        # Normalize each column for better visualization
        pivot_df_norm = pivot_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(pivot_df_norm.values, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(pivot_df_norm.columns)))
        ax.set_yticks(range(len(pivot_df_norm.index)))
        ax.set_xticklabels(pivot_df_norm.columns, rotation=45, ha='right')
        ax.set_yticklabels(pivot_df_norm.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value', rotation=270, labelpad=20)
        
        # Add text annotations with actual values
        for i in range(len(pivot_df_norm.index)):
            for j in range(len(pivot_df_norm.columns)):
                value = pivot_df.iloc[i, j]
                if pd.notna(value):
                    text = ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                                  color='white' if pivot_df_norm.iloc[i, j] > 0.5 else 'black',
                                  fontweight='bold')
        
        ax.set_title('Parameter Heatmap Across Exposures', fontsize=16, pad=20)
        plt.tight_layout()
        
        return fig
    
    def create_interactive_dashboard(self) -> go.Figure:
        """Create interactive Plotly dashboard.
        
        Returns:
            Plotly figure with interactive dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Method Distribution', 'Lookback Distribution', 
                           'Score vs Lookback', 'Parameter Consistency'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Method distribution (pie chart)
        method_dist = self.analyzer.get_method_distribution()
        if method_dist:
            component = list(method_dist.keys())[0]  # Take first component
            methods = method_dist[component]
            
            fig.add_trace(
                go.Pie(
                    labels=list(methods.keys()),
                    values=list(methods.values()),
                    name=component
                ),
                row=1, col=1
            )
        
        # Lookback distribution
        lookback_data = self.summary_df['lookback_days'].dropna()
        if len(lookback_data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=lookback_data,
                    name="Lookback Days",
                    nbinsx=min(15, len(lookback_data))
                ),
                row=1, col=2
            )
        
        # Score vs Lookback scatter
        for component in self.summary_df['component'].unique():
            component_data = self.summary_df[self.summary_df['component'] == component]
            if len(component_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=component_data['lookback_days'],
                        y=component_data['score'],
                        mode='markers',
                        name=component.replace('_', ' ').title(),
                        text=component_data['exposure_id'],
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Lookback: %{x}<br>' +
                                    'Score: %{y:.4f}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Parameter consistency
        consistency = self.analyzer.get_parameter_consistency()
        if consistency:
            components = list(consistency.keys())
            consistency_scores = []
            
            for component in components:
                # Simple consistency metric
                method_consistent = consistency[component]['method_consistency']
                freq_consistent = consistency[component]['frequency_consistency']
                score = (method_consistent + freq_consistent) / 2
                consistency_scores.append(score)
            
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=consistency_scores,
                    name="Consistency Score",
                    text=[f'{score:.1%}' for score in consistency_scores],
                    textposition='auto'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Optimization Analysis Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_summary_report(self, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Create comprehensive summary report with all visualizations.
        
        Args:
            save_path: Optional path to save figures
            
        Returns:
            Dictionary mapping figure names to matplotlib figures
        """
        figures = {}
        
        # Method distribution
        figures['method_distribution'] = self.plot_method_distribution()
        
        # Lookback distribution
        figures['lookback_distribution'] = self.plot_lookback_distribution()
        
        # Score analysis
        figures['score_analysis'] = self.plot_score_analysis()
        
        # Parameter heatmap
        figures['parameter_heatmap'] = self.plot_parameter_heatmap()
        
        # Save figures if path provided
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/{name}.png", dpi=300, bbox_inches='tight')
                print(f"Saved {name}.png")
        
        return figures


class ComparisonVisualizer:
    """Create visualizations for comparing optimization results."""
    
    def __init__(self, comparator: ParameterComparator):
        """Initialize with parameter comparator.
        
        Args:
            comparator: ParameterComparator instance
        """
        self.comparator = comparator
    
    def plot_score_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create score comparison plot.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        score_comparison = self.comparator.compare_scores()
        
        if not score_comparison:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No score data to compare', 
                   ha='center', va='center', fontsize=16)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract data for plotting
        components = list(score_comparison.keys())
        baseline_scores = [score_comparison[comp]['baseline_mean'] for comp in components]
        comparison_scores = [score_comparison[comp]['comparison_mean'] for comp in components]
        improvements = [score_comparison[comp]['improvement'] for comp in components]
        improvement_pcts = [score_comparison[comp]['improvement_pct'] for comp in components]
        
        # Score comparison bar chart
        ax1 = axes[0, 0]
        x = np.arange(len(components))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline', alpha=0.7)
        bars2 = ax1.bar(x + width/2, comparison_scores, width, label='Comparison', alpha=0.7)
        
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Score')
        ax1.set_title('Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace('_', ' ').title() for c in components])
        ax1.legend()
        
        # Improvement bar chart
        ax2 = axes[0, 1]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(components, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Score Improvement')
        ax2.set_title('Absolute Improvement')
        ax2.set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, improvements):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Percentage improvement bar chart
        ax3 = axes[1, 0]
        colors = ['green' if imp > 0 else 'red' for imp in improvement_pcts]
        bars = ax3.bar(components, improvement_pcts, color=colors, alpha=0.7)
        ax3.set_xlabel('Component')
        ax3.set_ylabel('Improvement %')
        ax3.set_title('Percentage Improvement')
        ax3.set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, improvement_pcts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Scatter plot of baseline vs comparison
        ax4 = axes[1, 1]
        ax4.scatter(baseline_scores, comparison_scores, alpha=0.7, s=100)
        
        # Add diagonal line
        min_score = min(min(baseline_scores), min(comparison_scores))
        max_score = max(max(baseline_scores), max(comparison_scores))
        ax4.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5)
        
        ax4.set_xlabel('Baseline Score')
        ax4.set_ylabel('Comparison Score')
        ax4.set_title('Baseline vs Comparison Scores')
        
        # Add component labels
        for i, component in enumerate(components):
            ax4.annotate(component.replace('_', ' ').title(), 
                        (baseline_scores[i], comparison_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_method_changes(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Create method changes visualization.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        method_comparison = self.comparator.compare_methods()
        
        if not method_comparison:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No method data to compare', 
                   ha='center', va='center', fontsize=16)
            return fig
        
        fig, axes = plt.subplots(1, len(method_comparison), figsize=figsize)
        if len(method_comparison) == 1:
            axes = [axes]
        
        for i, (component, data) in enumerate(method_comparison.items()):
            ax = axes[i]
            
            method_changes = data['method_changes']
            methods = list(method_changes.keys())
            changes = list(method_changes.values())
            
            # Create bar chart
            colors = ['green' if change > 0 else 'red' if change < 0 else 'gray' 
                     for change in changes]
            bars = ax.bar(methods, changes, color=colors, alpha=0.7)
            
            ax.set_title(f'{component.replace("_", " ").title()} Method Changes')
            ax.set_xlabel('Method')
            ax.set_ylabel('Change in Count')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, changes):
                if value != 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:+d}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_dashboard(self) -> go.Figure:
        """Create interactive comparison dashboard.
        
        Returns:
            Plotly figure with interactive comparison dashboard
        """
        score_comparison = self.comparator.compare_scores()
        
        if not score_comparison:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for comparison",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Score Comparison', 'Improvement Distribution', 
                           'Baseline vs Comparison', 'Method Changes'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Extract data
        components = list(score_comparison.keys())
        baseline_scores = [score_comparison[comp]['baseline_mean'] for comp in components]
        comparison_scores = [score_comparison[comp]['comparison_mean'] for comp in components]
        improvements = [score_comparison[comp]['improvement'] for comp in components]
        improvement_pcts = [score_comparison[comp]['improvement_pct'] for comp in components]
        
        # Score comparison
        fig.add_trace(
            go.Bar(
                x=components,
                y=baseline_scores,
                name='Baseline',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=components,
                y=comparison_scores,
                name='Comparison',
                marker_color='lightgreen'
            ),
            row=1, col=1
        )
        
        # Improvement distribution
        fig.add_trace(
            go.Histogram(
                x=improvement_pcts,
                name="Improvement %",
                nbinsx=10
            ),
            row=1, col=2
        )
        
        # Baseline vs Comparison scatter
        fig.add_trace(
            go.Scatter(
                x=baseline_scores,
                y=comparison_scores,
                mode='markers+text',
                text=components,
                textposition='top center',
                name='Components',
                marker=dict(size=10)
            ),
            row=2, col=1
        )
        
        # Add diagonal line
        min_score = min(min(baseline_scores), min(comparison_scores))
        max_score = max(max(baseline_scores), max(comparison_scores))
        fig.add_trace(
            go.Scatter(
                x=[min_score, max_score],
                y=[min_score, max_score],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Perfect Match',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Improvement bars
        colors = ['green' if imp > 0 else 'red' for imp in improvement_pcts]
        fig.add_trace(
            go.Bar(
                x=components,
                y=improvement_pcts,
                name='Improvement %',
                marker_color=colors,
                text=[f'{imp:.1f}%' for imp in improvement_pcts],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Optimization Comparison Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig


def create_optimization_summary(analyzer: ParameterAnalyzer, 
                              save_path: Optional[str] = None) -> Dict[str, Any]:
    """Create comprehensive optimization analysis summary.
    
    Args:
        analyzer: ParameterAnalyzer instance
        save_path: Optional path to save visualizations
        
    Returns:
        Dictionary with analysis results and figures
    """
    visualizer = OptimizationVisualizer(analyzer)
    
    # Create all visualizations
    figures = visualizer.create_summary_report(save_path)
    
    # Generate insights
    insights = analyzer.generate_optimization_insights()
    
    return {
        'insights': insights,
        'figures': figures,
        'interactive_dashboard': visualizer.create_interactive_dashboard()
    }


def compare_optimizations(baseline_analyzer: ParameterAnalyzer,
                         comparison_analyzer: ParameterAnalyzer,
                         save_path: Optional[str] = None) -> Dict[str, Any]:
    """Compare two optimization results with full analysis.
    
    Args:
        baseline_analyzer: ParameterAnalyzer for baseline
        comparison_analyzer: ParameterAnalyzer for comparison
        save_path: Optional path to save visualizations
        
    Returns:
        Dictionary with comparison results and figures
    """
    # Create comparator
    comparator = ParameterComparator(baseline_analyzer.params, comparison_analyzer.params)
    
    # Create visualizer
    comp_visualizer = ComparisonVisualizer(comparator)
    
    # Generate comparison report
    comparison_report = comparator.generate_comparison_report()
    
    # Create figures
    figures = {
        'score_comparison': comp_visualizer.plot_score_comparison(),
        'method_changes': comp_visualizer.plot_method_changes()
    }
    
    # Save figures if path provided
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for name, fig in figures.items():
            fig.savefig(f"{save_path}/{name}.png", dpi=300, bbox_inches='tight')
            print(f"Saved {name}.png")
    
    return {
        'comparison_report': comparison_report,
        'figures': figures,
        'interactive_dashboard': comp_visualizer.create_comparison_dashboard()
    }