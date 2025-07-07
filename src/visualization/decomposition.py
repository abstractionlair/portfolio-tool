"""
Return decomposition visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class DecompositionVisualizer:
    """Visualize return decomposition results."""
    
    def __init__(self, color_palette: Optional[List[str]] = None):
        """
        Initialize decomposition visualizer.
        
        Args:
            color_palette: Custom color palette
        """
        self.color_palette = color_palette or [
            '#1f77b4',  # Blue - Total return
            '#ff7f0e',  # Orange - Inflation
            '#2ca02c',  # Green - Real risk-free rate
            '#d62728',  # Red - Risk premium/spread
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ]
    
    def plot_return_components(
        self,
        decomposition: pd.DataFrame,
        title: str = "Return Decomposition",
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Create stacked area chart of return components over time.
        
        Args:
            decomposition: DataFrame from ReturnDecomposer with components
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Prepare components for stacking
        components = ['inflation', 'real_rf_rate', 'spread']
        component_labels = ['Inflation', 'Real Risk-Free Rate', 'Risk Premium']
        
        # Calculate cumulative components for stacking
        cum_inflation = (1 + decomposition['inflation']).cumprod()
        cum_real_rf = (1 + decomposition['real_rf_rate']).cumprod()
        cum_spread = (1 + decomposition['spread']).cumprod()
        cum_total = (1 + decomposition['total_return']).cumprod()
        
        # Plot 1: Stacked area chart of components
        ax1.fill_between(decomposition.index, 0, cum_inflation - 1,
                        color=self.color_palette[1], alpha=0.7, label='Inflation')
        ax1.fill_between(decomposition.index, cum_inflation - 1, 
                        cum_inflation * cum_real_rf - 1,
                        color=self.color_palette[2], alpha=0.7, label='Real Risk-Free Rate')
        ax1.fill_between(decomposition.index, cum_inflation * cum_real_rf - 1,
                        cum_total - 1,
                        color=self.color_palette[3], alpha=0.7, label='Risk Premium')
        
        # Overlay total return line
        ax1.plot(decomposition.index, cum_total - 1, 
                color='black', linewidth=2, label='Total Return')
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Component returns over time
        ax2.plot(decomposition.index, decomposition['inflation'] * 252,
                color=self.color_palette[1], label='Inflation', alpha=0.8)
        ax2.plot(decomposition.index, decomposition['real_rf_rate'] * 252,
                color=self.color_palette[2], label='Real Risk-Free', alpha=0.8)
        ax2.plot(decomposition.index, decomposition['spread'] * 252,
                color=self.color_palette[3], label='Risk Premium', alpha=0.8)
        
        ax2.set_ylabel('Annualized Return')
        ax2.set_xlabel('Date')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_component_comparison(
        self,
        decompositions: Dict[str, pd.DataFrame],
        title: str = "Return Component Comparison",
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """
        Compare decomposition results across multiple assets.
        
        Args:
            decompositions: Dict mapping asset names to decomposition DataFrames
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        n_assets = len(decompositions)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Calculate annualized average returns for each component
        summary_data = []
        for asset, decomp in decompositions.items():
            summary_data.append({
                'Asset': asset,
                'Total Return': decomp['total_return'].mean() * 252,
                'Inflation': decomp['inflation'].mean() * 252,
                'Real Risk-Free': decomp['real_rf_rate'].mean() * 252,
                'Risk Premium': decomp['spread'].mean() * 252
            })
        
        summary_df = pd.DataFrame(summary_data).set_index('Asset')
        
        # 1. Stacked bar chart of components
        ax = axes[0, 0]
        components = ['Inflation', 'Real Risk-Free', 'Risk Premium']
        bottom = np.zeros(len(summary_df))
        
        for i, component in enumerate(components):
            values = summary_df[component].values
            ax.bar(summary_df.index, values, bottom=bottom,
                  label=component, color=self.color_palette[i+1], alpha=0.8)
            bottom += values
        
        # Overlay total return markers
        ax.scatter(summary_df.index, summary_df['Total Return'].values,
                  color='red', s=50, marker='D', label='Actual Total', zorder=5)
        
        ax.set_title('Average Return Components', fontweight='bold')
        ax.set_ylabel('Annualized Return')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Risk premium vs total return scatter
        ax = axes[0, 1]
        x_vals = summary_df['Total Return'].values
        y_vals = summary_df['Risk Premium'].values
        
        ax.scatter(x_vals, y_vals, s=80, alpha=0.7, 
                  c=range(len(summary_df)), cmap='tab10')
        
        for i, asset in enumerate(summary_df.index):
            ax.annotate(asset, (x_vals[i], y_vals[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add diagonal line for reference
        max_val = max(x_vals.max(), y_vals.max())
        min_val = min(x_vals.min(), y_vals.min())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', alpha=0.5, label='Risk Premium = Total Return')
        
        ax.set_xlabel('Total Return')
        ax.set_ylabel('Risk Premium')
        ax.set_title('Risk Premium vs Total Return', fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Volatility comparison
        ax = axes[1, 0]
        vol_data = []
        for asset, decomp in decompositions.items():
            vol_data.append({
                'Asset': asset,
                'Total Return Vol': decomp['total_return'].std() * np.sqrt(252),
                'Inflation Vol': decomp['inflation'].std() * np.sqrt(252),
                'Real Risk-Free Vol': decomp['real_rf_rate'].std() * np.sqrt(252),
                'Risk Premium Vol': decomp['spread'].std() * np.sqrt(252)
            })
        
        vol_df = pd.DataFrame(vol_data).set_index('Asset')
        
        x_pos = np.arange(len(vol_df))
        width = 0.2
        
        for i, component in enumerate(['Total Return Vol', 'Risk Premium Vol']):
            offset = (i - 0.5) * width
            ax.bar(x_pos + offset, vol_df[component].values, width,
                  label=component, color=self.color_palette[i], alpha=0.8)
        
        ax.set_title('Volatility Comparison', fontweight='bold')
        ax.set_ylabel('Annualized Volatility')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(vol_df.index, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Component correlation heatmap
        ax = axes[1, 1]
        
        # Calculate correlation between risk premiums across assets
        risk_premiums = pd.DataFrame({
            asset: decomp['spread'] for asset, decomp in decompositions.items()
        })
        corr_matrix = risk_premiums.corr()
        
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.index)
        ax.set_title('Risk Premium Correlations', fontweight='bold')
        
        # Add correlation values to heatmap
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_inflation_impact(
        self,
        nominal_returns: pd.Series,
        real_returns: pd.Series,
        inflation: pd.Series,
        title: str = "Impact of Inflation on Returns",
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Show the impact of inflation on asset returns.
        
        Args:
            nominal_returns: Nominal return series
            real_returns: Real return series
            inflation: Inflation series
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Calculate cumulative returns
        cum_nominal = (1 + nominal_returns).cumprod()
        cum_real = (1 + real_returns).cumprod()
        cum_inflation = (1 + inflation).cumprod()
        
        # 1. Cumulative returns comparison
        ax = axes[0, 0]
        ax.plot(cum_nominal.index, cum_nominal.values, 
               color=self.color_palette[0], linewidth=2, label='Nominal Return')
        ax.plot(cum_real.index, cum_real.values,
               color=self.color_palette[2], linewidth=2, label='Real Return')
        ax.plot(cum_inflation.index, cum_inflation.values,
               color=self.color_palette[1], linewidth=2, label='Inflation', linestyle='--')
        
        ax.set_title('Cumulative Returns: Nominal vs Real', fontweight='bold')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Return difference over time
        ax = axes[0, 1]
        return_diff = nominal_returns - real_returns
        rolling_diff = return_diff.rolling(252).mean() * 252  # Annual rolling average
        
        ax.plot(rolling_diff.index, rolling_diff.values,
               color=self.color_palette[1], linewidth=2)
        ax.fill_between(rolling_diff.index, 0, rolling_diff.values,
                       color=self.color_palette[1], alpha=0.3)
        
        ax.set_title('Inflation Impact (Rolling 1Y)', fontweight='bold')
        ax.set_ylabel('Nominal - Real Return')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        # 3. Scatter plot: nominal vs real returns
        ax = axes[1, 0]
        
        # Sample data for readability (every 20th point)
        sample_indices = range(0, len(nominal_returns), max(1, len(nominal_returns) // 200))
        nom_sample = nominal_returns.iloc[sample_indices]
        real_sample = real_returns.iloc[sample_indices]
        
        ax.scatter(nom_sample, real_sample, alpha=0.6, s=20, color=self.color_palette[0])
        
        # Add diagonal line
        min_val = min(nom_sample.min(), real_sample.min())
        max_val = max(nom_sample.max(), real_sample.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='No Inflation')
        
        ax.set_xlabel('Nominal Return')
        ax.set_ylabel('Real Return')
        ax.set_title('Nominal vs Real Returns', fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Distribution comparison
        ax = axes[1, 1]
        
        # Create histograms
        ax.hist(nominal_returns.dropna(), bins=50, alpha=0.7, density=True,
               color=self.color_palette[0], label='Nominal', edgecolor='black', linewidth=0.5)
        ax.hist(real_returns.dropna(), bins=50, alpha=0.7, density=True,
               color=self.color_palette[2], label='Real', edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for means
        ax.axvline(nominal_returns.mean(), color=self.color_palette[0], 
                  linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(real_returns.mean(), color=self.color_palette[2],
                  linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Density')
        ax.set_title('Return Distributions', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_waterfall_chart(
        self,
        decomposition_summary: pd.Series,
        title: str = "Return Decomposition Waterfall",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Create waterfall chart showing how components build up to total return.
        
        Args:
            decomposition_summary: Series with component values
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define components and their order
        components = ['inflation', 'real_rf_rate', 'spread']
        component_labels = ['Inflation', 'Real Risk-Free Rate', 'Risk Premium']
        
        # Prepare data for waterfall
        values = [decomposition_summary.get(comp, 0) for comp in components]
        total_value = decomposition_summary.get('total_return', sum(values))
        
        # Calculate cumulative positions
        cumulative = np.cumsum([0] + values)
        
        # Create bars
        colors = [self.color_palette[1], self.color_palette[2], self.color_palette[3]]
        
        for i, (label, value, color) in enumerate(zip(component_labels, values, colors)):
            # Draw the bar
            bar = ax.bar(i, value, bottom=cumulative[i], color=color, alpha=0.8,
                        edgecolor='black', linewidth=1)
            
            # Add value label on bar
            ax.text(i, cumulative[i] + value/2, f'{value:.2%}',
                   ha='center', va='center', fontweight='bold', color='white', fontsize=11)
            
            # Add connecting line to next bar (if not last)
            if i < len(values) - 1:
                ax.plot([i + 0.4, i + 0.6], [cumulative[i+1], cumulative[i+1]],
                       'k--', alpha=0.5, linewidth=1)
        
        # Add total return bar
        total_bar = ax.bar(len(values), total_value, color='darkblue', alpha=0.8,
                          edgecolor='black', linewidth=2)
        ax.text(len(values), total_value/2, f'{total_value:.2%}',
               ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        
        # Formatting
        labels = component_labels + ['Total Return']
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Return Contribution')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        # Add total calculation text
        calc_text = f"Total: {values[0]:.2%} + {values[1]:.2%} + {values[2]:.2%} = {total_value:.2%}"
        ax.text(0.02, 0.98, calc_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        return fig
    
    def create_interactive_decomposition_dashboard(
        self,
        decompositions: Dict[str, pd.DataFrame],
        title: str = "Interactive Return Decomposition Dashboard"
    ) -> go.Figure:
        """
        Create interactive dashboard for return decomposition analysis.
        
        Args:
            decompositions: Dict mapping asset names to decomposition DataFrames
            title: Dashboard title
            
        Returns:
            plotly Figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive charts")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Component Returns Over Time', 'Average Component Breakdown',
                          'Risk Premium vs Total Return', 'Cumulative Performance'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        # 1. Component returns over time (first asset only for clarity)
        if decompositions:
            first_asset = list(decompositions.keys())[0]
            decomp = decompositions[first_asset]
            
            # Annualize for display
            fig.add_trace(
                go.Scatter(
                    x=decomp.index,
                    y=decomp['inflation'] * 252,
                    name='Inflation',
                    line=dict(color=colors[1]),
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=decomp.index,
                    y=decomp['real_rf_rate'] * 252,
                    name='Real Risk-Free',
                    line=dict(color=colors[2]),
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=decomp.index,
                    y=decomp['spread'] * 252,
                    name='Risk Premium',
                    line=dict(color=colors[3]),
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # 2. Average component breakdown
        assets = list(decompositions.keys())
        components = ['inflation', 'real_rf_rate', 'spread']
        component_labels = ['Inflation', 'Real Risk-Free', 'Risk Premium']
        
        for i, component in enumerate(components):
            values = [decompositions[asset][component].mean() * 252 for asset in assets]
            fig.add_trace(
                go.Bar(
                    name=component_labels[i],
                    x=assets,
                    y=values,
                    marker_color=colors[i+1],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Risk premium vs total return scatter
        total_returns = [decompositions[asset]['total_return'].mean() * 252 for asset in assets]
        risk_premiums = [decompositions[asset]['spread'].mean() * 252 for asset in assets]
        
        fig.add_trace(
            go.Scatter(
                x=total_returns,
                y=risk_premiums,
                mode='markers+text',
                text=assets,
                textposition="top center",
                marker=dict(size=12, color=colors[:len(assets)]),
                showlegend=False,
                hovertemplate='%{text}<br>Total Return: %{x:.2%}<br>Risk Premium: %{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Cumulative performance comparison
        for i, (asset, decomp) in enumerate(decompositions.items()):
            cum_total = (1 + decomp['total_return']).cumprod()
            cum_real = (1 + decomp['total_return'] - decomp['inflation']).cumprod()
            
            fig.add_trace(
                go.Scatter(
                    x=cum_total.index,
                    y=cum_total.values,
                    name=f'{asset} Nominal',
                    line=dict(color=colors[i], width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=cum_real.index,
                    y=cum_real.values,
                    name=f'{asset} Real',
                    line=dict(color=colors[i], width=2, dash='dash'),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Return", tickformat=".1%", row=1, col=1)
        
        fig.update_xaxes(title_text="Asset", row=1, col=2)
        fig.update_yaxes(title_text="Component Return", tickformat=".1%", row=1, col=2)
        
        fig.update_xaxes(title_text="Total Return", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Risk Premium", tickformat=".1%", row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Return", row=2, col=2)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig