"""
Optimization result visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Optional imports for type hints
try:
    from optimization.engine import OptimizationResult
except ImportError:
    # Define stub for typing purposes
    class OptimizationResult:
        def __init__(self, weights, expected_return, volatility, sharpe_ratio, metadata=None):
            self.weights = weights
            self.expected_return = expected_return
            self.volatility = volatility
            self.sharpe_ratio = sharpe_ratio
            self.metadata = metadata or {}


class OptimizationVisualizer:
    """Visualize optimization results and efficient frontiers."""
    
    def __init__(self, color_palette: Optional[List[str]] = None):
        """
        Initialize optimization visualizer.
        
        Args:
            color_palette: Custom color palette
        """
        self.color_palette = color_palette or [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def plot_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_portfolios: int = 10000,
        optimal_points: Optional[List[OptimizationResult]] = None,
        risk_free_rate: float = 0.02,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot efficient frontier with random portfolios and optimal points.
        
        Args:
            returns: Historical returns dataframe
            n_portfolios: Number of random portfolios to generate
            optimal_points: List of optimal portfolios to highlight
            risk_free_rate: Risk-free rate for CAL
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate random portfolios
        n_assets = len(returns.columns)
        results = np.zeros((3, n_portfolios))
        
        # Calculate expected returns and covariance
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        for i in range(n_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            # Calculate portfolio return and risk
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Calculate Sharpe ratio
            sharpe = (portfolio_return - risk_free_rate) / portfolio_std
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std
            results[2, i] = sharpe
        
        # Create scatter plot of random portfolios
        scatter = ax.scatter(results[1], results[0], c=results[2], cmap='viridis', 
                           alpha=0.5, s=10, label='Random Portfolios')
        
        # Add colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
        
        # Plot optimal points if provided
        if optimal_points:
            for i, result in enumerate(optimal_points):
                color = self.color_palette[i % len(self.color_palette)]
                volatility = getattr(result, 'volatility', getattr(result, 'expected_volatility', 0))
                ax.scatter(volatility, result.expected_return, 
                          color=color, s=100, marker='*', 
                          label=f'Optimal {i+1}', edgecolors='black', linewidth=1)
                
                # Add annotation
                ax.annotate(f'SR: {result.sharpe_ratio:.2f}',
                           (volatility, result.expected_return),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                           fontsize=9)
        
        # Plot Capital Allocation Line (CAL) for max Sharpe portfolio
        if optimal_points:
            max_sharpe_result = max(optimal_points, key=lambda x: x.sharpe_ratio)
            cal_x = np.linspace(0, max(results[1]) * 1.2, 100)
            cal_y = risk_free_rate + max_sharpe_result.sharpe_ratio * cal_x
            ax.plot(cal_x, cal_y, 'r--', linewidth=2, alpha=0.7, label='Capital Allocation Line')
        
        # Formatting
        ax.set_title('Efficient Frontier Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Volatility (Risk)')
        ax.set_ylabel('Expected Return')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_optimization_comparison(
        self,
        results: Dict[str, OptimizationResult],
        metrics: List[str] = ['expected_return', 'volatility', 'sharpe_ratio'],
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Compare multiple optimization results across key metrics.
        
        Args:
            results: Dict mapping strategy names to optimization results
            metrics: List of metrics to compare
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        strategies = list(results.keys())
        n_strategies = len(strategies)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract metric values
            values = []
            for strategy in strategies:
                result = results[strategy]
                if metric == 'volatility':
                    # Handle both volatility and expected_volatility
                    values.append(getattr(result, 'volatility', getattr(result, 'expected_volatility', 0)))
                elif hasattr(result, metric):
                    values.append(getattr(result, metric))
                else:
                    values.append(0)  # Default value if metric not found
            
            # Create bar chart
            bars = ax.bar(range(n_strategies), values, 
                         color=[self.color_palette[j % len(self.color_palette)] for j in range(n_strategies)],
                         alpha=0.8)
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                if metric in ['expected_return', 'volatility']:
                    label = f'{value:.1%}'
                else:
                    label = f'{value:.2f}'
                
                ax.text(bar.get_x() + bar.get_width()/2, value + max(values) * 0.01,
                       label, ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Formatting
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_xticks(range(n_strategies))
            ax.set_xticklabels(strategies, rotation=45, ha='right')
            
            if metric in ['expected_return', 'volatility']:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            
            ax.grid(True, alpha=0.3, axis='y')
            
            # Highlight best performer
            if metric == 'sharpe_ratio':
                best_idx = np.argmax(values)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        return fig
    
    def plot_weight_comparison(
        self,
        results: Dict[str, OptimizationResult],
        title: str = "Portfolio Weight Comparison",
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Create stacked bar chart comparing allocations across strategies.
        
        Args:
            results: Dict mapping strategy names to optimization results
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Combine all weights into DataFrame
        weights_dict = {name: result.weights for name, result in results.items()}
        weights_df = pd.DataFrame(weights_dict).fillna(0)
        
        # Create stacked bar chart
        bottom = np.zeros(len(weights_df.columns))
        
        for i, asset in enumerate(weights_df.index):
            values = weights_df.loc[asset].values
            color = self.color_palette[i % len(self.color_palette)]
            
            bars = ax.bar(weights_df.columns, values, bottom=bottom, 
                         label=asset, color=color, alpha=0.8)
            
            # Add labels for significant weights
            for j, (bar, value) in enumerate(zip(bars, values)):
                if abs(value) > 0.05:  # Only label if > 5%
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bottom[j] + value/2,
                           f'{value:.1%}', ha='center', va='center',
                           fontsize=9, fontweight='bold')
            
            bottom += values
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Optimization Strategy')
        ax.set_ylabel('Portfolio Weight')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def create_optimization_report(
        self,
        result: OptimizationResult,
        returns: Optional[pd.DataFrame] = None,
        title: str = "Optimization Report",
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Create comprehensive optimization report figure.
        
        Args:
            result: Optimization result to analyze
            returns: Historical returns for context
            title: Report title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=figsize)
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio weights (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        weights_to_plot = result.weights[result.weights.abs() > 0.01]  # Filter small weights
        
        if len(weights_to_plot) > 0:
            colors = self.color_palette[:len(weights_to_plot)]
            wedges, texts, autotexts = ax1.pie(
                weights_to_plot.abs().values,
                labels=weights_to_plot.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
        
        ax1.set_title('Portfolio Allocation', fontweight='bold')
        
        # 2. Weight bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        weight_bars = ax2.bar(range(len(result.weights)), result.weights.values,
                             color=[self.color_palette[i % len(self.color_palette)] 
                                   for i in range(len(result.weights))])
        ax2.set_title('Portfolio Weights', fontweight='bold')
        ax2.set_xticks(range(len(result.weights)))
        ax2.set_xticklabels(result.weights.index, rotation=45, ha='right')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linewidth=0.8)
        
        # 3. Key metrics
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        volatility = getattr(result, 'volatility', getattr(result, 'expected_volatility', 0))
        metrics_text = f"""
        Expected Return: {result.expected_return:.2%}
        Volatility: {volatility:.2%}
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        
        Total Gross Exposure: {result.weights.abs().sum():.1%}
        Number of Positions: {(result.weights.abs() > 0.01).sum()}
        Max Position: {result.weights.abs().max():.1%}
        """
        
        ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax3.set_title('Key Metrics', fontweight='bold')
        
        # 4. Risk decomposition (if available)
        ax4 = fig.add_subplot(gs[1, :2])
        if returns is not None:
            # Calculate contribution to portfolio risk
            cov_matrix = returns.cov() * 252
            portfolio_var = result.weights.T @ cov_matrix @ result.weights
            risk_contributions = (result.weights * (cov_matrix @ result.weights)) / portfolio_var
            
            # Plot risk contributions
            risk_bars = ax4.bar(range(len(risk_contributions)), risk_contributions.values * 100,
                               color=[self.color_palette[i % len(self.color_palette)] 
                                     for i in range(len(risk_contributions))])
            ax4.set_title('Risk Contribution by Asset (%)', fontweight='bold')
            ax4.set_xticks(range(len(risk_contributions)))
            ax4.set_xticklabels(risk_contributions.index, rotation=45, ha='right')
            ax4.set_ylabel('Risk Contribution (%)')
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Risk decomposition requires\nhistorical returns data',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Risk Decomposition', fontweight='bold')
        
        # 5. Optimization details
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        details_text = "Optimization Details:\n\n"
        if hasattr(result, 'metadata') and result.metadata:
            for key, value in result.metadata.items():
                if isinstance(value, float):
                    details_text += f"{key.replace('_', ' ').title()}: {value:.3f}\n"
                else:
                    details_text += f"{key.replace('_', ' ').title()}: {value}\n"
        else:
            details_text += "No metadata available"
        
        ax5.text(0.1, 0.9, details_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        ax5.set_title('Optimization Details', fontweight='bold')
        
        # 6. Portfolio composition table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create table data
        table_data = []
        for asset, weight in result.weights.items():
            if abs(weight) > 0.001:  # Only show significant weights
                table_data.append([
                    asset,
                    f'{weight:.2%}',
                    f'{weight * result.expected_return:.2%}' if hasattr(result, 'expected_return') else 'N/A',
                    'Long' if weight > 0 else 'Short' if weight < 0 else 'Zero'
                ])
        
        if table_data:
            table = ax6.table(
                cellText=table_data,
                colLabels=['Asset', 'Weight', 'Expected Contribution', 'Direction'],
                cellLoc='center',
                loc='center',
                bbox=[0.1, 0.1, 0.8, 0.8]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Style the table
            for i in range(len(table_data)):
                table[(i+1, 0)].set_facecolor('#f0f0f0')
                table[(i+1, 1)].set_facecolor('#ffffff')
                table[(i+1, 2)].set_facecolor('#f0f0f0')
                table[(i+1, 3)].set_facecolor('#ffffff')
            
            # Header styling
            for j in range(4):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Portfolio Composition', fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def plot_constraint_analysis(
        self,
        result: OptimizationResult,
        constraints: Dict[str, Any],
        title: str = "Constraint Analysis",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualize how the optimization result respects various constraints.
        
        Args:
            result: Optimization result
            constraints: Dict of constraints that were applied
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Weight bounds
        ax = axes[0]
        if 'weight_bounds' in constraints:
            bounds = constraints['weight_bounds']
            if isinstance(bounds, dict):
                assets = list(bounds.keys())
                lower_bounds = [bounds[asset][0] for asset in assets]
                upper_bounds = [bounds[asset][1] for asset in assets]
                actual_weights = [result.weights.get(asset, 0) for asset in assets]
                
                x_pos = np.arange(len(assets))
                
                # Plot bounds as error bars
                ax.errorbar(x_pos, actual_weights, 
                           yerr=[np.array(actual_weights) - np.array(lower_bounds),
                                 np.array(upper_bounds) - np.array(actual_weights)],
                           fmt='o', capsize=5, capthick=2)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(assets, rotation=45, ha='right')
                ax.set_ylabel('Weight')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
                ax.grid(True, alpha=0.3)
        
        ax.set_title('Weight Bounds Compliance', fontweight='bold')
        
        # 2. Sector/exposure limits
        ax = axes[1]
        ax.text(0.5, 0.5, 'Exposure Constraints\n(Implementation varies\nby constraint type)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Exposure Constraints', fontweight='bold')
        
        # 3. Turnover constraints
        ax = axes[2]
        if 'max_turnover' in constraints:
            # This would require current portfolio for comparison
            ax.text(0.5, 0.5, f"Max Turnover: {constraints['max_turnover']:.1%}\n(Actual turnover depends\non current portfolio)",
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No turnover constraints', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        ax.set_title('Turnover Constraints', fontweight='bold')
        
        # 4. Risk constraints
        ax = axes[3]
        constraint_text = ""
        if 'max_volatility' in constraints:
            constraint_text += f"Max Volatility: {constraints['max_volatility']:.1%}\n"
            constraint_text += f"Actual Volatility: {result.volatility:.1%}\n"
        if 'min_return' in constraints:
            constraint_text += f"Min Return: {constraints['min_return']:.1%}\n"
            constraint_text += f"Actual Return: {result.expected_return:.1%}\n"
        
        if constraint_text:
            ax.text(0.1, 0.9, constraint_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        else:
            ax.text(0.5, 0.5, 'No risk constraints', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title('Risk Constraints', fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_interactive_optimization_dashboard(
        self,
        results: Dict[str, OptimizationResult],
        title: str = "Interactive Optimization Dashboard"
    ) -> go.Figure:
        """
        Create interactive dashboard comparing multiple optimization results.
        
        Args:
            results: Dict mapping strategy names to results
            title: Dashboard title
            
        Returns:
            plotly Figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive charts")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk-Return Scatter', 'Weight Comparison', 
                          'Sharpe Ratio Comparison', 'Portfolio Composition'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        strategies = list(results.keys())
        colors = px.colors.qualitative.Set1
        
        # 1. Risk-Return scatter
        for i, (name, result) in enumerate(results.items()):
            fig.add_trace(
                go.Scatter(
                    x=[result.volatility],
                    y=[result.expected_return],
                    mode='markers+text',
                    name=name,
                    text=[name],
                    textposition="top center",
                    marker=dict(size=15, color=colors[i % len(colors)]),
                    hovertemplate=f'{name}<br>Return: %{{y:.2%}}<br>Risk: %{{x:.2%}}<br>Sharpe: {result.sharpe_ratio:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Weight comparison (stacked bar)
        weights_df = pd.DataFrame({name: result.weights for name, result in results.items()}).fillna(0)
        
        for i, asset in enumerate(weights_df.index):
            fig.add_trace(
                go.Bar(
                    name=asset,
                    x=strategies,
                    y=weights_df.loc[asset],
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Sharpe ratio comparison
        sharpe_ratios = [result.sharpe_ratio for result in results.values()]
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=sharpe_ratios,
                marker_color=colors[:len(strategies)],
                showlegend=False,
                text=[f'{sr:.2f}' for sr in sharpe_ratios],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # 4. Portfolio composition table
        # Prepare table data
        all_assets = set()
        for result in results.values():
            all_assets.update(result.weights.index)
        
        table_data = []
        for asset in sorted(all_assets):
            row = [asset]
            for result in results.values():
                weight = result.weights.get(asset, 0)
                row.append(f'{weight:.1%}')
            table_data.append(row)
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Asset'] + strategies,
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=list(zip(*table_data)),
                          fill_color='lavender',
                          align='left')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Volatility", tickformat=".1%", row=1, col=1)
        fig.update_yaxes(title_text="Expected Return", tickformat=".1%", row=1, col=1)
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_yaxes(title_text="Weight", tickformat=".0%", row=1, col=2)
        fig.update_xaxes(title_text="Strategy", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            barmode='stack'
        )
        
        return fig