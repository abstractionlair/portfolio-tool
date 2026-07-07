"""
Performance visualization tools for portfolio analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Union, List, Optional, Dict, Tuple
from datetime import datetime
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Optional import of portfolio analytics (for type hints only)
try:
    from portfolio.analytics import PortfolioAnalytics
except ImportError:
    # Define a stub for typing purposes
    class PortfolioAnalytics:
        pass


class PerformanceVisualizer:
    """Create performance-related visualizations."""
    
    def __init__(self, style: str = 'default', color_palette: Optional[List[str]] = None):
        """
        Initialize performance visualizer.
        
        Args:
            style: Matplotlib style ('default', 'seaborn', 'bmh')
            color_palette: Custom color palette
        """
        self.style = style
        self.color_palette = color_palette or [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Set style
        if style != 'default':
            try:
                plt.style.use(style)
            except OSError:
                warnings.warn(f"Style '{style}' not available, using default")
    
    def plot_cumulative_returns(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        title: str = "Cumulative Returns",
        log_scale: bool = False,
        benchmark: Optional[pd.Series] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot cumulative returns over time.
        
        Args:
            returns: Return series or dataframe
            title: Chart title
            log_scale: Use logarithmic scale
            benchmark: Benchmark returns for comparison
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative returns
        if isinstance(returns, pd.Series):
            cum_returns = (1 + returns).cumprod()
            ax.plot(cum_returns.index, cum_returns.values, 
                   label=returns.name or 'Portfolio', 
                   color=self.color_palette[0], linewidth=2)
        else:
            cum_returns = (1 + returns).cumprod()
            for i, col in enumerate(cum_returns.columns):
                ax.plot(cum_returns.index, cum_returns[col], 
                       label=col, color=self.color_palette[i % len(self.color_palette)],
                       linewidth=2)
        
        # Add benchmark if provided
        if benchmark is not None:
            cum_benchmark = (1 + benchmark).cumprod()
            ax.plot(cum_benchmark.index, cum_benchmark.values,
                   label='Benchmark', color='gray', linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (Base = 1.0)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Cumulative Return (Log Scale)')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        metrics: List[str] = ['volatility', 'sharpe'],
        window: int = 252,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Return series
            metrics: List of metrics to plot
            window: Rolling window size
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric == 'return':
                rolling_data = returns.rolling(window).mean() * 252
                ax.plot(rolling_data.index, rolling_data.values, 
                       color=self.color_palette[0], linewidth=2)
                ax.set_ylabel('Annual Return')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
                
            elif metric == 'volatility':
                rolling_data = returns.rolling(window).std() * np.sqrt(252)
                ax.plot(rolling_data.index, rolling_data.values,
                       color=self.color_palette[1], linewidth=2)
                ax.set_ylabel('Volatility')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
                
            elif metric == 'sharpe':
                rolling_mean = returns.rolling(window).mean()
                rolling_std = returns.rolling(window).std()
                rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
                ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                       color=self.color_palette[2], linewidth=2)
                ax.set_ylabel('Sharpe Ratio')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
            elif metric == 'drawdown':
                cum_returns = (1 + returns).cumprod()
                rolling_max = cum_returns.rolling(window, min_periods=1).max()
                drawdown = (cum_returns - rolling_max) / rolling_max
                ax.fill_between(drawdown.index, drawdown.values, 0,
                               color=self.color_palette[3], alpha=0.3)
                ax.plot(drawdown.index, drawdown.values,
                       color=self.color_palette[3], linewidth=1)
                ax.set_ylabel('Drawdown')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
                
            ax.set_title(f'Rolling {metric.title()} ({window} days)', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Format x-axis for bottom plot
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        axes[-1].set_xlabel('Date')
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(
        self,
        returns: pd.Series,
        title: str = "Drawdown Analysis",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot drawdown chart with underwater plot.
        
        Args:
            returns: Return series
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        # Calculate cumulative returns and drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        
        # Create subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
        
        # Cumulative returns plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(cum_returns.index, cum_returns.values, 
                color=self.color_palette[0], linewidth=2, label='Cumulative Return')
        ax1.plot(rolling_max.index, rolling_max.values,
                color='red', linestyle='--', alpha=0.7, label='Peak')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Drawdown plot
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax2.set_ylabel('Drawdown')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax2.grid(True, alpha=0.3)
        
        # Drawdown statistics
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        # Calculate drawdown statistics
        max_dd = drawdown.min()
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown)
        current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        stats_text = f"""
        Maximum Drawdown: {max_dd:.2%}
        Max DD Duration: {max_dd_duration} days
        Current Drawdown: {current_dd:.2%}
        Recovery Factor: {(cum_returns.iloc[-1] - 1) / abs(max_dd):.2f}
        """
        
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        ax2.set_xlabel('Date')
        
        plt.tight_layout()
        return fig
    
    def create_performance_dashboard(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Portfolio Performance Dashboard",
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Create comprehensive performance dashboard.
        
        Args:
            returns: Portfolio return series
            benchmark: Benchmark return series
            title: Dashboard title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        # Calculate metrics
        cum_returns = (1 + returns).cumprod()
        annual_return = (cum_returns.iloc[-1] ** (252 / len(returns))) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # 1. Cumulative Returns (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(cum_returns.index, cum_returns.values, 
                color=self.color_palette[0], linewidth=2, label='Portfolio')
        
        if benchmark is not None:
            cum_benchmark = (1 + benchmark).cumprod()
            ax1.plot(cum_benchmark.index, cum_benchmark.values,
                    color='gray', linestyle='--', alpha=0.7, label='Benchmark')
        
        ax1.set_title(f'{title} - Cumulative Returns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Rolling Volatility
        ax2 = fig.add_subplot(gs[1, 0])
        rolling_vol = returns.rolling(63).std() * np.sqrt(252)  # Quarterly rolling
        ax2.plot(rolling_vol.index, rolling_vol.values, color=self.color_palette[1])
        ax2.set_title('Rolling Volatility (63d)', fontweight='bold')
        ax2.set_ylabel('Volatility')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        ax3.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax3.set_title('Drawdown', fontweight='bold')
        ax3.set_ylabel('Drawdown')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax3.grid(True, alpha=0.3)
        
        # 4. Return Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(returns.dropna(), bins=50, alpha=0.7, color=self.color_palette[0], density=True)
        ax4.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax4.set_title('Return Distribution', fontweight='bold')
        ax4.set_xlabel('Daily Return')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Statistics Table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Calculate comprehensive statistics
        max_dd = drawdown.min()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else np.inf
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        stats_data = [
            ['Annual Return', f'{annual_return:.2%}'],
            ['Volatility', f'{annual_vol:.2%}'],
            ['Sharpe Ratio', f'{sharpe:.2f}'],
            ['Maximum Drawdown', f'{max_dd:.2%}'],
            ['Calmar Ratio', f'{calmar:.2f}'],
            ['VaR (95%)', f'{var_95:.2%}'],
            ['VaR (99%)', f'{var_99:.2%}'],
            ['Skewness', f'{returns.skew():.3f}'],
            ['Kurtosis', f'{returns.kurtosis():.3f}'],
        ]
        
        if benchmark is not None:
            excess_returns = returns - benchmark.reindex(returns.index).fillna(0)
            tracking_error = excess_returns.std() * np.sqrt(252)
            info_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
            
            stats_data.extend([
                ['Tracking Error', f'{tracking_error:.2%}'],
                ['Information Ratio', f'{info_ratio:.2f}']
            ])
        
        # Create table
        table = ax5.table(cellText=stats_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.2, 0.1, 0.6, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(stats_data)):
            table[(i+1, 0)].set_facecolor('#f0f0f0')
            table[(i+1, 1)].set_facecolor('#ffffff')
        
        # Header styling
        table[(0, 0)].set_facecolor('#4472C4')
        table[(0, 1)].set_facecolor('#4472C4')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_performance_chart(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        title: str = "Interactive Performance Analysis"
    ) -> go.Figure:
        """
        Create interactive Plotly chart with multiple features.
        
        Args:
            returns: Return series or dataframe
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive charts")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Cumulative Returns', 'Rolling Volatility', 'Drawdown'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        if isinstance(returns, pd.Series):
            returns = pd.DataFrame({returns.name or 'Portfolio': returns})
        
        colors = px.colors.qualitative.Set1
        
        for i, col in enumerate(returns.columns):
            color = colors[i % len(colors)]
            series = returns[col].dropna()
            
            # Cumulative returns
            cum_returns = (1 + series).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    name=col,
                    line=dict(color=color, width=2),
                    hovertemplate=f'{col}<br>Date: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Rolling volatility
            rolling_vol = series.rolling(63).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    name=f'{col} Vol',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hovertemplate=f'{col} Volatility<br>Date: %{{x}}<br>Volatility: %{{y:.2%}}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Drawdown
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    fill='tonexty' if i == 0 else None,
                    name=f'{col} DD',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hovertemplate=f'{col} Drawdown<br>Date: %{{x}}<br>Drawdown: %{{y:.2%}}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=False),
                type="date"
            )
        )
        
        # Format y-axes
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=3, col=1)
        
        return fig
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        is_dd = drawdown < 0
        dd_groups = (is_dd != is_dd.shift()).cumsum()
        dd_durations = is_dd.groupby(dd_groups).sum()
        return int(dd_durations.max()) if len(dd_durations) > 0 else 0