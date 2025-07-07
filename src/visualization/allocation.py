"""
Allocation and exposure visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
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
    from portfolio.portfolio import Portfolio
    from portfolio.fund_exposure import FundMap
except ImportError:
    # Define stubs for typing purposes
    class Portfolio:
        pass
    class FundMap:
        pass


class AllocationVisualizer:
    """Create allocation and exposure visualizations."""
    
    def __init__(self, color_palette: Optional[List[str]] = None):
        """
        Initialize allocation visualizer.
        
        Args:
            color_palette: Custom color palette
        """
        self.color_palette = color_palette or [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def plot_allocation_pie(
        self,
        weights: pd.Series,
        title: str = "Portfolio Allocation",
        figsize: Tuple[int, int] = (10, 8),
        threshold: float = 0.02
    ) -> plt.Figure:
        """
        Create pie chart of portfolio allocations.
        
        Args:
            weights: Portfolio weights (should sum to ~1.0)
            title: Chart title
            figsize: Figure size
            threshold: Minimum weight to show separately (others grouped as "Other")
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group small allocations
        weights_filtered = weights.abs()
        small_weights = weights_filtered[weights_filtered < threshold]
        large_weights = weights_filtered[weights_filtered >= threshold]
        
        # Prepare data for pie chart
        if len(small_weights) > 0:
            other_weight = small_weights.sum()
            plot_weights = large_weights.copy()
            plot_weights['Other'] = other_weight
        else:
            plot_weights = large_weights.copy()
        
        # Create pie chart
        colors = self.color_palette[:len(plot_weights)]
        wedges, texts, autotexts = ax.pie(
            plot_weights.values,
            labels=plot_weights.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend with actual values
        legend_labels = [f'{name}: {weight:.1%}' for name, weight in plot_weights.items()]
        ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        return fig
    
    def plot_allocation_bar(
        self,
        weights: pd.Series,
        title: str = "Portfolio Weights",
        figsize: Tuple[int, int] = (12, 6),
        show_net: bool = True
    ) -> plt.Figure:
        """
        Create bar chart showing positive and negative positions.
        
        Args:
            weights: Portfolio weights (can include negative values)
            title: Chart title
            figsize: Figure size
            show_net: Whether to show net exposure line
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Separate positive and negative weights
        pos_weights = weights.where(weights >= 0, 0)
        neg_weights = weights.where(weights < 0, 0)
        
        x_pos = np.arange(len(weights))
        
        # Create bars
        bars_pos = ax.bar(x_pos, pos_weights.values, 
                         color=self.color_palette[0], alpha=0.8, label='Long')
        bars_neg = ax.bar(x_pos, neg_weights.values,
                         color=self.color_palette[3], alpha=0.8, label='Short')
        
        # Add value labels on bars
        for i, (pos, neg) in enumerate(zip(pos_weights.values, neg_weights.values)):
            if abs(pos) > 0.01:  # Only label significant positions
                ax.text(i, pos + 0.01, f'{pos:.1%}', ha='center', va='bottom', fontsize=9)
            if abs(neg) > 0.01:
                ax.text(i, neg - 0.01, f'{neg:.1%}', ha='center', va='top', fontsize=9)
        
        # Show net exposure line if requested
        if show_net:
            net_exposure = weights.abs().sum()
            ax.axhline(y=net_exposure, color='red', linestyle='--', alpha=0.7,
                      label=f'Net Exposure: {net_exposure:.1%}')
            ax.axhline(y=-net_exposure, color='red', linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(weights.index, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        return fig
    
    def plot_exposure_breakdown(
        self,
        portfolio: Portfolio,
        fund_map: FundMap,
        exposure_type: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Show exposure decomposition of portfolio.
        
        Args:
            portfolio: Portfolio object
            fund_map: Fund exposure mapping
            exposure_type: Specific exposure type to filter by
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        # Calculate exposures
        exposures = portfolio.calculate_exposures(fund_map)
        
        if exposure_type:
            exposures = exposures[exposures.index.str.contains(exposure_type, case=False)]
        
        # Group by major categories
        exposure_categories = self._categorize_exposures(exposures)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Exposure by category (pie chart)
        category_sums = exposure_categories.groupby('category').sum()['exposure']
        colors1 = self.color_palette[:len(category_sums)]
        
        wedges, texts, autotexts = ax1.pie(
            category_sums.values,
            labels=category_sums.index,
            autopct='%1.1f%%',
            colors=colors1,
            startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax1.set_title('Exposure by Category', fontsize=12, fontweight='bold')
        
        # Plot 2: Individual exposures (bar chart)
        top_exposures = exposures.abs().nlargest(15)  # Show top 15
        colors2 = [self.color_palette[i % len(self.color_palette)] for i in range(len(top_exposures))]
        
        bars = ax2.barh(range(len(top_exposures)), top_exposures.values, color=colors2)
        ax2.set_yticks(range(len(top_exposures)))
        ax2.set_yticklabels(top_exposures.index, fontsize=10)
        ax2.set_xlabel('Exposure')
        ax2.set_title('Top Individual Exposures', fontsize=12, fontweight='bold')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_exposures.values)):
            ax2.text(value + 0.01, i, f'{value:.1%}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_allocation_comparison(
        self,
        allocations: Dict[str, pd.Series],
        title: str = "Portfolio Allocation Comparison",
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Compare multiple portfolio allocations side by side.
        
        Args:
            allocations: Dict mapping portfolio names to weight series
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Combine all allocations into DataFrame
        allocation_df = pd.DataFrame(allocations).fillna(0)
        
        # Create grouped bar chart
        x = np.arange(len(allocation_df.index))
        width = 0.8 / len(allocation_df.columns)
        
        for i, col in enumerate(allocation_df.columns):
            offset = (i - len(allocation_df.columns)/2) * width + width/2
            bars = ax.bar(x + offset, allocation_df[col].values, width,
                         label=col, color=self.color_palette[i % len(self.color_palette)],
                         alpha=0.8)
            
            # Add value labels on bars for significant allocations
            for j, (bar, value) in enumerate(zip(bars, allocation_df[col].values)):
                if abs(value) > 0.05:  # Only label if > 5%
                    ax.text(bar.get_x() + bar.get_width()/2, value + 0.01,
                           f'{value:.1%}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight')
        ax.set_xticks(x)
        ax.set_xticklabels(allocation_df.index, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_allocation_treemap(
        self,
        portfolio: Portfolio,
        fund_map: FundMap,
        title: str = "Portfolio Exposure Treemap"
    ) -> go.Figure:
        """
        Create interactive treemap of portfolio exposures.
        
        Args:
            portfolio: Portfolio object
            fund_map: Fund exposure mapping
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive charts")
        
        # Calculate exposures
        exposures = portfolio.calculate_exposures(fund_map)
        exposure_categories = self._categorize_exposures(exposures)
        
        # Prepare data for treemap
        categories = []
        subcategories = []
        values = []
        parents = []
        
        # Add top-level categories
        category_sums = exposure_categories.groupby('category').sum()['exposure']
        for cat in category_sums.index:
            categories.append(cat)
            values.append(category_sums[cat])
            parents.append("")
        
        # Add individual exposures
        for _, row in exposure_categories.iterrows():
            if row['exposure'] > 0.01:  # Only show significant exposures
                categories.append(row['exposure_name'])
                values.append(row['exposure'])
                parents.append(row['category'])
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=categories,
            values=values,
            parents=parents,
            textinfo="label+percent parent",
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<br>% of Parent: %{percentParent}<extra></extra>',
            maxdepth=2,
            branchvalues="total"
        ))
        
        fig.update_layout(
            title=title,
            font_size=12,
            height=600
        )
        
        return fig
    
    def plot_allocation_waterfall(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        title: str = "Portfolio Rebalancing Waterfall",
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Create waterfall chart showing changes from current to target allocation.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            title: Chart title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate changes
        all_assets = current_weights.index.union(target_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0)
        target_aligned = target_weights.reindex(all_assets, fill_value=0)
        changes = target_aligned - current_aligned
        
        # Filter to significant changes
        significant_changes = changes[changes.abs() > 0.01]
        
        # Separate increases and decreases
        increases = significant_changes[significant_changes > 0]
        decreases = significant_changes[significant_changes < 0]
        
        # Create waterfall effect
        x_pos = np.arange(len(significant_changes))
        
        # Plot bars
        colors = []
        for change in significant_changes.values:
            if change > 0:
                colors.append(self.color_palette[0])  # Green for increases
            else:
                colors.append(self.color_palette[3])  # Red for decreases
        
        bars = ax.bar(x_pos, significant_changes.values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, significant_changes.values)):
            label_y = value + (0.005 if value > 0 else -0.005)
            ax.text(bar.get_x() + bar.get_width()/2, label_y,
                   f'{value:+.1%}', ha='center',
                   va='bottom' if value > 0 else 'top', fontsize=9)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight Change')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(significant_changes.index, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:+.0%}'.format(y)))
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        # Add summary statistics
        total_change = changes.abs().sum()
        net_change = changes.sum()
        ax.text(0.02, 0.98, f'Total Turnover: {total_change:.1%}\nNet Change: {net_change:+.1%}',
               transform=ax.transAxes, va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        return fig
    
    def _categorize_exposures(self, exposures: pd.Series) -> pd.DataFrame:
        """
        Categorize exposures into major groups.
        
        Args:
            exposures: Series of exposure values
            
        Returns:
            DataFrame with exposure name, category, and value
        """
        categories = []
        
        for exposure_name in exposures.index:
            exposure_lower = exposure_name.lower()
            
            if any(keyword in exposure_lower for keyword in ['equity', 'stock', 'sp500', 'russell', 'msci']):
                category = 'Equity'
            elif any(keyword in exposure_lower for keyword in ['bond', 'treasury', 'corporate', 'credit']):
                category = 'Fixed Income'
            elif any(keyword in exposure_lower for keyword in ['commodity', 'gold', 'oil', 'real estate', 'reit']):
                category = 'Alternatives'
            elif any(keyword in exposure_lower for keyword in ['international', 'emerging', 'developed']):
                category = 'International'
            elif any(keyword in exposure_lower for keyword in ['currency', 'fx', 'dollar']):
                category = 'Currency'
            else:
                category = 'Other'
            
            categories.append(category)
        
        return pd.DataFrame({
            'exposure_name': exposures.index,
            'category': categories,
            'exposure': exposures.values
        })