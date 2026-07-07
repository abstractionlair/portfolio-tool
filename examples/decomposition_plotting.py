#!/usr/bin/env python3
"""
Plotting utilities for equity return decomposition analysis.

This module provides visualization functions for displaying
decomposition results in an intuitive way.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_decomposition_components(
    decomposition_result: Dict,
    ticker: str,
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Plot the main decomposition components over time.
    
    Args:
        decomposition_result: Results from decompose_equity_returns
        ticker: Stock ticker for title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{ticker} - Equity Return Decomposition Components', fontsize=16)
    
    # Get data
    data = decomposition_result['raw_data']
    
    # Main components
    main_components = ['nominal_return', 'dividend_yield', 'pe_change', 'nominal_earnings_growth']
    component_names = ['Total Return', 'Dividend Yield', 'P/E Change', 'Earnings Growth']
    
    for i, (component, name) in enumerate(zip(main_components, component_names)):
        ax = axes[i//2, i%2]
        
        if component in data:
            series = data[component].dropna()
            if len(series) > 0:
                # Plot time series
                ax.plot(series.index, series.values, alpha=0.7, linewidth=1)
                ax.set_title(f'{name}', fontsize=12)
                ax.set_ylabel('Daily Return')
                ax.grid(True, alpha=0.3)
                
                # Add zero line
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Format x-axis
                ax.tick_params(axis='x', rotation=45)
                
                # Add summary stats
                mean_annual = series.mean() * 252
                std_annual = series.std() * np.sqrt(252)
                ax.text(0.02, 0.98, f'Ann. Mean: {mean_annual:.1%}\nAnn. Std: {std_annual:.1%}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_real_vs_nominal_components(
    decomposition_result: Dict,
    ticker: str,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot real vs nominal return components.
    
    Args:
        decomposition_result: Results from decompose_equity_returns
        ticker: Stock ticker for title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{ticker} - Real vs Nominal Return Components', fontsize=16)
    
    data = decomposition_result['raw_data']
    
    # Nominal components
    ax1 = axes[0]
    nominal_components = ['dividend_yield', 'pe_change', 'nominal_earnings_growth']
    nominal_names = ['Dividend Yield', 'P/E Change', 'Earnings Growth']
    
    for component, name in zip(nominal_components, nominal_names):
        if component in data:
            series = data[component].dropna()
            if len(series) > 0:
                ax1.plot(series.index, series.cumsum(), label=name, linewidth=2)
    
    ax1.set_title('Nominal Components (Cumulative)', fontsize=12)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Real components
    ax2 = axes[1]
    real_components = ['dividend_yield', 'pe_change', 'real_earnings_excess']
    real_names = ['Dividend Yield', 'P/E Change', 'Real Earnings Excess']
    
    for component, name in zip(real_components, real_names):
        if component in data:
            series = data[component].dropna()
            if len(series) > 0:
                ax2.plot(series.index, series.cumsum(), label=name, linewidth=2)
    
    ax2.set_title('Real Components (Cumulative)', fontsize=12)
    ax2.set_ylabel('Cumulative Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_economic_context(
    decomposition_result: Dict,
    ticker: str,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot economic context (inflation, risk-free rates).
    
    Args:
        decomposition_result: Results from decompose_equity_returns
        ticker: Stock ticker for title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{ticker} - Economic Context', fontsize=16)
    
    data = decomposition_result['raw_data']
    
    # Inflation and risk-free rates
    ax1 = axes[0]
    economic_components = ['inflation', 'nominal_rf', 'real_rf']
    economic_names = ['Inflation', 'Nominal Risk-Free', 'Real Risk-Free']
    
    for component, name in zip(economic_components, economic_names):
        if component in data:
            series = data[component].dropna()
            if len(series) > 0:
                # Annualize for display
                annualized = series * 252
                ax1.plot(annualized.index, annualized.values, label=name, linewidth=2)
    
    ax1.set_title('Economic Rates (Annualized)', fontsize=12)
    ax1.set_ylabel('Annual Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Real vs nominal return
    ax2 = axes[1]
    if 'nominal_return' in data and 'real_risk_premium' in data:
        nominal = data['nominal_return'].dropna()
        real = data['real_risk_premium'].dropna()
        
        if len(nominal) > 0 and len(real) > 0:
            ax2.plot(nominal.index, nominal.cumsum(), label='Nominal Return', linewidth=2)
            ax2.plot(real.index, real.cumsum(), label='Real Risk Premium', linewidth=2)
            
            ax2.set_title('Nominal vs Real Returns (Cumulative)', fontsize=12)
            ax2.set_ylabel('Cumulative Return')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_component_distributions(
    decomposition_result: Dict,
    ticker: str,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot distributions of return components.
    
    Args:
        decomposition_result: Results from decompose_equity_returns
        ticker: Stock ticker for title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'{ticker} - Return Component Distributions', fontsize=16)
    
    data = decomposition_result['raw_data']
    
    components = [
        'nominal_return', 'dividend_yield', 'pe_change', 
        'nominal_earnings_growth', 'real_earnings_excess', 'real_risk_premium'
    ]
    
    component_names = [
        'Total Return', 'Dividend Yield', 'P/E Change',
        'Earnings Growth', 'Real Earnings Excess', 'Real Risk Premium'
    ]
    
    for i, (component, name) in enumerate(zip(components, component_names)):
        ax = axes[i//3, i%3]
        
        if component in data:
            series = data[component].dropna()
            if len(series) > 10:
                # Histogram
                ax.hist(series.values, bins=30, alpha=0.7, density=True, color='skyblue')
                
                # Add normal distribution overlay
                mu, sigma = series.mean(), series.std()
                x = np.linspace(series.min(), series.max(), 100)
                ax.plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2), 
                       'r-', linewidth=2, label='Normal')
                
                ax.set_title(f'{name}', fontsize=12)
                ax.set_xlabel('Daily Return')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                ax.text(0.02, 0.98, f'Mean: {mu:.4f}\nStd: {sigma:.4f}\nSkew: {series.skew():.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_stock_comparison(
    comparison_results: Dict,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comparison across multiple stocks.
    
    Args:
        comparison_results: Results from compare_stocks_decomposition
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Stock Comparison - Annualized Return Components', fontsize=16)
    
    # Extract data for comparison
    stocks = []
    components_data = {
        'nominal_return': [],
        'dividend_yield': [],
        'pe_change': [],
        'nominal_earnings_growth': []
    }
    
    for ticker, result in comparison_results.items():
        if 'summary' in result:
            stocks.append(ticker)
            for component in components_data.keys():
                if component in result['summary']:
                    components_data[component].append(result['summary'][component]['mean_annualized'])
                else:
                    components_data[component].append(0)
    
    if not stocks:
        # Create empty plot if no valid data
        ax = axes[0, 0]
        ax.text(0.5, 0.5, 'No valid data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Component comparison bar plots
    component_names = ['Total Return', 'Dividend Yield', 'P/E Change', 'Earnings Growth']
    
    for i, (component, name) in enumerate(zip(components_data.keys(), component_names)):
        ax = axes[i//2, i%2]
        
        bars = ax.bar(stocks, components_data[component], alpha=0.7)
        ax.set_title(f'{name} (Annualized)', fontsize=12)
        ax.set_ylabel('Return')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, components_data[component]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    decomposition_result: Dict,
    ticker: str,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot correlation heatmap of return components.
    
    Args:
        decomposition_result: Results from decompose_equity_returns
        ticker: Stock ticker for title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    data = decomposition_result['raw_data']
    
    # Select components for correlation analysis
    components = [
        'nominal_return', 'dividend_yield', 'pe_change', 
        'nominal_earnings_growth', 'real_earnings_excess',
        'inflation', 'real_rf'
    ]
    
    # Create DataFrame with available components
    df_data = {}
    for component in components:
        if component in data:
            series = data[component].dropna()
            if len(series) > 0:
                df_data[component] = series
    
    if df_data:
        df = pd.DataFrame(df_data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f')
        
        ax.set_title(f'{ticker} - Return Component Correlations', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
    else:
        ax.text(0.5, 0.5, 'No valid data available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


def create_decomposition_summary_table(
    decomposition_result: Dict,
    ticker: str
) -> pd.DataFrame:
    """
    Create a summary table of decomposition results.
    
    Args:
        decomposition_result: Results from decompose_equity_returns
        ticker: Stock ticker
        
    Returns:
        DataFrame with summary statistics
    """
    if 'summary' not in decomposition_result:
        return pd.DataFrame()
    
    summary_data = []
    
    components = [
        ('nominal_return', 'Total Nominal Return'),
        ('dividend_yield', 'Dividend Yield'),
        ('pe_change', 'P/E Change'),
        ('nominal_earnings_growth', 'Nominal Earnings Growth'),
        ('real_earnings_growth', 'Real Earnings Growth'),
        ('real_earnings_excess', 'Real Earnings Excess'),
        ('real_risk_premium', 'Real Risk Premium'),
        ('inflation', 'Inflation Rate'),
        ('nominal_rf', 'Nominal Risk-Free Rate'),
        ('real_rf', 'Real Risk-Free Rate')
    ]
    
    for component, name in components:
        if component in decomposition_result['summary']:
            stats = decomposition_result['summary'][component]
            summary_data.append({
                'Component': name,
                'Mean (Ann.)': f"{stats['mean_annualized']:.2%}",
                'Std (Ann.)': f"{stats['std_annualized']:.2%}",
                'Min': f"{stats['min']:.4f}",
                'Max': f"{stats['max']:.4f}",
                'Observations': stats['count']
            })
    
    return pd.DataFrame(summary_data)


def save_all_plots(
    decomposition_result: Dict,
    ticker: str,
    output_dir: str = "output/decomposition_plots"
) -> List[str]:
    """
    Save all plots for a decomposition analysis.
    
    Args:
        decomposition_result: Results from decompose_equity_returns
        ticker: Stock ticker
        output_dir: Output directory for plots
        
    Returns:
        List of saved file paths
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # Generate and save plots
    plots = [
        (plot_decomposition_components, 'components'),
        (plot_real_vs_nominal_components, 'real_vs_nominal'),
        (plot_economic_context, 'economic_context'),
        (plot_component_distributions, 'distributions'),
        (plot_correlation_heatmap, 'correlations')
    ]
    
    for plot_func, name in plots:
        try:
            fig = plot_func(decomposition_result, ticker)
            filepath = os.path.join(output_dir, f"{ticker}_{name}.png")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filepath)
        except Exception as e:
            print(f"Warning: Could not save {name} plot: {e}")
    
    return saved_files