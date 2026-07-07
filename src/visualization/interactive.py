"""
Interactive Visualization
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict

from src.portfolio.portfolio import Portfolio
from src.portfolio.exposures import FundExposureMap

def create_interactive_performance_chart(
    returns: pd.DataFrame,
    title: str = "Interactive Performance Analysis"
) -> go.Figure:
    """Create interactive Plotly chart with:
    - Multiple series
    - Zoom and pan
    - Hover details
    - Range selector
    """
    fig = go.Figure()
    for col in returns.columns:
        fig.add_trace(go.Scatter(x=returns.index, y=(1 + returns[col]).cumprod(), name=col))
        
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        legend_title="Assets",
        xaxis_rangeslider_visible=True
    )
    return fig

def create_interactive_allocation_treemap(
    portfolio: Portfolio,
    fund_map: FundExposureMap
) -> go.Figure:
    """Interactive treemap of exposures."""
    exposures = portfolio.get_exposure_breakdown(fund_map)
    
    # This is a simplified example, a more complex data structure would be needed for a multi-level treemap
    labels = exposures.columns
    parents = [""] * len(labels)
    values = exposures.sum().values
    
    fig = go.Figure(go.Treemap(
        labels = labels,
        parents = parents,
        values = values,
    ))
    
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    return fig
