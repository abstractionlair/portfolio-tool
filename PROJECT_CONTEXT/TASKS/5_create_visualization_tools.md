# Task 5: Create Portfolio Visualization Tools

**Status**: TODO  
**Priority**: HIGH - User experience  
**Estimated Time**: 4-5 hours  
**Dependencies**: Core data infrastructure working

## Context
We have powerful analytics and optimization, but no way to visualize results. Users need to see performance charts, allocation breakdowns, and optimization results visually.

## Problem
- No visual representation of portfolio performance
- Optimization results are just numbers
- Can't see exposure breakdowns graphically
- No way to compare different scenarios

## Requirements

### 1. Create Visualization Module
Location: `/src/visualization/__init__.py`

Main components:
```python
from .performance import PerformanceVisualizer
from .allocation import AllocationVisualizer
from .optimization import OptimizationVisualizer
from .decomposition import DecompositionVisualizer
```

### 2. Performance Visualizer
Location: `/src/visualization/performance.py`

```python
class PerformanceVisualizer:
    """Create performance-related visualizations."""
    
    def plot_cumulative_returns(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        title: str = "Cumulative Returns",
        log_scale: bool = False
    ) -> plt.Figure:
        """Plot cumulative returns over time."""
        
    def plot_rolling_metrics(
        self,
        analytics: PortfolioAnalytics,
        metrics: List[str] = ['return', 'volatility', 'sharpe'],
        window: int = 252
    ) -> plt.Figure:
        """Plot rolling performance metrics."""
        
    def plot_drawdown(
        self,
        returns: pd.Series,
        title: str = "Drawdown Analysis"
    ) -> plt.Figure:
        """Plot drawdown chart with underwater plot."""
        
    def create_performance_dashboard(
        self,
        analytics: PortfolioAnalytics
    ) -> plt.Figure:
        """Create comprehensive performance dashboard."""
        # 2x2 grid with:
        # - Cumulative returns
        # - Rolling volatility
        # - Drawdown
        # - Return distribution
```

### 3. Allocation Visualizer
Location: `/src/visualization/allocation.py`

```python
class AllocationVisualizer:
    """Create allocation and exposure visualizations."""
    
    def plot_allocation_pie(
        self,
        weights: pd.Series,
        title: str = "Portfolio Allocation"
    ) -> plt.Figure:
        """Simple pie chart of allocations."""
        
    def plot_allocation_bar(
        self,
        weights: pd.Series,
        title: str = "Portfolio Weights"
    ) -> plt.Figure:
        """Bar chart with positive/negative positions."""
        
    def plot_exposure_breakdown(
        self,
        portfolio: Portfolio,
        fund_map: FundMap,
        exposure_type: Optional[str] = None
    ) -> plt.Figure:
        """Show exposure decomposition."""
        
    def plot_exposure_sunburst(
        self,
        exposures: pd.DataFrame
    ) -> go.Figure:
        """Interactive sunburst chart of exposures using Plotly."""
```

### 4. Optimization Visualizer
Location: `/src/visualization/optimization.py`

```python
class OptimizationVisualizer:
    """Visualize optimization results and efficient frontiers."""
    
    def plot_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_portfolios: int = 10000,
        optimal_points: Optional[List[OptimizationResult]] = None
    ) -> plt.Figure:
        """Plot efficient frontier with random portfolios."""
        
    def plot_optimization_comparison(
        self,
        results: Dict[str, OptimizationResult],
        metrics: List[str] = ['return', 'risk', 'sharpe']
    ) -> plt.Figure:
        """Compare multiple optimization results."""
        
    def plot_weight_comparison(
        self,
        results: Dict[str, OptimizationResult]
    ) -> plt.Figure:
        """Stacked bar chart comparing allocations."""
        
    def create_optimization_report(
        self,
        result: OptimizationResult,
        analytics: PortfolioAnalytics
    ) -> plt.Figure:
        """Comprehensive optimization report figure."""
```

### 5. Decomposition Visualizer
Location: `/src/visualization/decomposition.py`

```python
class DecompositionVisualizer:
    """Visualize return decomposition results."""
    
    def plot_return_components(
        self,
        decomposition: pd.DataFrame,
        title: str = "Return Decomposition"
    ) -> plt.Figure:
        """Stacked area chart of return components."""
        
    def plot_component_comparison(
        self,
        decompositions: Dict[str, pd.DataFrame]
    ) -> plt.Figure:
        """Compare decompositions across assets."""
        
    def plot_inflation_impact(
        self,
        nominal_returns: pd.Series,
        real_returns: pd.Series,
        inflation: pd.Series
    ) -> plt.Figure:
        """Show impact of inflation on returns."""
```

### 6. Create Interactive Dashboards

Location: `/examples/visualization_demo.py`

```python
"""Interactive portfolio visualization examples."""

def create_portfolio_dashboard(portfolio_path: str):
    """Create full interactive dashboard."""
    
    # Load portfolio
    portfolio = Portfolio.from_csv(portfolio_path)
    
    # Initialize visualizers
    perf_viz = PerformanceVisualizer()
    alloc_viz = AllocationVisualizer()
    opt_viz = OptimizationVisualizer()
    decomp_viz = DecompositionVisualizer()
    
    # Create subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Performance section
    ax1 = plt.subplot(3, 3, 1)
    perf_viz.plot_cumulative_returns(returns, ax=ax1)
    
    # Allocation section
    ax2 = plt.subplot(3, 3, 2)
    alloc_viz.plot_allocation_pie(weights, ax=ax2)
    
    # ... more subplots
    
    plt.tight_layout()
    plt.show()
```

### 7. Add Plotly Interactive Charts

Location: `/src/visualization/interactive.py`

```python
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
    
def create_interactive_allocation_treemap(
    portfolio: Portfolio,
    fund_map: FundMap
) -> go.Figure:
    """Interactive treemap of exposures."""
```

### 8. Add Tests

Location: `/tests/test_visualization.py`

```python
def test_performance_charts():
    """Test performance visualization."""
    
def test_allocation_charts():
    """Test allocation visualization."""
    
def test_interactive_charts():
    """Test Plotly interactive charts."""
    
def test_save_figures():
    """Test saving charts to files."""
```

## Testing Instructions

1. Create sample visualizations:
```python
# Test with sample data
returns = pd.Series(np.random.normal(0.0008, 0.01, 252))
viz = PerformanceVisualizer()
fig = viz.plot_cumulative_returns(returns)
plt.show()
```

2. Test with real portfolio:
```python
# Load and visualize
portfolio = Portfolio.from_csv('sample_portfolio.csv')
analytics = PortfolioAnalytics(portfolio, market_data)

# Create dashboard
viz = PerformanceVisualizer()
fig = viz.create_performance_dashboard(analytics)
fig.savefig('portfolio_dashboard.png', dpi=300)
```

3. Test optimization visualization:
```python
# Run optimization
result = engine.optimize(...)

# Visualize
opt_viz = OptimizationVisualizer()
fig = opt_viz.create_optimization_report(result, analytics)
```

## Success Criteria
- [ ] Basic performance charts working (cumulative returns, drawdown)
- [ ] Allocation visualization (pie, bar, treemap)
- [ ] Optimization results visualization
- [ ] Return decomposition charts
- [ ] Interactive Plotly charts for web use
- [ ] Charts are publication-quality
- [ ] Can save charts to files
- [ ] Comprehensive example notebook

## Visual Style Guidelines
- Use consistent color palette
- Professional appearance (good for reports)
- Clear labels and titles
- Appropriate chart types for data
- Interactive where beneficial
- Mobile-friendly sizes

## Output Formats
- Static: PNG, PDF, SVG via matplotlib
- Interactive: HTML via Plotly
- Notebook: Inline display in Jupyter

## Notes
- Start with matplotlib for static charts
- Add Plotly for interactive/web features
- Keep visualizations modular and reusable
- Consider colorblind-friendly palettes
- Make sure charts work in both light and dark themes
