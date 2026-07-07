# Task: Real Return Optimization

**Status**: TODO  
**Priority**: HIGH - Core feature for sophisticated optimization  
**Estimated Time**: 8-10 hours  
**Dependencies**: Tasks 6 & 7 (EWMA and Multi-frequency)

## Objective
Implement optimization based on real (inflation-adjusted) returns, including finding the tangent portfolio that maximizes real Sharpe ratio based on our decomposition: nominal returns = inflation + real risk-free rate + risk premium.

## Motivation
- Investors care about purchasing power, not nominal returns
- Inflation impacts different assets differently
- Real return optimization can lead to very different portfolios
- Proper framework for long-term wealth preservation

## Mathematical Framework

### Return Decomposition
```
Nominal Return = Inflation + Real Risk-Free Rate + Risk Premium

Where:
- Inflation: From FRED (CPI, PCE, etc.)
- Real Risk-Free Rate: TIPS yields or nominal - expected inflation
- Risk Premium: Excess return for taking risk

Real Return = (1 + Nominal Return) / (1 + Inflation) - 1
           ≈ Nominal Return - Inflation (for small values)
```

### Real Tangent Portfolio
```
Objective: max (E[R_real] - R_f_real) / σ_real

Where:
- E[R_real]: Expected real returns
- R_f_real: Real risk-free rate
- σ_real: Standard deviation of real returns
```

## Implementation Tasks

### 1. Create Real Return Calculator
**File**: `/src/analysis/real_returns.py` (NEW)

```python
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Union
from datetime import datetime

from ..data.fred_data import FREDDataFetcher

class RealReturnCalculator:
    """Calculate and analyze real returns."""
    
    def __init__(
        self,
        inflation_series: Optional[pd.Series] = None,
        inflation_type: str = 'CPI'  # CPI, PCE, GDPDEF
    ):
        """
        Initialize with inflation data.
        
        Args:
            inflation_series: Pre-loaded inflation data
            inflation_type: Type of inflation measure
        """
        self.inflation = inflation_series
        self.inflation_type = inflation_type
        
        if self.inflation is None:
            self.fred = FREDDataFetcher()
            
    def load_inflation_data(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'monthly'
    ) -> pd.Series:
        """Load inflation data from FRED."""
        series_map = {
            'CPI': 'CPIAUCSL',      # CPI-U All Items SA
            'PCE': 'PCEPI',         # PCE Price Index
            'GDPDEF': 'GDPDEF',     # GDP Deflator
            'CORE_CPI': 'CPILFESL', # Core CPI
            'CORE_PCE': 'PCEPILFE'  # Core PCE
        }
        
        series_id = series_map.get(self.inflation_type, 'CPIAUCSL')
        
        # Fetch inflation index
        inflation_index = self.fred.fetch_series(
            series_id, start_date, end_date
        )
        
        # Convert to inflation rate
        if frequency == 'monthly':
            self.inflation = inflation_index.pct_change()
        elif frequency == 'annual':
            self.inflation = inflation_index.pct_change(12)
        else:  # daily
            # Interpolate monthly to daily
            daily_index = pd.date_range(start_date, end_date, freq='D')
            self.inflation = inflation_index.reindex(daily_index).interpolate()
            self.inflation = self.inflation.pct_change()
            
        return self.inflation
    
    def nominal_to_real(
        self,
        nominal_returns: Union[pd.Series, pd.DataFrame],
        method: str = 'exact'
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Convert nominal returns to real returns.
        
        Args:
            nominal_returns: Nominal return series or dataframe
            method: 'exact' or 'approximate'
        """
        if self.inflation is None:
            raise ValueError("Inflation data not loaded")
            
        # Align inflation with returns
        aligned_inflation = self.inflation.reindex(nominal_returns.index).fillna(method='ffill')
        
        if method == 'exact':
            # Real = (1 + Nominal) / (1 + Inflation) - 1
            if isinstance(nominal_returns, pd.DataFrame):
                real_returns = pd.DataFrame(index=nominal_returns.index, columns=nominal_returns.columns)
                for col in nominal_returns.columns:
                    real_returns[col] = (1 + nominal_returns[col]) / (1 + aligned_inflation) - 1
            else:
                real_returns = (1 + nominal_returns) / (1 + aligned_inflation) - 1
        else:
            # Approximate: Real ≈ Nominal - Inflation
            real_returns = nominal_returns.subtract(aligned_inflation, axis=0)
            
        return real_returns
    
    def decompose_returns(
        self,
        nominal_returns: pd.Series,
        risk_free_nominal: pd.Series,
        method: str = 'additive'
    ) -> Dict[str, pd.Series]:
        """
        Decompose returns into components.
        
        Returns dict with:
        - inflation: Inflation component
        - real_rf: Real risk-free rate
        - risk_premium: Risk premium
        - real_total: Total real return
        """
        # Align all series
        idx = nominal_returns.index
        inflation = self.inflation.reindex(idx).fillna(method='ffill')
        rf_nominal = risk_free_nominal.reindex(idx).fillna(method='ffill')
        
        if method == 'additive':
            # Simple decomposition
            real_rf = rf_nominal - inflation
            risk_premium = nominal_returns - rf_nominal
            real_total = nominal_returns - inflation
        else:
            # Geometric decomposition
            real_rf = (1 + rf_nominal) / (1 + inflation) - 1
            risk_premium = (1 + nominal_returns) / (1 + rf_nominal) - 1
            real_total = (1 + nominal_returns) / (1 + inflation) - 1
        
        return {
            'inflation': inflation,
            'real_rf': real_rf,
            'risk_premium': risk_premium,
            'real_total': real_total,
            'nominal_total': nominal_returns
        }
    
    def calculate_real_statistics(
        self,
        nominal_returns: pd.DataFrame,
        risk_free_nominal: pd.Series
    ) -> Dict:
        """Calculate real return statistics for portfolio optimization."""
        # Convert to real returns
        real_returns = self.nominal_to_real(nominal_returns)
        real_rf = self.nominal_to_real(risk_free_nominal).mean()
        
        # Calculate statistics
        stats = {
            'expected_real_returns': real_returns.mean(),
            'real_covariance': real_returns.cov(),
            'real_risk_free_rate': real_rf,
            'inflation_rate': self.inflation.mean(),
            'real_vs_nominal_correlation': {}
        }
        
        # Calculate correlation between real and nominal for each asset
        for col in nominal_returns.columns:
            corr = nominal_returns[col].corr(real_returns[col])
            stats['real_vs_nominal_correlation'][col] = corr
            
        return stats
```

### 2. Create Real Return Optimizer
**File**: `/src/optimization/real_return_optimizer.py` (NEW)

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cvxpy as cp

from .engine import OptimizationResult
from ..analysis.real_returns import RealReturnCalculator

class RealReturnOptimizer:
    """Optimization based on real returns."""
    
    def __init__(self, real_return_calculator: RealReturnCalculator):
        self.rr_calc = real_return_calculator
        
    def find_real_tangent_portfolio(
        self,
        expected_real_returns: pd.Series,
        real_covariance: pd.DataFrame,
        real_risk_free_rate: float,
        constraints: Optional[Dict] = None,
        leverage: float = 1.0
    ) -> OptimizationResult:
        """
        Find tangent portfolio maximizing real Sharpe ratio.
        
        Args:
            expected_real_returns: Expected real returns for each asset
            real_covariance: Covariance matrix of real returns
            real_risk_free_rate: Real risk-free rate
            constraints: Weight constraints
            leverage: Maximum leverage allowed
        """
        n_assets = len(expected_real_returns)
        
        # Setup optimization variables
        weights = cp.Variable(n_assets)
        
        # Expected portfolio real return
        expected_return = expected_real_returns.values @ weights
        
        # Portfolio variance (real returns)
        portfolio_variance = cp.quad_form(weights, real_covariance.values)
        portfolio_std = cp.sqrt(portfolio_variance)
        
        # Objective: Maximize real Sharpe ratio
        # max (E[R] - Rf) / σ
        # Equivalent to: max (E[R] - Rf) subject to σ = 1
        excess_return = expected_return - real_risk_free_rate
        
        # Constraints
        constraints_list = []
        
        # Normalize to unit volatility for numerical stability
        constraints_list.append(portfolio_std == 1.0)
        
        # Weight constraints
        if constraints:
            if 'long_only' in constraints and constraints['long_only']:
                constraints_list.append(weights >= 0)
            if 'weight_bounds' in constraints:
                for i, (lower, upper) in enumerate(constraints['weight_bounds']):
                    constraints_list.append(weights[i] >= lower)
                    constraints_list.append(weights[i] <= upper)
        
        # Leverage constraint
        constraints_list.append(cp.sum(cp.abs(weights)) <= leverage)
        
        # Solve
        objective = cp.Maximize(excess_return)
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status != 'optimal':
            raise ValueError(f"Optimization failed: {problem.status}")
        
        # Rescale weights to sum to 1
        final_weights = weights.value / np.sum(np.abs(weights.value))
        
        # Calculate real return metrics
        real_return = expected_real_returns.values @ final_weights
        real_volatility = np.sqrt(final_weights @ real_covariance.values @ final_weights)
        real_sharpe = (real_return - real_risk_free_rate) / real_volatility
        
        return OptimizationResult(
            weights=pd.Series(final_weights, index=expected_real_returns.index),
            expected_return=real_return,
            volatility=real_volatility,
            sharpe_ratio=real_sharpe,
            metadata={
                'optimization_type': 'real_tangent_portfolio',
                'real_risk_free_rate': real_risk_free_rate,
                'is_real_returns': True
            }
        )
    
    def compare_real_vs_nominal(
        self,
        nominal_returns: pd.DataFrame,
        risk_free_nominal: pd.Series,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Compare optimization results using real vs nominal returns.
        
        Returns comparison metrics and both portfolio weights.
        """
        # Get real return statistics
        real_stats = self.rr_calc.calculate_real_statistics(
            nominal_returns, risk_free_nominal
        )
        
        # Optimize with real returns
        real_result = self.find_real_tangent_portfolio(
            real_stats['expected_real_returns'],
            real_stats['real_covariance'],
            real_stats['real_risk_free_rate'],
            constraints
        )
        
        # Optimize with nominal returns (standard approach)
        from .methods import MeanVarianceOptimizer
        mv_opt = MeanVarianceOptimizer()
        
        nominal_result = mv_opt.optimize(
            expected_returns=nominal_returns.mean(),
            covariance_matrix=nominal_returns.cov(),
            risk_free_rate=risk_free_nominal.mean(),
            objective='max_sharpe',
            constraints=constraints
        )
        
        # Compare results
        comparison = {
            'real_optimized': real_result,
            'nominal_optimized': nominal_result,
            'weight_difference': real_result.weights - nominal_result.weights,
            'metrics': {
                'real_sharpe_real_opt': real_result.sharpe_ratio,
                'real_sharpe_nominal_opt': self._calculate_real_sharpe(
                    nominal_result.weights, real_stats
                ),
                'nominal_sharpe_real_opt': self._calculate_nominal_sharpe(
                    real_result.weights, nominal_returns, risk_free_nominal.mean()
                ),
                'nominal_sharpe_nominal_opt': nominal_result.sharpe_ratio
            }
        }
        
        return comparison
    
    def _calculate_real_sharpe(
        self,
        weights: pd.Series,
        real_stats: Dict
    ) -> float:
        """Calculate real Sharpe ratio for given weights."""
        real_return = real_stats['expected_real_returns'] @ weights
        real_vol = np.sqrt(weights @ real_stats['real_covariance'] @ weights)
        return (real_return - real_stats['real_risk_free_rate']) / real_vol
    
    def _calculate_nominal_sharpe(
        self,
        weights: pd.Series,
        nominal_returns: pd.DataFrame,
        risk_free_rate: float
    ) -> float:
        """Calculate nominal Sharpe ratio for given weights."""
        nom_return = nominal_returns.mean() @ weights
        nom_vol = np.sqrt(weights @ nominal_returns.cov() @ weights)
        return (nom_return - risk_free_rate) / nom_vol
```

### 3. Integration with Main Optimization Engine
**File**: `/src/optimization/engine.py` (UPDATE)

Add real return optimization method:
```python
def optimize(
    self,
    symbols: List[str],
    expected_returns: Optional[pd.Series] = None,
    covariance_matrix: Optional[pd.DataFrame] = None,
    # ... existing parameters ...
    use_real_returns: bool = False,  # NEW
    inflation_type: str = 'CPI'      # NEW
) -> OptimizationResult:
    """
    Enhanced optimization with real return support.
    """
    if use_real_returns:
        # Load inflation data
        rr_calc = RealReturnCalculator(inflation_type=inflation_type)
        rr_calc.load_inflation_data(self.start_date, self.end_date)
        
        # Convert to real returns
        real_returns = rr_calc.nominal_to_real(self.returns[symbols])
        
        if expected_returns is None:
            expected_returns = real_returns.mean()
        if covariance_matrix is None:
            covariance_matrix = real_returns.cov()
            
        # Note in metadata
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {}
        kwargs['metadata']['use_real_returns'] = True
        kwargs['metadata']['inflation_type'] = inflation_type
```

### 4. Create Analysis Scripts
**File**: `/scripts/experiments/real_return_tangent.py` (NEW)

```python
#!/usr/bin/env python
"""Find and analyze real return tangent portfolio."""

def main():
    # Setup
    from src.data.total_returns import TotalReturnFetcher
    from src.data.fred_data import FREDDataFetcher
    from src.analysis.real_returns import RealReturnCalculator
    from src.optimization.real_return_optimizer import RealReturnOptimizer
    
    # Define universe
    symbols = ['SPY', 'AGG', 'GLD', 'VNQ', 'TIP', 'RSSB']
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Fetch data
    print("Fetching market data...")
    fetcher = TotalReturnFetcher()
    returns_data = {}
    for symbol in symbols:
        returns_data[symbol] = fetcher.fetch_total_returns(
            symbol, start_date, end_date
        )
    returns_df = pd.DataFrame(returns_data)
    
    # Get risk-free rate
    fred = FREDDataFetcher()
    rf_rate = fred.fetch_series('DGS3MO', start_date, end_date) / 100 / 252
    
    # Setup real return calculator
    print("Loading inflation data...")
    rr_calc = RealReturnCalculator(inflation_type='CPI')
    rr_calc.load_inflation_data(start_date, end_date, frequency='daily')
    
    # Create optimizer
    optimizer = RealReturnOptimizer(rr_calc)
    
    # Compare real vs nominal optimization
    print("\nComparing real vs nominal optimization...")
    comparison = optimizer.compare_real_vs_nominal(
        returns_df,
        rf_rate,
        constraints={'long_only': True}
    )
    
    # Display results
    print("\n=== PORTFOLIO WEIGHTS ===")
    print("\nNominal-Optimized Portfolio:")
    print(comparison['nominal_optimized'].weights.round(3))
    
    print("\nReal-Optimized Portfolio:")
    print(comparison['real_optimized'].weights.round(3))
    
    print("\nWeight Differences (Real - Nominal):")
    print(comparison['weight_difference'].round(3))
    
    print("\n=== PERFORMANCE METRICS ===")
    metrics = comparison['metrics']
    print(f"Real Sharpe (Real-Optimized): {metrics['real_sharpe_real_opt']:.3f}")
    print(f"Real Sharpe (Nominal-Optimized): {metrics['real_sharpe_nominal_opt']:.3f}")
    print(f"Improvement: {metrics['real_sharpe_real_opt'] - metrics['real_sharpe_nominal_opt']:.3f}")
    
    # Decompose returns
    print("\n=== RETURN DECOMPOSITION ===")
    decomp = rr_calc.decompose_returns(
        returns_df['SPY'],
        rf_rate
    )
    
    print(f"Average Inflation: {decomp['inflation'].mean() * 252:.2%}")
    print(f"Average Real RF: {decomp['real_rf'].mean() * 252:.2%}")
    print(f"Average Risk Premium: {decomp['risk_premium'].mean() * 252:.2%}")
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Weight comparison
    ax = axes[0, 0]
    x = np.arange(len(symbols))
    width = 0.35
    ax.bar(x - width/2, comparison['nominal_optimized'].weights, width, label='Nominal')
    ax.bar(x + width/2, comparison['real_optimized'].weights, width, label='Real')
    ax.set_xticks(x)
    ax.set_xticklabels(symbols)
    ax.set_title('Portfolio Weights: Nominal vs Real Optimization')
    ax.legend()
    
    # Return decomposition
    ax = axes[0, 1]
    components = ['Inflation', 'Real RF', 'Risk Premium']
    values = [
        decomp['inflation'].mean() * 252,
        decomp['real_rf'].mean() * 252,
        decomp['risk_premium'].mean() * 252
    ]
    ax.bar(components, values)
    ax.set_title('S&P 500 Return Decomposition (Annual)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Real vs Nominal returns scatter
    ax = axes[1, 0]
    real_rets = rr_calc.nominal_to_real(returns_df)
    ax.scatter(returns_df.mean() * 252, real_rets.mean() * 252)
    for i, symbol in enumerate(symbols):
        ax.annotate(symbol, (returns_df[symbol].mean() * 252, real_rets[symbol].mean() * 252))
    ax.set_xlabel('Nominal Return')
    ax.set_ylabel('Real Return')
    ax.set_title('Nominal vs Real Returns by Asset')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Efficient frontiers
    ax = axes[1, 1]
    # Plot both efficient frontiers
    # (Implementation details omitted for brevity)
    ax.set_title('Efficient Frontiers: Nominal vs Real')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    
    plt.tight_layout()
    plt.savefig('real_vs_nominal_optimization.png')
    print("\nVisualization saved to real_vs_nominal_optimization.png")

if __name__ == "__main__":
    main()
```

### 5. Add Tests
**File**: `/tests/test_real_returns.py` (NEW)

Key test cases:
1. Inflation data loading and alignment
2. Nominal to real conversion accuracy
3. Return decomposition adds up correctly
4. Real optimization produces valid weights
5. Real Sharpe calculation accuracy

Example test:
```python
def test_return_decomposition():
    """Test that return components add up correctly."""
    # Create test data
    nominal_returns = pd.Series([0.10, 0.08, 0.12], index=pd.date_range('2023-01-01', periods=3, freq='Y'))
    inflation = pd.Series([0.03, 0.025, 0.035], index=nominal_returns.index)
    rf_nominal = pd.Series([0.04, 0.045, 0.05], index=nominal_returns.index)
    
    # Setup calculator
    calc = RealReturnCalculator(inflation_series=inflation)
    
    # Decompose
    decomp = calc.decompose_returns(nominal_returns, rf_nominal, method='additive')
    
    # Check additive decomposition
    reconstructed = decomp['inflation'] + decomp['real_rf'] + decomp['risk_premium']
    np.testing.assert_array_almost_equal(reconstructed.values, nominal_returns.values)
    
    # Check real return calculation
    expected_real = nominal_returns - inflation
    np.testing.assert_array_almost_equal(decomp['real_total'].values, expected_real.values)
```

## Success Criteria
- [ ] Real return calculator handles multiple inflation measures
- [ ] Optimization finds different portfolios for real vs nominal
- [ ] Return decomposition is mathematically correct
- [ ] Integration with main optimization engine works
- [ ] Comparison script shows meaningful differences
- [ ] Tests validate all calculations

## Important Considerations
1. **Inflation Lag**: CPI data has ~2 week lag; consider using market-based measures
2. **Frequency Matching**: Ensure inflation and return data align properly
3. **International Assets**: May need different inflation measures
4. **TIPS**: Already inflation-adjusted; handle specially

## Next Steps
1. Run analysis on historical data to quantify impact
2. Document typical differences in allocation
3. Add inflation scenario analysis
4. Consider breakeven inflation from TIPS spreads

## Progress Updates
- [ ] Started: [timestamp]
- [ ] Real return calculator implemented: [status]
- [ ] Optimizer created: [status]
- [ ] Integration complete: [status]
- [ ] Analysis script working: [status]
- [ ] Tests passing: [status]
- [ ] Documentation updated: [status]
