# Enhanced Equity Return Decomposition Examples

This directory contains comprehensive examples demonstrating the enhanced equity return decomposition functionality.

## 📁 Files Overview

### Core Examples
- **`simple_decomposition_example.py`** - Simple standalone example with realistic data
- **`enhanced_equity_decomposition_demo.py`** - Comprehensive demo with multiple stocks
- **`equity_decomposition_examples.py`** - Modular functions for different analysis types
- **`decomposition_plotting.py`** - Visualization utilities for charts and plots

### Jupyter Notebook
- **`../notebooks/enhanced_equity_decomposition_showcase.ipynb`** - Interactive demonstration

## 🚀 Quick Start

### Run Simple Example
```bash
python examples/simple_decomposition_example.py
```

### Run Comprehensive Demo
```bash
python examples/enhanced_equity_decomposition_demo.py
```

### Use in Jupyter Notebook
```bash
jupyter notebook notebooks/enhanced_equity_decomposition_showcase.ipynb
```

## 📊 Example Output

The examples demonstrate realistic values:

```
📊 AAPL Return Decomposition Summary
==================================================

Annualized Return Components:
------------------------------
Total Nominal Return     : 46.24% ± 19.95%
Dividend Yield           :  0.00% ± 0.00%
P/E Change               : 39.91% ± 19.86%
Nominal Earnings Growth  :  5.82% ± 4.08%
Real Earnings Growth     :  2.82% ± 4.08%
Real Earnings Excess     :  0.78% ± 4.08%
Real Risk Premium        : 40.69% ± 19.95%

Economic Context:
------------------------------
Inflation Rate           :  3.00%
Nominal Risk-Free Rate   :  4.98%
Real Risk-Free Rate      :  2.03%

Quality Metrics:
------------------------------
Identity Error (max)     : 0.0015 ✅ Good
Decomposition Error (max): 0.0015 ✅ Good
```

## 🎯 Key Features Demonstrated

### 1. Economic Framework
- Separates returns into dividend yield, P/E change, and real earnings excess
- Properly adjusts earnings growth for inflation and real risk-free rate
- Mathematical identity: `r_real_risk_premium = r_dividend + r_pe_change + r_real_earnings_excess`

### 2. Analysis Types
- **Single Stock Analysis**: Deep dive into one stock's decomposition
- **Stock Comparison**: Compare multiple stocks side-by-side
- **Sector Analysis**: Analyze patterns across different sectors
- **Time Series Analysis**: Statistical properties of components
- **Frequency Comparison**: Daily vs monthly analysis

### 3. Visualization
- Time series plots of components
- Distribution analysis
- Correlation heatmaps
- Economic context charts
- Multi-stock comparison bars

### 4. Quality Validation
- Identity checks (components sum to total return)
- Decomposition error bounds
- Statistical consistency verification

## 🔧 Customization

### Custom Analysis
```python
from examples.equity_decomposition_examples import analyze_single_stock, setup_provider

# Setup
provider = setup_provider()

# Analyze custom stock
result = analyze_single_stock(
    provider=provider,
    ticker='YOUR_TICKER',
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31),
    frequency='daily'
)
```

### Custom Visualization
```python
from examples.decomposition_plotting import plot_decomposition_components

# Create custom plots
fig = plot_decomposition_components(result, 'YOUR_TICKER')
```

## 🧪 Realistic Data

The examples use realistic economic data:
- **Inflation**: ~3% annually (converted to daily rates)
- **Risk-Free Rate**: ~5% annually (converted to daily rates)
- **Real Risk-Free Rate**: ~2% annually (risk-free minus inflation)

This produces economically meaningful results instead of the unrealistic values that would come from using raw daily rates.

## 📈 Use Cases

### Research Applications
- Understand return drivers across different stocks
- Analyze sector-specific return patterns
- Study time series properties of components
- Compare real vs nominal return contributions

### Portfolio Management
- Component-based risk assessment
- Return attribution analysis
- Factor exposure analysis
- Performance decomposition

### Academic Analysis
- Empirical finance research
- Asset pricing studies
- Return predictability analysis
- Risk premium decomposition

## 🔍 Technical Details

### Data Requirements
- Price data (provided by data layer)
- Earnings data (simulated in examples)
- Economic data (inflation, risk-free rates)

### Mathematical Framework
```
r_nominal = r_dividend + r_pe_change + r_nominal_earnings

r_real_risk_premium = r_nominal - r_inflation - r_real_rf
                    = r_dividend + r_pe_change + r_real_earnings_excess

where:
r_real_earnings_excess = r_nominal_earnings - r_inflation - r_real_rf
```

### Quality Checks
- Identity validation: Components sum to total return
- Decomposition error bounds: Real vs nominal consistency
- Statistical validity: Reasonable ranges and distributions

## 📝 Next Steps

1. **Modify Parameters**: Adjust earnings growth rates, economic assumptions
2. **Add New Tickers**: Extend analysis to other stocks or ETFs
3. **Custom Time Periods**: Analyze different market regimes
4. **Enhanced Visualization**: Create custom charts and dashboards
5. **Integration**: Use in portfolio optimization and risk management

## 🐛 Troubleshooting

### Common Issues
1. **Timezone Errors**: Ensure consistent timezone handling
2. **Data Availability**: Check network connection for live data
3. **Matplotlib Issues**: Ensure proper display backend for plots

### Performance Tips
- Use realistic date ranges (avoid very long periods)
- Consider monthly frequency for long-term analysis
- Cache results for repeated analysis

---

**The enhanced equity return decomposition provides a powerful foundation for understanding the economic drivers of equity returns and building sophisticated investment analysis tools.**