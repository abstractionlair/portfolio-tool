# Quick Task Summary for Implementation

## What We're Building
A data infrastructure system that:
1. Loads exposure definitions from `config/exposure_universe.yaml`
2. Fetches total returns (including dividends) for each exposure
3. Adds inflation data to calculate real returns
4. Estimates expected returns and covariances for optimization

## Start Here
1. Read `/config/exposure_universe.yaml` - this defines what we want to track
2. Create `src/data/exposure_universe.py` - classes to load and manage the universe
3. Enhance `src/data/market_data.py` - ensure we get total returns with dividends
4. Add `src/data/inflation_data.py` - FRED integration for CPI data
5. Create `src/estimation/` - return and risk estimation module

## Data Source Testing Priority
1. **Test these first** (likely to work): SPY, TLT, VNQ, GLD, all major ETFs
2. **Test these second** (uncertain): QSPIX, QMNIX, ABYIX, AHLIX, AQMNX, ASFYX
3. **Known to need FRED**: Risk-free rate (DGS3MO), CPI data (CPIAUCSL)
4. **Build fallbacks**: For any mutual funds that fail in yfinance

## Key Requirements
- Must get TOTAL returns (use yfinance Adj Close, not Close)
- Support weighted composites (e.g., equity factor = 25% each of 4 ETFs)
- Handle different start dates gracefully
- Convert everything to real returns using CPI data
- Calculate both arithmetic and geometric returns
- Implement hierarchical structure (5 categories, 16 exposures)
- TIPS classified as Real Assets, not Nominal Fixed Income
- Cash/Risk-Free Rate for leverage cost modeling

## Simple Test
```python
# This should work when done:
universe = ExposureUniverse.from_yaml("config/exposure_universe.yaml")
data = universe.fetch_returns(start="2015-01-01", end="2023-12-31")
real_returns = universe.to_real_returns(data)
params = universe.estimate_parameters(real_returns)
# params.expected_returns -> array of expected real returns
# params.covariance -> covariance matrix
```

## Priority Order
1. Basic universe loader
2. Total return fetching 
3. Composite calculations
4. Inflation integration
5. Parameter estimation
