# Asset Universe Definition for Portfolio Optimizer

## Core Asset Classes

### Equity Beta Exposures
- **US Large Cap Equity**: Core exposure to US large cap stocks
- **US Small Cap Equity**: US small cap premium exposure  
- **International Developed Equity**: EAFE exposure
- **Emerging Markets Equity**: EM equity exposure
- **US Total Market**: Broad US equity exposure

### Fixed Income Exposures  
- **US Aggregate Bonds**: Core US bond exposure
- **US Treasuries**: Government bond exposure
- **International Bonds**: Global ex-US bonds
- **High Yield Bonds**: Credit risk premium
- **TIPS**: Inflation-protected bonds

### Alternative Strategies
- **Trend Following**: Managed futures/CTA strategies
- **Equity Long/Short**: Market neutral equity
- **Global Macro**: Discretionary macro strategies
- **Merger Arbitrage**: Event-driven strategies

### Factor Exposures
- **Equity Factors (Composite)**: Blend of value, momentum, quality, low-vol
- **Value Factor**: Traditional value exposure
- **Momentum Factor**: Cross-sectional momentum
- **Quality Factor**: Profitable, stable companies
- **Low Volatility Factor**: Defensive equity

### Real Assets
- **Commodities**: Broad commodity exposure
- **Gold**: Precious metals
- **Real Estate**: REITs and real property
- **Infrastructure**: Infrastructure equity

### Specialized Exposures
- **Convertible Bonds**: Hybrid equity/fixed income
- **Preferred Stocks**: Hybrid securities
- **Bank Loans**: Floating rate credit
- **Emerging Market Bonds**: EM fixed income

## Implementation Priorities

### Phase 1 (Essential)
1. US Large Cap Equity (SPY, IVV, VOO)
2. US Aggregate Bonds (AGG, BND)  
3. International Equity (EFA, IEFA)
4. Trend Following (DBMF, KMLM)
5. Commodities (DJP, DBC)

### Phase 2 (Important)
1. US Small Cap (IWM, IJR)
2. Emerging Markets (EEM, VWO)
3. Real Estate (VNQ, XLRE)
4. Equity Factors Composite
5. Gold (GLD, IAU)

### Phase 3 (Nice to Have)
1. High Yield Bonds (HYG, JNK)
2. TIPS (TIP, SCHP)
3. International Bonds (IAGG)
4. Other alternatives

## Data Source Mapping

### Direct ETF Mapping
Most exposures can be captured via liquid ETFs with good history

### Composite Strategies
Some strategies need weighted combinations:
- Equity Factors = 25% each of MTUM, VLUE, QUAL, USMV
- Real Assets = 50% DJP + 50% VNQ
- Balanced Risk = Equal weight government/corporate/inflation bonds

### Index Data Needs
For strategies without good ETF proxies:
- SG Trend Index for trend following validation
- HFRX indices for hedge fund strategies
- MSCI factor indices for factor validation

## Historical Data Requirements

### Minimum History
- 10 years for covariance estimation
- 5 years absolute minimum
- Daily data for accuracy

### Total Return Requirements  
- Must include dividends/distributions
- Adjust for splits and corporate actions
- Validate against known benchmarks

### Quality Checks
- No extended gaps in data
- Reasonable return magnitudes
- Consistent with asset class behavior
