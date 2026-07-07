# Research Summary: Perplexity Finance API

## Quick Take
Perplexity's new finance features and Sonar API are **not a data API** - they're an AI search/synthesis service. Not suitable for replacing yfinance/FRED but could enhance user experience with narrative explanations.

## What Perplexity Offers
- **Perplexity Finance**: Web interface with real-time quotes sourced from Financial Modeling Prep
- **Sonar API**: AI-powered search that returns natural language answers with citations
- **SEC Integration**: Can analyze regulatory filings (but via natural language, not structured data)

## What It Doesn't Offer
- ❌ Direct access to time series data
- ❌ Structured price/return endpoints  
- ❌ Bulk data downloads
- ❌ Programmatic data API

## Potential Use in Portfolio Optimizer
✅ Generate portfolio explanations for users
✅ Research obscure funds we can't find data for
✅ Provide market context and commentary
✅ Create narrative summaries of optimization results

❌ Replace yfinance for price data
❌ Systematic data collection
❌ Backtesting data source
❌ Real-time portfolio analytics

## Cost Considerations
- $5 per 1,000 searches (plus token costs)
- Would be expensive for repeated/systematic use
- Better suited for on-demand, high-value queries

## Recommendation
**Don't pursue this now**. It's solving a different problem than our data infrastructure needs. Consider it as a future enhancement for user-facing features after core system is complete.

If we need better financial data access, look at:
1. **Financial Modeling Prep** - Direct API to Perplexity's data source
2. **Alpha Vantage** - Good mutual fund coverage
3. **IEX Cloud** - Reliable with good docs

Full analysis saved in: `/PROJECT_CONTEXT/RESEARCH/perplexity_finance_api_analysis.md`