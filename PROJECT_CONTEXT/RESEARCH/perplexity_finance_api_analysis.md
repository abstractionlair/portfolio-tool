# Perplexity Finance API Analysis for Portfolio Optimizer
*Created: January 7, 2025*
*Author: Desktop Claude*

## Executive Summary

Perplexity has recently launched financial capabilities and the Sonar API, which could potentially enhance our portfolio optimizer's data infrastructure. However, based on current research, **the Perplexity API does not appear to provide direct structured financial data access**. Instead, it offers AI-powered search and synthesis capabilities that could complement but not replace our current data sources.

## Key Findings

### 1. Perplexity Finance Features

**What it is:**
- A finance-focused interface at perplexity.ai/finance
- Provides real-time stock quotes, market data, and financial analysis
- Recently integrated SEC/EDGAR data for deeper financial research

**Data Sources:**
- Financial Modeling Prep (FMP) - for market data
- Unusual Whales - for options data  
- Quartr - for earnings transcripts
- FinChat.io - for earnings data
- SEC/EDGAR - for regulatory filings

**Current Capabilities:**
- Real-time stock prices and charts
- Company earnings reports
- Industry peer comparisons
- Basic financial metrics
- SEC filing analysis (new feature)

### 2. Sonar API Overview

**What it provides:**
- AI-powered search with real-time web access
- Natural language Q&A with citations
- NOT a traditional data API with structured financial data

**Pricing Tiers:**
1. **Sonar (Base)**
   - $5 per 1,000 searches
   - $1 per 1M input tokens (~750K words)
   - $1 per 1M output tokens
   - Lightweight, fast responses

2. **Sonar Pro**
   - $5 per 1,000 searches
   - $3 per 1M input tokens
   - $15 per 1M output tokens
   - More detailed answers, 2x citations
   - 200K token context window

**Key Features:**
- Real-time web search integration
- Source citations for all responses
- Domain filtering capabilities
- JSON mode for structured outputs
- OpenAI-compatible API format

### 3. Relevance to Portfolio Optimizer

#### Potential Use Cases

1. **Market Commentary & Analysis**
   - Generate narrative explanations of market conditions
   - Summarize earnings reports for holdings
   - Provide context for portfolio decisions
   - Create investor-friendly explanations

2. **Research Augmentation**
   - Query latest information about funds/ETFs
   - Find details about new financial products
   - Research alternative investment strategies
   - Validate exposure assumptions

3. **Data Gap Filling**
   - When yfinance lacks mutual fund data, could query for basic info
   - Find fund fact sheets or prospectuses
   - Locate hard-to-find alternative fund details

#### NOT Suitable For

1. **Primary Data Source**
   - No structured price/return data
   - No programmatic access to time series
   - Cannot replace yfinance/FRED

2. **High-Frequency Data Needs**
   - Cost prohibitive for repeated queries
   - Not designed for systematic data collection
   - No bulk data endpoints

### 4. Implementation Considerations

#### Pros
- Easy integration (OpenAI-compatible)
- High accuracy with citations
- Real-time information
- Could enhance user experience

#### Cons
- Usage-based pricing (could get expensive)
- Not a data API - it's a search/synthesis API
- Requires parsing natural language responses
- No guaranteed data structure/format

### 5. Example Integration Pattern

```python
from openai import OpenAI

class PerplexityFinanceHelper:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
    
    def get_fund_overview(self, ticker: str) -> dict:
        """Get narrative overview of a fund using Perplexity."""
        response = self.client.chat.completions.create(
            model="sonar",
            messages=[{
                "role": "user",
                "content": f"Provide key details about {ticker} ETF including: "
                           f"expense ratio, AUM, top holdings, and investment strategy. "
                           f"Return as JSON with these fields."
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def explain_portfolio_allocation(self, weights: dict) -> str:
        """Generate human-readable explanation of portfolio allocation."""
        weight_str = ", ".join([f"{k}: {v:.1%}" for k, v in weights.items()])
        
        response = self.client.chat.completions.create(
            model="sonar-pro",
            messages=[{
                "role": "user", 
                "content": f"Explain this portfolio allocation to an investor: {weight_str}. "
                           f"Include market context and rationale for the weightings."
            }]
        )
        return response.choices[0].message.content
```

### 6. Recommended Approach

**Phase 1: Experiment with Free Features**
- Use Perplexity Finance web interface to research funds
- Test quality of financial information
- Identify specific use cases where it adds value

**Phase 2: Limited API Integration**
- Add optional PerplexityFinanceHelper class
- Use for user-facing explanations and summaries
- Cache responses to minimize API costs
- Focus on high-value queries only

**Phase 3: Evaluate ROI**
- Track API usage and costs
- Measure user engagement with AI features
- Decide if deeper integration warranted

### 7. Alternative Considerations

Given that Perplexity doesn't provide structured financial data, consider these alternatives for our data gaps:

1. **Financial Modeling Prep API** (FMP)
   - Direct access to the data Perplexity uses
   - Structured API with time series data
   - More suitable for systematic data needs

2. **Alpha Vantage**
   - Free tier available
   - Good mutual fund coverage
   - Proper API for programmatic access

3. **IEX Cloud**
   - Reliable data source
   - Good documentation
   - Reasonable pricing

## Conclusion

Perplexity's finance capabilities are impressive for **narrative intelligence** but not suitable as a **primary data source** for our portfolio optimizer. The Sonar API could add value for:

1. Generating user-friendly explanations
2. Providing market context
3. Researching obscure funds
4. Creating portfolio summaries

However, it should not replace our current data infrastructure (yfinance, FRED) but rather complement it for specific use cases where natural language synthesis adds value.

**Recommendation**: Focus on completing our core data infrastructure with traditional APIs first. Consider Perplexity integration as a "Phase 5" enhancement for user experience, not as a solution to our current data challenges.

## Next Steps

1. Continue with current data infrastructure plan
2. Implement rate series handling for risk-free rate
3. Add traditional data API fallbacks (FMP, Alpha Vantage)
4. Consider Perplexity integration only after core functionality complete
5. If pursuing, start with free web interface testing before API investment

## References

- Perplexity Finance: https://www.perplexity.ai/finance
- Sonar API Docs: https://docs.perplexity.ai/
- API Pricing: https://www.perplexity.ai/hub/faq/pplx-api
- SEC Integration Announcement: https://www.perplexity.ai/hub/blog/answers-for-every-investor