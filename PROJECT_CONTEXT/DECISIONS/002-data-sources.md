# ADR-002: Data Source Strategy

**Status**: Accepted  
**Date**: 2025-01-04  
**Author**: Scott McGuire  

## Context

Financial data is critical for portfolio optimization but comes with challenges:
- API costs and rate limits
- Data quality and reliability varies
- Different sources have different coverage
- Need both real-time and historical data
- Free vs paid trade-offs

## Decision

Implement a multi-tier data source strategy with automatic fallback:

### Primary Tier (Free)
1. **yfinance**: Default for most equity/ETF data
   - Pros: Free, reliable, good coverage
   - Cons: No official API, may break

2. **FRED**: For economic data
   - Pros: Official API, high quality
   - Cons: Limited to economic indicators

### Secondary Tier (Free with limits)
3. **Alpha Vantage**: Backup for price data
   - Pros: Official API, reliable
   - Cons: 5 calls/minute limit

4. **Polygon.io**: For comprehensive data
   - Pros: Good free tier, quality data
   - Cons: Rate limited

### Caching Layer
- Cache all API responses locally
- Default 24-hour TTL for daily data
- Shorter TTL for intraday data
- Persistent cache in `data/cache/`

### Implementation Pattern
```python
data = cache.get(symbol)
if not data:
    for source in [YFinance, AlphaVantage, Polygon]:
        try:
            data = source.fetch(symbol)
            cache.set(symbol, data)
            break
        except SourceUnavailable:
            continue
```

## Consequences

### Positive
- Resilient to single source failures
- Minimizes API costs
- Good performance with caching
- Extensible to add new sources

### Negative
- More complex than single source
- Cache invalidation complexity
- Potential data inconsistencies
- Need to handle different data formats

## Alternatives Considered

1. **Single Premium Source**: Use only one paid service
   - Rejected: Too expensive for personal project
   - Rejected: Single point of failure

2. **Web Scraping**: Scrape data from websites
   - Rejected: Fragile and potentially violates ToS
   - Rejected: Maintenance burden

3. **Manual Data Entry**: Download and store CSVs
   - Rejected: Not scalable
   - Rejected: Stale data issues

## Future Considerations

- Add support for fundamental data
- Consider paid tier for production use
- Implement data quality scoring
- Add real-time WebSocket feeds
- Consider local database for historical data

## References
- yfinance documentation
- "Building Reliable Financial Data Pipelines" - O'Reilly
- Alpha Vantage API documentation
