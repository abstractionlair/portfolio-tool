# ADR-001: Leverage Approach in Portfolio Optimization

**Status**: Accepted  
**Date**: 2025-01-04  
**Author**: Scott McGuire  

## Context

Traditional portfolio optimization assumes unleveraged positions where weights sum to 1.0. However, modern portfolio construction often includes:
- Leveraged ETFs (UPRO, TQQQ, etc.)
- Return Stacked ETFs that embed leverage
- Short positions
- Derivatives with embedded leverage

This creates challenges in portfolio optimization and risk measurement.

## Decision

We will implement a dual-view approach to portfolio positions:

1. **Notional View**: Actual dollars invested (what you pay)
2. **Economic View**: Effective market exposure (what you get)

### Implementation Details

Each position will track:
- `notional_value`: Actual dollars invested
- `leverage_factor`: Embedded leverage (e.g., 3.0 for UPRO)
- `economic_value`: notional_value × leverage_factor

Portfolio analytics will provide both views:
- Notional weights (sum to 1.0)
- Economic weights (may sum > 1.0)
- Volatility-adjusted weights

### Optimization Approach

The optimization engine will support multiple modes:
1. **Notional-constrained**: Traditional approach, notional weights sum to 1.0
2. **Volatility-targeted**: Target portfolio volatility regardless of leverage
3. **Economic-exposure**: Optimize based on economic exposure

## Consequences

### Positive
- Accurately represents leveraged portfolios
- Enables sophisticated strategies like Return Stacking
- Aligns with modern portfolio construction techniques
- Provides flexibility for different use cases

### Negative
- More complex than traditional approaches
- Requires user education on the dual views
- May confuse users familiar with traditional tools
- Requires careful handling in reporting

## Alternatives Considered

1. **Ignore Leverage**: Treat all positions as unleveraged
   - Rejected: Misrepresents risk and exposure

2. **Separate Leveraged Portfolio**: Have different portfolio types
   - Rejected: Too restrictive, many portfolios mix both

3. **Only Economic View**: Show only the leveraged exposure
   - Rejected: Hides actual capital requirements

## References
- "Return Stacking: Strategies for Overcoming a Low Return Environment" - ReSolve Asset Management
- "Leveraged ETFs: The Trojan Horse Has Arrived" - Cheng & Madhavan
- Discussion with Corey Hoffstein on Twitter regarding leverage implementation
