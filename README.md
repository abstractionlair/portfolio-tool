# Portfolio Optimization Tool

## Vision
A comprehensive portfolio building and optimization tool that serves both as a practical personal investment management system and as a demonstration of software engineering capabilities for prospective employers.

## Goals

### Primary Goals
1. **Personal Use**: Create a functional tool for managing and optimizing my investment portfolio
2. **Professional Portfolio**: Develop a well-architected project that demonstrates software engineering skills

### Secondary Goals
- Replace/improve upon portfoliovisualizer.com functionality
- Enable portfolio management across multiple machines/locations
- Support sophisticated strategies including leverage and long-short positions

## High-Level Requirements

### Core Functionality
1. **Environment Setup**: Configurable development environment accessible from multiple machines
2. **Portfolio Data Access**: Import and analyze current portfolio holdings
3. **Historical Data**: Access historical portfolio performance data
4. **Investment Universe Definition**: Define and manage the set of available investment vehicles
5. **Market Data Integration**: Access current and historical data for all securities in the investment universe
6. **Benchmark Data**: Retrieve benchmark indices, returns, and compositions
7. **Portfolio Optimization**: Solve for optimal asset allocation given constraints
8. **Implementation Mapping**: Map theoretical allocations to available investment vehicles

### Investment Strategy Context
- **Starting Point**: Global Multi-Asset Market Portfolio (Doeswijk et al.)
- **Enhancements**: 
  - Alternative investments
  - Factor exposure
  - Trend following strategies
  - Multi-asset futures carry
- **Leverage Approach**: Target volatility similar to global equity portfolio using internally leveraged funds
  - Return Stacked ETFs
  - PIMCO StocksPLUS funds
  - PIMCO RAE Plus funds

### Technical Considerations
- **Methodological Challenges**: 
  - Handling leverage in allocation calculations
  - Long-short position representation
  - Choice between dollar-weighted vs. volatility-weighted allocations
- **Constraints**:
  - Investment minimums
  - Additional constraints to be determined

## Technology Preferences
- **Primary Language**: Python
- **Development Tools**: Compatible with Claude Code + web interface
- **Portability**: Easy setup on new machines OR remote accessibility
- **Data Sources**: Preference for free APIs, willing to pay for high-value services
- **UI Evolution**: Start simple, evolve to web interface with persistent storage
- **Current Brokerage**: Consolidating at Vanguard

## Success Criteria
1. Successfully manages personal portfolio with sophisticated optimization
2. Demonstrates clean architecture and professional coding practices
3. Includes comprehensive documentation and tests
4. Showcases relevant skills for quantitative development roles
5. Provides insights beyond simple commercial tools

## Project Phases (TBD)
- Phase 1: Environment setup and basic data access
- Phase 2: Portfolio import and analysis
- Phase 3: Optimization engine
- Phase 4: Web interface
- Phase 5: Advanced features and polish

## Open Questions
- Specific portfolio constraints beyond minimums?
- Performance requirements?
- Specific employer audiences to target?
- Integration with brokerage APIs vs. manual import/export?

