# Portfolio Optimizer - Architecture & Technical Decisions

## Architecture Overview

### Core Design Principles
1. **Modular**: Each component has a single, well-defined responsibility
2. **Testable**: All business logic is unit-testable
3. **Extensible**: Easy to add new data sources, optimization methods, or UI features
4. **Production-Ready**: Include logging, error handling, and configuration management
5. **Type-Safe**: Use type hints throughout for better IDE support and fewer bugs

### System Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web UI        │     │   CLI Tools     │     │ Jupyter         │
│   (FastAPI)     │     │   (Click)       │     │ Notebooks       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Core Library         │
                    │  (portfolio-optimizer)  │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼────────┐    ┌──────────▼─────────┐   ┌────────▼────────┐
│ Portfolio      │    │ Data Layer         │   │ Optimization    │
│ Management     │    │ - Market Data      │   │ Engine          │
│ - Portfolio    │    │ - Fundamentals     │   │ - Constraints   │
│ - Position     │    │ - Benchmarks       │   │ - Objectives    │
│ - Analytics    │    │ - Import/Export    │   │ - Solvers       │
└────────────────┘    └────────────────────┘   └─────────────────┘
```

## Key Technical Decisions

### 1. Python Package Structure
**Decision**: Use a proper package structure with setup.py for development mode installation

**Rationale**: 
- Solves import issues cleanly
- Allows for future PyPI distribution
- Standard Python best practice

### 2. Data Layer Design
**Decision**: Abstraction layer over multiple data sources with caching

**Structure**:
```python
DataSource (Protocol)
├── YFinanceSource
├── AlphaVantageSource
├── PolygonSource
└── CachedDataSource (Decorator)
```

**Benefits**:
- Easy to add new data sources
- Fallback options when APIs fail
- Reduced API calls via caching

### 3. Portfolio Representation
**Decision**: Separate Position and Portfolio classes with clear responsibilities

**Key Classes**:
- `Position`: Single holding with quantity, cost basis, current value
- `Portfolio`: Collection of positions with analytics
- `Trade`: Representation of a buy/sell action
- `Allocation`: Target weights for optimization

### 4. Leverage Handling
**Decision**: Track both notional and economic exposure

**Implementation**:
- Positions have `leverage_factor` attribute
- Portfolio analytics show both dollar and volatility-weighted views
- Optimization can target either metric

### 5. Fund Exposure Model
**Decision**: Separate fund positions from their underlying exposures

**Rationale**:
- Internally leveraged funds (Return Stacked, PIMCO) contain multiple exposures
- Optimization must work on true exposures, not fund weights
- Need to validate that funds deliver their promised exposures

**Implementation**:
- `FundDefinition`: Maps funds to their exposure breakdown
- `ExposureCalculator`: Converts positions to true exposures
- Return replication to validate exposure assumptions

**Key Classes**:
- `Exposure`: Represents exposure to a specific asset class/factor
- `FundExposureMap`: Database of fund → exposures mapping
- `ReturnReplicator`: Validates exposure assumptions against actual returns

**Example**:
```python
# RSSB (Return Stacked Bonds) provides:
# - 100% US Equity exposure
# - 100% US Bond exposure
# - 200% total notional exposure
```

### 6. Configuration Management
**Decision**: Environment variables with .env file support

**Hierarchy**:
1. Environment variables (highest priority)
2. .env file
3. Default values in code

### 7. Testing Strategy
**Decision**: Pytest with fixtures and comprehensive mocking

**Structure**:
- Unit tests for each module
- Integration tests for data fetching
- End-to-end tests for optimization
- Property-based tests for edge cases

### 8. AI Coordination (PROJECT_CONTEXT)
**Decision**: Shared filesystem structure for AI coordination

**Benefits**:
- Simple and direct communication via task files
- No complex message queue needed
- Clear task specifications and progress tracking
- Git-trackable changes and history

## Data Flow

### Portfolio Import Flow
```
CSV/JSON File → Parser → Position Objects → Portfolio Object → Database/Cache
```

### Optimization Flow
```
Portfolio → Exposure Calculator → True Exposures + Constraints + Objectives → Optimizer → Allocation → Trade List
```

### Market Data Flow
```
API Request → Data Source → Normalizer → Cache → Application
                    ↓ (fallback)
              Alternative Source
```

## Security Considerations
- API keys in environment variables only
- No credentials in code or git
- Validate all user inputs
- Sanitize file uploads
- Rate limit API calls

## Performance Targets
- Portfolio analytics: < 100ms for 1000 positions
- Optimization: < 10s for standard problems
- Data fetch: < 2s with cache, < 10s without
- Web UI response: < 200ms for most operations

## Future Architecture Considerations
1. **Microservices**: Could split data, optimization, and web into separate services
2. **Message Queue**: For long-running optimizations
3. **Database**: PostgreSQL for production deployment
4. **Caching**: Redis for distributed cache
5. **Monitoring**: Prometheus + Grafana for production
