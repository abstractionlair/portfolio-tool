# Portfolio Optimizer - Task Backlog

## Immediate Priority - Exposure-Based Optimization Foundation

### 1. ✅ Implement Global Forecast Horizon (CURRENT TASK)
- **Status**: ACTIVE - See current_task.md
- Ensure consistent forecast horizon across all exposures
- Modify parameter optimization to target specific horizons
- Create horizon-aware configuration structure

### 2. Create Fund Exposure Mapping System
- **Status**: READY (Task spec created)
- Build fund_exposures.yaml database
- Implement FundExposureManager class
- Create exposure replication validator
- Enable two-stage optimization workflow

### 3. Build Exposure-Level Optimization Engine
- **Status**: READY (Task spec created)
- Create optimization engine working with exposures
- Implement multiple objectives (Sharpe, risk parity, etc.)
- Add exposure-specific constraints
- Integrate with risk model

## High Priority - Complete the Vision

### 4. Expected Return Estimation at Exposure Level
- Implement risk premium framework for exposures
- Add shrinkage/Bayesian methods
- Support for views (Black-Litterman)
- Cross-sectional return models

### 5. Advanced Fund Selection Optimizer
- Minimize tracking error to target exposures
- Handle transaction costs
- Consider tax implications
- Multi-account optimization

### 6. Leverage Cost Modeling
- Model funding costs for leveraged exposures
- Time-varying spread over cash rate
- Fund-specific leverage costs
- Impact on optimization

## Medium Priority - Production Features

### 7. Web Interface Development
- **Note**: Now depends on exposure-based optimization
- RESTful API for exposure optimization
- Interactive exposure allocation UI
- Fund selection interface
- Real-time rebalancing suggestions

### 8. Backtesting Framework
- Test exposure-based strategies
- Compare with traditional optimization
- Out-of-sample validation
- Performance attribution

### 9. Risk Monitoring System
- Real-time exposure tracking
- Risk limit monitoring
- Drawdown alerts
- Correlation regime detection

### 10. Data Quality Layer
- Validate exposure mappings
- Detect data anomalies
- Handle corporate actions
- Quality scoring system

## Lower Priority - Advanced Features

### 11. Machine Learning Integration
- ML-based return forecasting
- Regime detection models
- Dynamic exposure adjustment
- Anomaly detection

### 12. Alternative Data Integration
- Sentiment indicators
- Macro nowcasting
- Factor timing signals
- Cross-asset momentum

### 13. Reporting System
- Exposure-based attribution
- Risk factor analysis
- Custom report builder
- Automated distribution

### 14. Multi-Strategy Integration
- Combine systematic strategies
- Dynamic strategy allocation
- Cross-strategy risk management
- Unified performance tracking

## Technical Infrastructure

### Code Quality & Testing
- Increase test coverage to 90%+
- Performance benchmarks
- API documentation
- Continuous integration

### Deployment & Scaling
- Docker containerization
- Cloud deployment (AWS/GCP)
- Database integration
- Monitoring and alerting

### Documentation
- Mathematical framework docs
- API reference
- Video tutorials
- Case studies

## Research Topics

### Exposure Modeling
- Time-varying exposures
- Non-linear exposure relationships
- Regime-dependent mappings
- Higher-order effects

### Risk Modeling
- Fat-tail risk measures
- Liquidity-adjusted risk
- Concentration penalties
- Stress testing

### Optimization Methods
- Hierarchical risk parity
- Nested optimization
- Online/adaptive optimization
- Reinforcement learning

## Notes
- Exposure-based optimization is the core differentiator
- Each task should maintain mathematical rigor
- Focus on practical implementation
- Regular validation against real portfolios
- Keep the long (10+ year) history requirement in mind

## Completed Tasks (Archive)
- See `/PROJECT_CONTEXT/TASKS/completed/` for completed work
- Data layer implementation ✅
- Portfolio optimization integration ✅
- Raw and transformed data providers ✅
