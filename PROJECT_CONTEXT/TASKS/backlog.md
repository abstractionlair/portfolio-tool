# Portfolio Optimizer - Task Backlog

## 🎯 NEW APPROACH: Build on Existing Code!
**IMPORTANT**: All tasks should EXTEND existing systems, not create new ones. The codebase is already sophisticated - we need surgical modifications, not rewrites.

## Immediate Priority - Exposure-Based Optimization Foundation

### 1. ✅ Complete Analysis with All 16 Exposures Including Alternatives (COMPLETED)
- **Status**: COMPLETE ✅ (2025-07-13)
- **Approach**: Extended parameter optimization to include full exposure universe
- **Key**: Fixed data availability issues, added alternative strategies, generated optimal parameters

### 2. Add Fund Exposure Mappings to Existing System
- **Status**: READY (Task spec: add_fund_exposure_mappings.md)
- **Approach**: Extend existing exposures.py and portfolio.py
- **Key**: Use existing ReturnReplicator for validation

### 3. Enable Exposure-Level Optimization in Existing Engine
- **Status**: READY (Task spec: enable_exposure_optimization.md)
- **Approach**: Create adapter for existing OptimizationEngine
- **Key**: The engine already works - just feed it exposure data

## High Priority - Complete the Vision

### 4. Expected Return Estimation at Exposure Level
- **Approach**: Extend existing return estimation
- Add to src/optimization/estimators.py
- Use existing data providers
- Build on current shrinkage methods

### 5. Fund Selection Using Existing Optimizer
- **Approach**: Second optimization problem using existing engine
- Minimize tracking error to target exposures
- Reuse existing constraint system
- Add to portfolio_optimizer.py

### 6. Leverage Cost Modeling
- **Approach**: Extend existing Position class
- Add funding cost to position.py
- Modify return calculations in existing calculators
- Update optimization to account for costs

## Medium Priority - Production Features

### 7. Web Interface Development
- **Approach**: API wrapper around existing functionality
- FastAPI endpoints calling existing optimizers
- Don't duplicate business logic
- Thin presentation layer only

### 8. Backtesting Using Existing Analytics
- **Approach**: Extend portfolio/analytics.py
- Use existing performance calculations
- Add time series of optimizations
- Reuse existing metrics

### 9. Risk Monitoring System
- **Approach**: Build on ExposureRiskEstimator
- Add monitoring methods to existing classes
- Use existing risk calculations
- Add alerting layer

### 10. Data Quality Layer
- **Approach**: Extend existing data providers
- Add quality scores to data interfaces
- Use existing validation in return_replicator
- Build on current error handling

## Key Principles Going Forward

### ✅ DO:
- Read existing code first
- Extend classes rather than create new ones
- Use existing calculations and algorithms
- Build adapters when needed
- Keep changes minimal and surgical

### ❌ DON'T:
- Create parallel systems
- Reimplement existing functionality
- Start from scratch
- Ignore what's already built
- Make sweeping changes

## Examples of Good Approach

1. **Parameter Optimization**: Add `target_horizon` parameter to existing methods
2. **Fund Exposures**: Add methods to existing Portfolio class
3. **Exposure Optimization**: Wrap existing OptimizationEngine
4. **Return Calculation**: Use existing ReturnCalculator, don't create new

## Technical Debt Reduction

### Consolidation Opportunities
- Merge duplicate return calculations
- Unify configuration formats
- Standardize data interfaces
- Reduce redundant validation

### Documentation Needs
- Document existing system architecture
- Create dependency diagrams
- Show data flow through system
- Explain design decisions

## Completed Tasks (Archive)
- See `/PROJECT_CONTEXT/TASKS/completed/` for completed work
- Data layer implementation ✅
- Portfolio optimization integration ✅
- Raw and transformed data providers ✅
- Component parameter optimization ✅
- Risk premium estimation ✅

## Remember
The existing codebase is sophisticated and well-designed. Our job is to:
1. Understand what exists
2. Identify minimal changes needed
3. Extend rather than replace
4. Test that existing functionality still works

Each task should start with "What already exists that does something similar?"
