# ADR-006: Production Optimization Failure Investigation

## Status
Proposed

## Context
On July 10, 2025, we discovered that the production parameter optimization is failing universally across all 29 parameter sets (14 volatility, 1 correlation, 14 expected returns). This is occurring despite:

1. **Sufficient Data**: Data retention was fixed earlier today, improving from 28 to 863+ data points
2. **Complete Framework**: All optimization components are implemented and tested
3. **Working Analysis**: The analysis framework successfully handles the failed optimization data

The failure pattern shows:
- All optimization scores returning NaN
- All validation metrics showing `optimization_failed` errors
- Universal failure suggests a systematic issue in the execution pipeline

## Decision
We need to implement a systematic debugging approach to identify and fix the root cause of the optimization failure. The approach will include:

1. **Detailed Execution Tracing**: Add comprehensive logging to trace the exact failure point
2. **Minimal Reproduction**: Create isolated test cases to reproduce the failure
3. **Common Failure Point Analysis**: Check data pipeline, validation setup, and scoring functions
4. **Diagnostic Tool Creation**: Build tools to systematically check pipeline health

## Consequences

### Positive
- Will unblock production deployment once fixed
- Creates diagnostic tools for future debugging
- Adds regression tests to prevent recurrence
- Improves system observability

### Negative
- Delays production readiness
- Requires additional development time
- May reveal deeper architectural issues

### Risks
- The issue might be in multiple places
- Fix might require significant refactoring
- Could be a fundamental issue with the approach

## Alternatives Considered

1. **Rollback Recent Changes**: Could undo the data alignment fix, but that fix is needed
2. **Use Simpler Optimization**: Could bypass component optimization, but loses the benefits
3. **Manual Parameter Selection**: Could hand-pick parameters, but not scalable

## Implementation Notes

Key areas to investigate:
1. **Data Loading**: Is data being fetched correctly for the validation period?
2. **Return Decomposition**: Is the decomposition producing valid risk premium data?
3. **Time Series Validation**: Is the TimeSeriesSplit configured correctly?
4. **Scoring Functions**: Are MSE/QLIKE calculations working properly?
5. **Date Handling**: Are estimation and validation dates properly aligned?

The debugging task has been created as the current priority task.

## Related Documents
- `/PROJECT_CONTEXT/TASKS/current_task.md` - Debug optimization failure task
- `/config/optimal_parameters.yaml` - Failed optimization results
- `/docs/analysis_framework_demo.md` - Shows framework handling failures correctly
