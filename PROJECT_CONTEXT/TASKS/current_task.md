# Current Task: Implement Leverage-Aware Optimization Engine

**Active Task**: Implement Leverage-Aware Optimization Engine
**Updated**: 2025-07-07 by Gemini

## Objective
Implement a leverage-aware optimization engine that uses the newly created risk premium covariance matrices to generate optimal portfolio allocations.

## Key Requirements
- Utilize the risk premium covariance matrices from the `RiskPremiumEstimator`.
- Support various optimization methods (e.g., Mean-Variance, Risk Parity).
- Handle leverage constraints, both at the individual position and portfolio level.
- Optimize on true exposures, not fund positions.
- Generate a trade list as output.

## Next Steps
1.  Review the detailed requirements in `PROJECT_CONTEXT/TASKS/backlog.md`.
2.  Create a new, detailed task specification file in the `TASKS/` directory.
3.  Begin implementation, starting with the core optimization engine.