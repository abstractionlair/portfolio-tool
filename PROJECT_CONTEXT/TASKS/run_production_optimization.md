# Task: Run Production Parameter Optimization on Real Data

**Status**: TODO  
**Priority**: CRITICAL - Need optimal parameters for production use  
**Estimated Time**: 4-6 hours (mostly compute time)  
**Dependencies**: Component optimization framework (complete)

## Overview

The component optimization framework is complete and tested. Now we need to run it on real data to generate the actual optimal parameters for production use. This will discover the best parameters for:
- Volatility estimation (for each exposure)
- Correlation estimation (single set)
- Expected return estimation (for each exposure)

## Phase 1: Pre-flight Checks (30 minutes)

### 1.1 Verify Data Access
```python
# Check that all exposures have data
from src.data.exposure_universe import ExposureUniverse
from src.optimization.risk_premium_estimator import RiskPremiumEstimator
from src.data.return_decomposition import ReturnDecomposer

# Load universe
universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')

# Test data access for each exposure
for exposure_id in universe.exposures.keys():
    print(f"Testing {exposure_id}...")
    try:
        # Try to load recent data
        estimator = RiskPremiumEstimator(universe, ReturnDecomposer())
        decomp = estimator.load_and_decompose_exposure_returns(
            exposure_id=exposure_id,
            estimation_date=datetime.now(),
            lookback_days=252,
            frequency='daily'
        )
        print(f"  ✓ {len(decomp)} data points available")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
```

### 1.2 Resource Planning
- Estimate compute time based on:
  - Number of exposures (14-16)
  - Parameter grid size (50-200 combinations per component)
  - Cross-validation splits (5 recommended)
  - Expected time: 1-2 minutes per exposure for volatility
  
- Memory requirements:
  - Estimate 2-4GB for full optimization
  - Consider reducing parallel workers if memory constrained

## Phase 2: Run Component Optimization (3-4 hours)

### 2.1 Create Optimization Script
Location: `/scripts/run_production_optimization.py`

```python
#!/usr/bin/env python3
"""
Run production parameter optimization on real exposure universe.
This script discovers optimal parameters for all components.
"""

import logging
import sys
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import json
import time

# Setup paths
sys.path.append('..')

from src.optimization.component_optimizers import ComponentOptimizationOrchestrator
from src.optimization.risk_premium_estimator import RiskPremiumEstimator
from src.data.exposure_universe import ExposureUniverse
from src.data.return_decomposition import ReturnDecomposer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run production parameter optimization."""
    
    logger.info("=" * 80)
    logger.info("PRODUCTION PARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    
    # Configuration
    OPTIMIZATION_END_DATE = datetime.now()
    OPTIMIZATION_START_DATE = OPTIMIZATION_END_DATE - pd.DateOffset(years=5)
    VALIDATION_SPLITS = 5
    OUTPUT_DIR = Path("optimization_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
    decomposer = ReturnDecomposer()
    risk_estimator = RiskPremiumEstimator(universe, decomposer)
    
    # Get all available exposures
    all_exposure_ids = list(universe.exposures.keys())
    logger.info(f"Found {len(all_exposure_ids)} exposures in universe")
    
    # Filter to implementable exposures
    implementable_exposures = []
    for exp_id in all_exposure_ids:
        try:
            # Test data availability
            decomp = risk_estimator.load_and_decompose_exposure_returns(
                exposure_id=exp_id,
                estimation_date=OPTIMIZATION_END_DATE,
                lookback_days=1260,  # 5 years
                frequency='daily'
            )
            if len(decomp) >= 252:  # At least 1 year of data
                implementable_exposures.append(exp_id)
                logger.info(f"  ✓ {exp_id}: {len(decomp)} data points")
            else:
                logger.warning(f"  ✗ {exp_id}: Insufficient data ({len(decomp)} points)")
        except Exception as e:
            logger.error(f"  ✗ {exp_id}: Failed to load data - {e}")
    
    logger.info(f"\nOptimizing {len(implementable_exposures)} exposures with sufficient data")
    
    # Initialize orchestrator
    orchestrator = ComponentOptimizationOrchestrator(
        risk_estimator=risk_estimator,
        parallel=True,  # Use parallel processing
        max_workers=4   # Adjust based on your CPU
    )
    
    # Run optimization
    logger.info("\nStarting component optimization...")
    logger.info(f"Period: {OPTIMIZATION_START_DATE.date()} to {OPTIMIZATION_END_DATE.date()}")
    logger.info(f"Validation splits: {VALIDATION_SPLITS}")
    
    start_time = time.time()
    
    try:
        optimal_params = orchestrator.optimize_all_components(
            exposure_ids=implementable_exposures,
            start_date=OPTIMIZATION_START_DATE,
            end_date=OPTIMIZATION_END_DATE,
            n_splits=VALIDATION_SPLITS
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nOptimization completed in {elapsed_time/60:.1f} minutes")
        
        # Save results
        output_file = OUTPUT_DIR / f"optimal_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        orchestrator.save_optimal_parameters(optimal_params, str(output_file))
        logger.info(f"Saved optimal parameters to: {output_file}")
        
        # Also save to production location
        prod_file = "config/optimal_parameters.yaml"
        orchestrator.save_optimal_parameters(optimal_params, prod_file)
        logger.info(f"Saved to production location: {prod_file}")
        
        # Generate summary report
        generate_optimization_report(optimal_params, implementable_exposures, OUTPUT_DIR)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)


def generate_optimization_report(optimal_params, exposure_ids, output_dir):
    """Generate summary report of optimization results."""
    
    report = {
        'optimization_date': optimal_params.optimization_date.isoformat(),
        'validation_period': [
            optimal_params.validation_period[0].isoformat(),
            optimal_params.validation_period[1].isoformat()
        ],
        'exposures_optimized': len(exposure_ids),
        'summary': {}
    }
    
    # Volatility summary
    vol_methods = {}
    for exp_id, params in optimal_params.volatility_params.items():
        method = params.method
        vol_methods[method] = vol_methods.get(method, 0) + 1
    report['summary']['volatility_methods'] = vol_methods
    
    # Correlation summary
    report['summary']['correlation'] = {
        'method': optimal_params.correlation_params.method,
        'lookback_days': optimal_params.correlation_params.lookback_days,
        'score': optimal_params.correlation_params.score
    }
    
    # Expected return summary
    ret_methods = {}
    for exp_id, params in optimal_params.expected_return_params.items():
        method = params.method
        ret_methods[method] = ret_methods.get(method, 0) + 1
    report['summary']['return_methods'] = ret_methods
    
    # Save report
    report_file = output_dir / "optimization_summary.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Generated summary report: {report_file}")


if __name__ == "__main__":
    main()
```

### 2.2 Run with Progress Monitoring
```bash
# Run the optimization
cd /Users/scottmcguire/portfolio-tool
python scripts/run_production_optimization.py

# Monitor progress in another terminal
tail -f optimization_run.log
```

### 2.3 Checkpoint Strategy
- The orchestrator should save intermediate results
- If optimization fails, can resume from checkpoint
- Consider running in screen/tmux for long runs

## Phase 3: Validation and Quality Checks (1 hour)

### 3.1 Validate Results
```python
# Load and validate the generated parameters
from src.optimization.component_optimizers import ComponentOptimizationOrchestrator

orchestrator = ComponentOptimizationOrchestrator(risk_estimator)
params = orchestrator.load_optimal_parameters("config/optimal_parameters.yaml")

# Validation checks
validation_report = orchestrator.validate_parameters(params, exposure_ids)

# Check for reasonable values
for exp_id, vol_params in params.volatility_params.items():
    assert vol_params.lookback_days >= 20  # Minimum lookback
    assert vol_params.score < 0  # Negative score (minimizing error)
    assert vol_params.method in ['historical', 'ewma', 'garch']
```

### 3.2 Create Backup
```bash
# Backup the production parameters
cp config/optimal_parameters.yaml config/optimal_parameters_backup_$(date +%Y%m%d).yaml
```

## Phase 4: Test Production Interface (30 minutes)

### 4.1 Verify OptimizedRiskEstimator Works
```python
from src.optimization import OptimizedRiskEstimator

# Test that it loads the new parameters
estimator = OptimizedRiskEstimator()

# Get sample estimates
test_exposures = ['us_large_equity', 'bonds', 'commodities']
cov_matrix = estimator.get_covariance_matrix(test_exposures, datetime.now())

print("Covariance Matrix:")
print(cov_matrix)

# Verify reasonable values
assert np.all(np.diag(cov_matrix) > 0)  # Positive variances
assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric
```

## Expected Outputs

1. **Primary Output**: `config/optimal_parameters.yaml`
   - Production-ready parameter file
   - Used automatically by OptimizedRiskEstimator

2. **Backup/Analysis Files**:
   - `optimization_results/optimal_parameters_YYYYMMDD_HHMMSS.yaml`
   - `optimization_results/optimization_summary.json`
   - `optimization_run.log`

3. **Summary Report** showing:
   - Methods selected for each component
   - Score improvements achieved
   - Any exposures that failed optimization

## Success Criteria

- [ ] Optimization completes without errors
- [ ] At least 80% of exposures successfully optimized
- [ ] Parameters show meaningful differences between components
- [ ] Correlation parameters use longer lookback than volatility
- [ ] Production interface works with new parameters
- [ ] Results are reproducible

## Troubleshooting

**If optimization fails for specific exposures:**
- Check data availability for that exposure
- Reduce parameter grid size
- Use fallback parameters

**If memory issues occur:**
- Reduce parallel workers
- Optimize in batches
- Use smaller parameter grids

**If optimization takes too long:**
- Reduce validation splits (minimum 3)
- Use constrained parameter grids
- Run overnight or on cloud instance

## Notes

- First run will take longest (3-5 hours typical)
- Subsequent runs can use previous results as starting point
- Consider running monthly/quarterly to update parameters
- Save all outputs for analysis notebook
