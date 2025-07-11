#!/usr/bin/env python3
"""
Run production parameter optimization on real exposure universe.

This script discovers optimal parameters for all portfolio components:
- Volatility estimation (per exposure)
- Correlation estimation (single set)  
- Expected return estimation (per exposure)

Based on pre-flight checks, we'll optimize 14 implementable exposures.
"""

import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import json
import time

# Setup paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
    
    # Configuration based on pre-flight check results
    OPTIMIZATION_END_DATE = datetime.now()
    OPTIMIZATION_START_DATE = OPTIMIZATION_END_DATE - pd.DateOffset(years=5)
    VALIDATION_SPLITS = 3  # Reduced from 5 due to limited data
    OUTPUT_DIR = Path("optimization_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    logger.info(f"Optimization period: {OPTIMIZATION_START_DATE.date()} to {OPTIMIZATION_END_DATE.date()}")
    logger.info(f"Validation splits: {VALIDATION_SPLITS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Initialize components
    logger.info("\nInitializing components...")
    try:
        universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
        decomposer = ReturnDecomposer()
        risk_estimator = RiskPremiumEstimator(universe, decomposer)
        logger.info("✓ Components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False
    
    # Define implementable exposures based on pre-flight check
    # Excluding: trend_following, cash_rate (implementation issues)
    implementable_exposures = [
        'us_large_equity',
        'us_small_equity', 
        'intl_developed_large_equity',
        'intl_developed_small_equity',
        'emerging_equity',
        'factor_style_equity',
        'factor_style_other',
        'short_ust',
        'broad_ust', 
        'dynamic_global_bonds',
        'real_estate',
        'commodities',
        'gold',
        'tips'
    ]
    
    logger.info(f"\nOptimizing {len(implementable_exposures)} exposures:")
    for i, exp_id in enumerate(implementable_exposures, 1):
        logger.info(f"  {i:2d}. {exp_id}")
    
    # Verify data access for each exposure before starting
    logger.info("\nVerifying data access...")
    verified_exposures = []
    for exp_id in implementable_exposures:
        try:
            # Quick test load with longer lookback to get sufficient data
            decomp = risk_estimator.load_and_decompose_exposure_returns(
                exposure_id=exp_id,
                estimation_date=OPTIMIZATION_END_DATE,
                lookback_days=1260,  # 5 years (gives ~28 monthly points after decomposition)
                frequency='monthly'
            )
            if len(decomp) >= 15:  # Minimum 15 months (adjusted for decomposition data loss)
                verified_exposures.append(exp_id)
                logger.info(f"  ✓ {exp_id}: {len(decomp)} data points")
            else:
                logger.warning(f"  ✗ {exp_id}: Insufficient data ({len(decomp)} points)")
        except Exception as e:
            logger.error(f"  ✗ {exp_id}: Data access failed - {e}")
    
    if len(verified_exposures) == 0:
        logger.error("No exposures have sufficient data - aborting optimization")
        return False
    
    logger.info(f"\nProceeding with {len(verified_exposures)} verified exposures")
    
    # Initialize orchestrator
    logger.info("\nInitializing component optimization orchestrator...")
    try:
        orchestrator = ComponentOptimizationOrchestrator(
            risk_estimator=risk_estimator,
            parallel=True,  # Use parallel processing
            max_workers=2   # Conservative to avoid memory issues
        )
        logger.info("✓ Orchestrator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        return False
    
    # Run optimization
    logger.info("\nStarting component optimization...")
    logger.info("This will optimize parameters separately for:")
    logger.info("  - Volatility (forecast accuracy)")
    logger.info("  - Correlation (stability)")  
    logger.info("  - Expected Returns (directional accuracy)")
    
    start_time = time.time()
    
    try:
        optimal_params = orchestrator.optimize_all_components(
            exposure_ids=verified_exposures,
            start_date=OPTIMIZATION_START_DATE,
            end_date=OPTIMIZATION_END_DATE,
            n_splits=VALIDATION_SPLITS
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n✓ Optimization completed in {elapsed_time/60:.1f} minutes")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = OUTPUT_DIR / f"optimal_parameters_{timestamp}.yaml"
        orchestrator.save_optimal_parameters(optimal_params, str(output_file))
        logger.info(f"✓ Saved optimal parameters to: {output_file}")
        
        # Also save to production location
        prod_file = "config/optimal_parameters.yaml"
        orchestrator.save_optimal_parameters(optimal_params, prod_file)
        logger.info(f"✓ Saved to production location: {prod_file}")
        
        # Generate summary report
        generate_optimization_report(optimal_params, verified_exposures, OUTPUT_DIR, timestamp)
        
        # Print parameter summary
        print_parameter_summary(optimal_params, verified_exposures)
        
        return True
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return False
    
    finally:
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)


def generate_optimization_report(optimal_params, exposure_ids, output_dir, timestamp):
    """Generate summary report of optimization results."""
    
    logger.info("\nGenerating optimization report...")
    
    report = {
        'timestamp': timestamp,
        'optimization_date': optimal_params.optimization_date.isoformat(),
        'validation_period': [
            optimal_params.validation_period[0].isoformat(),
            optimal_params.validation_period[1].isoformat()
        ],
        'exposures_optimized': len(exposure_ids),
        'exposure_list': exposure_ids,
        'summary': {},
        'parameter_details': {}
    }
    
    # Volatility summary
    vol_methods = {}
    vol_scores = []
    for exp_id, params in optimal_params.volatility_params.items():
        method = params.method
        vol_methods[method] = vol_methods.get(method, 0) + 1
        vol_scores.append(params.score)
    
    report['summary']['volatility_methods'] = vol_methods
    report['summary']['volatility_scores'] = {
        'mean': float(np.mean(vol_scores)) if vol_scores else 0.0,
        'std': float(np.std(vol_scores)) if vol_scores else 0.0,
        'min': float(np.min(vol_scores)) if vol_scores else 0.0,
        'max': float(np.max(vol_scores)) if vol_scores else 0.0
    }
    
    # Correlation summary
    corr_params = optimal_params.correlation_params
    report['summary']['correlation'] = {
        'method': corr_params.method,
        'lookback_days': corr_params.lookback_days,
        'frequency': corr_params.frequency,
        'score': corr_params.score
    }
    
    # Expected return summary
    ret_methods = {}
    ret_scores = []
    for exp_id, params in optimal_params.expected_return_params.items():
        method = params.method
        ret_methods[method] = ret_methods.get(method, 0) + 1
        ret_scores.append(params.score)
    
    report['summary']['return_methods'] = ret_methods
    report['summary']['return_scores'] = {
        'mean': float(np.mean(ret_scores)) if ret_scores else 0.0,
        'std': float(np.std(ret_scores)) if ret_scores else 0.0,
        'min': float(np.min(ret_scores)) if ret_scores else 0.0,
        'max': float(np.max(ret_scores)) if ret_scores else 0.0
    }
    
    # Detailed parameter breakdown
    report['parameter_details']['volatility'] = {}
    for exp_id, params in optimal_params.volatility_params.items():
        report['parameter_details']['volatility'][exp_id] = {
            'method': params.method,
            'lookback_days': params.lookback_days,
            'frequency': params.frequency,
            'score': params.score,
            'parameters': params.parameters
        }
    
    report['parameter_details']['expected_returns'] = {}
    for exp_id, params in optimal_params.expected_return_params.items():
        report['parameter_details']['expected_returns'][exp_id] = {
            'method': params.method,
            'lookback_days': params.lookback_days,
            'frequency': params.frequency,
            'score': params.score,
            'parameters': params.parameters
        }
    
    # Save report
    report_file = output_dir / f"optimization_summary_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Generated summary report: {report_file}")


def print_parameter_summary(optimal_params, exposure_ids):
    """Print detailed summary of optimal parameters."""
    
    print("\n" + "=" * 80)
    print("OPTIMAL PARAMETER SUMMARY")
    print("=" * 80)
    
    # Volatility parameters
    print("\nVolatility Parameters (per exposure):")
    print("-" * 80)
    print(f"{'Exposure':<25} {'Method':<12} {'Lookback':<10} {'Frequency':<10} {'Score':<12}")
    print("-" * 80)
    
    for exp_id in exposure_ids:
        if exp_id in optimal_params.volatility_params:
            params = optimal_params.volatility_params[exp_id]
            print(f"{exp_id:<25} {params.method:<12} "
                  f"{params.lookback_days:<10} {params.frequency:<10} {params.score:<12.6f}")
        else:
            print(f"{exp_id:<25} {'MISSING':<12} {'':>10} {'':>10} {'':>12}")
    
    # Correlation parameters
    print("\nCorrelation Parameters (shared across all exposures):")
    print("-" * 80)
    cp = optimal_params.correlation_params
    print(f"Method: {cp.method}")
    print(f"Lookback: {cp.lookback_days} days")
    print(f"Frequency: {cp.frequency}")
    print(f"Score: {cp.score:.6f}")
    
    # Expected return parameters
    print("\nExpected Return Parameters (per exposure):")
    print("-" * 80)
    print(f"{'Exposure':<25} {'Method':<12} {'Lookback':<10} {'Frequency':<10} {'Score':<12}")
    print("-" * 80)
    
    for exp_id in exposure_ids:
        if exp_id in optimal_params.expected_return_params:
            params = optimal_params.expected_return_params[exp_id]
            print(f"{exp_id:<25} {params.method:<12} "
                  f"{params.lookback_days:<10} {params.frequency:<10} {params.score:<12.6f}")
        else:
            print(f"{exp_id:<25} {'MISSING':<12} {'':>10} {'':>10} {'':>12}")
    
    # Summary statistics
    print("\nOptimization Summary:")
    print("-" * 80)
    
    # Method distribution
    vol_methods = {}
    for params in optimal_params.volatility_params.values():
        vol_methods[params.method] = vol_methods.get(params.method, 0) + 1
    
    ret_methods = {}
    for params in optimal_params.expected_return_params.values():
        ret_methods[params.method] = ret_methods.get(params.method, 0) + 1
    
    print(f"Volatility methods: {dict(vol_methods)}")
    print(f"Correlation method: {cp.method}")
    print(f"Return methods: {dict(ret_methods)}")
    
    print(f"\nExposures optimized: {len(exposure_ids)}")
    print(f"Optimization completed: {optimal_params.optimization_date}")


if __name__ == "__main__":
    import numpy as np
    success = main()
    sys.exit(0 if success else 1)