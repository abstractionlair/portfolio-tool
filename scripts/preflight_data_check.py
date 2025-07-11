#!/usr/bin/env python3
"""
Pre-flight data check for production parameter optimization.

Tests data availability for all exposures in the universe to ensure
we can run optimization without errors.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
import pandas as pd
from src.data.exposure_universe import ExposureUniverse
from src.optimization.risk_premium_estimator import RiskPremiumEstimator
from src.data.return_decomposition import ReturnDecomposer


def main():
    """Run pre-flight checks on all exposures."""
    
    print("=" * 80)
    print("PRE-FLIGHT DATA CHECK")
    print("=" * 80)
    
    # Initialize components
    print("Initializing components...")
    try:
        universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
        decomposer = ReturnDecomposer()
        risk_estimator = RiskPremiumEstimator(universe, decomposer)
        print("✓ Components initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize components: {e}")
        return False
    
    # Get all exposure IDs
    exposure_ids = list(universe.exposures.keys())
    print(f"\nFound {len(exposure_ids)} exposures in universe:")
    for exp_id in exposure_ids:
        print(f"  - {exp_id}")
    
    # Test data access for each exposure
    print(f"\nTesting data access for each exposure...")
    print("-" * 80)
    
    test_date = datetime.now()
    implementable_exposures = []
    problematic_exposures = []
    
    for i, exposure_id in enumerate(exposure_ids, 1):
        print(f"[{i:2d}/{len(exposure_ids)}] Testing {exposure_id}...")
        
        try:
            # Test basic data loading (1 year) - use monthly like working examples
            decomp_1y = risk_estimator.load_and_decompose_exposure_returns(
                exposure_id=exposure_id,
                estimation_date=test_date,
                lookback_days=365,
                frequency='monthly'
            )
            
            # Test 5-year data loading (needed for optimization) - use monthly like working examples
            decomp_5y = risk_estimator.load_and_decompose_exposure_returns(
                exposure_id=exposure_id,
                estimation_date=test_date,
                lookback_days=1260,
                frequency='monthly'
            )
            
            # Check data quality (adjusted for monthly frequency)
            if len(decomp_5y) >= 48:  # ~4 years minimum at monthly
                status = "✓ GOOD"
                implementable_exposures.append(exposure_id)
            elif len(decomp_5y) >= 24:  # ~2 years minimum at monthly
                status = "⚠ LIMITED"
                implementable_exposures.append(exposure_id)
            else:
                status = "✗ INSUFFICIENT"
                problematic_exposures.append(exposure_id)
            
            print(f"    {status} - 1Y: {len(decomp_1y)} points, 5Y: {len(decomp_5y)} points")
            
            # Check for NaN values
            nan_count = decomp_5y['spread'].isna().sum()
            if nan_count > 0:
                print(f"    ⚠ Warning: {nan_count} NaN values in risk premium data")
            
            # Check for recent data
            last_date = decomp_5y.index[-1] if len(decomp_5y) > 0 else None
            if last_date:
                days_since_last = (test_date.date() - last_date.date()).days
                if days_since_last > 7:
                    print(f"    ⚠ Warning: Last data point is {days_since_last} days old")
                    
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            problematic_exposures.append(exposure_id)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"Total exposures: {len(exposure_ids)}")
    print(f"Implementable: {len(implementable_exposures)} ({len(implementable_exposures)/len(exposure_ids)*100:.1f}%)")
    print(f"Problematic: {len(problematic_exposures)} ({len(problematic_exposures)/len(exposure_ids)*100:.1f}%)")
    
    if implementable_exposures:
        print(f"\nImplementable exposures ({len(implementable_exposures)}):")
        for exp_id in implementable_exposures:
            print(f"  ✓ {exp_id}")
    
    if problematic_exposures:
        print(f"\nProblematic exposures ({len(problematic_exposures)}):")
        for exp_id in problematic_exposures:
            print(f"  ✗ {exp_id}")
    
    # Resource estimation
    print(f"\nResource estimation:")
    print(f"  Exposures to optimize: {len(implementable_exposures)}")
    print(f"  Estimated optimization time: {len(implementable_exposures) * 2:.0f}-{len(implementable_exposures) * 4:.0f} minutes")
    print(f"  Memory requirement: ~2-4GB")
    
    # Recommendations
    success_rate = len(implementable_exposures) / len(exposure_ids)
    if success_rate >= 0.8:
        print(f"\n✓ READY TO PROCEED - {success_rate:.1%} success rate")
        return True
    elif success_rate >= 0.6:
        print(f"\n⚠ CAUTION - {success_rate:.1%} success rate, consider investigating problems")
        return True
    else:
        print(f"\n✗ NOT READY - {success_rate:.1%} success rate, fix data issues first")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)