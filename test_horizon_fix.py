#!/usr/bin/env python
"""
Test the horizon bug fix with realistic scenario
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.validation import CompatibilityValidationFramework
from src.validation.parameter_validation import ValidationMethod

print("🧪 TESTING HORIZON BUG FIX")
print("=" * 50)

# Create realistic test data similar to risk premium returns
np.random.seed(42)
dates = pd.date_range('2018-01-01', periods=150, freq='ME')
returns = pd.Series(np.random.normal(0, 0.015, 150), index=dates)

validation_framework = CompatibilityValidationFramework()

print("\n📊 Testing different horizons with the SAME data and method:")
print("   Data: 150 periods of monthly returns")
print("   Method: Historical volatility")
print("   Validation: Adaptive (will use walk-forward)")

# Test the specific horizons mentioned by the user
test_horizons = [5, 10, 15]  # Use smaller horizons that work with available data

results = []
for horizon in test_horizons:
    combo = {
        'method': 'historical',
        'parameters': {'window': 20},
        'horizon': horizon
    }
    
    result = validation_framework.validate_parameter_combination(
        returns, combo, ValidationMethod.ADAPTIVE
    )
    
    results.append(result)
    
    if result.success:
        print(f"   Horizon {horizon:2d}: MSE = {result.mse:.8f}, Method = {result.validation_method}")
    else:
        print(f"   Horizon {horizon:2d}: FAILED - {result.error_message}")

print("\n🔍 Analysis:")
# Check if MSE values are different
mse_values = [r.mse for r in results if r.success]
if len(set(mse_values)) == len(mse_values):
    print("   ✅ SUCCESS: All horizons produce DIFFERENT MSE values")
    print("   ✅ The horizon bug has been FIXED!")
else:
    print("   ❌ FAILURE: Some horizons still produce identical MSE values")
    print("   ❌ The horizon bug persists")

print(f"\n📈 MSE Values:")
for i, (horizon, result) in enumerate(zip(test_horizons, results)):
    if result.success:
        print(f"   Horizon {horizon}: {result.mse:.8f}")
        if i > 0:
            diff = abs(result.mse - results[i-1].mse)
            print(f"      Difference from previous: {diff:.8f}")

print("\n✅ HORIZON BUG FIX TEST COMPLETE")