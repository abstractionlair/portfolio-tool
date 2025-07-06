# 🚨 Critical Fix for Negative Real Returns in Notebooks

## The Problem

If you're seeing **all assets with negative real returns** (like -38% to -53%), this is caused by comparing monthly nominal returns against annualized inflation rates.

Example of what you're probably seeing:
```
us_large_equity: 15.8% nominal → -38.3% real return ❌
```

## The Root Cause

**WRONG CODE** (causing the issue):
```python
# ❌ This annualizes monthly inflation rates to ~4-5%
inflation_rates = fred_fetcher.calculate_inflation_rate(
    inflation_index, periods=1, annualize=True  # 🚨 WRONG for monthly returns!
)

# Then comparing:
# Monthly nominal return: ~1.5% 
# "Monthly" inflation: ~4.5% (actually annualized!)
# Result: Massive negative real returns
```

## The Solution

**CORRECT CODE** (fixes the issue):

### Option 1: Use the convenience method (Recommended)
```python
# ✅ Automatically matches frequencies
inflation_rates = fred_fetcher.get_inflation_rates_for_returns(
    start_date, end_date, return_frequency='monthly'
)
```

### Option 2: Explicitly disable annualization
```python
# ✅ Keep monthly inflation as monthly
inflation_rates = fred_fetcher.calculate_inflation_rate(
    inflation_index, periods=1, annualize=False  # FALSE for monthly returns
)
```

## Expected Results After Fix

With the correct calculation, you should see realistic results like:
```
us_large_equity: 17.9% nominal → 15.1% real return ✅
```

## Quick Test

Run this to verify your fix worked:
```python
# Should show small monthly values (~0.2%)
print(f"Monthly inflation rate: {inflation_rates.mean():.4%}")
print(f"Approximate annual: {inflation_rates.mean() * 12:.2%}")

# Real returns should be positive for most equity assets
print(f"US Large Equity real return: {real_returns_data['us_large_equity'].mean() * 12:.1%}")
```

## Files to Check

1. **Example Script**: `examples/correct_real_returns_calculation.py`
2. **Fixed Framework**: Uses `ReturnEstimationFramework.estimate_real_returns()` (already fixed)
3. **This Fix**: For direct FRED fetcher usage in notebooks

---

**Rule of Thumb**: Monthly returns need monthly inflation rates, not annualized ones!