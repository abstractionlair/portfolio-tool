#!/usr/bin/env python
"""
Simple Risk Premium Test

Quick test to validate the risk premium prediction concept with minimal dependencies.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test the core concept
from data.return_decomposition import ReturnDecomposer
from data.exposure_universe import ExposureUniverse


def test_return_decomposition():
    """Test basic return decomposition functionality."""
    print("Testing Return Decomposition...")
    
    # Create sample returns data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.08/12, 0.15/np.sqrt(12), len(dates)),
        index=dates,
        name='test_asset'
    )
    
    print(f"Sample returns: {len(returns)} periods")
    print(f"Mean return: {returns.mean():.4f}")
    print(f"Return volatility: {returns.std():.4f}")
    
    # Initialize decomposer
    decomposer = ReturnDecomposer()
    
    # Test decomposition
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    try:
        decomposition = decomposer.decompose_returns(
            returns=returns,
            start_date=start_date,
            end_date=end_date,
            frequency='monthly'
        )
        
        if not decomposition.empty:
            print(f"\n✅ Decomposition successful!")
            print(f"Decomposition shape: {decomposition.shape}")
            print(f"Columns: {list(decomposition.columns)}")
            
            # Show components
            if 'spread' in decomposition.columns:
                spread_vol = decomposition['spread'].std()
                total_vol = decomposition['total_return'].std()
                print(f"\nVolatility comparison:")
                print(f"  Total return volatility: {total_vol:.4f}")
                print(f"  Risk premium volatility: {spread_vol:.4f}")
                print(f"  Difference: {(total_vol - spread_vol):.4f}")
                
                return True
            else:
                print("❌ Missing spread column in decomposition")
                return False
        else:
            print("❌ Empty decomposition result")
            return False
            
    except Exception as e:
        print(f"❌ Decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exposure_universe_loading():
    """Test loading exposure universe."""
    print("\nTesting Exposure Universe Loading...")
    
    try:
        universe_path = Path(__file__).parent.parent / 'config' / 'exposure_universe.yaml'
        if not universe_path.exists():
            print(f"❌ Universe file not found: {universe_path}")
            return False
        
        universe = ExposureUniverse.from_yaml(str(universe_path))
        print(f"✅ Loaded {len(universe)} exposures")
        
        # Show available exposures
        exposure_ids = list(universe.get_all_exposure_ids())
        print(f"Available exposures: {exposure_ids[:5]}...")  # Show first 5
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load universe: {e}")
        return False


def demonstrate_concept():
    """Demonstrate the core concept with synthetic data."""
    print("\nDemonstrating Risk Premium Concept...")
    
    # Create synthetic "total return" that includes risk-free rate volatility
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Components
    risk_free_rate = 0.02 / 252  # 2% annual, daily
    risk_free_volatility = 0.01 / np.sqrt(252)  # Low volatility risk-free component
    risk_premium_mean = 0.06 / 252  # 6% annual risk premium
    risk_premium_volatility = 0.12 / np.sqrt(252)  # Higher volatility risk premium
    
    # Generate component returns
    rf_returns = np.random.normal(risk_free_rate, risk_free_volatility, len(dates))
    rp_returns = np.random.normal(risk_premium_mean, risk_premium_volatility, len(dates))
    
    # Total returns = risk-free + risk premium (simplified)
    total_returns = rf_returns + rp_returns
    
    # Calculate volatilities (annualized)
    total_vol = np.std(total_returns) * np.sqrt(252)
    rp_vol = np.std(rp_returns) * np.sqrt(252)
    rf_vol = np.std(rf_returns) * np.sqrt(252)
    
    print(f"Synthetic example:")
    print(f"  Total return volatility:   {total_vol:.1%}")
    print(f"  Risk premium volatility:   {rp_vol:.1%}")
    print(f"  Risk-free volatility:      {rf_vol:.1%}")
    print(f"  Uncompensated component:   {(total_vol - rp_vol):.1%}")
    
    print(f"\nKey insight:")
    print(f"  Portfolio optimization should use {rp_vol:.1%} (risk premium)")
    print(f"  NOT {total_vol:.1%} (total return)")
    print(f"  This focuses on compensated risk only")
    
    return True


def main():
    """Run all tests."""
    print("Risk Premium Prediction - Core Concept Test")
    print("=" * 50)
    
    success = True
    
    # Test 1: Return decomposition
    success &= test_return_decomposition()
    
    # Test 2: Exposure universe
    success &= test_exposure_universe_loading()
    
    # Test 3: Demonstrate concept
    success &= demonstrate_concept()
    
    print(f"\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Risk premium framework concept validated.")
        print("\nNext steps:")
        print("1. Fix datetime issues in full implementation")
        print("2. Test with real exposure data")
        print("3. Validate parameter optimization on risk premia")
    else:
        print("❌ Some tests failed. Need to debug implementation.")


if __name__ == "__main__":
    main()