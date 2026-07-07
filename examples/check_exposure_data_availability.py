"""Check which exposures have data available."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datetime import datetime
from data.exposure_universe import ExposureUniverse

def check_data_availability():
    """Check data availability for all exposures."""
    universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
    
    print("Checking data availability for all exposures...")
    print("=" * 80)
    
    # Try to import data fetcher
    try:
        from data.total_returns import TotalReturnFetcher
        fetcher = TotalReturnFetcher()
        print("✅ Using TotalReturnFetcher for data availability check")
    except ImportError:
        print("⚠️  TotalReturnFetcher not available, checking exposure definitions only")
        fetcher = None
    
    results = {}
    for exp_id, exposure in universe.exposures.items():
        print(f"\n{exp_id}:")
        
        # Get implementations
        implementations = exposure.implementations
        if not implementations:
            print(f"  ✗ No implementations defined")
            results[exp_id] = {'available': False, 'reason': 'No implementations'}
            continue
            
        # Check primary implementation
        primary_impl = implementations[0]  # First implementation is primary
        
        if fetcher:
            # Try to fetch actual data using correct method
            try:
                # Use fetch_returns_for_exposure which expects Exposure object
                returns, description = fetcher.fetch_returns_for_exposure(
                    exposure,
                    start_date=datetime(2020, 1, 1),
                    end_date=datetime(2024, 12, 31),
                    frequency='monthly'
                )
                
                if returns is not None and len(returns) > 0:
                    print(f"  ✅ {description}: {len(returns)} months of return data")
                    results[exp_id] = {
                        'available': True, 
                        'periods': len(returns),
                        'description': description,
                        'impl_type': primary_impl.type
                    }
                else:
                    print(f"  ❌ {description}: No return data")
                    results[exp_id] = {
                        'available': False, 
                        'description': description,
                        'reason': 'No data returned'
                    }
                        
            except Exception as e:
                print(f"  ❌ Error fetching data: {e}")
                results[exp_id] = {'available': False, 'error': str(e)}
        else:
            # Just check configuration
            if hasattr(primary_impl, 'ticker') and primary_impl.ticker:
                print(f"  📋 Configured: {primary_impl.ticker} ({primary_impl.type})")
                results[exp_id] = {'available': 'unknown', 'ticker': primary_impl.ticker, 'type': primary_impl.type}
            elif hasattr(primary_impl, 'tickers') and primary_impl.tickers:
                print(f"  📋 Configured: ETF Average {primary_impl.tickers}")
                results[exp_id] = {'available': 'unknown', 'tickers': primary_impl.tickers, 'type': primary_impl.type}
            else:
                print(f"  📋 Configured: {primary_impl.type}")
                results[exp_id] = {'available': 'unknown', 'type': primary_impl.type}
                
        # Show alternative implementations
        if len(implementations) > 1:
            print(f"    Alternative implementations available: {len(implementations) - 1}")
            for i, impl in enumerate(implementations[1:], 1):
                if hasattr(impl, 'ticker') and impl.ticker:
                    print(f"      {i}. {impl.ticker} ({impl.type})")
                elif hasattr(impl, 'tickers') and impl.tickers:
                    print(f"      {i}. ETF Average {impl.tickers}")
                else:
                    print(f"      {i}. {impl.type}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    if fetcher:
        available = sum(1 for r in results.values() if r.get('available') == True)
        unavailable = sum(1 for r in results.values() if r.get('available') == False)
        print(f"✅ Data available: {available}/{len(results)} exposures")
        print(f"❌ Data unavailable: {unavailable}/{len(results)} exposures")
        
        if unavailable > 0:
            print("\nMissing data for:")
            for exp_id, result in results.items():
                if result.get('available') == False:
                    reason = result.get('reason', result.get('error', 'Unknown'))
                    print(f"  - {exp_id}: {reason}")
    else:
        print(f"📋 Total exposures configured: {len(results)}")
        print("   (Data fetcher not available - run availability check for actual data verification)")
    
    print(f"\nExposure types in universe:")
    types = {}
    for exp_id, exposure in universe.exposures.items():
        asset_class = getattr(exposure, 'asset_class', 'unknown')
        if asset_class not in types:
            types[asset_class] = []
        types[asset_class].append(exp_id)
    
    for asset_class, exposures in types.items():
        print(f"  {asset_class}: {len(exposures)} exposures")
        for exp in exposures:
            status = "✅" if results.get(exp, {}).get('available') == True else "❌" if results.get(exp, {}).get('available') == False else "📋"
            print(f"    {status} {exp}")
    
    return results

if __name__ == "__main__":
    check_data_availability()