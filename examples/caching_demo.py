#!/usr/bin/env python3
"""
Caching Performance Demo

This module demonstrates the performance benefits of the caching layer.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from datetime import date
from typing import Dict, List

from src.data.interfaces import RawDataType
from src.data.providers import TransformedDataProvider, RawDataProviderCoordinator
from src.data.cache import create_cached_provider


class CachingDemo:
    """Demonstrate caching layer performance benefits."""
    
    def __init__(self):
        """Initialize providers."""
        self.raw_coordinator = RawDataProviderCoordinator()
        self.uncached_provider = TransformedDataProvider(self.raw_coordinator)
        
        # Create cached version
        self.cached_provider = create_cached_provider(
            provider=self.uncached_provider,
            cache_type='memory',
            cache_config={'max_size': 100, 'max_memory_mb': 50}
        )
    
    def measure_performance(
        self, 
        symbol: str,
        start_date: date,
        end_date: date,
        num_calls: int = 3
    ) -> Dict[str, float]:
        """
        Measure performance comparison between cached and uncached providers.
        
        Args:
            symbol: Ticker symbol to test
            start_date: Start date for data
            end_date: End date for data
            num_calls: Number of repeated calls to measure
            
        Returns:
            Dictionary with timing results
        """
        print(f"⚡ Testing performance for {symbol}")
        print(f"📅 Date range: {start_date} to {end_date}")
        
        # Test uncached performance
        print(f"\n1️⃣ Testing uncached provider ({num_calls} calls)...")
        uncached_times = []
        
        for i in range(num_calls):
            start_time = time.time()
            try:
                data = self.uncached_provider.get_data(
                    data_type=RawDataType.ADJUSTED_CLOSE,
                    start=start_date,
                    end=end_date,
                    ticker=symbol,
                    frequency="daily"
                )
                elapsed = time.time() - start_time
                uncached_times.append(elapsed)
                print(f"   Call {i+1}: {elapsed:.3f}s ({len(data)} data points)")
                
            except Exception as e:
                print(f"   Call {i+1}: Failed - {str(e)}")
                uncached_times.append(float('inf'))
        
        avg_uncached = np.mean([t for t in uncached_times if t != float('inf')])
        
        # Test cached performance (first call = cache miss)
        print(f"\n2️⃣ Testing cached provider...")
        
        cache_miss_time = None
        try:
            start_time = time.time()
            cached_data = self.cached_provider.get_data(
                data_type=RawDataType.ADJUSTED_CLOSE,
                start=start_date,
                end=end_date,
                ticker=symbol,
                frequency="daily"
            )
            cache_miss_time = time.time() - start_time
            print(f"   Cache miss: {cache_miss_time:.3f}s ({len(cached_data)} data points)")
            
        except Exception as e:
            print(f"   Cache miss: Failed - {str(e)}")
            cache_miss_time = float('inf')
        
        # Test cache hits
        cached_hit_times = []
        for i in range(num_calls):
            start_time = time.time()
            try:
                data = self.cached_provider.get_data(
                    data_type=RawDataType.ADJUSTED_CLOSE,
                    start=start_date,
                    end=end_date,
                    ticker=symbol,
                    frequency="daily"
                )
                elapsed = time.time() - start_time
                cached_hit_times.append(elapsed)
                print(f"   Cache hit {i+1}: {elapsed:.4f}s")
                
            except Exception as e:
                print(f"   Cache hit {i+1}: Failed - {str(e)}")
                cached_hit_times.append(float('inf'))
        
        avg_cached_hit = np.mean([t for t in cached_hit_times if t != float('inf')])
        
        # Calculate metrics
        results = {
            'avg_uncached': avg_uncached,
            'cache_miss_time': cache_miss_time,
            'avg_cached_hit': avg_cached_hit,
            'speedup': avg_uncached / avg_cached_hit if avg_cached_hit > 0 else 0,
            'efficiency_gain': ((avg_uncached - avg_cached_hit) / avg_uncached) * 100 if avg_uncached > 0 else 0
        }
        
        print(f"\n⚡ PERFORMANCE RESULTS:")
        print(f"   Uncached average: {results['avg_uncached']:.3f}s")
        print(f"   Cached hit average: {results['avg_cached_hit']:.4f}s")
        print(f"   🚀 Speedup: {results['speedup']:.1f}x faster")
        print(f"   📈 Efficiency gain: {results['efficiency_gain']:.1f}%")
        
        return results
    
    def get_cache_statistics(self) -> Dict[str, any]:
        """Get current cache statistics."""
        try:
            stats = self.cached_provider.get_cache_stats()
            return stats
        except Exception as e:
            print(f"❌ Failed to get cache stats: {str(e)}")
            return {}
    
    def test_multiple_symbols(
        self, 
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, any]:
        """
        Test caching performance across multiple symbols.
        
        Args:
            symbols: List of symbols to test
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with aggregated results
        """
        print(f"🔥 Testing cache with {len(symbols)} symbols")
        
        # Warm the cache
        print("\n1️⃣ Warming cache...")
        warm_start = time.time()
        warmed_count = 0
        
        for symbol in symbols:
            try:
                self.cached_provider.get_data(
                    data_type=RawDataType.ADJUSTED_CLOSE,
                    start=start_date,
                    end=end_date,
                    ticker=symbol,
                    frequency="daily"
                )
                warmed_count += 1
                print(f"   ✅ {symbol}")
                
            except Exception as e:
                print(f"   ❌ {symbol}: {str(e)}")
        
        warm_time = time.time() - warm_start
        print(f"Cache warming: {warm_time:.2f}s for {warmed_count}/{len(symbols)} symbols")
        
        # Test warm performance
        print(f"\n2️⃣ Testing warm cache performance...")
        warm_times = []
        
        for symbol in symbols[:warmed_count]:  # Only test successfully warmed symbols
            start_time = time.time()
            try:
                self.cached_provider.get_data(
                    data_type=RawDataType.ADJUSTED_CLOSE,
                    start=start_date,
                    end=end_date,
                    ticker=symbol,
                    frequency="daily"
                )
                elapsed = time.time() - start_time
                warm_times.append(elapsed)
                
            except Exception as e:
                print(f"   ❌ {symbol}: {str(e)}")
        
        avg_warm_time = np.mean(warm_times) if warm_times else 0
        
        results = {
            'symbols_tested': len(symbols),
            'symbols_warmed': warmed_count,
            'warm_time': warm_time,
            'avg_warm_request': avg_warm_time,
            'cache_stats': self.get_cache_statistics()
        }
        
        print(f"\n📊 MULTI-SYMBOL RESULTS:")
        print(f"   Symbols warmed: {warmed_count}/{len(symbols)}")
        print(f"   Average warm request: {avg_warm_time:.4f}s")
        
        return results


def demo_caching_performance():
    """Run caching performance demonstration."""
    print("⚡ Cache Performance Demo")
    print("=" * 50)
    
    demo = CachingDemo()
    
    # Test parameters
    test_symbol = 'AAPL'
    start_date = date(2023, 1, 1)
    end_date = date(2023, 3, 31)
    
    # Single symbol performance test
    print("\n🎯 Single Symbol Performance Test")
    performance_results = demo.measure_performance(
        symbol=test_symbol,
        start_date=start_date,
        end_date=end_date,
        num_calls=3
    )
    
    # Multiple symbols test
    print("\n🎯 Multiple Symbols Cache Test")
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    multi_results = demo.test_multiple_symbols(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Final cache statistics
    print("\n📊 Final Cache Statistics")
    final_stats = demo.get_cache_statistics()
    if final_stats:
        provider_stats = final_stats.get('provider_stats', {})
        cache_stats = final_stats.get('cache_stats', {})
        
        print(f"   Total requests: {provider_stats.get('requests', 0)}")
        print(f"   Cache hits: {provider_stats.get('cache_hits', 0)}")
        print(f"   Hit rate: {provider_stats.get('hit_rate', 0):.1f}%")
        print(f"   Memory usage: {cache_stats.get('memory_usage_mb', 0):.2f} MB")
    
    print("\n✅ Cache demo completed!")
    return {
        'performance': performance_results,
        'multi_symbol': multi_results,
        'final_stats': final_stats
    }


if __name__ == "__main__":
    # Run the demo when script is executed directly
    results = demo_caching_performance()