"""
Multi-level cache implementation.

This module provides a cache that coordinates multiple cache levels,
typically combining fast memory cache (L1) with persistent disk cache (L2).
"""

from typing import Any, Optional, Dict, List
from datetime import timedelta
import logging

from .interface import Cache, CacheStats, CacheManager
from .memory import MemoryCache
from .disk import DiskCache

logger = logging.getLogger(__name__)


class MultiLevelCache(Cache):
    """
    Multi-level cache that coordinates memory and disk caches.
    
    Cache hierarchy:
    1. L1 (Memory): Ultra-fast for recently accessed data
    2. L2 (Disk): Persistent storage for larger datasets
    
    Read strategy:
    - Check L1 first (memory cache)
    - If miss, check L2 (disk cache)
    - If L2 hit, promote to L1
    - If both miss, return None
    
    Write strategy:
    - Always write to L1
    - Optionally write to L2 based on TTL and size
    """
    
    def __init__(
        self,
        memory_cache: Optional[MemoryCache] = None,
        disk_cache: Optional[DiskCache] = None,
        promote_to_memory: bool = True,
        memory_ttl_threshold: Optional[timedelta] = None
    ):
        """
        Initialize the multi-level cache.
        
        Args:
            memory_cache: L1 memory cache (created if None)
            disk_cache: L2 disk cache (created if None)
            promote_to_memory: Whether to promote L2 hits to L1
            memory_ttl_threshold: Only promote to memory if TTL is above this threshold
        """
        self._l1_cache = memory_cache or MemoryCache()
        self._l2_cache = disk_cache
        self._promote_to_memory = promote_to_memory
        self._memory_ttl_threshold = memory_ttl_threshold
        
        # Statistics tracking
        self._l1_stats = CacheStats()
        self._l2_stats = CacheStats()
        
        logger.info(f"Initialized MultiLevelCache: L1={type(self._l1_cache).__name__}, "
                   f"L2={type(self._l2_cache).__name__ if self._l2_cache else 'None'}")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value using multi-level strategy."""
        # Try L1 first (memory)
        value = self._l1_cache.get(key)
        if value is not None:
            self._l1_stats.hits += 1
            return value
        
        self._l1_stats.misses += 1
        
        # Try L2 (disk) if available
        if self._l2_cache:
            value = self._l2_cache.get(key)
            if value is not None:
                self._l2_stats.hits += 1
                
                # Promote to L1 if enabled
                if self._promote_to_memory:
                    try:
                        self._l1_cache.put(key, value)
                        logger.debug(f"Promoted {key} from L2 to L1")
                    except Exception as e:
                        logger.warning(f"Failed to promote {key} to L1: {e}")
                
                return value
            
            self._l2_stats.misses += 1
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Store a value using multi-level strategy."""
        # Always store in L1 (memory)
        try:
            self._l1_cache.put(key, value, ttl)
        except Exception as e:
            logger.warning(f"Failed to store {key} in L1: {e}")
        
        # Store in L2 (disk) based on criteria
        if self._l2_cache and self._should_store_in_l2(key, value, ttl):
            try:
                self._l2_cache.put(key, value, ttl)
                logger.debug(f"Stored {key} in both L1 and L2")
            except Exception as e:
                logger.warning(f"Failed to store {key} in L2: {e}")
    
    def invalidate(self, key: str) -> bool:
        """Remove a key from all cache levels."""
        l1_removed = self._l1_cache.invalidate(key)
        l2_removed = self._l2_cache.invalidate(key) if self._l2_cache else False
        
        return l1_removed or l2_removed
    
    def clear(self) -> None:
        """Clear all cache levels."""
        self._l1_cache.clear()
        if self._l2_cache:
            self._l2_cache.clear()
        
        # Reset stats
        self._l1_stats = CacheStats()
        self._l2_stats = CacheStats()
        
        logger.info("Multi-level cache cleared")
    
    def stats(self) -> CacheStats:
        """Get combined statistics from all levels."""
        l1_stats = self._l1_cache.stats()
        l2_stats = self._l2_cache.stats() if self._l2_cache else CacheStats()
        
        # Combine statistics
        combined_stats = CacheStats(
            hits=l1_stats.hits + l2_stats.hits,
            misses=l1_stats.misses + l2_stats.misses,
            evictions=l1_stats.evictions + l2_stats.evictions,
            size=l1_stats.size + l2_stats.size,
            memory_usage_bytes=l1_stats.memory_usage_bytes + l2_stats.memory_usage_bytes
        )
        
        return combined_stats
    
    def stats_by_level(self) -> Dict[str, CacheStats]:
        """Get statistics broken down by cache level."""
        stats = {'L1': self._l1_cache.stats()}
        
        if self._l2_cache:
            stats['L2'] = self._l2_cache.stats()
        
        return stats
    
    def cleanup_expired(self) -> int:
        """Clean up expired items from all levels."""
        l1_cleaned = self._l1_cache.cleanup_expired()
        l2_cleaned = self._l2_cache.cleanup_expired() if self._l2_cache else 0
        
        total_cleaned = l1_cleaned + l2_cleaned
        
        if total_cleaned > 0:
            logger.debug(f"Cleaned up {total_cleaned} expired items "
                        f"(L1: {l1_cleaned}, L2: {l2_cleaned})")
        
        return total_cleaned
    
    def get_l1_cache(self) -> Cache:
        """Get the L1 (memory) cache."""
        return self._l1_cache
    
    def get_l2_cache(self) -> Optional[Cache]:
        """Get the L2 (disk) cache."""
        return self._l2_cache
    
    def warm_cache(self, keys_and_values: List[tuple], ttl: Optional[timedelta] = None) -> int:
        """
        Warm the cache with multiple key-value pairs.
        
        Args:
            keys_and_values: List of (key, value) tuples to pre-load
            ttl: TTL for all pre-loaded items
            
        Returns:
            Number of items successfully loaded
        """
        loaded_count = 0
        
        for key, value in keys_and_values:
            try:
                self.put(key, value, ttl)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for {key}: {e}")
        
        logger.info(f"Warmed cache with {loaded_count}/{len(keys_and_values)} items")
        return loaded_count
    
    def _should_store_in_l2(self, key: str, value: Any, ttl: Optional[timedelta]) -> bool:
        """Determine if a value should be stored in L2 cache."""
        # Don't store in L2 if it doesn't exist
        if not self._l2_cache:
            return False
        
        # Don't store if TTL is too short (L2 is for longer-term storage)
        if ttl and self._memory_ttl_threshold and ttl < self._memory_ttl_threshold:
            return False
        
        # For now, store everything in L2
        # In the future, this could be more sophisticated:
        # - Size-based decisions
        # - Access pattern analysis
        # - Data type specific rules
        return True


class CacheCluster(CacheManager):
    """
    Manager for multiple named caches.
    
    Allows different data types or use cases to have their own
    cache configurations while providing centralized management.
    """
    
    def __init__(self):
        """Initialize the cache cluster."""
        self._caches: Dict[str, Cache] = {}
        logger.info("Initialized CacheCluster")
    
    def add_cache(self, name: str, cache: Cache) -> None:
        """Add a named cache to the cluster."""
        self._caches[name] = cache
        logger.debug(f"Added cache '{name}' to cluster")
    
    def get_cache(self, cache_name: str) -> Optional[Cache]:
        """Get a specific cache by name."""
        return self._caches.get(cache_name)
    
    def remove_cache(self, cache_name: str) -> bool:
        """Remove a cache from the cluster."""
        if cache_name in self._caches:
            del self._caches[cache_name]
            logger.debug(f"Removed cache '{cache_name}' from cluster")
            return True
        return False
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all managed caches."""
        return {name: cache.stats() for name, cache in self._caches.items()}
    
    def clear_all(self) -> None:
        """Clear all managed caches."""
        for cache in self._caches.values():
            cache.clear()
        logger.info("Cleared all caches in cluster")
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired items from all caches."""
        cleanup_counts = {}
        
        for name, cache in self._caches.items():
            try:
                count = cache.cleanup_expired()
                cleanup_counts[name] = count
            except Exception as e:
                logger.warning(f"Error cleaning up cache '{name}': {e}")
                cleanup_counts[name] = 0
        
        total_cleaned = sum(cleanup_counts.values())
        logger.info(f"Cleaned up {total_cleaned} expired items across all caches")
        
        return cleanup_counts
    
    def list_caches(self) -> List[str]:
        """Get list of all cache names."""
        return list(self._caches.keys())
    
    def cache_sizes(self) -> Dict[str, int]:
        """Get size of each cache."""
        return {name: cache.stats().size for name, cache in self._caches.items()}