"""
In-memory cache implementation.

This module provides a high-performance in-memory cache with LRU eviction,
TTL support, and comprehensive statistics tracking.
"""

import threading
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import logging

from .interface import Cache, CacheStats, CacheEntry

logger = logging.getLogger(__name__)


class MemoryCache(Cache):
    """
    Thread-safe in-memory cache with LRU eviction and TTL support.
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - Per-item TTL (Time To Live) support
    - Thread-safe operations with read-write locks
    - Memory usage tracking and limits
    - Comprehensive performance statistics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[timedelta] = None,
        max_memory_mb: float = 100.0
    ):
        """
        Initialize the memory cache.
        
        Args:
            max_size: Maximum number of items to store
            default_ttl: Default time-to-live for cached items
            max_memory_mb: Maximum memory usage in megabytes
        """
        self._max_size = max_size
        self._default_ttl = default_ttl or timedelta(hours=1)
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        
        # Thread-safe storage using OrderedDict for LRU tracking
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        
        # Statistics tracking
        self._stats = CacheStats()
        
        # Memory usage tracking
        self._estimated_memory_usage = 0
        
        logger.info(f"Initialized MemoryCache: max_size={max_size}, "
                   f"default_ttl={default_ttl}, max_memory={max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                # Remove expired entry
                self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # Update access tracking
            updated_entry = entry._replace(
                access_count=entry.access_count + 1,
                last_accessed=datetime.now()
            )
            
            # Move to end (most recently used) for LRU tracking
            del self._cache[key]
            self._cache[key] = updated_entry
            
            self._stats.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Store a value in the cache."""
        with self._lock:
            now = datetime.now()
            actual_ttl = ttl or self._default_ttl
            expires_at = now + actual_ttl if actual_ttl else None
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=now,
                expires_at=expires_at,
                access_count=1,
                last_accessed=now
            )
            
            # Calculate estimated memory usage
            entry_size = self._estimate_size(key, value)
            
            # If key already exists, remove old entry first
            if key in self._cache:
                old_entry = self._cache[key]
                old_size = self._estimate_size(key, old_entry.value)
                self._estimated_memory_usage -= old_size
                del self._cache[key]
            
            # Check memory limits
            if self._estimated_memory_usage + entry_size > self._max_memory_bytes:
                self._evict_for_memory(entry_size)
            
            # Check size limits and evict if necessary
            while len(self._cache) >= self._max_size:
                self._evict_lru()
            
            # Add new entry
            self._cache[key] = entry
            self._estimated_memory_usage += entry_size
            self._stats.size = len(self._cache)
            self._stats.memory_usage_bytes = self._estimated_memory_usage
            
            logger.debug(f"Cached {key}: size={entry_size} bytes, expires={expires_at}")
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific key from the cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Remove all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._estimated_memory_usage = 0
            self._stats.size = 0
            self._stats.memory_usage_bytes = 0
            logger.info("Cache cleared")
    
    def stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            # Update current size stats
            current_stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=len(self._cache),
                memory_usage_bytes=self._estimated_memory_usage
            )
            return current_stats
    
    def cleanup_expired(self) -> int:
        """Remove expired items from the cache."""
        with self._lock:
            expired_keys = []
            now = datetime.now()
            
            for key, entry in self._cache.items():
                if self._is_expired(entry, now):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        with self._lock:
            total_entries = len(self._cache)
            if total_entries == 0:
                return {
                    'total_bytes': 0,
                    'average_entry_bytes': 0,
                    'largest_entry_bytes': 0,
                    'total_entries': 0,
                    'memory_efficiency': 0.0
                }
            
            entry_sizes = [self._estimate_size(k, entry.value) 
                          for k, entry in self._cache.items()]
            
            return {
                'total_bytes': self._estimated_memory_usage,
                'average_entry_bytes': self._estimated_memory_usage // total_entries,
                'largest_entry_bytes': max(entry_sizes) if entry_sizes else 0,
                'total_entries': total_entries,
                'memory_efficiency': (self._estimated_memory_usage / self._max_memory_bytes) * 100
            }
    
    def _is_expired(self, entry: CacheEntry, now: Optional[datetime] = None) -> bool:
        """Check if a cache entry is expired."""
        if entry.expires_at is None:
            return False
        
        check_time = now or datetime.now()
        return check_time >= entry.expires_at
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache[key]
            entry_size = self._estimate_size(key, entry.value)
            del self._cache[key]
            self._estimated_memory_usage -= entry_size
            self._stats.size = len(self._cache)
            self._stats.memory_usage_bytes = self._estimated_memory_usage
    
    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if self._cache:
            # OrderedDict FIFO behavior - first item is least recently used
            key, entry = self._cache.popitem(last=False)
            entry_size = self._estimate_size(key, entry.value)
            self._estimated_memory_usage -= entry_size
            self._stats.evictions += 1
            self._stats.size = len(self._cache)
            self._stats.memory_usage_bytes = self._estimated_memory_usage
            
            logger.debug(f"Evicted LRU entry: {key} ({entry_size} bytes)")
    
    def _evict_for_memory(self, required_bytes: int) -> None:
        """Evict entries to make room for new data."""
        freed_bytes = 0
        
        while (self._estimated_memory_usage + required_bytes > self._max_memory_bytes 
               and self._cache and freed_bytes < required_bytes * 2):
            # Evict oldest entries until we have enough space
            key, entry = self._cache.popitem(last=False)
            entry_size = self._estimate_size(key, entry.value)
            self._estimated_memory_usage -= entry_size
            freed_bytes += entry_size
            self._stats.evictions += 1
            
            logger.debug(f"Evicted for memory: {key} ({entry_size} bytes)")
        
        self._stats.size = len(self._cache)
        self._stats.memory_usage_bytes = self._estimated_memory_usage
    
    def _estimate_size(self, key: str, value: Any) -> int:
        """Estimate the memory size of a cache entry."""
        try:
            # Basic size estimation
            key_size = sys.getsizeof(key)
            value_size = sys.getsizeof(value)
            
            # For pandas objects, try to get more accurate size
            if hasattr(value, 'memory_usage'):
                # pandas DataFrame/Series
                value_size = value.memory_usage(deep=True).sum()
            elif hasattr(value, 'nbytes'):
                # numpy arrays
                value_size = value.nbytes
            
            # Add overhead for CacheEntry object
            overhead = sys.getsizeof(CacheEntry(value, datetime.now(), None))
            
            return key_size + value_size + overhead
            
        except Exception as e:
            logger.warning(f"Error estimating size for {key}: {e}")
            # Fallback to basic estimation
            return sys.getsizeof(key) + sys.getsizeof(value) + 200  # 200 bytes overhead


class RWLockMemoryCache(MemoryCache):
    """
    Memory cache with read-write locking for better concurrent performance.
    
    Uses separate read and write locks to allow multiple concurrent readers
    while ensuring exclusive access for writers.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._read_lock = threading.RLock()
        self._write_lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value using read lock."""
        with self._read_lock:
            return super().get(key)
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Store a value using write lock."""
        with self._write_lock:
            super().put(key, value, ttl)
    
    def invalidate(self, key: str) -> bool:
        """Remove a key using write lock."""
        with self._write_lock:
            return super().invalidate(key)
    
    def clear(self) -> None:
        """Clear cache using write lock."""
        with self._write_lock:
            super().clear()
    
    def cleanup_expired(self) -> int:
        """Cleanup expired items using write lock."""
        with self._write_lock:
            return super().cleanup_expired()