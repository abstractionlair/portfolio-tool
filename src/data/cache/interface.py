"""
Cache interfaces and protocols.

This module defines the core caching interfaces that all cache implementations
must follow, providing a consistent API across different cache types.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate as a percentage."""
        return 100.0 - self.hit_rate


class CacheEntry(NamedTuple):
    """Represents a cached item with metadata."""
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class Cache(ABC):
    """
    Abstract base class for all cache implementations.
    
    Provides a consistent interface for storing and retrieving cached data
    with support for time-to-live (TTL) expiration and cache statistics.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: The cache key to look up
            
        Returns:
            The cached value if found and not expired, None otherwise
        """
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The cache key to store under
            value: The value to cache
            ttl: Optional time-to-live for the cached item
        """
        pass
    
    @abstractmethod
    def invalidate(self, key: str) -> bool:
        """
        Remove a specific key from the cache.
        
        Args:
            key: The cache key to remove
            
        Returns:
            True if the key was found and removed, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the cache."""
        pass
    
    @abstractmethod
    def stats(self) -> CacheStats:
        """
        Get current cache statistics.
        
        Returns:
            CacheStats object with current performance metrics
        """
        pass
    
    @abstractmethod
    def cleanup_expired(self) -> int:
        """
        Remove expired items from the cache.
        
        Returns:
            Number of expired items removed
        """
        pass
    
    def contains(self, key: str) -> bool:
        """
        Check if a key exists in the cache (without retrieving the value).
        
        Args:
            key: The cache key to check
            
        Returns:
            True if the key exists and is not expired, False otherwise
        """
        return self.get(key) is not None
    
    def touch(self, key: str, ttl: Optional[timedelta] = None) -> bool:
        """
        Update the expiration time of a cached item without changing its value.
        
        Args:
            key: The cache key to touch
            ttl: New time-to-live for the item
            
        Returns:
            True if the key was found and updated, False otherwise
        """
        value = self.get(key)
        if value is not None:
            self.put(key, value, ttl)
            return True
        return False


class CacheManager(ABC):
    """
    Abstract base class for cache managers that coordinate multiple caches.
    
    Cache managers can implement sophisticated caching strategies like
    multi-level caching, cache warming, and intelligent eviction.
    """
    
    @abstractmethod
    def get_cache(self, cache_name: str) -> Optional[Cache]:
        """Get a specific cache by name."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all managed caches."""
        pass
    
    @abstractmethod
    def clear_all(self) -> None:
        """Clear all managed caches."""
        pass
    
    @abstractmethod
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired items from all caches."""
        pass


class Serializable(ABC):
    """
    Interface for objects that can be cached with custom serialization.
    
    Useful for complex objects that need special handling when being
    stored in persistent caches.
    """
    
    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize the object to bytes for storage."""
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: bytes) -> 'Serializable':
        """Deserialize bytes back to the original object."""
        pass


class CacheKeyGenerator(ABC):
    """
    Abstract base class for cache key generation strategies.
    
    Different data types may require different key generation strategies
    to ensure uniqueness and optimize cache performance.
    """
    
    @abstractmethod
    def generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from the given arguments."""
        pass
    
    @abstractmethod
    def parse_key(self, key: str) -> Dict[str, Any]:
        """Parse a cache key back into its component parts."""
        pass