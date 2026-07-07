"""
Cache-aware data provider implementations.

This module provides wrapper classes that add caching capabilities to existing
data providers, improving performance while maintaining the same interface.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

from ..interfaces import DataProvider
from .interface import Cache
from .memory import MemoryCache
from .disk import DiskCache
from .multilevel import MultiLevelCache
from .utils import generate_data_cache_key, get_cache_ttl, create_cache_namespace, should_cache_data

logger = logging.getLogger(__name__)


class CachedDataProvider(DataProvider):
    """
    A wrapper that adds caching capabilities to any DataProvider.
    
    This class implements the decorator pattern, wrapping an existing
    DataProvider with transparent caching functionality.
    """
    
    def __init__(
        self,
        provider: DataProvider,
        cache: Optional[Cache] = None,
        cache_namespace: Optional[str] = None,
        enable_cache_on_error: bool = True,
        cache_stats_log_interval: int = 100
    ):
        """
        Initialize the cached data provider.
        
        Args:
            provider: The underlying data provider to wrap
            cache: Cache instance to use (creates default if None)
            cache_namespace: Namespace for cache keys
            enable_cache_on_error: Whether to return cached data on provider errors
            cache_stats_log_interval: Log cache stats every N requests
        """
        self._provider = provider
        self._cache = cache or self._create_default_cache()
        self._namespace = cache_namespace or create_cache_namespace(
            provider.__class__.__name__, 'data'
        )
        self._enable_cache_on_error = enable_cache_on_error
        self._cache_stats_log_interval = cache_stats_log_interval
        
        # Statistics
        self._request_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._error_recoveries = 0
        
        logger.info(f"Initialized CachedDataProvider: provider={provider.__class__.__name__}, "
                   f"namespace={self._namespace}")
    
    def get_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        **kwargs
    ) -> pd.DataFrame:
        """Get data with caching support."""
        self._request_count += 1
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            'get_data', symbols, start_date, end_date, **kwargs
        )
        
        # Try cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self._cache_hits += 1
            self._log_stats_if_needed()
            return cached_data
        
        self._cache_misses += 1
        
        # Fetch from provider
        try:
            data = self._provider.get_data(symbols, start_date, end_date, **kwargs)
            
            # Cache the result if it's worth caching
            if should_cache_data(data):
                ttl = get_cache_ttl('prices', kwargs.get('frequency'))
                self._put_in_cache(cache_key, data, ttl)
            
            self._log_stats_if_needed()
            return data
            
        except Exception as e:
            # Try to return cached data if available and error recovery is enabled
            if self._enable_cache_on_error:
                stale_data = self._get_from_cache(cache_key, allow_expired=True)
                if stale_data is not None:
                    self._error_recoveries += 1
                    logger.warning(f"Returning stale cached data due to provider error: {e}")
                    return stale_data
            
            # Re-raise if no cached data available
            raise
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols with caching."""
        cache_key = self._generate_cache_key('get_available_symbols')
        
        # Try cache first
        cached_symbols = self._get_from_cache(cache_key)
        if cached_symbols is not None:
            return cached_symbols
        
        # Fetch from provider
        try:
            symbols = self._provider.get_available_symbols()
            
            # Cache with longer TTL since symbols change infrequently
            ttl = get_cache_ttl('symbols')
            self._put_in_cache(cache_key, symbols, ttl)
            
            return symbols
            
        except Exception as e:
            # Try to return cached data on error
            if self._enable_cache_on_error:
                stale_symbols = self._get_from_cache(cache_key, allow_expired=True)
                if stale_symbols is not None:
                    logger.warning(f"Returning stale symbol list due to provider error: {e}")
                    return stale_symbols
            
            raise
    
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get symbol metadata with caching."""
        cache_key = self._generate_cache_key('get_metadata', symbol)
        
        # Try cache first
        cached_metadata = self._get_from_cache(cache_key)
        if cached_metadata is not None:
            return cached_metadata
        
        # Fetch from provider
        try:
            metadata = self._provider.get_metadata(symbol)
            
            # Cache with longer TTL since metadata changes infrequently
            ttl = get_cache_ttl('metadata')
            self._put_in_cache(cache_key, metadata, ttl)
            
            return metadata
            
        except Exception as e:
            # Try to return cached data on error
            if self._enable_cache_on_error:
                stale_metadata = self._get_from_cache(cache_key, allow_expired=True)
                if stale_metadata is not None:
                    logger.warning(f"Returning stale metadata for {symbol} due to provider error: {e}")
                    return stale_metadata
            
            raise
    
    def invalidate_cache(self, symbol: Optional[str] = None) -> int:
        """
        Invalidate cached data.
        
        Args:
            symbol: Specific symbol to invalidate, or None for all data
            
        Returns:
            Number of cache entries invalidated
        """
        if symbol:
            # Invalidate entries for specific symbol
            # Note: This is a simplified approach - in practice, we'd need
            # more sophisticated key pattern matching
            keys_to_invalidate = [
                self._generate_cache_key('get_data', symbol),
                self._generate_cache_key('get_metadata', symbol)
            ]
            
            invalidated = 0
            for key in keys_to_invalidate:
                if self._cache.invalidate(key):
                    invalidated += 1
            
            logger.info(f"Invalidated {invalidated} cache entries for symbol {symbol}")
            return invalidated
        else:
            # Clear all cache
            self._cache.clear()
            logger.info("Cleared all cached data")
            return 0  # Can't count entries in a full clear
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_stats = self._cache.stats()
        
        return {
            'provider_stats': {
                'requests': self._request_count,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'error_recoveries': self._error_recoveries,
                'hit_rate': (self._cache_hits / max(1, self._request_count)) * 100
            },
            'cache_stats': {
                'hits': cache_stats.hits,
                'misses': cache_stats.misses,
                'evictions': cache_stats.evictions,
                'size': cache_stats.size,
                'memory_usage_mb': cache_stats.memory_usage_bytes / (1024 * 1024),
                'hit_rate': cache_stats.hit_rate
            },
            'namespace': self._namespace
        }
    
    def warm_cache(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Pre-populate cache with data for given symbols.
        
        Args:
            symbols: List of symbols to warm cache for
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters for data fetching
            
        Returns:
            Dictionary with warming results
        """
        logger.info(f"Warming cache for {len(symbols)} symbols")
        
        warmed_count = 0
        failed_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            try:
                # This will fetch and cache the data
                self.get_data(symbol, start_date, end_date, **kwargs)
                warmed_count += 1
            except Exception as e:
                failed_count += 1
                failed_symbols.append(symbol)
                logger.warning(f"Failed to warm cache for {symbol}: {e}")
        
        result = {
            'warmed_count': warmed_count,
            'failed_count': failed_count,
            'failed_symbols': failed_symbols,
            'total_symbols': len(symbols)
        }
        
        logger.info(f"Cache warming completed: {warmed_count}/{len(symbols)} successful")
        return result
    
    def _create_default_cache(self) -> Cache:
        """Create a default multi-level cache."""
        # Create memory cache with reasonable defaults
        memory_cache = MemoryCache(
            max_size=1000,
            default_ttl=timedelta(minutes=30),
            max_memory_mb=256
        )
        
        # Create disk cache in temp directory
        cache_dir = Path.home() / '.portfolio_tool_cache' / 'data'
        disk_cache = DiskCache(
            cache_dir=cache_dir,
            max_size_gb=2.0,
            compression=True
        )
        
        # Combine in multi-level cache
        return MultiLevelCache(
            memory_cache=memory_cache,
            disk_cache=disk_cache,
            promote_to_memory=True
        )
    
    def _generate_cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate a namespaced cache key."""
        key = generate_data_cache_key(
            provider_type=self._provider.__class__.__name__,
            data_type=method,
            symbols=args[0] if args else 'none',
            start_date=args[1] if len(args) > 1 else 'none',
            end_date=args[2] if len(args) > 2 else 'none',
            **kwargs
        )
        return f"{self._namespace}:{key}"
    
    def _get_from_cache(self, key: str, allow_expired: bool = False) -> Optional[Any]:
        """Get data from cache with optional expired data retrieval."""
        try:
            # For now, just use regular get - expired data handling would require
            # direct access to cache internals or extended interface
            return self._cache.get(key)
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
            return None
    
    def _put_in_cache(self, key: str, data: Any, ttl: timedelta) -> None:
        """Put data in cache with error handling."""
        try:
            self._cache.put(key, data, ttl)
        except Exception as e:
            logger.warning(f"Error storing in cache: {e}")
    
    def _log_stats_if_needed(self) -> None:
        """Log cache statistics periodically."""
        if self._request_count % self._cache_stats_log_interval == 0:
            stats = self.get_cache_stats()
            provider_stats = stats['provider_stats']
            cache_stats = stats['cache_stats']
            
            logger.info(
                f"Cache stats after {self._request_count} requests: "
                f"Provider hit rate: {provider_stats['hit_rate']:.1f}%, "
                f"Cache hit rate: {cache_stats['hit_rate']:.1f}%, "
                f"Memory usage: {cache_stats['memory_usage_mb']:.1f}MB"
            )


class CachedTransformedDataProvider(CachedDataProvider):
    """
    Specialized cached provider for transformed data with custom TTL logic.
    
    Transformed data often has different caching characteristics than raw data,
    so this class provides specialized behavior.
    """
    
    def get_returns(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        **kwargs
    ) -> pd.DataFrame:
        """Get returns data with caching."""
        cache_key = self._generate_cache_key(
            'get_returns', symbols, start_date, end_date, **kwargs
        )
        
        # Try cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch from provider
        try:
            data = self._provider.get_returns(symbols, start_date, end_date, **kwargs)
            
            # Use shorter TTL for returns as they're computed from price data
            ttl = get_cache_ttl('returns', kwargs.get('frequency'))
            self._put_in_cache(cache_key, data, ttl)
            
            return data
            
        except Exception as e:
            if self._enable_cache_on_error:
                stale_data = self._get_from_cache(cache_key, allow_expired=True)
                if stale_data is not None:
                    logger.warning(f"Returning stale returns data due to error: {e}")
                    return stale_data
            raise
    
    def get_volatility(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        **kwargs
    ) -> pd.DataFrame:
        """Get volatility data with caching."""
        cache_key = self._generate_cache_key(
            'get_volatility', symbols, start_date, end_date, **kwargs
        )
        
        # Try cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch from provider
        try:
            data = self._provider.get_volatility(symbols, start_date, end_date, **kwargs)
            
            # Moderate TTL for volatility calculations
            ttl = get_cache_ttl('volatility', kwargs.get('frequency'))
            self._put_in_cache(cache_key, data, ttl)
            
            return data
            
        except Exception as e:
            if self._enable_cache_on_error:
                stale_data = self._get_from_cache(cache_key, allow_expired=True)
                if stale_data is not None:
                    logger.warning(f"Returning stale volatility data due to error: {e}")
                    return stale_data
            raise


def create_cached_provider(
    provider: DataProvider,
    cache_type: str = 'multilevel',
    cache_config: Optional[Dict[str, Any]] = None
) -> CachedDataProvider:
    """
    Factory function to create cached providers with different cache configurations.
    
    Args:
        provider: The provider to wrap with caching
        cache_type: Type of cache ('memory', 'disk', 'multilevel')
        cache_config: Optional configuration for the cache
        
    Returns:
        A cached data provider
    """
    config = cache_config or {}
    
    if cache_type == 'memory':
        cache = MemoryCache(
            max_size=config.get('max_size', 1000),
            default_ttl=timedelta(minutes=config.get('default_ttl_minutes', 30)),
            max_memory_mb=config.get('max_memory_mb', 256)
        )
    elif cache_type == 'disk':
        cache_dir = config.get('cache_dir', Path.home() / '.portfolio_tool_cache')
        cache = DiskCache(
            cache_dir=Path(cache_dir),
            max_size_gb=config.get('max_size_gb', 2.0),
            compression=config.get('compression', True)
        )
    elif cache_type == 'multilevel':
        memory_cache = MemoryCache(
            max_size=config.get('memory_max_size', 1000),
            default_ttl=timedelta(minutes=config.get('memory_ttl_minutes', 30)),
            max_memory_mb=config.get('memory_max_mb', 256)
        )
        
        cache_dir = config.get('disk_cache_dir', Path.home() / '.portfolio_tool_cache')
        disk_cache = DiskCache(
            cache_dir=Path(cache_dir),
            max_size_gb=config.get('disk_max_gb', 2.0),
            compression=config.get('disk_compression', True)
        )
        
        cache = MultiLevelCache(
            memory_cache=memory_cache,
            disk_cache=disk_cache,
            promote_to_memory=config.get('promote_to_memory', True)
        )
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
    
    # Use specialized provider for transformed data if applicable
    if hasattr(provider, 'get_returns') or hasattr(provider, 'get_volatility'):
        return CachedTransformedDataProvider(
            provider=provider,
            cache=cache,
            enable_cache_on_error=config.get('enable_cache_on_error', True)
        )
    else:
        return CachedDataProvider(
            provider=provider,
            cache=cache,
            enable_cache_on_error=config.get('enable_cache_on_error', True)
        )