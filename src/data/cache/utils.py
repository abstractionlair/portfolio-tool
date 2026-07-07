"""
Cache utilities for key generation and TTL management.

This module provides utility functions for generating cache keys,
managing TTL values, and other common caching operations.
"""

import hashlib
import json
from datetime import timedelta, datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a deterministic cache key from arguments.
    
    Args:
        *args: Positional arguments to include in the key
        **kwargs: Keyword arguments to include in the key
        
    Returns:
        A unique string key for the given arguments
    """
    # Create a dictionary with all arguments
    key_data = {
        'args': _serialize_args(args),
        'kwargs': _serialize_kwargs(kwargs)
    }
    
    # Convert to JSON string for consistent ordering
    key_string = json.dumps(key_data, sort_keys=True, default=_json_serializer)
    
    # Generate hash for consistent key length
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


def generate_data_cache_key(
    provider_type: str,
    data_type: str,
    symbols: Union[str, List[str]],
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
    frequency: Optional[str] = None,
    **additional_params
) -> str:
    """
    Generate a cache key specifically for financial data requests.
    
    Args:
        provider_type: Type of data provider (e.g., 'yfinance', 'fred')
        data_type: Type of data (e.g., 'prices', 'dividends', 'economic')
        symbols: Symbol or list of symbols
        start_date: Start date for data
        end_date: End date for data
        frequency: Data frequency (optional)
        **additional_params: Any additional parameters
        
    Returns:
        A cache key specific to financial data requests
    """
    # Normalize symbols to a sorted list
    if isinstance(symbols, str):
        symbol_list = [symbols]
    else:
        symbol_list = sorted(list(symbols))
    
    # Normalize dates to ISO format strings
    start_str = _normalize_date(start_date)
    end_str = _normalize_date(end_date)
    
    # Create key components
    key_components = {
        'provider': provider_type,
        'data_type': data_type,
        'symbols': symbol_list,
        'start_date': start_str,
        'end_date': end_str,
        'frequency': frequency,
        'params': _serialize_kwargs(additional_params)
    }
    
    # Generate key
    return generate_cache_key(**key_components)


def get_cache_ttl(data_type: str, frequency: Optional[str] = None) -> timedelta:
    """
    Get appropriate TTL for different types of financial data.
    
    Args:
        data_type: Type of data being cached
        frequency: Data frequency (daily, monthly, etc.)
        
    Returns:
        Appropriate TTL for the data type
    """
    # Base TTL values based on data type
    ttl_mapping = {
        # Price data - shorter TTL for real-time needs
        'prices': timedelta(minutes=15),
        'adjusted_prices': timedelta(minutes=15),
        'ohlc': timedelta(minutes=15),
        
        # Corporate actions - longer TTL as they change infrequently
        'dividends': timedelta(hours=6),
        'splits': timedelta(hours=12),
        
        # Economic data - moderate TTL
        'economic': timedelta(hours=2),
        'rates': timedelta(hours=1),
        
        # Computed metrics - shorter TTL as they depend on price data
        'returns': timedelta(minutes=30),
        'volatility': timedelta(hours=1),
        'correlations': timedelta(hours=2),
        
        # Portfolio data - very short TTL as it changes frequently
        'portfolio_returns': timedelta(minutes=5),
        'portfolio_metrics': timedelta(minutes=10),
        
        # Reference data - longer TTL as it's relatively static
        'symbols': timedelta(days=1),
        'metadata': timedelta(hours=6),
        
        # Default fallback
        'default': timedelta(hours=1)
    }
    
    base_ttl = ttl_mapping.get(data_type, ttl_mapping['default'])
    
    # Adjust based on frequency
    if frequency:
        frequency_multipliers = {
            'minute': 0.1,   # Very short for minute data
            '5min': 0.2,
            '15min': 0.3,
            '30min': 0.5,
            'hourly': 0.7,
            'daily': 1.0,    # Base multiplier
            'weekly': 2.0,
            'monthly': 4.0,
            'quarterly': 8.0,
            'yearly': 16.0
        }
        
        multiplier = frequency_multipliers.get(frequency.lower(), 1.0)
        base_ttl = timedelta(seconds=base_ttl.total_seconds() * multiplier)
    
    # Ensure minimum TTL of 1 minute
    min_ttl = timedelta(minutes=1)
    return max(base_ttl, min_ttl)


def create_cache_namespace(provider_name: str, data_category: str) -> str:
    """
    Create a namespace for cache keys to avoid collisions.
    
    Args:
        provider_name: Name of the data provider
        data_category: Category of data (e.g., 'raw', 'transformed')
        
    Returns:
        A namespace string for cache keys
    """
    return f"{provider_name}:{data_category}"


def parse_cache_key(key: str) -> Dict[str, Any]:
    """
    Parse a cache key back into its components (for debugging).
    
    Args:
        key: The cache key to parse
        
    Returns:
        Dictionary with key components if parseable, empty dict otherwise
    """
    try:
        # For our hashed keys, we can't reverse engineer the original data
        # This is mainly useful for debugging and logging
        return {
            'key_hash': key,
            'key_length': len(key),
            'algorithm': 'sha256_truncated'
        }
    except Exception:
        return {}


def estimate_cache_memory_usage(data: Any) -> int:
    """
    Estimate memory usage of data for cache size management.
    
    Args:
        data: The data to estimate size for
        
    Returns:
        Estimated size in bytes
    """
    try:
        import sys
        
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, pd.Series):
            return data.memory_usage(deep=True)
        elif hasattr(data, 'nbytes'):  # numpy arrays
            return data.nbytes
        else:
            return sys.getsizeof(data)
    except Exception:
        # Fallback estimation
        return 1024  # 1KB default


def should_cache_data(
    data: Any,
    min_size_bytes: int = 100,
    max_size_bytes: int = 100 * 1024 * 1024  # 100MB
) -> bool:
    """
    Determine if data should be cached based on size constraints.
    
    Args:
        data: The data to evaluate
        min_size_bytes: Minimum size to be worth caching
        max_size_bytes: Maximum size that can be cached
        
    Returns:
        True if data should be cached, False otherwise
    """
    try:
        size = estimate_cache_memory_usage(data)
        return min_size_bytes <= size <= max_size_bytes
    except Exception:
        # If we can't estimate size, err on the side of caching
        return True


def _serialize_args(args) -> List[Any]:
    """Serialize positional arguments for key generation."""
    return [_serialize_value(arg) for arg in args]


def _serialize_kwargs(kwargs) -> Dict[str, Any]:
    """Serialize keyword arguments for key generation."""
    return {k: _serialize_value(v) for k, v in kwargs.items()}


def _serialize_value(value) -> Any:
    """Serialize a single value for consistent key generation."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    elif isinstance(value, pd.DatetimeIndex):
        return [dt.isoformat() for dt in value]
    elif hasattr(value, 'to_dict'):  # pandas objects
        return value.to_dict()
    else:
        # Fallback to string representation
        return str(value)


def _json_serializer(obj) -> str:
    """Custom JSON serializer for complex objects."""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):  # Other datetime-like objects
        return obj.isoformat()
    else:
        return str(obj)


def _normalize_date(date_value: Union[str, datetime, pd.Timestamp]) -> str:
    """Normalize a date value to ISO format string."""
    if isinstance(date_value, str):
        try:
            # Try to parse and reformat for consistency
            dt = pd.to_datetime(date_value)
            return dt.isoformat()
        except Exception:
            return date_value
    elif isinstance(date_value, (datetime, pd.Timestamp)):
        return date_value.isoformat()
    else:
        return str(date_value)