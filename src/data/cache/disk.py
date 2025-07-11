"""
Persistent disk cache implementation.

This module provides a disk-based cache for long-term storage of data
with compression, expiration, and automatic cleanup capabilities.
"""

import os
import json
import pickle
import gzip
import hashlib
import threading
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Set
import logging

from .interface import Cache, CacheStats, CacheEntry

logger = logging.getLogger(__name__)


class DiskCache(Cache):
    """
    Persistent disk-based cache with compression and automatic cleanup.
    
    Features:
    - Persistent storage survives application restarts
    - Optional gzip compression to save disk space
    - Automatic cleanup of expired items
    - Directory structure optimization for performance
    - Metadata tracking for efficient operations
    - Thread-safe operations
    """
    
    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 1.0,
        compression: bool = True,
        cleanup_interval_hours: int = 24
    ):
        """
        Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_gb: Maximum disk space to use in gigabytes
            compression: Whether to compress cache files
            cleanup_interval_hours: How often to run automatic cleanup
        """
        self._cache_dir = Path(cache_dir)
        self._max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._compression = compression
        self._cleanup_interval = timedelta(hours=cleanup_interval_hours)
        
        # Create cache directory structure
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self._cache_dir / "data"
        self._data_dir.mkdir(exist_ok=True)
        
        # Metadata file for tracking cache entries
        self._metadata_file = self._cache_dir / "metadata.json"
        self._metadata: Dict[str, Dict] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = CacheStats()
        
        # Last cleanup time
        self._last_cleanup = datetime.now()
        
        # Load existing metadata
        self._load_metadata()
        
        # Initialize statistics from metadata
        self._update_stats_from_metadata()
        
        logger.info(f"Initialized DiskCache: dir={cache_dir}, max_size={max_size_gb}GB, "
                   f"compression={compression}")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the disk cache."""
        with self._lock:
            # Check if we need to run cleanup
            self._maybe_cleanup()
            
            # Check metadata first
            if key not in self._metadata:
                self._stats.misses += 1
                return None
            
            entry_meta = self._metadata[key]
            
            # Check if expired
            if self._is_expired_meta(entry_meta):
                self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # Load from disk
            try:
                file_path = self._get_file_path(key)
                if not file_path.exists():
                    # File missing - remove from metadata
                    self._remove_entry(key)
                    self._stats.misses += 1
                    return None
                
                value = self._load_file(file_path)
                
                # Update access tracking
                entry_meta['access_count'] = entry_meta.get('access_count', 0) + 1
                entry_meta['last_accessed'] = datetime.now().isoformat()
                self._save_metadata()
                
                self._stats.hits += 1
                return value
                
            except Exception as e:
                logger.warning(f"Error loading cache entry {key}: {e}")
                self._remove_entry(key)
                self._stats.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Store a value in the disk cache."""
        with self._lock:
            now = datetime.now()
            expires_at = (now + ttl).isoformat() if ttl else None
            
            # Calculate file path
            file_path = self._get_file_path(key)
            
            try:
                # Save to disk
                self._save_file(file_path, value)
                file_size = file_path.stat().st_size
                
                # Update metadata
                self._metadata[key] = {
                    'created_at': now.isoformat(),
                    'expires_at': expires_at,
                    'file_path': str(file_path.relative_to(self._cache_dir)),
                    'file_size': file_size,
                    'access_count': 1,
                    'last_accessed': now.isoformat()
                }
                
                # Check disk space and cleanup if necessary
                self._ensure_disk_space()
                
                # Save metadata
                self._save_metadata()
                
                # Update stats
                self._stats.size = len(self._metadata)
                self._stats.memory_usage_bytes = sum(
                    meta.get('file_size', 0) for meta in self._metadata.values()
                )
                
                logger.debug(f"Cached {key} to disk: {file_size} bytes, expires={expires_at}")
                
            except Exception as e:
                logger.error(f"Error saving cache entry {key}: {e}")
                # Clean up partial write
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                raise
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific key from the cache."""
        with self._lock:
            if key in self._metadata:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Remove all items from the cache."""
        with self._lock:
            # Remove all data files
            if self._data_dir.exists():
                shutil.rmtree(self._data_dir)
                self._data_dir.mkdir()
            
            # Clear metadata
            self._metadata.clear()
            self._save_metadata()
            
            # Reset stats
            self._stats = CacheStats()
            
            logger.info("Disk cache cleared")
    
    def stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            # Update size information
            current_stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=len(self._metadata),
                memory_usage_bytes=sum(
                    meta.get('file_size', 0) for meta in self._metadata.values()
                )
            )
            return current_stats
    
    def cleanup_expired(self) -> int:
        """Remove expired items from the cache."""
        with self._lock:
            return self._cleanup_expired_items()
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get detailed disk usage information."""
        with self._lock:
            total_size = sum(meta.get('file_size', 0) for meta in self._metadata.values())
            total_files = len(self._metadata)
            
            if total_files == 0:
                return {
                    'total_bytes': 0,
                    'total_files': 0,
                    'average_file_bytes': 0,
                    'largest_file_bytes': 0,
                    'disk_efficiency': 0.0,
                    'compression_ratio': 0.0
                }
            
            file_sizes = [meta.get('file_size', 0) for meta in self._metadata.values()]
            
            return {
                'total_bytes': total_size,
                'total_files': total_files,
                'average_file_bytes': total_size // total_files,
                'largest_file_bytes': max(file_sizes) if file_sizes else 0,
                'disk_efficiency': (total_size / self._max_size_bytes) * 100,
                'compression_enabled': self._compression
            }
    
    def _get_file_path(self, key: str) -> Path:
        """Generate file path for a cache key."""
        # Use hash to create safe filename and distribute files across subdirectories
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir = key_hash[:2]  # First 2 chars for subdirectory
        filename = key_hash[2:] + ('.gz' if self._compression else '.pkl')
        
        subdir_path = self._data_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        
        return subdir_path / filename
    
    def _save_file(self, file_path: Path, value: Any) -> None:
        """Save a value to disk with optional compression."""
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self._compression:
            with gzip.open(file_path, 'wb') as f:
                f.write(data)
        else:
            with open(file_path, 'wb') as f:
                f.write(data)
    
    def _load_file(self, file_path: Path) -> Any:
        """Load a value from disk with optional decompression."""
        if self._compression:
            with gzip.open(file_path, 'rb') as f:
                data = f.read()
        else:
            with open(file_path, 'rb') as f:
                data = f.read()
        
        return pickle.loads(data)
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, 'r') as f:
                    self._metadata = json.load(f)
                logger.debug(f"Loaded metadata for {len(self._metadata)} cache entries")
            else:
                self._metadata = {}
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
            self._metadata = {}
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _update_stats_from_metadata(self) -> None:
        """Initialize statistics from loaded metadata."""
        self._stats.size = len(self._metadata)
        self._stats.memory_usage_bytes = sum(
            meta.get('file_size', 0) for meta in self._metadata.values()
        )
    
    def _is_expired_meta(self, entry_meta: Dict, now: Optional[datetime] = None) -> bool:
        """Check if a metadata entry is expired."""
        expires_at_str = entry_meta.get('expires_at')
        if not expires_at_str:
            return False
        
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            check_time = now or datetime.now()
            return check_time >= expires_at
        except Exception:
            return False
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry and its file."""
        if key in self._metadata:
            entry_meta = self._metadata[key]
            
            # Remove file
            try:
                file_path = self._cache_dir / entry_meta['file_path']
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Error removing cache file for {key}: {e}")
            
            # Remove from metadata
            del self._metadata[key]
            self._save_metadata()
    
    def _cleanup_expired_items(self) -> int:
        """Remove expired items and return count."""
        expired_keys = []
        now = datetime.now()
        
        for key, entry_meta in self._metadata.items():
            if self._is_expired_meta(entry_meta, now):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired disk cache entries")
        
        self._last_cleanup = now
        return len(expired_keys)
    
    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        if datetime.now() - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired_items()
    
    def _ensure_disk_space(self) -> None:
        """Ensure we don't exceed disk space limits."""
        current_size = sum(meta.get('file_size', 0) for meta in self._metadata.values())
        
        if current_size > self._max_size_bytes:
            # Sort by last access time (oldest first)
            entries_by_access = sorted(
                self._metadata.items(),
                key=lambda x: x[1].get('last_accessed', '1970-01-01')
            )
            
            # Remove oldest entries until under limit
            target_size = int(self._max_size_bytes * 0.8)  # Remove to 80% of limit
            
            for key, entry_meta in entries_by_access:
                if current_size <= target_size:
                    break
                
                file_size = entry_meta.get('file_size', 0)
                self._remove_entry(key)
                current_size -= file_size
                self._stats.evictions += 1
                
                logger.debug(f"Evicted for disk space: {key} ({file_size} bytes)")