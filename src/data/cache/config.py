"""
Cache configuration management.

This module provides configuration classes and utilities for managing
cache settings across the application.
"""

import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryCacheConfig:
    """Configuration for memory caches."""
    max_size: int = 1000
    default_ttl_minutes: int = 30
    max_memory_mb: float = 256.0
    cleanup_interval_minutes: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'max_size': self.max_size,
            'default_ttl_minutes': self.default_ttl_minutes,
            'max_memory_mb': self.max_memory_mb,
            'cleanup_interval_minutes': self.cleanup_interval_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryCacheConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DiskCacheConfig:
    """Configuration for disk caches."""
    cache_dir: Path = field(default_factory=lambda: Path.home() / '.portfolio_tool_cache')
    max_size_gb: float = 2.0
    compression: bool = True
    cleanup_interval_hours: int = 24
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'cache_dir': str(self.cache_dir),
            'max_size_gb': self.max_size_gb,
            'compression': self.compression,
            'cleanup_interval_hours': self.cleanup_interval_hours
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiskCacheConfig':
        """Create from dictionary."""
        config_data = data.copy()
        if 'cache_dir' in config_data:
            config_data['cache_dir'] = Path(config_data['cache_dir'])
        return cls(**config_data)


@dataclass
class CacheConfig:
    """Comprehensive cache configuration."""
    
    # Cache types
    enabled: bool = True
    cache_type: str = 'multilevel'  # 'memory', 'disk', 'multilevel'
    
    # Cache behavior
    enable_cache_on_error: bool = True
    cache_stats_log_interval: int = 100
    promote_to_memory: bool = True
    
    # Memory cache settings
    memory: MemoryCacheConfig = field(default_factory=MemoryCacheConfig)
    
    # Disk cache settings  
    disk: DiskCacheConfig = field(default_factory=DiskCacheConfig)
    
    # TTL settings (in minutes)
    ttl_settings: Dict[str, int] = field(default_factory=lambda: {
        'prices': 15,
        'adjusted_prices': 15,
        'ohlc': 15,
        'dividends': 360,  # 6 hours
        'splits': 720,     # 12 hours
        'economic': 120,   # 2 hours
        'rates': 60,       # 1 hour
        'returns': 30,
        'volatility': 60,
        'correlations': 120,
        'portfolio_returns': 5,
        'portfolio_metrics': 10,
        'symbols': 1440,   # 24 hours
        'metadata': 360,   # 6 hours
        'default': 60
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'enabled': self.enabled,
            'cache_type': self.cache_type,
            'enable_cache_on_error': self.enable_cache_on_error,
            'cache_stats_log_interval': self.cache_stats_log_interval,
            'promote_to_memory': self.promote_to_memory,
            'memory': self.memory.to_dict(),
            'disk': self.disk.to_dict(),
            'ttl_settings': self.ttl_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheConfig':
        """Create from dictionary."""
        config_data = data.copy()
        
        if 'memory' in config_data:
            config_data['memory'] = MemoryCacheConfig.from_dict(config_data['memory'])
        
        if 'disk' in config_data:
            config_data['disk'] = DiskCacheConfig.from_dict(config_data['disk'])
        
        return cls(**config_data)
    
    def get_ttl_minutes(self, data_type: str) -> int:
        """Get TTL in minutes for a specific data type."""
        return self.ttl_settings.get(data_type, self.ttl_settings['default'])
    
    def get_ttl_timedelta(self, data_type: str) -> timedelta:
        """Get TTL as timedelta for a specific data type."""
        return timedelta(minutes=self.get_ttl_minutes(data_type))


class CacheConfigManager:
    """Manages cache configuration from environment and defaults."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self._config_file = config_file
        self._config = self._load_config()
    
    def get_config(self) -> CacheConfig:
        """Get the current cache configuration."""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        # Update the configuration object
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Save if we have a config file
        if self._config_file:
            self._save_config()
    
    def reload_config(self) -> None:
        """Reload configuration from sources."""
        self._config = self._load_config()
    
    def _load_config(self) -> CacheConfig:
        """Load configuration from environment and files."""
        # Start with defaults
        config = CacheConfig()
        
        # Apply environment variable overrides
        self._apply_env_overrides(config)
        
        # Apply file-based config if available
        if self._config_file and self._config_file.exists():
            try:
                import json
                with open(self._config_file, 'r') as f:
                    file_config = json.load(f)
                config = CacheConfig.from_dict(file_config)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error loading config file {self._config_file}: {e}")
        
        return config
    
    def _apply_env_overrides(self, config: CacheConfig) -> None:
        """Apply environment variable overrides to configuration."""
        # Cache enabled
        if os.getenv('PORTFOLIO_CACHE_ENABLED'):
            config.enabled = os.getenv('PORTFOLIO_CACHE_ENABLED', '').lower() == 'true'
        
        # Cache type
        if os.getenv('PORTFOLIO_CACHE_TYPE'):
            config.cache_type = os.getenv('PORTFOLIO_CACHE_TYPE', 'multilevel')
        
        # Memory cache settings
        if os.getenv('PORTFOLIO_CACHE_MEMORY_SIZE'):
            config.memory.max_size = int(os.getenv('PORTFOLIO_CACHE_MEMORY_SIZE'))
        
        if os.getenv('PORTFOLIO_CACHE_MEMORY_MB'):
            config.memory.max_memory_mb = float(os.getenv('PORTFOLIO_CACHE_MEMORY_MB'))
        
        # Disk cache settings
        if os.getenv('PORTFOLIO_CACHE_DISK_DIR'):
            config.disk.cache_dir = Path(os.getenv('PORTFOLIO_CACHE_DISK_DIR'))
        
        if os.getenv('PORTFOLIO_CACHE_DISK_GB'):
            config.disk.max_size_gb = float(os.getenv('PORTFOLIO_CACHE_DISK_GB'))
        
        if os.getenv('PORTFOLIO_CACHE_COMPRESSION'):
            config.disk.compression = os.getenv('PORTFOLIO_CACHE_COMPRESSION', '').lower() == 'true'
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        if not self._config_file:
            return
        
        try:
            import json
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_file, 'w') as f:
                json.dump(self._config.to_dict(), f, indent=2)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error saving config file {self._config_file}: {e}")


# Global configuration manager instance
_config_manager: Optional[CacheConfigManager] = None


def get_cache_config() -> CacheConfig:
    """Get the global cache configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = CacheConfigManager()
    return _config_manager.get_config()


def set_cache_config_file(config_file: Path) -> None:
    """Set the configuration file path."""
    global _config_manager
    _config_manager = CacheConfigManager(config_file)


def update_cache_config(**kwargs) -> None:
    """Update the global cache configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = CacheConfigManager()
    _config_manager.update_config(**kwargs)


# Environment-based configuration helpers
def is_cache_enabled() -> bool:
    """Check if caching is enabled via environment or config."""
    return get_cache_config().enabled


def get_default_cache_dir() -> Path:
    """Get the default cache directory."""
    return get_cache_config().disk.cache_dir


def get_cache_type() -> str:
    """Get the configured cache type."""
    return get_cache_config().cache_type