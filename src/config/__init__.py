"""
Configuration module for portfolio optimizer.

Handles loading configuration from environment variables and config files.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CACHE_DIR = DATA_DIR / "cache"
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
    TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
    
    # Data source preferences (in order of preference)
    DATA_SOURCES = ["yfinance", "alpha_vantage", "polygon", "twelve_data"]
    
    # Cache settings
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", None)
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        if cls.ENABLE_CACHE:
            cls.CACHE_DIR.mkdir(exist_ok=True)
            
    @classmethod
    def get_api_key(cls, service: str) -> Optional[str]:
        """Get API key for a specific service."""
        key_map = {
            "alpha_vantage": cls.ALPHA_VANTAGE_API_KEY,
            "fred": cls.FRED_API_KEY,
            "polygon": cls.POLYGON_API_KEY,
            "twelve_data": cls.TWELVE_DATA_API_KEY,
        }
        return key_map.get(service.lower())


# Ensure directories exist when module is imported
Config.ensure_directories()
