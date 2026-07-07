"""Fund exposure decomposition system."""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import yaml
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExposureType(Enum):
    """Standard exposure types for portfolio analysis."""
    
    # Equity Exposures
    US_LARGE_EQUITY = "US_LARGE_EQUITY"
    US_SMALL_EQUITY = "US_SMALL_EQUITY"
    US_VALUE_EQUITY = "US_VALUE_EQUITY"
    US_SMALL_VALUE_EQUITY = "US_SMALL_VALUE_EQUITY"
    INTL_EQUITY = "INTL_EQUITY"
    INTL_VALUE_EQUITY = "INTL_VALUE_EQUITY"
    EM_VALUE_EQUITY = "EM_VALUE_EQUITY"
    GLOBAL_EQUITY = "GLOBAL_EQUITY"
    
    # Fixed Income Exposures
    BONDS = "BONDS"
    US_BONDS = "US_BONDS"
    INTL_BONDS = "INTL_BONDS"
    LONG_DURATION_BONDS = "LONG_DURATION_BONDS"
    
    # Alternative Exposures
    COMMODITIES = "COMMODITIES"
    REAL_ESTATE = "REAL_ESTATE"
    MANAGED_FUTURES = "MANAGED_FUTURES"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    LONG_SHORT = "LONG_SHORT"
    
    # Factor Exposures
    VALUE_FACTOR = "VALUE_FACTOR"
    MOMENTUM_FACTOR = "MOMENTUM_FACTOR"
    QUALITY_FACTOR = "QUALITY_FACTOR"
    LOW_VOL_FACTOR = "LOW_VOL_FACTOR"
    CARRY = "CARRY"
    
    @classmethod
    def from_string(cls, exposure_str: str) -> 'ExposureType':
        """Convert string to ExposureType, handling case variations."""
        try:
            return cls(exposure_str.upper())
        except ValueError:
            # Try to find a close match
            for exposure_type in cls:
                if exposure_type.value.replace('_', '').lower() == exposure_str.replace('_', '').lower():
                    return exposure_type
            raise ValueError(f"Unknown exposure type: {exposure_str}")


@dataclass
class Exposure:
    """Represents an exposure to a specific asset class or strategy.
    
    Attributes:
        exposure_type: The type of exposure
        amount: The exposure amount (can be negative for short exposures)
    """
    
    exposure_type: ExposureType
    amount: float
    
    def scale(self, factor: float) -> 'Exposure':
        """Scale exposure by a factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New scaled Exposure object
            
        Example:
            >>> exp = Exposure(ExposureType.US_LARGE_EQUITY, 1.0)
            >>> scaled = exp.scale(0.5)
            >>> scaled.amount
            0.5
        """
        return Exposure(self.exposure_type, self.amount * factor)
    
    def __add__(self, other: 'Exposure') -> 'Exposure':
        """Add two exposures of the same type."""
        if self.exposure_type != other.exposure_type:
            raise ValueError(f"Cannot add exposures of different types: {self.exposure_type} and {other.exposure_type}")
        return Exposure(self.exposure_type, self.amount + other.amount)
    
    def __repr__(self) -> str:
        """String representation of exposure."""
        return f"Exposure({self.exposure_type.value}, {self.amount:.4f})"


@dataclass
class FundDefinition:
    """Definition of a fund's underlying exposures.
    
    Attributes:
        symbol: Fund ticker symbol
        name: Full fund name
        exposures: Dictionary mapping exposure types to amounts
        total_notional: Total notional exposure (sum of absolute exposures)
        category: Fund category (optional)
    """
    
    symbol: str
    name: str
    exposures: Dict[ExposureType, float]
    total_notional: float
    category: str = "Unknown"
    
    def get_exposures(self, position_value: float) -> List[Exposure]:
        """Convert position value to list of exposures.
        
        Args:
            position_value: Market value of the position
            
        Returns:
            List of Exposure objects scaled by position value
            
        Example:
            >>> fund_def = FundDefinition("RSSB", "Return Stacked Stocks & Bonds", 
            ...                          {ExposureType.US_LARGE_EQUITY: 1.0, ExposureType.BONDS: 1.0}, 2.0)
            >>> exposures = fund_def.get_exposures(10000.0)
            >>> len(exposures)
            2
            >>> exposures[0].amount
            10000.0
        """
        exposure_list = []
        for exposure_type, exposure_amount in self.exposures.items():
            scaled_amount = position_value * exposure_amount
            exposure_list.append(Exposure(exposure_type, scaled_amount))
        
        return exposure_list
    
    def validate(self) -> bool:
        """Validate the fund definition.
        
        Returns:
            True if valid, False otherwise
        """
        if not self.symbol or not self.name:
            return False
        
        if not self.exposures:
            return False
        
        # Check that total notional is reasonable
        calculated_notional = sum(abs(amount) for amount in self.exposures.values())
        if abs(calculated_notional - self.total_notional) > 0.01:
            logger.warning(f"Fund {self.symbol}: calculated notional {calculated_notional:.3f} "
                         f"differs from stated {self.total_notional:.3f}")
        
        return True
    
    @classmethod
    def from_dict(cls, symbol: str, fund_data: Dict) -> 'FundDefinition':
        """Create FundDefinition from dictionary data.
        
        Args:
            symbol: Fund symbol
            fund_data: Dictionary with fund information
            
        Returns:
            FundDefinition object
        """
        # Convert exposure strings to ExposureType enums
        exposures = {}
        for exp_str, amount in fund_data.get('exposures', {}).items():
            try:
                exposure_type = ExposureType.from_string(exp_str)
                exposures[exposure_type] = float(amount)
            except ValueError as e:
                logger.warning(f"Skipping unknown exposure type for {symbol}: {exp_str}")
                continue
        
        return cls(
            symbol=symbol,
            name=fund_data.get('name', ''),
            exposures=exposures,
            total_notional=float(fund_data.get('total_notional', 1.0)),
            category=fund_data.get('category', 'Unknown')
        )


class FundExposureMap:
    """Map of fund symbols to their exposure definitions.
    
    Handles loading and saving fund definitions from YAML/JSON files.
    """
    
    def __init__(self, definitions_path: Optional[str] = None):
        """Initialize with optional path to definitions file.
        
        Args:
            definitions_path: Path to YAML or JSON file with fund definitions
        """
        self.definitions: Dict[str, FundDefinition] = {}
        
        if definitions_path:
            self.load_definitions(definitions_path)
    
    def load_definitions(self, file_path: str) -> None:
        """Load fund definitions from YAML or JSON file.
        
        Args:
            file_path: Path to definitions file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Definitions file not found: {file_path}")
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # Extract fund definitions from the data structure
            funds_data = data.get('funds', {})
            
            for symbol, fund_info in funds_data.items():
                try:
                    fund_def = FundDefinition.from_dict(symbol, fund_info)
                    if fund_def.validate():
                        self.definitions[symbol] = fund_def
                    else:
                        logger.warning(f"Invalid fund definition for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading fund definition for {symbol}: {e}")
            
            logger.info(f"Loaded {len(self.definitions)} fund definitions from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading definitions file {file_path}: {e}")
            raise ValueError(f"Failed to load definitions: {e}")
    
    def get_fund_definition(self, symbol: str) -> Optional[FundDefinition]:
        """Get definition for a specific fund.
        
        Args:
            symbol: Fund ticker symbol
            
        Returns:
            FundDefinition if found, None otherwise
        """
        return self.definitions.get(symbol.upper())
    
    def add_fund_definition(self, definition: FundDefinition) -> None:
        """Add or update a fund definition.
        
        Args:
            definition: FundDefinition to add
        """
        if definition.validate():
            self.definitions[definition.symbol.upper()] = definition
            logger.debug(f"Added fund definition for {definition.symbol}")
        else:
            raise ValueError(f"Invalid fund definition for {definition.symbol}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all available fund symbols.
        
        Returns:
            List of fund symbols
        """
        return list(self.definitions.keys())
    
    def save_definitions(self, file_path: str, format: str = 'yaml') -> None:
        """Save current definitions to file.
        
        Args:
            file_path: Path to save file
            format: 'yaml' or 'json'
        """
        data = {
            'metadata': {
                'last_updated': 'auto-generated',
                'source': 'FundExposureMap'
            },
            'funds': {}
        }
        
        for symbol, fund_def in self.definitions.items():
            data['funds'][symbol] = {
                'name': fund_def.name,
                'category': fund_def.category,
                'exposures': {exp.value: amount for exp, amount in fund_def.exposures.items()},
                'total_notional': fund_def.total_notional
            }
        
        path = Path(file_path)
        with open(path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(data, f, default_flow_style=False, sort_keys=True)
            elif format.lower() == 'json':
                json.dump(data, f, indent=2, sort_keys=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(self.definitions)} fund definitions to {file_path}")
    
    def __len__(self) -> int:
        """Return number of fund definitions."""
        return len(self.definitions)
    
    def __contains__(self, symbol: str) -> bool:
        """Check if symbol exists in definitions."""
        return symbol.upper() in self.definitions


class ExposureCalculator:
    """Calculator for position and portfolio exposures."""
    
    def __init__(self, fund_map: FundExposureMap):
        """Initialize with fund exposure map.
        
        Args:
            fund_map: FundExposureMap containing fund definitions
        """
        self.fund_map = fund_map
    
    def calculate_position_exposures(self, position, current_price: float) -> List[Exposure]:
        """Calculate exposures for a single position.
        
        Args:
            position: Position object
            current_price: Current price per share
            
        Returns:
            List of Exposure objects
        """
        from .position import Position  # Import here to avoid circular import
        
        market_value = position.market_value(current_price)
        fund_def = self.fund_map.get_fund_definition(position.symbol)
        
        if fund_def:
            # Use fund definition to decompose exposures
            return fund_def.get_exposures(market_value)
        else:
            # Fall back to simple exposure based on asset class
            # Try to map asset class to exposure type
            exposure_type = self._map_asset_class_to_exposure(position.asset_class)
            return [Exposure(exposure_type, market_value * position.leverage_factor)]
    
    def calculate_portfolio_exposures(self, portfolio, prices: Dict[str, float]) -> Dict[ExposureType, float]:
        """Calculate total exposures for entire portfolio.
        
        Args:
            portfolio: Portfolio object
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary mapping exposure types to total amounts
        """
        total_exposures = {}
        
        for symbol, position in portfolio.positions.items():
            if symbol in prices:
                position_exposures = self.calculate_position_exposures(position, prices[symbol])
                
                for exposure in position_exposures:
                    if exposure.exposure_type not in total_exposures:
                        total_exposures[exposure.exposure_type] = 0.0
                    total_exposures[exposure.exposure_type] += exposure.amount
        
        return total_exposures
    
    def _map_asset_class_to_exposure(self, asset_class: str) -> ExposureType:
        """Map asset class string to ExposureType.
        
        Args:
            asset_class: Asset class string
            
        Returns:
            Corresponding ExposureType
        """
        mapping = {
            'equity': ExposureType.US_LARGE_EQUITY,
            'bond': ExposureType.BONDS,
            'bonds': ExposureType.BONDS,
            'commodity': ExposureType.COMMODITIES,
            'commodities': ExposureType.COMMODITIES,
            'reit': ExposureType.REAL_ESTATE,
            'real estate': ExposureType.REAL_ESTATE,
            'managed futures': ExposureType.MANAGED_FUTURES,
        }
        
        return mapping.get(asset_class.lower(), ExposureType.US_LARGE_EQUITY)