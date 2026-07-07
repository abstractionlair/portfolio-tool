"""Position class for portfolio management."""

from datetime import datetime
from typing import Optional, List
from decimal import Decimal, ROUND_HALF_UP
import logging

logger = logging.getLogger(__name__)


class Position:
    """Represents a single position in a portfolio.
    
    Attributes:
        symbol: Stock/ETF ticker symbol
        quantity: Number of shares (can be negative for shorts)
        cost_basis: Average price paid per share
        purchase_date: When position was acquired
        leverage_factor: For leveraged ETFs (default 1.0)
        asset_class: Equity, Bond, Commodity, etc.
    """
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        cost_basis: float,
        purchase_date: datetime,
        leverage_factor: float = 1.0,
        asset_class: str = "Unknown"
    ) -> None:
        """Initialize a position.
        
        Args:
            symbol: Stock/ETF ticker symbol
            quantity: Number of shares (can be negative for shorts)
            cost_basis: Average price paid per share
            purchase_date: When position was acquired
            leverage_factor: For leveraged ETFs (default 1.0)
            asset_class: Equity, Bond, Commodity, etc.
            
        Raises:
            ValueError: If symbol is empty or cost_basis is negative
        """
        if not symbol.strip():
            raise ValueError("Symbol cannot be empty")
        if cost_basis < 0:
            raise ValueError("Cost basis cannot be negative")
        if leverage_factor <= 0:
            raise ValueError("Leverage factor must be positive")
            
        self.symbol = symbol.strip().upper()
        self.quantity = quantity
        self.cost_basis = cost_basis
        self.purchase_date = purchase_date
        self.leverage_factor = leverage_factor
        self.asset_class = asset_class
        
        logger.debug(f"Created position: {self.symbol} x {self.quantity} @ ${self.cost_basis}")
    
    def market_value(self, current_price: float) -> float:
        """Calculate current market value of the position.
        
        Args:
            current_price: Current price per share
            
        Returns:
            Market value of the position
            
        Example:
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> pos.market_value(410.0)
            41000.0
        """
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss.
        
        Args:
            current_price: Current price per share
            
        Returns:
            Unrealized profit/loss
            
        Example:
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> pos.unrealized_pnl(410.0)
            1000.0
        """
        return self.quantity * (current_price - self.cost_basis)
    
    def add_shares(self, quantity: float, price: float) -> None:
        """Add shares to the position with proper cost basis averaging.
        
        Args:
            quantity: Number of shares to add
            price: Price per share for the new shares
            
        Raises:
            ValueError: If price is negative
            
        Example:
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> pos.add_shares(50, 420.0)  # Buy 50 more shares at $420
            >>> pos.quantity
            150.0
            >>> pos.cost_basis  # Weighted average: (100*400 + 50*420) / 150
            406.6666666666667
        """
        if price < 0:
            raise ValueError("Price cannot be negative")
            
        # Calculate new cost basis using weighted average
        current_value = self.quantity * self.cost_basis
        new_value = quantity * price
        total_quantity = self.quantity + quantity
        
        if total_quantity != 0:
            self.cost_basis = (current_value + new_value) / total_quantity
        
        self.quantity = total_quantity
        
        logger.debug(f"Added {quantity} shares to {self.symbol} at ${price}, new cost basis: ${self.cost_basis:.2f}")
    
    def remove_shares(self, quantity: float) -> None:
        """Remove shares from the position (FIFO).
        
        Args:
            quantity: Number of shares to remove
            
        Raises:
            ValueError: If trying to remove more shares than available
            
        Example:
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> pos.remove_shares(25)
            >>> pos.quantity
            75.0
        """
        if abs(quantity) > abs(self.quantity):
            raise ValueError(f"Cannot remove {quantity} shares, only {self.quantity} available")
            
        self.quantity -= quantity
        
        logger.debug(f"Removed {quantity} shares from {self.symbol}, new quantity: {self.quantity}")
    
    def get_exposures(self, fund_map=None, current_price: Optional[float] = None) -> List:
        """Return list of Exposure objects based on fund definition.
        
        Args:
            fund_map: FundExposureMap containing fund definitions
            current_price: Current price per share (uses cost_basis if not provided)
            
        Returns:
            List of Exposure objects
            
        Example:
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> exposures = pos.get_exposures()
            >>> len(exposures)
            1
        """
        if current_price is None:
            current_price = self.cost_basis
            
        if fund_map:
            from .exposures import ExposureCalculator
            calculator = ExposureCalculator(fund_map)
            return calculator.calculate_position_exposures(self, current_price)
        else:
            # Fall back to simple exposure based on asset class
            from .exposures import Exposure, ExposureType
            
            # Map asset class to exposure type
            mapping = {
                'equity': ExposureType.US_LARGE_EQUITY,
                'bond': ExposureType.BONDS,
                'bonds': ExposureType.BONDS,
                'commodity': ExposureType.COMMODITIES,
                'commodities': ExposureType.COMMODITIES,
                'reit': ExposureType.REAL_ESTATE,
                'real estate': ExposureType.REAL_ESTATE,
            }
            
            exposure_type = mapping.get(self.asset_class.lower(), ExposureType.US_LARGE_EQUITY)
            market_value = self.market_value(current_price)
            
            return [Exposure(exposure_type, market_value * self.leverage_factor)]
    
    def to_dict(self) -> dict:
        """Convert position to dictionary for serialization.
        
        Returns:
            Dictionary representation of the position
        """
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'cost_basis': self.cost_basis,
            'purchase_date': self.purchase_date.isoformat(),
            'leverage_factor': self.leverage_factor,
            'asset_class': self.asset_class
        }
    
    def __repr__(self) -> str:
        """String representation of the position."""
        return (f"Position(symbol='{self.symbol}', quantity={self.quantity}, "
                f"cost_basis={self.cost_basis:.2f}, asset_class='{self.asset_class}')")
    
    def __eq__(self, other) -> bool:
        """Check equality with another position."""
        if not isinstance(other, Position):
            return False
        return (self.symbol == other.symbol and 
                abs(self.quantity - other.quantity) < 1e-6 and
                abs(self.cost_basis - other.cost_basis) < 1e-6)