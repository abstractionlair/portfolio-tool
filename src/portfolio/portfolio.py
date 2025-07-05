"""Portfolio class for managing multiple positions."""

from datetime import datetime
from typing import Dict, Optional, List
import csv
import pandas as pd
import logging
from .position import Position

logger = logging.getLogger(__name__)


class Portfolio:
    """Portfolio containing multiple positions.
    
    Attributes:
        positions: Dictionary of Position objects keyed by symbol
        cash: Cash balance
        name: Portfolio name/identifier
        last_update: Timestamp of last update
    """
    
    def __init__(self, name: str = "Portfolio", cash: float = 0.0) -> None:
        """Initialize a portfolio.
        
        Args:
            name: Portfolio name/identifier
            cash: Initial cash balance
            
        Raises:
            ValueError: If cash is negative
        """
        if cash < 0:
            raise ValueError("Cash cannot be negative")
            
        self.name = name
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.last_update: Optional[datetime] = None
        
        logger.debug(f"Created portfolio '{self.name}' with ${cash:.2f} cash")
    
    def add_position(self, position: Position) -> None:
        """Add or update a position.
        
        Args:
            position: Position to add
            
        Example:
            >>> portfolio = Portfolio("My Portfolio")
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> portfolio.add_position(pos)
            >>> len(portfolio.positions)
            1
        """
        if position.symbol in self.positions:
            # Merge with existing position
            existing = self.positions[position.symbol]
            existing.add_shares(position.quantity, position.cost_basis)
        else:
            self.positions[position.symbol] = position
        
        self.last_update = datetime.now()
        logger.debug(f"Added position {position.symbol} to portfolio '{self.name}'")
    
    def remove_position(self, symbol: str) -> None:
        """Remove position entirely.
        
        Args:
            symbol: Symbol to remove
            
        Raises:
            KeyError: If symbol not found in portfolio
            
        Example:
            >>> portfolio = Portfolio("My Portfolio")
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> portfolio.add_position(pos)
            >>> portfolio.remove_position("SPY")
            >>> len(portfolio.positions)
            0
        """
        if symbol not in self.positions:
            raise KeyError(f"Position {symbol} not found in portfolio")
        
        del self.positions[symbol]
        self.last_update = datetime.now()
        logger.debug(f"Removed position {symbol} from portfolio '{self.name}'")
    
    def add_trade(self, symbol: str, quantity: float, price: float) -> None:
        """Execute a trade.
        
        Args:
            symbol: Stock/ETF ticker symbol
            quantity: Number of shares (positive for buy, negative for sell)
            price: Price per share
            
        Raises:
            ValueError: If price is negative
            
        Example:
            >>> portfolio = Portfolio("My Portfolio", cash=50000.0)
            >>> portfolio.add_trade("SPY", 100, 400.0)  # Buy 100 shares
            >>> portfolio.cash
            10000.0
        """
        if price < 0:
            raise ValueError("Price cannot be negative")
        
        trade_value = quantity * price
        
        if symbol in self.positions:
            self.positions[symbol].add_shares(quantity, price)
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                cost_basis=price,
                purchase_date=datetime.now()
            )
        
        # Update cash balance
        self.cash -= trade_value
        self.last_update = datetime.now()
        
        logger.debug(f"Executed trade: {quantity} shares of {symbol} at ${price}")
    
    def get_weights(self, prices_dict: Dict[str, float]) -> Dict[str, float]:
        """Calculate position weights.
        
        Args:
            prices_dict: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of weights by symbol
            
        Example:
            >>> portfolio = Portfolio("My Portfolio")
            >>> pos1 = Position("SPY", 100, 400.0, datetime.now())
            >>> pos2 = Position("QQQ", 50, 300.0, datetime.now())
            >>> portfolio.add_position(pos1)
            >>> portfolio.add_position(pos2)
            >>> weights = portfolio.get_weights({"SPY": 410.0, "QQQ": 310.0})
            >>> weights["SPY"]  # 41000 / (41000 + 15500)
            0.7256637168141593
        """
        total_value = self.total_value(prices_dict)
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in self.positions.items():
            if symbol in prices_dict:
                market_value = position.market_value(prices_dict[symbol])
                weights[symbol] = market_value / total_value
        
        return weights
    
    def get_exposure(self, asset_class: str, prices_dict: Dict[str, float]) -> float:
        """Calculate exposure by asset class.
        
        Args:
            asset_class: Asset class to calculate exposure for
            prices_dict: Dictionary of current prices by symbol
            
        Returns:
            Total exposure for the asset class
            
        Example:
            >>> portfolio = Portfolio("My Portfolio")
            >>> pos = Position("SPY", 100, 400.0, datetime.now(), asset_class="Equity")
            >>> portfolio.add_position(pos)
            >>> exposure = portfolio.get_exposure("Equity", {"SPY": 410.0})
            >>> exposure
            41000.0
        """
        total_exposure = 0.0
        
        for symbol, position in self.positions.items():
            if position.asset_class == asset_class and symbol in prices_dict:
                market_value = position.market_value(prices_dict[symbol])
                total_exposure += market_value * position.leverage_factor
        
        return total_exposure
    
    def calculate_total_exposures(self, fund_map=None, prices: Optional[Dict[str, float]] = None) -> Dict:
        """Aggregate true exposures across all positions.
        
        Args:
            fund_map: FundExposureMap containing fund definitions
            prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of total exposures by ExposureType
            
        Example:
            >>> portfolio = Portfolio("My Portfolio")
            >>> pos = Position("SPY", 100, 400.0, datetime.now(), asset_class="Equity")
            >>> portfolio.add_position(pos)
            >>> exposures = portfolio.calculate_total_exposures()
            >>> exposures
            {<ExposureType.US_LARGE_EQUITY: 'US_LARGE_EQUITY'>: 40000.0}
        """
        if fund_map and prices:
            from .exposures import ExposureCalculator
            calculator = ExposureCalculator(fund_map)
            return calculator.calculate_portfolio_exposures(self, prices)
        else:
            # Fall back to simple aggregation
            from .exposures import ExposureType
            exposures = {}
            
            for position in self.positions.values():
                current_price = prices.get(position.symbol, position.cost_basis) if prices else position.cost_basis
                position_exposures = position.get_exposures(fund_map, current_price)
                
                for exposure in position_exposures:
                    if exposure.exposure_type not in exposures:
                        exposures[exposure.exposure_type] = 0.0
                    exposures[exposure.exposure_type] += exposure.amount
            
            return exposures
    
    def total_value(self, prices_dict: Dict[str, float]) -> float:
        """Calculate total portfolio value.
        
        Args:
            prices_dict: Dictionary of current prices by symbol
            
        Returns:
            Total portfolio value including cash
            
        Example:
            >>> portfolio = Portfolio("My Portfolio", cash=1000.0)
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> portfolio.add_position(pos)
            >>> total = portfolio.total_value({"SPY": 410.0})
            >>> total
            42000.0
        """
        positions_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in prices_dict:
                positions_value += position.market_value(prices_dict[symbol])
        
        return positions_value + self.cash
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export positions as pandas DataFrame.
        
        Returns:
            DataFrame with position data
            
        Example:
            >>> portfolio = Portfolio("My Portfolio")
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> portfolio.add_position(pos)
            >>> df = portfolio.to_dataframe()
            >>> df.shape[0]
            1
        """
        if not self.positions:
            return pd.DataFrame(columns=[
                'symbol', 'quantity', 'cost_basis', 'purchase_date', 
                'leverage_factor', 'asset_class'
            ])
        
        data = []
        for position in self.positions.values():
            data.append({
                'symbol': position.symbol,
                'quantity': position.quantity,
                'cost_basis': position.cost_basis,
                'purchase_date': position.purchase_date,
                'leverage_factor': position.leverage_factor,
                'asset_class': position.asset_class
            })
        
        return pd.DataFrame(data)
    
    def from_csv(self, filepath: str) -> None:
        """Import positions from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV format is invalid
            
        Example:
            >>> portfolio = Portfolio("My Portfolio")
            >>> portfolio.from_csv("positions.csv")
        """
        try:
            df = pd.read_csv(filepath)
            required_columns = ['symbol', 'quantity', 'cost_basis', 'purchase_date']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV")
            
            # Clear existing positions
            self.positions.clear()
            
            for _, row in df.iterrows():
                position = Position(
                    symbol=row['symbol'],
                    quantity=float(row['quantity']),
                    cost_basis=float(row['cost_basis']),
                    purchase_date=pd.to_datetime(row['purchase_date']),
                    leverage_factor=float(row.get('leverage_factor', 1.0)),
                    asset_class=row.get('asset_class', 'Unknown')
                )
                self.add_position(position)
            
            logger.info(f"Imported {len(self.positions)} positions from {filepath}")
            
        except FileNotFoundError:
            logger.error(f"File {filepath} not found")
            raise
        except Exception as e:
            logger.error(f"Error importing CSV: {e}")
            raise ValueError(f"Error importing CSV: {e}")
    
    def to_csv(self, filepath: str) -> None:
        """Export positions to CSV file.
        
        Args:
            filepath: Path to save CSV file
            
        Example:
            >>> portfolio = Portfolio("My Portfolio")
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> portfolio.add_position(pos)
            >>> portfolio.to_csv("positions.csv")
        """
        try:
            df = self.to_dataframe()
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(self.positions)} positions to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            raise
    
    def get_summary(self, prices_dict: Dict[str, float]) -> Dict[str, float]:
        """Get portfolio summary statistics.
        
        Args:
            prices_dict: Dictionary of current prices by symbol
            
        Returns:
            Dictionary with summary statistics
            
        Example:
            >>> portfolio = Portfolio("My Portfolio", cash=1000.0)
            >>> pos = Position("SPY", 100, 400.0, datetime.now())
            >>> portfolio.add_position(pos)
            >>> summary = portfolio.get_summary({"SPY": 410.0})
            >>> summary["total_value"]
            42000.0
        """
        total_val = self.total_value(prices_dict)
        total_cost = sum(pos.quantity * pos.cost_basis for pos in self.positions.values())
        total_unrealized_pnl = sum(
            pos.unrealized_pnl(prices_dict.get(pos.symbol, pos.cost_basis)) 
            for pos in self.positions.values()
        )
        
        return {
            'total_value': total_val,
            'total_cost': total_cost + self.cash,
            'cash': self.cash,
            'positions_count': len(self.positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_return_pct': (total_unrealized_pnl / total_cost * 100) if total_cost > 0 else 0.0
        }
    
    def __repr__(self) -> str:
        """String representation of the portfolio."""
        return (f"Portfolio(name='{self.name}', positions={len(self.positions)}, "
                f"cash=${self.cash:.2f})")
    
    def __len__(self) -> int:
        """Return number of positions in portfolio."""
        return len(self.positions)