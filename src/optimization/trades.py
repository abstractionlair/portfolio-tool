"""Trade generation utilities for portfolio optimization."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal, ROUND_HALF_UP
import logging

try:
    from portfolio import Portfolio
except ImportError:
    try:
        from ..portfolio import Portfolio
    except ImportError:
        # Define stub for standalone usage
        class Portfolio:
            pass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade to execute."""
    symbol: str
    quantity: float  # Number of shares/units to trade (positive = buy, negative = sell)
    direction: str  # 'BUY' or 'SELL'
    trade_value: float  # Dollar value of trade
    current_price: float
    
    def __post_init__(self):
        """Validate trade data."""
        if self.quantity > 0 and self.direction != 'BUY':
            raise ValueError("Positive quantity must be BUY direction")
        if self.quantity < 0 and self.direction != 'SELL':
            raise ValueError("Negative quantity must be SELL direction")
        
        # Ensure trade_value matches quantity * price
        expected_value = abs(self.quantity * self.current_price)
        if abs(self.trade_value - expected_value) > 0.01:
            logger.warning(f"Trade value mismatch for {self.symbol}: "
                         f"expected {expected_value}, got {self.trade_value}")


class TradeGenerator:
    """Generates trades from optimization results."""
    
    def __init__(self, min_trade_value: float = 100.0, round_lots: bool = True):
        """Initialize trade generator.
        
        Args:
            min_trade_value: Minimum dollar value for a trade
            round_lots: Whether to round trades to whole shares
        """
        self.min_trade_value = min_trade_value
        self.round_lots = round_lots
        
        logger.debug(f"Initialized TradeGenerator with min_trade_value=${min_trade_value}")
    
    def generate_trades(
        self,
        target_weights: Dict[str, float],
        current_portfolio: Portfolio,
        prices: Dict[str, float],
        total_portfolio_value: float
    ) -> List[Trade]:
        """Generate trades to reach target weights from current portfolio.
        
        Args:
            target_weights: Desired portfolio weights (symbol -> weight)
            current_portfolio: Current portfolio holdings
            prices: Current market prices (symbol -> price)
            total_portfolio_value: Total portfolio value for calculating targets
            
        Returns:
            List of Trade objects to execute
        """
        trades = []
        
        # Get current positions as weights
        current_weights = self._get_current_weights(current_portfolio, prices, total_portfolio_value)
        
        # Calculate required trades for each symbol
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        
        for symbol in all_symbols:
            target_weight = target_weights.get(symbol, 0.0)
            current_weight = current_weights.get(symbol, 0.0)
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < 1e-6:  # Skip tiny differences
                continue
            
            if symbol not in prices:
                logger.warning(f"No price available for {symbol}, skipping trade")
                continue
            
            price = prices[symbol]
            target_value = target_weight * total_portfolio_value
            current_value = current_weight * total_portfolio_value
            trade_value = target_value - current_value
            
            if abs(trade_value) < self.min_trade_value:
                logger.debug(f"Trade value ${abs(trade_value):.2f} for {symbol} below minimum ${self.min_trade_value}")
                continue
            
            # Calculate quantity needed
            if price <= 0:
                logger.warning(f"Invalid price ${price} for {symbol}, skipping trade")
                continue
            
            raw_quantity = trade_value / price
            
            # Round to whole shares if requested
            if self.round_lots:
                quantity = self._round_quantity(raw_quantity)
            else:
                quantity = raw_quantity
            
            if abs(quantity) < 1e-6:  # Skip if rounded to zero
                continue
            
            # Determine direction
            direction = 'BUY' if quantity > 0 else 'SELL'
            actual_trade_value = abs(quantity * price)
            
            trade = Trade(
                symbol=symbol,
                quantity=quantity,
                direction=direction,
                trade_value=actual_trade_value,
                current_price=price
            )
            
            trades.append(trade)
            
            logger.debug(f"Generated trade: {direction} {abs(quantity):.2f} shares of {symbol} "
                        f"at ${price:.2f} (value: ${actual_trade_value:.2f})")
        
        # Sort trades by value (largest first) for execution priority
        trades.sort(key=lambda t: t.trade_value, reverse=True)
        
        logger.info(f"Generated {len(trades)} trades with total value "
                   f"${sum(t.trade_value for t in trades):.2f}")
        
        return trades
    
    def _get_current_weights(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        total_value: float
    ) -> Dict[str, float]:
        """Calculate current portfolio weights."""
        weights = {}
        
        for symbol, position in portfolio.positions.items():
            if symbol in prices and prices[symbol] > 0:
                position_value = position.quantity * prices[symbol]
                weights[symbol] = position_value / total_value if total_value > 0 else 0.0
        
        return weights
    
    def _round_quantity(self, quantity: float) -> float:
        """Round quantity to appropriate number of shares."""
        if abs(quantity) < 1.0:
            # For fractional shares, round to 2 decimal places
            decimal_qty = Decimal(str(quantity)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            return float(decimal_qty)
        else:
            # For whole shares, round to nearest integer
            return round(quantity)
    
    def calculate_trade_costs(self, trades: List[Trade], cost_per_trade: float = 0.0) -> float:
        """Calculate total transaction costs for a list of trades.
        
        Args:
            trades: List of trades
            cost_per_trade: Fixed cost per trade (e.g., commission)
            
        Returns:
            Total transaction costs
        """
        if not trades:
            return 0.0
        
        # Fixed costs
        fixed_costs = len(trades) * cost_per_trade
        
        # Variable costs (assume small bid-ask spread impact)
        variable_costs = sum(t.trade_value * 0.0005 for t in trades)  # 5 bps impact
        
        total_costs = fixed_costs + variable_costs
        
        logger.debug(f"Transaction costs: fixed=${fixed_costs:.2f}, "
                    f"variable=${variable_costs:.2f}, total=${total_costs:.2f}")
        
        return total_costs
    
    def simulate_execution(
        self,
        trades: List[Trade],
        execution_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Simulate trade execution and return realized weights.
        
        Args:
            trades: List of trades to execute
            execution_prices: Actual execution prices (if different from current)
            
        Returns:
            Dictionary of realized portfolio weights after execution
        """
        if execution_prices is None:
            execution_prices = {t.symbol: t.current_price for t in trades}
        
        # Calculate total portfolio value after trades
        total_value = 0.0
        position_values = {}
        
        for trade in trades:
            execution_price = execution_prices.get(trade.symbol, trade.current_price)
            executed_value = trade.quantity * execution_price
            
            if trade.symbol in position_values:
                position_values[trade.symbol] += executed_value
            else:
                position_values[trade.symbol] = executed_value
            
            total_value += abs(executed_value)
        
        # Calculate realized weights
        realized_weights = {}
        if total_value > 0:
            for symbol, value in position_values.items():
                realized_weights[symbol] = value / total_value
        
        logger.info(f"Simulated execution: {len(realized_weights)} positions, "
                   f"total value ${total_value:.2f}")
        
        return realized_weights
    
    def optimize_trade_order(self, trades: List[Trade]) -> List[Trade]:
        """Optimize the order of trade execution.
        
        Args:
            trades: List of trades to optimize
            
        Returns:
            Reordered list of trades for optimal execution
        """
        if not trades:
            return trades
        
        # Separate buys and sells
        buys = [t for t in trades if t.direction == 'BUY']
        sells = [t for t in trades if t.direction == 'SELL']
        
        # Sort sells by value (largest first) to free up capital
        sells.sort(key=lambda t: t.trade_value, reverse=True)
        
        # Sort buys by value (smallest first) to use freed capital efficiently
        buys.sort(key=lambda t: t.trade_value)
        
        # Execute sells first, then buys
        optimized_trades = sells + buys
        
        logger.debug(f"Optimized trade order: {len(sells)} sells first, then {len(buys)} buys")
        
        return optimized_trades