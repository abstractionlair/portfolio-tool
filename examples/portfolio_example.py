"""Example usage of the Portfolio and Position classes."""

from datetime import datetime
import tempfile
import os
from src.portfolio import Portfolio, Position
from src.data.market_data import MarketDataFetcher


def main():
    """Demonstrate portfolio functionality."""
    print("=== Portfolio Management Example ===\n")
    
    # Create a new portfolio
    portfolio = Portfolio("My Investment Portfolio", cash=50000.0)
    print(f"Created portfolio: {portfolio}")
    
    # Create some positions
    print("\n1. Creating positions...")
    positions = [
        Position("SPY", 100, 400.0, datetime(2024, 1, 15), asset_class="Equity"),
        Position("QQQ", 50, 300.0, datetime(2024, 1, 10), asset_class="Equity"),
        Position("TLT", 200, 95.0, datetime(2024, 1, 20), asset_class="Bond"),
        Position("UPRO", 25, 75.0, datetime(2024, 1, 25), leverage_factor=3.0, asset_class="Equity"),
    ]
    
    for pos in positions:
        portfolio.add_position(pos)
        print(f"Added: {pos}")
    
    # Show portfolio summary
    print(f"\nPortfolio now contains {len(portfolio)} positions")
    print(f"Cash balance: ${portfolio.cash:.2f}")
    
    # Fetch current market data
    print("\n2. Fetching current market data...")
    fetcher = MarketDataFetcher()
    symbols = ["SPY", "QQQ", "TLT", "UPRO"]
    
    try:
        market_data = fetcher.fetch_current_prices(symbols)
        if hasattr(market_data, 'to_dict'):
            current_prices = market_data.to_dict()
        else:
            current_prices = dict(market_data)
        print("Current prices:")
        for symbol, price in current_prices.items():
            print(f"  {symbol}: ${price:.2f}")
    except Exception as e:
        print(f"Error fetching market data: {e}")
        # Use mock prices for demo
        current_prices = {"SPY": 410.0, "QQQ": 310.0, "TLT": 98.0, "UPRO": 82.0}
        print("Using mock prices for demo:", current_prices)
    
    # Calculate portfolio metrics
    print("\n3. Portfolio Analysis...")
    
    # Total value
    total_value = portfolio.total_value(current_prices)
    print(f"Total portfolio value: ${total_value:.2f}")
    
    # Position weights
    weights = portfolio.get_weights(current_prices)
    print("Position weights:")
    for symbol, weight in weights.items():
        print(f"  {symbol}: {weight:.1%}")
    
    # Exposures by asset class
    exposures = portfolio.calculate_total_exposures()
    print("Total exposures by asset class:")
    for asset_class, exposure in exposures.items():
        print(f"  {asset_class}: ${exposure:.2f}")
    
    # Individual position analysis
    print("\n4. Individual Position Analysis...")
    for symbol, position in portfolio.positions.items():
        current_price = current_prices.get(symbol, position.cost_basis)
        market_value = position.market_value(current_price)
        unrealized_pnl = position.unrealized_pnl(current_price)
        pnl_pct = (unrealized_pnl / (position.quantity * position.cost_basis)) * 100
        
        print(f"{symbol}:")
        print(f"  Quantity: {position.quantity}")
        print(f"  Cost Basis: ${position.cost_basis:.2f}")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Market Value: ${market_value:.2f}")
        print(f"  Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_pct:.1f}%)")
        print(f"  Asset Class: {position.asset_class}")
        print(f"  Leverage Factor: {position.leverage_factor}x")
        print()
    
    # Portfolio summary
    summary = portfolio.get_summary(current_prices)
    print("5. Portfolio Summary:")
    print(f"  Total Value: ${summary['total_value']:.2f}")
    print(f"  Total Cost: ${summary['total_cost']:.2f}")
    print(f"  Cash: ${summary['cash']:.2f}")
    print(f"  Positions: {summary['positions_count']}")
    print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
    print(f"  Total Return: {summary['total_return_pct']:.1f}%")
    
    # Execute a trade
    print("\n6. Executing a trade...")
    print("Buying 50 more shares of SPY at current price...")
    portfolio.add_trade("SPY", 50, current_prices["SPY"])
    print(f"New SPY position: {portfolio.positions['SPY']}")
    print(f"Updated cash balance: ${portfolio.cash:.2f}")
    
    # CSV export/import example
    print("\n7. CSV Export/Import Example...")
    
    # Export to CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        portfolio.to_csv(csv_path)
        print(f"Exported portfolio to: {csv_path}")
        
        # Show CSV content
        with open(csv_path, 'r') as f:
            print("CSV content:")
            print(f.read())
        
        # Import to new portfolio
        new_portfolio = Portfolio("Imported Portfolio")
        new_portfolio.from_csv(csv_path)
        print(f"Imported portfolio: {new_portfolio}")
        print(f"Imported {len(new_portfolio)} positions")
        
    finally:
        os.unlink(csv_path)
    
    # DataFrame export
    print("\n8. DataFrame Export...")
    df = portfolio.to_dataframe()
    print("Portfolio as DataFrame:")
    print(df.to_string(index=False))
    
    # Demonstrate position modifications
    print("\n9. Position Modifications...")
    spy_position = portfolio.positions["SPY"]
    print(f"Original SPY position: {spy_position}")
    
    # Add more shares
    spy_position.add_shares(25, 415.0)
    print(f"After adding 25 shares at $415: {spy_position}")
    
    # Remove some shares
    spy_position.remove_shares(25)
    print(f"After removing 25 shares: {spy_position}")
    
    # Demonstrate leveraged position exposure
    print("\n10. Leveraged Position Exposure...")
    upro_position = portfolio.positions["UPRO"]
    print(f"UPRO position: {upro_position}")
    
    upro_exposures = upro_position.get_exposures()
    print("UPRO exposures:")
    for exposure in upro_exposures:
        print(f"  Asset Class: {exposure['asset_class']}")
        print(f"  Leverage Factor: {exposure['exposure']}x")
        print(f"  Notional Exposure: ${exposure['notional']:.2f}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()