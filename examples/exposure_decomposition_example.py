"""Example usage of the Fund Exposure Decomposition System."""

from datetime import datetime
import os
from src.portfolio import Portfolio, Position
from src.portfolio.exposures import (
    ExposureType, Exposure, FundDefinition, FundExposureMap, ExposureCalculator
)
from src.portfolio.return_replicator import ReturnReplicator, HAS_SKLEARN
from src.data.market_data import MarketDataFetcher


def main():
    """Demonstrate fund exposure decomposition functionality."""
    print("=== Fund Exposure Decomposition Example ===\n")
    
    # 1. Create simple fund definitions
    print("1. Creating Fund Definitions...")
    
    # Create a fund exposure map
    fund_map = FundExposureMap()
    
    # Add some fund definitions
    fund_definitions = [
        FundDefinition(
            symbol="SPY",
            name="SPDR S&P 500 ETF",
            exposures={ExposureType.US_LARGE_EQUITY: 1.0},
            total_notional=1.0,
            category="Equity"
        ),
        FundDefinition(
            symbol="RSSB", 
            name="Return Stacked US Stocks & Bonds ETF",
            exposures={
                ExposureType.US_LARGE_EQUITY: 1.0,
                ExposureType.BONDS: 1.0
            },
            total_notional=2.0,
            category="Return Stacked"
        ),
        FundDefinition(
            symbol="RSST",
            name="Return Stacked US Stocks & Managed Futures ETF", 
            exposures={
                ExposureType.US_LARGE_EQUITY: 1.0,
                ExposureType.MANAGED_FUTURES: 0.5
            },
            total_notional=1.5,
            category="Return Stacked"
        ),
        FundDefinition(
            symbol="AGG",
            name="iShares Core US Aggregate Bond ETF",
            exposures={ExposureType.BONDS: 1.0},
            total_notional=1.0,
            category="Bond"
        )
    ]
    
    for fund_def in fund_definitions:
        fund_map.add_fund_definition(fund_def)
        print(f"Added: {fund_def.symbol} - {fund_def.name}")
        print(f"  Exposures: {[(exp.value, amt) for exp, amt in fund_def.exposures.items()]}")
        print(f"  Total Notional: {fund_def.total_notional}x")
        print()
    
    # 2. Load real fund universe data if available
    print("2. Loading Real Fund Universe Data...")
    fund_universe_path = "data/fund_universe.yaml"
    if os.path.exists(fund_universe_path):
        try:
            real_fund_map = FundExposureMap(fund_universe_path)
            print(f"Loaded {len(real_fund_map)} funds from {fund_universe_path}")
            
            # Show some examples
            available_symbols = real_fund_map.get_available_symbols()
            for symbol in available_symbols[:3]:  # Show first 3
                fund_def = real_fund_map.get_fund_definition(symbol)
                print(f"  {symbol}: {fund_def.name}")
                print(f"    Exposures: {[(exp.value, amt) for exp, amt in fund_def.exposures.items()]}")
            
            # Use the real fund map for the rest of the example
            fund_map = real_fund_map
            
        except Exception as e:
            print(f"Error loading real fund data: {e}")
            print("Using synthetic fund definitions")
    else:
        print("Real fund universe data not found, using synthetic definitions")
    
    print()
    
    # 3. Create a portfolio with various funds
    print("3. Creating Portfolio with Mixed Funds...")
    
    portfolio = Portfolio("Exposure Demo Portfolio", cash=25000.0)
    
    # Add positions with different exposure profiles
    positions = [
        Position("SPY", 100, 420.0, datetime(2024, 12, 1), asset_class="Equity"),
        Position("AGG", 200, 95.0, datetime(2024, 12, 1), asset_class="Bond"),
    ]
    
    # Add Return Stacked positions if available
    if "RSSB" in fund_map:
        positions.append(Position("RSSB", 50, 25.0, datetime(2024, 12, 1), asset_class="Equity"))
    if "RSST" in fund_map:
        positions.append(Position("RSST", 30, 26.0, datetime(2024, 12, 1), asset_class="Equity"))
    
    # Add any real funds from the data
    available_symbols = fund_map.get_available_symbols()
    if available_symbols:
        # Add a position in the first available fund
        real_symbol = available_symbols[0]
        positions.append(Position(real_symbol, 20, 50.0, datetime(2024, 12, 1), asset_class="Other"))
    
    for pos in positions:
        portfolio.add_position(pos)
        print(f"Added: {pos}")
    
    print(f"\nPortfolio contains {len(portfolio)} positions")
    print(f"Cash: ${portfolio.cash:,.2f}")
    
    # 4. Calculate exposures
    print("\n4. Calculating Portfolio Exposures...")
    
    # Create mock current prices
    current_prices = {}
    for position in portfolio.positions.values():
        current_prices[position.symbol] = position.cost_basis * 1.05  # 5% gain
    
    print("Current prices:")
    for symbol, price in current_prices.items():
        print(f"  {symbol}: ${price:.2f}")
    
    # Calculate exposures using the exposure calculator
    calculator = ExposureCalculator(fund_map)
    total_exposures = calculator.calculate_portfolio_exposures(portfolio, current_prices)
    
    print("\nTotal Portfolio Exposures:")
    total_exposure_value = sum(abs(amount) for amount in total_exposures.values())
    
    for exposure_type, amount in sorted(total_exposures.items(), key=lambda x: abs(x[1]), reverse=True):
        percentage = (amount / total_exposure_value) * 100 if total_exposure_value > 0 else 0
        print(f"  {exposure_type.value}: ${amount:,.2f} ({percentage:.1f}%)")
    
    print(f"\nTotal Exposure Value: ${total_exposure_value:,.2f}")
    
    # 5. Compare with simple asset class breakdown
    print("\n5. Comparison with Simple Asset Class Breakdown...")
    
    simple_exposures = portfolio.calculate_total_exposures(None, current_prices)
    print("Simple asset class exposures (without fund decomposition):")
    for exposure_type, amount in simple_exposures.items():
        print(f"  {exposure_type.value}: ${amount:,.2f}")
    
    # 6. Individual position analysis
    print("\n6. Individual Position Exposure Analysis...")
    
    for symbol, position in portfolio.positions.items():
        current_price = current_prices[symbol]
        market_value = position.market_value(current_price)
        
        print(f"\n{symbol} ({market_value:,.0f} market value):")
        
        # Get exposures for this position
        position_exposures = calculator.calculate_position_exposures(position, current_price)
        
        for exposure in position_exposures:
            leverage = exposure.amount / market_value if market_value != 0 else 0
            print(f"  {exposure.exposure_type.value}: ${exposure.amount:,.2f} ({leverage:.1f}x leverage)")
    
    # 7. Fund definition examples
    print("\n7. Fund Definition Details...")
    
    for symbol in portfolio.positions.keys():
        fund_def = fund_map.get_fund_definition(symbol)
        if fund_def:
            print(f"\n{symbol} - {fund_def.name}")
            print(f"  Category: {fund_def.category}")
            print(f"  Total Notional: {fund_def.total_notional}x")
            print("  Stated Exposures:")
            for exp_type, amount in fund_def.exposures.items():
                print(f"    {exp_type.value}: {amount:.2f}")
        else:
            print(f"\n{symbol} - No fund definition available (using asset class mapping)")
    
    # 8. Return replication validation (if sklearn available)
    print("\n8. Return Replication Validation...")
    
    if HAS_SKLEARN:
        print("Sklearn available - return replication analysis possible")
        
        try:
            fetcher = MarketDataFetcher()
            replicator = ReturnReplicator(fetcher)
            
            # Try to validate one of the funds
            validation_symbols = [symbol for symbol in portfolio.positions.keys() if fund_map.get_fund_definition(symbol)]
            
            if validation_symbols:
                test_symbol = validation_symbols[0]
                fund_def = fund_map.get_fund_definition(test_symbol)
                
                print(f"Attempting to validate {test_symbol}...")
                
                try:
                    # This would normally run the full validation
                    # For demo purposes, just show the structure
                    print(f"  Fund: {fund_def.name}")
                    print(f"  Stated exposures: {list(fund_def.exposures.keys())}")
                    print("  Note: Full validation requires historical data and can take time")
                    
                except Exception as e:
                    print(f"  Validation failed: {e}")
            else:
                print("No funds available for validation")
                
        except Exception as e:
            print(f"Error setting up return replication: {e}")
    else:
        print("Sklearn not available - install with 'pip install scikit-learn' for return replication")
    
    # 9. Save fund definitions
    print("\n9. Saving Fund Definitions...")
    
    # Save the current fund map to a file
    output_path = "temp_fund_definitions.yaml"
    try:
        fund_map.save_definitions(output_path, 'yaml')
        print(f"Saved {len(fund_map)} fund definitions to {output_path}")
        
        # Show a few lines of the saved file
        with open(output_path, 'r') as f:
            lines = f.readlines()[:10]  # First 10 lines
            print("Sample content:")
            for line in lines:
                print(f"  {line.strip()}")
        
        # Clean up
        os.unlink(output_path)
        print(f"Cleaned up {output_path}")
        
    except Exception as e:
        print(f"Error saving definitions: {e}")
    
    # 10. Summary
    print("\n10. Summary...")
    
    total_value = portfolio.total_value(current_prices)
    print(f"Portfolio Total Value: ${total_value:,.2f}")
    print(f"Total Exposure Value: ${total_exposure_value:,.2f}")
    print(f"Exposure Ratio: {total_exposure_value / (total_value - portfolio.cash):.2f}x")
    
    print("\nKey Benefits of Exposure Decomposition:")
    print("- Reveals true underlying exposures of complex funds")
    print("- Enables proper portfolio optimization with leveraged funds")
    print("- Identifies concentration risks across different fund structures")
    print("- Supports return attribution analysis")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()