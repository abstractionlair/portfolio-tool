# Current Task: Implement Portfolio Data Structures

**Status**: COMPLETED  
**Assigned**: Claude Code  
**Priority**: High  
**GitHub Issue**: #1  

## Objective
Implement the core Portfolio and Position classes that will form the foundation of the portfolio management system.

## Requirements

### Position Class
Create a `Position` class that represents a single holding with the following features:

1. **Attributes**:
   - `symbol`: Stock/ETF ticker symbol
   - `quantity`: Number of shares (can be negative for shorts)
   - `cost_basis`: Average price paid per share
   - `purchase_date`: When position was acquired
   - `leverage_factor`: For leveraged ETFs (default 1.0)
   - `asset_class`: Equity, Bond, Commodity, etc.

2. **Methods**:
   - `market_value(current_price)`: Calculate current value
   - `unrealized_pnl(current_price)`: Calculate profit/loss
   - `add_shares(quantity, price)`: Add to position with proper cost basis averaging
   - `remove_shares(quantity)`: Remove shares (FIFO)
   - `get_exposures(fund_definitions)`: Return list of Exposure objects based on fund definition

### Portfolio Class
Create a `Portfolio` class that manages multiple positions:

1. **Attributes**:
   - `positions`: Dictionary of Position objects keyed by symbol
   - `cash`: Cash balance
   - `name`: Portfolio name/identifier
   - `last_update`: Timestamp of last update

2. **Methods**:
   - `add_position(position)`: Add or update position
   - `remove_position(symbol)`: Remove position entirely
   - `add_trade(symbol, quantity, price)`: Execute a trade
   - `get_weights()`: Calculate position weights
   - `get_exposure(asset_class)`: Calculate exposure by asset class
   - `calculate_total_exposures(fund_definitions)`: Aggregate true exposures across all positions
   - `total_value(prices_dict)`: Calculate total portfolio value
   - `to_dataframe()`: Export positions as pandas DataFrame
   - `from_csv(filepath)`: Import positions from CSV
   - `to_csv(filepath)`: Export positions to CSV

### CSV Format
Support importing/exporting portfolios with this format:
```csv
symbol,quantity,cost_basis,purchase_date,leverage_factor,asset_class
SPY,100,400.50,2024-01-15,1.0,Equity
TLT,200,95.25,2024-01-10,1.0,Bond
UPRO,50,75.00,2024-01-20,3.0,Equity
```

## Implementation Steps

1. Create `src/portfolio/position.py` with Position class
2. Create `src/portfolio/portfolio.py` with Portfolio class
3. Create comprehensive tests in `tests/test_portfolio.py`
4. Add example usage in `examples/portfolio_example.py`
5. Update the notebook to demonstrate portfolio functionality

## Key Considerations

- **Leverage Handling**: Positions in leveraged ETFs should properly track their leverage factor
- **Exposure Decomposition**: Consider how positions map to underlying exposures (e.g., RSSB = 100% equity + 100% bonds)
- **Short Positions**: Support negative quantities for short positions
- **Cost Basis**: Implement proper averaging when adding to positions
- **Type Safety**: Use type hints throughout
- **Validation**: Validate inputs (e.g., no zero prices, valid dates)
- **Edge Cases**: Handle empty portfolios, zero quantities, etc.

## Note on Exposure Model
While the full exposure decomposition system will be implemented in a separate task, the Position and Portfolio classes should be designed with this future enhancement in mind. Consider adding placeholder methods or structure that can be extended later.

## Test Cases to Include

1. Creating positions with various asset types
2. Adding/removing shares with cost basis calculation
3. Portfolio weight calculation with different position sizes
4. CSV import/export round trip
5. Handling leveraged positions correctly
6. Short position handling
7. Edge cases (empty portfolio, single position, etc.)

## Success Criteria

- [x] All tests pass
- [x] Code follows project style guide
- [x] Type hints on all public methods
- [x] Docstrings with examples
- [x] Example script demonstrates key features
- [x] CSV import/export works correctly

## Notes for Implementation

- Start with simple versions and iterate
- Focus on correctness over optimization initially
- Use `decimal.Decimal` for monetary calculations if precision is critical
- Consider using `@dataclass` for simple data containers
- Log important operations for debugging
