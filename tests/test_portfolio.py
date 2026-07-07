"""Tests for portfolio module."""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, mock_open
import pandas as pd

from src.portfolio.position import Position
from src.portfolio.portfolio import Portfolio


class TestPosition:
    """Tests for Position class."""
    
    @pytest.fixture
    def position(self):
        """Create a test position."""
        return Position(
            symbol="SPY",
            quantity=100,
            cost_basis=400.0,
            purchase_date=datetime(2024, 1, 15),
            leverage_factor=1.0,
            asset_class="Equity"
        )
    
    @pytest.fixture
    def leveraged_position(self):
        """Create a leveraged position."""
        return Position(
            symbol="UPRO",
            quantity=50,
            cost_basis=75.0,
            purchase_date=datetime(2024, 1, 20),
            leverage_factor=3.0,
            asset_class="Equity"
        )
    
    def test_position_creation(self, position):
        """Test creating a position."""
        assert position.symbol == "SPY"
        assert position.quantity == 100
        assert position.cost_basis == 400.0
        assert position.leverage_factor == 1.0
        assert position.asset_class == "Equity"
    
    def test_position_validation(self):
        """Test position validation."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            Position("", 100, 400.0, datetime.now())
        
        with pytest.raises(ValueError, match="Cost basis cannot be negative"):
            Position("SPY", 100, -400.0, datetime.now())
        
        with pytest.raises(ValueError, match="Leverage factor must be positive"):
            Position("SPY", 100, 400.0, datetime.now(), leverage_factor=-1.0)
    
    def test_symbol_normalization(self):
        """Test symbol normalization."""
        pos = Position("  spy  ", 100, 400.0, datetime.now())
        assert pos.symbol == "SPY"
    
    def test_market_value(self, position):
        """Test market value calculation."""
        assert position.market_value(410.0) == 41000.0
        assert position.market_value(390.0) == 39000.0
    
    def test_unrealized_pnl(self, position):
        """Test unrealized P&L calculation."""
        assert position.unrealized_pnl(410.0) == 1000.0
        assert position.unrealized_pnl(390.0) == -1000.0
        assert position.unrealized_pnl(400.0) == 0.0
    
    def test_add_shares(self, position):
        """Test adding shares with cost basis averaging."""
        # Original: 100 shares at $400 = $40,000
        # Add: 50 shares at $420 = $21,000
        # Total: 150 shares at $406.67 = $61,000
        position.add_shares(50, 420.0)
        
        assert position.quantity == 150
        assert abs(position.cost_basis - 406.6666666666667) < 1e-10
    
    def test_add_shares_validation(self, position):
        """Test add_shares validation."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            position.add_shares(50, -10.0)
    
    def test_remove_shares(self, position):
        """Test removing shares."""
        position.remove_shares(25)
        assert position.quantity == 75
        assert position.cost_basis == 400.0  # Cost basis unchanged
    
    def test_remove_shares_validation(self, position):
        """Test remove_shares validation."""
        with pytest.raises(ValueError, match="Cannot remove 150 shares"):
            position.remove_shares(150)
    
    def test_get_exposures_simple(self, position):
        """Test getting exposures without fund definitions."""
        exposures = position.get_exposures()
        assert len(exposures) == 1
        assert exposures[0].exposure_type.value == 'US_LARGE_EQUITY'  # Mapped from 'Equity' asset class
        assert exposures[0].amount == 40000.0  # 100 * 400 * 1.0
    
    def test_get_exposures_leveraged(self, leveraged_position):
        """Test getting exposures for leveraged position."""
        exposures = leveraged_position.get_exposures()
        assert len(exposures) == 1
        assert exposures[0].exposure_type.value == 'US_LARGE_EQUITY'  # Mapped from 'Equity' asset class
        assert exposures[0].amount == 11250.0  # 50 * 75 * 3.0 (leverage factor)
    
    def test_to_dict(self, position):
        """Test converting position to dictionary."""
        pos_dict = position.to_dict()
        assert pos_dict['symbol'] == 'SPY'
        assert pos_dict['quantity'] == 100
        assert pos_dict['cost_basis'] == 400.0
        assert pos_dict['leverage_factor'] == 1.0
        assert pos_dict['asset_class'] == 'Equity'
    
    def test_repr(self, position):
        """Test string representation."""
        repr_str = repr(position)
        assert "Position(symbol='SPY'" in repr_str
        assert "quantity=100" in repr_str
        assert "cost_basis=400.00" in repr_str
    
    def test_equality(self, position):
        """Test position equality."""
        other = Position("SPY", 100, 400.0, datetime(2024, 1, 15))
        assert position == other
        
        different = Position("QQQ", 100, 400.0, datetime(2024, 1, 15))
        assert position != different
    
    def test_short_position(self):
        """Test short position handling."""
        short_pos = Position("SPY", -100, 400.0, datetime.now())
        assert short_pos.quantity == -100
        assert short_pos.market_value(410.0) == -41000.0
        assert short_pos.unrealized_pnl(410.0) == -1000.0  # Loss on short


class TestPortfolio:
    """Tests for Portfolio class."""
    
    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio."""
        return Portfolio("Test Portfolio", cash=10000.0)
    
    @pytest.fixture
    def position_spy(self):
        """Create SPY position."""
        return Position("SPY", 100, 400.0, datetime(2024, 1, 15), asset_class="Equity")
    
    @pytest.fixture
    def position_qqq(self):
        """Create QQQ position."""
        return Position("QQQ", 50, 300.0, datetime(2024, 1, 10), asset_class="Equity")
    
    @pytest.fixture
    def prices(self):
        """Current market prices."""
        return {"SPY": 410.0, "QQQ": 310.0}
    
    def test_portfolio_creation(self, portfolio):
        """Test creating a portfolio."""
        assert portfolio.name == "Test Portfolio"
        assert portfolio.cash == 10000.0
        assert len(portfolio.positions) == 0
    
    def test_portfolio_validation(self):
        """Test portfolio validation."""
        with pytest.raises(ValueError, match="Cash cannot be negative"):
            Portfolio("Test", cash=-1000.0)
    
    def test_add_position(self, portfolio, position_spy):
        """Test adding a position."""
        portfolio.add_position(position_spy)
        assert len(portfolio.positions) == 1
        assert "SPY" in portfolio.positions
        assert portfolio.last_update is not None
    
    def test_add_position_merge(self, portfolio, position_spy):
        """Test merging positions with same symbol."""
        portfolio.add_position(position_spy)
        
        # Add another SPY position
        another_spy = Position("SPY", 50, 420.0, datetime(2024, 1, 20), asset_class="Equity")
        portfolio.add_position(another_spy)
        
        assert len(portfolio.positions) == 1
        assert portfolio.positions["SPY"].quantity == 150
        assert abs(portfolio.positions["SPY"].cost_basis - 406.6666666666667) < 1e-10
    
    def test_remove_position(self, portfolio, position_spy):
        """Test removing a position."""
        portfolio.add_position(position_spy)
        portfolio.remove_position("SPY")
        assert len(portfolio.positions) == 0
    
    def test_remove_position_not_found(self, portfolio):
        """Test removing non-existent position."""
        with pytest.raises(KeyError, match="Position AAPL not found"):
            portfolio.remove_position("AAPL")
    
    def test_add_trade(self, portfolio):
        """Test executing a trade."""
        portfolio.add_trade("SPY", 100, 400.0)
        
        assert len(portfolio.positions) == 1
        assert portfolio.positions["SPY"].quantity == 100
        assert portfolio.positions["SPY"].cost_basis == 400.0
        assert portfolio.cash == -30000.0  # 10000 - 40000
    
    def test_add_trade_validation(self, portfolio):
        """Test trade validation."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            portfolio.add_trade("SPY", 100, -400.0)
    
    def test_get_weights(self, portfolio, position_spy, position_qqq, prices):
        """Test calculating position weights."""
        portfolio.add_position(position_spy)
        portfolio.add_position(position_qqq)
        
        weights = portfolio.get_weights(prices)
        
        # SPY: 100 * 410 = 41000
        # QQQ: 50 * 310 = 15500
        # Total value: 56500 + 10000 (cash) = 66500
        assert abs(weights["SPY"] - 41000/66500) < 1e-10
        assert abs(weights["QQQ"] - 15500/66500) < 1e-10
    
    def test_get_weights_empty_portfolio(self, portfolio):
        """Test weights for empty portfolio."""
        weights = portfolio.get_weights({})
        assert weights == {}
    
    def test_get_exposure(self, portfolio, position_spy, prices):
        """Test calculating exposure by asset class."""
        portfolio.add_position(position_spy)
        exposure = portfolio.get_exposure("Equity", prices)
        assert exposure == 41000.0  # 100 * 410 * 1.0 (leverage factor)
    
    def test_calculate_total_exposures(self, portfolio, position_spy, position_qqq):
        """Test calculating total exposures."""
        portfolio.add_position(position_spy)
        portfolio.add_position(position_qqq)
        
        exposures = portfolio.calculate_total_exposures()
        from src.portfolio.exposures import ExposureType
        assert exposures[ExposureType.US_LARGE_EQUITY] == 55000.0  # 40000 + 15000
    
    def test_total_value(self, portfolio, position_spy, position_qqq, prices):
        """Test calculating total portfolio value."""
        portfolio.add_position(position_spy)
        portfolio.add_position(position_qqq)
        
        total = portfolio.total_value(prices)
        assert total == 66500.0  # 41000 + 15500 + 10000 (cash)
    
    def test_to_dataframe(self, portfolio, position_spy, position_qqq):
        """Test converting to DataFrame."""
        portfolio.add_position(position_spy)
        portfolio.add_position(position_qqq)
        
        df = portfolio.to_dataframe()
        assert df.shape[0] == 2
        assert set(df['symbol']) == {'SPY', 'QQQ'}
    
    def test_to_dataframe_empty(self, portfolio):
        """Test converting empty portfolio to DataFrame."""
        df = portfolio.to_dataframe()
        assert df.shape[0] == 0
        assert 'symbol' in df.columns
    
    def test_csv_round_trip(self, portfolio, position_spy, position_qqq):
        """Test CSV export and import."""
        portfolio.add_position(position_spy)
        portfolio.add_position(position_qqq)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export to CSV
            portfolio.to_csv(temp_path)
            
            # Create new portfolio and import
            new_portfolio = Portfolio("New Portfolio")
            new_portfolio.from_csv(temp_path)
            
            assert len(new_portfolio.positions) == 2
            assert "SPY" in new_portfolio.positions
            assert "QQQ" in new_portfolio.positions
            assert new_portfolio.positions["SPY"].quantity == 100
            assert new_portfolio.positions["QQQ"].quantity == 50
            
        finally:
            os.unlink(temp_path)
    
    def test_csv_import_missing_columns(self, portfolio):
        """Test CSV import with missing columns."""
        csv_content = "symbol,quantity\nSPY,100\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Required column 'cost_basis' not found"):
                portfolio.from_csv(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_csv_import_file_not_found(self, portfolio):
        """Test CSV import with non-existent file."""
        with pytest.raises(FileNotFoundError):
            portfolio.from_csv("nonexistent.csv")
    
    def test_get_summary(self, portfolio, position_spy, position_qqq, prices):
        """Test getting portfolio summary."""
        portfolio.add_position(position_spy)
        portfolio.add_position(position_qqq)
        
        summary = portfolio.get_summary(prices)
        
        assert summary['total_value'] == 66500.0
        assert summary['total_cost'] == 65000.0  # 40000 + 15000 + 10000 (cash)
        assert summary['cash'] == 10000.0
        assert summary['positions_count'] == 2
        assert summary['total_unrealized_pnl'] == 1500.0  # 1000 + 500
        assert abs(summary['total_return_pct'] - 2.727272727272727) < 1e-10
    
    def test_repr(self, portfolio):
        """Test string representation."""
        repr_str = repr(portfolio)
        assert "Portfolio(name='Test Portfolio'" in repr_str
        assert "positions=0" in repr_str
        assert "cash=$10000.00" in repr_str
    
    def test_len(self, portfolio, position_spy):
        """Test portfolio length."""
        assert len(portfolio) == 0
        portfolio.add_position(position_spy)
        assert len(portfolio) == 1
    
    def test_leveraged_position_handling(self, portfolio):
        """Test handling of leveraged positions."""
        leveraged_pos = Position("UPRO", 50, 75.0, datetime.now(), leverage_factor=3.0, asset_class="Equity")
        portfolio.add_position(leveraged_pos)
        
        exposure = portfolio.get_exposure("Equity", {"UPRO": 80.0})
        assert exposure == 12000.0  # 50 * 80 * 3.0
    
    def test_short_position_handling(self, portfolio):
        """Test handling of short positions."""
        short_pos = Position("SPY", -100, 400.0, datetime.now(), asset_class="Equity")
        portfolio.add_position(short_pos)
        
        total_value = portfolio.total_value({"SPY": 410.0})
        assert total_value == -31000.0  # 10000 - 41000
    
    def test_mixed_long_short_portfolio(self, portfolio):
        """Test portfolio with both long and short positions."""
        long_pos = Position("SPY", 100, 400.0, datetime.now(), asset_class="Equity")
        short_pos = Position("QQQ", -50, 300.0, datetime.now(), asset_class="Equity")
        
        portfolio.add_position(long_pos)
        portfolio.add_position(short_pos)
        
        total_value = portfolio.total_value({"SPY": 410.0, "QQQ": 310.0})
        # SPY: 100 * 410 = 41000 (long)
        # QQQ: -50 * 310 = -15500 (short)
        # Cash: 10000
        # Total: 41000 + (-15500) + 10000 = 35500
        assert total_value == 35500.0
        
        weights = portfolio.get_weights({"SPY": 410.0, "QQQ": 310.0})
        assert weights["SPY"] > 0
        assert weights["QQQ"] < 0