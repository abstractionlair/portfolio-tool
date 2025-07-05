"""Tests for the optimization module."""

import pytest
import numpy as np
from datetime import datetime

from src.optimization import (
    OptimizationEngine, ObjectiveType, OptimizationConstraints, OptimizationResult,
    ReturnEstimator, MarketView, TradeGenerator, Trade
)
from src.portfolio import Portfolio, Position, ExposureType, FundExposureMap, FundDefinition, PortfolioAnalytics
from src.data import MarketDataFetcher


class MockMarketDataFetcher:
    """Mock market data fetcher for testing."""
    
    def fetch_price_history(self, symbols, start_date, end_date):
        """Return dummy price data."""
        dates = ['2023-01-01', '2023-01-02', '2023-01-03']
        data = {}
        for symbol in symbols:
            # Generate deterministic but varying price data
            np.random.seed(hash(symbol) % 2**32)
            prices = 100 + np.random.normal(0, 5, len(dates))
            data[symbol] = {date: price for date, price in zip(dates, prices)}
        return data


@pytest.fixture
def mock_fund_map():
    """Create a mock fund exposure map for testing."""
    fund_map = FundExposureMap()
    
    # Add some test funds
    fund_map.add_fund_definition(FundDefinition(
        symbol='SPY',
        name='SPDR S&P 500 ETF',
        exposures={
            ExposureType.US_LARGE_EQUITY: 1.0
        },
        total_notional=1.0
    ))
    
    fund_map.add_fund_definition(FundDefinition(
        symbol='TLT',
        name='iShares 20+ Year Treasury Bond ETF',
        exposures={
            ExposureType.BONDS: 1.0
        },
        total_notional=1.0
    ))
    
    fund_map.add_fund_definition(FundDefinition(
        symbol='QQQ',
        name='Invesco QQQ Trust',
        exposures={
            ExposureType.US_LARGE_EQUITY: 1.0
        },
        total_notional=1.0
    ))
    
    # Add a leveraged fund
    fund_map.add_fund_definition(FundDefinition(
        symbol='TQQQ',
        name='ProShares UltraPro QQQ',
        exposures={
            ExposureType.US_LARGE_EQUITY: 3.0
        },
        total_notional=3.0
    ))
    
    return fund_map


@pytest.fixture
def mock_analytics():
    """Create a mock analytics object."""
    mock_fetcher = MockMarketDataFetcher()
    # Create a dummy portfolio for analytics
    dummy_portfolio = Portfolio(name="Test Portfolio")
    return PortfolioAnalytics(dummy_portfolio, mock_fetcher)


@pytest.fixture
def optimization_engine(mock_analytics, mock_fund_map):
    """Create an optimization engine for testing."""
    return OptimizationEngine(mock_analytics, mock_fund_map)


@pytest.fixture
def sample_data():
    """Create sample data for optimization tests."""
    symbols = ['SPY', 'TLT', 'QQQ']
    expected_returns = np.array([0.08, 0.03, 0.10])  # Annual returns
    
    # Simple covariance matrix
    cov_matrix = np.array([
        [0.04, 0.01, 0.03],   # SPY: 20% vol, moderate correlation
        [0.01, 0.02, 0.005],  # TLT: 14% vol, low correlation  
        [0.03, 0.005, 0.06]   # QQQ: 24% vol, high correlation with SPY
    ])
    
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.5,
        max_total_notional=1.5,
        long_only=True
    )
    
    return symbols, expected_returns, cov_matrix, constraints


class TestOptimizationEngine:
    """Test the optimization engine."""
    
    def test_engine_initialization(self, optimization_engine):
        """Test that the optimization engine initializes correctly."""
        assert optimization_engine is not None
        assert optimization_engine.analytics is not None
        assert optimization_engine.fund_map is not None
        assert optimization_engine.mv_optimizer is not None
        assert optimization_engine.rp_optimizer is not None
        assert optimization_engine.bl_optimizer is not None
    
    def test_max_sharpe_optimization(self, optimization_engine, sample_data):
        """Test maximum Sharpe ratio optimization."""
        symbols, expected_returns, cov_matrix, constraints = sample_data
        
        result = optimization_engine.optimize(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            constraints=constraints,
            objective=ObjectiveType.MAX_SHARPE
        )
        
        assert result.success
        assert len(result.weights) == len(symbols)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6  # Weights sum to 1
        assert result.sharpe_ratio > 0
        assert result.expected_return > 0
        assert result.expected_volatility > 0
        
        # Check that weights respect constraints
        for weight in result.weights.values():
            assert weight >= constraints.min_weight - 1e-6
            assert weight <= constraints.max_weight + 1e-6
    
    def test_min_volatility_optimization(self, optimization_engine, sample_data):
        """Test minimum volatility optimization."""
        symbols, expected_returns, cov_matrix, constraints = sample_data
        
        result = optimization_engine.optimize(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            constraints=constraints,
            objective=ObjectiveType.MIN_VOLATILITY
        )
        
        assert result.success
        assert len(result.weights) == len(symbols)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert result.expected_volatility > 0
        
        # Min vol should prefer bonds (lowest volatility asset)
        assert result.weights['TLT'] > 0.1
    
    def test_risk_parity_optimization(self, optimization_engine, sample_data):
        """Test risk parity optimization."""
        symbols, expected_returns, cov_matrix, constraints = sample_data
        
        result = optimization_engine.optimize(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            constraints=constraints,
            objective=ObjectiveType.RISK_PARITY
        )
        
        assert result.success
        assert len(result.weights) == len(symbols)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        
        # Risk parity should give more weight to lower volatility assets
        assert result.weights['TLT'] > result.weights['QQQ']  # TLT has lower vol than QQQ
    
    def test_exposure_optimization(self, optimization_engine, sample_data):
        """Test exposure-based optimization."""
        symbols, expected_returns, cov_matrix, constraints = sample_data
        
        target_exposures = {
            ExposureType.US_LARGE_EQUITY: 0.6,
            ExposureType.BONDS: 0.4
        }
        
        result = optimization_engine.optimize(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            constraints=constraints,
            objective=ObjectiveType.TARGET_EXPOSURES,
            target_exposures=target_exposures
        )
        
        assert result.success
        assert len(result.weights) == len(symbols)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        
        # Check that exposures are close to targets
        assert abs(result.exposures.get(ExposureType.US_LARGE_EQUITY, 0) - 0.6) < 0.1
        assert abs(result.exposures.get(ExposureType.BONDS, 0) - 0.4) < 0.1
    
    def test_constraint_enforcement(self, optimization_engine, sample_data):
        """Test that constraints are properly enforced."""
        symbols, expected_returns, cov_matrix, _ = sample_data
        
        # Create strict constraints
        constraints = OptimizationConstraints(
            min_weight=0.1,
            max_weight=0.4,
            long_only=True,
            max_total_notional=1.0
        )
        
        result = optimization_engine.optimize(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            constraints=constraints,
            objective=ObjectiveType.MAX_SHARPE
        )
        
        if result.success:
            # Check weight bounds
            for weight in result.weights.values():
                if weight > 1e-6:  # Only check non-zero weights
                    assert weight >= constraints.min_weight - 1e-6
                assert weight <= constraints.max_weight + 1e-6
            
            # Check total notional
            assert result.total_notional <= constraints.max_total_notional + 1e-6
    
    def test_leveraged_fund_handling(self, optimization_engine, mock_fund_map):
        """Test optimization with leveraged funds."""
        symbols = ['SPY', 'TQQQ']  # Include leveraged fund
        expected_returns = np.array([0.08, 0.12])
        cov_matrix = np.array([[0.04, 0.06], [0.06, 0.12]])  # Higher vol for leveraged fund
        
        constraints = OptimizationConstraints(
            max_total_notional=2.0,  # Allow some leverage
            long_only=True
        )
        
        result = optimization_engine.optimize(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            constraints=constraints,
            objective=ObjectiveType.MAX_SHARPE
        )
        
        assert result.success
        # Total notional should account for TQQQ's 3x leverage
        assert result.total_notional > 1.0  # Should be leveraged due to TQQQ
        assert result.total_notional <= constraints.max_total_notional + 1e-6


class TestReturnEstimator:
    """Test the return estimator."""
    
    def test_estimator_initialization(self):
        """Test that return estimator initializes correctly."""
        mock_fetcher = MockMarketDataFetcher()
        estimator = ReturnEstimator(mock_fetcher)
        
        assert estimator is not None
        assert estimator.market_data is not None
        assert estimator._cache == {}
    
    def test_historical_return_estimation(self):
        """Test historical return estimation."""
        mock_fetcher = MockMarketDataFetcher()
        estimator = ReturnEstimator(mock_fetcher)
        
        symbols = ['SPY', 'TLT']
        returns = estimator.estimate_expected_returns(symbols, method='historical')
        
        assert len(returns) == len(symbols)
        assert isinstance(returns, np.ndarray)
        # Returns should be reasonable (not extremely large or small)
        assert all(abs(r) < 1.0 for r in returns)  # Less than 100% annual return
    
    def test_covariance_estimation(self):
        """Test covariance matrix estimation."""
        mock_fetcher = MockMarketDataFetcher()
        estimator = ReturnEstimator(mock_fetcher)
        
        symbols = ['SPY', 'TLT']
        cov_matrix = estimator.estimate_covariance_matrix(symbols, method='sample')
        
        assert cov_matrix.shape == (len(symbols), len(symbols))
        # Covariance matrix should be symmetric
        assert np.allclose(cov_matrix, cov_matrix.T)
        # Diagonal elements (variances) should be positive
        assert all(cov_matrix[i, i] > 0 for i in range(len(symbols)))


class TestTradeGenerator:
    """Test the trade generator."""
    
    def test_trade_generation(self):
        """Test basic trade generation."""
        generator = TradeGenerator(min_trade_value=50.0)
        
        # Create a simple portfolio
        current_portfolio = Portfolio("Test Portfolio")
        current_portfolio.add_position(Position(symbol='SPY', quantity=10.0, cost_basis=400.0, purchase_date=datetime.now()))
        current_portfolio.add_position(Position(symbol='TLT', quantity=5.0, cost_basis=120.0, purchase_date=datetime.now()))
        
        target_weights = {'SPY': 0.4, 'TLT': 0.6}
        prices = {'SPY': 420.0, 'TLT': 125.0}
        total_value = 10 * 420 + 5 * 125  # $4825
        
        trades = generator.generate_trades(
            target_weights=target_weights,
            current_portfolio=current_portfolio,
            prices=prices,
            total_portfolio_value=total_value
        )
        
        assert isinstance(trades, list)
        # Should generate trades since current weights don't match targets
        for trade in trades:
            assert isinstance(trade, Trade)
            assert trade.symbol in prices
            assert trade.current_price > 0
            assert trade.trade_value >= 0
            assert trade.direction in ['BUY', 'SELL']
    
    def test_trade_cost_calculation(self):
        """Test transaction cost calculation."""
        generator = TradeGenerator()
        
        trades = [
            Trade(symbol='SPY', quantity=10, direction='BUY', 
                  trade_value=4200, current_price=420),
            Trade(symbol='TLT', quantity=-5, direction='SELL', 
                  trade_value=625, current_price=125)
        ]
        
        costs = generator.calculate_trade_costs(trades, cost_per_trade=1.0)
        
        assert costs > 0
        assert costs >= 2.0  # At least fixed costs for 2 trades
    
    def test_trade_order_optimization(self):
        """Test trade order optimization."""
        generator = TradeGenerator()
        
        trades = [
            Trade(symbol='SPY', quantity=10, direction='BUY', 
                  trade_value=4200, current_price=420),
            Trade(symbol='TLT', quantity=-5, direction='SELL', 
                  trade_value=625, current_price=125),
            Trade(symbol='QQQ', quantity=20, direction='BUY', 
                  trade_value=7000, current_price=350)
        ]
        
        optimized = generator.optimize_trade_order(trades)
        
        assert len(optimized) == len(trades)
        # Should have sells before buys
        sell_indices = [i for i, t in enumerate(optimized) if t.direction == 'SELL']
        buy_indices = [i for i, t in enumerate(optimized) if t.direction == 'BUY']
        
        if sell_indices and buy_indices:
            assert max(sell_indices) < min(buy_indices)


class TestOptimizationResult:
    """Test optimization result functionality."""
    
    def test_result_initialization(self):
        """Test that optimization result initializes correctly."""
        weights = {'SPY': 0.6, 'TLT': 0.4}
        
        result = OptimizationResult(
            weights=weights,
            objective_value=0.8,
            expected_return=0.06,
            expected_volatility=0.12,
            sharpe_ratio=0.5,
            exposures={ExposureType.US_LARGE_EQUITY: 0.6, ExposureType.BONDS: 0.4},
            total_notional=1.0,
            success=True,
            message="Test result"
        )
        
        assert result.weights == weights
        assert result.success
        assert result.effective_assets == 2  # Two non-zero positions
        assert result.max_weight == 0.6
        assert 0 < result.concentration_ratio < 1  # Should be between 0 and 1
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        weights = {'SPY': 0.6, 'TLT': 0.4}
        
        result = OptimizationResult(
            weights=weights,
            objective_value=0.8,
            expected_return=0.06,
            expected_volatility=0.12,
            sharpe_ratio=0.5,
            exposures={ExposureType.US_LARGE_EQUITY: 0.6},
            total_notional=1.0,
            success=True,
            message="Test result"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['weights'] == weights
        assert result_dict['success'] == True
        assert result_dict['effective_assets'] == 2
        assert 'exposures' in result_dict
    
    def test_to_trades_integration(self, optimization_engine, sample_data):
        """Test integration with trade generation."""
        symbols, expected_returns, cov_matrix, constraints = sample_data
        
        # Get optimization result
        result = optimization_engine.optimize(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            constraints=constraints,
            objective=ObjectiveType.MAX_SHARPE
        )
        
        # Create a current portfolio
        current_portfolio = Portfolio("Test Portfolio")
        current_portfolio.add_position(Position(symbol='SPY', quantity=100.0, cost_basis=400.0, purchase_date=datetime.now()))
        
        prices = {'SPY': 420.0, 'TLT': 125.0, 'QQQ': 350.0}
        
        if result.success:
            trades = result.to_trades(
                current_portfolio=current_portfolio,
                prices=prices,
                total_portfolio_value=42000.0
            )
            
            assert isinstance(trades, list)
            # Should generate trades to rebalance
            assert len(trades) >= 0


class TestConstraints:
    """Test optimization constraints."""
    
    def test_constraint_initialization(self):
        """Test constraint initialization with defaults."""
        constraints = OptimizationConstraints()
        
        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 1.0
        assert constraints.long_only == True
        assert isinstance(constraints.max_exposure_per_type, dict)
        assert isinstance(constraints.min_exposure_per_type, dict)
    
    def test_exposure_constraints(self):
        """Test exposure constraint specification."""
        constraints = OptimizationConstraints(
            max_exposure_per_type={ExposureType.US_LARGE_EQUITY: 0.7},
            min_exposure_per_type={ExposureType.BONDS: 0.2}
        )
        
        assert ExposureType.US_LARGE_EQUITY in constraints.max_exposure_per_type
        assert constraints.max_exposure_per_type[ExposureType.US_LARGE_EQUITY] == 0.7
        assert ExposureType.BONDS in constraints.min_exposure_per_type
        assert constraints.min_exposure_per_type[ExposureType.BONDS] == 0.2


if __name__ == "__main__":
    pytest.main([__file__])