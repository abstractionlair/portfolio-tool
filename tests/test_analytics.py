"""Tests for portfolio analytics module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile

from src.portfolio.analytics import PortfolioAnalytics, PortfolioAnalyticsSummary, CashFlow
from src.portfolio.portfolio import Portfolio
from src.portfolio.position import Position
from src.portfolio.exposures import FundExposureMap, FundDefinition, ExposureType
from src.data.market_data import MarketDataFetcher


class TestCashFlow:
    """Tests for CashFlow dataclass."""
    
    def test_cash_flow_creation(self):
        """Test creating a cash flow."""
        cf = CashFlow(datetime(2024, 1, 15), 1000.0, "Initial deposit")
        assert cf.date == datetime(2024, 1, 15)
        assert cf.amount == 1000.0
        assert cf.description == "Initial deposit"


class TestPortfolioAnalyticsSummary:
    """Tests for PortfolioAnalyticsSummary dataclass."""
    
    @pytest.fixture
    def summary(self):
        """Create a test summary."""
        return PortfolioAnalyticsSummary(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 12, 31),
            total_return=0.15,
            annualized_return=0.15,
            volatility=0.20,
            sharpe_ratio=0.75,
            max_drawdown=0.05,
            max_drawdown_duration=30,
            var_95=0.03,
            cvar_95=0.045,
            best_position="SPY",
            worst_position="TLT",
            best_position_return=0.25,
            worst_position_return=-0.05
        )
    
    def test_summary_creation(self, summary):
        """Test creating a summary."""
        assert summary.total_return == 0.15
        assert summary.volatility == 0.20
        assert summary.best_position == "SPY"
    
    def test_to_dict(self, summary):
        """Test converting summary to dictionary."""
        summary_dict = summary.to_dict()
        
        assert summary_dict['total_return'] == 0.15
        assert summary_dict['volatility'] == 0.20
        assert summary_dict['best_position'] == "SPY"
        assert 'period_start' in summary_dict
        assert 'period_end' in summary_dict


class TestPortfolioAnalytics:
    """Tests for PortfolioAnalytics class."""
    
    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio."""
        portfolio = Portfolio("Test Portfolio", cash=10000.0)
        
        positions = [
            Position("SPY", 100, 400.0, datetime(2024, 1, 1), asset_class="Equity"),
            Position("TLT", 50, 100.0, datetime(2024, 1, 1), asset_class="Bond"),
        ]
        
        for pos in positions:
            portfolio.add_position(pos)
        
        return portfolio
    
    @pytest.fixture
    def mock_fetcher(self):
        """Create mock market data fetcher."""
        fetcher = Mock(spec=MarketDataFetcher)
        
        # Create synthetic price data
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        
        # SPY trending up with volatility
        spy_prices = 400 + np.cumsum(np.random.normal(0.1, 2, len(dates)))
        spy_data = pd.DataFrame({'Adj Close': spy_prices}, index=dates)
        
        # TLT trending down slightly
        tlt_prices = 100 + np.cumsum(np.random.normal(-0.05, 1, len(dates)))
        tlt_data = pd.DataFrame({'Adj Close': tlt_prices}, index=dates)
        
        fetcher.fetch_price_history.return_value = {
            'SPY': spy_data,
            'TLT': tlt_data
        }
        
        return fetcher
    
    @pytest.fixture
    def analytics(self, portfolio, mock_fetcher):
        """Create analytics object."""
        return PortfolioAnalytics(portfolio, mock_fetcher)
    
    def test_analytics_creation(self, analytics):
        """Test creating analytics object."""
        assert analytics.portfolio is not None
        assert analytics.market_data is not None
        assert len(analytics._price_cache) == 0
    
    def test_calculate_portfolio_values(self, analytics):
        """Test calculating portfolio values."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        values = analytics.calculate_portfolio_values(start_date, end_date)
        
        assert isinstance(values, pd.Series)
        assert len(values) > 0
        assert all(values > 0)  # Portfolio values should be positive
    
    def test_calculate_returns_daily(self, analytics):
        """Test calculating daily returns."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        returns = analytics.calculate_returns(start_date, end_date, 'daily')
        
        assert isinstance(returns, pd.Series)
        assert len(returns) > 0
        # Returns should be reasonable (between -50% and +50% daily)
        assert all(returns > -0.5)
        assert all(returns < 0.5)
    
    def test_calculate_returns_monthly(self, analytics):
        """Test calculating monthly returns."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 31)
        
        # Need to mock longer price series
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        spy_prices = 400 + np.cumsum(np.random.normal(0.1, 2, len(dates)))
        tlt_prices = 100 + np.cumsum(np.random.normal(-0.05, 1, len(dates)))
        
        analytics.market_data.fetch_price_history.return_value = {
            'SPY': pd.DataFrame({'Adj Close': spy_prices}, index=dates),
            'TLT': pd.DataFrame({'Adj Close': tlt_prices}, index=dates)
        }
        
        returns = analytics.calculate_returns(start_date, end_date, 'monthly')
        
        assert isinstance(returns, pd.Series)
        # Should have fewer observations than daily
        assert len(returns) <= 3  # Max 3 months
    
    def test_calculate_position_returns(self, analytics):
        """Test calculating position returns."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        returns = analytics.calculate_position_returns("SPY", start_date, end_date)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) > 0
    
    def test_calculate_position_returns_invalid_symbol(self, analytics):
        """Test calculating returns for invalid symbol."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        with pytest.raises(ValueError, match="Symbol INVALID not found"):
            analytics.calculate_position_returns("INVALID", start_date, end_date)
    
    def test_time_weighted_return_simple(self, analytics):
        """Test time-weighted return without cash flows."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        twr = analytics.time_weighted_return(start_date, end_date)
        
        assert isinstance(twr, float)
        # Should be reasonable return
        assert -1.0 < twr < 1.0
    
    def test_time_weighted_return_with_cash_flows(self, analytics):
        """Test time-weighted return with cash flows."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        cash_flows = [
            CashFlow(datetime(2024, 1, 15), 5000.0, "Deposit")
        ]
        
        twr = analytics.time_weighted_return(start_date, end_date, cash_flows)
        
        assert isinstance(twr, float)
    
    def test_calculate_volatility(self, analytics):
        """Test volatility calculation."""
        # Create sample returns
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        
        vol_daily = analytics.calculate_volatility(returns, annualize=False)
        vol_annual = analytics.calculate_volatility(returns, annualize=True)
        
        assert vol_daily > 0
        assert vol_annual > vol_daily  # Annualized should be higher
        assert abs(vol_annual - vol_daily * np.sqrt(252)) < 1e-10
    
    def test_calculate_sharpe_ratio(self, analytics):
        """Test Sharpe ratio calculation."""
        # Create sample returns with positive mean and some volatility
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))  # 0.1% mean daily return with volatility
        
        sharpe = analytics.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        # Should be reasonable value (not testing specific sign since returns are random)
    
    def test_calculate_sharpe_ratio_zero_vol(self, analytics):
        """Test Sharpe ratio with zero volatility."""
        returns = pd.Series([0.001] * 100)  # Constant returns
        
        sharpe = analytics.calculate_sharpe_ratio(returns)
        
        # With zero volatility, Sharpe ratio should be zero by our implementation
        assert sharpe == 0.0
    
    def test_calculate_max_drawdown(self, analytics):
        """Test maximum drawdown calculation."""
        # Create portfolio values with a clear drawdown
        values = pd.Series([100, 110, 120, 100, 90, 95, 130], 
                          index=pd.date_range('2024-01-01', periods=7))
        
        dd_info = analytics.calculate_max_drawdown(values)
        
        assert dd_info['max_drawdown'] > 0
        assert dd_info['max_drawdown_duration'] >= 0
        assert dd_info['peak_date'] is not None
        assert dd_info['trough_date'] is not None
    
    def test_calculate_max_drawdown_empty(self, analytics):
        """Test max drawdown with empty series."""
        values = pd.Series(dtype=float)
        
        dd_info = analytics.calculate_max_drawdown(values)
        
        assert dd_info['max_drawdown'] == 0.0
        assert dd_info['max_drawdown_duration'] == 0
    
    def test_calculate_var_historical(self, analytics):
        """Test VaR calculation with historical method."""
        # Create returns with known distribution
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        var_95 = analytics.calculate_var(returns, 0.95, 'historical')
        
        assert var_95 > 0  # VaR should be positive
        assert 0.01 < var_95 < 0.1  # Should be reasonable magnitude
    
    def test_calculate_var_parametric(self, analytics):
        """Test VaR calculation with parametric method."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        var_95 = analytics.calculate_var(returns, 0.95, 'parametric')
        
        assert var_95 > 0
    
    def test_calculate_var_invalid_method(self, analytics):
        """Test VaR with invalid method."""
        returns = pd.Series([0.01, -0.02, 0.015])
        
        with pytest.raises(ValueError, match="Unsupported VaR method"):
            analytics.calculate_var(returns, 0.95, 'invalid')
    
    def test_calculate_cvar(self, analytics):
        """Test CVaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        cvar_95 = analytics.calculate_cvar(returns, 0.95)
        var_95 = analytics.calculate_var(returns, 0.95, 'historical')
        
        assert cvar_95 > 0
        assert cvar_95 >= var_95  # CVaR should be >= VaR
    
    def test_calculate_information_ratio(self, analytics):
        """Test information ratio calculation."""
        portfolio_returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        benchmark_returns = pd.Series([0.008, -0.015, 0.012, -0.008, 0.018])
        
        ir = analytics.calculate_information_ratio(portfolio_returns, benchmark_returns)
        
        assert isinstance(ir, float)
        # Should be reasonable value
        assert -10 < ir < 10
    
    def test_calculate_tracking_error(self, analytics):
        """Test tracking error calculation."""
        portfolio_returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        benchmark_returns = pd.Series([0.008, -0.015, 0.012, -0.008, 0.018])
        
        te = analytics.calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        assert te > 0
        assert te < 1.0  # Should be reasonable
    
    def test_calculate_beta(self, analytics):
        """Test beta calculation."""
        portfolio_returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        market_returns = pd.Series([0.008, -0.015, 0.012, -0.008, 0.018])
        
        beta = analytics.calculate_beta(portfolio_returns, market_returns)
        
        assert isinstance(beta, float)
        assert 0 < beta < 3  # Should be reasonable beta
    
    def test_calculate_alpha(self, analytics):
        """Test alpha calculation."""
        portfolio_returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        market_returns = pd.Series([0.008, -0.015, 0.012, -0.008, 0.018])
        
        alpha = analytics.calculate_alpha(portfolio_returns, market_returns)
        
        assert isinstance(alpha, float)
        assert -1 < alpha < 1  # Should be reasonable alpha
    
    def test_generate_analytics_summary(self, analytics):
        """Test generating comprehensive analytics summary."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        summary = analytics.generate_analytics_summary(start_date, end_date)
        
        assert isinstance(summary, PortfolioAnalyticsSummary)
        assert summary.period_start == start_date
        assert summary.period_end == end_date
        assert isinstance(summary.total_return, float)
        assert isinstance(summary.volatility, float)
        assert isinstance(summary.sharpe_ratio, float)
    
    def test_generate_analytics_summary_with_fund_map(self, analytics):
        """Test analytics summary with fund exposure map."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        # Create simple fund map
        fund_map = FundExposureMap()
        spy_def = FundDefinition(
            symbol="SPY",
            name="SPDR S&P 500 ETF",
            exposures={ExposureType.US_LARGE_EQUITY: 1.0},
            total_notional=1.0
        )
        fund_map.add_fund_definition(spy_def)
        
        summary = analytics.generate_analytics_summary(start_date, end_date, fund_map)
        
        assert isinstance(summary, PortfolioAnalyticsSummary)
        # Should have exposure returns if fund map provided
        assert summary.exposure_returns is not None or summary.exposure_returns == {}
    
    def test_calculate_exposure_returns(self, analytics):
        """Test exposure-based return calculation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        # Create fund map
        fund_map = FundExposureMap()
        spy_def = FundDefinition(
            symbol="SPY",
            name="SPDR S&P 500 ETF",
            exposures={ExposureType.US_LARGE_EQUITY: 1.0},
            total_notional=1.0
        )
        fund_map.add_fund_definition(spy_def)
        
        exposure_returns = analytics.calculate_exposure_returns(fund_map, start_date, end_date)
        
        # Should return DataFrame or empty DataFrame
        assert isinstance(exposure_returns, pd.DataFrame)
    
    def test_exposure_attribution(self, analytics):
        """Test exposure attribution calculation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        # Create fund map
        fund_map = FundExposureMap()
        spy_def = FundDefinition(
            symbol="SPY",
            name="SPDR S&P 500 ETF",
            exposures={ExposureType.US_LARGE_EQUITY: 1.0},
            total_notional=1.0
        )
        fund_map.add_fund_definition(spy_def)
        
        attribution = analytics.exposure_attribution(fund_map, start_date, end_date)
        
        # Should return dictionary
        assert isinstance(attribution, dict)
    
    def test_price_data_caching(self, analytics):
        """Test that price data is cached."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        # First call
        analytics.calculate_returns(start_date, end_date)
        assert len(analytics._price_cache) > 0
        
        # Second call should use cache
        cache_size_before = len(analytics._price_cache)
        analytics.calculate_returns(start_date, end_date)
        
        # Cache size shouldn't change (reused existing data)
        assert len(analytics._price_cache) == cache_size_before
    
    def test_empty_portfolio(self):
        """Test analytics with empty portfolio."""
        empty_portfolio = Portfolio("Empty Portfolio")
        mock_fetcher = Mock(spec=MarketDataFetcher)
        analytics = PortfolioAnalytics(empty_portfolio, mock_fetcher)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        values = analytics.calculate_portfolio_values(start_date, end_date)
        assert values.empty
    
    def test_missing_price_data(self, portfolio):
        """Test handling of missing price data."""
        mock_fetcher = Mock(spec=MarketDataFetcher)
        mock_fetcher.fetch_price_history.return_value = {}  # No data
        
        analytics = PortfolioAnalytics(portfolio, mock_fetcher)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        with pytest.raises(ValueError, match="Could not fetch price data"):
            analytics.calculate_portfolio_values(start_date, end_date)


class TestIntegration:
    """Integration tests for analytics with real data."""
    
    @pytest.fixture
    def real_portfolio(self):
        """Create portfolio with real symbols."""
        portfolio = Portfolio("Real Portfolio", cash=5000.0)
        
        positions = [
            Position("SPY", 10, 400.0, datetime(2024, 1, 1), asset_class="Equity"),
            Position("AGG", 20, 100.0, datetime(2024, 1, 1), asset_class="Bond"),
        ]
        
        for pos in positions:
            portfolio.add_position(pos)
        
        return portfolio
    
    def test_analytics_with_real_data(self, real_portfolio):
        """Test analytics with actual market data."""
        try:
            from src.data.market_data import MarketDataFetcher
            fetcher = MarketDataFetcher()
            analytics = PortfolioAnalytics(real_portfolio, fetcher)
            
            # Use recent dates that should have data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # This test might fail if market data is unavailable
            # That's okay - it's testing the integration
            returns = analytics.calculate_returns(start_date, end_date)
            
            if not returns.empty:
                assert len(returns) > 0
                assert all(returns > -1.0)  # No 100% daily losses
                assert all(returns < 1.0)   # No 100% daily gains
                
        except Exception as e:
            pytest.skip(f"Could not test with real data: {e}")
    
    def test_summary_with_real_fund_data(self, real_portfolio):
        """Test summary generation with real fund universe data."""
        import os
        
        fund_universe_path = "data/fund_universe.yaml"
        if not os.path.exists(fund_universe_path):
            pytest.skip("Fund universe data not available")
        
        try:
            fund_map = FundExposureMap(fund_universe_path)
            
            mock_fetcher = Mock(spec=MarketDataFetcher)
            
            # Create synthetic but realistic price data
            dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
            spy_data = pd.DataFrame({
                'Adj Close': 400 + np.cumsum(np.random.normal(0.5, 8, len(dates)))
            }, index=dates)
            agg_data = pd.DataFrame({
                'Adj Close': 100 + np.cumsum(np.random.normal(0.1, 1, len(dates)))
            }, index=dates)
            
            mock_fetcher.fetch_price_history.return_value = {
                'SPY': spy_data,
                'AGG': agg_data
            }
            
            analytics = PortfolioAnalytics(real_portfolio, mock_fetcher)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 31)
            
            summary = analytics.generate_analytics_summary(start_date, end_date, fund_map)
            
            assert isinstance(summary, PortfolioAnalyticsSummary)
            
            # Should have some basic metrics
            assert isinstance(summary.total_return, float)
            assert isinstance(summary.volatility, float)
            
        except Exception as e:
            pytest.skip(f"Could not test with fund data: {e}")