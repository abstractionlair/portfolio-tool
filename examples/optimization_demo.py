#!/usr/bin/env python3
"""
Demonstration of the Portfolio Optimization Engine

This script shows how to use the optimization engine to create portfolios
with different objectives and constraints, particularly highlighting the
leverage-aware features for handling leveraged funds.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.optimization import (
    OptimizationEngine, ObjectiveType, OptimizationConstraints,
    ReturnEstimator, MarketView
)
from src.portfolio import Portfolio, Position, ExposureType, FundExposureMap, FundDefinition, PortfolioAnalytics
from src.data import MarketDataFetcher


class DemoMarketDataFetcher:
    """Demo market data fetcher with realistic example data."""
    
    def fetch_price_history(self, symbols, start_date, end_date):
        """Generate realistic price histories for demo."""
        # Generate dates
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Base parameters for different asset classes
        params = {
            'SPY': {'initial': 400, 'drift': 0.08/252, 'vol': 0.16/np.sqrt(252)},
            'QQQ': {'initial': 350, 'drift': 0.10/252, 'vol': 0.20/np.sqrt(252)},
            'TLT': {'initial': 120, 'drift': 0.02/252, 'vol': 0.12/np.sqrt(252)},
            'IWM': {'initial': 200, 'drift': 0.09/252, 'vol': 0.22/np.sqrt(252)},
            'GLD': {'initial': 180, 'drift': 0.04/252, 'vol': 0.18/np.sqrt(252)},
            'VNQ': {'initial': 90, 'drift': 0.07/252, 'vol': 0.25/np.sqrt(252)},
            'TQQQ': {'initial': 35, 'drift': 0.30/252, 'vol': 0.60/np.sqrt(252)},  # 3x leveraged
            'RSSB': {'initial': 25, 'drift': 0.05/252, 'vol': 0.14/np.sqrt(252)},  # Balanced fund
        }
        
        price_data = {}
        
        for symbol in symbols:
            if symbol in params:
                p = params[symbol]
                
                # Generate geometric Brownian motion
                np.random.seed(hash(symbol) % 2**32)  # Deterministic for demo
                returns = np.random.normal(p['drift'], p['vol'], len(dates))
                prices = [p['initial']]
                
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                # Convert to pandas Series with date index
                price_series = pd.Series(prices, index=dates, name=symbol)
                price_data[symbol] = price_series
            else:
                # Default for unknown symbols
                np.random.seed(hash(symbol) % 2**32)
                returns = np.random.normal(0.06/252, 0.18/np.sqrt(252), len(dates))
                prices = [100]
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                price_series = pd.Series(prices, index=dates, name=symbol)
                price_data[symbol] = price_series
        
        return price_data


def create_demo_fund_map():
    """Create a comprehensive fund universe for demonstration."""
    fund_map = FundExposureMap()
    
    # Traditional ETFs
    fund_map.add_fund_definition(FundDefinition(
        symbol='SPY',
        name='SPDR S&P 500 ETF',
        exposures={ExposureType.US_LARGE_EQUITY: 1.0},
        total_notional=1.0
    ))
    
    fund_map.add_fund_definition(FundDefinition(
        symbol='QQQ',
        name='Invesco QQQ Trust',
        exposures={
            ExposureType.US_LARGE_EQUITY: 1.0,
            ExposureType.QUALITY_FACTOR: 0.55
        },
        total_notional=1.0
    ))
    
    fund_map.add_fund_definition(FundDefinition(
        symbol='IWM',
        name='iShares Russell 2000 ETF',
        exposures={ExposureType.US_SMALL_EQUITY: 1.0},
        total_notional=1.0
    ))
    
    fund_map.add_fund_definition(FundDefinition(
        symbol='TLT',
        name='iShares 20+ Year Treasury Bond ETF',
        exposures={ExposureType.BONDS: 1.0},
        total_notional=1.0
    ))
    
    fund_map.add_fund_definition(FundDefinition(
        symbol='GLD',
        name='SPDR Gold Shares',
        exposures={ExposureType.COMMODITIES: 1.0},
        total_notional=1.0
    ))
    
    fund_map.add_fund_definition(FundDefinition(
        symbol='VNQ',
        name='Vanguard Real Estate ETF',
        exposures={ExposureType.REAL_ESTATE: 1.0},
        total_notional=1.0
    ))
    
    # Leveraged fund - 3x QQQ
    fund_map.add_fund_definition(FundDefinition(
        symbol='TQQQ',
        name='ProShares UltraPro QQQ',
        exposures={
            ExposureType.US_LARGE_EQUITY: 3.0,
            ExposureType.QUALITY_FACTOR: 1.65
        },
        total_notional=3.0
    ))
    
    # Multi-asset fund with complex exposures
    fund_map.add_fund_definition(FundDefinition(
        symbol='RSSB',
        name='Invesco Russell 1000 Equal Weight ETF',
        exposures={
            ExposureType.US_LARGE_EQUITY: 1.0,
            ExposureType.BONDS: 1.0  # Hypothetical balanced fund
        },
        total_notional=2.0  # 100% stocks + 100% bonds
    ))
    
    return fund_map


def demo_basic_optimization():
    """Demonstrate basic portfolio optimization."""
    print("=" * 60)
    print("BASIC PORTFOLIO OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Setup
    fund_map = create_demo_fund_map()
    mock_fetcher = DemoMarketDataFetcher()
    dummy_portfolio = Portfolio("Demo Portfolio")
    analytics = PortfolioAnalytics(dummy_portfolio, mock_fetcher)
    engine = OptimizationEngine(analytics, fund_map)
    
    # Universe of assets
    symbols = ['SPY', 'TLT', 'GLD', 'VNQ']
    
    # Estimate returns and covariance
    estimator = ReturnEstimator(mock_fetcher)
    expected_returns = estimator.estimate_expected_returns(symbols, lookback_years=2)
    cov_matrix = estimator.estimate_covariance_matrix(symbols, lookback_years=2)
    
    print(f"\\nUniverse: {symbols}")
    print(f"Expected Returns: {[f'{r:.1%}' for r in expected_returns]}")
    print(f"Volatilities: {[f'{np.sqrt(cov_matrix[i,i]):.1%}' for i in range(len(symbols))]}")
    
    # Basic constraints
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.5,
        long_only=True,
        max_total_notional=1.0
    )
    
    # Test different objectives
    objectives = [
        (ObjectiveType.MAX_SHARPE, "Maximum Sharpe Ratio"),
        (ObjectiveType.MIN_VOLATILITY, "Minimum Volatility"),
        (ObjectiveType.RISK_PARITY, "Risk Parity")
    ]
    
    for obj_type, obj_name in objectives:
        print(f"\\n--- {obj_name} ---")
        result = engine.optimize(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            constraints=constraints,
            objective=obj_type
        )
        
        if result.success:
            print(f"Expected Return: {result.expected_return:.1%}")
            print(f"Volatility: {result.expected_volatility:.1%}")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print("Weights:")
            for symbol, weight in result.weights.items():
                if weight > 0.01:  # Only show meaningful weights
                    print(f"  {symbol}: {weight:.1%}")
        else:
            print(f"Optimization failed: {result.message}")


def demo_leveraged_optimization():
    """Demonstrate optimization with leveraged funds."""
    print("\\n" + "=" * 60)
    print("LEVERAGED FUND OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Setup
    fund_map = create_demo_fund_map()
    mock_fetcher = DemoMarketDataFetcher()
    dummy_portfolio = Portfolio("Demo Portfolio")
    analytics = PortfolioAnalytics(dummy_portfolio, mock_fetcher)
    engine = OptimizationEngine(analytics, fund_map)
    
    # Include leveraged fund
    symbols = ['SPY', 'TLT', 'TQQQ']  # Include 3x leveraged QQQ
    
    estimator = ReturnEstimator(mock_fetcher)
    expected_returns = estimator.estimate_expected_returns(symbols, lookback_years=2)
    cov_matrix = estimator.estimate_covariance_matrix(symbols, lookback_years=2)
    
    print(f"\\nUniverse: {symbols}")
    print("Fund Details:")
    for symbol in symbols:
        fund_def = fund_map.get_fund_definition(symbol)
        if fund_def:
            print(f"  {symbol}: {fund_def.total_notional}x leverage")
            for exp_type, amount in fund_def.exposures.items():
                print(f"    {exp_type.value}: {amount:.1f}")
    
    # Allow leverage but with limits
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.4,
        long_only=True,
        max_total_notional=2.0  # Allow up to 200% notional exposure
    )
    
    print(f"\\n--- Maximum Sharpe with Leverage Constraint ---")
    result = engine.optimize(
        symbols=symbols,
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        constraints=constraints,
        objective=ObjectiveType.MAX_SHARPE
    )
    
    if result.success:
        print(f"Expected Return: {result.expected_return:.1%}")
        print(f"Volatility: {result.expected_volatility:.1%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Total Notional Exposure: {result.total_notional:.1f}x")
        print("Weights:")
        for symbol, weight in result.weights.items():
            if weight > 0.01:
                print(f"  {symbol}: {weight:.1%}")
        
        print("\\nExposure Breakdown:")
        for exp_type, exposure in result.exposures.items():
            if abs(exposure) > 0.01:
                print(f"  {exp_type.value}: {exposure:.1%}")
    else:
        print(f"Optimization failed: {result.message}")


def demo_exposure_targeting():
    """Demonstrate exposure-based optimization."""
    print("\\n" + "=" * 60)
    print("EXPOSURE-BASED OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Setup
    fund_map = create_demo_fund_map()
    mock_fetcher = DemoMarketDataFetcher()
    dummy_portfolio = Portfolio("Demo Portfolio")
    analytics = PortfolioAnalytics(dummy_portfolio, mock_fetcher)
    engine = OptimizationEngine(analytics, fund_map)
    
    # Multi-asset universe including complex funds
    symbols = ['SPY', 'TLT', 'GLD', 'TQQQ', 'RSSB']
    
    print(f"\\nUniverse: {symbols}")
    print("Fund Exposures:")
    for symbol in symbols:
        fund_def = fund_map.get_fund_definition(symbol)
        if fund_def:
            print(f"  {symbol}:")
            for exp_type, amount in fund_def.exposures.items():
                print(f"    {exp_type.value}: {amount:.1f}")
    
    # Target allocation: 60% equity, 25% bonds, 10% commodities, 5% real estate
    target_exposures = {
        ExposureType.US_LARGE_EQUITY: 0.60,
        ExposureType.BONDS: 0.25,
        ExposureType.COMMODITIES: 0.10,
        ExposureType.REAL_ESTATE: 0.05
    }
    
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.5,
        long_only=True,
        max_total_notional=1.5
    )
    
    print(f"\\n--- Target Exposure Optimization ---")
    print("Target Exposures:")
    for exp_type, target in target_exposures.items():
        print(f"  {exp_type.value}: {target:.1%}")
    
    result = engine.optimize(
        symbols=symbols,
        expected_returns=np.ones(len(symbols)) * 0.06,  # Dummy returns for exposure optimization
        covariance_matrix=np.eye(len(symbols)) * 0.04,  # Dummy covariance
        constraints=constraints,
        objective=ObjectiveType.TARGET_EXPOSURES,
        target_exposures=target_exposures
    )
    
    if result.success:
        print(f"\\nOptimal Weights:")
        for symbol, weight in result.weights.items():
            if weight > 0.01:
                print(f"  {symbol}: {weight:.1%}")
        
        print(f"\\nActual vs Target Exposures:")
        for exp_type, target in target_exposures.items():
            actual = result.exposures.get(exp_type, 0.0)
            diff = actual - target
            print(f"  {exp_type.value}:")
            print(f"    Target: {target:.1%}, Actual: {actual:.1%}, Diff: {diff:+.1%}")
        
        print(f"\\nTotal Notional Exposure: {result.total_notional:.2f}x")
    else:
        print(f"Optimization failed: {result.message}")


def demo_black_litterman():
    """Demonstrate Black-Litterman optimization with views."""
    print("\\n" + "=" * 60)
    print("BLACK-LITTERMAN WITH MARKET VIEWS DEMO")
    print("=" * 60)
    
    # Setup
    fund_map = create_demo_fund_map()
    mock_fetcher = DemoMarketDataFetcher()
    dummy_portfolio = Portfolio("Demo Portfolio")
    analytics = PortfolioAnalytics(dummy_portfolio, mock_fetcher)
    engine = OptimizationEngine(analytics, fund_map)
    
    symbols = ['SPY', 'QQQ', 'TLT', 'GLD']
    
    # Market cap weights (simplified)
    estimator = ReturnEstimator(mock_fetcher)
    market_weights = estimator.estimate_market_weights(symbols)
    cov_matrix = estimator.estimate_covariance_matrix(symbols, lookback_years=2)
    
    print(f"\\nUniverse: {symbols}")
    print(f"Market Weights: {[f'{w:.1%}' for w in market_weights]}")
    
    # Create market views
    views = [
        MarketView(
            assets=['QQQ'],
            view_type='absolute',
            expected_return=0.12,  # 12% expected return for tech
            confidence=0.8,
            description="Tech will outperform due to AI trends"
        ),
        MarketView(
            assets=['SPY', 'TLT'],
            view_type='relative',
            expected_return=0.03,  # Equities outperform bonds by 3%
            confidence=0.6,
            description="Equities over bonds in growth environment"
        )
    ]
    
    print(f"\\nMarket Views:")
    for i, view in enumerate(views, 1):
        print(f"  {i}. {view.description}")
        print(f"     Assets: {view.assets}")
        print(f"     Expected: {view.expected_return:.1%} ({view.view_type})")
        print(f"     Confidence: {view.confidence:.0%}")
    
    # Calculate Black-Litterman returns
    bl_returns = engine.bl_optimizer.black_litterman_returns(
        symbols=symbols,
        market_weights=market_weights,
        covariance_matrix=cov_matrix,
        views=views
    )
    
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.6,
        long_only=True
    )
    
    # Compare standard vs Black-Litterman optimization
    print(f"\\n--- Standard Max Sharpe ---")
    hist_returns = estimator.estimate_expected_returns(symbols, method='historical')
    result_standard = engine.optimize(
        symbols=symbols,
        expected_returns=hist_returns,
        covariance_matrix=cov_matrix,
        constraints=constraints,
        objective=ObjectiveType.MAX_SHARPE
    )
    
    if result_standard.success:
        print("Weights:")
        for symbol, weight in result_standard.weights.items():
            if weight > 0.01:
                print(f"  {symbol}: {weight:.1%}")
    
    print(f"\\n--- Black-Litterman Max Sharpe ---")
    result_bl = engine.optimize(
        symbols=symbols,
        expected_returns=bl_returns,
        covariance_matrix=cov_matrix,
        constraints=constraints,
        objective=ObjectiveType.MAX_SHARPE
    )
    
    if result_bl.success:
        print("Weights:")
        for symbol, weight in result_bl.weights.items():
            if weight > 0.01:
                print(f"  {symbol}: {weight:.1%}")
        
        print(f"\\nComparison:")
        print(f"  Standard Sharpe: {result_standard.sharpe_ratio:.2f}")
        print(f"  BL Sharpe: {result_bl.sharpe_ratio:.2f}")


def demo_trade_generation():
    """Demonstrate trade generation from optimization results."""
    print("\\n" + "=" * 60)
    print("TRADE GENERATION DEMO")
    print("=" * 60)
    
    # Setup
    fund_map = create_demo_fund_map()
    mock_fetcher = DemoMarketDataFetcher()
    dummy_portfolio = Portfolio("Demo Portfolio")
    analytics = PortfolioAnalytics(dummy_portfolio, mock_fetcher)
    engine = OptimizationEngine(analytics, fund_map)
    
    # Create current portfolio
    current_portfolio = Portfolio("Demo Portfolio")
    current_portfolio.add_position(Position(symbol='SPY', quantity=100.0, cost_basis=400.0, purchase_date=datetime.now()))
    current_portfolio.add_position(Position(symbol='TLT', quantity=200.0, cost_basis=120.0, purchase_date=datetime.now()))
    
    # Current market prices
    prices = {
        'SPY': 420.0,
        'TLT': 125.0,
        'QQQ': 350.0,
        'GLD': 185.0
    }
    
    current_value = sum(pos.quantity * prices[symbol] for symbol, pos in current_portfolio.positions.items())
    print(f"Current Portfolio Value: ${current_value:,.0f}")
    print("Current Holdings:")
    for symbol, pos in current_portfolio.positions.items():
        value = pos.quantity * prices[symbol]
        weight = value / current_value
        print(f"  {symbol}: {pos.quantity:.0f} shares = ${value:,.0f} ({weight:.1%})")
    
    # Optimize for new target
    symbols = ['SPY', 'TLT', 'QQQ', 'GLD']
    estimator = ReturnEstimator(mock_fetcher)
    expected_returns = estimator.estimate_expected_returns(symbols, lookback_years=2)
    cov_matrix = estimator.estimate_covariance_matrix(symbols, lookback_years=2)
    
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.5,
        long_only=True
    )
    
    result = engine.optimize(
        symbols=symbols,
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        constraints=constraints,
        objective=ObjectiveType.MAX_SHARPE
    )
    
    if result.success:
        print(f"\\n--- Target Portfolio ---")
        print("Target Weights:")
        for symbol, weight in result.weights.items():
            if weight > 0.01:
                target_value = weight * current_value
                print(f"  {symbol}: {weight:.1%} = ${target_value:,.0f}")
        
        # Generate trades
        trades = result.to_trades(
            current_portfolio=current_portfolio,
            prices=prices,
            total_portfolio_value=current_value
        )
        
        print(f"\\n--- Required Trades ---")
        if trades:
            total_trade_value = sum(t.trade_value for t in trades)
            print(f"Total Trade Value: ${total_trade_value:,.0f}")
            
            for trade in trades:
                print(f"  {trade.direction} {abs(trade.quantity):.1f} shares of {trade.symbol}")
                print(f"    @ ${trade.current_price:.2f} = ${trade.trade_value:,.0f}")
        else:
            print("No trades required - portfolio already optimal!")
    else:
        print(f"Optimization failed: {result.message}")


def main():
    """Run all optimization demos."""
    print("Portfolio Optimization Engine Demo")
    print("This demonstration shows various optimization capabilities")
    print("including leverage-aware optimization for complex funds.\\n")
    
    try:
        demo_basic_optimization()
        demo_leveraged_optimization()
        demo_exposure_targeting()
        demo_black_litterman()
        demo_trade_generation()
        
        print("\\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\\nKey Features Demonstrated:")
        print("• Multiple optimization objectives (Sharpe, min vol, risk parity)")
        print("• Leverage-aware optimization for complex funds")
        print("• Exposure-based portfolio construction")
        print("• Black-Litterman model with market views")
        print("• Trade generation from optimization results")
        print("• Comprehensive constraint handling")
        
    except Exception as e:
        print(f"\\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()