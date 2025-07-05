"""Example usage of the Portfolio Analytics system."""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os

from src.portfolio import Portfolio, Position
from src.portfolio.analytics import PortfolioAnalytics, CashFlow
from src.portfolio.exposures import FundExposureMap
from src.data.market_data import MarketDataFetcher


def create_mock_market_data(symbols, start_date, end_date):
    """Create realistic mock market data for demonstration."""
    
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Different return characteristics for different assets
    asset_params = {
        'SPY': {'drift': 0.0008, 'vol': 0.015},    # S&P 500: 20% annual return, 24% vol
        'QQQ': {'drift': 0.001, 'vol': 0.018},     # Nasdaq: 25% annual return, 28% vol
        'TLT': {'drift': 0.0002, 'vol': 0.012},    # Long bonds: 5% annual return, 19% vol
        'GLD': {'drift': 0.0003, 'vol': 0.013},    # Gold: 8% annual return, 21% vol
        'VNQ': {'drift': 0.0006, 'vol': 0.016},    # REITs: 15% annual return, 25% vol
        'RSSB': {'drift': 0.0007, 'vol': 0.014},   # Return stacked: 18% annual return, 22% vol
    }
    
    price_data = {}
    
    for symbol in symbols:
        if symbol in asset_params:
            params = asset_params[symbol]
        else:
            # Default parameters for unknown symbols
            params = {'drift': 0.0005, 'vol': 0.015}
        
        # Generate price series with realistic starting values
        start_prices = {'SPY': 420.0, 'QQQ': 350.0, 'TLT': 95.0, 'GLD': 180.0, 'VNQ': 90.0, 'RSSB': 25.0}
        start_price = start_prices.get(symbol, 100.0)
        
        # Generate random returns
        returns = np.random.normal(params['drift'], params['vol'], len(dates))
        
        # Create price series
        prices = [start_price]
        for ret in returns[1:]:  # Skip first return
            prices.append(prices[-1] * (1 + ret))
        
        price_data[symbol] = pd.DataFrame({
            'Adj Close': prices
        }, index=dates)
    
    return price_data


def main():
    """Demonstrate portfolio analytics functionality."""
    print("=== Portfolio Analytics Example ===\n")
    
    # 1. Create a sample portfolio
    print("1. Creating Sample Portfolio...")
    
    portfolio = Portfolio("Diversified Growth Portfolio", cash=15000.0)
    
    # Add various positions with different risk/return profiles
    positions = [
        Position("SPY", 50, 415.0, datetime(2024, 1, 15), asset_class="Equity"),      # Core US equity
        Position("QQQ", 25, 345.0, datetime(2024, 1, 10), asset_class="Equity"),      # Growth tech
        Position("TLT", 100, 96.0, datetime(2024, 1, 20), asset_class="Bond"),        # Long duration bonds
        Position("GLD", 15, 175.0, datetime(2024, 2, 1), asset_class="Commodity"),    # Gold hedge
        Position("VNQ", 30, 88.0, datetime(2024, 2, 15), asset_class="REIT"),         # Real estate
    ]
    
    # Add a leveraged fund if we have exposure data
    if os.path.exists("data/fund_universe.yaml"):
        positions.append(Position("RSSB", 40, 25.5, datetime(2024, 2, 20), asset_class="Equity"))
    
    for pos in positions:
        portfolio.add_position(pos)
        print(f"Added: {pos}")
    
    print(f"\nPortfolio Summary:")
    print(f"  Positions: {len(portfolio)}")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    
    # Create analytics object with mock data
    print("\n2. Setting Up Analytics...")
    
    # For this example, we'll use mock data since we can't guarantee market data access
    class MockMarketDataFetcher:
        def fetch_price_history(self, symbols, start_date, end_date):
            return create_mock_market_data(symbols, start_date, end_date)
    
    market_data = MockMarketDataFetcher()
    analytics = PortfolioAnalytics(portfolio, market_data)
    
    # Define analysis period
    end_date = datetime(2024, 3, 31)
    start_date = datetime(2024, 1, 1)
    
    print(f"Analysis Period: {start_date.date()} to {end_date.date()}")
    
    # 3. Calculate portfolio returns
    print("\n3. Portfolio Return Analysis...")
    
    try:
        # Daily returns
        daily_returns = analytics.calculate_returns(start_date, end_date, 'daily')
        print(f"Daily returns calculated: {len(daily_returns)} observations")
        
        # Portfolio values
        portfolio_values = analytics.calculate_portfolio_values(start_date, end_date)
        print(f"Portfolio values calculated: {len(portfolio_values)} observations")
        
        # Basic statistics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        print(f"Total Return: {total_return:.2%}")
        
        # Show some daily returns
        print(f"\nSample Daily Returns:")
        for i in range(min(5, len(daily_returns))):
            date = daily_returns.index[i]
            ret = daily_returns.iloc[i]
            print(f"  {date.date()}: {ret:.3%}")
        
    except Exception as e:
        print(f"Error calculating returns: {e}")
        return
    
    # 4. Risk metrics
    print("\n4. Risk Analysis...")
    
    try:
        volatility = analytics.calculate_volatility(daily_returns)
        sharpe_ratio = analytics.calculate_sharpe_ratio(daily_returns)
        
        drawdown_info = analytics.calculate_max_drawdown(portfolio_values)
        
        var_95 = analytics.calculate_var(daily_returns, 0.95)
        cvar_95 = analytics.calculate_cvar(daily_returns, 0.95)
        
        print(f"Annualized Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {drawdown_info['max_drawdown']:.2%}")
        print(f"Max Drawdown Duration: {drawdown_info['max_drawdown_duration']} days")
        print(f"95% VaR (daily): {var_95:.3%}")
        print(f"95% CVaR (daily): {cvar_95:.3%}")
        
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
    
    # 5. Position-level analysis
    print("\n5. Position-Level Analysis...")
    
    position_stats = []
    
    for symbol in portfolio.positions.keys():
        try:
            pos_returns = analytics.calculate_position_returns(symbol, start_date, end_date)
            
            if not pos_returns.empty:
                total_pos_return = (1 + pos_returns).prod() - 1
                pos_volatility = analytics.calculate_volatility(pos_returns)
                pos_sharpe = analytics.calculate_sharpe_ratio(pos_returns)
                
                position_stats.append({
                    'Symbol': symbol,
                    'Return': total_pos_return,
                    'Volatility': pos_volatility,
                    'Sharpe': pos_sharpe
                })
                
        except Exception as e:
            print(f"Could not analyze {symbol}: {e}")
    
    if position_stats:
        print("Position Performance:")
        print(f"{'Symbol':<8} {'Return':<10} {'Volatility':<12} {'Sharpe':<8}")
        print("-" * 40)
        
        for stat in sorted(position_stats, key=lambda x: x['Return'], reverse=True):
            print(f"{stat['Symbol']:<8} {stat['Return']:<10.2%} {stat['Volatility']:<12.2%} {stat['Sharpe']:<8.2f}")
    
    # 6. Cash flow analysis
    print("\n6. Cash Flow Analysis...")
    
    # Simulate some cash flows during the period
    cash_flows = [
        CashFlow(datetime(2024, 2, 1), 5000.0, "Monthly contribution"),
        CashFlow(datetime(2024, 3, 1), 5000.0, "Monthly contribution"),
        CashFlow(datetime(2024, 3, 15), -2000.0, "Partial withdrawal"),
    ]
    
    print("Simulated Cash Flows:")
    for cf in cash_flows:
        flow_type = "Inflow" if cf.amount > 0 else "Outflow"
        print(f"  {cf.date.date()}: ${abs(cf.amount):,.2f} ({flow_type}) - {cf.description}")
    
    try:
        twr_simple = analytics.time_weighted_return(start_date, end_date)
        twr_adjusted = analytics.time_weighted_return(start_date, end_date, cash_flows)
        
        print(f"\nTime-Weighted Returns:")
        print(f"  Without cash flows: {twr_simple:.2%}")
        print(f"  With cash flows:    {twr_adjusted:.2%}")
        
    except Exception as e:
        print(f"Error calculating time-weighted returns: {e}")
    
    # 7. Benchmark comparison
    print("\n7. Benchmark Comparison...")
    
    try:
        # Create a simple 60/40 benchmark (60% SPY, 40% TLT)
        spy_data = create_mock_market_data(['SPY'], start_date, end_date)['SPY']
        tlt_data = create_mock_market_data(['TLT'], start_date, end_date)['TLT']
        
        spy_returns = spy_data['Adj Close'].pct_change().dropna()
        tlt_returns = tlt_data['Adj Close'].pct_change().dropna()
        
        # Align dates
        common_dates = spy_returns.index.intersection(tlt_returns.index).intersection(daily_returns.index)
        
        benchmark_returns = (0.6 * spy_returns[common_dates] + 0.4 * tlt_returns[common_dates])
        portfolio_returns_aligned = daily_returns[common_dates]
        
        if len(benchmark_returns) > 0 and len(portfolio_returns_aligned) > 0:
            beta = analytics.calculate_beta(portfolio_returns_aligned, benchmark_returns)
            alpha = analytics.calculate_alpha(portfolio_returns_aligned, benchmark_returns)
            info_ratio = analytics.calculate_information_ratio(portfolio_returns_aligned, benchmark_returns)
            tracking_error = analytics.calculate_tracking_error(portfolio_returns_aligned, benchmark_returns)
            
            print(f"vs 60/40 Benchmark:")
            print(f"  Beta: {beta:.2f}")
            print(f"  Alpha (annualized): {alpha:.2%}")
            print(f"  Information Ratio: {info_ratio:.2f}")
            print(f"  Tracking Error: {tracking_error:.2%}")
            
    except Exception as e:
        print(f"Error in benchmark analysis: {e}")
    
    # 8. Exposure attribution (if fund data available)
    print("\n8. Exposure-Based Attribution...")
    
    fund_map = None
    if os.path.exists("data/fund_universe.yaml"):
        try:
            fund_map = FundExposureMap("data/fund_universe.yaml")
            print(f"Loaded fund exposure map with {len(fund_map)} funds")
            
            attribution = analytics.exposure_attribution(fund_map, start_date, end_date)
            
            if attribution:
                print("Return Attribution by Exposure Type:")
                for exposure_type, return_contrib in sorted(attribution.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {exposure_type.value}: {return_contrib:.2%}")
            else:
                print("No exposure attribution data available")
                
        except Exception as e:
            print(f"Error loading fund exposure data: {e}")
    else:
        print("Fund exposure data not available")
    
    # 9. Comprehensive summary
    print("\n9. Comprehensive Analytics Summary...")
    
    try:
        summary = analytics.generate_analytics_summary(start_date, end_date, fund_map)
        
        print(f"Portfolio Analytics Summary:")
        print(f"  Period: {summary.period_start.date()} to {summary.period_end.date()}")
        print(f"  Total Return: {summary.total_return:.2%}")
        print(f"  Annualized Return: {summary.annualized_return:.2%}")
        print(f"  Volatility: {summary.volatility:.2%}")
        print(f"  Sharpe Ratio: {summary.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {summary.max_drawdown:.2%}")
        print(f"  Max DD Duration: {summary.max_drawdown_duration} days")
        print(f"  95% VaR: {summary.var_95:.3%}")
        print(f"  95% CVaR: {summary.cvar_95:.3%}")
        print(f"  Best Position: {summary.best_position} ({summary.best_position_return:.2%})")
        print(f"  Worst Position: {summary.worst_position} ({summary.worst_position_return:.2%})")
        
        if summary.exposure_returns:
            print(f"  Exposure Returns:")
            for exposure, ret in summary.exposure_returns.items():
                print(f"    {exposure}: {ret:.2%}")
        
        # Export summary to dictionary for further use
        summary_dict = summary.to_dict()
        print(f"\nSummary exported to dictionary with {len(summary_dict)} fields")
        
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    # 10. Advanced analysis demonstrations
    print("\n10. Advanced Analysis Examples...")
    
    try:
        # Monthly returns analysis
        monthly_returns = analytics.calculate_returns(start_date, end_date, 'monthly')
        if not monthly_returns.empty:
            print(f"Monthly Returns Analysis:")
            print(f"  Number of months: {len(monthly_returns)}")
            print(f"  Best month: {monthly_returns.max():.2%}")
            print(f"  Worst month: {monthly_returns.min():.2%}")
            print(f"  Positive months: {(monthly_returns > 0).sum()}/{len(monthly_returns)}")
        
        # Drawdown recovery analysis
        if drawdown_info['recovery_date']:
            print(f"\nDrawdown Recovery Analysis:")
            print(f"  Peak date: {drawdown_info['peak_date'].date()}")
            print(f"  Trough date: {drawdown_info['trough_date'].date()}")
            print(f"  Recovery date: {drawdown_info['recovery_date'].date()}")
            print(f"  Time to recover: {drawdown_info['max_drawdown_duration']} days")
        
    except Exception as e:
        print(f"Error in advanced analysis: {e}")
    
    print("\n=== Analytics Example Complete ===")
    print("\nKey Insights:")
    print("- Portfolio analytics provide comprehensive risk and return measurement")
    print("- Integration with exposure decomposition enables attribution analysis")
    print("- Multiple time horizons (daily, monthly, annual) supported")
    print("- Standard risk metrics (VaR, CVaR, Sharpe ratio) calculated")
    print("- Benchmark comparison and cash flow analysis available")
    print("- Suitable for both simple and complex leveraged portfolios")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()