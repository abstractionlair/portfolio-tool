#!/usr/bin/env python3
"""
Portfolio Optimization Integration Demo

This script demonstrates the complete end-to-end portfolio optimization workflow,
connecting the data layer to the optimization engine with real market data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.optimization.portfolio_optimizer import PortfolioOptimizer, PortfolioOptimizationConfig
from src.optimization.engine import OptimizationConstraints, ObjectiveType

def main():
    """Demonstrate complete portfolio optimization workflow."""
    print("🚀 Portfolio Optimization Integration Demo")
    print("=" * 50)
    
    # Step 1: Initialize the portfolio optimizer
    print("\n1️⃣ Initializing Portfolio Optimizer...")
    optimizer = PortfolioOptimizer()
    print("   ✅ Portfolio optimizer initialized with data layer")
    
    # Step 2: Define portfolio universe
    portfolio_assets = [
        'AAPL',   # Large cap tech
        'GOOGL',  # Large cap tech
        'MSFT',   # Large cap tech
        'TLT',    # Long-term treasury bonds
        'GLD',    # Gold ETF
        'VTI',    # Total stock market
        'AGG'     # Aggregate bonds
    ]
    
    print(f"\n2️⃣ Portfolio Universe: {len(portfolio_assets)} assets")
    for symbol in portfolio_assets:
        print(f"   - {symbol}")
    
    # Step 3: Define optimization period
    end_date = date.today()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    lookback_years = 2  # Use 2 years for estimation
    
    print(f"\n3️⃣ Analysis Period:")
    print(f"   Start Date: {start_date}")
    print(f"   End Date: {end_date}")
    print(f"   Lookback: {lookback_years} years")
    
    # Step 4: Get optimization inputs (for inspection)
    print(f"\n4️⃣ Preparing optimization inputs...")
    try:
        inputs = optimizer.get_optimization_inputs(
            symbols=portfolio_assets,
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
            frequency="daily"
        )
        
        print(f"   ✅ Data prepared:")
        print(f"      Returns data: {len(inputs['returns_data'])} observations")
        print(f"      Expected returns range: {inputs['expected_returns'].min():.3f} to {inputs['expected_returns'].max():.3f}")
        print(f"      Volatilities range: {inputs['volatilities'].min():.3f} to {inputs['volatilities'].max():.3f}")
        print(f"      Risk-free rate: {inputs['risk_free_rate']:.3f}")
        
        # Show correlation matrix summary
        corr_matrix = inputs['correlation_matrix']
        avg_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        print(f"      Average correlation: {avg_correlation:.3f}")
        
    except Exception as e:
        print(f"   ❌ Failed to prepare inputs: {e}")
        return
    
    # Step 5: Define optimization configurations
    print(f"\n5️⃣ Running multiple optimization strategies...")
    
    # Configuration 1: Maximum Sharpe Ratio
    constraints_standard = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.4,  # No single asset > 40%
        long_only=True,
        min_position_size=0.01  # Minimum 1% position
    )
    
    config_sharpe = PortfolioOptimizationConfig(
        symbols=portfolio_assets,
        start_date=start_date,
        end_date=end_date,
        lookback_years=lookback_years,
        objective=ObjectiveType.MAX_SHARPE,
        constraints=constraints_standard,
        return_estimation_method="historical",
        covariance_estimation_method="sample"
    )
    
    # Configuration 2: Minimum Volatility
    config_min_vol = PortfolioOptimizationConfig(
        symbols=portfolio_assets,
        start_date=start_date,
        end_date=end_date,
        lookback_years=lookback_years,
        objective=ObjectiveType.MIN_VOLATILITY,
        constraints=constraints_standard,
        return_estimation_method="historical",
        covariance_estimation_method="shrinkage"  # Use shrinkage for min vol
    )
    
    # Configuration 3: Maximum Sharpe with Shrinkage
    config_sharpe_shrink = PortfolioOptimizationConfig(
        symbols=portfolio_assets,
        start_date=start_date,
        end_date=end_date,
        lookback_years=lookback_years,
        objective=ObjectiveType.MAX_SHARPE,
        constraints=constraints_standard,
        return_estimation_method="shrinkage",
        covariance_estimation_method="shrinkage"
    )
    
    # Run optimizations
    results = {}
    
    print(f"\n   📊 Max Sharpe (Historical)...")
    try:
        results['max_sharpe'] = optimizer.optimize_portfolio(config_sharpe)
        if results['max_sharpe'].success:
            print(f"      ✅ Success - Return: {results['max_sharpe'].expected_return:.3f}, "
                  f"Vol: {results['max_sharpe'].expected_volatility:.3f}, "
                  f"Sharpe: {results['max_sharpe'].sharpe_ratio:.3f}")
        else:
            print(f"      ❌ Failed: {results['max_sharpe'].message}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
    
    print(f"\n   📊 Min Volatility (Shrinkage)...")
    try:
        results['min_vol'] = optimizer.optimize_portfolio(config_min_vol)
        if results['min_vol'].success:
            print(f"      ✅ Success - Return: {results['min_vol'].expected_return:.3f}, "
                  f"Vol: {results['min_vol'].expected_volatility:.3f}, "
                  f"Sharpe: {results['min_vol'].sharpe_ratio:.3f}")
        else:
            print(f"      ❌ Failed: {results['min_vol'].message}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
    
    print(f"\n   📊 Max Sharpe (Shrinkage)...")
    try:
        results['max_sharpe_shrink'] = optimizer.optimize_portfolio(config_sharpe_shrink)
        if results['max_sharpe_shrink'].success:
            print(f"      ✅ Success - Return: {results['max_sharpe_shrink'].expected_return:.3f}, "
                  f"Vol: {results['max_sharpe_shrink'].expected_volatility:.3f}, "
                  f"Sharpe: {results['max_sharpe_shrink'].sharpe_ratio:.3f}")
        else:
            print(f"      ❌ Failed: {results['max_sharpe_shrink'].message}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
    
    # Step 6: Compare results
    print(f"\n6️⃣ Portfolio Comparison:")
    print(f"{'Strategy':<20} {'Return':<8} {'Vol':<8} {'Sharpe':<8} {'Positions':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        if result.success:
            positions = sum(1 for w in result.weights.values() if abs(w) > 0.01)
            print(f"{name:<20} {result.expected_return:<8.3f} {result.expected_volatility:<8.3f} "
                  f"{result.sharpe_ratio:<8.3f} {positions:<10}")
    
    # Step 7: Show detailed allocations for best strategy
    best_strategy = None
    best_sharpe = -999
    
    for name, result in results.items():
        if result.success and result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_strategy = (name, result)
    
    if best_strategy:
        name, result = best_strategy
        print(f"\n7️⃣ Best Strategy: {name} (Sharpe: {result.sharpe_ratio:.3f})")
        print(f"{'Asset':<8} {'Weight':<8} {'%':<8}")
        print("-" * 25)
        
        # Sort by weight for easier reading
        sorted_weights = sorted(result.weights.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for symbol, weight in sorted_weights:
            if abs(weight) > 0.001:  # Only show meaningful positions
                print(f"{symbol:<8} {weight:<8.3f} {weight*100:<8.1f}%")
    
    # Step 8: Performance attribution
    if best_strategy and inputs:
        name, result = best_strategy
        print(f"\n8️⃣ Risk Attribution for {name}:")
        
        weights_array = np.array([result.weights.get(symbol, 0.0) for symbol in portfolio_assets])
        individual_vols = inputs['volatilities']
        
        # Risk contribution calculation
        portfolio_variance = np.dot(weights_array, np.dot(inputs['covariance_matrix'], weights_array))
        marginal_contrib = np.dot(inputs['covariance_matrix'], weights_array) / np.sqrt(portfolio_variance)
        risk_contrib = weights_array * marginal_contrib
        
        print(f"{'Asset':<8} {'Weight':<8} {'Vol':<8} {'Risk%':<8}")
        print("-" * 35)
        
        for i, symbol in enumerate(portfolio_assets):
            if abs(weights_array[i]) > 0.001:
                risk_pct = (risk_contrib[i] / np.sum(risk_contrib)) * 100
                print(f"{symbol:<8} {weights_array[i]:<8.3f} {individual_vols[i]:<8.3f} {risk_pct:<8.1f}%")
    
    print(f"\n🎉 Portfolio optimization integration demo completed!")
    print(f"   Data layer successfully connected to optimization engine")
    print(f"   {len([r for r in results.values() if r.success])} out of {len(results)} optimizations succeeded")

if __name__ == "__main__":
    main()