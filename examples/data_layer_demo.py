#!/usr/bin/env python3
"""
Data Layer Demo Examples

This module contains clean, testable examples of using the portfolio tool's data layer.
These examples can be run independently or called from notebooks.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional

from src.data.interfaces import LogicalDataType, RawDataType
from src.data.providers import (
    RawDataProviderCoordinator, 
    TransformedDataProvider
)


class DataLayerDemo:
    """Clean examples demonstrating data layer capabilities."""
    
    def __init__(self):
        """Initialize the data providers."""
        self.raw_coordinator = RawDataProviderCoordinator()
        self.transformed_provider = TransformedDataProvider(self.raw_coordinator)
        
    def get_basic_price_data(
        self, 
        symbols: List[str], 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        Get basic price data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with symbols as columns, dates as index
        """
        print(f"📈 Fetching price data for {symbols}")
        
        try:
            prices_df = self.transformed_provider.get_universe_data(
                data_type=RawDataType.ADJUSTED_CLOSE,
                tickers=symbols,
                start=start_date,
                end=end_date,
                frequency="daily"
            )
            
            print(f"✅ Successfully fetched data for {len(prices_df.columns)} symbols")
            print(f"📊 Dataset shape: {prices_df.shape}")
            
            return prices_df
            
        except Exception as e:
            print(f"❌ Error fetching price data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_returns_comparison(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> Dict[str, pd.Series]:
        """
        Calculate different types of returns for comparison.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of return series by type
        """
        print(f"🧮 Calculating returns for {symbol}")
        
        returns_data = {}
        
        # Total returns (includes dividends)
        try:
            total_returns = self.transformed_provider.get_data(
                data_type=LogicalDataType.TOTAL_RETURN,
                start=start_date,
                end=end_date,
                ticker=symbol,
                frequency="daily"
            )
            returns_data['Total Returns'] = total_returns
            valid_count = len(total_returns.dropna())
            print(f"✅ Total Returns: {valid_count} valid values")
            
        except Exception as e:
            print(f"❌ Total Returns failed: {str(e)}")
        
        # Simple returns (price only)
        try:
            simple_returns = self.transformed_provider.get_data(
                data_type=LogicalDataType.SIMPLE_RETURN,
                start=start_date,
                end=end_date,
                ticker=symbol,
                frequency="daily"
            )
            returns_data['Simple Returns'] = simple_returns
            valid_count = len(simple_returns.dropna())
            print(f"✅ Simple Returns: {valid_count} valid values")
            
        except Exception as e:
            print(f"❌ Simple Returns failed: {str(e)}")
        
        # Log returns
        try:
            log_returns = self.transformed_provider.get_data(
                data_type=LogicalDataType.LOG_RETURN,
                start=start_date,
                end=end_date,
                ticker=symbol,
                frequency="daily"
            )
            returns_data['Log Returns'] = log_returns
            valid_count = len(log_returns.dropna())
            print(f"✅ Log Returns: {valid_count} valid values")
            
        except Exception as e:
            print(f"❌ Log Returns failed: {str(e)}")
        
        return returns_data
    
    def get_economic_indicators(
        self, 
        start_date: date, 
        end_date: date
    ) -> Dict[str, pd.Series]:
        """
        Get economic indicator data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of economic data series
        """
        print("🏛️ Fetching economic indicators...")
        
        economic_data = {}
        indicators = {
            'Treasury 3M': RawDataType.TREASURY_3M,
            'Fed Funds Rate': RawDataType.FED_FUNDS,
            'CPI Index': RawDataType.CPI_INDEX
        }
        
        for name, data_type in indicators.items():
            try:
                data = self.raw_coordinator.get_data(
                    data_type=data_type,
                    start=start_date,
                    end=end_date,
                    frequency="daily"
                )
                
                if not data.empty:
                    economic_data[name] = data
                    print(f"✅ {name}: {len(data)} data points")
                else:
                    print(f"⚠️ {name}: No data returned")
                    
            except Exception as e:
                print(f"❌ {name}: {str(e)}")
        
        return economic_data
    
    def analyze_portfolio_returns(
        self, 
        portfolio: Dict[str, float],
        start_date: date,
        end_date: date
    ) -> Dict[str, any]:
        """
        Analyze returns for a portfolio of assets.
        
        Args:
            portfolio: Dictionary of {symbol: weight}
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with portfolio analysis results
        """
        print(f"📊 Analyzing portfolio: {list(portfolio.keys())}")
        
        # Get returns for each asset
        asset_returns = {}
        for symbol in portfolio.keys():
            try:
                returns = self.transformed_provider.get_data(
                    data_type=LogicalDataType.TOTAL_RETURN,
                    start=start_date,
                    end=end_date,
                    ticker=symbol,
                    frequency="daily"
                )
                asset_returns[symbol] = returns
                print(f"✅ {symbol}: {len(returns.dropna())} returns")
                
            except Exception as e:
                print(f"❌ {symbol}: {str(e)}")
        
        if not asset_returns:
            print("⚠️ No asset returns available")
            return {}
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(asset_returns)
        
        # Calculate portfolio metrics
        weights = pd.Series(portfolio)
        portfolio_returns = (returns_df.fillna(0) * weights).sum(axis=1)
        
        results = {
            'returns_df': returns_df,
            'portfolio_returns': portfolio_returns,
            'total_return': portfolio_returns.sum(),
            'volatility': portfolio_returns.std(),
            'best_day': portfolio_returns.max(),
            'worst_day': portfolio_returns.min(),
            'weights': weights
        }
        
        print(f"📈 Portfolio total return: {results['total_return']:.2%}")
        print(f"📊 Portfolio volatility: {results['volatility']:.2%}")
        
        return results


def demo_basic_usage():
    """Run basic data layer demonstration."""
    print("🚀 Data Layer Basic Demo")
    print("=" * 50)
    
    demo = DataLayerDemo()
    
    # Test parameters
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date = date(2023, 1, 1)
    end_date = date(2023, 6, 30)
    
    # 1. Basic price data
    print("\n1️⃣ Basic Price Data")
    prices_df = demo.get_basic_price_data(symbols, start_date, end_date)
    if not prices_df.empty:
        print(f"Price range: ${prices_df.min().min():.2f} - ${prices_df.max().max():.2f}")
    
    # 2. Returns calculation
    print("\n2️⃣ Returns Calculation")
    returns_data = demo.calculate_returns_comparison('AAPL', start_date, end_date)
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        print(f"Returns data shape: {returns_df.shape}")
        
        # Show performance comparison
        print("\n📊 Performance Comparison (6 months):")
        for col in returns_df.columns:
            if not returns_df[col].empty:
                cumulative = (1 + returns_df[col].fillna(0)).prod() - 1
                print(f"   {col}: {cumulative:.2%}")
    
    # 3. Economic data
    print("\n3️⃣ Economic Indicators")
    economic_data = demo.get_economic_indicators(start_date, end_date)
    print(f"Retrieved {len(economic_data)} economic series")
    
    # 4. Portfolio analysis
    print("\n4️⃣ Portfolio Analysis")
    portfolio = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
    portfolio_results = demo.analyze_portfolio_returns(portfolio, start_date, end_date)
    
    print("\n✅ Demo completed successfully!")
    return {
        'prices': prices_df,
        'returns': returns_data,
        'economic': economic_data,
        'portfolio': portfolio_results
    }


if __name__ == "__main__":
    # Run the demo when script is executed directly
    results = demo_basic_usage()