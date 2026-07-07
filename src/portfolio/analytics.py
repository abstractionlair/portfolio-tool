"""Portfolio analytics and performance measurement."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path

from .portfolio import Portfolio
from .position import Position
from .exposures import ExposureType, FundExposureMap, ExposureCalculator
from ..data.market_data import MarketDataFetcher, calculate_returns

logger = logging.getLogger(__name__)


@dataclass
class CashFlow:
    """Represents a cash flow into or out of the portfolio."""
    date: datetime
    amount: float  # Positive for inflows, negative for outflows
    description: str = ""


@dataclass
class PortfolioAnalyticsSummary:
    """Summary of portfolio analytics."""
    period_start: datetime
    period_end: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # Days
    var_95: float
    cvar_95: float
    best_position: str
    worst_position: str
    best_position_return: float
    worst_position_return: float
    exposure_returns: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration_days': self.max_drawdown_duration,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'best_position': self.best_position,
            'best_position_return': self.best_position_return,
            'worst_position': self.worst_position,
            'worst_position_return': self.worst_position_return,
            'exposure_returns': self.exposure_returns or {}
        }


class PortfolioAnalytics:
    """Comprehensive analytics for portfolio performance and risk measurement."""
    
    def __init__(self, portfolio: Portfolio, market_data: MarketDataFetcher):
        """Initialize portfolio analytics.
        
        Args:
            portfolio: Portfolio object to analyze
            market_data: MarketDataFetcher for price data
        """
        self.portfolio = portfolio
        self.market_data = market_data
        
        # Cache for price data to avoid repeated API calls
        self._price_cache: Dict[str, pd.DataFrame] = {}
        
        logger.debug(f"Initialized analytics for portfolio '{portfolio.name}' with {len(portfolio)} positions")
    
    def _get_price_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Get price data with caching.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbols to price DataFrames
        """
        cache_key = f"{'-'.join(sorted(symbols))}_{start_date.date()}_{end_date.date()}"
        
        if cache_key in self._price_cache:
            logger.debug(f"Using cached price data for {len(symbols)} symbols")
            return self._price_cache[cache_key]
        
        try:
            price_data = self.market_data.fetch_price_history(symbols, start_date, end_date)
            self._price_cache[cache_key] = price_data
            logger.debug(f"Fetched price data for {len(price_data)} symbols")
            return price_data
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return {}
    
    def calculate_portfolio_values(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """Calculate portfolio values over time.
        
        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            Series of portfolio values indexed by date
        """
        symbols = list(self.portfolio.positions.keys())
        if not symbols:
            logger.warning("Portfolio has no positions")
            return pd.Series(dtype=float)
        
        price_data = self._get_price_data(symbols, start_date, end_date)
        if not price_data:
            raise ValueError("Could not fetch price data for portfolio positions")
        
        # Align all price series
        all_prices = {}
        for symbol in symbols:
            if symbol in price_data and not price_data[symbol].empty:
                all_prices[symbol] = price_data[symbol]['Adj Close']
        
        if not all_prices:
            raise ValueError("No valid price data available")
        
        # Create aligned DataFrame
        prices_df = pd.DataFrame(all_prices).dropna()
        
        if prices_df.empty:
            raise ValueError("No overlapping price data found")
        
        # Calculate portfolio values
        portfolio_values = pd.Series(index=prices_df.index, dtype=float)
        
        for date in prices_df.index:
            daily_value = self.portfolio.cash  # Start with cash
            
            for symbol, position in self.portfolio.positions.items():
                if symbol in prices_df.columns:
                    price = prices_df.loc[date, symbol]
                    daily_value += position.market_value(price)
            
            portfolio_values[date] = daily_value
        
        return portfolio_values
    
    def calculate_returns(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'daily'
    ) -> pd.Series:
        """Calculate portfolio returns over time.
        
        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            frequency: 'daily', 'monthly', or 'annual'
            
        Returns:
            Series of returns
        """
        portfolio_values = self.calculate_portfolio_values(start_date, end_date)
        
        if portfolio_values.empty:
            return pd.Series(dtype=float)
        
        # Calculate daily returns
        daily_returns = portfolio_values.pct_change().dropna()
        
        if frequency == 'daily':
            return daily_returns
        elif frequency == 'monthly':
            # Resample to month-end
            monthly_values = portfolio_values.resample('ME').last()
            return monthly_values.pct_change().dropna()
        elif frequency == 'annual':
            # Resample to year-end
            annual_values = portfolio_values.resample('YE').last()
            return annual_values.pct_change().dropna()
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
    
    def calculate_position_returns(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """Calculate returns for a specific position.
        
        Args:
            symbol: Symbol to analyze
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            Series of position returns
        """
        if symbol not in self.portfolio.positions:
            raise ValueError(f"Symbol {symbol} not found in portfolio")
        
        price_data = self._get_price_data([symbol], start_date, end_date)
        
        if symbol not in price_data or price_data[symbol].empty:
            raise ValueError(f"No price data available for {symbol}")
        
        return calculate_returns(price_data[symbol], period='daily')
    
    def time_weighted_return(
        self,
        start_date: datetime,
        end_date: datetime,
        cash_flows: Optional[List[CashFlow]] = None
    ) -> float:
        """Calculate time-weighted return accounting for cash flows.
        
        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            cash_flows: List of cash flows during period
            
        Returns:
            Time-weighted return
        """
        portfolio_values = self.calculate_portfolio_values(start_date, end_date)
        
        if portfolio_values.empty or len(portfolio_values) < 2:
            return 0.0
        
        if not cash_flows:
            # Simple return calculation
            return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0
        
        # Complex calculation with cash flows
        # This is a simplified implementation - more sophisticated methods exist
        adjusted_values = portfolio_values.copy()
        
        for cf in cash_flows:
            # Adjust values after cash flow date
            cf_date = pd.Timestamp(cf.date)
            mask = adjusted_values.index >= cf_date
            adjusted_values.loc[mask] -= cf.amount
        
        # Calculate return on adjusted values
        return (adjusted_values.iloc[-1] / adjusted_values.iloc[0]) - 1.0
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """Calculate return volatility (standard deviation).
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize the volatility
            
        Returns:
            Volatility
        """
        if returns.empty:
            return 0.0
        
        vol = returns.std()
        
        if annualize:
            # Assume daily returns, annualize with sqrt(252)
            vol *= np.sqrt(252)
        
        return vol
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if returns.empty:
            return 0.0
        
        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        excess_returns = returns - daily_rf
        
        excess_std = excess_returns.std()
        if excess_std == 0 or np.isclose(excess_std, 0):
            return 0.0
        
        return (excess_returns.mean() * 252) / (excess_std * np.sqrt(252))
    
    def calculate_max_drawdown(self, values: pd.Series) -> Dict[str, Any]:
        """Calculate maximum drawdown and recovery information.
        
        Args:
            values: Series of portfolio values
            
        Returns:
            Dictionary with drawdown statistics
        """
        if values.empty or len(values) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'peak_date': None,
                'trough_date': None,
                'recovery_date': None
            }
        
        # Calculate running maximum (peak)
        peak = values.expanding().max()
        
        # Calculate drawdown
        drawdown = (values - peak) / peak
        
        # Find maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_drawdown = drawdown.min()
        
        # Find the peak before max drawdown
        peak_idx = peak.loc[:max_dd_idx].idxmax()
        
        # Find recovery date (if any)
        recovery_idx = None
        post_trough = values.loc[max_dd_idx:]
        peak_value = values.loc[peak_idx]
        
        recovery_mask = post_trough >= peak_value
        if recovery_mask.any():
            recovery_idx = post_trough[recovery_mask].index[0]
        
        # Calculate duration
        if recovery_idx:
            duration = (recovery_idx - peak_idx).days
        else:
            duration = (values.index[-1] - peak_idx).days
        
        return {
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_duration': duration,
            'peak_date': peak_idx,
            'trough_date': max_dd_idx,
            'recovery_date': recovery_idx
        }
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """Calculate Value at Risk.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical' or 'parametric'
            
        Returns:
            Value at Risk (positive number for losses)
        """
        if returns.empty:
            return 0.0
        
        if method == 'historical':
            # Historical VaR - use empirical quantile
            return -returns.quantile(1 - confidence_level)
        elif method == 'parametric':
            # Parametric VaR - assume normal distribution
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence_level)
            return -(returns.mean() + z_score * returns.std())
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR (positive number for losses)
        """
        if returns.empty:
            return 0.0
        
        var_threshold = self.calculate_var(returns, confidence_level, 'historical')
        
        # CVaR is the mean of returns worse than VaR
        tail_returns = returns[returns <= -var_threshold]
        
        if tail_returns.empty:
            return var_threshold
        
        return -tail_returns.mean()
    
    def calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate information ratio vs benchmark.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0
        
        # Align series
        aligned_data = pd.DataFrame({'portfolio': returns, 'benchmark': benchmark_returns}).dropna()
        
        if aligned_data.empty:
            return 0.0
        
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return (excess_returns.mean() * 252) / (tracking_error * np.sqrt(252))
    
    def calculate_tracking_error(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate tracking error vs benchmark.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Annualized tracking error
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0
        
        # Align series
        aligned_data = pd.DataFrame({'portfolio': returns, 'benchmark': benchmark_returns}).dropna()
        
        if aligned_data.empty:
            return 0.0
        
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        return excess_returns.std() * np.sqrt(252)
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate portfolio beta.
        
        Args:
            returns: Portfolio returns
            market_returns: Market returns
            
        Returns:
            Beta
        """
        if returns.empty or market_returns.empty:
            return 0.0
        
        # Align series
        aligned_data = pd.DataFrame({'portfolio': returns, 'market': market_returns}).dropna()
        
        if aligned_data.empty or aligned_data['market'].var() == 0:
            return 0.0
        
        return aligned_data['portfolio'].cov(aligned_data['market']) / aligned_data['market'].var()
    
    def calculate_alpha(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate portfolio alpha (Jensen's alpha).
        
        Args:
            returns: Portfolio returns
            market_returns: Market returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Annualized alpha
        """
        if returns.empty or market_returns.empty:
            return 0.0
        
        beta = self.calculate_beta(returns, market_returns)
        
        # Convert to daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate excess returns
        portfolio_excess = returns.mean() - daily_rf
        market_excess = market_returns.mean() - daily_rf
        
        # Jensen's alpha: excess portfolio return - beta * excess market return
        alpha_daily = portfolio_excess - beta * market_excess
        
        return alpha_daily * 252  # Annualize
    
    def calculate_exposure_returns(
        self,
        fund_map: FundExposureMap,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Calculate returns by exposure type.
        
        Args:
            fund_map: Fund exposure map
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with exposure returns over time
        """
        calculator = ExposureCalculator(fund_map)
        
        # Get portfolio values over time
        symbols = list(self.portfolio.positions.keys())
        price_data = self._get_price_data(symbols, start_date, end_date)
        
        if not price_data:
            return pd.DataFrame()
        
        # Align all price data
        all_prices = {}
        for symbol in symbols:
            if symbol in price_data and not price_data[symbol].empty:
                all_prices[symbol] = price_data[symbol]['Adj Close']
        
        prices_df = pd.DataFrame(all_prices).dropna()
        
        if prices_df.empty:
            return pd.DataFrame()
        
        # Calculate exposure values over time
        exposure_values = {}
        
        for date in prices_df.index:
            daily_prices = prices_df.loc[date].to_dict()
            exposures = calculator.calculate_portfolio_exposures(self.portfolio, daily_prices)
            
            for exposure_type, value in exposures.items():
                if exposure_type not in exposure_values:
                    exposure_values[exposure_type] = {}
                exposure_values[exposure_type][date] = value
        
        # Convert to DataFrame and calculate returns
        exposure_df = pd.DataFrame(exposure_values).fillna(0)
        return exposure_df.pct_change().dropna()
    
    def exposure_attribution(
        self,
        fund_map: FundExposureMap,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[ExposureType, float]:
        """Attribute portfolio returns to each exposure type.
        
        Args:
            fund_map: Fund exposure map
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping exposure types to attributed returns
        """
        exposure_returns = self.calculate_exposure_returns(fund_map, start_date, end_date)
        
        if exposure_returns.empty:
            return {}
        
        # Calculate total returns for each exposure
        attribution = {}
        for exposure_type in exposure_returns.columns:
            total_return = (1 + exposure_returns[exposure_type]).prod() - 1
            attribution[exposure_type] = total_return
        
        return attribution
    
    def generate_analytics_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        fund_map: Optional[FundExposureMap] = None
    ) -> PortfolioAnalyticsSummary:
        """Generate comprehensive analytics summary.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            fund_map: Optional fund exposure map
            
        Returns:
            Analytics summary
        """
        try:
            # Calculate returns
            returns = self.calculate_returns(start_date, end_date, 'daily')
            
            if returns.empty:
                logger.warning("No returns data available for summary")
                return PortfolioAnalyticsSummary(
                    period_start=start_date,
                    period_end=end_date,
                    total_return=0.0,
                    annualized_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    max_drawdown_duration=0,
                    var_95=0.0,
                    cvar_95=0.0,
                    best_position="N/A",
                    worst_position="N/A",
                    best_position_return=0.0,
                    worst_position_return=0.0
                )
            
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            days = (end_date - start_date).days
            annualized_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
            
            volatility = self.calculate_volatility(returns)
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            
            # Portfolio values for drawdown calculation
            portfolio_values = self.calculate_portfolio_values(start_date, end_date)
            drawdown_info = self.calculate_max_drawdown(portfolio_values)
            
            var_95 = self.calculate_var(returns, 0.95)
            cvar_95 = self.calculate_cvar(returns, 0.95)
            
            # Position-level analysis
            best_position = "N/A"
            worst_position = "N/A"
            best_return = 0.0
            worst_return = 0.0
            
            try:
                position_returns = {}
                for symbol in self.portfolio.positions.keys():
                    try:
                        pos_returns = self.calculate_position_returns(symbol, start_date, end_date)
                        if not pos_returns.empty:
                            total_pos_return = (1 + pos_returns).prod() - 1
                            position_returns[symbol] = total_pos_return
                    except Exception as e:
                        logger.debug(f"Could not calculate returns for {symbol}: {e}")
                
                if position_returns:
                    best_symbol = max(position_returns, key=position_returns.get)
                    worst_symbol = min(position_returns, key=position_returns.get)
                    
                    best_position = best_symbol
                    best_return = position_returns[best_symbol]
                    worst_position = worst_symbol
                    worst_return = position_returns[worst_symbol]
                    
            except Exception as e:
                logger.debug(f"Error calculating position returns: {e}")
            
            # Exposure attribution
            exposure_returns = None
            if fund_map:
                try:
                    exposure_attr = self.exposure_attribution(fund_map, start_date, end_date)
                    exposure_returns = {exp.value: ret for exp, ret in exposure_attr.items()}
                except Exception as e:
                    logger.debug(f"Error calculating exposure attribution: {e}")
            
            return PortfolioAnalyticsSummary(
                period_start=start_date,
                period_end=end_date,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=drawdown_info['max_drawdown'],
                max_drawdown_duration=drawdown_info['max_drawdown_duration'],
                var_95=var_95,
                cvar_95=cvar_95,
                best_position=best_position,
                worst_position=worst_position,
                best_position_return=best_return,
                worst_position_return=worst_return,
                exposure_returns=exposure_returns
            )
            
        except Exception as e:
            logger.error(f"Error generating analytics summary: {e}")
            raise