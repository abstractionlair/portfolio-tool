"""
Portfolio Optimization Integration Layer

This module provides the main integration between the data layer and optimization engine,
enabling end-to-end portfolio optimization workflows with real market data.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..data.interfaces import LogicalDataType, RawDataType
from ..data.providers import TransformedDataProvider, RawDataProviderCoordinator
from .engine import OptimizationEngine, OptimizationConstraints, OptimizationResult, ObjectiveType
from .estimators import ReturnEstimator

logger = logging.getLogger(__name__)


@dataclass
class PortfolioOptimizationConfig:
    """Configuration for end-to-end portfolio optimization."""
    # Asset universe
    symbols: List[str]
    
    # Time period
    start_date: date
    end_date: date
    lookback_years: int = 2
    
    # Optimization settings
    objective: ObjectiveType = ObjectiveType.MAX_SHARPE
    constraints: Optional[OptimizationConstraints] = None
    
    # Estimation methods
    return_estimation_method: str = "historical"  # historical, capm, shrinkage
    covariance_estimation_method: str = "sample"  # sample, shrinkage, ewma, garch
    frequency: str = "daily"
    
    # Risk-free rate settings
    risk_free_rate: Optional[float] = None  # If None, will be estimated from data


class PortfolioOptimizer:
    """
    Main portfolio optimization class that integrates data layer with optimization engine.
    
    This class provides a simplified interface for end-to-end portfolio optimization
    using real market data from the data layer.
    """
    
    def __init__(self, data_provider: Optional[TransformedDataProvider] = None):
        """
        Initialize portfolio optimizer.
        
        Args:
            data_provider: Data provider for market data. If None, creates default.
        """
        if data_provider is None:
            raw_coordinator = RawDataProviderCoordinator()
            self.data_provider = TransformedDataProvider(raw_coordinator)
        else:
            self.data_provider = data_provider
        
        # Will be initialized when needed with portfolio analytics
        self._optimization_engine = None
        
        logger.info("Initialized PortfolioOptimizer with data provider")
    
    def optimize_portfolio(
        self,
        config: PortfolioOptimizationConfig
    ) -> OptimizationResult:
        """
        Perform end-to-end portfolio optimization.
        
        Args:
            config: Portfolio optimization configuration
            
        Returns:
            OptimizationResult with optimal weights and diagnostics
        """
        logger.info(f"Starting portfolio optimization for {len(config.symbols)} assets")
        
        try:
            # Step 1: Prepare return estimation data
            returns_data = self._prepare_returns_data(config)
            
            # Step 2: Estimate expected returns
            expected_returns = self._estimate_expected_returns(returns_data, config)
            
            # Step 3: Estimate covariance matrix
            covariance_matrix = self._estimate_covariance_matrix(returns_data, config)
            
            # Step 4: Get risk-free rate
            risk_free_rate = self._get_risk_free_rate(config)
            
            # Step 5: Setup constraints
            constraints = config.constraints or OptimizationConstraints()
            
            # Step 6: Perform optimization (simplified without portfolio analytics)
            result = self._optimize_with_data(
                config.symbols,
                expected_returns,
                covariance_matrix,
                constraints,
                config.objective,
                risk_free_rate
            )
            
            logger.info(f"Optimization completed: {result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return OptimizationResult(
                weights={symbol: 0.0 for symbol in config.symbols},
                objective_value=0.0,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                exposures={},
                total_notional=0.0,
                success=False,
                message=f"Optimization failed: {str(e)}"
            )
    
    def _prepare_returns_data(self, config: PortfolioOptimizationConfig) -> pd.DataFrame:
        """Prepare returns data for all symbols."""
        logger.debug("Preparing returns data for optimization")
        
        # Calculate extended start date for return calculations
        lookback_days = config.lookback_years * 365
        extended_start = config.start_date - timedelta(days=lookback_days + 30)
        
        returns_data = {}
        
        for symbol in config.symbols:
            try:
                # Get total returns including dividends
                returns = self.data_provider.get_data(
                    LogicalDataType.TOTAL_RETURN,
                    start=extended_start,
                    end=config.end_date,
                    ticker=symbol,
                    frequency=config.frequency
                )
                
                if returns is not None and not returns.empty:
                    # Filter to actual analysis period - handle timezone-aware indices
                    start_timestamp = pd.Timestamp(config.start_date)
                    if hasattr(returns.index, 'tz') and returns.index.tz is not None:
                        # Localize start timestamp to match data timezone
                        start_timestamp = start_timestamp.tz_localize(returns.index.tz)
                    
                    returns = returns[returns.index >= start_timestamp]
                    returns_data[symbol] = returns
                    logger.debug(f"Loaded {len(returns)} returns for {symbol}")
                else:
                    logger.warning(f"No returns data for {symbol}")
                    
            except Exception as e:
                logger.warning(f"Failed to load returns for {symbol}: {e}")
        
        if not returns_data:
            raise ValueError("No returns data available for any symbols")
        
        # Align all return series
        returns_df = pd.DataFrame(returns_data).dropna()
        logger.info(f"Prepared returns data: {len(returns_df)} observations, {len(returns_df.columns)} assets")
        
        return returns_df
    
    def _estimate_expected_returns(
        self, 
        returns_data: pd.DataFrame, 
        config: PortfolioOptimizationConfig
    ) -> np.ndarray:
        """Estimate expected returns using specified method."""
        logger.debug(f"Estimating expected returns using {config.return_estimation_method} method")
        
        if config.return_estimation_method == "historical":
            # Simple historical mean
            mean_returns = returns_data.mean()
            
            # Annualize based on frequency
            if config.frequency == "daily":
                annual_returns = mean_returns * 252
            elif config.frequency == "weekly":
                annual_returns = mean_returns * 52
            elif config.frequency == "monthly":
                annual_returns = mean_returns * 12
            else:
                annual_returns = mean_returns
                
        elif config.return_estimation_method == "shrinkage":
            # Shrinkage toward grand mean
            historical_returns = returns_data.mean()
            if config.frequency == "daily":
                historical_returns *= 252
            elif config.frequency == "weekly":
                historical_returns *= 52
            elif config.frequency == "monthly":
                historical_returns *= 12
                
            grand_mean = historical_returns.mean()
            shrinkage_intensity = 0.2
            annual_returns = (1 - shrinkage_intensity) * historical_returns + shrinkage_intensity * grand_mean
            
        else:
            # Default to historical
            mean_returns = returns_data.mean()
            if config.frequency == "daily":
                annual_returns = mean_returns * 252
            else:
                annual_returns = mean_returns
        
        # Convert to array in symbol order
        expected_returns = np.array([annual_returns.get(symbol, 0.0) for symbol in config.symbols])
        
        logger.debug(f"Expected returns range: {expected_returns.min():.3f} to {expected_returns.max():.3f}")
        return expected_returns
    
    def _estimate_covariance_matrix(
        self, 
        returns_data: pd.DataFrame, 
        config: PortfolioOptimizationConfig
    ) -> np.ndarray:
        """Estimate covariance matrix using specified method."""
        logger.debug(f"Estimating covariance matrix using {config.covariance_estimation_method} method")
        
        # Select only symbols we need
        available_symbols = [s for s in config.symbols if s in returns_data.columns]
        
        if not available_symbols:
            # Return identity matrix as fallback
            return np.eye(len(config.symbols)) * 0.04
        
        if config.covariance_estimation_method == "sample":
            # Sample covariance
            cov_matrix = returns_data[available_symbols].cov().values
            
        elif config.covariance_estimation_method == "shrinkage":
            # Shrinkage toward diagonal
            sample_cov = returns_data[available_symbols].cov().values
            target = np.diag(np.diag(sample_cov))
            shrinkage_intensity = 0.1
            cov_matrix = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target
            
        else:
            # Default to sample
            cov_matrix = returns_data[available_symbols].cov().values
        
        # Annualize
        if config.frequency == "daily":
            cov_matrix *= 252
        elif config.frequency == "weekly":
            cov_matrix *= 52
        elif config.frequency == "monthly":
            cov_matrix *= 12
        
        # Handle missing symbols by padding with identity
        if len(available_symbols) < len(config.symbols):
            full_cov = np.eye(len(config.symbols)) * 0.04
            symbol_map = {symbol: i for i, symbol in enumerate(config.symbols)}
            available_indices = [symbol_map[s] for s in available_symbols]
            
            for i, idx_i in enumerate(available_indices):
                for j, idx_j in enumerate(available_indices):
                    full_cov[idx_i, idx_j] = cov_matrix[i, j]
            
            return full_cov
        
        logger.debug(f"Covariance matrix shape: {cov_matrix.shape}")
        return cov_matrix
    
    def _get_risk_free_rate(self, config: PortfolioOptimizationConfig) -> float:
        """Get risk-free rate for optimization."""
        if config.risk_free_rate is not None:
            return config.risk_free_rate
        
        try:
            # Try to get Treasury 3-month rate
            rf_data = self.data_provider.get_data(
                LogicalDataType.NOMINAL_RISK_FREE,
                start=config.start_date,
                end=config.end_date,
                frequency=config.frequency
            )
            
            if rf_data is not None and not rf_data.empty:
                # Convert to annual rate
                rf_rate = rf_data.iloc[-1] / 100.0  # Convert from percentage
                logger.debug(f"Using estimated risk-free rate: {rf_rate:.3f}")
                return rf_rate
            
        except Exception as e:
            logger.warning(f"Failed to get risk-free rate from data: {e}")
        
        # Default risk-free rate
        default_rf = 0.02
        logger.debug(f"Using default risk-free rate: {default_rf:.3f}")
        return default_rf
    
    def _optimize_with_data(
        self,
        symbols: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        objective: ObjectiveType,
        risk_free_rate: float
    ) -> OptimizationResult:
        """Perform optimization using prepared data (simplified version without portfolio analytics)."""
        logger.debug("Performing optimization with prepared data")
        
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization. Install with: pip install cvxpy")
        
        n = len(symbols)
        weights = cp.Variable(n)
        
        # Build constraints
        constraints_list = []
        
        # Weight sum constraint
        constraints_list.append(cp.sum(weights) == 1.0)
        
        # Individual weight bounds
        if constraints.long_only:
            constraints_list.append(weights >= constraints.min_weight)
        else:
            constraints_list.append(weights >= constraints.min_weight)
        
        constraints_list.append(weights <= constraints.max_weight)
        
        # Build objective based on type
        if objective == ObjectiveType.MAX_SHARPE:
            # Portfolio return and variance
            portfolio_return = expected_returns.T @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            
            # Maximize Sharpe ratio (approximate via maximize return / sqrt(variance))
            # Using a reformulation that's convex: maximize (return - rf) / sqrt(variance)
            excess_return = portfolio_return - risk_free_rate
            
            # Since we can't directly maximize Sharpe, we'll minimize -Sharpe approximation
            # This is an approximation - exact Sharpe optimization requires more complex formulation
            objective_func = cp.Minimize(-excess_return + 0.5 * portfolio_variance)
            
        elif objective == ObjectiveType.MIN_VOLATILITY:
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            objective_func = cp.Minimize(portfolio_variance)
            
        elif objective == ObjectiveType.MAX_RETURN:
            portfolio_return = expected_returns.T @ weights
            objective_func = cp.Maximize(portfolio_return)
            
        else:
            # Default to min volatility
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            objective_func = cp.Minimize(portfolio_variance)
        
        # Solve optimization
        problem = cp.Problem(objective_func, constraints_list)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=10000)
            
            if problem.status not in ["infeasible", "unbounded", "infeasible_inaccurate"]:
                weights_result = weights.value
                if weights_result is not None:
                    # Clean up tiny weights
                    weights_result[np.abs(weights_result) < constraints.min_position_size] = 0.0
                    
                    # Normalize to ensure weights sum to 1
                    if np.sum(weights_result) > 0:
                        weights_result = weights_result / np.sum(weights_result)
                    
                    weights_dict = {symbol: float(w) for symbol, w in zip(symbols, weights_result)}
                    
                    # Calculate portfolio metrics
                    portfolio_return, portfolio_volatility, sharpe_ratio = self._calculate_portfolio_metrics(
                        weights_result, expected_returns, covariance_matrix, risk_free_rate
                    )
                    
                    return OptimizationResult(
                        weights=weights_dict,
                        objective_value=float(problem.value),
                        expected_return=portfolio_return,
                        expected_volatility=portfolio_volatility,
                        sharpe_ratio=sharpe_ratio,
                        exposures={},  # Simplified - no exposure calculation
                        total_notional=1.0,
                        success=True,
                        message="Optimization completed successfully",
                        solver_status=problem.status
                    )
            
            return OptimizationResult(
                weights={symbol: 0.0 for symbol in symbols},
                objective_value=0.0,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                exposures={},
                total_notional=0.0,
                success=False,
                message=f"Optimization failed with status: {problem.status}",
                solver_status=problem.status
            )
            
        except Exception as e:
            logger.error(f"Solver error: {e}")
            return OptimizationResult(
                weights={symbol: 0.0 for symbol in symbols},
                objective_value=0.0,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                exposures={},
                total_notional=0.0,
                success=False,
                message=f"Solver error: {str(e)}"
            )
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float
    ) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio."""
        # Portfolio expected return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0))
        
        # Sharpe ratio
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def get_optimization_inputs(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        lookback_years: int = 2,
        frequency: str = "daily"
    ) -> Dict:
        """
        Get optimization inputs (returns, covariance) for analysis.
        
        This method allows users to inspect the data that would be used
        for optimization before running the actual optimization.
        """
        config = PortfolioOptimizationConfig(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
            frequency=frequency
        )
        
        returns_data = self._prepare_returns_data(config)
        expected_returns = self._estimate_expected_returns(returns_data, config)
        covariance_matrix = self._estimate_covariance_matrix(returns_data, config)
        risk_free_rate = self._get_risk_free_rate(config)
        
        return {
            "returns_data": returns_data,
            "expected_returns": expected_returns,
            "covariance_matrix": covariance_matrix,
            "correlation_matrix": self._cov_to_corr(covariance_matrix),
            "volatilities": np.sqrt(np.diag(covariance_matrix)),
            "risk_free_rate": risk_free_rate,
            "symbols": symbols
        }
    
    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        volatilities = np.sqrt(np.diag(cov_matrix))
        # Avoid division by zero
        volatilities[volatilities == 0] = 1.0
        corr_matrix = cov_matrix / np.outer(volatilities, volatilities)
        return corr_matrix