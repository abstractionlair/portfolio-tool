"""Specific optimization method implementations."""

import numpy as np
from typing import List, Dict, Optional
import logging
from scipy.optimize import minimize
import time

from .engine import OptimizationResult, OptimizationConstraints

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    """Mean-variance optimization methods (Markowitz framework)."""
    
    def __init__(self, engine):
        """Initialize with reference to main optimization engine."""
        self.engine = engine
    
    def optimize_max_sharpe(
        self,
        symbols: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        risk_free_rate: float = 0.02
    ) -> OptimizationResult:
        """Optimize portfolio to maximize Sharpe ratio.
        
        Args:
            symbols: List of asset symbols
            expected_returns: Expected returns vector
            covariance_matrix: Return covariance matrix
            constraints: Optimization constraints
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            OptimizationResult with optimal weights
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization. Install with: pip install cvxpy")
        
        start_time = time.time()
        n = len(symbols)
        weights = cp.Variable(n)
        
        # Portfolio return and variance
        portfolio_return = expected_returns.T @ weights
        portfolio_variance = cp.quad_form(weights, covariance_matrix)
        
        # For max Sharpe, we solve the equivalent problem of maximizing
        # (return - rf) / sqrt(variance) which can be reformulated
        excess_return = portfolio_return - risk_free_rate
        
        # Use the standard quadratic formulation: max (μ - rf)^T w subject to w^T Σ w <= 1
        # Then scale the result
        objective = cp.Maximize(excess_return)
        
        # Build constraints
        constraints_list = self.engine._build_basic_constraints(weights, symbols, constraints)
        
        # Add variance constraint for Sharpe optimization
        constraints_list.append(portfolio_variance <= 1.0)
        
        # Add exposure constraints if any
        exposure_constraints = self._build_exposure_constraints_mv(weights, symbols, constraints)
        constraints_list.extend(exposure_constraints)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=10000)
            solve_time = time.time() - start_time
            
            if problem.status in cp.settings.SOLUTION_PRESENT:
                weights_raw = weights.value
                if weights_raw is not None:
                    # Scale weights to sum to 1
                    weights_sum = np.sum(weights_raw)
                    if weights_sum > 1e-10:
                        weights_scaled = weights_raw / weights_sum
                    else:
                        weights_scaled = np.ones(n) / n  # Equal weights fallback
                    
                    # Clean up tiny weights
                    weights_scaled[np.abs(weights_scaled) < constraints.min_position_size] = 0.0
                    
                    # Renormalize
                    if np.sum(weights_scaled) > 0:
                        weights_scaled = weights_scaled / np.sum(weights_scaled)
                    
                    weights_dict = {symbol: float(w) for symbol, w in zip(symbols, weights_scaled)}
                    
                    # Calculate portfolio metrics with scaled weights
                    portfolio_return, portfolio_vol, sharpe_ratio = self.engine.calculate_portfolio_metrics(
                        weights_dict, expected_returns, covariance_matrix, symbols, risk_free_rate
                    )
                    
                    # Calculate exposures
                    exposures = self._calculate_exposures(symbols, weights_dict)
                    
                    # Calculate total notional
                    total_notional = self.engine._calculate_total_notional(symbols, weights_dict)
                    
                    return OptimizationResult(
                        weights=weights_dict,
                        objective_value=float(sharpe_ratio),
                        expected_return=float(portfolio_return),
                        expected_volatility=float(portfolio_vol),
                        sharpe_ratio=float(sharpe_ratio),
                        exposures=exposures,
                        total_notional=total_notional,
                        success=True,
                        message="Max Sharpe optimization completed successfully",
                        solver_status=problem.status,
                        solve_time=solve_time
                    )
            
            return self._create_failed_result(symbols, f"Optimization failed with status: {problem.status}")
            
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"Solver error in max Sharpe optimization: {e}")
            return self._create_failed_result(symbols, f"Solver error: {str(e)}", solve_time)
    
    def optimize_min_volatility(
        self,
        symbols: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Optimize portfolio to minimize volatility.
        
        Args:
            symbols: List of asset symbols
            expected_returns: Expected returns vector
            covariance_matrix: Return covariance matrix
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult with minimum variance weights
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        start_time = time.time()
        n = len(symbols)
        weights = cp.Variable(n)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, covariance_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Build constraints
        constraints_list = self.engine._build_basic_constraints(weights, symbols, constraints)
        
        # Add exposure constraints
        exposure_constraints = self._build_exposure_constraints_mv(weights, symbols, constraints)
        constraints_list.extend(exposure_constraints)
        
        # Target volatility constraint if specified
        if constraints.target_volatility is not None:
            constraints_list.append(cp.sqrt(portfolio_variance) <= constraints.target_volatility)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=10000)
            solve_time = time.time() - start_time
            
            if problem.status in cp.settings.SOLUTION_PRESENT:
                weights_result = weights.value
                if weights_result is not None:
                    # Clean up tiny weights
                    weights_result[np.abs(weights_result) < constraints.min_position_size] = 0.0
                    
                    # Normalize
                    if np.sum(weights_result) > 0:
                        weights_result = weights_result / np.sum(weights_result)
                    
                    weights_dict = {symbol: float(w) for symbol, w in zip(symbols, weights_result)}
                    
                    # Calculate portfolio metrics
                    portfolio_return, portfolio_vol, sharpe_ratio = self.engine.calculate_portfolio_metrics(
                        weights_dict, expected_returns, covariance_matrix, symbols
                    )
                    
                    # Calculate exposures
                    exposures = self._calculate_exposures(symbols, weights_dict)
                    
                    # Calculate total notional
                    total_notional = self.engine._calculate_total_notional(symbols, weights_dict)
                    
                    return OptimizationResult(
                        weights=weights_dict,
                        objective_value=float(portfolio_vol),
                        expected_return=float(portfolio_return),
                        expected_volatility=float(portfolio_vol),
                        sharpe_ratio=float(sharpe_ratio),
                        exposures=exposures,
                        total_notional=total_notional,
                        success=True,
                        message="Min volatility optimization completed successfully",
                        solver_status=problem.status,
                        solve_time=solve_time
                    )
            
            return self._create_failed_result(symbols, f"Optimization failed with status: {problem.status}")
            
        except Exception as e:
            solve_time = time.time() - start_time
            return self._create_failed_result(symbols, f"Solver error: {str(e)}", solve_time)
    
    def optimize_max_return(
        self,
        symbols: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Optimize portfolio to maximize expected return.
        
        Args:
            symbols: List of asset symbols
            expected_returns: Expected returns vector
            covariance_matrix: Return covariance matrix
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult with maximum return weights
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        start_time = time.time()
        n = len(symbols)
        weights = cp.Variable(n)
        
        # Objective: maximize portfolio return
        portfolio_return = expected_returns.T @ weights
        objective = cp.Maximize(portfolio_return)
        
        # Build constraints
        constraints_list = self.engine._build_basic_constraints(weights, symbols, constraints)
        
        # Add volatility constraint if target specified
        if constraints.target_volatility is not None:
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            constraints_list.append(cp.sqrt(portfolio_variance) <= constraints.target_volatility)
        
        # Add exposure constraints
        exposure_constraints = self._build_exposure_constraints_mv(weights, symbols, constraints)
        constraints_list.extend(exposure_constraints)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=10000)
            solve_time = time.time() - start_time
            
            if problem.status in cp.settings.SOLUTION_PRESENT:
                weights_result = weights.value
                if weights_result is not None:
                    # Clean and normalize weights
                    weights_result[np.abs(weights_result) < constraints.min_position_size] = 0.0
                    if np.sum(weights_result) > 0:
                        weights_result = weights_result / np.sum(weights_result)
                    
                    weights_dict = {symbol: float(w) for symbol, w in zip(symbols, weights_result)}
                    
                    # Calculate portfolio metrics
                    portfolio_return, portfolio_vol, sharpe_ratio = self.engine.calculate_portfolio_metrics(
                        weights_dict, expected_returns, covariance_matrix, symbols
                    )
                    
                    # Calculate exposures
                    exposures = self._calculate_exposures(symbols, weights_dict)
                    
                    # Calculate total notional
                    total_notional = self.engine._calculate_total_notional(symbols, weights_dict)
                    
                    return OptimizationResult(
                        weights=weights_dict,
                        objective_value=float(portfolio_return),
                        expected_return=float(portfolio_return),
                        expected_volatility=float(portfolio_vol),
                        sharpe_ratio=float(sharpe_ratio),
                        exposures=exposures,
                        total_notional=total_notional,
                        success=True,
                        message="Max return optimization completed successfully",
                        solver_status=problem.status,
                        solve_time=solve_time
                    )
            
            return self._create_failed_result(symbols, f"Optimization failed with status: {problem.status}")
            
        except Exception as e:
            solve_time = time.time() - start_time
            return self._create_failed_result(symbols, f"Solver error: {str(e)}", solve_time)
    
    def _build_exposure_constraints_mv(self, weights, symbols, constraints):
        """Build exposure constraints for mean-variance optimization."""
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        constraints_list = []
        
        if not constraints.max_exposure_per_type and not constraints.min_exposure_per_type:
            return constraints_list
        
        # Get all exposure types mentioned in constraints
        all_exposure_types = set(constraints.max_exposure_per_type.keys()) | set(constraints.min_exposure_per_type.keys())
        
        for exp_type in all_exposure_types:
            # Build exposure vector for this type
            exposure_vector = np.zeros(len(symbols))
            for i, symbol in enumerate(symbols):
                fund_def = self.engine.fund_map.get_fund_definition(symbol)
                if fund_def:
                    exposure_vector[i] = fund_def.exposures.get(exp_type, 0.0)
                else:
                    # Default exposure mapping
                    if symbol.upper() in ['SPY', 'QQQ', 'IWM', 'VTI']:
                        if exp_type.value == 'US_LARGE_EQUITY':
                            exposure_vector[i] = 1.0
                    elif symbol.upper() in ['TLT', 'AGG', 'IEF']:
                        if exp_type.value == 'BONDS':
                            exposure_vector[i] = 1.0
            
            # Add constraints
            exposure_value = exposure_vector @ weights
            
            if exp_type in constraints.max_exposure_per_type:
                max_exp = constraints.max_exposure_per_type[exp_type]
                constraints_list.append(exposure_value <= max_exp)
            
            if exp_type in constraints.min_exposure_per_type:
                min_exp = constraints.min_exposure_per_type[exp_type]
                constraints_list.append(exposure_value >= min_exp)
        
        return constraints_list
    
    def _calculate_exposures(self, symbols: List[str], weights: Dict[str, float]) -> Dict:
        """Calculate portfolio exposures given weights."""
        exposures = {}
        
        for symbol, weight in weights.items():
            if abs(weight) > 1e-6:
                fund_def = self.engine.fund_map.get_fund_definition(symbol)
                if fund_def:
                    for exp_type, exp_amount in fund_def.exposures.items():
                        if exp_type not in exposures:
                            exposures[exp_type] = 0.0
                        exposures[exp_type] += weight * exp_amount
        
        return exposures
    
    def _create_failed_result(self, symbols: List[str], message: str, solve_time: float = 0.0) -> OptimizationResult:
        """Create a failed optimization result."""
        return OptimizationResult(
            weights={symbol: 0.0 for symbol in symbols},
            objective_value=0.0,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            exposures={},
            total_notional=0.0,
            success=False,
            message=message,
            solve_time=solve_time
        )


class RiskParityOptimizer:
    """Risk parity optimization methods."""
    
    def __init__(self, engine):
        """Initialize with reference to main optimization engine."""
        self.engine = engine
    
    def optimize_risk_parity(
        self,
        symbols: List[str],
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        use_leverage: bool = True,
        target_volatility: Optional[float] = None
    ) -> OptimizationResult:
        """Allocate risk equally across assets.
        
        Args:
            symbols: List of asset symbols
            covariance_matrix: Return covariance matrix
            constraints: Optimization constraints
            use_leverage: Whether to use leverage to achieve target volatility
            target_volatility: Target portfolio volatility
            
        Returns:
            OptimizationResult with risk parity weights
        """
        start_time = time.time()
        n = len(symbols)
        
        # Risk parity objective function
        def risk_parity_objective(weights):
            """Objective function for risk parity optimization."""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            if portfolio_vol == 0:
                return 1e10  # Large penalty for zero volatility
            
            # Marginal risk contributions
            marginal_contribs = np.dot(covariance_matrix, weights) / portfolio_vol
            
            # Risk contributions
            risk_contribs = weights * marginal_contribs
            
            # Target risk contribution (equal for all assets)
            target_contrib = portfolio_vol / n
            
            # Sum of squared deviations from equal risk contribution
            return np.sum((risk_contribs - target_contrib) ** 2)
        
        # Initial guess - equal weights
        x0 = np.ones(n) / n
        
        # Bounds for weights
        if constraints.long_only:
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
        else:
            bounds = [(-constraints.max_weight, constraints.max_weight) for _ in range(n)]
        
        # Constraints
        constraint_list = []
        
        # Sum to 1 constraint
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Total notional constraint if specified
        if constraints.max_total_notional < float('inf'):
            def notional_constraint(w):
                total_notional = 0.0
                for i, symbol in enumerate(symbols):
                    fund_def = self.engine.fund_map.get_fund_definition(symbol)
                    notional_factor = fund_def.total_notional if fund_def else 1.0
                    total_notional += abs(w[i]) * notional_factor
                return constraints.max_total_notional - total_notional
            
            constraint_list.append({
                'type': 'ineq',
                'fun': notional_constraint
            })
        
        # Optimize
        try:
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            solve_time = time.time() - start_time
            
            if result.success:
                weights_result = result.x
                
                # Clean up tiny weights
                weights_result[np.abs(weights_result) < constraints.min_position_size] = 0.0
                
                # Renormalize
                if np.sum(weights_result) > 0:
                    weights_result = weights_result / np.sum(weights_result)
                
                # Apply leverage if targeting specific volatility
                if target_volatility is not None and use_leverage:
                    current_vol = np.sqrt(np.dot(weights_result, np.dot(covariance_matrix, weights_result)))
                    if current_vol > 0:
                        leverage_factor = target_volatility / current_vol
                        # Check if leverage is within constraints
                        total_notional_leveraged = self.engine._calculate_total_notional(symbols, 
                            {symbol: w * leverage_factor for symbol, w in zip(symbols, weights_result)})
                        
                        if total_notional_leveraged <= constraints.max_total_notional:
                            weights_result *= leverage_factor
                
                weights_dict = {symbol: float(w) for symbol, w in zip(symbols, weights_result)}
                
                # Calculate portfolio metrics (using zero expected returns for risk parity)
                expected_returns = np.zeros(n)  # Risk parity doesn't use expected returns
                portfolio_return, portfolio_vol, sharpe_ratio = self.engine.calculate_portfolio_metrics(
                    weights_dict, expected_returns, covariance_matrix, symbols
                )
                
                # Calculate exposures
                exposures = {}
                for symbol, weight in weights_dict.items():
                    if abs(weight) > 1e-6:
                        fund_def = self.engine.fund_map.get_fund_definition(symbol)
                        if fund_def:
                            for exp_type, exp_amount in fund_def.exposures.items():
                                if exp_type not in exposures:
                                    exposures[exp_type] = 0.0
                                exposures[exp_type] += weight * exp_amount
                
                # Calculate total notional
                total_notional = self.engine._calculate_total_notional(symbols, weights_dict)
                
                return OptimizationResult(
                    weights=weights_dict,
                    objective_value=float(result.fun),
                    expected_return=float(portfolio_return),
                    expected_volatility=float(portfolio_vol),
                    sharpe_ratio=float(sharpe_ratio),
                    exposures=exposures,
                    total_notional=total_notional,
                    success=True,
                    message="Risk parity optimization completed successfully",
                    solve_time=solve_time
                )
            else:
                return OptimizationResult(
                    weights={symbol: 0.0 for symbol in symbols},
                    objective_value=0.0,
                    expected_return=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    exposures={},
                    total_notional=0.0,
                    success=False,
                    message=f"Risk parity optimization failed: {result.message}",
                    solve_time=solve_time
                )
                
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"Risk parity optimization error: {e}")
            return OptimizationResult(
                weights={symbol: 0.0 for symbol in symbols},
                objective_value=0.0,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                exposures={},
                total_notional=0.0,
                success=False,
                message=f"Risk parity optimization error: {str(e)}",
                solve_time=solve_time
            )


class BlackLittermanOptimizer:
    """Black-Litterman model implementation."""
    
    def __init__(self, engine):
        """Initialize with reference to main optimization engine."""
        self.engine = engine
    
    def black_litterman_returns(
        self,
        symbols: List[str],
        market_weights: np.ndarray,
        covariance_matrix: np.ndarray,
        views: List,  # MarketView objects
        risk_aversion: float = 2.5,
        tau: float = 0.025
    ) -> np.ndarray:
        """Calculate Black-Litterman expected returns.
        
        Args:
            symbols: List of asset symbols
            market_weights: Market capitalization weights
            covariance_matrix: Return covariance matrix
            views: List of MarketView objects
            risk_aversion: Risk aversion parameter
            tau: Scaling parameter for uncertainty
            
        Returns:
            Black-Litterman expected returns vector
        """
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(covariance_matrix, market_weights)
        
        if not views:
            return pi
        
        # Build view matrix P and view vector Q
        n = len(symbols)
        k = len(views)
        
        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.eye(k)  # View uncertainty matrix (simplified)
        
        for i, view in enumerate(views):
            Q[i] = view.expected_return
            
            # Build view vector
            if view.view_type == 'absolute':
                # Absolute view on specific assets
                for asset in view.assets:
                    if asset in symbols:
                        idx = symbols.index(asset)
                        P[i, idx] = 1.0 / len(view.assets)
            elif view.view_type == 'relative':
                # Relative view (first asset outperforms second)
                if len(view.assets) >= 2:
                    asset1, asset2 = view.assets[0], view.assets[1]
                    if asset1 in symbols and asset2 in symbols:
                        idx1, idx2 = symbols.index(asset1), symbols.index(asset2)
                        P[i, idx1] = 1.0
                        P[i, idx2] = -1.0
            
            # Set view uncertainty based on confidence
            Omega[i, i] = tau * np.dot(P[i, :], np.dot(covariance_matrix, P[i, :].T)) / view.confidence
        
        # Black-Litterman formula
        tau_sigma = tau * covariance_matrix
        
        # Precision matrices
        M1 = np.linalg.inv(tau_sigma)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
        M3 = np.dot(np.linalg.inv(tau_sigma), pi) + np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        
        # New expected returns
        mu_bl = np.dot(np.linalg.inv(M1 + M2), M3)
        
        return mu_bl