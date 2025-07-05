"""Core optimization engine and data structures."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from ..portfolio import ExposureType, FundExposureMap, ExposureCalculator, PortfolioAnalytics

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MAX_SHARPE = "maximize_sharpe_ratio"
    MIN_VOLATILITY = "minimize_volatility"
    MAX_RETURN = "maximize_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "maximize_diversification"
    MIN_TRACKING_ERROR = "minimize_tracking_error"
    TARGET_EXPOSURES = "target_exposures"


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0  # Minimum position weight
    max_weight: float = 1.0  # Maximum position weight
    max_total_notional: float = 2.0  # Maximum leverage (200% notional)
    target_volatility: Optional[float] = None  # Target portfolio volatility
    max_exposure_per_type: Optional[Dict[ExposureType, float]] = None
    min_exposure_per_type: Optional[Dict[ExposureType, float]] = None
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None
    long_only: bool = True
    max_positions: Optional[int] = None  # Maximum number of positions
    min_position_size: float = 0.001  # Minimum position size to avoid micro positions
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.max_exposure_per_type is None:
            self.max_exposure_per_type = {}
        if self.min_exposure_per_type is None:
            self.min_exposure_per_type = {}
        if self.sector_limits is None:
            self.sector_limits = {}


@dataclass
class OptimizationResult:
    """Results from portfolio optimization."""
    weights: Dict[str, float]
    objective_value: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    exposures: Dict[ExposureType, float]
    total_notional: float
    success: bool
    message: str
    
    # Additional diagnostics
    effective_assets: int = 0  # Number of non-zero positions
    concentration_ratio: float = 0.0  # Sum of squared weights (Herfindahl index)
    diversification_ratio: float = 0.0  # Portfolio vol / weighted avg individual vols
    max_weight: float = 0.0  # Maximum individual weight
    
    # Optimization details
    solver_status: str = ""
    solve_time: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.weights:
            weights_array = np.array(list(self.weights.values()))
            
            # Count effective assets (non-zero positions)
            self.effective_assets = np.sum(weights_array > 1e-6)
            
            # Concentration ratio (Herfindahl index)
            self.concentration_ratio = np.sum(weights_array ** 2)
            
            # Maximum weight
            self.max_weight = np.max(np.abs(weights_array))
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            'weights': self.weights,
            'objective_value': self.objective_value,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'exposures': {exp.value: val for exp, val in self.exposures.items()},
            'total_notional': self.total_notional,
            'success': self.success,
            'message': self.message,
            'effective_assets': self.effective_assets,
            'concentration_ratio': self.concentration_ratio,
            'diversification_ratio': self.diversification_ratio,
            'max_weight': self.max_weight,
            'solver_status': self.solver_status,
            'solve_time': self.solve_time
        }
    
    def to_trades(self, current_portfolio, prices: Dict[str, float], total_portfolio_value: Optional[float] = None):
        """Convert optimization result to list of trades.
        
        Args:
            current_portfolio: Current Portfolio object
            prices: Current market prices
            total_portfolio_value: Total portfolio value (calculated if not provided)
            
        Returns:
            List of Trade objects to reach target weights
        """
        from .trades import TradeGenerator
        
        if not self.success:
            raise ValueError("Cannot generate trades from failed optimization result")
        
        # Calculate total portfolio value if not provided
        if total_portfolio_value is None:
            total_portfolio_value = 0.0
            for symbol, position in current_portfolio.positions.items():
                if symbol in prices:
                    total_portfolio_value += position.quantity * prices[symbol]
        
        # Generate trades using TradeGenerator
        trade_generator = TradeGenerator()
        trades = trade_generator.generate_trades(
            target_weights=self.weights,
            current_portfolio=current_portfolio,
            prices=prices,
            total_portfolio_value=total_portfolio_value
        )
        
        return trades


class OptimizationEngine:
    """Main optimization engine that coordinates different optimization methods."""
    
    def __init__(self, analytics: PortfolioAnalytics, fund_map: FundExposureMap):
        """Initialize optimization engine.
        
        Args:
            analytics: PortfolioAnalytics instance for return/risk calculations
            fund_map: FundExposureMap for exposure decomposition
        """
        self.analytics = analytics
        self.fund_map = fund_map
        self.calculator = ExposureCalculator(fund_map)
        
        # Initialize method-specific optimizers
        from .methods import MeanVarianceOptimizer, RiskParityOptimizer, BlackLittermanOptimizer
        
        self.mv_optimizer = MeanVarianceOptimizer(self)
        self.rp_optimizer = RiskParityOptimizer(self)
        self.bl_optimizer = BlackLittermanOptimizer(self)
        
        logger.debug("Initialized optimization engine with fund map containing "
                    f"{len(fund_map)} funds")
    
    def optimize(
        self,
        symbols: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        objective: ObjectiveType = ObjectiveType.MAX_SHARPE,
        **kwargs
    ) -> OptimizationResult:
        """Main optimization dispatch method.
        
        Args:
            symbols: List of symbols to optimize
            expected_returns: Expected returns vector
            covariance_matrix: Return covariance matrix
            constraints: Optimization constraints
            objective: Optimization objective
            **kwargs: Additional method-specific parameters
            
        Returns:
            OptimizationResult with optimal weights and diagnostics
        """
        if len(symbols) != len(expected_returns) or len(symbols) != covariance_matrix.shape[0]:
            raise ValueError("Dimension mismatch between symbols, returns, and covariance matrix")
        
        if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")
        
        logger.info(f"Starting optimization with {len(symbols)} assets, objective: {objective.value}")
        
        try:
            if objective == ObjectiveType.MAX_SHARPE:
                return self.mv_optimizer.optimize_max_sharpe(
                    symbols, expected_returns, covariance_matrix, constraints, **kwargs
                )
            elif objective == ObjectiveType.MIN_VOLATILITY:
                return self.mv_optimizer.optimize_min_volatility(
                    symbols, expected_returns, covariance_matrix, constraints, **kwargs
                )
            elif objective == ObjectiveType.MAX_RETURN:
                return self.mv_optimizer.optimize_max_return(
                    symbols, expected_returns, covariance_matrix, constraints, **kwargs
                )
            elif objective == ObjectiveType.RISK_PARITY:
                return self.rp_optimizer.optimize_risk_parity(
                    symbols, covariance_matrix, constraints, **kwargs
                )
            elif objective == ObjectiveType.TARGET_EXPOSURES:
                target_exposures = kwargs.get('target_exposures')
                if target_exposures is None:
                    raise ValueError("target_exposures required for TARGET_EXPOSURES objective")
                return self.optimize_exposures(symbols, target_exposures, constraints, **kwargs)
            else:
                raise ValueError(f"Unsupported objective type: {objective}")
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                weights={symbol: 0.0 for symbol in symbols},
                objective_value=0.0,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                exposures={},
                total_notional=0.0,
                success=False,
                message=f"Optimization failed: {str(e)}"
            )
    
    def optimize_exposures(
        self,
        symbols: List[str],
        target_exposures: Dict[ExposureType, float],
        constraints: OptimizationConstraints,
        minimize_cost: bool = True
    ) -> OptimizationResult:
        """Find fund weights that best match target exposure profile.
        
        Args:
            symbols: Available symbols to optimize over
            target_exposures: Target exposure by type
            constraints: Optimization constraints
            minimize_cost: Whether to minimize total positions/costs
            
        Returns:
            OptimizationResult with weights that best match target exposures
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization. Install with: pip install cvxpy")
        
        n = len(symbols)
        weights = cp.Variable(n)
        
        # Build exposure matrix - each row is an exposure type, each column is a symbol
        exposure_types = list(target_exposures.keys())
        exposure_matrix = np.zeros((len(exposure_types), n))
        
        for j, symbol in enumerate(symbols):
            fund_def = self.fund_map.get_fund_definition(symbol)
            if fund_def:
                for i, exp_type in enumerate(exposure_types):
                    exposure_matrix[i, j] = fund_def.exposures.get(exp_type, 0.0)
            else:
                # Default exposure based on symbol name or assume simple equity
                if 'bond' in symbol.lower() or 'tlt' in symbol.lower():
                    if ExposureType.BONDS in exposure_types:
                        idx = exposure_types.index(ExposureType.BONDS)
                        exposure_matrix[idx, j] = 1.0
                else:
                    # Default to equity exposure
                    if ExposureType.US_LARGE_EQUITY in exposure_types:
                        idx = exposure_types.index(ExposureType.US_LARGE_EQUITY)
                        exposure_matrix[idx, j] = 1.0
        
        # Calculate actual exposures from weights
        actual_exposures = exposure_matrix @ weights
        
        # Target exposure vector
        target_vector = np.array([target_exposures[exp_type] for exp_type in exposure_types])
        
        # Objective: minimize tracking error to target exposures
        tracking_error = cp.sum_squares(actual_exposures - target_vector)
        
        if minimize_cost:
            # Add penalty for number of positions (L1 regularization)
            cost_penalty = 0.01 * cp.norm(weights, 1)
            objective = cp.Minimize(tracking_error + cost_penalty)
        else:
            objective = cp.Minimize(tracking_error)
        
        # Build constraints
        constraints_list = self._build_basic_constraints(weights, symbols, constraints)
        
        # Add exposure constraints if specified
        exposure_constraints = self._build_exposure_constraints(
            weights, symbols, constraints, exposure_matrix, exposure_types
        )
        constraints_list.extend(exposure_constraints)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        
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
                    
                    # Calculate resulting exposures
                    result_exposures = {}
                    for i, exp_type in enumerate(exposure_types):
                        result_exposures[exp_type] = float(np.dot(exposure_matrix[i, :], weights_result))
                    
                    # Calculate total notional exposure
                    total_notional = self._calculate_total_notional(symbols, weights_dict)
                    
                    return OptimizationResult(
                        weights=weights_dict,
                        objective_value=float(problem.value),
                        expected_return=0.0,  # Not calculated for exposure optimization
                        expected_volatility=0.0,  # Not calculated for exposure optimization
                        sharpe_ratio=0.0,
                        exposures=result_exposures,
                        total_notional=total_notional,
                        success=True,
                        message="Exposure optimization completed successfully",
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
    
    def _build_basic_constraints(self, weights, symbols, constraints):
        """Build basic optimization constraints."""
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        constraints_list = []
        
        # Weight sum constraint
        constraints_list.append(cp.sum(weights) == 1.0)
        
        # Individual weight bounds
        if constraints.long_only:
            constraints_list.append(weights >= constraints.min_weight)
        else:
            constraints_list.append(weights >= -constraints.max_weight)
        
        constraints_list.append(weights <= constraints.max_weight)
        
        # Total notional constraint (leverage constraint)
        if constraints.max_total_notional < float('inf'):
            # Calculate notional exposure for each position
            notional_weights = []
            for i, symbol in enumerate(symbols):
                fund_def = self.fund_map.get_fund_definition(symbol)
                if fund_def:
                    notional_factor = fund_def.total_notional
                else:
                    notional_factor = 1.0  # Simple funds have 1x notional
                notional_weights.append(notional_factor * cp.abs(weights[i]))
            
            constraints_list.append(cp.sum(notional_weights) <= constraints.max_total_notional)
        
        return constraints_list
    
    def _build_exposure_constraints(self, weights, symbols, constraints, exposure_matrix, exposure_types):
        """Build exposure-based constraints."""
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        constraints_list = []
        
        # Maximum exposure constraints
        for exp_type, max_exposure in constraints.max_exposure_per_type.items():
            if exp_type in exposure_types:
                idx = exposure_types.index(exp_type)
                exposure_value = exposure_matrix[idx, :] @ weights
                constraints_list.append(exposure_value <= max_exposure)
        
        # Minimum exposure constraints
        for exp_type, min_exposure in constraints.min_exposure_per_type.items():
            if exp_type in exposure_types:
                idx = exposure_types.index(exp_type)
                exposure_value = exposure_matrix[idx, :] @ weights
                constraints_list.append(exposure_value >= min_exposure)
        
        return constraints_list
    
    def _calculate_total_notional(self, symbols: List[str], weights: Dict[str, float]) -> float:
        """Calculate total notional exposure for a set of weights."""
        total_notional = 0.0
        
        for symbol, weight in weights.items():
            if abs(weight) > 1e-6:
                fund_def = self.fund_map.get_fund_definition(symbol)
                if fund_def:
                    notional_factor = fund_def.total_notional
                else:
                    notional_factor = 1.0
                
                total_notional += abs(weight) * notional_factor
        
        return total_notional
    
    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        symbols: List[str],
        risk_free_rate: float = 0.02
    ) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            symbols: Symbol list
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        weights_array = np.array([weights.get(symbol, 0.0) for symbol in symbols])
        
        # Portfolio expected return
        portfolio_return = np.dot(weights_array, expected_returns)
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights_array, np.dot(covariance_matrix, weights_array))
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0))
        
        # Sharpe ratio
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio