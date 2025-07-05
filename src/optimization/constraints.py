"""Advanced constraint building utilities."""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConstraintBuilder:
    """Builder for complex portfolio optimization constraints."""
    
    def __init__(self, fund_map):
        """Initialize with fund exposure map.
        
        Args:
            fund_map: FundExposureMap for exposure calculations
        """
        self.fund_map = fund_map
    
    def build_turnover_constraint(self, weights_var, current_weights: Dict[str, float], 
                                 symbols: List[str], max_turnover: float):
        """Build constraint limiting portfolio turnover.
        
        Args:
            weights_var: CVXPY variable for weights
            current_weights: Current portfolio weights
            symbols: List of symbols
            max_turnover: Maximum allowed turnover (0-1)
            
        Returns:
            CVXPY constraint object
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        current_weights_array = np.array([current_weights.get(symbol, 0.0) for symbol in symbols])
        weight_changes = cp.abs(weights_var - current_weights_array)
        turnover = cp.sum(weight_changes) / 2  # Divide by 2 since buys + sells = 2 * net change
        
        return turnover <= max_turnover
    
    def build_sector_constraints(self, weights_var, symbols: List[str], 
                               sector_map: Dict[str, str],
                               sector_limits: Dict[str, Tuple[float, float]]):
        """Build constraints limiting exposure to sectors.
        
        Args:
            weights_var: CVXPY variable for weights
            symbols: List of symbols
            sector_map: Mapping from symbol to sector
            sector_limits: Dictionary of (min_weight, max_weight) per sector
            
        Returns:
            List of CVXPY constraint objects
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        constraints = []
        
        # Group symbols by sector
        sectors = {}
        for i, symbol in enumerate(symbols):
            sector = sector_map.get(symbol, 'Other')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(i)
        
        # Build constraints for each sector
        for sector, indices in sectors.items():
            if sector in sector_limits:
                min_weight, max_weight = sector_limits[sector]
                sector_weight = cp.sum([weights_var[i] for i in indices])
                
                if min_weight > 0:
                    constraints.append(sector_weight >= min_weight)
                if max_weight < 1.0:
                    constraints.append(sector_weight <= max_weight)
        
        return constraints
    
    def build_tracking_error_constraint(self, weights_var, symbols: List[str],
                                      benchmark_weights: Dict[str, float],
                                      covariance_matrix: np.ndarray,
                                      max_tracking_error: float):
        """Build constraint limiting tracking error vs benchmark.
        
        Args:
            weights_var: CVXPY variable for weights
            symbols: List of symbols
            benchmark_weights: Benchmark weights
            covariance_matrix: Return covariance matrix
            max_tracking_error: Maximum tracking error (annualized)
            
        Returns:
            CVXPY constraint object
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        benchmark_array = np.array([benchmark_weights.get(symbol, 0.0) for symbol in symbols])
        active_weights = weights_var - benchmark_array
        tracking_variance = cp.quad_form(active_weights, covariance_matrix)
        tracking_error = cp.sqrt(tracking_variance)
        
        return tracking_error <= max_tracking_error
    
    def build_factor_exposure_constraints(self, weights_var, symbols: List[str],
                                        factor_loadings: Dict[str, np.ndarray],
                                        factor_limits: Dict[str, Tuple[float, float]]):
        """Build constraints on factor exposures.
        
        Args:
            weights_var: CVXPY variable for weights
            symbols: List of symbols
            factor_loadings: Factor loadings for each factor
            factor_limits: Limits on factor exposures
            
        Returns:
            List of CVXPY constraint objects
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        constraints = []
        
        for factor_name, (min_exposure, max_exposure) in factor_limits.items():
            if factor_name in factor_loadings:
                loadings = factor_loadings[factor_name]
                factor_exposure = loadings @ weights_var
                
                if min_exposure > -float('inf'):
                    constraints.append(factor_exposure >= min_exposure)
                if max_exposure < float('inf'):
                    constraints.append(factor_exposure <= max_exposure)
        
        return constraints
    
    def build_transaction_cost_penalty(self, weights_var, current_weights: Dict[str, float],
                                     symbols: List[str], transaction_costs: Dict[str, float]):
        """Build transaction cost penalty for objective function.
        
        Args:
            weights_var: CVXPY variable for weights
            current_weights: Current portfolio weights
            symbols: List of symbols
            transaction_costs: Transaction cost per symbol (as fraction)
            
        Returns:
            CVXPY expression for transaction costs
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        current_weights_array = np.array([current_weights.get(symbol, 0.0) for symbol in symbols])
        cost_vector = np.array([transaction_costs.get(symbol, 0.001) for symbol in symbols])
        
        weight_changes = cp.abs(weights_var - current_weights_array)
        transaction_cost = cost_vector @ weight_changes
        
        return transaction_cost
    
    def build_concentration_constraint(self, weights_var, max_concentration: float):
        """Build constraint limiting portfolio concentration (Herfindahl index).
        
        Args:
            weights_var: CVXPY variable for weights
            max_concentration: Maximum concentration (sum of squared weights)
            
        Returns:
            CVXPY constraint object
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        concentration = cp.sum_squares(weights_var)
        return concentration <= max_concentration
    
    def build_leverage_constraint(self, weights_var, symbols: List[str], 
                                max_gross_exposure: float, max_net_exposure: float = 1.0):
        """Build constraints on gross and net leverage.
        
        Args:
            weights_var: CVXPY variable for weights
            symbols: List of symbols
            max_gross_exposure: Maximum gross exposure (sum of absolute weights)
            max_net_exposure: Maximum net exposure (sum of weights)
            
        Returns:
            List of CVXPY constraint objects
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        constraints = []
        
        # Gross exposure constraint (includes leveraged funds)
        gross_exposure = 0
        for i, symbol in enumerate(symbols):
            fund_def = self.fund_map.get_fund_definition(symbol)
            notional_factor = fund_def.total_notional if fund_def else 1.0
            gross_exposure += cp.abs(weights_var[i]) * notional_factor
        
        constraints.append(gross_exposure <= max_gross_exposure)
        
        # Net exposure constraint
        net_exposure = cp.sum(weights_var)
        constraints.append(cp.abs(net_exposure) <= max_net_exposure)
        
        return constraints
    
    def build_min_position_constraint(self, weights_var, min_positions: int, 
                                    min_position_size: float = 0.001):
        """Build constraint ensuring minimum number of positions.
        
        Args:
            weights_var: CVXPY variable for weights
            min_positions: Minimum number of non-zero positions
            min_position_size: Minimum size for a position to count
            
        Returns:
            CVXPY constraint object (approximate)
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        # This is an approximation since we can't directly count non-zero elements
        # We use the fact that ||w||_1 / ||w||_inf gives a measure of diversification
        
        # Alternatively, we can use a big-M formulation with binary variables
        # For simplicity, we'll use a penalty approach in the objective instead
        
        # For now, return a dummy constraint
        return cp.sum(cp.abs(weights_var)) >= min_positions * min_position_size
    
    def build_regime_aware_constraints(self, weights_var, symbols: List[str],
                                     regime_probabilities: np.ndarray,
                                     regime_covariances: List[np.ndarray],
                                     max_conditional_var: float):
        """Build constraints that are aware of different market regimes.
        
        Args:
            weights_var: CVXPY variable for weights
            symbols: List of symbols
            regime_probabilities: Probability of each regime
            regime_covariances: Covariance matrix for each regime
            max_conditional_var: Maximum VaR in any regime
            
        Returns:
            List of CVXPY constraint objects
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for optimization")
        
        constraints = []
        
        # For each regime, limit the portfolio variance
        for i, (prob, cov_matrix) in enumerate(zip(regime_probabilities, regime_covariances)):
            if prob > 0.01:  # Only consider regimes with meaningful probability
                regime_variance = cp.quad_form(weights_var, cov_matrix)
                regime_vol = cp.sqrt(regime_variance)
                
                # Approximate VaR constraint (assuming normal distribution)
                # VaR_95% ≈ 1.645 * volatility for daily returns
                var_constraint = 1.645 * regime_vol <= max_conditional_var
                constraints.append(var_constraint)
        
        return constraints