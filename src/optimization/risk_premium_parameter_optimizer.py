"""
Risk Premium Parameter Optimizer

This module implements parameter optimization specifically for risk premium estimation.
Unlike traditional volatility forecasting, this focuses on optimizing parameters for
the risk premium component of returns, which should provide superior inputs for
portfolio optimization.

Key Features:
- Optimizes EWMA lambda parameters for risk premium volatility forecasting
- Supports GARCH parameter optimization for risk premium estimation
- Uses walk-forward validation on out-of-sample risk premium forecasts
- Implements multiple objective functions (MSE, MAE, QLIKE)
- Provides parameter stability analysis across different time periods
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from scipy.optimize import minimize

from .risk_premium_estimator import RiskPremiumEstimator
from .parameter_optimizer import ParameterOptimizer, OptimizationResult


@dataclass
class RiskPremiumOptimizationResult:
    """Results from risk premium parameter optimization."""
    exposure_id: str
    method: str
    optimal_parameters: Dict
    optimization_score: float
    objective_function: str
    validation_results: Dict
    parameter_stability: Dict
    sample_size: int
    optimization_date: datetime
    
    def __post_init__(self):
        """Validate the optimization result."""
        if self.optimization_score < 0:
            warnings.warn(f"Negative optimization score: {self.optimization_score}")
        if self.sample_size < 50:
            warnings.warn(f"Small sample size for optimization: {self.sample_size}")


class RiskPremiumParameterOptimizer:
    """
    Optimizes parameters for risk premium estimation methods.
    
    This class focuses specifically on optimizing parameters for risk premium
    volatility and correlation forecasting, which is theoretically superior
    to optimizing on total returns for portfolio optimization purposes.
    """
    
    def __init__(self, 
                 risk_premium_estimator: RiskPremiumEstimator,
                 validation_periods: int = 12,
                 min_estimation_periods: int = 60):
        """
        Initialize the risk premium parameter optimizer.
        
        Args:
            risk_premium_estimator: RiskPremiumEstimator instance
            validation_periods: Number of periods for walk-forward validation
            min_estimation_periods: Minimum periods needed for parameter estimation
        """
        self.risk_premium_estimator = risk_premium_estimator
        self.validation_periods = validation_periods
        self.min_estimation_periods = min_estimation_periods
        
        # Parameter search spaces
        self.ewma_lambda_range = np.arange(0.85, 0.99, 0.01)
        self.garch_param_ranges = {
            'alpha': np.arange(0.01, 0.15, 0.01),
            'beta': np.arange(0.80, 0.95, 0.01),
            'omega': np.arange(0.00001, 0.001, 0.0001)
        }
        
        # Objective functions
        self.objective_functions = {
            'mse': self._mse_objective,
            'mae': self._mae_objective,
            'qlike': self._qlike_objective
        }
    
    def optimize_exposure_parameters(self,
                                   exposure_id: str,
                                   estimation_date: datetime,
                                   method: str = 'ewma',
                                   objective: str = 'mse',
                                   frequency: str = 'monthly',
                                   lookback_days: int = 1260) -> Optional[RiskPremiumOptimizationResult]:
        """
        Optimize parameters for a specific exposure's risk premium estimation.
        
        Args:
            exposure_id: Exposure identifier
            estimation_date: Date for parameter optimization
            method: Estimation method ('ewma', 'garch')
            objective: Objective function ('mse', 'mae', 'qlike')
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            lookback_days: Historical data lookback period
            
        Returns:
            RiskPremiumOptimizationResult or None if optimization fails
        """
        try:
            print(f"\nOptimizing {method} parameters for {exposure_id} risk premium estimation...")
            
            # Get decomposed risk premium data
            decomposed_data = self.risk_premium_estimator.load_and_decompose_exposure_returns(
                exposure_id=exposure_id,
                estimation_date=estimation_date,
                lookback_days=lookback_days,
                frequency=frequency
            )
            
            if not decomposed_data or decomposed_data.risk_premium.empty:
                print(f"❌ No risk premium data available for {exposure_id}")
                return None
            
            risk_premium_returns = decomposed_data.risk_premium.dropna()
            
            if len(risk_premium_returns) < self.min_estimation_periods:
                print(f"❌ Insufficient data for {exposure_id}: {len(risk_premium_returns)} periods")
                return None
            
            # Perform walk-forward optimization
            if method == 'ewma':
                optimal_params, score, validation_results = self._optimize_ewma_parameters(
                    risk_premium_returns, objective
                )
            elif method == 'garch':
                optimal_params, score, validation_results = self._optimize_garch_parameters(
                    risk_premium_returns, objective
                )
            else:
                print(f"❌ Unsupported method: {method}")
                return None
            
            if optimal_params is None:
                print(f"❌ Optimization failed for {exposure_id}")
                return None
            
            # Analyze parameter stability
            stability_analysis = self._analyze_parameter_stability(
                risk_premium_returns, method, optimal_params, objective
            )
            
            result = RiskPremiumOptimizationResult(
                exposure_id=exposure_id,
                method=method,
                optimal_parameters=optimal_params,
                optimization_score=score,
                objective_function=objective,
                validation_results=validation_results,
                parameter_stability=stability_analysis,
                sample_size=len(risk_premium_returns),
                optimization_date=estimation_date
            )
            
            print(f"✅ Optimized {method} parameters for {exposure_id}")
            print(f"   Optimal parameters: {optimal_params}")
            print(f"   Optimization score: {score:.6f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error optimizing parameters for {exposure_id}: {e}")
            return None
    
    def optimize_universe_parameters(self,
                                   exposures: List[str],
                                   estimation_date: datetime,
                                   method: str = 'ewma',
                                   objective: str = 'mse',
                                   frequency: str = 'monthly',
                                   lookback_days: int = 1260) -> Dict[str, RiskPremiumOptimizationResult]:
        """
        Optimize parameters for multiple exposures in the universe.
        
        Args:
            exposures: List of exposure identifiers
            estimation_date: Date for parameter optimization
            method: Estimation method ('ewma', 'garch')
            objective: Objective function ('mse', 'mae', 'qlike')
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            lookback_days: Historical data lookback period
            
        Returns:
            Dictionary mapping exposure IDs to optimization results
        """
        print(f"\nOptimizing {method} parameters for {len(exposures)} exposures...")
        print(f"Method: {method}, Objective: {objective}, Frequency: {frequency}")
        
        results = {}
        successful_optimizations = 0
        
        for exposure_id in exposures:
            result = self.optimize_exposure_parameters(
                exposure_id=exposure_id,
                estimation_date=estimation_date,
                method=method,
                objective=objective,
                frequency=frequency,
                lookback_days=lookback_days
            )
            
            if result:
                results[exposure_id] = result
                successful_optimizations += 1
        
        print(f"\n✅ Parameter optimization complete: {successful_optimizations}/{len(exposures)} successful")
        
        return results
    
    def _optimize_ewma_parameters(self, 
                                 risk_premium_returns: pd.Series,
                                 objective: str) -> Tuple[Optional[Dict], float, Dict]:
        """Optimize EWMA lambda parameter for risk premium volatility forecasting."""
        best_lambda = None
        best_score = np.inf
        validation_results = {}
        
        # Walk-forward validation
        total_periods = len(risk_premium_returns)
        validation_start = total_periods - self.validation_periods
        
        if validation_start < self.min_estimation_periods:
            print(f"❌ Insufficient data for validation: need {self.min_estimation_periods + self.validation_periods}, have {total_periods}")
            return None, np.inf, {}
        
        for lambda_val in self.ewma_lambda_range:
            scores = []
            
            # Walk-forward validation
            for i in range(validation_start, total_periods):
                # Use data up to period i for estimation
                estimation_data = risk_premium_returns.iloc[:i]
                
                # Forecast volatility for period i
                forecast_vol = self._ewma_volatility_forecast(estimation_data, lambda_val)
                
                # Actual realized volatility (using return at period i)
                actual_vol = abs(risk_premium_returns.iloc[i])
                
                # Calculate objective score
                score = self.objective_functions[objective](forecast_vol, actual_vol)
                scores.append(score)
            
            avg_score = np.mean(scores)
            validation_results[lambda_val] = {
                'avg_score': avg_score,
                'scores': scores
            }
            
            if avg_score < best_score:
                best_score = avg_score
                best_lambda = lambda_val
        
        if best_lambda is None:
            return None, np.inf, validation_results
        
        return {'lambda': best_lambda}, best_score, validation_results
    
    def _optimize_garch_parameters(self, 
                                  risk_premium_returns: pd.Series,
                                  objective: str) -> Tuple[Optional[Dict], float, Dict]:
        """Optimize GARCH parameters for risk premium volatility forecasting."""
        try:
            # Grid search over GARCH parameters
            best_params = None
            best_score = np.inf
            validation_results = {}
            
            # Simplified grid search (computationally intensive)
            alpha_range = np.arange(0.05, 0.15, 0.05)
            beta_range = np.arange(0.80, 0.95, 0.05)
            omega_range = np.arange(0.0001, 0.001, 0.0005)
            
            for alpha in alpha_range:
                for beta in beta_range:
                    for omega in omega_range:
                        # Constraint: alpha + beta < 1 for stationarity
                        if alpha + beta >= 1:
                            continue
                        
                        params = {'alpha': alpha, 'beta': beta, 'omega': omega}
                        score = self._evaluate_garch_parameters(risk_premium_returns, params, objective)
                        
                        validation_results[f"alpha_{alpha}_beta_{beta}_omega_{omega}"] = score
                        
                        if score < best_score:
                            best_score = score
                            best_params = params
            
            return best_params, best_score, validation_results
            
        except Exception as e:
            print(f"❌ GARCH optimization failed: {e}")
            return None, np.inf, {}
    
    def _evaluate_garch_parameters(self, 
                                  risk_premium_returns: pd.Series,
                                  params: Dict,
                                  objective: str) -> float:
        """Evaluate GARCH parameters using walk-forward validation."""
        try:
            # Simplified GARCH(1,1) evaluation
            scores = []
            total_periods = len(risk_premium_returns)
            validation_start = total_periods - self.validation_periods
            
            for i in range(validation_start, total_periods):
                estimation_data = risk_premium_returns.iloc[:i]
                
                # Fit GARCH model (simplified)
                forecast_vol = self._garch_volatility_forecast(estimation_data, params)
                actual_vol = abs(risk_premium_returns.iloc[i])
                
                score = self.objective_functions[objective](forecast_vol, actual_vol)
                scores.append(score)
            
            return np.mean(scores)
            
        except Exception as e:
            print(f"❌ GARCH evaluation failed: {e}")
            return np.inf
    
    def _ewma_volatility_forecast(self, returns: pd.Series, lambda_val: float) -> float:
        """Generate EWMA volatility forecast for risk premium."""
        if len(returns) < 2:
            return np.std(returns) if len(returns) > 0 else 0.01
        
        # Calculate EWMA variance
        weights = [(1 - lambda_val) * (lambda_val ** i) for i in range(len(returns))]
        weights = np.array(weights[::-1])  # Reverse for most recent first
        weights = weights / np.sum(weights)  # Normalize
        
        squared_returns = returns.values ** 2
        ewma_variance = np.sum(weights * squared_returns)
        
        return np.sqrt(ewma_variance)
    
    def _garch_volatility_forecast(self, returns: pd.Series, params: Dict) -> float:
        """Generate GARCH volatility forecast for risk premium."""
        try:
            # Simplified GARCH(1,1) forecast
            alpha, beta, omega = params['alpha'], params['beta'], params['omega']
            
            # Initialize with sample variance
            variance = np.var(returns)
            
            # Iterate GARCH equation
            for ret in returns.values:
                variance = omega + alpha * (ret ** 2) + beta * variance
            
            return np.sqrt(variance)
            
        except Exception as e:
            print(f"❌ GARCH forecast failed: {e}")
            return np.std(returns)
    
    def _analyze_parameter_stability(self, 
                                   risk_premium_returns: pd.Series,
                                   method: str,
                                   optimal_params: Dict,
                                   objective: str) -> Dict:
        """Analyze parameter stability across different time periods."""
        try:
            # Split data into different periods
            total_periods = len(risk_premium_returns)
            period_size = total_periods // 4  # Quarterly analysis
            
            stability_results = {}
            
            for i in range(4):
                start_idx = i * period_size
                end_idx = start_idx + period_size if i < 3 else total_periods
                
                period_data = risk_premium_returns.iloc[start_idx:end_idx]
                
                if len(period_data) < 20:  # Minimum data for analysis
                    continue
                
                # Test optimal parameters on this period
                if method == 'ewma':
                    forecast_vol = self._ewma_volatility_forecast(period_data, optimal_params['lambda'])
                    actual_vol = np.std(period_data)
                else:  # garch
                    forecast_vol = self._garch_volatility_forecast(period_data, optimal_params)
                    actual_vol = np.std(period_data)
                
                score = self.objective_functions[objective](forecast_vol, actual_vol)
                
                stability_results[f"period_{i+1}"] = {
                    'score': score,
                    'forecast_vol': forecast_vol,
                    'actual_vol': actual_vol,
                    'periods': len(period_data)
                }
            
            # Calculate stability metrics
            scores = [result['score'] for result in stability_results.values()]
            stability_metrics = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else np.inf
            }
            
            return {
                'period_results': stability_results,
                'stability_metrics': stability_metrics
            }
            
        except Exception as e:
            print(f"❌ Stability analysis failed: {e}")
            return {}
    
    def _mse_objective(self, forecast: float, actual: float) -> float:
        """Mean Squared Error objective function."""
        return (forecast - actual) ** 2
    
    def _mae_objective(self, forecast: float, actual: float) -> float:
        """Mean Absolute Error objective function."""
        return abs(forecast - actual)
    
    def _qlike_objective(self, forecast: float, actual: float) -> float:
        """Quasi-likelihood objective function."""
        if forecast <= 0:
            return np.inf
        return (actual ** 2) / (forecast ** 2) + np.log(forecast ** 2)
    
    def generate_optimization_report(self, 
                                   results: Dict[str, RiskPremiumOptimizationResult],
                                   save_path: Optional[str] = None) -> pd.DataFrame:
        """Generate comprehensive optimization report."""
        try:
            report_data = []
            
            for exposure_id, result in results.items():
                # Basic optimization results
                row = {
                    'exposure_id': exposure_id,
                    'method': result.method,
                    'optimization_score': result.optimization_score,
                    'objective_function': result.objective_function,
                    'sample_size': result.sample_size,
                    'optimization_date': result.optimization_date
                }
                
                # Add optimal parameters
                for param, value in result.optimal_parameters.items():
                    row[f'optimal_{param}'] = value
                
                # Add stability metrics
                if result.parameter_stability and 'stability_metrics' in result.parameter_stability:
                    stability = result.parameter_stability['stability_metrics']
                    row['stability_mean_score'] = stability.get('mean_score', np.nan)
                    row['stability_std_score'] = stability.get('std_score', np.nan)
                    row['stability_cv'] = stability.get('coefficient_of_variation', np.nan)
                
                report_data.append(row)
            
            report_df = pd.DataFrame(report_data)
            
            if save_path:
                report_df.to_csv(save_path, index=False)
                print(f"✅ Optimization report saved to {save_path}")
            
            return report_df
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            return pd.DataFrame()