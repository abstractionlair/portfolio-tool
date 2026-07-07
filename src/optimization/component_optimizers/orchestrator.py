"""Orchestrator for component-specific optimization."""

from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from pathlib import Path

from .base import ComponentOptimalParameters, UnifiedOptimalParameters
from .volatility import VolatilityOptimizer
from .correlation import CorrelationOptimizer
from .returns import ExpectedReturnOptimizer

logger = logging.getLogger(__name__)


class ComponentOptimizationOrchestrator:
    """Orchestrate optimization across all components."""
    
    def __init__(self, risk_estimator, parallel: bool = True, max_workers: Optional[int] = None):
        """Initialize the orchestrator.
        
        Args:
            risk_estimator: RiskPremiumEstimator instance
            parallel: Whether to run component optimizations in parallel
            max_workers: Maximum number of parallel workers (None for auto)
        """
        self.risk_estimator = risk_estimator
        self.parallel = parallel
        self.max_workers = max_workers or 3  # One per component type
        
        # Initialize component optimizers
        self.volatility_optimizer = VolatilityOptimizer(risk_estimator, logger)
        self.correlation_optimizer = CorrelationOptimizer(risk_estimator, logger)
        self.return_optimizer = ExpectedReturnOptimizer(risk_estimator, logger)
        
        logger.info(f"Initialized ComponentOptimizationOrchestrator (parallel={parallel})")
    
    def optimize_all_components(self,
                               exposure_ids: List[str],
                               start_date: datetime,
                               end_date: datetime,
                               validation_start: Optional[datetime] = None,
                               validation_end: Optional[datetime] = None,
                               n_splits: int = 5) -> UnifiedOptimalParameters:
        """Run optimization for all components.
        
        Args:
            exposure_ids: List of exposure IDs to optimize
            start_date: Start of optimization period
            end_date: End of optimization period
            validation_start: Start of validation period (defaults to end_date)
            validation_end: End of validation period (defaults to end_date + 3 months)
            n_splits: Number of cross-validation splits
            
        Returns:
            UnifiedOptimalParameters with optimal parameters for all components
        """
        logger.info(f"Starting component optimization for {len(exposure_ids)} exposures")
        logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Set default validation period
        if validation_start is None:
            validation_start = end_date
        if validation_end is None:
            validation_end = end_date.replace(month=min(end_date.month + 3, 12))
        
        optimization_start_time = time.time()
        
        if self.parallel:
            results = self._optimize_parallel(exposure_ids, start_date, end_date, n_splits)
        else:
            results = self._optimize_sequential(exposure_ids, start_date, end_date, n_splits)
        
        optimization_time = time.time() - optimization_start_time
        
        # Extract results by component
        volatility_params = results.get('volatility', {})
        correlation_params = results.get('correlation', {}).get('correlation')
        expected_return_params = results.get('expected_returns', {})
        
        # Create unified parameters
        unified_params = UnifiedOptimalParameters(
            volatility_params=volatility_params,
            correlation_params=correlation_params,
            expected_return_params=expected_return_params,
            optimization_date=datetime.now(),
            validation_period=(validation_start, validation_end),
            metadata={
                'optimization_time_seconds': optimization_time,
                'n_exposures': len(exposure_ids),
                'n_splits': n_splits,
                'parallel': self.parallel,
                'exposure_ids': exposure_ids
            }
        )
        
        logger.info(f"Component optimization completed in {optimization_time:.1f} seconds")
        return unified_params
    
    def _optimize_parallel(self,
                          exposure_ids: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          n_splits: int) -> Dict[str, Dict]:
        """Run component optimizations in parallel."""
        logger.info(f"Running parallel optimization with {self.max_workers} workers")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit optimization tasks
            future_to_component = {
                executor.submit(
                    self.volatility_optimizer.optimize_parameters,
                    exposure_ids, start_date, end_date, n_splits
                ): 'volatility',
                
                executor.submit(
                    self.correlation_optimizer.optimize_parameters,
                    exposure_ids, start_date, end_date, n_splits
                ): 'correlation',
                
                executor.submit(
                    self.return_optimizer.optimize_parameters,
                    exposure_ids, start_date, end_date, n_splits
                ): 'expected_returns'
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_component):
                component = future_to_component[future]
                try:
                    result = future.result()
                    results[component] = result
                    logger.info(f"Completed {component} optimization")
                except Exception as e:
                    logger.error(f"Failed {component} optimization: {e}")
                    results[component] = {}
        
        return results
    
    def _optimize_sequential(self,
                           exposure_ids: List[str],
                           start_date: datetime,
                           end_date: datetime,
                           n_splits: int) -> Dict[str, Dict]:
        """Run component optimizations sequentially."""
        logger.info("Running sequential optimization")
        
        results = {}
        
        # Volatility optimization
        try:
            logger.info("Starting volatility optimization")
            start_time = time.time()
            results['volatility'] = self.volatility_optimizer.optimize_parameters(
                exposure_ids, start_date, end_date, n_splits
            )
            logger.info(f"Volatility optimization completed in {time.time() - start_time:.1f}s")
        except Exception as e:
            logger.error(f"Volatility optimization failed: {e}")
            results['volatility'] = {}
        
        # Correlation optimization
        try:
            logger.info("Starting correlation optimization")
            start_time = time.time()
            results['correlation'] = self.correlation_optimizer.optimize_parameters(
                exposure_ids, start_date, end_date, n_splits
            )
            logger.info(f"Correlation optimization completed in {time.time() - start_time:.1f}s")
        except Exception as e:
            logger.error(f"Correlation optimization failed: {e}")
            results['correlation'] = {}
        
        # Expected returns optimization
        try:
            logger.info("Starting expected returns optimization")
            start_time = time.time()
            results['expected_returns'] = self.return_optimizer.optimize_parameters(
                exposure_ids, start_date, end_date, n_splits
            )
            logger.info(f"Expected returns optimization completed in {time.time() - start_time:.1f}s")
        except Exception as e:
            logger.error(f"Expected returns optimization failed: {e}")
            results['expected_returns'] = {}
        
        return results
    
    def save_optimal_parameters(self,
                               params: UnifiedOptimalParameters,
                               path: Union[str, Path] = "config/optimal_parameters.yaml") -> None:
        """Save parameters for production use.
        
        Args:
            params: Unified optimal parameters
            path: File path to save parameters
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            yaml_content = params.to_yaml()
            with open(path, 'w') as f:
                f.write(yaml_content)
            
            logger.info(f"Saved optimal parameters to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save parameters to {path}: {e}")
            raise
    
    def load_optimal_parameters(self, path: Union[str, Path]) -> UnifiedOptimalParameters:
        """Load previously optimized parameters.
        
        Args:
            path: File path to load parameters from
            
        Returns:
            UnifiedOptimalParameters loaded from file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Parameters file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                yaml_content = f.read()
            
            params = UnifiedOptimalParameters.from_yaml(yaml_content)
            logger.info(f"Loaded optimal parameters from {path}")
            return params
            
        except Exception as e:
            logger.error(f"Failed to load parameters from {path}: {e}")
            raise
    
    def generate_optimization_report(self, params: UnifiedOptimalParameters) -> Dict[str, any]:
        """Generate detailed report on optimization results.
        
        Args:
            params: Unified optimal parameters
            
        Returns:
            Dictionary with optimization report
        """
        report = {
            'summary': {
                'optimization_date': params.optimization_date.isoformat(),
                'validation_period': [
                    params.validation_period[0].isoformat(),
                    params.validation_period[1].isoformat()
                ],
                'n_exposures': len(params.volatility_params),
                'optimization_time': params.metadata.get('optimization_time_seconds', 0),
                'parallel_execution': params.metadata.get('parallel', False)
            },
            'volatility_results': {},
            'correlation_results': {},
            'expected_return_results': {}
        }
        
        # Volatility results
        if params.volatility_params:
            vol_methods = {}
            vol_scores = []
            vol_lookbacks = []
            vol_frequencies = {}
            
            for exp_id, vol_params in params.volatility_params.items():
                vol_methods[vol_params.method] = vol_methods.get(vol_params.method, 0) + 1
                if not np.isnan(vol_params.score):
                    vol_scores.append(vol_params.score)
                vol_lookbacks.append(vol_params.lookback_days)
                vol_frequencies[vol_params.frequency] = vol_frequencies.get(vol_params.frequency, 0) + 1
            
            report['volatility_results'] = {
                'method_distribution': vol_methods,
                'avg_score': np.mean(vol_scores) if vol_scores else None,
                'score_range': [np.min(vol_scores), np.max(vol_scores)] if vol_scores else None,
                'avg_lookback_days': np.mean(vol_lookbacks),
                'frequency_distribution': vol_frequencies,
                'exposures_optimized': list(params.volatility_params.keys())
            }
        
        # Correlation results
        if params.correlation_params:
            corr_params = params.correlation_params
            report['correlation_results'] = {
                'method': corr_params.method,
                'lookback_days': corr_params.lookback_days,
                'frequency': corr_params.frequency,
                'score': corr_params.score,
                'validation_metrics': corr_params.validation_metrics
            }
        
        # Expected return results
        if params.expected_return_params:
            ret_methods = {}
            ret_scores = []
            ret_lookbacks = []
            ret_frequencies = {}
            
            for exp_id, ret_params in params.expected_return_params.items():
                ret_methods[ret_params.method] = ret_methods.get(ret_params.method, 0) + 1
                if not np.isnan(ret_params.score):
                    ret_scores.append(ret_params.score)
                ret_lookbacks.append(ret_params.lookback_days)
                ret_frequencies[ret_params.frequency] = ret_frequencies.get(ret_params.frequency, 0) + 1
            
            report['expected_return_results'] = {
                'method_distribution': ret_methods,
                'avg_score': np.mean(ret_scores) if ret_scores else None,
                'score_range': [np.min(ret_scores), np.max(ret_scores)] if ret_scores else None,
                'avg_lookback_days': np.mean(ret_lookbacks),
                'frequency_distribution': ret_frequencies,
                'exposures_optimized': list(params.expected_return_params.keys())
            }
        
        return report
    
    def validate_parameters(self,
                           params: UnifiedOptimalParameters,
                           exposure_ids: List[str]) -> Dict[str, any]:
        """Validate that parameters are complete and reasonable.
        
        Args:
            params: Parameters to validate
            exposure_ids: Expected exposure IDs
            
        Returns:
            Validation report with issues and recommendations
        """
        issues = []
        warnings = []
        recommendations = []
        
        # Check volatility parameters
        missing_vol = set(exposure_ids) - set(params.volatility_params.keys())
        if missing_vol:
            issues.append(f"Missing volatility parameters for: {sorted(missing_vol)}")
        
        for exp_id, vol_params in params.volatility_params.items():
            if np.isnan(vol_params.score):
                warnings.append(f"Volatility optimization failed for {exp_id}")
            elif vol_params.score > 1.0:  # High MSE
                warnings.append(f"High volatility forecast error for {exp_id}: {vol_params.score:.3f}")
        
        # Check correlation parameters
        if params.correlation_params is None:
            issues.append("Missing correlation parameters")
        elif np.isnan(params.correlation_params.score):
            warnings.append("Correlation optimization failed")
        
        # Check expected return parameters
        missing_ret = set(exposure_ids) - set(params.expected_return_params.keys())
        if missing_ret:
            issues.append(f"Missing expected return parameters for: {sorted(missing_ret)}")
        
        for exp_id, ret_params in params.expected_return_params.items():
            if np.isnan(ret_params.score):
                warnings.append(f"Expected return optimization failed for {exp_id}")
            elif ret_params.score < 0.55:  # Poor directional accuracy
                recommendations.append(f"Low directional accuracy for {exp_id}: {ret_params.score:.3f}")
        
        # General recommendations
        if params.metadata.get('optimization_time_seconds', 0) > 3600:
            recommendations.append("Consider using constrained parameter grids to reduce optimization time")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'summary': {
                'n_issues': len(issues),
                'n_warnings': len(warnings),
                'n_recommendations': len(recommendations)
            }
        }


# Import numpy here to avoid circular imports
try:
    import numpy as np
except ImportError:
    np = None