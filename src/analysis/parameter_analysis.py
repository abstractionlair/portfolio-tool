"""
Analysis utilities for component optimization parameters.

This module provides reusable tools for analyzing optimization results,
comparing parameters across components, and generating insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path

from ..optimization.component_optimizers import UnifiedOptimalParameters

logger = logging.getLogger(__name__)


class ParameterAnalyzer:
    """Analyze component optimization parameters."""
    
    def __init__(self, optimal_params: UnifiedOptimalParameters):
        """Initialize with optimal parameters.
        
        Args:
            optimal_params: UnifiedOptimalParameters from optimization
        """
        self.params = optimal_params
        self._summary_df = None
    
    def create_parameter_summary(self) -> pd.DataFrame:
        """Create comprehensive parameter summary DataFrame.
        
        Returns:
            DataFrame with all parameters organized by component and exposure
        """
        if self._summary_df is not None:
            return self._summary_df
        
        rows = []
        
        # Volatility parameters
        for exp_id, params in self.params.volatility_params.items():
            # Handle failed optimization scores
            score = self._handle_failed_optimization(params)
            optimization_status = 'failed' if score == -999.0 else 'success'
            
            rows.append({
                'exposure_id': exp_id,
                'component': 'volatility',
                'method': params.method,
                'lookback_days': params.lookback_days,
                'frequency': params.frequency,
                'score': score,
                'optimization_status': optimization_status,
                'sample_size': params.validation_metrics.get('sample_size', np.nan),
                'validation_error': params.validation_metrics.get('error', 'none'),
                **self._flatten_parameters(params.parameters)
            })
        
        # Correlation parameters (single set)
        corr_params = self.params.correlation_params
        corr_score = self._handle_failed_optimization(corr_params)
        corr_status = 'failed' if corr_score == -999.0 else 'success'
        
        rows.append({
            'exposure_id': 'ALL',
            'component': 'correlation',
            'method': corr_params.method,
            'lookback_days': corr_params.lookback_days,
            'frequency': corr_params.frequency,
            'score': corr_score,
            'optimization_status': corr_status,
            'validation_error': corr_params.validation_metrics.get('error', 'none'),
            **self._flatten_parameters(corr_params.parameters)
        })
        
        # Expected return parameters
        for exp_id, params in self.params.expected_return_params.items():
            # Handle failed optimization scores
            score = self._handle_failed_optimization(params)
            optimization_status = 'failed' if score == -999.0 else 'success'
            
            rows.append({
                'exposure_id': exp_id,
                'component': 'expected_returns',
                'method': params.method,
                'lookback_days': params.lookback_days,
                'frequency': params.frequency,
                'score': score,
                'optimization_status': optimization_status,
                'validation_error': params.validation_metrics.get('error', 'none'),
                **self._flatten_parameters(params.parameters)
            })
        
        self._summary_df = pd.DataFrame(rows)
        return self._summary_df
    
    def _flatten_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested parameter dictionaries for tabular display."""
        flattened = {}
        
        for key, value in params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened[f"{key}_{sub_key}"] = sub_value
            else:
                flattened[key] = value
        
        return flattened
    
    def _handle_failed_optimization(self, params) -> float:
        """Handle cases where optimization failed and score is NaN."""
        if hasattr(params, 'score'):
            score = params.score
            if pd.isna(score) or score is None:
                # Return a default penalty score for failed optimizations
                return -999.0
            return float(score)
        return -999.0
    
    def get_method_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of methods across components.
        
        Returns:
            Dictionary mapping component to method counts
        """
        summary = self.create_parameter_summary()
        
        method_dist = {}
        for component in summary['component'].unique():
            component_data = summary[summary['component'] == component]
            method_dist[component] = component_data['method'].value_counts().to_dict()
        
        return method_dist
    
    def get_lookback_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get lookback period statistics by component.
        
        Returns:
            Dictionary with lookback statistics per component
        """
        summary = self.create_parameter_summary()
        
        lookback_stats = {}
        for component in summary['component'].unique():
            component_data = summary[summary['component'] == component]
            lookback_days = component_data['lookback_days'].dropna()
            
            if len(lookback_days) > 0:
                lookback_stats[component] = {
                    'mean': float(lookback_days.mean()),
                    'median': float(lookback_days.median()),
                    'std': float(lookback_days.std()),
                    'min': float(lookback_days.min()),
                    'max': float(lookback_days.max()),
                    'count': len(lookback_days)
                }
        
        return lookback_stats
    
    def get_score_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze optimization scores by component.
        
        Returns:
            Dictionary with score statistics per component
        """
        summary = self.create_parameter_summary()
        
        score_analysis = {}
        for component in summary['component'].unique():
            component_data = summary[summary['component'] == component]
            scores = component_data['score'].dropna()
            
            if len(scores) > 0:
                score_analysis[component] = {
                    'mean': float(scores.mean()),
                    'median': float(scores.median()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'count': len(scores)
                }
        
        return score_analysis
    
    def compare_exposure_parameters(self, exposure_ids: List[str]) -> pd.DataFrame:
        """Compare parameters across specific exposures.
        
        Args:
            exposure_ids: List of exposure IDs to compare
            
        Returns:
            DataFrame with parameter comparison
        """
        summary = self.create_parameter_summary()
        
        # Filter for requested exposures
        exposure_data = summary[summary['exposure_id'].isin(exposure_ids)]
        
        # Pivot to compare parameters across exposures
        comparison_df = exposure_data.pivot_table(
            index='component',
            columns='exposure_id',
            values=['method', 'lookback_days', 'frequency', 'score'],
            aggfunc='first'
        )
        
        return comparison_df
    
    def get_parameter_consistency(self) -> Dict[str, Dict[str, Any]]:
        """Analyze parameter consistency across exposures.
        
        Returns:
            Dictionary with consistency metrics per component
        """
        summary = self.create_parameter_summary()
        
        consistency = {}
        for component in summary['component'].unique():
            component_data = summary[summary['component'] == component]
            
            # Skip correlation (only one set of parameters)
            if component == 'correlation':
                continue
            
            consistency[component] = {
                'method_consistency': len(component_data['method'].unique()) == 1,
                'dominant_method': component_data['method'].mode().iloc[0] if len(component_data) > 0 else None,
                'frequency_consistency': len(component_data['frequency'].unique()) == 1,
                'dominant_frequency': component_data['frequency'].mode().iloc[0] if len(component_data) > 0 else None,
                'lookback_cv': float(component_data['lookback_days'].std() / component_data['lookback_days'].mean()) if component_data['lookback_days'].mean() > 0 else np.nan,
                'score_cv': float(component_data['score'].std() / component_data['score'].mean()) if component_data['score'].mean() > 0 else np.nan
            }
        
        return consistency
    
    def identify_outliers(self, component: str, metric: str = 'score') -> List[str]:
        """Identify outlier exposures for a specific component and metric.
        
        Args:
            component: Component name ('volatility', 'correlation', 'expected_returns')
            metric: Metric to analyze ('score', 'lookback_days')
            
        Returns:
            List of exposure IDs that are outliers
        """
        summary = self.create_parameter_summary()
        component_data = summary[summary['component'] == component]
        
        if len(component_data) == 0 or metric not in component_data.columns:
            return []
        
        values = component_data[metric].dropna()
        
        if len(values) < 3:  # Need at least 3 points for outlier detection
            return []
        
        # Use IQR method for outlier detection
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = component_data[
            (component_data[metric] < lower_bound) | 
            (component_data[metric] > upper_bound)
        ]['exposure_id'].tolist()
        
        return outliers
    
    def generate_optimization_insights(self) -> Dict[str, Any]:
        """Generate comprehensive optimization insights.
        
        Returns:
            Dictionary with optimization insights and recommendations
        """
        insights = {
            'parameter_summary': self.create_parameter_summary(),
            'method_distribution': self.get_method_distribution(),
            'lookback_statistics': self.get_lookback_statistics(),
            'score_analysis': self.get_score_analysis(),
            'parameter_consistency': self.get_parameter_consistency(),
            'outliers': {},
            'recommendations': []
        }
        
        # Find outliers for each component
        for component in ['volatility', 'expected_returns']:
            insights['outliers'][component] = {
                'score_outliers': self.identify_outliers(component, 'score'),
                'lookback_outliers': self.identify_outliers(component, 'lookback_days')
            }
        
        # Generate recommendations
        consistency = insights['parameter_consistency']
        for component, metrics in consistency.items():
            if not metrics['method_consistency']:
                insights['recommendations'].append(
                    f"Consider standardizing {component} methods - multiple methods detected"
                )
            
            if not metrics['frequency_consistency']:
                insights['recommendations'].append(
                    f"Consider standardizing {component} frequencies - multiple frequencies detected"
                )
            
            if metrics['lookback_cv'] > 0.3:  # High coefficient of variation
                insights['recommendations'].append(
                    f"High variability in {component} lookback periods (CV={metrics['lookback_cv']:.2f}) - review optimization"
                )
        
        return insights


class ParameterComparator:
    """Compare parameters across different optimization runs."""
    
    def __init__(self, baseline_params: UnifiedOptimalParameters, 
                 comparison_params: UnifiedOptimalParameters):
        """Initialize with baseline and comparison parameters.
        
        Args:
            baseline_params: Baseline parameter set
            comparison_params: Comparison parameter set
        """
        self.baseline = ParameterAnalyzer(baseline_params)
        self.comparison = ParameterAnalyzer(comparison_params)
    
    def compare_methods(self) -> Dict[str, Dict[str, Any]]:
        """Compare method selections between parameter sets.
        
        Returns:
            Dictionary with method comparison results
        """
        baseline_methods = self.baseline.get_method_distribution()
        comparison_methods = self.comparison.get_method_distribution()
        
        comparison_results = {}
        
        for component in set(baseline_methods.keys()) | set(comparison_methods.keys()):
            baseline_dist = baseline_methods.get(component, {})
            comparison_dist = comparison_methods.get(component, {})
            
            comparison_results[component] = {
                'baseline_methods': baseline_dist,
                'comparison_methods': comparison_dist,
                'method_changes': self._calculate_method_changes(baseline_dist, comparison_dist)
            }
        
        return comparison_results
    
    def _calculate_method_changes(self, baseline: Dict[str, int], 
                                 comparison: Dict[str, int]) -> Dict[str, int]:
        """Calculate method changes between two distributions."""
        changes = {}
        
        all_methods = set(baseline.keys()) | set(comparison.keys())
        
        for method in all_methods:
            baseline_count = baseline.get(method, 0)
            comparison_count = comparison.get(method, 0)
            changes[method] = comparison_count - baseline_count
        
        return changes
    
    def compare_scores(self) -> Dict[str, Dict[str, float]]:
        """Compare optimization scores between parameter sets.
        
        Returns:
            Dictionary with score comparison results
        """
        baseline_scores = self.baseline.get_score_analysis()
        comparison_scores = self.comparison.get_score_analysis()
        
        comparison_results = {}
        
        for component in set(baseline_scores.keys()) | set(comparison_scores.keys()):
            baseline_stats = baseline_scores.get(component, {})
            comparison_stats = comparison_scores.get(component, {})
            
            if baseline_stats and comparison_stats:
                comparison_results[component] = {
                    'baseline_mean': baseline_stats['mean'],
                    'comparison_mean': comparison_stats['mean'],
                    'improvement': comparison_stats['mean'] - baseline_stats['mean'],
                    'improvement_pct': ((comparison_stats['mean'] - baseline_stats['mean']) / 
                                       abs(baseline_stats['mean'])) * 100 if baseline_stats['mean'] != 0 else 0
                }
        
        return comparison_results
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report.
        
        Returns:
            Dictionary with complete comparison analysis
        """
        return {
            'method_comparison': self.compare_methods(),
            'score_comparison': self.compare_scores(),
            'baseline_insights': self.baseline.generate_optimization_insights(),
            'comparison_insights': self.comparison.generate_optimization_insights()
        }


def load_parameters_from_yaml(yaml_path: str) -> UnifiedOptimalParameters:
    """Load parameters from YAML file.
    
    Args:
        yaml_path: Path to YAML file with parameters
        
    Returns:
        UnifiedOptimalParameters object
    """
    import yaml
    from ..optimization.component_optimizers import UnifiedOptimalParameters
    
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    
    return UnifiedOptimalParameters.from_yaml(yaml_content)


def analyze_parameter_file(yaml_path: str) -> Dict[str, Any]:
    """Analyze parameters from a YAML file.
    
    Args:
        yaml_path: Path to YAML file with parameters
        
    Returns:
        Dictionary with analysis results
    """
    params = load_parameters_from_yaml(yaml_path)
    analyzer = ParameterAnalyzer(params)
    
    return analyzer.generate_optimization_insights()


def compare_parameter_files(baseline_path: str, comparison_path: str) -> Dict[str, Any]:
    """Compare parameters from two YAML files.
    
    Args:
        baseline_path: Path to baseline parameters
        comparison_path: Path to comparison parameters
        
    Returns:
        Dictionary with comparison results
    """
    baseline_params = load_parameters_from_yaml(baseline_path)
    comparison_params = load_parameters_from_yaml(comparison_path)
    
    comparator = ParameterComparator(baseline_params, comparison_params)
    
    return comparator.generate_comparison_report()