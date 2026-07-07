"""
Results Analysis

This module provides functions for analyzing the results of parameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field


@dataclass
class OptimalParameterResult:
    """Dataclass for optimal parameter results."""
    horizon_days: int
    horizon_months: float
    method: str
    parameters: Dict[str, Any]
    history_length: int
    frequency: str
    validation_method: str
    forecasting_error: float
    success_rate: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ForecastabilityAnalysis:
    """Dataclass for forecastability analysis."""
    horizon_days: int
    horizon_months: float
    min_error: float
    mean_error: float
    median_error: float
    std_error: float
    num_combinations: int
    best_success_rate: float

@dataclass
class MethodPerformanceAnalysis:
    """Dataclass for method performance analysis."""
    method: str
    min_error: float
    mean_error: float
    median_error: float
    std_error: float
    combinations_tested: int
    avg_success_rate: float
    best_success_rate: float

@dataclass
class ComprehensiveInsights:
    """Dataclass for comprehensive insights."""
    optimal_by_horizon: Dict[str, OptimalParameterResult]
    forecastability_by_horizon: Dict[int, ForecastabilityAnalysis]
    method_performance: Dict[str, MethodPerformanceAnalysis]
    frequency_impact: Dict[str, Dict[str, float]]
    key_findings: List[str]
    summary_statistics: Dict[str, Any]

class ParameterSearchResultsAnalyzer:
    """
    A class for analyzing the results of parameter optimization.
    """

    def __init__(self, search_results: Dict):
        """
        Initialize the ResultsAnalyzer.

        Args:
            objective_functions: A dictionary of objective functions.
        """
        self.search_results = search_results
        self.results_df = self._create_results_dataframe()
        self.insights = None

    def _create_results_dataframe(self):
        if not self.search_results or 'results' not in self.search_results or not self.search_results['results']:
            return pd.DataFrame()
        
        records = []
        for result in self.search_results['results']:
            if not result.get('success', False):
                continue
            
            combo = result.get('combination', {})
            metrics = result.get('aggregate_metrics', {})
            
            record = {
                'method': combo.get('method'),
                'history_length': combo.get('history_length'),
                'horizon': combo.get('horizon'),
                'frequency': combo.get('frequency'),
                'validation_method': combo.get('validation_method'),
                'mean_mse': metrics.get('mean_mse'),
                'mean_mae': metrics.get('mean_mae'),
                'mean_hit_rate': metrics.get('mean_hit_rate'),
                'success_rate': result.get('success_rate_across_exposures')
            }
            
            for param, value in combo.get('parameters', {}).items():
                record[f'param_{param}'] = value
            
            records.append(record)
            
        return pd.DataFrame(records)

    def analyze_optimal_parameters_by_horizon(self):
        if self.results_df.empty:
            return {}
        
        optimal_params = {}
        for horizon, group in self.results_df.groupby('horizon'):
            best_idx = group['mean_mse'].idxmin()
            if pd.isna(best_idx):
                continue
            best = group.loc[best_idx]
            optimal_params[f"{horizon}_days"] = OptimalParameterResult(
                horizon_days=horizon,
                horizon_months=horizon / 21.0,
                method=best['method'],
                parameters={k.replace('param_', ''): v for k, v in best.items() if k.startswith('param_') and pd.notna(v)},
                history_length=best['history_length'],
                frequency=best['frequency'],
                validation_method=best['validation_method'],
                forecasting_error=best['mean_mse'],
                success_rate=best['success_rate']
            )
        return optimal_params

    def analyze_forecastability_by_horizon(self):
        if self.results_df.empty:
            return {}
            
        forecastability = {}
        for horizon, group in self.results_df.groupby('horizon'):
            forecastability[horizon] = ForecastabilityAnalysis(
                horizon_days=horizon,
                horizon_months=horizon / 21.0,
                min_error=group['mean_mse'].min(),
                mean_error=group['mean_mse'].mean(),
                median_error=group['mean_mse'].median(),
                std_error=group['mean_mse'].std(),
                num_combinations=len(group),
                best_success_rate=group['success_rate'].max()
            )
        return forecastability

    def analyze_method_performance(self):
        if self.results_df.empty:
            return {}
            
        performance = {}
        for method, group in self.results_df.groupby('method'):
            if group['mean_mse'].isnull().all():
                continue
            performance[method] = MethodPerformanceAnalysis(
                method=method,
                min_error=group['mean_mse'].min(),
                mean_error=group['mean_mse'].mean(),
                median_error=group['mean_mse'].median(),
                std_error=group['mean_mse'].std(),
                combinations_tested=len(group),
                avg_success_rate=group['success_rate'].mean(),
                best_success_rate=group['success_rate'].max()
            )
        return performance

    def analyze_frequency_impact(self):
        if self.results_df.empty:
            return {}
            
        impact = {}
        for freq, group in self.results_df.groupby('frequency'):
            if group['mean_mse'].isnull().all():
                continue
            impact[freq] = {
                'min_error': group['mean_mse'].min(),
                'mean_error': group['mean_mse'].mean(),
                'median_error': group['mean_mse'].median(),
                'std_error': group['mean_mse'].std(),
                'combinations_tested': len(group),
                'avg_success_rate': group['success_rate'].mean()
            }
        return impact

    def analyze_parameter_stability(self):
        if self.results_df.empty:
            return {}
            
        stability = {}
        for col in self.results_df.columns:
            if col.startswith('param_'):
                param_name = col.replace('param_', '')
                stability[f"{self.results_df['method'].iloc[0]}_{param_name}"] = {
                    'mean': self.results_df[col].mean(),
                    'std': self.results_df[col].std(),
                    'min': self.results_df[col].min(),
                    'max': self.results_df[col].max(),
                    'most_common': self.results_df[col].mode()[0] if not self.results_df[col].mode().empty else None
                }
        return stability

    def generate_key_findings(self, insights):
        findings = []
        if not insights or not insights.get('method_performance'):
            return ["No key findings could be generated due to lack of data."]
        if insights['method_performance']:
            best_method = min(insights['method_performance'], key=lambda k: insights['method_performance'][k].mean_error)
            findings.append(f"The '{best_method}' method provides the best overall performance.")
        if insights['forecastability_by_horizon']:
            best_horizon = min(insights['forecastability_by_horizon'], key=lambda k: insights['forecastability_by_horizon'][k].min_error)
            findings.append(f"The {best_horizon}-day horizon is the most forecastable.")
        if insights['frequency_impact']:
            best_freq = min(insights['frequency_impact'], key=lambda k: insights['frequency_impact'][k]['mean_error'])
            findings.append(f"The '{best_freq}' frequency provides the best overall performance.")
        return findings

    def generate_comprehensive_insights(self):
        if self.results_df.empty:
            return ComprehensiveInsights({}, {}, {}, {}, [], {})
            
        optimal_by_horizon = self.analyze_optimal_parameters_by_horizon()
        forecastability_by_horizon = self.analyze_forecastability_by_horizon()
        method_performance = self.analyze_method_performance()
        frequency_impact = self.analyze_frequency_impact()
        
        summary_statistics = {
            'total_combinations': len(self.results_df),
            'unique_methods': self.results_df['method'].nunique(),
            'unique_horizons': self.results_df['horizon'].nunique(),
            'unique_frequencies': self.results_df['frequency'].nunique(),
            'average_success_rate': self.results_df['success_rate'].mean(),
            'best_overall_success_rate': self.results_df['success_rate'].max(),
            'best_mean_mse': self.results_df['mean_mse'].min(),
            'average_mean_mse': self.results_df['mean_mse'].mean()
        }

        insights = {
            'optimal_by_horizon': optimal_by_horizon,
            'forecastability_by_horizon': forecastability_by_horizon,
            'method_performance': method_performance,
            'frequency_impact': frequency_impact,
            'summary_statistics': summary_statistics
        }
        
        key_findings = self.generate_key_findings(insights)
        
        self.insights = ComprehensiveInsights(
            optimal_by_horizon=optimal_by_horizon,
            forecastability_by_horizon=forecastability_by_horizon,
            method_performance=method_performance,
            frequency_impact=frequency_impact,
            key_findings=key_findings,
            summary_statistics=summary_statistics
        )
        return self.insights

    def export_detailed_report(self, path):
        if not self.insights:
            self.generate_comprehensive_insights()
            
        with open(path, 'w') as f:
            f.write("PARAMETER SEARCH RESULTS ANALYSIS REPORT\n")
            f.write("="*40 + "\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            for finding in self.insights.key_findings:
                f.write(f"- {finding}\n")
            f.write("\n")
            f.write("KEY FINDINGS\n")
            f.write("-" * 20 + "\n")
            for finding in self.insights.key_findings:
                f.write(f"- {finding}\n")
            f.write("\n")
            f.write("OPTIMAL PARAMETERS BY HORIZON\n")
            f.write("-" * 20 + "\n")
            for horizon, result in self.insights.optimal_by_horizon.items():
                f.write(f"Horizon: {horizon}\n")
                f.write(f"  Method: {result.method}\n")
                f.write(f"  Parameters: {result.parameters}\n")
                f.write(f"  MSE: {result.forecasting_error}\n")
            f.write("\n")
            f.write("METHOD PERFORMANCE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            for method, result in self.insights.method_performance.items():
                f.write(f"Method: {method}\n")
                f.write(f"  Min MSE: {result.min_error}\n")
                f.write(f"  Mean MSE: {result.mean_error}\n")

    def get_recommendations(self):
        if not self.insights:
            self.generate_comprehensive_insights()
        
        recommendations = []
        if self.insights and self.insights.key_findings:
            recommendations.extend(self.insights.key_findings)
        
        if not recommendations:
            recommendations.append("No recommendations could be generated due to lack of data.")
            
        return recommendations

    def create_visualization_dashboard(self):
        pass
			
