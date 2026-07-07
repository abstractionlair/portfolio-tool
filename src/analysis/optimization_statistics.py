"""
Statistical analysis tools for optimization results.

This module provides advanced statistical analysis capabilities for
optimization results, including significance testing, confidence intervals,
and robustness analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, normaltest
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from dataclasses import dataclass

from ..analysis.parameter_analysis import ParameterAnalyzer


@dataclass
class StatisticalTest:
    """Results of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    interpretation: str
    effect_size: Optional[float] = None


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    mean: float
    std_error: float


class OptimizationStatistics:
    """Advanced statistical analysis for optimization results."""
    
    def __init__(self, analyzer: ParameterAnalyzer, confidence_level: float = 0.95):
        """Initialize with parameter analyzer.
        
        Args:
            analyzer: ParameterAnalyzer instance
            confidence_level: Confidence level for statistical tests (default: 0.95)
        """
        self.analyzer = analyzer
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.summary_df = analyzer.create_parameter_summary()
    
    def test_score_differences(self, component1: str, component2: str) -> StatisticalTest:
        """Test for significant differences in scores between components.
        
        Args:
            component1: First component name
            component2: Second component name
            
        Returns:
            StatisticalTest result
        """
        # Extract scores for each component
        comp1_scores = self.summary_df[
            self.summary_df['component'] == component1
        ]['score'].dropna()
        
        comp2_scores = self.summary_df[
            self.summary_df['component'] == component2
        ]['score'].dropna()
        
        if len(comp1_scores) == 0 or len(comp2_scores) == 0:
            return StatisticalTest(
                test_name="Score Difference Test",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                confidence_level=self.confidence_level,
                interpretation="Insufficient data for testing"
            )
        
        # Test normality
        _, p_norm1 = normaltest(comp1_scores)
        _, p_norm2 = normaltest(comp2_scores)
        
        # Choose appropriate test
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            # Both normal - use t-test
            statistic, p_value = ttest_ind(comp1_scores, comp2_scores)
            test_name = "Independent t-test"
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(comp1_scores) - 1) * comp1_scores.var() + 
                                 (len(comp2_scores) - 1) * comp2_scores.var()) / 
                                (len(comp1_scores) + len(comp2_scores) - 2))
            effect_size = (comp1_scores.mean() - comp2_scores.mean()) / pooled_std
        else:
            # Non-normal - use Mann-Whitney U test
            statistic, p_value = mannwhitneyu(comp1_scores, comp2_scores, 
                                            alternative='two-sided')
            test_name = "Mann-Whitney U test"
            
            # Calculate effect size (rank biserial correlation)
            n1, n2 = len(comp1_scores), len(comp2_scores)
            effect_size = (2 * statistic) / (n1 * n2) - 1
        
        significant = p_value < self.alpha
        
        # Interpretation
        if significant:
            direction = "higher" if comp1_scores.mean() > comp2_scores.mean() else "lower"
            interpretation = f"{component1} scores are significantly {direction} than {component2}"
        else:
            interpretation = f"No significant difference between {component1} and {component2} scores"
        
        return StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            effect_size=effect_size
        )
    
    def test_method_independence(self, component: str) -> StatisticalTest:
        """Test independence of method selection across exposures.
        
        Args:
            component: Component name to test
            
        Returns:
            StatisticalTest result
        """
        component_data = self.summary_df[self.summary_df['component'] == component]
        
        if len(component_data) < 2:
            return StatisticalTest(
                test_name="Method Independence Test",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                confidence_level=self.confidence_level,
                interpretation="Insufficient data for testing"
            )
        
        # Create contingency table
        contingency_table = pd.crosstab(component_data['exposure_id'], 
                                       component_data['method'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        significant = p_value < self.alpha
        
        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        effect_size = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        # Interpretation
        if significant:
            interpretation = f"Method selection for {component} is not independent of exposure"
        else:
            interpretation = f"Method selection for {component} appears independent of exposure"
        
        return StatisticalTest(
            test_name="Chi-square test of independence",
            statistic=chi2,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            effect_size=effect_size
        )
    
    def calculate_score_confidence_intervals(self, component: str) -> Dict[str, ConfidenceInterval]:
        """Calculate confidence intervals for scores by component.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary mapping exposure IDs to confidence intervals
        """
        component_data = self.summary_df[self.summary_df['component'] == component]
        
        confidence_intervals = {}
        
        for exposure_id in component_data['exposure_id'].unique():
            exposure_scores = component_data[
                component_data['exposure_id'] == exposure_id
            ]['score'].dropna()
            
            if len(exposure_scores) == 0:
                continue
            
            # Calculate confidence interval
            mean_score = exposure_scores.mean()
            std_error = stats.sem(exposure_scores)
            
            # Use t-distribution for small samples
            if len(exposure_scores) < 30:
                t_critical = stats.t.ppf((1 + self.confidence_level) / 2, 
                                       len(exposure_scores) - 1)
                margin_error = t_critical * std_error
            else:
                z_critical = stats.norm.ppf((1 + self.confidence_level) / 2)
                margin_error = z_critical * std_error
            
            confidence_intervals[exposure_id] = ConfidenceInterval(
                lower_bound=mean_score - margin_error,
                upper_bound=mean_score + margin_error,
                confidence_level=self.confidence_level,
                mean=mean_score,
                std_error=std_error
            )
        
        return confidence_intervals
    
    def analyze_parameter_stability(self, component: str, 
                                  parameter: str = 'lookback_days') -> Dict[str, Any]:
        """Analyze stability of parameters across exposures.
        
        Args:
            component: Component name
            parameter: Parameter to analyze
            
        Returns:
            Dictionary with stability analysis results
        """
        component_data = self.summary_df[self.summary_df['component'] == component]
        
        if parameter not in component_data.columns:
            return {'error': f'Parameter {parameter} not found in component {component}'}
        
        param_values = component_data[parameter].dropna()
        
        if len(param_values) < 2:
            return {'error': 'Insufficient data for stability analysis'}
        
        # Basic statistics
        mean_val = param_values.mean()
        std_val = param_values.std()
        cv = std_val / mean_val if mean_val != 0 else np.inf
        
        # Stability classification
        if cv < 0.1:
            stability_class = "Very Stable"
        elif cv < 0.2:
            stability_class = "Stable"
        elif cv < 0.3:
            stability_class = "Moderate"
        elif cv < 0.5:
            stability_class = "Unstable"
        else:
            stability_class = "Very Unstable"
        
        # Outlier detection using IQR
        q1 = param_values.quantile(0.25)
        q3 = param_values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = param_values[(param_values < lower_bound) | 
                               (param_values > upper_bound)]
        
        # Test for normality
        _, p_normal = normaltest(param_values)
        is_normal = p_normal > 0.05
        
        return {
            'mean': mean_val,
            'std': std_val,
            'coefficient_of_variation': cv,
            'stability_class': stability_class,
            'outlier_count': len(outliers),
            'outlier_values': outliers.tolist(),
            'is_normal': is_normal,
            'normality_p_value': p_normal,
            'quartiles': {
                'q1': q1,
                'q2': param_values.median(),
                'q3': q3
            }
        }
    
    def perform_robustness_analysis(self, component: str) -> Dict[str, Any]:
        """Perform robustness analysis for component optimization.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with robustness analysis results
        """
        component_data = self.summary_df[self.summary_df['component'] == component]
        
        if len(component_data) < 3:
            return {'error': 'Insufficient data for robustness analysis'}
        
        # Score robustness
        scores = component_data['score'].dropna()
        score_robustness = {
            'score_range': float(scores.max() - scores.min()),
            'score_iqr': float(scores.quantile(0.75) - scores.quantile(0.25)),
            'score_mad': float(stats.median_absolute_deviation(scores)),
            'score_skewness': float(stats.skew(scores)),
            'score_kurtosis': float(stats.kurtosis(scores))
        }
        
        # Method diversity
        method_counts = component_data['method'].value_counts()
        method_diversity = {
            'unique_methods': len(method_counts),
            'method_entropy': float(stats.entropy(method_counts.values)),
            'dominant_method': method_counts.index[0],
            'dominant_method_pct': float(method_counts.iloc[0] / len(component_data))
        }
        
        # Parameter robustness
        param_robustness = {}
        for param in ['lookback_days', 'frequency']:
            if param in component_data.columns:
                param_values = component_data[param].dropna()
                if len(param_values) > 0:
                    if param == 'lookback_days':
                        param_robustness[param] = {
                            'range': float(param_values.max() - param_values.min()),
                            'cv': float(param_values.std() / param_values.mean()) if param_values.mean() != 0 else np.inf,
                            'unique_values': len(param_values.unique())
                        }
                    else:
                        param_robustness[param] = {
                            'unique_values': len(param_values.unique()),
                            'most_common': param_values.mode().iloc[0] if len(param_values) > 0 else None
                        }
        
        # Overall robustness score (0-100)
        robustness_score = 100
        
        # Penalize high score variation
        if score_robustness['score_iqr'] > 0.1:
            robustness_score -= 20
        
        # Penalize lack of method diversity
        if method_diversity['unique_methods'] < 2:
            robustness_score -= 30
        
        # Penalize high parameter variation
        if 'lookback_days' in param_robustness:
            if param_robustness['lookback_days']['cv'] > 0.3:
                robustness_score -= 25
        
        # Penalize extreme skewness
        if abs(score_robustness['score_skewness']) > 2:
            robustness_score -= 15
        
        robustness_score = max(0, robustness_score)
        
        return {
            'robustness_score': robustness_score,
            'score_robustness': score_robustness,
            'method_diversity': method_diversity,
            'parameter_robustness': param_robustness,
            'interpretation': self._interpret_robustness(robustness_score)
        }
    
    def _interpret_robustness(self, score: float) -> str:
        """Interpret robustness score."""
        if score >= 80:
            return "Highly robust optimization with consistent results"
        elif score >= 60:
            return "Moderately robust optimization with some variation"
        elif score >= 40:
            return "Somewhat robust optimization with notable variation"
        elif score >= 20:
            return "Low robustness with significant variation in results"
        else:
            return "Very low robustness with highly variable results"
    
    def compare_component_performance(self) -> Dict[str, Any]:
        """Compare performance across all components statistically.
        
        Returns:
            Dictionary with component comparison results
        """
        components = self.summary_df['component'].unique()
        
        if len(components) < 2:
            return {'error': 'Need at least 2 components for comparison'}
        
        # Pairwise comparisons
        pairwise_tests = {}
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1, comp2 = components[i], components[j]
                test_key = f"{comp1}_vs_{comp2}"
                pairwise_tests[test_key] = self.test_score_differences(comp1, comp2)
        
        # Overall ANOVA test
        score_groups = []
        component_labels = []
        
        for component in components:
            comp_scores = self.summary_df[
                self.summary_df['component'] == component
            ]['score'].dropna()
            
            if len(comp_scores) > 0:
                score_groups.append(comp_scores.values)
                component_labels.extend([component] * len(comp_scores))
        
        if len(score_groups) > 1:
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*score_groups)
            
            # Effect size (eta-squared)
            all_scores = np.concatenate(score_groups)
            ss_total = np.sum((all_scores - np.mean(all_scores)) ** 2)
            ss_between = sum(len(group) * (np.mean(group) - np.mean(all_scores)) ** 2 
                           for group in score_groups)
            eta_squared = ss_between / ss_total
            
            anova_result = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'eta_squared': eta_squared,
                'interpretation': "Significant differences between components" if p_value < self.alpha 
                               else "No significant differences between components"
            }
        else:
            anova_result = {'error': 'Insufficient data for ANOVA'}
        
        return {
            'pairwise_tests': pairwise_tests,
            'anova_result': anova_result,
            'component_summaries': {
                comp: {
                    'mean_score': float(self.summary_df[self.summary_df['component'] == comp]['score'].mean()),
                    'std_score': float(self.summary_df[self.summary_df['component'] == comp]['score'].std()),
                    'count': len(self.summary_df[self.summary_df['component'] == comp])
                }
                for comp in components
            }
        }
    
    def generate_statistical_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis report.
        
        Returns:
            Dictionary with complete statistical analysis
        """
        report = {
            'summary_statistics': {},
            'hypothesis_tests': {},
            'confidence_intervals': {},
            'robustness_analysis': {},
            'component_comparison': self.compare_component_performance(),
            'recommendations': []
        }
        
        # Summary statistics for each component
        for component in self.summary_df['component'].unique():
            component_data = self.summary_df[self.summary_df['component'] == component]
            
            if len(component_data) > 0:
                report['summary_statistics'][component] = {
                    'count': len(component_data),
                    'mean_score': float(component_data['score'].mean()),
                    'std_score': float(component_data['score'].std()),
                    'median_score': float(component_data['score'].median()),
                    'score_range': float(component_data['score'].max() - component_data['score'].min()),
                    'parameter_stability': self.analyze_parameter_stability(component)
                }
                
                # Confidence intervals
                report['confidence_intervals'][component] = self.calculate_score_confidence_intervals(component)
                
                # Robustness analysis
                report['robustness_analysis'][component] = self.perform_robustness_analysis(component)
                
                # Method independence test
                report['hypothesis_tests'][f'{component}_method_independence'] = self.test_method_independence(component)
        
        # Generate recommendations
        for component, robustness in report['robustness_analysis'].items():
            if 'robustness_score' in robustness:
                if robustness['robustness_score'] < 60:
                    report['recommendations'].append(
                        f"Consider re-optimizing {component} - low robustness score ({robustness['robustness_score']:.0f})"
                    )
                
                if robustness['method_diversity']['unique_methods'] == 1:
                    report['recommendations'].append(
                        f"Consider testing alternative methods for {component} - only using {robustness['method_diversity']['dominant_method']}"
                    )
        
        return report


class ComparisonStatistics:
    """Statistical analysis for comparing optimization runs."""
    
    def __init__(self, baseline_analyzer: ParameterAnalyzer, 
                 comparison_analyzer: ParameterAnalyzer, 
                 confidence_level: float = 0.95):
        """Initialize with baseline and comparison analyzers.
        
        Args:
            baseline_analyzer: Baseline parameter analyzer
            comparison_analyzer: Comparison parameter analyzer
            confidence_level: Confidence level for statistical tests
        """
        self.baseline_stats = OptimizationStatistics(baseline_analyzer, confidence_level)
        self.comparison_stats = OptimizationStatistics(comparison_analyzer, confidence_level)
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def test_improvement_significance(self, component: str) -> StatisticalTest:
        """Test if improvement in component scores is statistically significant.
        
        Args:
            component: Component name
            
        Returns:
            StatisticalTest result
        """
        baseline_scores = self.baseline_stats.summary_df[
            self.baseline_stats.summary_df['component'] == component
        ]['score'].dropna()
        
        comparison_scores = self.comparison_stats.summary_df[
            self.comparison_stats.summary_df['component'] == component
        ]['score'].dropna()
        
        if len(baseline_scores) == 0 or len(comparison_scores) == 0:
            return StatisticalTest(
                test_name="Improvement Test",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                confidence_level=self.confidence_level,
                interpretation="Insufficient data for testing"
            )
        
        # Test for improvement (one-tailed test)
        statistic, p_value = ttest_ind(comparison_scores, baseline_scores)
        
        # Convert to one-tailed test
        if comparison_scores.mean() > baseline_scores.mean():
            p_value = p_value / 2  # One-tailed test for improvement
        else:
            p_value = 1 - p_value / 2  # One-tailed test for deterioration
        
        significant = p_value < self.alpha
        
        # Calculate effect size
        pooled_std = np.sqrt(((len(comparison_scores) - 1) * comparison_scores.var() + 
                             (len(baseline_scores) - 1) * baseline_scores.var()) / 
                            (len(comparison_scores) + len(baseline_scores) - 2))
        effect_size = (comparison_scores.mean() - baseline_scores.mean()) / pooled_std
        
        # Interpretation
        if significant:
            if comparison_scores.mean() > baseline_scores.mean():
                interpretation = f"Significant improvement in {component} scores"
            else:
                interpretation = f"Significant deterioration in {component} scores"
        else:
            interpretation = f"No significant change in {component} scores"
        
        return StatisticalTest(
            test_name="Improvement t-test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            effect_size=effect_size
        )
    
    def analyze_improvement_distribution(self, component: str) -> Dict[str, Any]:
        """Analyze the distribution of improvements across exposures.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with improvement distribution analysis
        """
        baseline_data = self.baseline_stats.summary_df[
            self.baseline_stats.summary_df['component'] == component
        ].set_index('exposure_id')
        
        comparison_data = self.comparison_stats.summary_df[
            self.comparison_stats.summary_df['component'] == component
        ].set_index('exposure_id')
        
        # Calculate improvements for common exposures
        common_exposures = set(baseline_data.index) & set(comparison_data.index)
        
        if len(common_exposures) == 0:
            return {'error': 'No common exposures found'}
        
        improvements = []
        for exposure in common_exposures:
            baseline_score = baseline_data.loc[exposure, 'score']
            comparison_score = comparison_data.loc[exposure, 'score']
            improvement = comparison_score - baseline_score
            improvements.append({
                'exposure_id': exposure,
                'baseline_score': baseline_score,
                'comparison_score': comparison_score,
                'improvement': improvement,
                'improvement_pct': (improvement / abs(baseline_score)) * 100 if baseline_score != 0 else 0
            })
        
        improvements_df = pd.DataFrame(improvements)
        
        # Calculate statistics
        improvement_values = improvements_df['improvement']
        improvement_pcts = improvements_df['improvement_pct']
        
        # Test if improvements are consistently positive
        _, p_value = stats.wilcoxon(improvement_values)
        
        analysis = {
            'improvement_data': improvements,
            'statistics': {
                'mean_improvement': float(improvement_values.mean()),
                'median_improvement': float(improvement_values.median()),
                'std_improvement': float(improvement_values.std()),
                'mean_improvement_pct': float(improvement_pcts.mean()),
                'median_improvement_pct': float(improvement_pcts.median()),
                'positive_improvements': int(sum(improvement_values > 0)),
                'negative_improvements': int(sum(improvement_values < 0)),
                'no_change': int(sum(improvement_values == 0)),
                'total_exposures': len(improvements)
            },
            'wilcoxon_test': {
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'interpretation': "Improvements are significantly different from zero" if p_value < self.alpha 
                               else "Improvements are not significantly different from zero"
            }
        }
        
        return analysis
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison statistical report.
        
        Returns:
            Dictionary with complete comparison analysis
        """
        report = {
            'significance_tests': {},
            'improvement_analysis': {},
            'summary': {
                'baseline_components': len(self.baseline_stats.summary_df['component'].unique()),
                'comparison_components': len(self.comparison_stats.summary_df['component'].unique()),
                'common_components': len(set(self.baseline_stats.summary_df['component'].unique()) & 
                                       set(self.comparison_stats.summary_df['component'].unique()))
            },
            'recommendations': []
        }
        
        # Analyze each component
        common_components = (set(self.baseline_stats.summary_df['component'].unique()) & 
                           set(self.comparison_stats.summary_df['component'].unique()))
        
        for component in common_components:
            # Significance test
            report['significance_tests'][component] = self.test_improvement_significance(component)
            
            # Improvement analysis
            report['improvement_analysis'][component] = self.analyze_improvement_distribution(component)
            
            # Generate recommendations
            sig_test = report['significance_tests'][component]
            if sig_test.significant:
                if sig_test.effect_size and sig_test.effect_size > 0:
                    report['recommendations'].append(
                        f"Deploy {component} optimization - significant improvement detected"
                    )
                else:
                    report['recommendations'].append(
                        f"Revert {component} optimization - significant deterioration detected"
                    )
            else:
                report['recommendations'].append(
                    f"Consider further optimization for {component} - no significant improvement"
                )
        
        return report


def calculate_statistical_power(effect_size: float, sample_size: int, 
                              alpha: float = 0.05) -> float:
    """Calculate statistical power for given effect size and sample size.
    
    Args:
        effect_size: Effect size (Cohen's d)
        sample_size: Sample size
        alpha: Significance level
        
    Returns:
        Statistical power (0-1)
    """
    from scipy.stats import norm
    
    # Calculate critical value
    z_alpha = norm.ppf(1 - alpha/2)
    
    # Calculate power
    z_beta = z_alpha - effect_size * np.sqrt(sample_size / 2)
    power = 1 - norm.cdf(z_beta)
    
    return power


def recommend_sample_size(effect_size: float, power: float = 0.8, 
                         alpha: float = 0.05) -> int:
    """Recommend sample size for desired statistical power.
    
    Args:
        effect_size: Desired effect size to detect
        power: Desired statistical power
        alpha: Significance level
        
    Returns:
        Recommended sample size
    """
    from scipy.stats import norm
    
    # Calculate required sample size
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    
    return int(np.ceil(n))