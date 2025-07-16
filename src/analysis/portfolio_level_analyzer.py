"""
Portfolio-Level Optimization Results Analyzer

This module provides comprehensive analysis capabilities for portfolio-level parameter optimization results.
It loads results from the two-stage optimization and provides methods to analyze parameter quality,
validation metrics, and method selection performance.
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OptimizationResults:
    """Container for portfolio-level optimization results"""
    optimal_horizon: int
    portfolio_rmse: float
    goodness_score: float
    volatility_parameters: Dict
    correlation_parameters: Dict
    return_parameters: Dict
    validation_metrics: Dict
    horizon_comparison: Dict
    optimization_metadata: Dict

class PortfolioLevelAnalyzer:
    """
    Analyzes portfolio-level parameter optimization results
    
    This class loads results from the two-stage optimization and provides
    methods to analyze parameter quality, validation performance, and
    method selection effectiveness.
    """
    
    def __init__(self, results_dir: str = "output/portfolio_level_optimization"):
        # Handle relative paths from different working directories
        if not Path(results_dir).exists():
            # Try from parent directory (for notebooks)
            parent_results_dir = Path("..") / results_dir
            if parent_results_dir.exists():
                self.results_dir = parent_results_dir
            else:
                self.results_dir = Path(results_dir)
        else:
            self.results_dir = Path(results_dir)
            
        # Same logic for config directory  
        if not Path("config").exists():
            parent_config_dir = Path("../config")
            if parent_config_dir.exists():
                self.config_dir = parent_config_dir
            else:
                self.config_dir = Path("config")
        else:
            self.config_dir = Path("config")
            
        self.results = None
        self.detailed_results = None
        
    def load_results(self) -> OptimizationResults:
        """Load optimization results from files"""
        
        # Load detailed JSON results (primary source)
        results_file = self.results_dir / "portfolio_level_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
            
        with open(results_file, 'r') as f:
            detailed_data = json.load(f)
            
        # Try to load YAML configuration for additional metadata
        config_file = self.config_dir / "optimal_parameters_portfolio_level.yaml"
        config_data = None
        
        if config_file.exists():
            try:
                # Try to parse YAML, extracting just the structural info we need
                import re
                with open(config_file, 'r') as f:
                    content = f.read()
                    
                # Extract optimal horizon
                horizon_match = re.search(r'optimal_horizon:\s*(\d+)', content)
                optimal_horizon = int(horizon_match.group(1)) if horizon_match else 63
                
                # Extract parameter structure info
                volatility_params = {}
                
                # Look for parameter blocks
                param_blocks = re.findall(r'(\w+):\s*\n\s*horizon:\s*(\d+)\s*\n\s*method:\s*(\w+)', content)
                for exposure, horizon, method in param_blocks:
                    # Extract lambda if present  
                    lambda_match = re.search(rf'{exposure}:.*?lambda:\s*([\d.]+)', content, re.DOTALL)
                    lookback_match = re.search(rf'{exposure}:.*?lookback_days:\s*(\d+)', content, re.DOTALL)
                    
                    volatility_params[exposure] = {
                        'method': method,
                        'horizon': int(horizon),
                        'parameters': {
                            'lambda': float(lambda_match.group(1)) if lambda_match else None,
                            'lookback_days': int(lookback_match.group(1)) if lookback_match else 252
                        },
                        'validation_score': 0.0  # Default value since we can't parse numpy scalars
                    }
                
                config_data = {
                    'optimal_horizon': optimal_horizon,
                    'volatility_parameters': volatility_params
                }
                
            except Exception as e:
                print(f"Warning: Could not parse YAML config: {e}")
                config_data = None
        
        # Use JSON data as primary source, supplement with YAML if available
        optimal_horizon = detailed_data['optimal_horizon']
        validation_metrics = detailed_data['optimal_parameters']['validation_metrics']
        
        # Extract volatility parameters from JSON or YAML
        if config_data and 'volatility_parameters' in config_data:
            volatility_parameters = config_data['volatility_parameters']
        else:
            # Fallback: extract from JSON structure if available
            volatility_parameters = {}
            
        # Extract horizon comparison from JSON data
        horizon_comparison = {}
        if 'all_horizon_results' in detailed_data:
            for horizon_str, result in detailed_data['all_horizon_results'].items():
                horizon_comparison[f"{horizon_str}_day"] = {
                    'goodness_score': result['goodness_score'],
                    'portfolio_rmse': result['validation_metrics'].get('vol_rmse', 
                                                                     result['validation_metrics'].get('rmse', 0.0))
                }
        
        # Extract return parameters from JSON or YAML
        if config_data and 'return_parameters' in config_data:
            return_parameters = config_data['return_parameters']
        else:
            # Fallback: extract from JSON structure if available
            return_parameters = detailed_data['optimal_parameters'].get('return_params', {})
            
        # Create results object
        self.results = OptimizationResults(
            optimal_horizon=optimal_horizon,
            portfolio_rmse=validation_metrics.get('vol_rmse', validation_metrics.get('rmse', 0.0)),
            goodness_score=detailed_data['optimal_parameters']['goodness_score'],
            volatility_parameters=volatility_parameters,
            correlation_parameters={},  # Will be populated if needed
            return_parameters=return_parameters,
            validation_metrics=validation_metrics,
            horizon_comparison=horizon_comparison,
            optimization_metadata=detailed_data.get('optimization_metadata', {})
        )
        
        self.detailed_results = detailed_data
        return self.results
        
    def get_exposure_summary(self) -> pd.DataFrame:
        """Get summary of all exposures with their optimal parameters"""
        
        if self.results is None:
            self.load_results()
            
        summary_data = []
        
        # If we have volatility parameters from YAML
        if self.results.volatility_parameters:
            for exposure_id, params in self.results.volatility_parameters.items():
                row = {
                    'exposure': exposure_id,
                    'method': params['method'],
                    'horizon': params['horizon'],
                    'validation_score': params.get('validation_score', np.nan)
                }
                
                # Add method-specific parameters
                if params['method'] == 'ewma':
                    row['lambda'] = params['parameters']['lambda']
                    row['lookback_days'] = params['parameters']['lookback_days']
                elif params['method'] == 'historical':
                    row['lambda'] = np.nan
                    row['lookback_days'] = params['parameters']['lookback_days']
                elif params['method'] == 'garch':
                    row['lambda'] = np.nan
                    row['lookback_days'] = params['parameters'].get('lookback_days', np.nan)
                    
                summary_data.append(row)
        else:
            # Fallback: create basic summary from validation results
            detailed_metrics = self.get_detailed_validation_results()
            if detailed_metrics:
                # Get exposure list from first test portfolio
                exposures = list(detailed_metrics[0]['portfolio'].keys())
                for exposure_id in exposures:
                    row = {
                        'exposure': exposure_id,
                        'method': 'unknown',
                        'horizon': self.results.optimal_horizon,
                        'validation_score': np.nan,
                        'lambda': np.nan,
                        'lookback_days': np.nan
                    }
                    summary_data.append(row)
            
        return pd.DataFrame(summary_data)
        
    def get_method_distribution(self) -> pd.Series:
        """Get distribution of methods across exposures"""
        
        if self.results is None:
            self.load_results()
            
        if self.results.volatility_parameters:
            methods = [params['method'] for params in self.results.volatility_parameters.values()]
            return pd.Series(methods).value_counts()
        else:
            # Fallback: return empty series or estimated distribution
            return pd.Series({'unknown': len(self.get_exposure_summary())})
        
    def get_validation_metrics(self) -> Dict:
        """Get portfolio-level validation metrics"""
        
        if self.results is None:
            self.load_results()
            
        return {
            'portfolio_rmse': self.results.portfolio_rmse,
            'portfolio_mse': self.results.validation_metrics['mse'],
            'goodness_score': self.results.goodness_score,
            'n_tests': self.results.validation_metrics['n_tests'],
            'optimal_horizon': self.results.optimal_horizon
        }
        
    def get_horizon_comparison(self) -> pd.DataFrame:
        """Get comparison of different horizons tested"""
        
        if self.results is None:
            self.load_results()
            
        comparison_data = []
        
        if self.results.horizon_comparison:
            for horizon, metrics in self.results.horizon_comparison.items():
                # Extract numeric values from numpy scalars if needed
                if hasattr(metrics['goodness_score'], 'item'):
                    goodness_score = metrics['goodness_score'].item()
                    portfolio_rmse = metrics['portfolio_rmse'].item()
                else:
                    goodness_score = metrics['goodness_score']
                    portfolio_rmse = metrics['portfolio_rmse']
                    
                comparison_data.append({
                    'horizon': horizon,
                    'goodness_score': goodness_score,
                    'portfolio_rmse': portfolio_rmse
                })
        else:
            # Fallback: create single entry for current optimal horizon
            comparison_data.append({
                'horizon': f"{self.results.optimal_horizon}_day",
                'goodness_score': self.results.goodness_score,
                'portfolio_rmse': self.results.portfolio_rmse
            })
            
        df = pd.DataFrame(comparison_data)
        df['horizon_days'] = df['horizon'].str.replace('_day', '').astype(int)
        return df.sort_values('horizon_days')
        
    def get_detailed_validation_results(self) -> List[Dict]:
        """Get detailed validation results for each test portfolio"""
        
        if self.results is None:
            self.load_results()
            
        return self.results.validation_metrics.get('detailed_metrics', [])
        
    def get_test_portfolios_performance(self) -> pd.DataFrame:
        """Analyze performance across different test portfolios"""
        
        detailed_metrics = self.get_detailed_validation_results()
        
        portfolio_data = []
        for i, test_result in enumerate(detailed_metrics):
            # Identify portfolio type based on weights
            weights = test_result['portfolio']
            weight_values = list(weights.values())
            
            # Simple classification based on weight distribution
            if all(abs(w - weight_values[0]) < 0.001 for w in weight_values if w > 0):
                portfolio_type = "Equal Weight"
            elif any(w > 0.15 for w in weight_values):
                portfolio_type = "Concentrated"
            else:
                portfolio_type = "Diversified"
                
            portfolio_data.append({
                'test_id': i,
                'portfolio_type': portfolio_type,
                'predicted_vol': test_result['predicted_vol'],
                'realized_vol': test_result['realized_vol'],
                'error': test_result['error'],
                'abs_error': abs(test_result['error']),
                'relative_error': test_result['error'] / test_result['realized_vol']
            })
            
        return pd.DataFrame(portfolio_data)
        
    def get_lambda_distribution(self) -> pd.DataFrame:
        """Analyze lambda parameter distribution for EWMA methods"""
        
        if self.results is None:
            self.load_results()
            
        lambda_data = []
        for exposure_id, params in self.results.volatility_parameters.items():
            if params['method'] == 'ewma':
                lambda_data.append({
                    'exposure': exposure_id,
                    'lambda': params['parameters']['lambda'],
                    'lookback_days': params['parameters']['lookback_days']
                })
                
        return pd.DataFrame(lambda_data)
        
    def get_lookback_analysis(self) -> pd.DataFrame:
        """Analyze lookback period distribution across methods"""
        
        if self.results is None:
            self.load_results()
            
        lookback_data = []
        for exposure_id, params in self.results.volatility_parameters.items():
            lookback_data.append({
                'exposure': exposure_id,
                'method': params['method'],
                'lookback_days': params['parameters']['lookback_days']
            })
            
        return pd.DataFrame(lookback_data)
        
    def calculate_improvement_metrics(self) -> Dict:
        """Calculate improvement metrics from horizon comparison"""
        
        horizon_df = self.get_horizon_comparison()
        
        if len(horizon_df) < 2:
            # Only one horizon available, no improvement to calculate
            single_horizon = horizon_df.iloc[0]
            return {
                'best_horizon': single_horizon['horizon'],
                'worst_horizon': single_horizon['horizon'],
                'rmse_improvement': 0.0,
                'goodness_improvement': 0.0,
                'best_rmse': single_horizon['portfolio_rmse'],
                'worst_rmse': single_horizon['portfolio_rmse']
            }
        
        # Find best and worst performing horizons
        best_horizon = horizon_df.loc[horizon_df['goodness_score'].idxmax()]
        worst_horizon = horizon_df.loc[horizon_df['goodness_score'].idxmin()]
        
        # Calculate improvements
        rmse_improvement = (worst_horizon['portfolio_rmse'] - best_horizon['portfolio_rmse']) / worst_horizon['portfolio_rmse']
        goodness_improvement = (best_horizon['goodness_score'] - worst_horizon['goodness_score']) / abs(worst_horizon['goodness_score'])
        
        return {
            'best_horizon': best_horizon['horizon'],
            'worst_horizon': worst_horizon['horizon'],
            'rmse_improvement': rmse_improvement,
            'goodness_improvement': goodness_improvement,
            'best_rmse': best_horizon['portfolio_rmse'],
            'worst_rmse': worst_horizon['portfolio_rmse']
        }
        
    def get_correlation_parameters(self) -> Dict:
        """Get correlation estimation parameters"""
        
        if self.results is None:
            self.load_results()
            
        return self.results.correlation_parameters
        
    def export_summary_report(self, output_file: str = None) -> str:
        """Export a comprehensive summary report"""
        
        if self.results is None:
            self.load_results()
            
        if output_file is None:
            output_file = self.results_dir / "analysis_summary.txt"
        else:
            output_file = Path(output_file)
            
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
            
        report_lines = []
        report_lines.append("PORTFOLIO-LEVEL OPTIMIZATION ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Optimization overview
        report_lines.append("OPTIMIZATION OVERVIEW")
        report_lines.append("-" * 20)
        report_lines.append(f"Optimal Horizon: {self.results.optimal_horizon} days")
        report_lines.append(f"Portfolio RMSE: {self.results.portfolio_rmse:.4f}")
        report_lines.append(f"Goodness Score: {self.results.goodness_score:.2e}")
        report_lines.append(f"Number of Exposures: {len(self.results.volatility_parameters)}")
        report_lines.append("")
        
        # Method distribution
        method_dist = self.get_method_distribution()
        report_lines.append("METHOD DISTRIBUTION")
        report_lines.append("-" * 18)
        for method, count in method_dist.items():
            report_lines.append(f"{method}: {count} exposures")
        report_lines.append("")
        
        # Horizon comparison
        horizon_df = self.get_horizon_comparison()
        report_lines.append("HORIZON COMPARISON")
        report_lines.append("-" * 18)
        for _, row in horizon_df.iterrows():
            report_lines.append(f"{row['horizon']}: RMSE={row['portfolio_rmse']:.4f}, Score={row['goodness_score']:.2e}")
        report_lines.append("")
        
        # Improvement metrics
        improvements = self.calculate_improvement_metrics()
        report_lines.append("IMPROVEMENT METRICS")
        report_lines.append("-" * 18)
        report_lines.append(f"Best Horizon: {improvements['best_horizon']}")
        report_lines.append(f"RMSE Improvement: {improvements['rmse_improvement']:.1%}")
        report_lines.append(f"Goodness Improvement: {improvements['goodness_improvement']:.1%}")
        report_lines.append("")
        
        # Validation summary
        validation = self.get_validation_metrics()
        report_lines.append("VALIDATION SUMMARY")
        report_lines.append("-" * 17)
        report_lines.append(f"Number of Tests: {validation['n_tests']}")
        report_lines.append(f"Portfolio MSE: {validation['portfolio_mse']:.2e}")
        report_lines.append(f"Portfolio RMSE: {validation['portfolio_rmse']:.4f}")
        report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        with open(output_file, 'w') as f:
            f.write(report_content)
            
        return str(output_file)
        
    def get_optimization_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        
        if self.results is None:
            self.load_results()
            
        horizon_df = self.get_horizon_comparison()
        
        # Calculate performance metrics
        best_rmse = horizon_df['portfolio_rmse'].min()
        worst_rmse = horizon_df['portfolio_rmse'].max()
        median_rmse = horizon_df['portfolio_rmse'].median()
        
        # Find optimal vs suboptimal
        optimal_horizon = horizon_df.loc[horizon_df['portfolio_rmse'].idxmin()]
        
        return {
            'optimal_horizon_days': optimal_horizon['horizon_days'],
            'optimal_rmse': best_rmse,
            'worst_rmse': worst_rmse,
            'median_rmse': median_rmse,
            'rmse_range': worst_rmse - best_rmse,
            'relative_improvement': (worst_rmse - best_rmse) / worst_rmse,
            'horizons_tested': len(horizon_df),
            'performance_curve': horizon_df[['horizon_days', 'portfolio_rmse']].to_dict('records')
        }
    
    def get_return_prediction_analysis(self) -> Dict:
        """Analyze return prediction performance across exposures, methods, and horizons"""
        
        if self.results is None:
            self.load_results()
            
        if not self.results.return_parameters:
            return {'note': 'No return parameters found in results'}
            
        # Analyze return prediction methods
        return_analysis = {
            'method_distribution': {},
            'exposure_performance': {},
            'parameter_analysis': {},
            'horizon_analysis': {}
        }
        
        # Method distribution
        for exposure_id, params in self.results.return_parameters.items():
            method = params['method']
            score = params['validation_score']
            
            if method not in return_analysis['method_distribution']:
                return_analysis['method_distribution'][method] = {
                    'count': 0,
                    'avg_score': 0.0,
                    'scores': []
                }
            
            return_analysis['method_distribution'][method]['count'] += 1
            return_analysis['method_distribution'][method]['scores'].append(score)
            
            # Exposure performance
            return_analysis['exposure_performance'][exposure_id] = {
                'method': method,
                'score': score,
                'horizon': params['horizon'],
                'parameters': params['parameters']
            }
        
        # Calculate average scores for each method
        for method, data in return_analysis['method_distribution'].items():
            data['avg_score'] = np.mean(data['scores'])
            data['std_score'] = np.std(data['scores'])
            data['min_score'] = np.min(data['scores'])
            data['max_score'] = np.max(data['scores'])
        
        # Parameter analysis by method
        for method in return_analysis['method_distribution'].keys():
            method_params = []
            for exposure_id, params in self.results.return_parameters.items():
                if params['method'] == method:
                    method_params.append({
                        'exposure': exposure_id,
                        'score': params['validation_score'],
                        'parameters': params['parameters']
                    })
            
            return_analysis['parameter_analysis'][method] = method_params
        
        # Horizon analysis - compare across all horizons if available
        if self.detailed_results and 'all_horizon_results' in self.detailed_results:
            for horizon_str, result in self.detailed_results['all_horizon_results'].items():
                horizon = int(horizon_str)
                
                if 'return_params' in result:
                    horizon_return_accuracy = []
                    for exposure_id, params in result['return_params'].items():
                        if 'validation_score' in params:
                            horizon_return_accuracy.append(params['validation_score'])
                    
                    if horizon_return_accuracy:
                        return_analysis['horizon_analysis'][horizon] = {
                            'avg_return_accuracy': np.mean(horizon_return_accuracy),
                            'std_return_accuracy': np.std(horizon_return_accuracy),
                            'min_return_accuracy': np.min(horizon_return_accuracy),
                            'max_return_accuracy': np.max(horizon_return_accuracy),
                            'n_exposures': len(horizon_return_accuracy)
                        }
        
        return return_analysis
    
    def get_return_prediction_errors_by_exposure(self) -> pd.DataFrame:
        """Get detailed return prediction errors by exposure"""
        
        if self.results is None:
            self.load_results()
            
        if not self.results.return_parameters:
            return pd.DataFrame()
            
        error_data = []
        
        for exposure_id, params in self.results.return_parameters.items():
            # Extract error information
            score = params['validation_score']
            method = params['method']
            horizon = params['horizon']
            
            # Directional accuracy interpretation
            # Higher score = better directional accuracy
            error_rate = 1.0 - score  # Error rate = 1 - accuracy
            
            error_data.append({
                'exposure': exposure_id,
                'method': method,
                'horizon': horizon,
                'directional_accuracy': score,
                'error_rate': error_rate,
                'parameters': str(params['parameters'])
            })
        
        return pd.DataFrame(error_data)
    
    def get_return_prediction_errors_by_method(self) -> pd.DataFrame:
        """Get return prediction errors aggregated by method"""
        
        errors_df = self.get_return_prediction_errors_by_exposure()
        if errors_df.empty:
            return pd.DataFrame()
        
        # Group by method and calculate statistics
        method_stats = errors_df.groupby('method').agg({
            'directional_accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'error_rate': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Flatten column names
        method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns]
        method_stats = method_stats.reset_index()
        
        return method_stats
    
    def get_return_prediction_errors_by_horizon(self) -> pd.DataFrame:
        """Get return prediction errors by horizon across all tested horizons"""
        
        if self.results is None:
            self.load_results()
            
        if not self.detailed_results or 'all_horizon_results' not in self.detailed_results:
            return pd.DataFrame()
        
        horizon_data = []
        
        for horizon_str, result in self.detailed_results['all_horizon_results'].items():
            horizon = int(horizon_str)
            
            if 'return_params' in result:
                return_accuracies = []
                method_counts = {}
                
                for exposure_id, params in result['return_params'].items():
                    if 'validation_score' in params:
                        return_accuracies.append(params['validation_score'])
                        method = params['method']
                        method_counts[method] = method_counts.get(method, 0) + 1
                
                if return_accuracies:
                    horizon_data.append({
                        'horizon': horizon,
                        'avg_directional_accuracy': np.mean(return_accuracies),
                        'std_directional_accuracy': np.std(return_accuracies),
                        'min_directional_accuracy': np.min(return_accuracies),
                        'max_directional_accuracy': np.max(return_accuracies),
                        'avg_error_rate': 1.0 - np.mean(return_accuracies),
                        'n_exposures': len(return_accuracies),
                        'dominant_method': max(method_counts, key=method_counts.get) if method_counts else None,
                        'method_diversity': len(method_counts)
                    })
        
        return pd.DataFrame(horizon_data).sort_values('horizon')
    
    def get_return_prediction_parameter_analysis(self) -> Dict:
        """Analyze return prediction parameters for patterns and effectiveness"""
        
        if self.results is None:
            self.load_results()
            
        if not self.results.return_parameters:
            return {'note': 'No return parameters found in results'}
        
        parameter_analysis = {}
        
        # Analyze parameters by method
        for method in ['historical', 'ewma', 'momentum', 'mean_reversion']:
            method_params = []
            method_scores = []
            
            for exposure_id, params in self.results.return_parameters.items():
                if params['method'] == method:
                    method_params.append(params['parameters'])
                    method_scores.append(params['validation_score'])
            
            if method_params:
                parameter_analysis[method] = {
                    'count': len(method_params),
                    'avg_score': np.mean(method_scores),
                    'parameters': method_params,
                    'scores': method_scores
                }
                
                # Analyze parameter patterns
                if method == 'historical':
                    lookbacks = [p.get('lookback_days', 0) for p in method_params]
                    parameter_analysis[method]['avg_lookback'] = np.mean(lookbacks)
                    parameter_analysis[method]['lookback_range'] = [np.min(lookbacks), np.max(lookbacks)]
                    
                elif method == 'ewma':
                    decay_factors = [p.get('decay_factor', 0) for p in method_params]
                    parameter_analysis[method]['avg_decay_factor'] = np.mean(decay_factors)
                    parameter_analysis[method]['decay_range'] = [np.min(decay_factors), np.max(decay_factors)]
                    
                elif method == 'momentum':
                    momentum_periods = [p.get('momentum_period', 0) for p in method_params]
                    momentum_strengths = [p.get('momentum_strength', 0) for p in method_params]
                    parameter_analysis[method]['avg_momentum_period'] = np.mean(momentum_periods)
                    parameter_analysis[method]['avg_momentum_strength'] = np.mean(momentum_strengths)
                    
                elif method == 'mean_reversion':
                    recent_periods = [p.get('recent_period', 0) for p in method_params]
                    reversion_strengths = [p.get('reversion_strength', 0) for p in method_params]
                    parameter_analysis[method]['avg_recent_period'] = np.mean(recent_periods)
                    parameter_analysis[method]['avg_reversion_strength'] = np.mean(reversion_strengths)
        
        return parameter_analysis
        
    def get_parameter_effectiveness_analysis(self) -> Dict:
        """Analyze parameter effectiveness across methods"""
        
        if self.results is None:
            self.load_results()
            
        exposure_summary = self.get_exposure_summary()
        
        # Group by method
        method_analysis = {}
        for method in exposure_summary['method'].unique():
            method_data = exposure_summary[exposure_summary['method'] == method]
            
            method_analysis[method] = {
                'count': len(method_data),
                'avg_lambda': method_data['lambda'].mean() if method == 'ewma' else None,
                'avg_lookback': method_data['lookback_days'].mean(),
                'lambda_range': [method_data['lambda'].min(), method_data['lambda'].max()] if method == 'ewma' else None,
                'lookback_range': [method_data['lookback_days'].min(), method_data['lookback_days'].max()],
                'exposures': method_data['exposure'].tolist()
            }
            
        return method_analysis
        
    def get_horizon_efficiency_analysis(self) -> pd.DataFrame:
        """Analyze efficiency across horizons"""
        
        horizon_df = self.get_horizon_comparison()
        
        # Calculate efficiency metrics
        horizon_df['efficiency_score'] = 1.0 / horizon_df['portfolio_rmse']  # Higher is better
        horizon_df['relative_performance'] = horizon_df['portfolio_rmse'] / horizon_df['portfolio_rmse'].min()
        horizon_df['rebalancing_frequency'] = 252 / horizon_df['horizon_days']  # Times per year
        
        return horizon_df
        
    def get_final_risk_estimates(self) -> Dict:
        """Get the final volatilities and correlations using optimal parameters"""
        
        if self.results is None:
            self.load_results()
            
        # Check if we have detailed results with final estimates
        if self.detailed_results and 'final_estimates' in self.detailed_results:
            final_estimates = self.detailed_results['final_estimates']
            
            # Check if estimates were computed successfully
            if 'error' in final_estimates:
                return {
                    'note': f'Error computing final risk estimates: {final_estimates["error"]}',
                    'available_data': list(self.detailed_results.keys()) if self.detailed_results else []
                }
            else:
                return final_estimates
            
        # If not available in results, we need to compute them
        # This would require running the optimization with the optimal parameters
        # For now, return empty dict with instruction
        return {
            'note': 'Final risk estimates not available in stored results. Run portfolio optimization to generate them.',
            'available_data': list(self.detailed_results.keys()) if self.detailed_results else []
        }
        
    def get_exposure_volatilities(self) -> pd.DataFrame:
        """Get individual exposure volatilities using optimal parameters"""
        
        if self.results is None:
            self.load_results()
            
        # Try to get final risk estimates
        final_estimates = self.get_final_risk_estimates()
        
        # Get exposure parameter summary
        exposure_summary = self.get_exposure_summary()
        
        volatility_data = []
        for _, row in exposure_summary.iterrows():
            exposure_id = row['exposure']
            
            # Initialize data with parameter info
            vol_data = {
                'exposure': exposure_id,
                'method': row['method'],
                'lambda': row['lambda'],
                'lookback_days': row['lookback_days'],
                'annualized_volatility': None,
                'daily_volatility': None
            }
            
            # Try to get computed volatilities from final estimates
            if ('exposure_volatilities' in final_estimates and 
                exposure_id in final_estimates['exposure_volatilities']):
                
                exposure_vol = final_estimates['exposure_volatilities'][exposure_id]
                vol_data['annualized_volatility'] = exposure_vol.get('annualized_volatility')
                vol_data['daily_volatility'] = exposure_vol.get('daily_volatility')
            
            volatility_data.append(vol_data)
            
        return pd.DataFrame(volatility_data)
        
    def get_correlation_matrix_results(self) -> pd.DataFrame:
        """Get the correlation matrix using optimal parameters"""
        
        if self.results is None:
            self.load_results()
            
        # Try to get final risk estimates
        final_estimates = self.get_final_risk_estimates()
        
        # Check if correlation matrix is available in final estimates
        if ('correlation_matrix' in final_estimates and 
            final_estimates['correlation_matrix'] and 
            'note' not in final_estimates):
            
            corr_data = final_estimates['correlation_matrix']
            return pd.DataFrame(corr_data)
            
        # If not available, return placeholder
        exposures = self.get_exposure_summary()['exposure'].tolist()
        n_exposures = len(exposures)
        
        # Create identity matrix as placeholder
        import numpy as np
        identity_matrix = np.eye(n_exposures)
        
        return pd.DataFrame(identity_matrix, index=exposures, columns=exposures)
        
    def compute_portfolio_risk_breakdown(self) -> Dict:
        """Compute portfolio risk breakdown using optimal parameters"""
        
        if self.results is None:
            self.load_results()
            
        # Get test portfolio performance for risk analysis
        test_portfolios = self.get_test_portfolios_performance()
        
        # Analyze portfolio risk components
        risk_breakdown = {
            'total_portfolio_tests': len(test_portfolios),
            'average_predicted_vol': test_portfolios['predicted_vol'].mean(),
            'average_realized_vol': test_portfolios['realized_vol'].mean(),
            'volatility_range': {
                'min_predicted': test_portfolios['predicted_vol'].min(),
                'max_predicted': test_portfolios['predicted_vol'].max(),
                'min_realized': test_portfolios['realized_vol'].min(),
                'max_realized': test_portfolios['realized_vol'].max()
            },
            'prediction_accuracy': {
                'mean_absolute_error': test_portfolios['abs_error'].mean(),
                'mean_relative_error': test_portfolios['relative_error'].mean(),
                'rmse': np.sqrt(test_portfolios['error'].pow(2).mean())
            }
        }
        
        return risk_breakdown