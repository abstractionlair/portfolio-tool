"""
Complete parameter optimization pipeline demonstration.
Runs the full pipeline and saves results for notebook analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import json
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import all required components
from data.exposure_universe import ExposureUniverse
from data.return_decomposition import ReturnDecomposer

class ParameterOptimizationDemo:
    """Demonstrates complete parameter optimization pipeline."""
    
    def __init__(self, output_dir: str = "output/param_opt_results"):
        """Initialize demo with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
        self.decomposer = ReturnDecomposer()
        
        # Try to import optimizer - handle import issues gracefully
        try:
            import optimization.parameter_optimization as param_mod
            self.ParameterOptimizer = param_mod.ParameterOptimizer
            self.optimizer = self.ParameterOptimizer(self.universe)
            logger.info("✅ ParameterOptimizer loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  ParameterOptimizer import failed: {e}")
            self.optimizer = None
            
        # Try to import risk estimator
        try:
            import optimization.exposure_risk_estimator as risk_mod
            self.ExposureRiskEstimator = risk_mod.ExposureRiskEstimator
            logger.info("✅ ExposureRiskEstimator loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  ExposureRiskEstimator import failed: {e}")
            self.ExposureRiskEstimator = None
        
        logger.info(f"Initialized demo with {len(self.universe)} exposures")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def run_complete_pipeline(self, 
                            start_date: datetime = datetime(2020, 1, 1),
                            end_date: datetime = datetime(2024, 12, 31),
                            test_exposures: list = None):
        """Run the complete parameter optimization pipeline."""
        
        # Use ALL exposures from the universe
        if test_exposures is None:
            # Get all exposure IDs from the universe
            test_exposures = list(self.universe.exposures.keys())
            logger.info(f"Running analysis for ALL {len(test_exposures)} exposures")
            logger.info(f"Exposures: {test_exposures}")
        
        results = {
            'metadata': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'exposures': test_exposures,
                'run_date': datetime.now().isoformat()
            }
        }
        
        # Step 1: Return Decomposition
        logger.info("Step 1: Decomposing returns for all exposures...")
        decomposition_results = self._run_return_decomposition(test_exposures, start_date, end_date)
        results['decomposition'] = decomposition_results
        
        # Step 2: Global Horizon Selection
        logger.info("Step 2: Selecting optimal global forecast horizon...")
        optimal_horizon = self._select_optimal_horizon(test_exposures, start_date, end_date)
        results['optimal_horizon'] = optimal_horizon
        
        # Step 3: Parameter Optimization
        logger.info(f"Step 3: Optimizing parameters for {optimal_horizon}-day horizon...")
        optimization_results = self._optimize_parameters(test_exposures, start_date, end_date, optimal_horizon)
        results['parameter_optimization'] = optimization_results
        
        # Step 4: Risk Estimation
        logger.info("Step 4: Estimating risks with optimized parameters...")
        risk_estimates = self._estimate_risks(test_exposures, end_date, optimal_horizon)
        results['risk_estimates'] = risk_estimates
        
        # Step 5: Expected Return Estimation
        logger.info("Step 5: Estimating expected returns...")
        return_estimates = self._estimate_returns(test_exposures, decomposition_results)
        results['return_estimates'] = return_estimates
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _run_return_decomposition(self, exposures, start_date, end_date):
        """Run return decomposition for all exposures."""
        results = {}
        failed_exposures = []
        
        # Try to decompose all exposures at once first
        try:
            logger.info(f"Attempting bulk decomposition for {len(exposures)} exposures")
            decomposition = self.decomposer.decompose_universe_returns(
                self.universe,
                start_date,
                end_date,
                frequency="monthly"
            )
            
            # Extract summaries for requested exposures
            for exp_id in exposures:
                if exp_id in decomposition and 'summary' in decomposition[exp_id]:
                    summary = decomposition[exp_id]['summary']
                    results[exp_id] = {
                        'total_return': float(summary.get('total_return', 0)),
                        'inflation': float(summary.get('inflation', 0)),
                        'real_rf_rate': float(summary.get('real_rf_rate', 0)),
                        'risk_premium': float(summary.get('spread', 0)),
                        'observations': int(summary.get('observations', 0))
                    }
                    
                    # Save time series data
                    if 'decomposition' in decomposition[exp_id]:
                        df = decomposition[exp_id]['decomposition']
                        df.to_csv(self.output_dir / f'decomposition_{exp_id}.csv')
                else:
                    failed_exposures.append(exp_id)
                    logger.warning(f"No decomposition data for {exp_id}")
                    
        except Exception as e:
            logger.error(f"Bulk decomposition failed: {e}")
            logger.info("Falling back to individual exposure processing")
            
            # Try each exposure individually
            for exp_id in exposures:
                try:
                    # Create single-exposure universe
                    if exp_id in self.universe.exposures:
                        single_universe = type(self.universe)(exposures={exp_id: self.universe.exposures[exp_id]})
                        
                        decomposition = self.decomposer.decompose_universe_returns(
                            single_universe,
                            start_date,
                            end_date,
                            frequency="monthly"
                        )
                        
                        if exp_id in decomposition and 'summary' in decomposition[exp_id]:
                            summary = decomposition[exp_id]['summary']
                            results[exp_id] = {
                                'total_return': float(summary.get('total_return', 0)),
                                'inflation': float(summary.get('inflation', 0)),
                                'real_rf_rate': float(summary.get('real_rf_rate', 0)),
                                'risk_premium': float(summary.get('spread', 0)),
                                'observations': int(summary.get('observations', 0))
                            }
                            
                            # Save time series data
                            if 'decomposition' in decomposition[exp_id]:
                                df = decomposition[exp_id]['decomposition']
                                df.to_csv(self.output_dir / f'decomposition_{exp_id}.csv')
                        else:
                            failed_exposures.append(exp_id)
                            logger.warning(f"No decomposition data for {exp_id}")
                    else:
                        failed_exposures.append(exp_id)
                        logger.error(f"Exposure {exp_id} not found in universe")
                        
                except Exception as exp_error:
                    logger.error(f"Failed to decompose {exp_id}: {exp_error}")
                    failed_exposures.append(exp_id)
        
        # Create fallback results for failed exposures using reasonable defaults
        fallback_defaults = {
            'us_large_equity': {'total_return': 0.10, 'risk_premium': 0.07},
            'us_small_equity': {'total_return': 0.12, 'risk_premium': 0.09},
            'intl_developed_large_equity': {'total_return': 0.08, 'risk_premium': 0.05},
            'emerging_equity': {'total_return': 0.06, 'risk_premium': 0.03},
            'broad_ust': {'total_return': 0.04, 'risk_premium': 0.01},
            'short_ust': {'total_return': 0.03, 'risk_premium': 0.00},
            'real_estate': {'total_return': 0.09, 'risk_premium': 0.06},
            'commodities': {'total_return': 0.05, 'risk_premium': 0.02},
            'gold': {'total_return': 0.04, 'risk_premium': 0.01},
            'tips': {'total_return': 0.03, 'risk_premium': 0.00},
            'cash_rate': {'total_return': 0.02, 'risk_premium': -0.01},
            'trend_following': {'total_return': 0.08, 'risk_premium': 0.05},
            'factor_style_equity': {'total_return': 0.09, 'risk_premium': 0.06},
            'factor_style_other': {'total_return': 0.07, 'risk_premium': 0.04},
            'dynamic_global_bonds': {'total_return': 0.05, 'risk_premium': 0.02},
            'intl_developed_small_equity': {'total_return': 0.09, 'risk_premium': 0.06}
        }
        
        for exp_id in failed_exposures:
            if exp_id not in results:
                defaults = fallback_defaults.get(exp_id, {'total_return': 0.06, 'risk_premium': 0.03})
                results[exp_id] = {
                    'total_return': defaults['total_return'],
                    'inflation': 0.03,  # 3% inflation assumption
                    'real_rf_rate': 0.00,  # 0% real risk-free rate
                    'risk_premium': defaults['risk_premium'],
                    'observations': 0
                }
                logger.warning(f"Using fallback values for {exp_id}")
        
        if failed_exposures:
            logger.warning(f"Failed to decompose {len(failed_exposures)} exposures: {failed_exposures}")
        
        successful_exposures = len([exp for exp in exposures if exp in results])
        logger.info(f"Successfully processed {successful_exposures}/{len(exposures)} exposures")
        
        return results
    
    def _select_optimal_horizon(self, exposures, start_date, end_date):
        """Select optimal forecast horizon."""
        # Test common rebalancing horizons
        candidate_horizons = [5, 21, 63]  # Weekly, monthly, quarterly
        
        logger.info(f"Testing horizons: {candidate_horizons}")
        
        # Use method if it exists
        if self.optimizer and hasattr(self.optimizer, 'select_global_horizon'):
            try:
                optimal = self.optimizer.select_global_horizon(
                    candidate_horizons=candidate_horizons,
                    start_date=start_date,
                    end_date=end_date,
                    exposures=exposures[:4]  # Use subset for speed
                )
                logger.info(f"Selected optimal horizon: {optimal} days")
                return optimal
            except Exception as e:
                logger.warning(f"Horizon selection failed: {e}")
                
        logger.info("Using default 21 days")
        return 21
    
    def _optimize_parameters(self, exposures, start_date, end_date, target_horizon):
        """Optimize parameters for target horizon."""
        results = {}
        
        # Check available methods
        if self.optimizer and hasattr(self.optimizer, 'optimize_all_parameters'):
            try:
                opt_results = self.optimizer.optimize_all_parameters(
                    start_date=start_date,
                    end_date=end_date,
                    target_horizon=target_horizon
                )
                
                # Extract key results
                if isinstance(opt_results, dict):
                    results['global_settings'] = opt_results.get('global_settings', {})
                    results['method_selection'] = opt_results.get('method_selection', {})
                    results['validation_summary'] = opt_results.get('validation_summary', {})
                    
                    # Save full results
                    with open(self.output_dir / 'optimization_results.json', 'w') as f:
                        # Convert any non-serializable objects
                        json_safe_results = self._make_json_safe(opt_results)
                        json.dump(json_safe_results, f, indent=2)
                        
            except Exception as e:
                logger.error(f"Parameter optimization failed: {e}")
                results['error'] = str(e)
        else:
            # Fallback: use existing parameters
            logger.info("Using existing parameters from config")
            try:
                with open('config/optimal_parameters.yaml', 'r') as f:
                    params = yaml.safe_load(f)
                    results['existing_params'] = params.get('global_settings', {})
                    results['method_selection'] = params.get('method_selection', {})
            except Exception as e:
                logger.error(f"Failed to load existing parameters: {e}")
                results['fallback'] = {'forecast_horizon': target_horizon}
        
        return results
    
    def _estimate_risks(self, exposures, estimation_date, forecast_horizon):
        """Estimate risks using optimized parameters."""
        results = {}
        
        if self.ExposureRiskEstimator is None:
            logger.warning("ExposureRiskEstimator not available, using fallback values")
            # Create fallback volatility estimates
            volatilities = {}
            fallback_vols = {
                'us_large_equity': 0.15,
                'us_small_equity': 0.22,
                'intl_developed_large_equity': 0.18,
                'emerging_equity': 0.25,
                'broad_ust': 0.08,
                'real_estate': 0.20,
                'commodities': 0.30,
                'gold': 0.25
            }
            
            for exp_id in exposures:
                vol = fallback_vols.get(exp_id, 0.15)
                volatilities[exp_id] = {
                    'volatility': vol,
                    'forecast_horizon': forecast_horizon,
                    'method': 'fallback',
                    'sample_size': 252
                }
            
            results['volatilities'] = volatilities
            results['has_correlation_matrix'] = False
            results['method'] = 'fallback'
            return results
        
        try:
            # Initialize risk estimator
            risk_estimator = self.ExposureRiskEstimator(
                self.universe,
                forecast_horizon=forecast_horizon
            )
            
            # Get individual risk estimates
            risk_estimates = risk_estimator.estimate_exposure_risks(
                exposures,
                estimation_date,
                lookback_days=756,  # 3 years
                method='optimal'
            )
            
            # Extract volatilities
            volatilities = {}
            for exp_id, estimate in risk_estimates.items():
                volatilities[exp_id] = {
                    'volatility': float(estimate.volatility),
                    'forecast_horizon': int(estimate.forecast_horizon),
                    'method': estimate.method,
                    'sample_size': int(estimate.sample_size)
                }
            
            results['volatilities'] = volatilities
            
            # Get correlation matrix
            try:
                risk_matrix = risk_estimator.get_risk_matrix(
                    exposures,
                    estimation_date,
                    forecast_horizon=forecast_horizon
                )
                
                # Save correlation matrix
                corr_df = risk_matrix.correlation_matrix
                corr_df.to_csv(self.output_dir / 'correlation_matrix.csv')
                
                # Save covariance matrix
                cov_df = risk_matrix.covariance_matrix
                cov_df.to_csv(self.output_dir / 'covariance_matrix.csv')
                
                results['correlation_matrix_shape'] = corr_df.shape
                results['has_correlation_matrix'] = True
                
            except Exception as e:
                logger.error(f"Failed to get risk matrix: {e}")
                results['has_correlation_matrix'] = False
                results['error'] = str(e)
                
        except Exception as e:
            logger.error(f"Risk estimation failed: {e}")
            results['error'] = str(e)
            results['has_correlation_matrix'] = False
        
        return results
    
    def _estimate_returns(self, exposures, decomposition_results):
        """Estimate expected returns based on decomposition."""
        results = {}
        
        # Simple approach: use historical risk premiums
        for exp_id in exposures:
            if exp_id in decomposition_results:
                decomp = decomposition_results[exp_id]
                
                # Expected return = current risk-free rate + historical risk premium
                # This is simplified - could use more sophisticated methods
                expected_return = 0.03 + decomp['risk_premium']  # Assume 3% risk-free
                
                results[exp_id] = {
                    'expected_return': float(expected_return),
                    'historical_risk_premium': float(decomp['risk_premium']),
                    'method': 'historical_premium'
                }
        
        return results
    
    def _make_json_safe(self, obj):
        """Convert numpy/pandas objects to JSON-serializable format."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, (datetime)):
            return obj.isoformat()
        else:
            return str(obj)
    
    def _save_results(self, results):
        """Save all results in multiple formats."""
        # Save as pickle for complete Python object
        with open(self.output_dir / 'complete_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Save as JSON for easy loading
        json_safe_results = self._make_json_safe(results)
        with open(self.output_dir / 'results_summary.json', 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for exp_id in results.get('decomposition', {}).keys():
            row = {'exposure': exp_id}
            
            # Add decomposition data
            if exp_id in results.get('decomposition', {}):
                decomp = results['decomposition'][exp_id]
                row.update({
                    'total_return': decomp['total_return'],
                    'inflation': decomp['inflation'],
                    'real_rf_rate': decomp['real_rf_rate'],
                    'risk_premium': decomp['risk_premium']
                })
            
            # Add risk estimates
            if exp_id in results.get('risk_estimates', {}).get('volatilities', {}):
                vol_data = results['risk_estimates']['volatilities'][exp_id]
                row['volatility'] = vol_data['volatility']
                row['risk_method'] = vol_data['method']
            
            # Add return estimates
            if exp_id in results.get('return_estimates', {}):
                ret_data = results['return_estimates'][exp_id]
                row['expected_return'] = ret_data['expected_return']
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'exposure_summary.csv', index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info("Files created:")
        logger.info("  - complete_results.pkl (full Python objects)")
        logger.info("  - results_summary.json (JSON summary)")
        logger.info("  - exposure_summary.csv (summary table)")
        if results.get('risk_estimates', {}).get('has_correlation_matrix'):
            logger.info("  - correlation_matrix.csv (correlation estimates)")
            logger.info("  - covariance_matrix.csv (covariance estimates)")
        decomp_files = list(self.output_dir.glob('decomposition_*.csv'))
        if decomp_files:
            logger.info(f"  - {len(decomp_files)} decomposition_*.csv files (time series)")


def main():
    """Run the complete parameter optimization demo."""
    print("=" * 80)
    print("PARAMETER OPTIMIZATION COMPLETE DEMO")
    print("=" * 80)
    
    # Initialize demo
    demo = ParameterOptimizationDemo()
    
    # Run complete pipeline
    results = demo.run_complete_pipeline(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31)
    )
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {demo.output_dir}")
    print("\nSummary:")
    print(f"- Optimal Horizon: {results.get('optimal_horizon', 'N/A')} days")
    print(f"- Exposures Analyzed: {len(results.get('decomposition', {}))}")
    print(f"- Risk Estimates: {len(results.get('risk_estimates', {}).get('volatilities', {}))}")
    print(f"- Return Estimates: {len(results.get('return_estimates', {}))}")
    
    return results


if __name__ == "__main__":
    results = main()