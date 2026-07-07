"""
Parameter Search

This module provides the search space for the parameter optimization.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum
import time
from datetime import datetime
from data.multi_frequency import Frequency
from src.validation import CompatibilityValidationFramework
from src.validation.parameter_validation import ValidationMethod

@dataclass
class ParameterCombination:
    """Dataclass for a parameter combination."""
    method: str
    parameters: Dict[str, Any]
    history_length: int
    horizon: int
    frequency: Frequency
    validation_method: ValidationMethod
    method_description: str = ""

    def to_dict(self):
        return {
            "method": self.method,
            "parameters": self.parameters,
            "history_length": self.history_length,
            "horizon": self.horizon,
            "frequency": self.frequency.value,
            "validation_method": self.validation_method.value,
            "method_description": self.method_description,
        }

@dataclass
class SearchResult:
    """Dataclass for a search result."""
    combination: ParameterCombination
    success: bool
    successful_exposures: int
    failed_exposures: int
    total_exposures: int
    success_rate_across_exposures: float
    error_message: str = ""
    exposure_results: List[Dict] = field(default_factory=list)
    aggregate_metrics: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "combination": self.combination.to_dict(),
            "success": self.success,
            "successful_exposures": self.successful_exposures,
            "failed_exposures": self.failed_exposures,
            "total_exposures": self.total_exposures,
            "success_rate_across_exposures": self.success_rate_across_exposures,
            "error_message": self.error_message,
            "exposure_results": self.exposure_results,
            "aggregate_metrics": self.aggregate_metrics,
        }

@dataclass
class SearchConfiguration:
    """Dataclass for search configuration."""
    history_lengths: List[int] = field(default_factory=lambda: [126, 252, 504, 756, 1008])
    frequencies: List[Frequency] = field(default_factory=lambda: [Frequency.MONTHLY, Frequency.WEEKLY])
    horizons: List[int] = field(default_factory=lambda: [21, 42, 63, 126, 252])
    validation_methods: List[ValidationMethod] = field(default_factory=lambda: [ValidationMethod.ADAPTIVE])
    methods: Dict[str, Any] = field(default_factory=lambda: {
        'historical': {
            'description': 'Historical Volatility',
            'parameters': [{'window': w} for w in [20, 40, 60, 120]]
        },
        'ewma': {
            'description': 'Exponentially Weighted Moving Average',
            'parameters': [{'lambda': l, 'min_periods': 10} for l in [0.90, 0.94, 0.97, 0.99]]
        }
    })

    def get_total_combinations(self):
        return sum(len(m['parameters']) for m in self.methods.values()) * len(self.history_lengths) * len(self.frequencies) * len(self.horizons) * len(self.validation_methods)

    def validate(self):
        issues = []
        if not self.history_lengths:
            issues.append("history_lengths cannot be empty")
        if not self.frequencies:
            issues.append("frequencies cannot be empty")
        if not self.horizons:
            issues.append("horizons cannot be empty")
        if not self.validation_methods:
            issues.append("validation_methods cannot be empty")
        if not self.methods:
            issues.append("methods cannot be empty")
        for name, method in self.methods.items():
            if 'parameters' not in method or not method['parameters']:
                issues.append(f"Method '{name}' is missing 'parameters' key or it is empty.")
        return issues

@dataclass
class ProgressTracker:
    """Dataclass for tracking progress."""
    total_combinations: int
    completed: int = 0
    successful_results: List = field(default_factory=list)
    failed_results: List = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def update_progress(self, result):
        self.completed += 1
        if result.success:
            self.successful_results.append(result)
        else:
            self.failed_results.append(result)

    def should_report_progress(self, report_interval):
        return self.completed > 0 and self.completed % report_interval == 0

    def get_summary(self):
        return {
            'total_combinations': self.total_combinations,
            'completed': self.completed,
            'successful': len(self.successful_results),
            'failed': len(self.failed_results),
            'success_rate': len(self.successful_results) / self.completed if self.completed > 0 else 0,
            'runtime_hours': 0,
            'combinations_per_hour': 0
        }

class ParameterSearchEngine:
    """
    A class for defining the parameter search space.
    """

    def __init__(self, exposure_data, available_exposures, validation_framework, search_config):
        """
        Initialize the ParameterSearch.
        """
        self.exposure_data = exposure_data
        self.available_exposures = available_exposures
        self.validation_framework = validation_framework
        self.search_config = search_config
        if self.search_config.validate():
            raise ValueError("Invalid search configuration")
        self.total_combinations = self.search_config.get_total_combinations()

    def generate_parameter_combinations(self):
        combinations = []
        for method_name, method_config in self.search_config.methods.items():
            for params in method_config['parameters']:
                for history in self.search_config.history_lengths:
                    for freq in self.search_config.frequencies:
                        for horizon in self.search_config.horizons:
                            for val_method in self.search_config.validation_methods:
                                combinations.append(ParameterCombination(
                                    method=method_name,
                                    parameters=params,
                                    history_length=history,
                                    horizon=horizon,
                                    frequency=freq,
                                    validation_method=val_method,
                                    method_description=method_config.get('description', '')
                                ))
        return combinations

    def test_parameter_combination(self, combination, estimation_date):
        exposure_results = []
        for exposure_id in self.available_exposures:
            try:
                series = self.exposure_data[exposure_id]['spread']
                result = self.validation_framework.validate_parameter_combination(series, combination.__dict__, combination.validation_method)
                exposure_results.append({
                    'exposure_id': exposure_id,
                    'success': result.success,
                    'validation_result': result.to_dict()
                })
            except KeyError as e:
                # Handle missing 'spread' column or other key errors
                exposure_results.append({
                    'exposure_id': exposure_id,
                    'success': False,
                    'validation_result': {'error': f'Missing required column: {e}'}
                })
        
        successful_exposures = sum(1 for r in exposure_results if r['success'])
        total_exposures = len(exposure_results)
        
        return SearchResult(
            combination=combination,
            success=successful_exposures > 0,
            successful_exposures=successful_exposures,
            failed_exposures=total_exposures - successful_exposures,
            total_exposures=total_exposures,
            success_rate_across_exposures=successful_exposures / total_exposures if total_exposures > 0 else 0,
            exposure_results=exposure_results,
            aggregate_metrics=self._aggregate_exposure_results(exposure_results)
        )

    def _aggregate_exposure_results(self, exposure_results):
        metrics = {}
        valid_results = [r['validation_result'] for r in exposure_results if r['success']]
        if not valid_results:
            return {}
        
        for key in valid_results[0]:
            if isinstance(valid_results[0][key], (int, float)):
                metrics[f'mean_{key}'] = np.mean([r[key] for r in valid_results])
        
        metrics['total_exposures'] = len(exposure_results)
        metrics['exposures_with_valid_mse'] = len(valid_results)
        return metrics

    def run_search(self, estimation_date, save_results=False, report_interval=10):
        combinations = self.generate_parameter_combinations()
        tracker = ProgressTracker(len(combinations))
        
        results = []
        failed_combinations = []

        for combo in combinations:
            result = self.test_parameter_combination(combo, estimation_date)
            tracker.update_progress(result)
            if result.success:
                results.append(result.to_dict())
            else:
                failed_combinations.append(result.to_dict())
            
            if tracker.should_report_progress(report_interval):
                print(f"Progress: {tracker.completed}/{tracker.total_combinations}")

        summary = tracker.get_summary()
        
        results_data = {
            'summary': summary,
            'results': results,
            'failed_combinations': failed_combinations,
            'search_metadata': {
                'exposures_tested': len(self.available_exposures),
                'total_combinations': self.total_combinations,
                'framework_type': 'comprehensive'
            }
        }

        if save_results:
            self._save_results(results_data)

        return results_data

    def get_best_combination(self, results, metric, minimize=True):
        if not results:
            return None
        
        return sorted(results, key=lambda x: x['aggregate_metrics'].get(metric, np.inf if minimize else -np.inf), reverse=not minimize)[0]

    def _save_results(self, results_data):
        import pickle
        import json
        import os
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        pickle_file = f"comprehensive_parameter_search_{timestamp}.pkl"
        json_file = f"comprehensive_search_summary_{timestamp}.json"

        with open(pickle_file, 'wb') as f:
            pickle.dump(results_data, f)

        summary_data = {
            'summary': results_data['summary'],
            'top_10_combinations': [self.get_best_combination(results_data['results'], 'mean_mse', True)] if results_data['results'] else [],
            'search_metadata': results_data['search_metadata']
        }

        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=4)
            
        return {'pickle_file': pickle_file, 'json_file': json_file}
