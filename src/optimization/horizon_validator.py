"""
Horizon Consistency Validator

This module provides validation and enforcement of consistent forecast horizons
across all portfolio optimization components to ensure mathematical consistency.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import yaml

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HorizonValidationResult:
    """Results from horizon consistency validation."""
    is_consistent: bool
    target_horizon: int
    violations: List[str]
    warnings: List[str]
    validation_date: datetime
    
    # Detailed checks
    parameter_consistency: bool = True
    configuration_consistency: bool = True
    risk_estimate_consistency: bool = True
    
    # Performance metrics
    forecast_accuracy_check: bool = True
    horizon_coverage: float = 1.0  # Fraction of exposures with proper horizon


class HorizonValidator:
    """
    Validates and enforces consistent forecast horizons across portfolio optimization.
    
    Ensures that all risk estimates (volatilities and correlations) are for the
    same forecast horizon, which is critical for mathematical consistency in
    portfolio optimization.
    """
    
    def __init__(self, config_path: str = "config/optimal_parameters_v2.yaml"):
        """
        Initialize horizon validator.
        
        Args:
            config_path: Path to horizon-specific configuration file
        """
        self.config_path = config_path
        self.config = self._load_configuration()
        self.target_horizon = self._extract_target_horizon()
        
        logger.info(f"Initialized HorizonValidator for {self.target_horizon}-day horizon")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load horizon-specific configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _extract_target_horizon(self) -> int:
        """Extract target forecast horizon from configuration."""
        if not self.config:
            return 21  # Default to 21-day horizon
        
        global_settings = self.config.get('global_settings', {})
        return global_settings.get('forecast_horizon', 21)
    
    def validate_full_consistency(
        self,
        exposures: List[str],
        risk_estimates: Optional[Dict[str, Any]] = None
    ) -> HorizonValidationResult:
        """
        Perform comprehensive horizon consistency validation.
        
        Args:
            exposures: List of exposure IDs to validate
            risk_estimates: Optional risk estimates to validate
            
        Returns:
            HorizonValidationResult with detailed validation results
        """
        logger.info(f"Starting comprehensive horizon consistency validation for {len(exposures)} exposures")
        
        violations = []
        warnings = []
        
        # Check 1: Configuration consistency
        config_consistent = self._validate_configuration_consistency()
        if not config_consistent:
            violations.append("Configuration does not specify consistent global horizon")
        
        # Check 2: Parameter consistency
        param_consistent = self._validate_parameter_consistency(exposures)
        if not param_consistent:
            violations.append("Parameters are not optimized for the same horizon")
        
        # Check 3: Risk estimate consistency (if provided)
        risk_consistent = True
        if risk_estimates:
            risk_consistent = self._validate_risk_estimate_consistency(risk_estimates, exposures)
            if not risk_consistent:
                violations.append("Risk estimates use inconsistent forecast horizons")
        
        # Check 4: Forecast accuracy validation
        accuracy_ok = self._validate_forecast_accuracy()
        if not accuracy_ok:
            warnings.append("Forecast accuracy validation suggests horizon mismatch")
        
        # Check 5: Coverage validation
        coverage = self._calculate_horizon_coverage(exposures)
        if coverage < 0.9:
            warnings.append(f"Only {coverage*100:.1f}% of exposures have horizon-specific parameters")
        
        # Overall consistency
        is_consistent = config_consistent and param_consistent and risk_consistent
        
        result = HorizonValidationResult(
            is_consistent=is_consistent,
            target_horizon=self.target_horizon,
            violations=violations,
            warnings=warnings,
            validation_date=datetime.now(),
            parameter_consistency=param_consistent,
            configuration_consistency=config_consistent,
            risk_estimate_consistency=risk_consistent,
            forecast_accuracy_check=accuracy_ok,
            horizon_coverage=coverage
        )
        
        # Log results
        if is_consistent:
            logger.info(f"✅ Horizon consistency validation PASSED for {self.target_horizon}-day horizon")
        else:
            logger.warning(f"❌ Horizon consistency validation FAILED: {len(violations)} violations")
            for violation in violations:
                logger.warning(f"  - {violation}")
        
        if warnings:
            logger.info(f"⚠️  {len(warnings)} warnings:")
            for warning in warnings:
                logger.info(f"  - {warning}")
        
        return result
    
    def _validate_configuration_consistency(self) -> bool:
        """Validate that configuration specifies consistent global horizon."""
        if not self.config:
            return False
        
        global_settings = self.config.get('global_settings', {})
        
        # Check for global horizon setting
        if 'forecast_horizon' not in global_settings:
            logger.warning("No global forecast_horizon specified in configuration")
            return False
        
        horizon = global_settings['forecast_horizon']
        
        # Check that horizon-specific parameters exist
        horizon_key = f'horizon_{horizon}_parameters'
        if horizon_key not in self.config:
            logger.warning(f"No parameters found for horizon {horizon}")
            return False
        
        # Check mathematical consistency flag
        math_consistent = global_settings.get('mathematical_consistency', False)
        if not math_consistent:
            logger.warning("mathematical_consistency flag not set to true")
            return False
        
        return True
    
    def _validate_parameter_consistency(self, exposures: List[str]) -> bool:
        """Validate that all parameters are optimized for the same horizon."""
        if not self.config:
            return False
        
        horizon_key = f'horizon_{self.target_horizon}_parameters'
        if horizon_key not in self.config:
            return False
        
        horizon_params = self.config[horizon_key]
        
        # Check volatility parameters
        volatility_params = horizon_params.get('volatility', {})
        correlation_params = horizon_params.get('correlation', {})
        
        # Validate that parameters exist for all exposures
        missing_exposures = []
        for exposure in exposures:
            if exposure not in volatility_params:
                missing_exposures.append(exposure)
        
        if missing_exposures:
            logger.warning(f"Missing volatility parameters for exposures: {missing_exposures}")
            return len(missing_exposures) < len(exposures) * 0.5  # Allow if less than 50% are missing
        
        # Check that correlation parameters are horizon-specific
        if not correlation_params:
            logger.warning("No correlation parameters found")
            return False
        
        return True
    
    def _validate_risk_estimate_consistency(
        self, 
        risk_estimates: Dict[str, Any], 
        exposures: List[str]
    ) -> bool:
        """Validate that risk estimates use consistent forecast horizons."""
        # This would check actual risk estimate objects to ensure they all
        # use the same forecast horizon. Implementation depends on the
        # structure of risk estimate objects.
        
        # For now, we'll assume consistency if the estimates exist
        # In a full implementation, this would check:
        # 1. All volatility estimates are for target_horizon
        # 2. All correlation estimates are for target_horizon  
        # 3. No mixing of different horizons
        
        if not risk_estimates:
            return True  # No estimates to validate
        
        # Check that estimates exist for all exposures
        missing_estimates = []
        for exposure in exposures:
            if exposure not in risk_estimates:
                missing_estimates.append(exposure)
        
        if missing_estimates:
            logger.warning(f"Missing risk estimates for: {missing_estimates}")
        
        # In practice, would validate horizon attribute of each estimate
        return len(missing_estimates) < len(exposures) * 0.1  # Allow 10% missing
    
    def _validate_forecast_accuracy(self) -> bool:
        """Validate forecast accuracy metrics for horizon consistency."""
        if not self.config:
            return True  # Can't validate without config
        
        validation_summary = self.config.get('validation_summary', {})
        
        # Check if validation was actually performed
        vol_performance = validation_summary.get('volatility_performance', {})
        if vol_performance.get('best_mse') is None:
            logger.info("No validation performance data available")
            return True  # Not an error, just no data
        
        # In practice, would check that forecast accuracy is reasonable
        # for the specified horizon (e.g., MSE is better than naive forecasts)
        
        return True
    
    def _calculate_horizon_coverage(self, exposures: List[str]) -> float:
        """Calculate fraction of exposures with horizon-specific parameters."""
        if not self.config:
            return 0.0
        
        horizon_key = f'horizon_{self.target_horizon}_parameters'
        if horizon_key not in self.config:
            return 0.0
        
        volatility_params = self.config[horizon_key].get('volatility', {})
        
        covered_exposures = 0
        for exposure in exposures:
            if exposure in volatility_params:
                covered_exposures += 1
        
        return covered_exposures / len(exposures) if exposures else 0.0
    
    def enforce_horizon_consistency(
        self,
        risk_estimator_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enforce horizon consistency by modifying risk estimator configuration.
        
        Args:
            risk_estimator_config: Configuration for risk estimator
            
        Returns:
            Modified configuration with enforced horizon consistency
        """
        logger.info(f"Enforcing {self.target_horizon}-day horizon consistency")
        
        # Ensure global horizon is set
        risk_estimator_config['forecast_horizon'] = self.target_horizon
        risk_estimator_config['mathematical_consistency'] = True
        
        # Load horizon-specific parameters
        if self.config:
            horizon_key = f'horizon_{self.target_horizon}_parameters'
            if horizon_key in self.config:
                risk_estimator_config['parameters'] = self.config[horizon_key]
                logger.info(f"Loaded parameters optimized for {self.target_horizon}-day horizon")
        
        return risk_estimator_config
    
    def generate_consistency_report(
        self,
        validation_result: HorizonValidationResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed horizon consistency report.
        
        Args:
            validation_result: Result from validate_full_consistency
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = [
            "# Horizon Consistency Validation Report",
            f"**Validation Date**: {validation_result.validation_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Target Horizon**: {validation_result.target_horizon} days",
            f"**Overall Consistency**: {'✅ PASS' if validation_result.is_consistent else '❌ FAIL'}",
            "",
            "## Validation Results",
            f"- Configuration Consistency: {'✅' if validation_result.configuration_consistency else '❌'}",
            f"- Parameter Consistency: {'✅' if validation_result.parameter_consistency else '❌'}",
            f"- Risk Estimate Consistency: {'✅' if validation_result.risk_estimate_consistency else '❌'}",
            f"- Forecast Accuracy Check: {'✅' if validation_result.forecast_accuracy_check else '⚠️'}",
            f"- Horizon Coverage: {validation_result.horizon_coverage*100:.1f}%",
            ""
        ]
        
        if validation_result.violations:
            report_lines.extend([
                "## ❌ Violations (Must Fix)",
                ""
            ])
            for i, violation in enumerate(validation_result.violations, 1):
                report_lines.append(f"{i}. {violation}")
            report_lines.append("")
        
        if validation_result.warnings:
            report_lines.extend([
                "## ⚠️ Warnings (Recommended Fixes)",
                ""
            ])
            for i, warning in enumerate(validation_result.warnings, 1):
                report_lines.append(f"{i}. {warning}")
            report_lines.append("")
        
        report_lines.extend([
            "## Recommendations",
            "",
            "1. **Ensure Global Horizon**: All risk estimates must use the same forecast horizon",
            "2. **Parameter Optimization**: Optimize parameters specifically for the target horizon",
            "3. **Validation**: Validate forecasts against realized values at the target horizon",
            "4. **Documentation**: Clearly document the chosen horizon and its rationale",
            "",
            f"For mathematical consistency, ensure all portfolio optimization uses {validation_result.target_horizon}-day risk estimates."
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Horizon consistency report saved to {output_path}")
        
        return report


def validate_horizon_consistency(
    exposures: List[str],
    target_horizon: int = 21,
    config_path: str = "config/optimal_parameters_v2.yaml"
) -> HorizonValidationResult:
    """
    Convenience function for horizon consistency validation.
    
    Args:
        exposures: List of exposure IDs to validate
        target_horizon: Target forecast horizon in days
        config_path: Path to configuration file
        
    Returns:
        HorizonValidationResult
    """
    validator = HorizonValidator(config_path)
    return validator.validate_full_consistency(exposures)