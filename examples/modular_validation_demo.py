"""
Demonstration of the new modular parameter validation framework.

This script shows how to use the new modular architecture for parameter validation,
including both the new interface and the compatibility adapter for existing code.
"""

import numpy as np
import pandas as pd
from src.validation import (
    ValidationFrameworkFactory,
    ValidationConfig,
    CompatibilityValidationFramework
)
from src.validation.parameter_validation import ValidationMethod

def main():
    """Demonstrate the new modular validation framework."""
    print("=== Modular Parameter Validation Framework Demo ===\n")
    
    # Create sample data
    np.random.seed(42)
    data = pd.Series(np.random.normal(0, 0.02, 100))
    
    combination = {
        'method': 'historical',
        'parameters': {'window': 20},
        'horizon': 21
    }
    
    print("Sample data: 100 periods of risk premium returns")
    print(f"Combination to validate: {combination}\n")
    
    # === Demo 1: Using the new modular framework ===
    print("1. NEW MODULAR FRAMEWORK")
    print("-" * 40)
    
    # Create default framework
    framework = ValidationFrameworkFactory.create_default_framework()
    
    # Show available components
    supported = framework.get_supported_methods()
    print(f"Available forecasting methods: {supported['forecasting_methods']}")
    print(f"Available validation strategies: {supported['validation_strategies']}")
    print(f"Available metrics: {supported['metrics'][:5]}...")  # Show first 5
    
    # Run validation
    result = framework.validate_parameter_combination(data, combination)
    
    print(f"\nValidation Result:")
    print(f"  Success: {result.success}")
    print(f"  Method: {result.validation_method}")
    print(f"  Forecasts: {result.num_forecasts}")
    print(f"  MSE: {result.mse:.6f}")
    print(f"  MAE: {result.mae:.6f}")
    
    # === Demo 2: Custom configuration ===
    print("\n\n2. CUSTOM CONFIGURATION")
    print("-" * 40)
    
    # Create custom configuration
    custom_config = ValidationConfig(
        adaptive_thresholds={
            'walk_forward': 80,
            'reduced_walk_forward': 40,
            'simple_holdout': 20
        },
        default_forecasting_method='ewma',
        enabled_metrics=['mse', 'mae', 'bias']
    )
    
    custom_framework = ValidationFrameworkFactory.create_default_framework(custom_config)
    
    print(f"Custom thresholds: {custom_config.adaptive_thresholds}")
    print(f"Default method: {custom_config.default_forecasting_method}")
    
    # Get strategy recommendation
    recommendation = custom_framework.get_strategy_recommendation(len(data))
    print(f"Recommended strategy for {len(data)} periods: {recommendation}")
    
    # === Demo 3: Testing framework ===
    print("\n\n3. TESTING FRAMEWORK")
    print("-" * 40)
    
    testing_framework = ValidationFrameworkFactory.create_framework_for_testing()
    
    print("Testing framework with lower thresholds for faster validation:")
    print(f"  Thresholds: {testing_framework.config.adaptive_thresholds}")
    print(f"  Max forecasts: {testing_framework.config.max_forecasts_per_validation}")
    
    # === Demo 4: Compatibility adapter (drop-in replacement) ===
    print("\n\n4. COMPATIBILITY ADAPTER")
    print("-" * 40)
    
    # This can replace the old ParameterValidationFramework directly
    compat_framework = CompatibilityValidationFramework()
    
    # Use the exact same interface as the old framework
    compat_result = compat_framework.validate_parameter_combination(
        data, combination, ValidationMethod.ADAPTIVE
    )
    
    print("Using compatibility adapter with original interface:")
    print(f"  Success: {compat_result.success}")
    print(f"  Method: {compat_result.validation_method}")
    print(f"  Forecasts: {compat_result.num_forecasts}")
    
    # Access forecasting methods like the old framework
    methods = compat_framework.forecasting_methods
    print(f"\nAvailable methods: {list(methods.keys())}")
    
    # Use a method directly
    historical_method = methods['historical']
    forecast = historical_method.forecast(data, {'window': 20})
    print(f"Direct forecast from historical method: {forecast:.6f}")
    
    # === Demo 5: Different validation strategies ===
    print("\n\n5. DIFFERENT VALIDATION STRATEGIES")
    print("-" * 40)
    
    strategies = ['simple_holdout', 'reduced_walk_forward', 'walk_forward']
    
    for strategy in strategies:
        try:
            result = framework.validate_parameter_combination(
                data, combination, validation_method=strategy
            )
            print(f"{strategy:20}: {result.num_forecasts:2d} forecasts, MSE: {result.mse:.6f}")
        except Exception as e:
            print(f"{strategy:20}: Failed - {str(e)}")
    
    # === Demo 6: Framework diagnostics ===
    print("\n\n6. FRAMEWORK DIAGNOSTICS")
    print("-" * 40)
    
    diagnostics = framework.validate_configuration()
    print(f"Configuration status: {diagnostics['validation_status']}")
    print(f"Forecasting methods: {diagnostics['forecasting_service']['available_methods']}")
    print(f"Validation strategies: {diagnostics['validation_strategies']['available_strategies']}")
    print(f"Metrics: {diagnostics['metrics_service']['available_metrics']}")
    
    print("\n=== Demo Complete ===")
    print("\nKey benefits of the new modular architecture:")
    print("- Dependency injection enables focused development")
    print("- Clean interfaces make components easy to test")
    print("- Factory pattern simplifies framework construction")
    print("- Compatibility adapter enables smooth migration")
    print("- Modular design reduces context overload for AI development")


if __name__ == "__main__":
    main()