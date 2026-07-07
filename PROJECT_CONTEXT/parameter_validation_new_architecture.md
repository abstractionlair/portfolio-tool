# Parameter Validation - New Modular Architecture

## Design Principles

1. **Single Responsibility**: Each component has one clear purpose
2. **Dependency Injection**: Components receive dependencies, don't create them
3. **Open/Closed**: Easy to extend with new methods/strategies without modification
4. **Interface Segregation**: Clean, focused interfaces
5. **Inversion of Control**: Framework depends on abstractions, not concretions

## New Architecture Design

### Core Interfaces

```python
# === Configuration ===
@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    adaptive_thresholds: Dict[str, int] = field(default_factory=lambda: {
        'walk_forward': 60,
        'reduced_walk_forward': 35, 
        'simple_holdout': 25
    })
    default_forecasting_method: str = 'historical'
    enabled_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'rmse', 'hit_rate', 'bias'
    ])
    validation_timeout_seconds: Optional[float] = None

# === Forecasting Service ===
class ForecastingService:
    """Service for managing and executing forecasting methods."""
    
    def __init__(self, methods: Dict[str, ForecastingMethod]):
        self.methods = methods
    
    def forecast(self, method_name: str, data: pd.Series, 
                parameters: Dict[str, Any]) -> float:
        """Execute a forecasting method."""
        
    def get_available_methods(self) -> List[str]:
        """Get list of available forecasting methods."""
        
    def get_method_defaults(self, method_name: str) -> Dict[str, Any]:
        """Get default parameters for a method."""

# === Validation Strategy Interface ===
class ValidationStrategy(ABC):
    """Abstract base class for validation strategies."""
    
    @abstractmethod
    def validate(self, data: pd.Series, combination: Dict[str, Any],
                forecasting_service: ForecastingService) -> ValidationResult:
        """Execute validation strategy."""
        
    @abstractmethod
    def get_required_data_length(self, horizon: int) -> int:
        """Get minimum data length required for this strategy."""
        
    @abstractmethod  
    def get_strategy_name(self) -> str:
        """Get strategy name for reporting."""

# === Metrics Service ===
class MetricsService:
    """Service for calculating validation metrics."""
    
    def __init__(self, calculators: Dict[str, MetricCalculator]):
        self.calculators = calculators
    
    def calculate_all_metrics(self, forecasts: np.ndarray, 
                            actuals: np.ndarray) -> Dict[str, float]:
        """Calculate all configured metrics."""
        
    def calculate_metric(self, metric_name: str, forecasts: np.ndarray,
                        actuals: np.ndarray) -> float:
        """Calculate a specific metric."""

# === Adaptive Strategy Selector ===
class AdaptiveStrategySelector:
    """Selects appropriate validation strategy based on data characteristics."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def select_strategy(self, data_length: int, 
                       available_strategies: List[ValidationStrategy]) -> ValidationStrategy:
        """Select best strategy for given data length."""
```

### Concrete Implementations

```python
# === Validation Strategies ===
class WalkForwardValidation(ValidationStrategy):
    """Walk-forward validation implementation."""
    
    def __init__(self, min_train_periods: int = 40, max_test_points: int = 15):
        self.min_train_periods = min_train_periods
        self.max_test_points = max_test_points
    
    def validate(self, data: pd.Series, combination: Dict[str, Any],
                forecasting_service: ForecastingService) -> ValidationResult:
        # Implementation here
        
    def get_required_data_length(self, horizon: int) -> int:
        return self.min_train_periods + self.max_test_points + horizon

class SimpleHoldoutValidation(ValidationStrategy):
    """Simple holdout validation implementation."""
    
    def __init__(self, test_ratio: float = 0.2, min_test_size: int = 5):
        self.test_ratio = test_ratio
        self.min_test_size = min_test_size

# === Metric Calculators ===
class MetricCalculator(ABC):
    """Abstract base for metric calculators."""
    
    @abstractmethod
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate metric value."""
        
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""

class MSECalculator(MetricCalculator):
    def calculate(self, forecasts: np.ndarray, actuals: np.ndarray) -> float:
        return np.mean((forecasts - actuals) ** 2)
        
    def get_name(self) -> str:
        return "mse"

# === Main Orchestrator ===
class ValidationOrchestrator:
    """Main orchestrator that coordinates all validation services."""
    
    def __init__(self, 
                 forecasting_service: ForecastingService,
                 validation_strategies: Dict[str, ValidationStrategy],
                 metrics_service: MetricsService,
                 adaptive_selector: AdaptiveStrategySelector,
                 config: ValidationConfig):
        self.forecasting_service = forecasting_service
        self.validation_strategies = validation_strategies  
        self.metrics_service = metrics_service
        self.adaptive_selector = adaptive_selector
        self.config = config
    
    def validate_parameter_combination(self, 
                                     data: pd.Series,
                                     combination: Dict[str, Any],
                                     validation_method: ValidationMethod) -> ValidationResult:
        """Main validation entry point."""
        
        # Select strategy (adaptive or specified)
        if validation_method == ValidationMethod.ADAPTIVE:
            strategy = self.adaptive_selector.select_strategy(
                len(data), list(self.validation_strategies.values())
            )
        else:
            strategy = self.validation_strategies[validation_method.value]
        
        # Execute validation
        return strategy.validate(data, combination, self.forecasting_service)
```

### Factory for Easy Construction

```python
class ValidationFrameworkFactory:
    """Factory for creating configured validation framework."""
    
    @staticmethod
    def create_default_framework(config: Optional[ValidationConfig] = None) -> ValidationOrchestrator:
        """Create framework with default components."""
        
        if config is None:
            config = ValidationConfig()
        
        # Create forecasting methods
        forecasting_methods = {
            'historical': HistoricalMethod(),
            'ewma': EWMAMethod(),
            'exponential_smoothing': ExponentialSmoothingMethod(),
            'robust_mad': RobustMADMethod(),
            'quantile_range': QuantileRangeMethod()
        }
        forecasting_service = ForecastingService(forecasting_methods)
        
        # Create validation strategies
        validation_strategies = {
            'walk_forward': WalkForwardValidation(),
            'reduced_walk_forward': ReducedWalkForwardValidation(),
            'simple_holdout': SimpleHoldoutValidation()
        }
        
        # Create metrics
        metric_calculators = {
            'mse': MSECalculator(),
            'mae': MAECalculator(), 
            'rmse': RMSECalculator(),
            'hit_rate': HitRateCalculator(),
            'bias': BiasCalculator()
        }
        metrics_service = MetricsService(metric_calculators)
        
        # Create adaptive selector
        adaptive_selector = AdaptiveStrategySelector(config)
        
        return ValidationOrchestrator(
            forecasting_service=forecasting_service,
            validation_strategies=validation_strategies,
            metrics_service=metrics_service,
            adaptive_selector=adaptive_selector,
            config=config
        )
    
    @staticmethod
    def create_custom_framework(
        forecasting_methods: Dict[str, ForecastingMethod],
        validation_strategies: Dict[str, ValidationStrategy],
        metric_calculators: Dict[str, MetricCalculator],
        config: ValidationConfig
    ) -> ValidationOrchestrator:
        """Create framework with custom components."""
        # Implementation for custom configuration
```

## Benefits of New Architecture

### 1. **Modularity**
- Each component can be developed, tested, and modified independently
- Clear separation of concerns
- Easy to understand and reason about individual pieces

### 2. **Extensibility**  
- Add new forecasting methods by implementing `ForecastingMethod`
- Add new validation strategies by implementing `ValidationStrategy`
- Add new metrics by implementing `MetricCalculator`
- No need to modify existing code

### 3. **Testability**
- Each component can be unit tested in isolation
- Easy to mock dependencies for testing
- Clear interfaces make test setup straightforward

### 4. **Configurability**
- External configuration for all parameters
- Different configurations for different use cases
- Runtime selection of components

### 5. **AI-Friendly Development**
- Focused components reduce context overload
- Clear interfaces make dependencies explicit
- Single-responsibility classes are easier to understand and modify

## Migration Strategy

### Phase 1: Create New Interfaces
1. Define all abstract interfaces
2. Create configuration classes
3. Implement factory pattern

### Phase 2: Extract Components
1. Extract forecasting methods to service
2. Extract validation strategies  
3. Extract metrics calculations
4. Create orchestrator

### Phase 3: Maintain Compatibility
1. Create adapter for old interface
2. Update tests gradually
3. Deprecate old interface

### Phase 4: Full Migration
1. Update all usage to new interface
2. Remove old implementation
3. Update documentation

This architecture addresses all the coupling issues identified while maintaining backward compatibility during migration.