"""Base classes for component-specific optimization framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import yaml
import numpy as np


@dataclass
class ComponentOptimalParameters:
    """Optimal parameters for a specific component."""
    component_type: str  # 'volatility', 'correlation', 'expected_returns'
    exposure_id: str
    method: str
    parameters: Dict[str, Any]
    lookback_days: int
    frequency: str
    score: float
    validation_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate component_type."""
        valid_types = {'volatility', 'correlation', 'expected_returns'}
        if self.component_type not in valid_types:
            raise ValueError(f"component_type must be one of {valid_types}")


@dataclass
class UnifiedOptimalParameters:
    """Complete parameter set for all components."""
    volatility_params: Dict[str, ComponentOptimalParameters]
    correlation_params: ComponentOptimalParameters
    expected_return_params: Dict[str, ComponentOptimalParameters]
    optimization_date: datetime
    validation_period: Tuple[datetime, datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_yaml(self) -> str:
        """Convert to YAML for storage."""
        # Convert to serializable format
        data = {
            'optimization_date': self.optimization_date.isoformat(),
            'validation_period': [
                self.validation_period[0].isoformat(),
                self.validation_period[1].isoformat()
            ],
            'metadata': self.metadata,
            'volatility_params': {},
            'correlation_params': self._serialize_component_params(self.correlation_params),
            'expected_return_params': {}
        }
        
        # Serialize volatility parameters
        for exp_id, params in self.volatility_params.items():
            data['volatility_params'][exp_id] = self._serialize_component_params(params)
        
        # Serialize expected return parameters
        for exp_id, params in self.expected_return_params.items():
            data['expected_return_params'][exp_id] = self._serialize_component_params(params)
        
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'UnifiedOptimalParameters':
        """Load from YAML."""
        data = yaml.safe_load(yaml_str)
        
        # Parse dates
        optimization_date = datetime.fromisoformat(data['optimization_date'])
        validation_period = (
            datetime.fromisoformat(data['validation_period'][0]),
            datetime.fromisoformat(data['validation_period'][1])
        )
        
        # Parse component parameters
        volatility_params = {}
        for exp_id, params_data in data['volatility_params'].items():
            volatility_params[exp_id] = cls._deserialize_component_params(params_data)
        
        correlation_params = cls._deserialize_component_params(data['correlation_params'])
        
        expected_return_params = {}
        for exp_id, params_data in data['expected_return_params'].items():
            expected_return_params[exp_id] = cls._deserialize_component_params(params_data)
        
        return cls(
            volatility_params=volatility_params,
            correlation_params=correlation_params,
            expected_return_params=expected_return_params,
            optimization_date=optimization_date,
            validation_period=validation_period,
            metadata=data.get('metadata', {})
        )
    
    @staticmethod
    def _serialize_component_params(params: ComponentOptimalParameters) -> Dict[str, Any]:
        """Convert ComponentOptimalParameters to serializable format."""
        return {
            'component_type': params.component_type,
            'exposure_id': params.exposure_id,
            'method': params.method,
            'parameters': params.parameters,
            'lookback_days': params.lookback_days,
            'frequency': params.frequency,
            'score': params.score,
            'validation_metrics': params.validation_metrics,
            'metadata': params.metadata
        }
    
    @staticmethod
    def _deserialize_component_params(data: Dict[str, Any]) -> ComponentOptimalParameters:
        """Convert serialized data back to ComponentOptimalParameters."""
        return ComponentOptimalParameters(
            component_type=data['component_type'],
            exposure_id=data['exposure_id'],
            method=data['method'],
            parameters=data['parameters'],
            lookback_days=data['lookback_days'],
            frequency=data['frequency'],
            score=data['score'],
            validation_metrics=data['validation_metrics'],
            metadata=data.get('metadata', {})
        )


class ComponentOptimizer(ABC):
    """Base class for component-specific optimizers."""
    
    def __init__(self, risk_estimator, logger=None):
        """Initialize the component optimizer.
        
        Args:
            risk_estimator: RiskPremiumEstimator instance
            logger: Optional logger instance
        """
        self.risk_estimator = risk_estimator
        self.logger = logger
    
    @abstractmethod
    def optimize_parameters(self, 
                          exposure_ids: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          n_splits: int = 5) -> Dict[str, ComponentOptimalParameters]:
        """Optimize parameters for this component.
        
        Args:
            exposure_ids: List of exposure IDs to optimize
            start_date: Start of optimization period
            end_date: End of optimization period
            n_splits: Number of time series cross-validation splits
            
        Returns:
            Dictionary mapping exposure_id to optimal parameters
        """
        pass
    
    @abstractmethod
    def get_optimization_objectives(self) -> List[str]:
        """Get list of optimization objectives for this component.
        
        Returns:
            List of objective names (e.g., ['mse', 'qlike', 'realized_correlation'])
        """
        pass
    
    @abstractmethod
    def score_parameters(self,
                        exposure_id: str,
                        parameters: Dict[str, Any],
                        train_data: Any,
                        test_data: Any) -> Dict[str, float]:
        """Score a parameter set on test data.
        
        Args:
            exposure_id: Exposure being evaluated
            parameters: Parameter dictionary to score
            train_data: Training dataset
            test_data: Test dataset
            
        Returns:
            Dictionary of score_name -> score_value
        """
        pass
    
    def _log_info(self, message: str):
        """Log info message if logger available."""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"INFO: {message}")
    
    def _log_warning(self, message: str):
        """Log warning message if logger available."""
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")
    
    def _validate_time_period(self, start_date: datetime, end_date: datetime, min_days: int = 365):
        """Validate that time period is sufficient for optimization.
        
        Args:
            start_date: Start date
            end_date: End date
            min_days: Minimum required days
            
        Raises:
            ValueError: If period is too short
        """
        period_days = (end_date - start_date).days
        if period_days < min_days:
            raise ValueError(
                f"Optimization period too short: {period_days} days. "
                f"Minimum required: {min_days} days"
            )
    
    def _create_time_splits(self, 
                          start_date: datetime, 
                          end_date: datetime, 
                          n_splits: int) -> List[Tuple[datetime, datetime]]:
        """Create time series cross-validation splits.
        
        Args:
            start_date: Start of overall period
            end_date: End of overall period
            n_splits: Number of splits
            
        Returns:
            List of (train_end, test_end) tuples
        """
        total_days = (end_date - start_date).days
        split_days = total_days // (n_splits + 1)  # Leave room for initial training period
        
        splits = []
        for i in range(n_splits):
            train_end = start_date + pd.Timedelta(days=(i + 2) * split_days)
            test_end = start_date + pd.Timedelta(days=(i + 3) * split_days)
            
            # Ensure we don't exceed end_date
            if test_end > end_date:
                test_end = end_date
            if train_end >= test_end:
                break
                
            splits.append((train_end, test_end))
        
        return splits
    
    def _extract_parameter_grid(self, constrained: bool = True) -> Dict[str, List[Any]]:
        """Extract parameter grid for this component type.
        
        Args:
            constrained: Whether to use constrained parameter space
            
        Returns:
            Dictionary of parameter_name -> list_of_values
        """
        # Base parameter grid - subclasses should override for component-specific parameters
        if constrained:
            return {
                'lookback_days': [756, 1260],
                'frequency': ['monthly'],
                'method': ['historical', 'ewma']
            }
        else:
            return {
                'lookback_days': [504, 756, 1008, 1260],
                'frequency': ['weekly', 'monthly'],
                'method': ['historical', 'ewma', 'exponential_smoothing']
            }


# Import pandas here to avoid circular imports
try:
    import pandas as pd
except ImportError:
    pd = None