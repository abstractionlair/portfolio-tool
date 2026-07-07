"""Component-specific optimization framework.

This module provides specialized optimizers for different components of portfolio optimization:
- Volatility forecasting
- Correlation matrix estimation  
- Expected returns prediction

Each component can be optimized independently with appropriate objectives.
"""

from .base import ComponentOptimizer, ComponentOptimalParameters, UnifiedOptimalParameters
from .volatility import VolatilityOptimizer
from .correlation import CorrelationOptimizer
from .returns import ExpectedReturnOptimizer
from .orchestrator import ComponentOptimizationOrchestrator

__all__ = [
    'ComponentOptimizer',
    'ComponentOptimalParameters', 
    'UnifiedOptimalParameters',
    'VolatilityOptimizer',
    'CorrelationOptimizer',
    'ExpectedReturnOptimizer',
    'ComponentOptimizationOrchestrator'
]