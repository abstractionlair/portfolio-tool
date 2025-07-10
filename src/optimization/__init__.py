"""Portfolio optimization module."""

from .engine import OptimizationEngine, ObjectiveType, OptimizationConstraints, OptimizationResult
from .estimators import ReturnEstimator, MarketView
from .methods import MeanVarianceOptimizer, RiskParityOptimizer, BlackLittermanOptimizer
from .constraints import ConstraintBuilder
from .trades import Trade, TradeGenerator
from .ewma import EWMAEstimator, EWMAParameters, GARCHEstimator
from .parameter_optimization import ParameterOptimizer, OptimizationConfig
from .exposure_risk_estimator import (
    ExposureRiskEstimator,
    ExposureRiskEstimate, 
    ExposureRiskMatrix,
    build_portfolio_risk_matrix
)
from .risk_premium_estimator import (
    RiskPremiumEstimator,
    RiskPremiumEstimate,
    CombinedRiskEstimates,
    build_portfolio_risk_matrix_from_risk_premia
)
from .comprehensive_parameter_search import (
    ComprehensiveParameterEstimator,
    ComprehensiveParameterSearchEngine,
    ComprehensiveSearchResult,
    analyze_search_results
)

__all__ = [
    "OptimizationEngine",
    "ObjectiveType", 
    "OptimizationConstraints",
    "OptimizationResult",
    "ReturnEstimator",
    "MarketView",
    "MeanVarianceOptimizer",
    "RiskParityOptimizer", 
    "BlackLittermanOptimizer",
    "ConstraintBuilder",
    "Trade",
    "TradeGenerator",
    "EWMAEstimator",
    "EWMAParameters", 
    "GARCHEstimator",
    "ParameterOptimizer",
    "OptimizationConfig",
    "ExposureRiskEstimator",
    "ExposureRiskEstimate",
    "ExposureRiskMatrix",
    "build_portfolio_risk_matrix",
    "RiskPremiumEstimator",
    "RiskPremiumEstimate", 
    "CombinedRiskEstimates",
    "build_portfolio_risk_matrix_from_risk_premia",
    "ComprehensiveParameterEstimator",
    "ComprehensiveParameterSearchEngine",
    "ComprehensiveSearchResult",
    "analyze_search_results"
]