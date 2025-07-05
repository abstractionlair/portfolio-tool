# Portfolio Optimizer - Implementation Guide

## Code Style & Standards

### Python Style
- **Formatter**: Black with 100-character line length
- **Linter**: Pylint with project-specific rules
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all public APIs

### Example Code Template
```python
"""Module description."""

from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PortfolioComponent:
    """Class description.
    
    Attributes:
        attribute1: Description
        attribute2: Description
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None) -> None:
        """Initialize the component.
        
        Args:
            param1: Description of param1
            param2: Description of param2
            
        Raises:
            ValueError: If param1 is invalid
        """
        self.param1 = param1
        self.param2 = param2 or 42
    
    def process(self, data: List[float]) -> Dict[str, float]:
        """Process the data.
        
        Args:
            data: List of values to process
            
        Returns:
            Dictionary with results
            
        Example:
            >>> component = PortfolioComponent("test")
            >>> component.process([1.0, 2.0, 3.0])
            {'mean': 2.0, 'sum': 6.0}
        """
        try:
            result = {
                'mean': sum(data) / len(data),
                'sum': sum(data)
            }
            logger.info(f"Processed {len(data)} values")
            return result
        except ZeroDivisionError:
            logger.error("Cannot process empty data")
            raise ValueError("Data cannot be empty")
```

## Implementation Patterns

### 1. Data Source Pattern
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources."""
    
    def fetch_prices(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch price data."""
        ...
    
    def is_available(self) -> bool:
        """Check if source is available."""
        ...

class YFinanceSource:
    """Implementation using yfinance."""
    
    def fetch_prices(self, symbols: List[str]) -> pd.DataFrame:
        # Implementation
        pass
    
    def is_available(self) -> bool:
        # Check connectivity
        return True
```

### 2. Error Handling Pattern
```python
class PortfolioError(Exception):
    """Base exception for portfolio errors."""
    pass

class DataNotFoundError(PortfolioError):
    """Raised when requested data is not available."""
    pass

class OptimizationError(PortfolioError):
    """Raised when optimization fails."""
    pass

# Usage
try:
    data = fetcher.get_data(symbol)
except DataNotFoundError:
    logger.warning(f"No data for {symbol}, using fallback")
    data = get_fallback_data(symbol)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### 3. Configuration Pattern
```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class DataConfig:
    """Configuration for data sources."""
    
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    api_timeout: int = 30
    rate_limit_per_minute: int = 60
    
    @classmethod
    def from_env(cls) -> 'DataConfig':
        """Create config from environment variables."""
        return cls(
            cache_enabled=os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            cache_ttl_hours=int(os.getenv('CACHE_TTL_HOURS', '24')),
            api_timeout=int(os.getenv('API_TIMEOUT', '30')),
            rate_limit_per_minute=int(os.getenv('RATE_LIMIT', '60'))
        )
```

### 4. Testing Pattern
```python
import pytest
from unittest.mock import Mock, patch

class TestPortfolioComponent:
    """Tests for PortfolioComponent."""
    
    @pytest.fixture
    def component(self):
        """Create a test component."""
        return PortfolioComponent("test")
    
    @pytest.fixture
    def mock_data_source(self):
        """Create a mock data source."""
        source = Mock(spec=DataSource)
        source.fetch_prices.return_value = pd.DataFrame({
            'SPY': [400, 401, 402],
            'QQQ': [300, 301, 302]
        })
        return source
    
    def test_process_valid_data(self, component):
        """Test processing with valid data."""
        result = component.process([1.0, 2.0, 3.0])
        assert result['mean'] == 2.0
        assert result['sum'] == 6.0
    
    def test_process_empty_data(self, component):
        """Test processing with empty data."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            component.process([])
    
    @patch('src.data.market_data.yfinance')
    def test_fetch_with_mock(self, mock_yf, mock_data_source):
        """Test fetching with mocked external dependency."""
        # Test implementation
        pass
```

## Specific Implementation Guidelines

### Portfolio Class Design
```python
class Portfolio:
    """Portfolio containing multiple positions."""
    
    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._cash: float = 0.0
        self._last_update: Optional[datetime] = None
    
    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        if position.symbol in self._positions:
            self._positions[position.symbol].add_shares(position.quantity)
        else:
            self._positions[position.symbol] = position
        self._last_update = datetime.now()
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(pos.market_value for pos in self._positions.values()) + self._cash
    
    @property
    def weights(self) -> Dict[str, float]:
        """Calculate position weights."""
        total = self.total_value
        if total == 0:
            return {}
        return {
            symbol: pos.market_value / total 
            for symbol, pos in self._positions.items()
        }
```

### Optimization Pattern
```python
import cvxpy as cp

class PortfolioOptimizer:
    """Optimize portfolio allocations."""
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: List[Constraint]
    ) -> np.ndarray:
        """Find optimal weights."""
        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        
        # Objective: maximize return - risk_penalty * variance
        portfolio_return = expected_returns @ weights
        portfolio_variance = cp.quad_form(weights, covariance_matrix)
        objective = cp.Maximize(portfolio_return - self.risk_penalty * portfolio_variance)
        
        # Standard constraints
        constraints_cp = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= 0,           # Long only (can be modified)
        ]
        
        # Add custom constraints
        for constraint in constraints:
            constraints_cp.append(constraint.to_cvxpy(weights))
        
        # Solve
        problem = cp.Problem(objective, constraints_cp)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            raise OptimizationError(f"Optimization failed: {problem.status}")
        
        return weights.value
```

## Common Pitfalls to Avoid

1. **Don't hardcode data sources** - Always use the abstraction layer
2. **Don't ignore timezone issues** - Use UTC internally, convert for display
3. **Don't forget error handling** - Every external call can fail
4. **Don't skip tests** - Especially for financial calculations
5. **Don't mix business logic and I/O** - Keep them separate for testing

## Performance Tips

1. **Cache aggressively** - Market data doesn't change that often
2. **Batch API calls** - Request multiple symbols at once
3. **Use numpy operations** - Avoid Python loops for numerical work
4. **Profile before optimizing** - Measure, don't guess
5. **Consider async for I/O** - Especially for multiple data sources

## AI Assistant Collaboration Tips

### For Claude Desktop
- Focus on architecture and design decisions
- Review code for best practices
- Help with complex algorithms
- Create comprehensive test cases

### For Claude Code
- Implement based on specifications
- Refactor for clarity and performance
- Debug specific issues
- Add detailed logging

### Handoff Pattern
1. Desktop creates specification in PROJECT_CONTEXT/TASKS/current_task.md
2. Code implements and updates progress in task file
3. Desktop reviews code and adds feedback to task file
4. Iterate until complete
