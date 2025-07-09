# Modular Refactoring Plan

**Goal**: Improve code modularity and orthogonality to reduce coupling and enable easier development for both AI and human developers.

## Problem Statement
As the codebase has grown, we've observed:
- Increasing difficulty for AI agents to make progress without breaking other components
- Tight coupling between modules making changes risky
- Test coverage gaps that allowed subtle bugs to slip through
- Context overload making it hard to focus on specific areas

## Strategy

### Phase 1: Test Foundation (CURRENT)
1. **Audit current test coverage** - Identify gaps and blind spots
2. **Write comprehensive unit tests** - Focus on areas with complex logic
3. **Add integration tests** - Catch cross-module interaction bugs
4. **Implement property-based testing** - Find edge cases we haven't thought of

### Phase 2: Identify Refactoring Targets
1. **Analyze coupling metrics** - Which modules are most tightly coupled?
2. **Map dependencies** - Understand current architecture
3. **Prioritize by impact** - Which areas would benefit most from modularity?

### Phase 3: Systematic Refactoring (One Module at a Time)
1. **Design abstract interfaces** - Define clean contracts
2. **Implement dependency injection** - Remove hard dependencies
3. **Extract focused modules** - Single responsibility principle
4. **Maintain test coverage** - Tests as safety net during refactoring

## Initial Target Areas

### 1. Parameter Validation Framework
**Why**: Complex logic with multiple validation methods, prone to the horizon duplication bugs
**Current Issues**: 
- Mixed validation logic and data processing
- Hard-coded validation methods
- Difficult to test edge cases

**Target Design**:
```python
class ValidationMethod(ABC):
    @abstractmethod
    def validate(self, data: pd.Series, params: Dict) -> ValidationResult: pass

class ParameterValidator:
    def __init__(self, methods: List[ValidationMethod]):
        self.methods = methods
    
    def validate_combination(self, data: pd.Series, combination: Dict) -> ValidationResult:
        method = self.get_method(combination['method'])
        return method.validate(data, combination)
```

### 2. Risk Estimation
**Why**: Core functionality used by multiple modules
**Current Issues**:
- EWMA, GARCH, Historical methods tightly coupled
- Difficult to add new estimation methods
- Parameter optimization mixed with estimation logic

### 3. Optimization Engine
**Why**: Complex with multiple optimizers and constraints
**Current Issues**:
- Multiple optimization methods in single class
- Constraint handling mixed with optimization logic
- Hard to test individual components

## Testing Strategy

### Current Test Issues We've Seen
1. **Horizon duplication bug** - Same error values across different horizons
2. **Phantom correlations** - Test data that accidentally confirmed bugs
3. **Edge case blindness** - Missing validation for boundary conditions
4. **Integration gaps** - Components worked individually but failed together

### Improved Testing Approach
1. **Property-based testing** - Use hypothesis to generate edge cases
2. **Mutation testing** - Verify tests actually catch bugs
3. **Integration test matrix** - Test all component combinations
4. **Regression test suite** - Prevent previously fixed bugs from returning

## Success Metrics
- **Test coverage**: >95% line coverage, >90% branch coverage
- **Coupling metrics**: Reduced cyclomatic complexity
- **Development velocity**: Faster feature development
- **Bug reduction**: Fewer integration issues
- **AI effectiveness**: AI agents can work on modules independently

## Next Steps
1. Complete current test audit
2. Write missing unit tests for parameter validation
3. Implement property-based tests for validation edge cases
4. Begin refactoring parameter validation module