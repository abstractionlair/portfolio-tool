"""Check what optimization methods actually exist."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.exposure_universe import ExposureUniverse
from optimization.parameter_optimization import ParameterOptimizer
import inspect

# Load universe
universe = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
optimizer = ParameterOptimizer(universe)

print("ParameterOptimizer methods:")
print("-" * 40)

# Get all methods
for name, method in inspect.getmembers(optimizer, predicate=inspect.ismethod):
    if not name.startswith('_'):
        # Get method signature
        sig = inspect.signature(method)
        print(f"{name}{sig}")

print("\n" + "-" * 40)
print("Attributes:")
for attr in dir(optimizer):
    if not attr.startswith('_') and not callable(getattr(optimizer, attr)):
        print(f"- {attr}: {type(getattr(optimizer, attr))}")