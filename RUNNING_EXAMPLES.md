# Running Examples and Demos

This guide helps you run the portfolio analysis examples and visualization demos.

## Common Import Issues

If you encounter `ImportError` related to relative imports when running examples, this is a common Python packaging issue. Here are the solutions:

### Solution 1: Use Standalone Scripts (Easiest)

Run the standalone versions that handle imports automatically:

```bash
# Visualization demo (no import issues)
python examples/visualization_standalone_demo.py

# Simple visualization demo 
python examples/visualization_simple_demo.py
```

### Solution 2: Use Shell Scripts

Use the provided shell scripts to set up your environment:

```bash
# Linux/Mac - source the script to modify your current shell
source setup_env.sh

# Windows - run the batch file  
setup_env.bat

# Then run any example
python examples/visualization_simple_demo.py
```

### Solution 3: Set PYTHONPATH

Set the PYTHONPATH to include the src directory:

```bash
# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python examples/visualization_demo.py

# Windows
set PYTHONPATH=%PYTHONPATH%;%cd%\src
python examples\visualization_demo.py
```

### Solution 4: Run as Module

Run scripts as modules from the project root:

```bash
# From the portfolio-tool directory
python -m examples.visualization_simple_demo
```

## Available Examples

### Visualization Demos
- `examples/visualization_standalone_demo.py` - Basic visualization test
- `examples/visualization_simple_demo.py` - Comprehensive visualization showcase  
- `examples/visualization_demo.py` - Full demo (requires proper imports)

### Data Analysis Examples  
- `examples/exposure_universe_demo.py` - Exposure universe exploration
- `examples/return_decomposition_demo.py` - Return decomposition analysis
- `examples/cash_rate_demo.py` - Rate series examples

### Optimization Examples
- `examples/optimization_demo.py` - Portfolio optimization showcase

### Analysis Scripts
- `scripts/test_data_availability.py` - Data availability validation
- `scripts/test_return_decomposition.py` - Return decomposition testing

## Jupyter Notebooks

For interactive analysis, use the Jupyter notebooks:

```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks in the notebooks/ directory
# - exposure_universe_analysis.ipynb
# - exposure_universe_decomposition.ipynb
```

## Testing

Run tests to verify everything is working:

```bash
# Standalone visualization tests
python tests/test_visualization_standalone.py

# All tests (requires proper environment)
python -m pytest tests/
```

## Troubleshooting

If you still have import issues:

1. **Check your current directory**: Run commands from the project root (`portfolio-tool/`)

2. **Verify Python version**: Use Python 3.9+ 

3. **Check virtual environment**: Make sure you're in the correct venv:
   ```bash
   which python  # Should point to your venv
   ```

4. **Verify installation**: Check that required packages are installed:
   ```bash
   pip list | grep -E "(pandas|numpy|matplotlib|plotly)"
   ```

5. **Use absolute paths**: If relative imports fail, use standalone scripts that handle paths automatically

## Need Help?

If you continue to have issues:
1. Try the standalone scripts first (`visualization_standalone_demo.py`)
2. Check that you're in the project root directory
3. Verify your Python environment is set up correctly
4. Use the environment setup script: `python setup_environment.py`