#!/bin/bash
# Portfolio Tool Environment Setup
# Source this script to set up your environment:
# source setup_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Add src directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/src"

echo "Portfolio Tool environment configured!"
echo "PYTHONPATH now includes: ${SCRIPT_DIR}/src"
echo ""
echo "You can now run:"
echo "  python examples/visualization_simple_demo.py"
echo "  python examples/visualization_demo.py"
echo "  python -m pytest tests/"
echo ""
echo "To make this permanent, add this to your ~/.bashrc or ~/.zshrc:"
echo "export PYTHONPATH=\"\${PYTHONPATH}:${SCRIPT_DIR}/src\""