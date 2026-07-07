#!/bin/bash
# Script to run Python scripts with the correct PYTHONPATH

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include the project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the Python script with all arguments passed to this script
python "$@"
