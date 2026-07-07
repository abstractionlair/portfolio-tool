#!/usr/bin/env python
"""
Setup script to configure the Python environment for running examples.
This script ensures the src directory is in the Python path.
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Add src directory to Python path."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    src_dir = project_root / 'src'
    
    # Add to Python path if not already there
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        print(f"Added {src_str} to Python path")
    
    # Verify the path exists
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    
    print(f"Environment setup complete!")
    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    
    return project_root, src_dir

def test_imports():
    """Test that key modules can be imported."""
    try:
        from visualization import (
            PerformanceVisualizer,
            AllocationVisualizer, 
            OptimizationVisualizer,
            DecompositionVisualizer
        )
        print("✓ All visualization modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def print_setup_instructions():
    """Print instructions for setting up the environment."""
    project_root = Path(__file__).parent.absolute()
    src_dir = project_root / 'src'
    
    print("\n" + "=" * 60)
    print("ENVIRONMENT SETUP INSTRUCTIONS")
    print("=" * 60)
    print("\nNOTE: This Python script can only test imports within this process.")
    print("To set up your shell environment, use one of these methods:")
    print("\n1. Use shell scripts (RECOMMENDED):")
    print("   Linux/Mac:   source setup_env.sh")
    print("   Windows:     setup_env.bat")
    
    print("\n2. Set PYTHONPATH manually:")
    print(f"   Linux/Mac:   export PYTHONPATH=\"${{PYTHONPATH}}:{src_dir}\"")
    print(f"   Windows:     set PYTHONPATH=%PYTHONPATH%;{src_dir}")
    
    print("\n3. Use standalone scripts (NO SETUP NEEDED):")
    print("   python examples/visualization_standalone_demo.py")
    
    print("\n4. Run as modules:")
    print("   python -m examples.visualization_simple_demo")

def main():
    """Setup environment and test imports."""
    print("Portfolio Tool Environment Setup")
    print("=" * 40)
    
    try:
        setup_environment()
        success = test_imports()
        
        if success:
            print("\n✅ Imports work within this Python process!")
            print_setup_instructions()
        else:
            print("\n❌ Import test failed")
            print_setup_instructions()
            return False
            
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print_setup_instructions()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)