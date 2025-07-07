@echo off
REM Portfolio Tool Environment Setup for Windows
REM Run this script to set up your environment

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Add src directory to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%SCRIPT_DIR%src

echo Portfolio Tool environment configured!
echo PYTHONPATH now includes: %SCRIPT_DIR%src
echo.
echo You can now run:
echo   python examples\visualization_simple_demo.py
echo   python examples\visualization_demo.py
echo   python -m pytest tests\
echo.
echo To make this permanent, add this to your system environment variables:
echo PYTHONPATH=%PYTHONPATH%;%SCRIPT_DIR%src