"""
Setup configuration for portfolio-optimizer package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="portfolio-optimizer",
    version="0.1.0",
    author="Scott McGuire",
    author_email="scottvmcguire@fastmail.fm",
    description="Portfolio optimization tool with support for leverage and alternative investments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abstractionlair/portfolio-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            # We can add command-line scripts here later
        ],
    },
)
