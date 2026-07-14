# Portfolio Optimization Tool

*See [PROVENANCE.md](PROVENANCE.md) for this repo's public-release date and real project timeframe.*

## Overview
This project had dual purposes.
First, I actually did want this tool for my own use in managing my investments.
Second, I wanted to see how far I could push getting AI to write, test, and debug code while I acted mostly as an architect and manager.
This wasn't quite vibe coding. I did review the code and ask for specific additions or changes, but in a "last line of defense" way.

## Learnings
(Things are moving fast. There have already been updates to the models since I did this and functionality is likely better.)

### Claude Code
Claude Code is great. In the end it did almost all the (good) work. I like the CLI interface a lot, as apparently many people do. Must be something about coders' brains.
There were rough spots and I think they were the same complaints that other people have had.
There was placeholder code, like methods with hard-coded returns when they were meant to compute or retrieve something. And this was _without_ acknowledgement in the chat. (Though there were at least comments.)
Code for new features tended to be just "added to the pile" rather than organized.
Relatedly, sometimes Claude would write new code for something when a method already existed somewhere.
Despite those issues, this _felt_ more productive than writing it all myself.
(I am aware of "Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity".)

This is the way.

### Gemini
I had heard someone say at the time that Gemini excelled at big picture stuff like architecture and interfaces, so I decided to let it try to reorganize the "pile of code" I had ended up with.
This looked like it was making progress, so I started letting it continue on its own and I went out.
It made a little progress on a nicer interface, but threw away the real code and replaced it with placeholder code, then got stuck in an infinite loop, and cost me $200.

### OpenAI Codex
I couldn't get it to do anything useful. I suspect this was their rushed out the door, "me too" version. I have heard it is much better now.

### Qwen
I couldn't get it to do anything useful.

### What a second review pass added (July 2026)
I later ran this repo through a multi-model code review. The reviews keep finding exactly what vibe-coding predicts: documentation describing the project the agent intended to build rather than the one it built.

- The docs claim "420+ tests" and "202 contract tests." The review counted 218 test functions, and PROJECT_CONTEXT/CURRENT_STATE.md names a test file (`tests/data/test_return_calculation_fixes.py`) that does not exist.
- The "production" parameter file, `config/optimal_parameters_portfolio_level.yaml`, can't be read with `yaml.safe_load` — it is full of numpy object tags from a plain `yaml.dump`. And the docs' optimal horizon (189 days) disagrees with the YAML's (252).
- In `src/optimization/portfolio_optimizer.py`, the `long_only=True` and `long_only=False` branches are byte-identical, `max_total_notional` is never referenced, and the success path hardcodes `exposures={}` and `total_notional=1.0`.
- The docs point to results at `output/portfolio_level_optimization/portfolio_level_results.json` and `plots/`. Neither exists, and the analyzer that would read them raises FileNotFoundError.
- Three checked-in artifacts disagree on how many exposures worked: 17, 14, or 16, depending on which one you read.

None of this was visible from the demos, which is the point. Demo-driven QC catches math that produces visibly wrong numbers. It does not catch documentation that outruns the code.


## Goals

### Primary Goals
Create a functional tool for managing and optimizing my investment portfolio

### Secondary Goals
- Replace/improve upon portfoliovisualizer.com functionality
- Enable portfolio management across multiple machines/locations
- Support sophisticated strategies including leverage and long-short positions
- Experiment with developing using Claude, Gemini, and Codex in one project

## High-Level Requirements

### Core Functionality
1. **Environment Setup**: Configurable development environment accessible from multiple machines
2. **Portfolio Data Access**: Import and analyze current portfolio holdings
3. **Historical Data**: Access historical portfolio performance data
4. **Investment Universe Definition**: Define and manage the set of available investment vehicles
5. **Market Data Integration**: Access current and historical data for all securities in the investment universe
6. **Benchmark Data**: Retrieve benchmark indices, returns, and compositions
7. **Portfolio Optimization**: Solve for optimal asset allocation given constraints
8. **Implementation Mapping**: Map theoretical allocations to available investment vehicles

### Investment Strategy Context
- **Starting Point**: Global Multi-Asset Market Portfolio (Doeswijk et al.)
- **Enhancements**: 
  - Alternative investments
  - Factor exposure
  - Trend following strategies
  - Multi-asset futures carry
- **Leverage Approach**: Target volatility similar to global equity portfolio using internally leveraged funds
  - Return Stacked ETFs
  - PIMCO StocksPLUS funds
  - PIMCO RAE Plus funds
  - ...

### Technical Considerations
- **Methodological Challenges**: 
  - Handling leverage in allocation calculations
  - Long-short position representation
  - Choice between dollar-weighted vs. volatility-weighted allocations
- **Constraints**:
  - Investment minimums
  - Additional constraints to be determined

## Technology Preferences
- **Primary Language**: Python
- **Development Tools**: Compatible with Claude Code + web interface
- **Portability**: Easy setup on new machines OR remote accessibility
- **Data Sources**: Preference for free APIs, willing to pay for high-value services
- **UI Evolution**: Start simple, evolve to web interface with persistent storage
- **Current Brokerage**: Consolidating at Vanguard

## Success Criteria
1. Successfully manages personal portfolio with sophisticated optimization
2. Demonstrates clean architecture and professional coding practices
3. Includes comprehensive documentation and tests
4. Showcases relevant skills for quantitative development roles
5. Provides insights beyond simple commercial tools

## Project Phases (TBD)
- Phase 1: Environment setup and basic data access
- Phase 2: Portfolio import and analysis
- Phase 3: Optimization engine
- Phase 4: Web interface
- Phase 5: Advanced features and polish

## Open Questions
- Specific portfolio constraints beyond minimums?
- Performance requirements?
- Specific employer audiences to target?
- Integration with brokerage APIs vs. manual import/export?

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/abstractionlair/portfolio-tool.git
   cd portfolio-tool
   ```

2. Set up Python environment:
   ```bash
   pyenv local 3.13.5  # or check .python-version
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies and package:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. Copy and configure environment variables:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

5. Run tests:
   ```bash
   pytest
   ```

6. Try the example:
   ```bash
   python examples/fetch_market_data.py
   ```

For detailed setup instructions, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

