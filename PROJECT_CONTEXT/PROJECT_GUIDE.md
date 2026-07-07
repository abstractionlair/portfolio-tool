# Portfolio Optimizer - Project Guide

## Project Overview
A sophisticated portfolio optimization tool for personal use, with a secondary goal of demonstrating professional software engineering skills to prospective employers.

## Multi-Agent Development Environment
This project uses multiple AI assistants working from a shared filesystem:
- **Desktop Claude**: Architecture, planning, code reviews, complex algorithms
- **Claude Code**: Implementation, debugging, testing, refactoring
- **Other assistants**: May include Codex, Gemini, Cursor as needed

**Critical**: All agents use the filesystem as the single source of truth. No separate knowledge bases.

## Filesystem-First Approach
Everything lives in `PROJECT_CONTEXT/`:
```
PROJECT_CONTEXT/
├── PROJECT_GUIDE.md          # This file - main entry point
├── CURRENT_STATE.md          # Current status and progress
├── ARCHITECTURE.md           # Technical decisions
├── IMPLEMENTATION_GUIDE.md   # Coding standards
├── TASKS/
│   ├── current_task.md       # Active work item
│   ├── backlog.md            # Future work
│   └── completed/            # Finished tasks
└── DECISIONS/                # Architecture Decision Records
```

## Project Goals
1. **Primary**: Build a portfolio optimizer that handles:
   - Traditional and leveraged assets
   - Multiple optimization objectives
   - Real-world constraints
   - Tax-aware rebalancing

2. **Secondary**: Showcase skills including:
   - Clean architecture
   - Professional coding practices
   - Comprehensive testing
   - Modern Python development

## Investment Philosophy
- Start with Global Multi-Asset Market Portfolio (Doeswijk et al.)
- Add alternatives: factors, trend following, carry strategies
- Use internal leverage via Return Stacked ETFs and PIMCO funds
- Target volatility similar to global equity portfolio

## Exposure-Based Analysis
- Funds are decomposed into their underlying exposures
- Example: RSSB = 100% Equity + 100% Bonds (200% total notional)
- Portfolio optimization works on exposures, not fund positions
- Return replication validates exposure assumptions
- Exposure mapping table captures how each fund maps to asset classes/strategies

## Getting Started (Any Agent)
1. **Check status**: Read `CURRENT_STATE.md`
2. **Find work**: Read `TASKS/current_task.md`
3. **Understand design**: Review relevant files in `ARCHITECTURE.md` and `DECISIONS/`
4. **Follow standards**: Use `IMPLEMENTATION_GUIDE.md` for coding
5. **Communicate**: Update task progress and questions directly in task files

## Workflow Patterns

### Desktop Claude Workflow
1. Review `CURRENT_STATE.md` and plan next steps
2. Create detailed task specifications in `TASKS/`
3. Answer questions added to task files
4. Review completed work by examining code
5. Update project documentation

### Claude Code Workflow
1. Read task specification
2. Implement following the coding guide
3. Update task progress (e.g., "45% complete")
4. Add questions directly to task file when blocked
5. Mark task complete when done

### Coordination Examples
```
Desktop: Creates "implement_portfolio_class.md" task
Code: Implements, updates task file to "75% complete, need review"
Desktop: Reviews code, adds feedback to task file
Code: Addresses feedback, marks complete in task file
Desktop: Moves task to completed/, updates CURRENT_STATE.md
```

## Key Technical Decisions
- **Python 3.13.5** via pyenv
- **Package structure** with `pip install -e .`
- **Data sources**: yfinance primary, multiple fallbacks
- **Leverage handling**: Dual notional/economic views
- **Testing**: pytest with comprehensive coverage
- **Style**: Black formatter, type hints everywhere

## Current Phase (July 2025)
- ✅ Phase 1: Environment setup and market data fetching (COMPLETE)
- ✅ Phase 2: Raw data provider infrastructure (COMPLETE 2025-07-11)
- ✅ Phase 3: Transformed data providers & optimization engine (COMPLETE 2025-07-11)
- ✅ Phase 4: Portfolio optimization integration (COMPLETE 2025-07-11)
- ✅ Phase 4.5: Portfolio-level two-stage parameter optimization (COMPLETE 2025-07-13)
- 🚀 Phase 5: Web interface development (READY TO START)
- 📋 Phase 6: Advanced analytics and monitoring
- 📋 Phase 7: Production deployment and scaling

## Quick Commands
```bash
# Activate environment
cd /Users/user/portfolio-tool
source venv/bin/activate

# Run tests
pytest

# Run examples
python examples/fetch_market_data.py
python examples/parameter_optimization_complete_demo.py
python examples/portfolio_level_optimization_demo.py

# Start notebook
jupyter lab
```

## Important Files
- Task spec: `TASKS/current_task.md`
- Coding standards: `IMPLEMENTATION_GUIDE.md`
- Architecture: `ARCHITECTURE.md`
- Setup guide: `docs/DEVELOPMENT.md`

## Remember
- **Update as you go**: Don't wait until task completion
- **Ask early**: Use bridge for questions
- **Test everything**: TDD preferred
- **Document decisions**: Update ADRs for design choices
- **Single source of truth**: Everything in filesystem
