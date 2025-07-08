# Multi-Agent Setup Guide for Portfolio Optimizer

## Agent Roles and Responsibilities

### Desktop Claude (Architect)
- Creates task specifications
- Reviews completed work
- Makes architectural decisions
- Updates project documentation

### Claude Code (Primary Implementer)
- Handles complex, multi-file implementations
- Best for architectural changes
- Use for Risk Premium Decomposition task

### OpenAI Codex CLI (Utility Developer)
- Quick functions and scripts
- Data fetching utilities
- Test generation
- Refactoring small components

### Google Gemini CLI (Analyzer)
- Project-wide analysis
- Documentation generation
- Finding patterns and inconsistencies
- Bulk operations

## Setup Instructions

### 1. OpenAI Codex CLI
```bash
# Install (if not already done)
pip install openai-codex-cli

# Configure for project
cd /Users/scottmcguire/portfolio-tool
codex init --project-path .
codex config set model gpt-4-turbo
codex config set context_path PROJECT_CONTEXT/
```

### 2. Google Gemini CLI
```bash
# Clone and install
git clone https://github.com/google/gemini-cli
cd gemini-cli
pip install -e .

# Configure
gemini auth login
gemini init --repo /Users/scottmcguire/portfolio-tool
```

### 3. Agent Coordination

All agents should:
1. Start by reading `/PROJECT_CONTEXT/CURRENT_STATE.md`
2. Check `/PROJECT_CONTEXT/TASKS/current_task.md`
3. Update task files with progress
4. Follow `/PROJECT_CONTEXT/IMPLEMENTATION_GUIDE.md`

## Current Priority Tasks by Agent

### Claude Code
- Implement Risk Premium Decomposition Framework
- Create `RiskPremiumEstimator` class
- Re-optimize parameters on risk premia

### OpenAI Codex
- Implement FRED data integration for risk-free rates
- Create rate series handlers in `TotalReturnFetcher`
- Add mutual fund data fallbacks

### Google Gemini
- Generate comprehensive documentation for exposure universe
- Analyze codebase for optimization opportunities
- Create visual documentation of architecture

## Communication Protocol

1. **Task Updates**: Edit task files directly
2. **Questions**: Add to task file with "QUESTION:" prefix
3. **Blockers**: Mark in task file with "BLOCKED:" prefix
4. **Completion**: Update percentage in task file

## Example Workflow

```bash
# Desktop Claude creates task
echo "New task created: implement_fred_integration.md" > TASKS/implement_fred_integration.md

# Codex implements
codex run "Implement FRED data integration as specified in TASKS/implement_fred_integration.md"

# Gemini documents
gemini analyze --task "Document the new FRED integration module"

# Claude Code handles complex integration
claude-code /init
claude-code "Integrate FRED data with Risk Premium Decomposition"
```

## Important Files for All Agents

- **Current State**: `/PROJECT_CONTEXT/CURRENT_STATE.md`
- **Active Task**: `/PROJECT_CONTEXT/TASKS/current_task.md`
- **Architecture**: `/PROJECT_CONTEXT/ARCHITECTURE.md`
- **Coding Standards**: `/PROJECT_CONTEXT/IMPLEMENTATION_GUIDE.md`
- **Exposure Config**: `/config/exposure_universe.yaml`

## Success Metrics

- All agents update task files regularly
- Code follows project standards
- Tests pass before marking complete
- Documentation updated with changes
