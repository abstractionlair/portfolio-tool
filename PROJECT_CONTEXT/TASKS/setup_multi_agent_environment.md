# Task: Multi-Agent Development Environment Setup

**Created**: 2025-01-07 by Desktop Claude
**Priority**: High
**Status**: Ready for implementation

## Objective
Set up OpenAI Codex CLI and Google Gemini CLI to work alongside Claude Code on the portfolio optimizer project. Establish clear workflows and division of responsibilities.

## Background
We have been using Desktop Claude (architect) and Claude Code (implementer) successfully. Now we want to add:
- **OpenAI Codex CLI**: For quick utilities, data integration, refactoring
- **Google Gemini CLI**: For documentation, codebase analysis, bulk operations

## Specific Setup Tasks

### 1. OpenAI Codex CLI Installation
```bash
# Expected installation method (verify current version)
pip install openai-codex-cli
# or
npm install -g @openai/codex-cli
```

**Configuration needed:**
- API key setup
- Project path configuration
- Model selection (gpt-4-turbo recommended)
- Integration with PROJECT_CONTEXT/

### 2. Google Gemini CLI Installation
```bash
# Based on research, likely:
git clone https://github.com/google/gemini-cli
cd gemini-cli
pip install -e .
```

**Configuration needed:**
- Google Cloud authentication
- Repository mapping
- Context window settings

### 3. Project Integration

Create configuration files for each agent:

**`.codex/config.json`**:
```json
{
  "project_root": "/Users/scottmcguire/portfolio-tool",
  "context_path": "PROJECT_CONTEXT/",
  "model": "gpt-4-turbo",
  "test_runner": "pytest",
  "style_guide": "black",
  "auto_test": true
}
```

**`.gemini/config.yaml`**:
```yaml
project:
  path: /Users/scottmcguire/portfolio-tool
  context: PROJECT_CONTEXT/
  
analysis:
  include_tests: true
  documentation_style: google
  
vm:
  auto_clone: true
  preserve_state: true
```

### 4. Workflow Integration

Update PROJECT_CONTEXT files:
- Add agent-specific instructions to IMPLEMENTATION_GUIDE.md
- Create AGENT_COORDINATION.md with clear boundaries
- Update task templates to include agent assignment

### 5. Test Tasks

Create simple test tasks for each agent:

**For Codex**: 
- Task: Implement a utility function to fetch Treasury rates from FRED
- File: `src/data/fred_client.py`

**For Gemini**:
- Task: Generate documentation for the exposure universe system
- Output: `docs/exposure_universe_guide.md`

## Division of Labor

### Complex Architectural Work → Claude Code
- Risk Premium Decomposition Framework
- Multi-file refactoring
- Core system design

### Quick Development → OpenAI Codex
- Single-file utilities
- Data fetching functions
- Test generation
- Small refactoring

### Analysis & Documentation → Google Gemini
- Codebase analysis
- Documentation generation
- Pattern detection
- Bulk updates

## Success Criteria

1. ✅ Both Codex and Gemini CLI tools installed and configured
2. ✅ Each tool can read/write to PROJECT_CONTEXT/
3. ✅ Test task completed successfully by each agent
4. ✅ Clear workflow documented in AGENT_COORDINATION.md
5. ✅ All agents following same coding standards

## Notes for Implementation

- Start with the agent comparison guide at `/PROJECT_CONTEXT/agent_comparison.md`
- Ensure all agents use the same Python environment (pyenv 3.13.5)
- Configure all agents to respect `.gitignore` and not touch `venv/`
- Set up rate limiting to avoid API cost overruns

## Questions/Blockers

(Agents should add any questions here)

## Progress

- [x] GitHub CLI installed (v2.74.2)
- [x] OpenAI Python SDK installed (v1.93.1)
- [x] Google GenerativeAI SDK installed (v0.8.5)
- [x] Helper scripts created (openai_assistant.py, gemini_assistant.py)
- [x] Configuration files created (.ai_config/)
- [ ] API keys configured (requires user action)
- [ ] GitHub authentication (requires user action)
- [ ] Test tasks completed

**Current Status**: 80% - Infrastructure complete, awaiting API keys

**See**: `/PROJECT_CONTEXT/TASKS/ai_environment_setup_complete.md` for detailed setup guide
