# Task: Realistic Multi-Agent Development Environment Setup

**Created**: 2025-07-07 by Claude Code  
**Priority**: High  
**Status**: Ready for implementation  

## Objective
Set up **actually available** AI coding tools to work alongside Claude Code on the portfolio optimizer project, replacing Desktop Claude's assumptions with real tools.

## Reality Check: What's Actually Available

### ❌ What Desktop Claude Assumed (Not Real)
- "OpenAI Codex CLI" - doesn't exist as standalone tool
- "Google Gemini CLI" - no official CLI yet

### ✅ What We Can Actually Set Up

**1. GitHub Copilot CLI** (Real OpenAI Integration)
```bash
# Install GitHub CLI first
brew install gh
# Install Copilot extension
gh extension install github/gh-copilot
```

**2. OpenAI API via Python SDK** (Direct Access)
```bash
pip install openai
```

**3. Google AI Studio API** (Gemini Access)
```bash
pip install google-generativeai
```

**4. Cursor IDE Integration** (Alternative)
- Download Cursor (VS Code fork with AI)
- Built-in GPT-4 and Claude integration

## Practical Implementation Plan

### Phase 1: Install Available Tools

```bash
# 1. GitHub CLI + Copilot
brew install gh
gh auth login
gh extension install github/gh-copilot

# 2. AI Python SDKs
pip install openai google-generativeai anthropic

# 3. Utility tools
pip install aider-chat  # AI pair programming
```

### Phase 2: Create AI Helper Scripts

**`scripts/ai_helpers/openai_assistant.py`**:
```python
#!/usr/bin/env python
"""OpenAI GPT-4 assistant for code generation"""
import openai
import sys
from pathlib import Path

def generate_code(prompt, file_context=""):
    # Code generation using OpenAI API
    pass

if __name__ == "__main__":
    # CLI interface for OpenAI assistance
    pass
```

**`scripts/ai_helpers/gemini_assistant.py`**:
```python
#!/usr/bin/env python
"""Google Gemini assistant for analysis"""
import google.generativeai as genai

def analyze_codebase(directory):
    # Codebase analysis using Gemini
    pass

def generate_docs(code_files):
    # Documentation generation
    pass
```

### Phase 3: Integration with PROJECT_CONTEXT

**`.ai_config/openai_config.json`**:
```json
{
  "model": "gpt-4-turbo-preview",
  "project_root": "/Users/scottmcguire/portfolio-tool",
  "context_files": [
    "PROJECT_CONTEXT/CURRENT_STATE.md",
    "PROJECT_CONTEXT/ARCHITECTURE.md"
  ],
  "coding_standards": {
    "formatter": "black",
    "linter": "ruff",
    "type_checker": "mypy"
  }
}
```

**`.ai_config/gemini_config.json`**:
```json
{
  "model": "gemini-pro",
  "project_root": "/Users/scottmcguire/portfolio-tool",
  "analysis_scope": [
    "src/",
    "tests/",
    "PROJECT_CONTEXT/"
  ],
  "documentation_style": "google"
}
```

## Realistic Division of Labor

### 🎯 Claude Code (Architecture & Complex Implementation)
- Risk premium frameworks
- Multi-file system design
- Portfolio optimization engine
- Complex algorithmic work

### 🚀 GitHub Copilot (Quick Code Generation)
```bash
# Example usage:
gh copilot suggest "Write a function to fetch Treasury rates from FRED API"
gh copilot explain "def optimize_portfolio_weights(returns, covariance):"
```

### 🤖 OpenAI GPT-4 (Utilities & Refactoring)
```bash
# Via our helper script:
python scripts/ai_helpers/openai_assistant.py \
  --task "refactor" \
  --file "src/data/market_data.py" \
  --goal "add error handling and logging"
```

### 📊 Google Gemini (Analysis & Documentation)
```bash
# Via our helper script:
python scripts/ai_helpers/gemini_assistant.py \
  --analyze src/optimization/ \
  --output docs/optimization_guide.md
```

## Implementation Steps

### Step 1: Install GitHub Copilot CLI
```bash
brew install gh
gh auth login
gh extension install github/gh-copilot
gh copilot --help
```

### Step 2: Install AI SDKs
```bash
pip install openai google-generativeai aider-chat
```

### Step 3: Set up API Keys
```bash
# OpenAI API key (user needs to provide)
export OPENAI_API_KEY="sk-..."

# Google AI Studio API key (user needs to provide) 
export GOOGLE_API_KEY="AI..."
```

### Step 4: Create Helper Scripts
- `scripts/ai_helpers/openai_assistant.py`
- `scripts/ai_helpers/gemini_assistant.py`
- `scripts/ai_helpers/multi_agent_coordinator.py`

### Step 5: Test Integration
Create test tasks:
- **Copilot**: Generate FRED data fetcher function
- **OpenAI**: Refactor existing utility with better error handling
- **Gemini**: Analyze risk premium framework and generate docs

## Success Criteria

1. ✅ GitHub Copilot CLI installed and working
2. ✅ OpenAI and Gemini APIs accessible via Python
3. ✅ Helper scripts created and tested
4. ✅ Each tool completes a test task successfully
5. ✅ PROJECT_CONTEXT integration working
6. ✅ API keys secured and configured

## Realistic Expectations

**What This Setup Provides**:
- AI-assisted code generation via Copilot
- Custom utilities leveraging OpenAI/Gemini APIs
- Integration with our existing Claude Code workflow

**What This Doesn't Provide**:
- Standalone "AI agents" working independently
- Magic automation of complex tasks
- Replacement for careful human oversight

## Questions for User

1. Do you have OpenAI API access/credits?
2. Do you have Google AI Studio access?
3. Would you prefer Cursor IDE integration instead/additionally?
4. Any specific tasks you want the AI helpers to focus on?

## Progress Tracking

- [ ] GitHub CLI installed
- [ ] Copilot extension installed  
- [ ] AI Python SDKs installed
- [ ] API keys configured
- [ ] Helper scripts created
- [ ] Test tasks completed
- [ ] Documentation updated

**Next**: Start with Step 1 (GitHub Copilot) as it's the most straightforward and powerful option.