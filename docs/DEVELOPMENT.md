# Development Machine Setup Guide

This guide covers setting up a new development machine to work on the portfolio optimizer project.

## Prerequisites
- Git installed
- Python 3.11 or higher
- GitHub account with repository access
- Claude Desktop app (for MCP integration)
- VS Code or preferred editor

## 1. Clone the Repository

```bash
# Create a workspace directory
mkdir -p ~/projects
cd ~/projects

# Clone the repository
git clone https://github.com/YOUR_USERNAME/portfolio-optimizer.git
cd portfolio-optimizer

# Set up git identity (if not already configured)
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## 2. Python Environment Setup

### Option A: Using venv (Recommended for simplicity)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using Poetry (Better dependency management)
```bash
# Install poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate shell
poetry shell
```

## 3. Environment Variables Setup

```bash
# Copy the template
cp config/.env.template .env

# Edit .env with your favorite editor
# Add your API keys:
# - ALPHAVANTAGE_API_KEY
# - FRED_API_KEY
# - Any other service keys
```

## 4. Claude Desktop MCP Configuration

### Install GitHub MCP Server
```bash
# Ensure npm is installed first
npm install -g @modelcontextprotocol/server-github
```

### Configure Claude Desktop
1. Open Claude Desktop settings
2. Locate config file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

3. Add/update configuration:
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-pat-here"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y", 
        "@modelcontextprotocol/server-filesystem",
        "~/projects/portfolio-optimizer"
      ]
    }
  }
}
```

4. Restart Claude Desktop

## 5. VS Code Setup (Recommended)

### Install Recommended Extensions
```bash
# Create .vscode/extensions.json if it doesn't exist
mkdir -p .vscode
```

Add to `.vscode/extensions.json`:
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "ms-toolsai.jupyter",
    "github.copilot",
    "eamodio.gitlens"
  ]
}
```

### Configure VS Code Settings
Add to `.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "python.linting.pylintArgs": ["--max-line-length", "100"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    "venv": true
  }
}
```

## 6. Verify Installation

### Run Tests
```bash
# Run pytest
pytest

# If no tests exist yet, create a simple one
mkdir -p tests
echo "def test_import():
    import src
    assert True" > tests/test_import.py
```

### Test Data Access
```python
# Create a test script
python -c "
import yfinance as yf
import os
from dotenv import load_dotenv

load_dotenv()

# Test yfinance
spy = yf.Ticker('SPY')
print(f'SPY current price: {spy.info.get(\"regularMarketPrice\", \"N/A\")}')

# Test environment variables
print(f'Environment loaded: {\".env\" if os.getenv(\"ALPHAVANTAGE_API_KEY\") else \"No .env found\"}')
"
```

## 7. Claude Code Setup

When using Claude Code on this machine:

1. **First time in a session**:
   ```bash
   cd ~/projects/portfolio-optimizer
   source venv/bin/activate  # or 'poetry shell'
   ```

2. **Share context from Claude Chat**:
   - Copy relevant artifacts or discussions
   - Reference GitHub issues: "Working on issue #5"
   - Share design decisions from chat

## 8. Optional: Jupyter Setup

```bash
# Install Jupyter in the virtual environment
pip install jupyter notebook ipykernel

# Add kernel to Jupyter
python -m ipykernel install --user --name=portfolio-optimizer

# Start Jupyter
jupyter notebook notebooks/
```

## 9. Multiple AI Assistant Setup (Optional but Recommended)

Using multiple AI coding assistants provides complementary strengths and fresh perspectives. All work with the same local files and git repository, ensuring coordination.

### Claude Code (Anthropic)
**Best for**: Complex architecture, thorough code review, following requirements
```bash
# Install via npm
npm install -g @anthropic-ai/claude-code

# Run in your project directory
cd ~/projects/portfolio-optimizer
claude

# Authentication options:
# 1. Anthropic Console (requires billing)
# 2. Claude Pro/Max subscription
# 3. API key from console.anthropic.com
```

### OpenAI Codex CLI
**Best for**: Fast reasoning, multi-threaded tasks, Python expertise
```bash
# Install via npm
npm install -g @openai/codex

# Set up API key
export OPENAI_API_KEY="your-api-key-here"

# Run in your project
codex

# Features:
# - Sandboxed execution (safe by default)
# - Supports o3/o4-mini models
# - Can run multiple tasks in parallel
```

### Google Gemini CLI
**Best for**: Large context windows (1M tokens), multimodal inputs, free tier
```bash
# Install via npm (no additional dependencies needed)
npx @google-gemini/gemini-cli

# Authentication:
# Sign in with personal Google account for free tier
# - 60 requests/minute
# - 1,000 requests/day

# Run in your project
gemini

# Features:
# - Built-in Google Search grounding
# - MCP server support
# - Handles images/PDFs as input
```

### Cursor (IDE with AI)
**Best for**: Inline editing, quick completions, IDE integration
- Download from cursor.com
- Install as you would VS Code
- Use `cursor` command from terminal to open projects
- $20/month subscription
- Note: This is an IDE, not a pure CLI tool

### Practical Multi-Assistant Workflow
```bash
# Example: Complex refactoring task
# 1. Use Claude Code for initial architecture
claude
> "Refactor the data fetcher module to use async/await"
git commit -m "refactor: initial async conversion"

# 2. Use Gemini CLI for optimization review
gemini
> "Review the performance of src/data/fetcher.py and suggest optimizations"
git commit -m "perf: optimize based on Gemini review"

# 3. Use Codex CLI for test coverage
codex
> "Add comprehensive tests for the refactored data fetcher"
git commit -m "test: add fetcher tests"

# 4. Review git log to see all changes
git log --oneline -n 3
```

### Best Practices
- **Clear commits**: Label which assistant made changes
- **Feature branches**: Use different branches for major work
- **Consistent formatting**: Run `black` after any changes
- **Trust but verify**: Review all AI-generated code
- **Leverage strengths**:
  - Claude Code: Architecture and complex logic
  - Codex CLI: Quick iterations and fixes
  - Gemini CLI: Large file analysis and research
  - Cursor: Rapid prototyping in IDE

## Troubleshooting

### Common Issues

1. **MCP not connecting**:
   - Check Claude Desktop is fully restarted
   - Verify GitHub PAT hasn't expired
   - Check config file JSON syntax

2. **Import errors**:
   - Ensure virtual environment is activated
   - Check all requirements are installed
   - Verify PYTHONPATH includes project root

3. **API errors**:
   - Verify .env file exists and has keys
   - Check API rate limits
   - Ensure keys are valid

4. **AI Assistant Issues**:
   - **Claude Code**: Ensure npm v18+ and proper authentication
   - **Codex CLI**: Check OPENAI_API_KEY is set and valid
   - **Gemini CLI**: Login with Google account, check quota
   - **Permission errors**: Never use sudo with npm installs

### Useful Commands
```bash
# Check Python environment
which python
python --version

# List installed packages
pip list

# Check git status
git status
git remote -v

# View environment variables (without showing secrets)
python -c "from dotenv import dotenv_values; print(list(dotenv_values('.env').keys()))"

# Check AI assistant versions
claude --version
codex --version
gemini --version
```

## Next Steps
1. Verify all tests pass
2. Try fetching some market data
3. Check Claude Desktop can access repository via MCP
4. Start developing!

## Notes
- Keep your .env file updated with any new API keys
- Pull latest changes before starting work: `git pull origin develop`
- Create feature branches for new work: `git checkout -b feature/your-feature`

