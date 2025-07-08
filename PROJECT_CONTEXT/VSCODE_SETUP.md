# VS Code Setup for Portfolio Optimizer

## Essential Extensions

### Python Development
- **Python** (Microsoft) - Core Python support
- **Pylance** - Fast, type-checking language server
- **Python Docstring Generator** - Matches your Google-style docstrings
- **Black Formatter** - Your project's formatter

### Project Navigation
- **Project Manager** - Quick switching between project contexts
- **Path Intellisense** - Autocompletes filenames
- **GitLens** - See who changed what (useful with multiple agents)

### Multi-Agent Workflow
- **Terminal Tabs** - Run different agents in different terminals
- **Task Runner** - Create tasks for common agent commands
- **File Watcher** - See when agents update files in real-time

## Workspace Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  
  // Multi-agent specific
  "files.watcherExclude": {
    "**/.git/**": true,
    "**/venv/**": true
  },
  
  // Show PROJECT_CONTEXT prominently
  "explorer.sortOrder": "mixed",
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  },
  
  // Terminal setup for agents
  "terminal.integrated.profiles.osx": {
    "Claude Code": {
      "path": "zsh",
      "args": ["-c", "cd ${workspaceFolder} && claude-code"]
    },
    "Codex CLI": {
      "path": "zsh", 
      "args": ["-c", "cd ${workspaceFolder} && codex"]
    },
    "Gemini CLI": {
      "path": "zsh",
      "args": ["-c", "cd ${workspaceFolder} && gemini"]
    }
  }
}
```

## Multi-Agent Workflow in VS Code

### 1. Layout Recommendation
- **Left Panel**: File explorer focused on src/
- **Center**: Main code editor
- **Right Panel**: Split - PROJECT_CONTEXT files on top, terminal on bottom
- **Bottom Panel**: Multiple terminals (one per agent)

### 2. Task Automation

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Check Current Task",
      "type": "shell",
      "command": "cat PROJECT_CONTEXT/TASKS/current_task.md",
      "group": "none",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "pytest -v",
      "group": {
        "kind": "test",
        "isDefault": true
      }
    },
    {
      "label": "Update Task Progress",
      "type": "shell",
      "command": "echo 'Progress: ${input:progress}%' >> PROJECT_CONTEXT/TASKS/current_task.md",
      "group": "none"
    }
  ],
  "inputs": [
    {
      "id": "progress",
      "type": "promptString",
      "description": "Progress percentage"
    }
  ]
}
```

### 3. Snippets for Common Patterns

Create `.vscode/python.code-snippets`:

```json
{
  "Task Question": {
    "prefix": "taskq",
    "body": [
      "# QUESTION: ${1:question}",
      "# Context: ${2:context}",
      "# Blocking: ${3:yes/no}"
    ]
  },
  "Progress Update": {
    "prefix": "taskp", 
    "body": [
      "# PROGRESS: ${1:percentage}% complete",
      "# Completed: ${2:what was done}",
      "# Next: ${3:what's next}"
    ]
  }
}
```

## Why This Setup Works Best

1. **Clear Separation**: Your agents handle coding, VS Code is for review/coordination
2. **File Watching**: See real-time updates as agents modify files
3. **Terminal Integration**: Run all agents from within VS Code
4. **Cost Effective**: No additional AI subscription needed
5. **Standard Tool**: Every developer knows VS Code

## Alternative Consideration

If you later find you want MORE AI assistance:
- **GitHub Copilot** ($10/month) integrates with VS Code
- Provides inline suggestions without disrupting your agent workflow
- Much cheaper than Cursor
- Can disable for files being actively edited by agents

This setup maximizes your ability to review and coordinate the multi-agent work while maintaining the "all in on AI" philosophy through your CLI agents rather than IDE features.
