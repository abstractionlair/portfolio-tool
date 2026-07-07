# Portfolio Optimizer Project Context

## Purpose
This directory contains all project context and knowledge that needs to be shared between different AI assistants (Desktop Claude, Claude Code, etc.) and across different devices/sessions.

## For All AI Assistants
1. **Always check CURRENT_STATE.md first** to understand project status
2. **Read TASKS/current_task.md** for the immediate work focus
3. **Update these files** as you make progress or decisions
4. All project knowledge lives in the filesystem - no separate project knowledge bases

## Directory Structure
```
PROJECT_CONTEXT/
├── README.md                 # This file - quick project overview
├── CURRENT_STATE.md          # What's done, what's in progress, what's next
├── ARCHITECTURE.md           # Key technical decisions and rationale
├── IMPLEMENTATION_GUIDE.md   # Detailed guidelines for development
├── TASKS/                    # Task management
│   ├── current_task.md       # What we're working on RIGHT NOW
│   ├── backlog.md            # Prioritized list of future tasks
│   └── completed/            # Archive of completed tasks
└── DECISIONS/                # Architecture Decision Records (ADRs)
    ├── 001-leverage-approach.md
    ├── 002-data-sources.md
    ├── 003-ai-coordination.md
    └── ...
```

## Quick Links
- Vision & Goals: See PROJECT_CHARTER.md in docs/
- Development Setup: docs/DEVELOPMENT.md
- Technical Stack: ARCHITECTURE.md in this directory
- Investment Strategy: DECISIONS/001-leverage-approach.md

## Coordination Protocol
- Desktop Claude: Planning, architecture, task breakdown, reviews
- Claude Code: Implementation, testing, debugging, refactoring
- Both update the same filesystem files for synchronization
- Use the Claude Bridge (.claude-bridge/) for async communication when needed
