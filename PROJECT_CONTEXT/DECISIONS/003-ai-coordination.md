# ADR-003: AI Assistant Coordination Strategy

**Status**: Accepted  
**Date**: 2025-01-04  
**Author**: Scott McGuire  

## Context

This project uses multiple AI assistants with different strengths:
- **Claude Desktop**: Architecture, planning, code review
- **Claude Code**: Implementation, debugging, refactoring
- **Other assistants**: Potentially Codex, Gemini, etc.

Challenge: How to coordinate work between assistants effectively without:
- Losing context between sessions
- Duplicating work
- Creating conflicts
- Missing important decisions

## Decision

Implement a filesystem-based coordination system using PROJECT_CONTEXT directory as the single source of truth for all AI assistants.

### PROJECT_CONTEXT Directory Structure
Centralized knowledge base accessible to all assistants:
```
PROJECT_CONTEXT/
├── CURRENT_STATE.md      # Project status
├── ARCHITECTURE.md       # Technical decisions
├── TASKS/
│   ├── current_task.md   # Active work
│   └── backlog.md        # Future work
└── DECISIONS/            # ADRs
```

All coordination happens through direct updates to these files. No separate messaging system is needed.

### Workflow Patterns

1. **Task Assignment**:
   - Desktop creates task specification in current_task.md
   - Code implements and updates progress in task file
   - Desktop reviews code and adds feedback to task file

2. **Question/Answer**:
   - Code adds questions to task file
   - Desktop provides answers in task file
   - Code continues with implementation

3. **Review Cycle**:
   - Code marks task as ready for review
   - Desktop reviews and adds feedback to task file
   - Code addresses feedback and updates task

## Consequences

### Positive
- No external dependencies
- Works across devices/sessions
- Clear audit trail
- Extensible to more assistants
- Git-trackable progress

### Negative
- Requires discipline to update files
- No automated notifications between assistants
- Human needs to relay information between sessions

## Alternatives Considered

1. **External Coordination Service**: Use a web service
   - Rejected: Complexity and dependencies
   - Rejected: Privacy concerns

2. **Git Commits Only**: Use only git for coordination
   - Rejected: Too coarse-grained
   - Rejected: Can't handle questions/reviews

3. **Shared Cloud Document**: Use Google Docs or similar
   - Rejected: Not developer-friendly
   - Rejected: Poor version control

4. **No Coordination**: Work independently
   - Rejected: Leads to conflicts and rework
   - Rejected: Loses architectural coherence

## Implementation Guidelines

### For All Assistants
1. Check CURRENT_STATE.md at session start
2. Read current_task.md for work focus
3. Update progress directly in task files
4. Add questions/notes to task files for the other assistant

### For Desktop Claude
1. Create clear task specifications
2. Review code by examining the actual files
3. Update architecture docs
4. Maintain project vision

### For Claude Code
1. Implement based on specifications
2. Add questions directly to task file when unclear
3. Mark task as ready for review when complete
4. Update task file with implementation details

## Future Enhancements

1. **Automation**: Scripts to check task updates
2. **Templates**: Standard task file formats
3. **Multi-Agent**: Support for specialized assistants
4. **Metrics**: Track task completion times
5. **Tooling**: Helper scripts for task management

## References
- "Distributed Systems for Fun and Profit"
- Unix philosophy of file-based communication
- MCP (Model Context Protocol) design patterns
