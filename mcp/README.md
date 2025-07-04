# AI Coordination MCP Server

This directory contains a custom MCP server that enables direct invocation of AI assistants from within Claude Desktop conversations.

## What This Enables

Instead of manually coordinating between AI assistants, Claude Desktop can now:
- **Directly invoke Claude Code** with specific tasks and contexts
- **Call Gemini CLI** for large context analysis or research
- **Execute Codex CLI** for fast Python development
- **Coordinate multiple assistants** in parallel or sequential workflows

## Installation

1. **Install MCP dependencies**:
```bash
pip install mcp anthropic-mcp
```

2. **Make the server executable**:
```bash
chmod +x ai_coordination_server.py
```

3. **Add to Claude Desktop configuration**:
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-pat-here"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y", 
        "@modelcontextprotocol/server-filesystem",
        "/Users/scottmcguire/portfolio-tool"
      ]
    },
    "ai-coordination": {
      "command": "python",
      "args": ["/Users/scottmcguire/portfolio-tool/mcp/ai_coordination_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-openai-key-here"
      }
    }
  }
}
```

4. **Restart Claude Desktop**

## Usage Examples

### Direct Claude Code Invocation
```
Can you invoke Claude Code to implement the portfolio data structures? 
Use the task definition in tasks/next_task.md and focus on 
src/portfolio/portfolio.py and src/portfolio/position.py.
```

### Multi-AI Coordination
```
I need to analyze the current market data architecture and propose 
improvements. Can you coordinate Gemini CLI (for analysis) and 
Claude Code (for implementation) to work on this in parallel?
```

### Sequential Workflow
```
Let's build the optimization engine:
1. Use Gemini CLI to research best practices for portfolio optimization
2. Have Claude Code implement the core algorithm
3. Use Codex CLI to optimize the performance
```

## Available Tools

### `invoke_claude_code`
- **Purpose**: Complex implementation tasks, architecture decisions
- **Parameters**: task, context, files, working_directory
- **Best for**: Deep development work, following specifications

### `invoke_gemini_cli`
- **Purpose**: Large context analysis, research with Google Search
- **Parameters**: task, context, use_search
- **Best for**: Research, large file analysis, market insights

### `invoke_codex_cli`
- **Purpose**: Fast Python development, optimization
- **Parameters**: task, context, model
- **Best for**: Quick iterations, performance tuning

### `coordinate_multi_ai`
- **Purpose**: Complex tasks requiring multiple perspectives
- **Parameters**: task, assistants, strategy (parallel/sequential)
- **Best for**: Comprehensive analysis, competitive approaches

## Benefits

1. **Seamless Integration**: No more manual context switching
2. **Persistent Context**: All assistants work with the same project state
3. **Specialized Strengths**: Each AI focuses on what they do best
4. **Real-time Coordination**: Results flow back to the main conversation
5. **Scalable Complexity**: Handle tasks that require multiple AI capabilities

## Next Steps

1. Install and configure the MCP server
2. Test with a simple Claude Code invocation
3. Try multi-AI coordination on a complex task
4. Extend with additional AI assistants as needed

This transforms Claude Desktop into a true AI development orchestrator!
