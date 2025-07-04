# Claude Desktop Configuration Setup

This guide explains how to configure Claude Desktop to work with the portfolio optimizer project.

## Configuration File Location

The Claude Desktop configuration file is located at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

## Setup Instructions

1. **Copy the template**
   ```bash
   cp config/claude_desktop_config.template.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. **Replace placeholder values**:
   - `YOUR_GITHUB_PAT_HERE`: Your GitHub Personal Access Token
     - Create at: https://github.com/settings/tokens
     - Required scopes: `repo`, `workflow`
   - `YOUR_GITHUB_USERNAME`: Your GitHub username (e.g., "abstractionlair")
   - `PYTHON_VENV_PATH`: Full path to your Python virtual environment
     - Example: `/Users/yourusername/portfolio-tool/venv`
   - `PROJECT_PATH`: Full path to your project directory
     - Example: `/Users/yourusername/portfolio-tool`
   - `YOUR_ANTHROPIC_API_KEY_HERE`: Your Anthropic API key (if using AI coordination)
     - Get from: https://console.anthropic.com/

3. **Restart Claude Desktop** for changes to take effect

## MCP Servers Explained

### GitHub Server
Provides access to GitHub repositories, issues, and pull requests directly from Claude Desktop.

### Filesystem Server
Allows Claude to read and write files in your project directory. The path `~/portfolio-tool` will expand to your home directory automatically.

### AI Coordination Server
Custom server for coordinating between Claude Desktop and Claude Code. This enables invoking Claude Code directly from conversations.

## Security Notes

- **Never commit** your actual config file with real API keys
- Store API keys securely in a password manager
- Rotate tokens periodically
- Use the template file for sharing configuration structure

## Troubleshooting

1. **MCP not connecting**: Ensure Claude Desktop is fully restarted
2. **Permission errors**: Check file permissions on the project directory
3. **Python path issues**: Verify the virtual environment path is correct
4. **API errors**: Confirm API keys are valid and have proper permissions

## Example Working Configuration

Here's an example of what a working configuration might look like (with fake keys):

```json
{
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_1234567890abcdefghijklmnopqrstuv",
                "GITHUB_OWNER": "abstractionlair"
            }
        },
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "~/portfolio-tool"]
        },
        "ai-coordination": {
            "command": "/Users/scottmcguire/portfolio-tool/venv/bin/python",
            "args": ["/Users/scottmcguire/portfolio-tool/mcp/ai_coordination_server.py"],
            "env": {
                "PYTHONPATH": "/Users/scottmcguire/portfolio-tool/mcp",
                "ANTHROPIC_API_KEY": "sk-ant-api03-fake-key-for-example"
            }
        }
    }
}
```
