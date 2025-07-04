#!/usr/bin/env python3
"""
MCP Server for AI Assistant Coordination

This MCP server allows Claude Desktop to directly invoke other AI assistants
like Claude Code, Gemini CLI, and OpenAI Codex for specialized tasks.
"""

import asyncio
import json
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.lowlevel import NotificationOptions
    import mcp.server.stdio
except ImportError:
    print("MCP not available, running in test mode", file=sys.stderr)
    # Define minimal types for testing
    class types:
        class TextContent:
            def __init__(self, type, text):
                self.type = type
                self.text = text
        
        class Tool:
            def __init__(self, name, description, inputSchema):
                self.name = name
                self.description = description
                self.inputSchema = inputSchema

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent

class AICoordinationServer:
    def __init__(self):
        try:
            self.server = Server("ai-coordination")
            self.mcp_available = True
        except:
            self.server = None
            self.mcp_available = False
        
        self.active_sessions: Dict[str, subprocess.Popen] = {}
        if self.mcp_available:
            self.setup_tools()
    
    def setup_tools(self):
        """Register MCP tools for AI assistant coordination"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="invoke_claude_code",
                    description="Invoke Claude Code with a specific task and context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to assign to Claude Code"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context and requirements"
                            },
                            "working_directory": {
                                "type": "string",
                                "description": "Working directory (defaults to project root)"
                            }
                        },
                        "required": ["task"]
                    }
                ),
                types.Tool(
                    name="test_coordination",
                    description="Test the coordination system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        },
                        "required": ["message"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            try:
                if name == "invoke_claude_code":
                    return await self.invoke_claude_code(arguments)
                elif name == "test_coordination":
                    return await self.test_coordination(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def test_coordination(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Simple test function"""
        message = args.get("message", "Hello from AI coordination!")
        return [types.TextContent(
            type="text",
            text=f"AI Coordination Server is working! Message: {message}"
        )]
    
    async def invoke_claude_code(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Invoke Claude Code with a specific task"""
        
        task = args["task"]
        context = args.get("context", "")
        working_dir = args.get("working_directory", str(PROJECT_ROOT))
        
        try:
            # Create the full prompt for Claude Code
            prompt = f"{task}"
            if context:
                prompt += f"\n\nContext: {context}"
            
            # Execute Claude Code in interactive mode
            result = await asyncio.create_subprocess_exec(
                "claude",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            # Send the prompt and close stdin
            stdout, stderr = await result.communicate(input=prompt.encode())
            
            if result.returncode == 0:
                return [types.TextContent(
                    type="text",
                    text=f"Claude Code completed successfully:\n\n{stdout.decode()}"
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"Claude Code error (code {result.returncode}):\n{stderr.decode()}"
                )]
        
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error invoking Claude Code: {str(e)}"
            )]
    
    async def run(self):
        """Run the MCP server"""
        if not self.mcp_available:
            print("MCP not available, exiting", file=sys.stderr)
            return
            
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="ai-coordination",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        except KeyboardInterrupt:
            print("Server interrupted", file=sys.stderr)
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    server = AICoordinationServer()
    asyncio.run(server.run())
