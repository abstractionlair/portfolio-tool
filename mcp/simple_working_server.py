#!/usr/bin/env python3
"""
Simple working MCP server for testing
"""

import asyncio
import sys
from typing import Dict, List, Any

try:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    MCP_AVAILABLE = True
except ImportError:
    print("MCP not available", file=sys.stderr)
    MCP_AVAILABLE = False

class SimpleServer:
    def __init__(self):
        if not MCP_AVAILABLE:
            return
        
        self.server = Server("ai-coordination")
        self.setup_tools()
    
    def setup_tools(self):
        """Register basic tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="test_tool",
                    description="A simple test tool",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Test message"}
                        },
                        "required": ["message"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            if name == "test_tool":
                message = arguments.get("message", "Hello")
                return [types.TextContent(
                    type="text",
                    text=f"Test successful: {message}"
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]
    
    async def run(self):
        """Run the server"""
        if not MCP_AVAILABLE:
            print("MCP not available", file=sys.stderr)
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
                            notification_options=types.NotificationParams(),
                            experimental_capabilities={}
                        )
                    )
                )
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    server = SimpleServer()
    asyncio.run(server.run())