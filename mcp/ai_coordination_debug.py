#!/usr/bin/env python3
"""
AI Coordination MCP Server with debugging
"""

import asyncio
import json
import sys
import os
import logging
from pathlib import Path
from typing import Any, Dict, List

# Setup logging to a file for debugging
log_file = Path(__file__).parent / "mcp_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting AI Coordination MCP Server")

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
    logger.info("MCP imports successful")
except ImportError as e:
    logger.error(f"Failed to import MCP: {e}")
    sys.exit(1)

PROJECT_ROOT = Path("/Users/scottmcguire/portfolio-tool")

class AICoordinationServer:
    def __init__(self):
        logger.info("Initializing AICoordinationServer")
        self.server = Server("ai-coordination")
        self.setup_tools()
        
    def setup_tools(self):
        """Register MCP tools"""
        logger.info("Setting up tools")
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            logger.info("Listing tools")
            tools = [
                types.Tool(
                    name="test_connection",
                    description="Test the AI coordination connection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        }
                    }
                ),
                types.Tool(
                    name="invoke_claude_code",
                    description="Invoke Claude Code with a task",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "context": {"type": "string"}
                        },
                        "required": ["task"]
                    }
                )
            ]
            logger.info(f"Returning {len(tools)} tools")
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            logger.info(f"Tool called: {name} with arguments: {arguments}")
            
            try:
                if name == "test_connection":
                    message = arguments.get("message", "Test successful!")
                    return [types.TextContent(
                        type="text",
                        text=f"AI Coordination Server is working! Message: {message}"
                    )]
                
                elif name == "invoke_claude_code":
                    task = arguments["task"]
                    context = arguments.get("context", "")
                    
                    # For now, just simulate the response
                    return [types.TextContent(
                        type="text",
                        text=f"Claude Code invocation simulated:\nTask: {task}\nContext: {context}\n\nThis would normally execute Claude Code."
                    )]
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting server run")
        
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                logger.info("Got stdio streams")
                
                init_options = InitializationOptions(
                    server_name="ai-coordination",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={}
                    )
                )
                
                logger.info(f"Running server with options: {init_options}")
                await self.server.run(read_stream, write_stream, init_options)
                
        except Exception as e:
            logger.error(f"Server run error: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        logger.info("Starting main")
        server = AICoordinationServer()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
