#!/usr/bin/env python3
"""
Simple MCP server that actually works
"""

import asyncio
import json
import sys
import subprocess
from pathlib import Path
from typing import Any, Sequence

# Simple JSON-RPC handling without the full MCP library complexity
class SimpleMCPServer:
    def __init__(self):
        self.name = "ai-coordination"
        self.version = "1.0.0"
    
    async def handle_message(self, message: dict) -> dict:
        """Handle a single JSON-RPC message"""
        method = message.get("method")
        id = message.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [
                        {
                            "name": "test_connection",
                            "description": "Test the AI coordination connection",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"}
                                }
                            }
                        },
                        {
                            "name": "invoke_claude_code",
                            "description": "Invoke Claude Code with a task",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "task": {"type": "string"},
                                    "context": {"type": "string"},
                                    "working_directory": {"type": "string"}
                                },
                                "required": ["task"]
                            }
                        }
                    ]
                }
            }
        
        elif method == "tools/call":
            tool_name = message.get("params", {}).get("name")
            arguments = message.get("params", {}).get("arguments", {})
            
            if tool_name == "test_connection":
                return {
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Connection test successful! Message: {arguments.get('message', 'Hello!')}"
                            }
                        ]
                    }
                }
            
            elif tool_name == "invoke_claude_code":
                task = arguments.get("task", "")
                context = arguments.get("context", "")
                working_dir = arguments.get("working_directory", str(Path(__file__).parent.parent))
                
                # Actually invoke Claude Code
                try:
                    # Create the full prompt
                    prompt = f"{task}"
                    if context:
                        prompt += f"\n\nContext: {context}"
                    
                    # Execute Claude Code in interactive mode
                    result = subprocess.run(
                        ["claude"],
                        input=prompt,
                        capture_output=True,
                        text=True,
                        timeout=120,  # 2 minute timeout
                        cwd=working_dir
                    )
                    
                    if result.returncode == 0:
                        output = f"Claude Code completed successfully:\n\n{result.stdout}"
                    else:
                        output = f"Claude Code error (code {result.returncode}):\n{result.stderr}"
                    
                except subprocess.TimeoutExpired:
                    output = "Claude Code execution timed out (2 minutes)"
                except Exception as e:
                    output = f"Error invoking Claude Code: {str(e)}"
                
                return {
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": output
                            }
                        ]
                    }
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        
        else:
            # Unknown method
            return {
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    async def run(self):
        """Main server loop"""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        
        w_transport, w_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(w_transport, w_protocol, reader, asyncio.get_event_loop())
        
        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                
                # Parse JSON-RPC message
                try:
                    message = json.loads(line.decode().strip())
                    response = await self.handle_message(message)
                    
                    # Send response
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    
                except json.JSONDecodeError:
                    # Invalid JSON, ignore
                    continue
                    
            except Exception as e:
                # Log error but keep running
                print(f"Error in message loop: {e}", file=sys.stderr)
                continue

if __name__ == "__main__":
    server = SimpleMCPServer()
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
