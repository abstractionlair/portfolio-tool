#!/usr/bin/env python3
"""
Minimal MCP server for testing
"""

import asyncio
import json
import sys
from typing import Any

async def main():
    """Simple MCP server that just handles basic initialization"""
    
    # Read from stdin, write to stdout (MCP protocol)
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    
    try:
        transport, _ = await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )
        
        writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_event_loop())
        
        # Simple message loop
        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                
                # Try to parse JSON-RPC message
                try:
                    message = json.loads(line.decode().strip())
                    
                    if message.get("method") == "initialize":
                        # Respond to initialize
                        response = {
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "result": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {
                                    "tools": {}
                                },
                                "serverInfo": {
                                    "name": "ai-coordination-test",
                                    "version": "1.0.0"
                                }
                            }
                        }
                        
                        writer.write((json.dumps(response) + "\n").encode())
                        await writer.drain()
                        
                    elif message.get("method") == "tools/list":
                        # Return empty tools list
                        response = {
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "result": {
                                "tools": []
                            }
                        }
                        
                        writer.write((json.dumps(response) + "\n").encode())
                        await writer.drain()
                        
                except json.JSONDecodeError:
                    continue
                    
            except Exception as e:
                print(f"Error in message loop: {e}", file=sys.stderr)
                break
                
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
