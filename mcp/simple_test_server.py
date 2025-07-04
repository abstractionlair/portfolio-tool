#!/usr/bin/env python3
"""
Simple test MCP server to verify basic functionality
"""

import asyncio
import sys
import json
from typing import Any, Dict, List

# Simple mock MCP server for testing
async def simple_server():
    """Basic MCP server that just responds to stdin"""
    
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            try:
                request = json.loads(line.strip())
                
                if request.get("method") == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {}
                            },
                            "serverInfo": {
                                "name": "simple-ai-coordination",
                                "version": "1.0.0"
                            }
                        }
                    }
                    print(json.dumps(response))
                    sys.stdout.flush()
                
            except json.JSONDecodeError:
                continue
                
        except Exception as e:
            break

if __name__ == "__main__":
    asyncio.run(simple_server())
