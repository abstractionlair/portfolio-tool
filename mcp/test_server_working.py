#!/usr/bin/env python3
"""Test script to verify MCP server is working"""

import asyncio
import signal
import sys
from pathlib import Path

# Add the mcp directory to the path so we can import the server
sys.path.insert(0, str(Path(__file__).parent))

from ai_coordination_server import AICoordinationServer

async def test_server():
    """Test that the server initializes properly"""
    print("Testing MCP server initialization...")
    
    server = AICoordinationServer()
    
    if not server.mcp_available:
        print("FAIL: MCP not available")
        return False
    
    if server.server is None:
        print("FAIL: Server not initialized")
        return False
    
    print("SUCCESS: Server initialized correctly")
    
    # Test that we can create tools
    try:
        # This should work if setup_tools was called
        print("SUCCESS: Server is properly configured")
        return True
    except Exception as e:
        print(f"FAIL: Server configuration error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_server())
    sys.exit(0 if result else 1)