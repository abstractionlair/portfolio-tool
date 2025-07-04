#!/usr/bin/env python3
"""
Test the MCP server functionality
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

def test_mcp_server():
    """Test the MCP server with a simple task"""
    
    # Test messages
    initialize_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    list_tools_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    
    invoke_claude_msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "invoke_claude_code",
            "arguments": {
                "task": "List the files in the current directory",
                "context": "This is a test to verify the MCP server works"
            }
        }
    }
    
    # Create input for the server
    input_data = "\n".join([
        json.dumps(initialize_msg),
        json.dumps(list_tools_msg),
        json.dumps(invoke_claude_msg)
    ]) + "\n"
    
    try:
        # Run the server
        result = subprocess.run(
            [sys.executable, "simple_mcp_server.py"],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent
        )
        
        print("Server output:")
        print(result.stdout)
        
        if result.stderr:
            print("Server errors:")
            print(result.stderr)
        
        # Parse responses
        responses = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {line}")
        
        print(f"\nReceived {len(responses)} responses:")
        for i, response in enumerate(responses):
            print(f"Response {i + 1}: {json.dumps(response, indent=2)}")
        
        return len(responses) > 0
        
    except subprocess.TimeoutExpired:
        print("Server test timed out")
        return False
    except Exception as e:
        print(f"Error testing server: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_server()
    if success:
        print("\n✅ MCP server test completed!")
    else:
        print("\n❌ MCP server test failed!")
        sys.exit(1)