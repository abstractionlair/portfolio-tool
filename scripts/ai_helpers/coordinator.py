#!/usr/bin/env python
"""
Multi-Agent AI Coordinator for Portfolio Optimizer

This script provides a unified interface to coordinate between different AI assistants
based on the task type. It routes requests to the most appropriate AI tool.

Usage:
    python scripts/ai_helpers/coordinator.py --help
    python scripts/ai_helpers/coordinator.py --auto "Create a FRED data fetcher"
    python scripts/ai_helpers/coordinator.py --route "analyze src/optimization/"
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class AICoordinator:
    """Coordinates between different AI assistants based on task type."""
    
    def __init__(self):
        """Initialize the coordinator."""
        self.openai_available = self._check_openai()
        self.gemini_available = self._check_gemini()
        self.copilot_available = self._check_copilot()
    
    def _check_openai(self) -> bool:
        """Check if OpenAI is available."""
        import os
        return bool(os.getenv('OPENAI_API_KEY'))
    
    def _check_gemini(self) -> bool:
        """Check if Gemini is available via Google Cloud or API key."""
        import os
        import subprocess
        
        # Check Google Cloud authentication first
        try:
            result = subprocess.run(['gcloud', 'auth', 'application-default', 'print-access-token'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
        except:
            pass
        
        # Fall back to API key
        return bool(os.getenv('GOOGLE_API_KEY'))
    
    def _check_copilot(self) -> bool:
        """Check if GitHub Copilot is available."""
        import subprocess
        try:
            result = subprocess.run(['gh', 'copilot', '--help'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def route_task(self, task_description: str) -> Tuple[str, str]:
        """
        Route task to the most appropriate AI assistant.
        
        Returns:
            Tuple of (tool_name, command_to_run)
        """
        task_lower = task_description.lower()
        
        # Analysis and documentation tasks → Gemini
        if any(word in task_lower for word in ['analyze', 'document', 'patterns', 'compare', 'architecture']):
            if self.gemini_available:
                return 'gemini', f"python scripts/ai_helpers/gemini_assistant.py --analyze src/"
            else:
                return 'unavailable', "❌ Gemini requires GOOGLE_API_KEY"
        
        # Code generation and refactoring → OpenAI
        elif any(word in task_lower for word in ['generate', 'create', 'refactor', 'implement', 'write']):
            if self.openai_available:
                return 'openai', f"python scripts/ai_helpers/openai_assistant.py --task generate --prompt \"{task_description}\""
            else:
                return 'unavailable', "❌ OpenAI requires OPENAI_API_KEY"
        
        # Quick suggestions → Copilot
        elif any(word in task_lower for word in ['suggest', 'quick', 'help', 'explain']):
            if self.copilot_available:
                return 'copilot', f"gh copilot suggest \"{task_description}\""
            else:
                return 'unavailable', "❌ GitHub Copilot requires 'gh auth login' and 'gh extension install github/gh-copilot'"
        
        # Default to OpenAI for general tasks
        else:
            if self.openai_available:
                return 'openai', f"python scripts/ai_helpers/openai_assistant.py --task generate --prompt \"{task_description}\""
            elif self.gemini_available:
                return 'gemini', f"python scripts/ai_helpers/gemini_assistant.py --analyze src/"
            else:
                return 'unavailable', "❌ No AI tools configured. Set OPENAI_API_KEY or GOOGLE_API_KEY"
    
    def get_status(self) -> str:
        """Get status of all AI tools."""
        status = "🤖 AI Tools Status:\n"
        status += f"  OpenAI GPT-4:      {'✅ Ready' if self.openai_available else '❌ Need OPENAI_API_KEY'}\n"
        
        # Check which Gemini auth method is available
        if self.gemini_available:
            import subprocess
            try:
                result = subprocess.run(['gcloud', 'auth', 'application-default', 'print-access-token'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    status += f"  Google Gemini:     ✅ Ready (Google Cloud auth)\n"
                else:
                    status += f"  Google Gemini:     ✅ Ready (API key)\n"
            except:
                status += f"  Google Gemini:     ✅ Ready (API key)\n"
        else:
            status += f"  Google Gemini:     ❌ Need 'gcloud auth application-default login' OR GOOGLE_API_KEY\n"
            
        status += f"  GitHub Copilot:    {'✅ Ready' if self.copilot_available else '❌ Need gh auth + extension'}\n"
        
        if not any([self.openai_available, self.gemini_available, self.copilot_available]):
            status += "\n⚠️  No AI tools are configured. See PROJECT_CONTEXT/TASKS/ai_environment_setup_complete.md"
        
        return status
    
    def suggest_tool(self, task_description: str) -> str:
        """Suggest which tool to use for a task."""
        tool, command = self.route_task(task_description)
        
        if tool == 'unavailable':
            return command
        
        suggestion = f"💡 Recommended tool for '{task_description}':\n"
        suggestion += f"   Tool: {tool.title()}\n"
        suggestion += f"   Command: {command}\n"
        
        return suggestion


def main():
    """Command-line interface for AI coordinator."""
    parser = argparse.ArgumentParser(description="Multi-Agent AI Coordinator")
    parser.add_argument("--status", action="store_true", help="Show status of all AI tools")
    parser.add_argument("--route", metavar="TASK", help="Route task to appropriate AI tool")
    parser.add_argument("--auto", metavar="TASK", help="Automatically execute task with best tool")
    
    args = parser.parse_args()
    
    coordinator = AICoordinator()
    
    if args.status:
        print(coordinator.get_status())
    
    elif args.route:
        suggestion = coordinator.suggest_tool(args.route)
        print(suggestion)
    
    elif args.auto:
        tool, command = coordinator.route_task(args.auto)
        
        if tool == 'unavailable':
            print(command)
            sys.exit(1)
        
        print(f"🚀 Executing with {tool.title()}: {args.auto}")
        print(f"Command: {command}")
        print("-" * 50)
        
        # Execute the command
        import subprocess
        result = subprocess.run(command, shell=True)
        sys.exit(result.returncode)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()