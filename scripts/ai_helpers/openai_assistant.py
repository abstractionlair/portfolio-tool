#!/usr/bin/env python
"""
OpenAI GPT-4 Assistant for Portfolio Optimizer

This script provides command-line access to OpenAI's GPT-4 for code generation,
refactoring, and utilities within the portfolio optimizer project.

Usage:
    python scripts/ai_helpers/openai_assistant.py --task generate --prompt "Create a FRED data fetcher"
    python scripts/ai_helpers/openai_assistant.py --task refactor --file src/data/market_data.py
    python scripts/ai_helpers/openai_assistant.py --task explain --file src/optimization/risk_premium_estimator.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import openai
except ImportError:
    print("❌ OpenAI package not installed. Run: pip install openai")
    sys.exit(1)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class OpenAIAssistant:
    """OpenAI GPT-4 assistant for portfolio optimizer development."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the OpenAI assistant."""
        self.config_path = config_path or PROJECT_ROOT / '.ai_config' / 'openai_config.json'
        self.config = self._load_config()
        self.client = self._init_client()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "model": "gpt-4-turbo-preview",
                "project_root": str(PROJECT_ROOT),
                "context_files": [
                    "PROJECT_CONTEXT/CURRENT_STATE.md",
                    "PROJECT_CONTEXT/ARCHITECTURE.md",
                    "CLAUDE.md"
                ],
                "coding_standards": {
                    "formatter": "black",
                    "linter": "ruff", 
                    "type_checker": "mypy",
                    "style": "Google docstrings, type hints, clear naming"
                },
                "max_tokens": 4000,
                "temperature": 0.1
            }
            
            # Create config directory and save default
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def _init_client(self) -> openai.OpenAI:
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            print("   Set it with: export OPENAI_API_KEY='sk-...'")
            sys.exit(1)
        
        return openai.OpenAI(api_key=api_key)
    
    def _get_project_context(self) -> str:
        """Get relevant project context for the AI."""
        context_parts = []
        
        for context_file in self.config["context_files"]:
            file_path = PROJECT_ROOT / context_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        context_parts.append(f"## {context_file}\n{content}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not read {context_file}: {e}")
        
        return "\n\n".join(context_parts)
    
    def _read_file_content(self, file_path: str) -> str:
        """Read and return file content."""
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            return f"❌ File not found: {file_path}"
        
        try:
            with open(full_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"❌ Error reading file: {e}"
    
    def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code based on prompt."""
        project_context = self._get_project_context()
        
        system_prompt = f"""You are a senior Python developer working on a portfolio optimization system.

Project Context:
{project_context}

Coding Standards:
- {self.config['coding_standards']['style']}
- Use type hints for all function parameters and returns
- Include comprehensive docstrings
- Follow the existing codebase patterns
- Add appropriate error handling and logging

Additional Context:
{context}

Generate clean, production-ready code that follows the project's existing patterns."""

        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ OpenAI API Error: {e}"
    
    def refactor_code(self, file_path: str, goal: str) -> str:
        """Refactor existing code file."""
        file_content = self._read_file_content(file_path)
        if file_content.startswith("❌"):
            return file_content
        
        prompt = f"""Refactor the following Python code with this goal: {goal}

Current code from {file_path}:
```python
{file_content}
```

Please provide the refactored code with:
1. Improved error handling
2. Better logging
3. Type hints
4. Clear docstrings
5. Adherence to the goal: {goal}

Return only the refactored code, ready to replace the original file."""
        
        return self.generate_code(prompt)
    
    def explain_code(self, file_path: str) -> str:
        """Explain existing code."""
        file_content = self._read_file_content(file_path)
        if file_content.startswith("❌"):
            return file_content
        
        prompt = f"""Analyze and explain this Python code from {file_path}:

```python
{file_content}
```

Provide:
1. High-level overview of what this code does
2. Key classes and functions
3. Dependencies and imports
4. Integration points with the rest of the portfolio optimizer system
5. Potential improvements or concerns
6. How it fits into the overall architecture

Make the explanation clear and detailed for a developer familiar with Python but new to this codebase."""
        
        project_context = self._get_project_context()
        
        system_prompt = f"""You are a senior code reviewer analyzing a portfolio optimization system.

Project Context:
{project_context}

Provide a thorough technical analysis that helps developers understand the code's role in the larger system."""

        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ OpenAI API Error: {e}"


def main():
    """Command-line interface for OpenAI assistant."""
    parser = argparse.ArgumentParser(description="OpenAI Assistant for Portfolio Optimizer")
    parser.add_argument("--task", required=True, choices=["generate", "refactor", "explain"],
                       help="Task type")
    parser.add_argument("--prompt", help="Prompt for code generation")
    parser.add_argument("--file", help="File path for refactor/explain tasks")
    parser.add_argument("--goal", help="Refactoring goal")
    parser.add_argument("--context", default="", help="Additional context")
    parser.add_argument("--output", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    assistant = OpenAIAssistant()
    
    # Execute the requested task
    if args.task == "generate":
        if not args.prompt:
            print("❌ --prompt required for generate task")
            sys.exit(1)
        result = assistant.generate_code(args.prompt, args.context)
        
    elif args.task == "refactor":
        if not args.file or not args.goal:
            print("❌ --file and --goal required for refactor task")
            sys.exit(1)
        result = assistant.refactor_code(args.file, args.goal)
        
    elif args.task == "explain":
        if not args.file:
            print("❌ --file required for explain task")
            sys.exit(1)
        result = assistant.explain_code(args.file)
    
    # Output result
    if args.output:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(result)
        print(f"✅ Output saved to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()