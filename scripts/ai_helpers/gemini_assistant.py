#!/usr/bin/env python
"""
Google Gemini Assistant for Portfolio Optimizer

This script provides command-line access to Google's Gemini AI for codebase analysis,
documentation generation, and pattern detection within the portfolio optimizer project.

Usage:
    python scripts/ai_helpers/gemini_assistant.py --analyze src/optimization/
    python scripts/ai_helpers/gemini_assistant.py --document src/data/exposure_universe.py
    python scripts/ai_helpers/gemini_assistant.py --patterns src/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import google.generativeai as genai
except ImportError:
    print("❌ Google GenerativeAI package not installed. Run: pip install google-generativeai")
    sys.exit(1)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class GeminiAssistant:
    """Google Gemini assistant for portfolio optimizer analysis and documentation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Gemini assistant."""
        self.config_path = config_path or PROJECT_ROOT / '.ai_config' / 'gemini_config.json'
        self.config = self._load_config()
        self.model = self._init_model()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "model": "gemini-pro",
                "project_root": str(PROJECT_ROOT),
                "analysis_scope": [
                    "src/",
                    "tests/",
                    "PROJECT_CONTEXT/"
                ],
                "documentation_style": "google",
                "temperature": 0.1,
                "max_output_tokens": 8192,
                "exclude_patterns": [
                    "*.pyc",
                    "__pycache__",
                    ".git",
                    "venv",
                    "*.egg-info"
                ]
            }
            
            # Create config directory and save default
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def _init_model(self) -> genai.GenerativeModel:
        """Initialize Gemini model with Google Cloud or API key authentication."""
        
        # Try Google Cloud authentication first (preferred)
        try:
            import subprocess
            result = subprocess.run(['gcloud', 'auth', 'application-default', 'print-access-token'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Use Google Cloud authentication
                print("✅ Using Google Cloud authentication for Gemini")
                # For Vertex AI, we'd use different endpoint, but for now use API key fallback
                pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fall back to API key authentication
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ Neither Google Cloud auth nor GOOGLE_API_KEY found")
            print("   Option 1 (Recommended): gcloud auth application-default login")
            print("   Option 2: Get API key from https://makersuite.google.com/app/apikey")
            print("             Set with: export GOOGLE_API_KEY='AI...'")
            sys.exit(1)
        
        print("✅ Using API key authentication for Gemini")
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": self.config["temperature"],
            "max_output_tokens": self.config["max_output_tokens"],
        }
        
        return genai.GenerativeModel(
            model_name=self.config["model"],
            generation_config=generation_config
        )
    
    def _scan_directory(self, directory: str, extensions: List[str] = None) -> List[Path]:
        """Scan directory for relevant files."""
        if extensions is None:
            extensions = ['.py', '.md', '.yaml', '.yml', '.json']
        
        directory_path = PROJECT_ROOT / directory
        if not directory_path.exists():
            return []
        
        files = []
        for ext in extensions:
            files.extend(directory_path.rglob(f'*{ext}'))
        
        # Filter out excluded patterns
        filtered_files = []
        for file_path in files:
            should_exclude = False
            for pattern in self.config["exclude_patterns"]:
                if pattern.replace('*', '') in str(file_path):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _read_file_safely(self, file_path: Path) -> str:
        """Read file content safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Limit content size for API
                if len(content) > 10000:
                    content = content[:10000] + "\n\n... [FILE TRUNCATED] ..."
                return content
        except Exception as e:
            return f"❌ Error reading {file_path}: {e}"
    
    def analyze_codebase(self, directory: str) -> str:
        """Analyze codebase structure and patterns."""
        files = self._scan_directory(directory)
        
        if not files:
            return f"❌ No files found in {directory}"
        
        # Build codebase overview
        file_contents = {}
        structure = []
        
        for file_path in files[:20]:  # Limit to avoid API limits
            rel_path = file_path.relative_to(PROJECT_ROOT)
            structure.append(str(rel_path))
            file_contents[str(rel_path)] = self._read_file_safely(file_path)
        
        prompt = f"""Analyze this portfolio optimizer codebase directory: {directory}

File Structure:
{chr(10).join(structure)}

File Contents:
"""
        
        for file_path, content in file_contents.items():
            prompt += f"\n## {file_path}\n```python\n{content}\n```\n"
        
        prompt += """
Please provide a comprehensive analysis including:

1. **Architecture Overview**: How the code is structured and organized
2. **Key Components**: Main classes, functions, and their responsibilities  
3. **Design Patterns**: What patterns are used (inheritance, composition, etc.)
4. **Dependencies**: How modules depend on each other
5. **Data Flow**: How data moves through the system
6. **Code Quality**: Assessment of documentation, testing, error handling
7. **Potential Issues**: Any concerns or improvement opportunities
8. **Integration Points**: How this code connects to the broader system

Focus on providing actionable insights for developers working on this codebase."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini API Error: {e}"
    
    def generate_documentation(self, file_path: str) -> str:
        """Generate comprehensive documentation for a file."""
        full_path = PROJECT_ROOT / file_path
        
        if not full_path.exists():
            return f"❌ File not found: {file_path}"
        
        content = self._read_file_safely(full_path)
        if content.startswith("❌"):
            return content
        
        prompt = f"""Generate comprehensive documentation for this Python file: {file_path}

```python
{content}
```

Create documentation in {self.config['documentation_style']} style that includes:

1. **Module Overview**: What this module does and its purpose
2. **API Reference**: Detailed documentation for all public classes and functions
3. **Usage Examples**: Code examples showing how to use the main functionality
4. **Dependencies**: What other modules this depends on
5. **Architecture Notes**: How this fits into the larger system
6. **Configuration**: Any configuration options or parameters
7. **Error Handling**: What errors can occur and how they're handled
8. **Performance Considerations**: Any performance notes or optimizations

Format as clear, professional documentation that would help both users and maintainers."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini API Error: {e}"
    
    def detect_patterns(self, directory: str) -> str:
        """Detect code patterns and anti-patterns."""
        files = self._scan_directory(directory, ['.py'])
        
        if not files:
            return f"❌ No Python files found in {directory}"
        
        # Collect code samples
        code_samples = []
        for file_path in files[:15]:  # Limit for API
            rel_path = file_path.relative_to(PROJECT_ROOT)
            content = self._read_file_safely(file_path)
            code_samples.append(f"## {rel_path}\n```python\n{content}\n```")
        
        prompt = f"""Analyze these Python files from the portfolio optimizer for patterns and anti-patterns:

{chr(10).join(code_samples)}

Please identify:

1. **Good Patterns**: Well-implemented design patterns, best practices
2. **Anti-Patterns**: Code smells, problematic patterns to avoid
3. **Consistency Issues**: Inconsistent naming, formatting, or approaches
4. **Architectural Patterns**: How the code is structured (MVC, layers, etc.)
5. **Error Handling Patterns**: How errors are handled consistently
6. **Testing Patterns**: How testing is approached
7. **Documentation Patterns**: How code is documented
8. **Refactoring Opportunities**: Areas that could be improved

Focus on actionable insights that help maintain code quality and consistency."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini API Error: {e}"
    
    def compare_implementations(self, file1: str, file2: str) -> str:
        """Compare two implementations."""
        content1 = self._read_file_safely(PROJECT_ROOT / file1)
        content2 = self._read_file_safely(PROJECT_ROOT / file2)
        
        if content1.startswith("❌") or content2.startswith("❌"):
            return f"❌ Error reading files: {file1} or {file2}"
        
        prompt = f"""Compare these two Python implementations:

## {file1}
```python
{content1}
```

## {file2}
```python
{content2}
```

Provide a detailed comparison including:

1. **Functionality**: What each implementation does differently
2. **Approach**: Different approaches or algorithms used
3. **Code Quality**: Readability, maintainability, documentation
4. **Performance**: Which might be more efficient and why
5. **Error Handling**: How each handles errors
6. **Testing**: Test coverage and approach
7. **Dependencies**: Different external dependencies
8. **Recommendations**: Which approach is better and why

Focus on helping developers choose the best implementation or merge benefits from both."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini API Error: {e}"


def main():
    """Command-line interface for Gemini assistant."""
    parser = argparse.ArgumentParser(description="Gemini Assistant for Portfolio Optimizer")
    parser.add_argument("--analyze", metavar="DIR", help="Analyze codebase directory")
    parser.add_argument("--document", metavar="FILE", help="Generate documentation for file")
    parser.add_argument("--patterns", metavar="DIR", help="Detect patterns in directory")
    parser.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"), 
                       help="Compare two implementations")
    parser.add_argument("--output", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.document, args.patterns, args.compare]):
        parser.print_help()
        sys.exit(1)
    
    assistant = GeminiAssistant()
    
    # Execute the requested task
    if args.analyze:
        result = assistant.analyze_codebase(args.analyze)
    elif args.document:
        result = assistant.generate_documentation(args.document)
    elif args.patterns:
        result = assistant.detect_patterns(args.patterns)
    elif args.compare:
        result = assistant.compare_implementations(args.compare[0], args.compare[1])
    
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