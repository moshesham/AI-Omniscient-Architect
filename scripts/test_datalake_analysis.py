#!/usr/bin/env python
"""Test script to analyze the local Datalake-Guide repository."""

import asyncio
import sys
from pathlib import Path

# Add package paths
packages_dir = Path(__file__).parent / "packages"
for pkg in ["core", "agents", "tools", "github", "api", "llm"]:
    src_path = packages_dir / pkg / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

from omniscient_core import EXCLUDE_DIRS


# Path to the local repository
REPO_PATH = Path(r"C:\Users\Moshe\Analytical_Guide\Datalake-Guide")


def find_python_files(repo_path: Path, max_files: int = 15) -> list[Path]:
    """Find Python files in the repository, excluding venv and cache."""
    files = []
    
    for py_file in repo_path.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in EXCLUDE_DIRS):
            continue
        files.append(py_file)
        if len(files) >= max_files:
            break
    
    return files


async def test_local_repo_analysis():
    """Test analyzing the local Datalake-Guide repository."""
    from omniscient_core import FileAnalysis, RepositoryInfo
    from omniscient_agents.llm_agent import CodeReviewAgent
    from omniscient_llm import OllamaProvider, LLMClient, ModelManager
    
    print("=" * 70)
    print("Omniscient Architect - Local Repository Analysis")
    print("=" * 70)
    print(f"\nRepository: {REPO_PATH}")
    
    if not REPO_PATH.exists():
        print(f"ERROR: Repository path does not exist: {REPO_PATH}")
        return
    
    # Check Ollama status
    print("\n--- Checking LLM Status ---")
    async with ModelManager() as manager:
        if await manager.is_available():
            models = await manager.list_models()
            print(f"✓ Ollama is running with {len(models)} model(s)")
            for model in models:
                print(f"  - {model.name} ({model.size_gb or '?'} GB)")
        else:
            print("✗ Ollama not running!")
            return
    
    # Find Python files
    print("\n--- Scanning Repository ---")
    python_files = find_python_files(REPO_PATH)
    print(f"✓ Found {len(python_files)} Python files")
    
    # Read file contents and create analyses
    print("\n--- Loading Files ---")
    file_analyses = []
    for file_path in python_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            # Get relative path for display
            rel_path = file_path.relative_to(REPO_PATH)
            
            fa = FileAnalysis(
                path=str(rel_path),
                content=content,
                language="Python",
                size=len(content)
            )
            file_analyses.append(fa)
            print(f"  ✓ {rel_path} ({len(content):,} bytes)")
        except Exception as e:
            print(f"  ✗ {file_path.name}: {e}")
    
    if not file_analyses:
        print("No files to analyze!")
        return
    
    # Create repository info
    repo_info = RepositoryInfo(
        path=str(REPO_PATH),
        url="local://Datalake-Guide",
        branch="main",
        name="Datalake-Guide",
        project_objective="Analyze code quality, patterns, and best practices in data lake examples"
    )
    
    print(f"\n--- Running Analysis with LLM ---")
    print(f"Files to analyze: {len(file_analyses)}")
    
    # Create LLM client with smaller model that fits in memory
    provider = OllamaProvider(model="qwen2.5-coder:1.5b")
    llm_client = LLMClient(provider=provider)
    
    async with llm_client:
        print("✓ LLM client initialized")
        
        # Create agent with LLM client
        agent = CodeReviewAgent(
            llm_client=llm_client,
            focus_areas=["code quality", "best practices", "data engineering patterns", "error handling"]
        )
        
        # Run analysis
        print("\nRunning Code Review Agent...")
        print("-" * 40)
        
        try:
            result = await agent.analyze(file_analyses, repo_info)
            
            print("\n" + "=" * 70)
            print("ANALYSIS RESULTS")
            print("=" * 70)
            
            # Display structured results
            if hasattr(result, 'summary') and result.summary:
                print(f"\n📋 Summary:\n{result.summary}")
            
            if hasattr(result, 'issues') and result.issues:
                print(f"\n📊 Issues Found: {len(result.issues)}")
                print("\n🔍 Detailed Issues:\n")
                for i, issue in enumerate(result.issues, 1):
                    # Handle both dict and object formats
                    if isinstance(issue, dict):
                        severity = issue.get('severity', 'N/A')
                        category = issue.get('category', 'General')
                        description = issue.get('description', 'No description')
                        file_path = issue.get('file_path')
                        line_number = issue.get('line_number')
                        suggestion = issue.get('suggestion')
                    else:
                        severity = getattr(issue, 'severity', 'N/A')
                        category = getattr(issue, 'category', 'General')
                        description = getattr(issue, 'description', 'No description')
                        file_path = getattr(issue, 'file_path', None)
                        line_number = getattr(issue, 'line_number', None)
                        suggestion = getattr(issue, 'suggestion', None)
                    
                    severity_emoji = {
                        'high': '🔴',
                        'medium': '🟡', 
                        'low': '🟢',
                        'critical': '🔴'
                    }.get(str(severity).lower(), '⚪')
                    
                    print(f"  {i}. {severity_emoji} [{str(severity).upper()}] {category}")
                    print(f"     {description}")
                    if file_path:
                        print(f"     📁 File: {file_path}")
                    if line_number:
                        print(f"     📍 Line: {line_number}")
                    if suggestion:
                        print(f"     💡 Suggestion: {suggestion}")
                    print()
            
            if hasattr(result, 'recommendations') and result.recommendations:
                print("💡 Recommendations:")
                for rec in result.recommendations:
                    print(f"  • {rec}")
                    
        except Exception as e:
            print(f"\n❌ Analysis Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


def main():
    """Entry point."""
    asyncio.run(test_local_repo_analysis())


if __name__ == "__main__":
    main()
