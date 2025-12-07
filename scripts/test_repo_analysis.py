#!/usr/bin/env python
"""Test script to analyze a GitHub repository using the platform."""

import asyncio
import sys
from pathlib import Path

# Add package paths
packages_dir = Path(__file__).parent.parent / "packages"
for pkg in ["core", "agents", "tools", "github", "api", "llm"]:
    src_path = packages_dir / pkg / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

from omniscient_core import CODE_EXTENSIONS


async def test_github_analysis():
    """Test analyzing a GitHub repository."""
    from omniscient_core import FileAnalysis, RepositoryInfo
    from omniscient_github import GitHubClient, parse_github_url
    from omniscient_agents import (
        ArchitectureAgent,
        EfficiencyAgent,
        AnalysisOrchestrator,
    )
    from omniscient_llm import OllamaProvider, LLMClient, ModelManager
    
    repo_url = "https://github.com/moshesham/Economic-Dashboard-API"
    
    print("=" * 60)
    print("Omniscient Architect - Repository Analysis Test")
    print("=" * 60)
    print(f"\nRepository: {repo_url}")
    
    # Parse URL
    owner, repo_name = parse_github_url(repo_url)
    print(f"Owner: {owner}, Repo: {repo_name}")
    
    # Check Ollama status
    print("\n--- Checking LLM Status ---")
    async with ModelManager() as manager:
        if await manager.is_available():
            models = await manager.list_models()
            print(f"✓ Ollama is running with {len(models)} model(s)")
            for model in models:
                print(f"  - {model.name} ({model.size_gb or '?'} GB)")
        else:
            print("✗ Ollama not running - analysis will be limited")
            return
    
    # Create GitHub client and fetch repository
    print("\n--- Fetching Repository ---")
    
    try:
        async with GitHubClient() as client:
            # Get repository info
            repo_data = await client.get_repository(owner, repo_name)
            print(f"✓ Repository: {repo_data.name}")
            print(f"  Description: {repo_data.description or 'N/A'}")
            print(f"  Language: {repo_data.language or 'N/A'}")
            print(f"  Stars: {repo_data.stars}")
            
            # Get repository contents
            print("\n--- Scanning Files ---")
            files = await client.list_files_recursive(owner, repo_name)
            
            # Filter for code files
            code_files = [f for f in files if f.type == 'file' and any(f.path.endswith(ext) for ext in CODE_EXTENSIONS)]
            print(f"✓ Found {len(code_files)} code files")
            
            # Fetch file contents (limit to first 10 for testing)
            files_to_analyze = code_files[:10]
            print(f"\n--- Fetching {len(files_to_analyze)} Files ---")
            
            file_analyses = []
            for file_info in files_to_analyze:
                try:
                    content = await client.get_file_content(owner, repo_name, file_info.path)
                    if content:
                        # Detect language from extension
                        ext = '.' + file_info.path.split('.')[-1] if '.' in file_info.path else ''
                        lang_map = {
                            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                            '.java': 'Java', '.go': 'Go', '.rs': 'Rust',
                            '.rb': 'Ruby', '.php': 'PHP'
                        }
                        language = lang_map.get(ext, 'Unknown')
                        
                        fa = FileAnalysis(
                            path=file_info.path,
                            content=content,
                            language=language,
                            size=len(content)
                        )
                        file_analyses.append(fa)
                        print(f"  ✓ {file_info.path} ({language}, {len(content)} bytes)")
                except Exception as e:
                    print(f"  ✗ {file_info.path}: {e}")
            
            if not file_analyses:
                print("No files to analyze!")
                return
            
            # Create repository info
            repo_info = RepositoryInfo(
                path=repo_url,
                url=repo_url,
                branch=repo_data.default_branch,
                name=repo_name,
                project_objective="Analyze code quality, architecture, and efficiency"
            )
            
            print(f"\n--- Running Analysis with LLM ---")
            print(f"Files: {len(file_analyses)}")
            
            # Create LLM provider and use new LLM agent
            from omniscient_agents.llm_agent import CodeReviewAgent
            from omniscient_llm import OllamaProvider, LLMClient
            
            # Use smaller model that fits in available memory
            provider = OllamaProvider(model="qwen2.5-coder:1.5b")
            llm_client = LLMClient(provider=provider)
            
            async with llm_client:
                # Create agent with LLM client
                agent = CodeReviewAgent(
                    llm_client=llm_client,
                    focus_areas=["architecture", "code quality", "best practices"]
                )
            
                # Run analysis
                print("\nRunning Code Review Agent...")
                try:
                    result = await agent.analyze(file_analyses, repo_info)
                    print(f"\n--- Code Review Results ---")
                    print(f"Summary: {result.summary[:500] if result.summary else 'N/A'}...")
                    print(f"Issues found: {len(result.issues)}")
                    
                    for i, issue in enumerate(result.issues[:5], 1):
                        print(f"\n  {i}. [{issue.severity}] {issue.category}")
                        print(f"     {issue.description[:100]}...")
                        if issue.file_path:
                            print(f"     File: {issue.file_path}")
                            
                except Exception as e:
                    print(f"Analysis error: {e}")
                    import traceback
                    traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Entry point."""
    asyncio.run(test_github_analysis())


if __name__ == "__main__":
    main()
