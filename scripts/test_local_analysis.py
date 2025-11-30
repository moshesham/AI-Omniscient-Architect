#!/usr/bin/env python
"""Test script to analyze files using the LLM agent (no GitHub API needed)."""

import asyncio
import sys
from pathlib import Path

# Add package paths (packages/ is sibling to scripts/)
packages_dir = Path(__file__).parent.parent / "packages"
for pkg in ["core", "agents", "tools", "github", "api", "llm", "rag"]:
    src_path = packages_dir / pkg / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))


# Sample code files for testing (simulating Economic-Dashboard-API structure)
SAMPLE_FILES = {
    "api/main.py": '''
"""FastAPI main application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Economic Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Security concern
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Economic Dashboard API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
''',
    "api/routes/data.py": '''
"""Data routes for economic indicators."""
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/data", tags=["data"])

# Global variable - potential issue
DATA_CACHE = {}

@router.get("/indicators")
async def get_indicators(start_date: str = None, end_date: str = None):
    """Get economic indicators."""
    try:
        # Hardcoded database connection - security issue
        connection_string = "postgresql://user:password123@localhost/db"
        data = fetch_data(connection_string, start_date, end_date)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_data(conn_str, start, end):
    """Fetch data from database."""
    # SQL injection vulnerability
    query = f"SELECT * FROM indicators WHERE date >= '{start}'"
    return {"query": query}
'''
}


async def test_local_analysis():
    """Test analyzing local code samples."""
    from omniscient_core import FileAnalysis, RepositoryInfo
    from omniscient_agents.llm_agent import CodeReviewAgent
    from omniscient_llm import OllamaProvider, LLMClient, ModelManager
    
    print("=" * 60)
    print("Omniscient Architect - Local Analysis Test")
    print("=" * 60)
    
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
    
    # Create file analyses from sample code
    print("\n--- Preparing Files ---")
    file_analyses = []
    for path, content in SAMPLE_FILES.items():
        fa = FileAnalysis(
            path=path,
            content=content.strip(),
            language="Python",
            size=len(content)
        )
        file_analyses.append(fa)
        print(f"  ✓ {path} ({len(content)} bytes)")
    
    # Create repository info
    repo_info = RepositoryInfo(
        path="./Economic-Dashboard-API",
        url="https://github.com/moshesham/Economic-Dashboard-API",
        branch="main",
        name="Economic-Dashboard-API",
        project_objective="Analyze code quality, security, and architecture patterns"
    )
    
    print(f"\n--- Running Analysis with LLM ---")
    print(f"Files: {len(file_analyses)}")
    
    # Create LLM client - using smaller model that fits in memory
    provider = OllamaProvider(model="qwen2.5-coder:1.5b")
    llm_client = LLMClient(provider=provider)
    
    async with llm_client:
        print("✓ LLM client initialized")
        
        # Create agent with LLM client
        agent = CodeReviewAgent(
            llm_client=llm_client,
            focus_areas=["security", "architecture", "code quality", "best practices"]
        )
        
        # Run analysis
        print("\nRunning Code Review Agent (this may take a minute)...")
        print("-" * 40)
        
        try:
            result = await agent.analyze(file_analyses, repo_info)
            
            print("\n" + "=" * 60)
            print("ANALYSIS RESULTS")
            print("=" * 60)
            
            if result.summary:
                print(f"\n📋 Summary:\n{result.summary}")
            
            print(f"\n📊 Issues Found: {len(result.issues)}")
            
            if result.issues:
                print("\n🔍 Detailed Issues:")
                for i, issue in enumerate(result.issues[:10], 1):
                    severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(issue.severity, "⚪")
                    print(f"\n  {i}. {severity_icon} [{issue.severity.upper()}] {issue.category}")
                    print(f"     {issue.description}")
                    if issue.file_path:
                        print(f"     📁 File: {issue.file_path}")
                    if issue.line_number:
                        print(f"     📍 Line: {issue.line_number}")
                    if hasattr(issue, 'suggestion') and issue.suggestion:
                        print(f"     💡 Suggestion: {issue.suggestion}")
            
            if hasattr(result, 'recommendations') and result.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in result.recommendations:
                    print(f"  • {rec}")
                    
        except Exception as e:
            print(f"\n✗ Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


def main():
    """Entry point."""
    asyncio.run(test_local_analysis())


if __name__ == "__main__":
    main()
