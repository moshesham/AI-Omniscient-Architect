#!/usr/bin/env python
"""Test script to verify all packages can be imported correctly."""

import sys
from pathlib import Path

# Add package paths
_r = Path(__file__).parent.parent
for _p in ["core", "agents", "tools", "github", "api", "llm"]:
    _path = _r / "packages" / _p / "src"
    if _path.exists(): sys.path.insert(0, str(_path))


def test_core_package():
    """Test omniscient-core package."""
    print("Testing omniscient-core...")
    
    from omniscient_core import (
        FileAnalysis,
        RepositoryInfo,
        AnalysisConfig,
        AgentFindings,
        ReviewResult,
        BaseAIAgent,
        AgentResponse,
        load_config,
        setup_logging,
        get_logger,
    )
    
    # Test model creation
    file = FileAnalysis(path="test.py", content="print('hello')", language="Python", size=15)
    assert file.path == "test.py"
    
    repo = RepositoryInfo(path="/tmp/test", url=None, branch="main")
    assert repo.branch == "main"
    
    config = AnalysisConfig()
    assert config.max_file_size > 0
    
    print("  ✓ Core models work")
    print("  ✓ omniscient-core OK")


def test_agents_package():
    """Test omniscient-agents package."""
    print("Testing omniscient-agents...")
    
    from omniscient_agents import (
        ArchitectureAgent,
        ReliabilityAgent,
        EfficiencyAgent,
        AlignmentAgent,
        AgentRegistry,
        AnalysisOrchestrator,
        StreamingOrchestrator,
        AnalysisProgress,
        AnalysisResult,
        AnalysisStatus,
    )
    from omniscient_agents.prompts import load_prompt, get_prompt_names
    
    # Test registry
    registry = AgentRegistry()
    assert registry is not None
    
    # Test prompt templates
    templates = get_prompt_names()
    assert len(templates) > 0
    
    # Test loading a prompt
    arch_prompt = load_prompt("architecture")
    assert len(arch_prompt) > 0
    
    # Test orchestrator
    orchestrator = AnalysisOrchestrator(agents=[], max_concurrent=2)
    assert orchestrator.max_concurrent == 2
    
    print("  ✓ Agents can be imported")
    print("  ✓ Prompt templates work")
    print("  ✓ Orchestrator works")
    print("  ✓ omniscient-agents OK")


def test_tools_package():
    """Test omniscient-tools package."""
    print("Testing omniscient-tools...")
    
    from omniscient_tools import (
        AnalysisCache,
        FileScanner,
    )
    
    # Test complexity analyzer (may require lizard)
    try:
        from omniscient_tools import ComplexityAnalyzer
        _ = ComplexityAnalyzer()
        print("  ✓ ComplexityAnalyzer works (lizard installed)")
    except ImportError:
        print("  ⚠ ComplexityAnalyzer skipped (lizard not installed)")
    
    # Test cache
    cache = AnalysisCache()
    assert cache is not None
    print("  ✓ AnalysisCache works")
    
    # Test file scanner
    scanner = FileScanner()
    assert scanner is not None
    print("  ✓ FileScanner works")
    
    print("  ✓ omniscient-tools OK")


def test_github_package():
    """Test omniscient-github package."""
    print("Testing omniscient-github...")
    
    from omniscient_github import (
        GitHubClient,
        GitHubConfig,
        GitHubRepo,
        GitHubFile,
        RepositoryScanner,
        RateLimitHandler,
        parse_github_url,
    )
    
    # Test URL parsing
    owner, repo = parse_github_url("https://github.com/microsoft/vscode")
    assert owner == "microsoft"
    assert repo == "vscode"
    
    # Test config
    config = GitHubConfig()
    assert config.api_url == "https://api.github.com"
    
    print("  ✓ GitHub utilities work")
    print("  ✓ omniscient-github OK")


def test_api_package():
    """Test omniscient-api package."""
    print("Testing omniscient-api...")
    
    # Test config and models (don't need multipart)
    from omniscient_api.config import APIConfig
    from omniscient_api.models import (
        AnalysisRequest,
        AnalysisResponse,
        AnalysisStatus,
        Finding,
    )
    
    # Test config
    config = APIConfig()
    assert config.port == 8000
    print("  ✓ APIConfig works")
    
    # Test models
    assert AnalysisStatus.PENDING.value == "pending"
    print("  ✓ API models work")
    
    # Try to import app (requires python-multipart)
    try:
        from omniscient_api import create_app
        print("  ✓ create_app available")
    except RuntimeError as e:
        if "multipart" in str(e).lower():
            print("  ⚠ create_app skipped (python-multipart not installed)")
        else:
            raise
    
    print("  ✓ omniscient-api OK")


def test_llm_package():
    """Test omniscient-llm package."""
    print("Testing omniscient-llm...")
    
    from omniscient_llm import (
        LLMClient,
        BaseLLMProvider,
        LLMConfig,
        LLMResponse,
        ProviderType,
        Role,
        ChatMessage,
        GenerationRequest,
        ProviderChain,
        ModelManager,
        LLMError,
        ProviderUnavailable,
    )
    
    # Test config
    config = LLMConfig(model="llama3.2:latest", provider=ProviderType.OLLAMA)
    assert config.model == "llama3.2:latest"
    print("  ✓ LLMConfig works")
    
    # Test ChatMessage
    msg = ChatMessage(Role.USER, "Hello")
    assert msg.role == Role.USER
    print("  ✓ ChatMessage works")
    
    # Test GenerationRequest
    req = GenerationRequest(prompt="Test prompt", temperature=0.5)
    assert req.temperature == 0.5
    print("  ✓ GenerationRequest works")
    
    # Test provider availability
    try:
        from omniscient_llm import OllamaProvider
        if OllamaProvider:
            print("  ✓ OllamaProvider available")
    except ImportError:
        print("  - OllamaProvider not available")
    
    # Test ModelManager recommendations
    manager = ModelManager()
    recs = manager.get_recommendations("code")
    assert len(recs) > 0
    print("  ✓ ModelManager works")
    
    print("  ✓ omniscient-llm OK")


def main():
    """Run all package tests."""
    print("=" * 50)
    print("Omniscient Architect Package Verification")
    print("=" * 50)
    print()
    
    try:
        test_core_package()
        print()
        
        test_agents_package()
        print()
        
        test_tools_package()
        print()
        
        test_github_package()
        print()
        
        test_api_package()
        print()
        
        test_llm_package()
        print()
        
        print("=" * 50)
        print("All packages verified successfully! ✓")
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
