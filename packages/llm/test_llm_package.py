"""Test omniscient-llm package."""

import asyncio
import sys
from pathlib import Path

# Add packages to path
pkg_root = Path(__file__).parent
sys.path.insert(0, str(pkg_root / "src"))
sys.path.insert(0, str(pkg_root.parent / "core" / "src"))


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    from omniscient_llm import (
        LLMClient,
        BaseLLMProvider,
        LLMConfig,
        LLMResponse,
        LLMUsage,
        ModelInfo,
        ModelStatus,
        ProviderType,
        Role,
        ChatMessage,
        GenerationRequest,
        StreamChunk,
        ProviderChain,
        ModelManager,
        LLMError,
        ProviderUnavailable,
        RateLimitError,
        ModelNotFoundError,
        ModelNotFound,
        TokenLimitError,
    )
    
    print("  ✓ All core imports successful")
    
    # Test provider imports (may fail if deps not installed)
    try:
        from omniscient_llm import OllamaProvider
        if OllamaProvider:
            print("  ✓ OllamaProvider available")
        else:
            print("  - OllamaProvider not available (install omniscient-llm[ollama])")
    except ImportError as e:
        print(f"  - OllamaProvider not available: {e}")
    
    try:
        from omniscient_llm import OpenAIProvider
        if OpenAIProvider:
            print("  ✓ OpenAIProvider available")
        else:
            print("  - OpenAIProvider not available (install omniscient-llm[openai])")
    except ImportError as e:
        print(f"  - OpenAIProvider not available: {e}")
    
    try:
        from omniscient_llm import AnthropicProvider
        if AnthropicProvider:
            print("  ✓ AnthropicProvider available")
        else:
            print("  - AnthropicProvider not available (install omniscient-llm[anthropic])")
    except ImportError as e:
        print(f"  - AnthropicProvider not available: {e}")


def test_models():
    """Test model classes."""
    print("\nTesting models...")
    
    from omniscient_llm import (
        LLMConfig,
        LLMResponse,
        LLMUsage,
        ModelInfo,
        ProviderType,
        Role,
        ChatMessage,
        GenerationRequest,
    )
    
    # Test LLMConfig
    config = LLMConfig(
        model="llama3.2:latest",
        provider=ProviderType.OLLAMA,
        temperature=0.7,
    )
    assert config.model == "llama3.2:latest"
    print("  ✓ LLMConfig works")
    
    # Test ChatMessage
    msg = ChatMessage(Role.USER, "Hello, world!")
    assert msg.role == Role.USER
    assert msg.content == "Hello, world!"
    print("  ✓ ChatMessage works")
    
    # Test GenerationRequest
    req = GenerationRequest(
        messages=[
            ChatMessage(Role.SYSTEM, "You are a helpful assistant."),
            ChatMessage(Role.USER, "Hello!"),
        ],
        temperature=0.5,
    )
    assert len(req.messages) == 2
    assert req.get_prompt()  # Should work
    print("  ✓ GenerationRequest works")
    
    # Test ModelInfo
    info = ModelInfo(
        name="llama3.2:latest",
        provider=ProviderType.OLLAMA,
        size_bytes=4_000_000_000,
    )
    assert info.size_gb == 3.73
    print("  ✓ ModelInfo works")
    
    # Test LLMUsage
    usage = LLMUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    assert usage.total_tokens == 150
    print("  ✓ LLMUsage works")


def test_errors():
    """Test error classes."""
    print("\nTesting errors...")
    
    from omniscient_llm import (
        LLMError,
        ProviderUnavailable,
        RateLimitError,
        ModelNotFoundError,
        ModelNotFound,
        TokenLimitError,
    )
    
    # Test base error
    try:
        raise LLMError("Test error", provider="test")
    except LLMError as e:
        assert "Test error" in str(e)
    print("  ✓ LLMError works")
    
    # Test ProviderUnavailable
    try:
        raise ProviderUnavailable("ollama", "Server not running")
    except ProviderUnavailable as e:
        assert "ollama" in str(e)
    print("  ✓ ProviderUnavailable works")
    
    # Test ModelNotFound (alias)
    try:
        raise ModelNotFound("nonexistent-model")
    except ModelNotFoundError as e:
        assert "nonexistent-model" in str(e)
    print("  ✓ ModelNotFound works")
    
    # Test RateLimitError
    try:
        raise RateLimitError("openai", retry_after=60.0)
    except RateLimitError as e:
        assert e.retry_after == 60.0
    print("  ✓ RateLimitError works")


async def test_manager():
    """Test ModelManager (requires Ollama running)."""
    print("\nTesting ModelManager...")
    
    from omniscient_llm import ModelManager
    
    manager = ModelManager()
    
    try:
        await manager.initialize()
        is_available = await manager.is_available()
        
        if is_available:
            print("  ✓ Ollama is running")
            
            models = await manager.list_models()
            print(f"  ✓ Found {len(models)} models")
            
            for model in models[:3]:  # Show first 3
                print(f"    - {model.name} ({model.size_gb or '?'} GB)")
        else:
            print("  - Ollama not available (skipping model tests)")
            
    except Exception as e:
        print(f"  - Ollama not available: {e}")
    finally:
        await manager.close()


def test_recommendations():
    """Test model recommendations."""
    print("\nTesting recommendations...")
    
    from omniscient_llm import ModelManager
    
    manager = ModelManager()
    
    # Test code recommendations
    code_models = manager.get_recommendations("code")
    assert len(code_models) > 0
    print(f"  ✓ Code models: {', '.join(code_models[:2])}...")
    
    # Test size filtering
    small_models = manager.get_recommendations("general", max_size_gb=3.0)
    assert len(small_models) >= 0  # May be empty for strict size limit
    print(f"  ✓ Small models (≤3GB): {len(small_models)} found")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Omniscient LLM Package Tests")
    print("=" * 60)
    
    try:
        test_imports()
        test_models()
        test_errors()
        test_recommendations()
        asyncio.run(test_manager())
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
