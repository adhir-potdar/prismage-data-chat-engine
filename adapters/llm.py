"""
LLMAdapter — creates LangChain-compatible chat model instances.
Supports OpenAI, Anthropic, and any LangChain BaseChatModel provider.
"""
from __future__ import annotations
from langchain_core.language_models import BaseChatModel


def create_llm(provider: str = "openai", model: str | None = None, **kwargs) -> BaseChatModel:
    """
    Factory for LangChain chat model instances.

    provider: "openai" | "anthropic"
    model:    model name (defaults to a sensible default per provider)

    Environment variables required:
      OpenAI:    OPENAI_API_KEY
      Anthropic: ANTHROPIC_API_KEY
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o", temperature=0, **kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-sonnet-4-6", temperature=0, **kwargs)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'anthropic'.")
