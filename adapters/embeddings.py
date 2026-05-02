"""
EmbeddingsAdapter — creates LangChain-compatible embedding model instances.
Supports OpenAI and Voyage AI providers.
"""
from __future__ import annotations
from langchain_core.embeddings import Embeddings


def create_embeddings(provider: str = "openai", model: str | None = None, **kwargs) -> Embeddings:
    """
    Factory for LangChain embeddings instances.

    provider: "openai" | "voyage"
    model:    model name (defaults to a sensible default per provider)

    Environment variables required:
      OpenAI: OPENAI_API_KEY
      Voyage: VOYAGE_API_KEY
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model or "text-embedding-3-small", **kwargs)  # default: text-embedding-3-small

    elif provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings
        return VoyageAIEmbeddings(model=model or "voyage-3", **kwargs)

    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}. Use 'openai' or 'voyage'.")
