"""
Embedding provider abstraction for NexusCRM RAG pipeline.

WHY ABSTRACTION:
  OpenAI embeddings cost money and require API key.
  HuggingFace sentence-transformers work offline and free.
  Swappable via EMBEDDING_PROVIDER env var without code changes.

EMBEDDING DIMENSIONS:
  - text-embedding-3-small: 1536 dims (OpenAI, most cost-effective)
  - text-embedding-ada-002: 1536 dims (OpenAI, legacy)
  - all-MiniLM-L6-v2: 384 dims (HuggingFace, fast + free)
  - all-mpnet-base-v2: 768 dims (HuggingFace, higher quality)

IMPORTANT: All documents in a vector store must use the SAME embedding model.
  Mixing models produces incorrect similarity scores.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol satisfied by both OpenAI and HuggingFace embedding wrappers."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def get_embedding_model() -> EmbeddingModel:
    """
    Return the configured embedding model.

    Priority:
      1. EMBEDDING_PROVIDER=openai → OpenAIEmbeddings (requires OPENAI_API_KEY)
      2. EMBEDDING_PROVIDER=huggingface → HuggingFaceEmbeddings (local, free)
      3. Fallback → HuggingFaceEmbeddings with all-MiniLM-L6-v2

    WHY lru_cache NOT used here:
      We call get_settings() inside; settings are already cached.
      Model initialization is done once in get_rag_pipeline() which is cached.
    """
    from app.config.settings import get_settings
    settings = get_settings()

    if settings.embedding_provider == "openai" and settings.openai_configured:
        logger.info("Using OpenAI embeddings (model=%s)", settings.embedding_model)
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        except ImportError:
            logger.warning("langchain-openai not installed, falling back to HuggingFace")

    # HuggingFace fallback (works offline, no API key)
    logger.info("Using HuggingFace embeddings (model=%s)", settings.hf_embedding_model)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        # Last resort: minimal mock that returns zero vectors
        # This lets the app start without ML libraries installed
        logger.error(
            "Neither langchain-openai nor langchain-huggingface is installed. "
            "RAG retrieval will not work. Install: pip install langchain-huggingface"
        )
        return _DummyEmbeddings()


class _DummyEmbeddings:
    """
    Zero-vector placeholder — ensures the app doesn't crash if no
    embedding library is installed. RAG queries will return no results
    but won't raise exceptions.
    """
    _dim = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self._dim
