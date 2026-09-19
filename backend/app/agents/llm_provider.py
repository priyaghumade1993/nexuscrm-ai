"""
LLM Provider abstraction layer.

WHY THIS EXISTS:
  The architecture is designed so the LLM vendor can be swapped without
  rewriting the agent system. The rest of the codebase imports from this
  module, not from langchain_openai directly.

CURRENT PROVIDER: OpenAI (via LangChain)

FUTURE PROVIDERS (not implemented yet — architecture is ready):
  - Google Gemini (langchain_google_genai)
  - Anthropic Claude (langchain_anthropic)
  - Ollama (langchain_community.llms.ollama) for local/private deployment

HOW TO ADD A PROVIDER:
  1. Implement a new subclass of BaseLLMProvider.
  2. Register it in get_llm_provider().
  3. Add the LANGCHAIN_LLM_PROVIDER env var option.
  No changes needed in agents, graph, or tools.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.language_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Interface every LLM provider must implement."""

    @abstractmethod
    def get_chat_model(
        self,
        temperature: float = 0.1,
        streaming: bool = False,
    ) -> BaseChatModel:
        """Return a LangChain-compatible chat model."""
        ...

    @abstractmethod
    def get_embedding_model(self):
        """Return a LangChain-compatible embeddings object."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        ...


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider using LangChain.

    WHY LOW TEMPERATURE (0.1):
      For CRM operations, we want deterministic, predictable outputs.
      High temperature causes hallucinations and inconsistent tool selection.
      Temperature is higher (0.7) only for email drafting.

    WHY gpt-4o-mini as default:
      Good balance of cost vs capability for tool calling.
      gpt-4o is available for complex multi-hop reasoning if needed.
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def is_configured(self) -> bool:
        return settings.openai_configured

    def get_chat_model(
        self,
        temperature: float = 0.1,
        streaming: bool = False,
    ) -> BaseChatModel:
        if not self.is_configured:
            raise RuntimeError(
                "OpenAI API key not configured. Set OPENAI_API_KEY in .env\n"
                "Integration implemented but requires API credentials to execute."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature,
            api_key=settings.openai_api_key,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            streaming=streaming,
        )

    def get_embedding_model(self):
        if settings.embedding_provider == "huggingface":
            return self._get_huggingface_embeddings()
        return self._get_openai_embeddings()

    def _get_openai_embeddings(self):
        if not self.is_configured:
            raise RuntimeError("OpenAI API key required for embeddings")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    def _get_huggingface_embeddings(self):
        """
        Local HuggingFace embeddings — no API key required.
        WHY: Useful for development, air-gapped environments, or cost reduction.
             Dimension: 384 (all-MiniLM-L6-v2) vs 1536 (text-embedding-3-small).
             Quality is lower but sufficient for many RAG use cases.
        """
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=settings.huggingface_embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            logger.warning("HuggingFace embeddings not available, falling back to OpenAI")
            return self._get_openai_embeddings()


# ── Provider registry ─────────────────────────────────────────────────────────

_providers: dict[str, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    # Future:
    # "gemini": GeminiProvider,
    # "claude": AnthropicProvider,
    # "ollama": OllamaProvider,
}


def get_llm_provider(name: Optional[str] = None) -> BaseLLMProvider:
    """Return the configured LLM provider."""
    provider_name = name or "openai"
    cls = _providers.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Available: {list(_providers.keys())}"
        )
    return cls()


# Singleton for the default provider
_default_provider: Optional[BaseLLMProvider] = None


def get_default_provider() -> BaseLLMProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = get_llm_provider()
    return _default_provider
