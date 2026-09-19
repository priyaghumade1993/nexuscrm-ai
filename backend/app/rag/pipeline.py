"""
NexusCRM RAG Pipeline — Retrieval-Augmented Generation.

PIPELINE ARCHITECTURE:
  Documents (MD/PDF/TXT)
       ↓
  DocumentLoader (LangChain)
       ↓
  TextSplitter (RecursiveCharacterTextSplitter)
       ↓
  EmbeddingModel (OpenAI / HuggingFace)
       ↓
  VectorStore (FAISS / Chroma / Pinecone)
       ↓
  Retriever (similarity_search with k=4)
       ↓
  Context Assembly
       ↓
  LLM (knowledge_agent uses context as grounding)

WHY RAG INSTEAD OF FINE-TUNING:
  1. Knowledge updates: RAG lets you add new pricing/policy docs without
     retraining the model. Fine-tuning requires expensive GPU training runs.
  2. Source citation: RAG returns the source document — you know WHERE
     the answer came from. Fine-tuned models hallucinate without attribution.
  3. Tenant isolation: RAG can filter by metadata (tenant_id) at query time.
     Fine-tuning bakes all tenants' data into one model — data leakage risk.
  4. Cost: Embedding + retrieval is cents per 1000 queries.
     Fine-tuning is $100s per run + ongoing hosting.

WHY CHUNKING MATTERS:
  Too small (< 100 tokens): chunks lose context — "the price is $X" without
    the product name is useless.
  Too large (> 1000 tokens): retrieval returns too much noise, context window
    fills up, and irrelevant info confuses the LLM.
  Sweet spot: 512 tokens with 50-token overlap preserves context boundaries.

INTERVIEW TALKING POINT:
  "We use RAG so that when a sales rep asks 'what's our enterprise pricing?',
   the agent retrieves the actual pricing document — not a hallucinated number.
   The knowledge base is updated by uploading new markdown files and running
   the ingestion script. No model retraining needed."
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512        # tokens per chunk
CHUNK_OVERLAP = 50      # token overlap between adjacent chunks
RETRIEVAL_K = 4         # top-k documents to retrieve per query


class RAGPipeline:
    """
    Encapsulates the vector store + retriever for NexusCRM knowledge base.

    Usage:
        pipeline = await get_rag_pipeline()
        context = pipeline.retrieve("What is our enterprise pricing?")
    """

    def __init__(self, vector_store, retriever):
        self._store = vector_store
        self._retriever = retriever
        logger.info("RAGPipeline initialized (store=%s)", type(vector_store).__name__)

    def retrieve(self, query: str, k: int = RETRIEVAL_K) -> str:
        """
        Retrieve the top-k most relevant document chunks for a query.

        Returns:
            Formatted string of document chunks with source attribution,
            ready to inject into an LLM system prompt as context.

        WHY STRING NOT LIST:
            The calling agent injects this directly into a prompt template.
            Formatting it here keeps the agent nodes clean.
        """
        try:
            docs = self._store.similarity_search(query, k=k)
            if not docs:
                return "No relevant information found in the knowledge base."

            chunks = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "knowledge_base")
                title = doc.metadata.get("title", Path(source).stem if source else "unknown")
                chunks.append(f"[Source {i}: {title}]\n{doc.page_content}")

            return "\n\n".join(chunks)

        except Exception as exc:
            logger.error("RAG retrieval error: %s", exc, exc_info=True)
            return "Knowledge base temporarily unavailable."

    def add_documents(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> None:
        """
        Add new document chunks to the vector store.
        Called by the ingestion script when new knowledge base files are added.
        """
        self._store.add_texts(texts, metadatas=metadatas or [{}] * len(texts))
        logger.info("Added %d chunks to vector store", len(texts))


def _build_pipeline() -> Optional[RAGPipeline]:
    """
    Initialize embedding model + vector store + retriever.
    Returns None if initialization fails (graceful degradation).
    """
    try:
        from app.rag.embeddings import get_embedding_model
        from app.rag.vector_store import get_vector_store

        embedding_model = get_embedding_model()
        vector_store = get_vector_store(embedding_model, index_name="nexuscrm")
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K},
        )
        return RAGPipeline(vector_store=vector_store, retriever=retriever)

    except Exception as exc:
        logger.warning(
            "RAG pipeline initialization failed — knowledge queries will be limited: %s", exc
        )
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────
_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> Optional[RAGPipeline]:
    """
    Return the singleton RAG pipeline, initializing on first call.

    Returns None gracefully if:
    - No embedding model is configured
    - Vector store files don't exist yet
    - Required packages aren't installed

    Callers must handle None: ``context = pipeline.retrieve(q) if pipeline else ""``.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


def reset_pipeline() -> None:
    """
    Force re-initialization of the pipeline singleton.
    Call after ingesting new documents so the retriever picks up the new index.
    """
    global _pipeline
    _pipeline = None
    logger.info("RAG pipeline cache cleared — will reinitialize on next call")


# ── Document Ingestion Helpers (called by scripts/ingest_knowledge_base.py) ───

def ingest_markdown_file(filepath: str | Path) -> int:
    """
    Load a markdown file, split into chunks, add to vector store.

    Returns:
        Number of chunks ingested.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {filepath}")

    try:
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise RuntimeError(
            "langchain-community not installed. "
            "Run: pip install langchain-community"
        )

    loader = TextLoader(str(filepath), encoding="utf-8")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(raw_docs)

    # Enrich metadata
    for chunk in chunks:
        chunk.metadata.update({
            "source": str(filepath),
            "title": filepath.stem.replace("_", " ").title(),
            "file_type": "markdown",
        })

    pipeline = get_rag_pipeline()
    if pipeline is None:
        raise RuntimeError("RAG pipeline not initialized")

    pipeline.add_documents(
        texts=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    # Persist if using FAISS
    try:
        from app.config.settings import get_settings
        if get_settings().vector_store == "faiss":
            from app.rag.vector_store import save_faiss_store
            save_faiss_store(pipeline._store)
    except Exception as exc:
        logger.warning("Could not persist FAISS index: %s", exc)

    logger.info("Ingested %d chunks from %s", len(chunks), filepath.name)
    return len(chunks)
