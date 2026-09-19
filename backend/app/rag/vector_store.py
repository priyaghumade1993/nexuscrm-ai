"""
Vector Store Adapters for NexusCRM RAG Pipeline.

WHY THREE STORES:
  - FAISS: Zero infrastructure, perfect for demos and development.
             Files saved to disk: knowledge_base/*.faiss + *.pkl
             No server needed. Memory-mapped for fast retrieval.
  - ChromaDB: Local persistent store with metadata filtering.
               Runs as an in-process library (no Docker needed for dev).
               Supports filtering by tenant_id or document_type.
  - Pinecone: Managed cloud vector DB. Required for production at scale.
               Serverless, always-on, no index management.
               Requires PINECONE_API_KEY + PINECONE_ENVIRONMENT.

SWITCHING STORES:
  Set VECTOR_STORE=faiss|chroma|pinecone in .env.
  The pipeline calls get_vector_store() which reads this setting.
  All three implement the same LangChain VectorStore interface.

INTERVIEW TALKING POINT:
  "I abstracted the vector store behind a factory function so the
   product team can switch from FAISS (dev) to Pinecone (prod) by
   changing one env var — zero code changes."
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_vector_store(embedding_model: Any, index_name: str = "nexuscrm"):
    """
    Factory: return the configured vector store (FAISS / Chroma / Pinecone).

    Args:
        embedding_model: LangChain-compatible embedding instance
        index_name: logical name for the index / collection

    Returns:
        LangChain VectorStore instance
    """
    from app.config.settings import get_settings
    settings = get_settings()
    store = settings.vector_store.lower()

    if store == "pinecone":
        return _get_pinecone(embedding_model, index_name, settings)
    elif store == "chroma":
        return _get_chroma(embedding_model, index_name, settings)
    else:
        return _get_faiss(embedding_model, index_name, settings)


# ── FAISS ─────────────────────────────────────────────────────────────────────
def _get_faiss(embedding_model: Any, index_name: str, settings: Any):
    """
    FAISS-backed vector store.

    WHY FAISS:
      Facebook AI Similarity Search — the gold standard for in-memory
      ANN (Approximate Nearest Neighbor) search. Zero infrastructure,
      ships as a pip package, persisted as flat files.

    LIMITATION: Not horizontally scalable. Fine for knowledge bases
      under ~1M vectors. For larger corpora, use Pinecone.
    """
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        raise RuntimeError(
            "langchain-community is not installed. "
            "Run: pip install langchain-community faiss-cpu"
        )

    persist_path = Path(settings.faiss_index_path)
    index_file = persist_path / f"{index_name}.faiss"
    pkl_file = persist_path / f"{index_name}.pkl"

    if index_file.exists() and pkl_file.exists():
        logger.info("Loading existing FAISS index from %s", persist_path)
        store = FAISS.load_local(
            str(persist_path),
            embedding_model,
            index_name=index_name,
            allow_dangerous_deserialization=True,  # safe: our own serialized files
        )
    else:
        logger.info("Creating new empty FAISS index at %s", persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)
        # Create with a placeholder doc to initialize index
        store = FAISS.from_texts(
            ["NexusCRM AI Knowledge Base initialized."],
            embedding_model,
        )
        store.save_local(str(persist_path), index_name=index_name)

    return store


def save_faiss_store(store: Any, index_name: str = "nexuscrm") -> None:
    """Persist FAISS index to disk after ingesting new documents."""
    from app.config.settings import get_settings
    settings = get_settings()
    persist_path = Path(settings.faiss_index_path)
    persist_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(persist_path), index_name=index_name)
    logger.info("FAISS index saved to %s", persist_path)


# ── ChromaDB ──────────────────────────────────────────────────────────────────
def _get_chroma(embedding_model: Any, index_name: str, settings: Any):
    """
    ChromaDB-backed vector store.

    WHY CHROMA:
      Runs in-process (no separate server for dev), supports metadata
      filtering natively (filter by tenant_id, document_type), and
      has a simple HTTP client for when you want a separate server.

    PRODUCTION SETUP: Run `chroma run --host 0.0.0.0 --port 8000`
      and set CHROMA_HOST + CHROMA_PORT in your env.
    """
    try:
        from langchain_chroma import Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise RuntimeError(
                "ChromaDB not installed. Run: pip install langchain-chroma chromadb"
            )

    persist_dir = settings.chroma_persist_dir

    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        # Remote Chroma server
        import chromadb
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        logger.info("Connecting to Chroma server at %s:%d", chroma_host, chroma_port)
        return Chroma(
            client=client,
            collection_name=index_name,
            embedding_function=embedding_model,
        )
    else:
        # In-process persistent Chroma
        logger.info("Using local Chroma at %s (collection=%s)", persist_dir, index_name)
        return Chroma(
            collection_name=index_name,
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )


# ── Pinecone ──────────────────────────────────────────────────────────────────
def _get_pinecone(embedding_model: Any, index_name: str, settings: Any):
    """
    Pinecone-backed vector store.

    WHY PINECONE:
      Managed, serverless, infinitely scalable. No index management.
      Supports metadata filtering at query time (filter by tenant_id).
      Production-grade choice for enterprise CRM RAG.

    REQUIRES:
      PINECONE_API_KEY and PINECONE_ENVIRONMENT set in .env
      Index must already exist in Pinecone console (or auto-created below).

    INTERVIEW TALKING POINT:
      "Pinecone handles billions of vectors with millisecond latency.
       I chose it for production because the fully managed nature means
       no ops burden — you pay per query, not per server."
    """
    if not settings.pinecone_configured:
        raise RuntimeError(
            "Pinecone is not configured. Set PINECONE_API_KEY in .env "
            "or switch VECTOR_STORE=faiss for local development."
        )

    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        raise RuntimeError(
            "Pinecone not installed. Run: pip install langchain-pinecone pinecone-client"
        )

    pc = Pinecone(api_key=settings.pinecone_api_key)
    environment = settings.pinecone_environment

    # Create index if it doesn't exist
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info("Creating Pinecone index '%s'", index_name)
        pc.create_index(
            name=index_name,
            dimension=1536,  # text-embedding-3-small / ada-002 dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=environment),
        )
    else:
        logger.info("Using existing Pinecone index '%s'", index_name)

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model,
        pinecone_api_key=settings.pinecone_api_key,
    )
