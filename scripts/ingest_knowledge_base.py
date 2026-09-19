"""
NexusCRM Knowledge Base Ingestion Script.

Loads markdown files from knowledge_base/ directory, chunks them,
generates embeddings, and stores in the configured vector store.

USAGE:
  cd backend
  python ../scripts/ingest_knowledge_base.py

  # Ingest a specific file:
  python ../scripts/ingest_knowledge_base.py --file pricing.md

  # Force re-ingestion (clears and rebuilds):
  python ../scripts/ingest_knowledge_base.py --reset

WHAT IT DOES:
  1. Reads all .md / .txt / .pdf files from knowledge_base/
  2. Splits into 512-token chunks with 50-token overlap
  3. Generates embeddings (OpenAI or HuggingFace depending on config)
  4. Stores in FAISS / ChromaDB / Pinecone (depends on VECTOR_STORE setting)
  5. Reports total chunks ingested

WHY SEPARATE SCRIPT (not auto-run at startup):
  Ingestion is expensive:
  - OpenAI embeddings: ~$0.02 per million tokens
  - HuggingFace: free but CPU-intensive
  Running at startup would slow every restart. Instead, run once when
  the knowledge base changes, and the vector store is persisted to disk.

INTERVIEW TALKING POINT:
  "The ingestion script is separated from the serve path intentionally.
   Knowledge base updates are infrequent (new pricing doc every quarter),
   so we ingest offline, persist the FAISS index, and the live app loads
   the pre-built index at startup in milliseconds."
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).parent.parent / "knowledge_base"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NexusCRM knowledge base into vector store")
    parser.add_argument(
        "--file", type=str, default=None,
        help="Ingest a specific file (relative to knowledge_base/). Default: ingest all files."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear existing index and rebuild from scratch (FAISS only)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count chunks that would be ingested without actually ingesting"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from app.config.settings import get_settings
    settings = get_settings()

    logger.info("NexusCRM Knowledge Base Ingestion")
    logger.info("Vector store:      %s", settings.vector_store)
    logger.info("Embedding model:   %s (%s)", settings.embedding_model, settings.embedding_provider)
    logger.info("Knowledge base dir: %s", KB_DIR)

    if not KB_DIR.exists():
        logger.error("Knowledge base directory not found: %s", KB_DIR)
        sys.exit(1)

    # Determine files to ingest
    if args.file:
        files = [KB_DIR / args.file]
        if not files[0].exists():
            logger.error("File not found: %s", files[0])
            sys.exit(1)
    else:
        files = [f for f in KB_DIR.rglob("*") if f.suffix in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning("No supported files found in %s", KB_DIR)
        sys.exit(0)

    logger.info("Files to ingest: %d", len(files))
    for f in files:
        logger.info("  → %s", f.name)

    if args.dry_run:
        logger.info("Dry run — counting chunks only")
        _count_chunks(files)
        return

    # Reset index if requested (FAISS: delete index files)
    if args.reset and settings.vector_store == "faiss":
        _reset_faiss_index(settings)

    # Initialize RAG pipeline (will build fresh index if reset)
    from app.rag.pipeline import get_rag_pipeline, reset_pipeline, ingest_markdown_file

    if args.reset:
        reset_pipeline()

    pipeline = get_rag_pipeline()
    if pipeline is None:
        logger.error(
            "Failed to initialize RAG pipeline. "
            "Check EMBEDDING_PROVIDER and OPENAI_API_KEY settings."
        )
        sys.exit(1)

    # Ingest each file
    total_chunks = 0
    start_time = time.time()

    for filepath in files:
        try:
            logger.info("Ingesting: %s", filepath.name)
            t0 = time.time()

            if filepath.suffix in (".md", ".txt"):
                chunks = ingest_markdown_file(filepath)
            elif filepath.suffix == ".pdf":
                chunks = _ingest_pdf(filepath, pipeline)
            else:
                logger.warning("Unsupported format: %s", filepath.suffix)
                continue

            elapsed = int((time.time() - t0) * 1000)
            logger.info("  → %d chunks in %dms", chunks, elapsed)
            total_chunks += chunks

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", filepath.name, exc, exc_info=True)

    elapsed_total = int((time.time() - start_time) * 1000)
    logger.info("=" * 50)
    logger.info("Ingestion complete!")
    logger.info("  Files processed: %d", len(files))
    logger.info("  Total chunks:    %d", total_chunks)
    logger.info("  Total time:      %dms", elapsed_total)
    logger.info("  Vector store:    %s", settings.vector_store)


def _count_chunks(files: list[Path]) -> None:
    """Count chunks without ingesting."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from app.rag.pipeline import CHUNK_SIZE, CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        for filepath in files:
            text = filepath.read_text(encoding="utf-8")
            chunks = splitter.split_text(text)
            logger.info("  %s → %d chunks", filepath.name, len(chunks))
    except ImportError:
        logger.warning("langchain not installed — cannot count chunks")


def _reset_faiss_index(settings) -> None:
    """Delete existing FAISS index files."""
    import shutil
    faiss_dir = Path(settings.faiss_index_path)
    if faiss_dir.exists():
        shutil.rmtree(faiss_dir)
        logger.info("Cleared existing FAISS index at %s", faiss_dir)


def _ingest_pdf(filepath: Path, pipeline) -> int:
    """Ingest a PDF file using LangChain's PyPDF loader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from app.rag.pipeline import CHUNK_SIZE, CHUNK_OVERLAP

        loader = PyPDFLoader(str(filepath))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source"] = str(filepath)
            chunk.metadata["title"] = filepath.stem.replace("_", " ").title()

        pipeline.add_documents(
            texts=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return 0


if __name__ == "__main__":
    main()
