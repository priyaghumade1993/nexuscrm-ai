"""
pytest configuration and shared fixtures for NexusCRM AI tests.

HOW TO RUN:
  cd backend
  pytest ../tests/ -v

  # Run only unit tests (fast, no DB):
  pytest ../tests/unit/ -v

  # Run with coverage:
  pytest ../tests/ --cov=app --cov-report=html
"""
import asyncio
import sys
from pathlib import Path

import pytest

# Add backend to path so tests can import app.*
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ── Event loop fixture (required for async tests) ─────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ── Mock settings for tests ───────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Override settings for tests.
    Prevents tests from requiring real API keys or database connections.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/nexuscrm_test")
    monkeypatch.setenv("DATABASE_URL_SYNC", "postgresql+psycopg2://test:test@localhost:5432/nexuscrm_test")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only-32chars")
    monkeypatch.setenv("VECTOR_STORE", "faiss")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    # Clear LangSmith to avoid tracing in tests
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)


# ── Sample test data ──────────────────────────────────────────────────────────
SAMPLE_TENANT_ID = "test-tenant-001"
SAMPLE_USER_ID = "test-user-001"

@pytest.fixture
def sample_tenant_id():
    return SAMPLE_TENANT_ID

@pytest.fixture
def sample_user_id():
    return SAMPLE_USER_ID
