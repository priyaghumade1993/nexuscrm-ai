"""
NexusCRM AI — Application Settings
WHY: Centralised, validated configuration using Pydantic Settings.
     All secrets come from environment variables — nothing hardcoded.
     One source of truth for every configurable value.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "NexusCRM AI"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    secret_key: str = "dev-secret-key-change-in-production"
    allowed_origins: List[str] = ["http://localhost:8501", "http://localhost:3000"]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v

    # --- JWT ---
    jwt_secret_key: str = "jwt-dev-secret-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # --- PostgreSQL ---
    database_url: str = (
        "postgresql+asyncpg://nexuscrm_user:nexuscrm_password@localhost:5432/nexuscrm"
    )
    database_url_sync: str = (
        "postgresql://nexuscrm_user:nexuscrm_password@localhost:5432/nexuscrm"
    )

    # --- MongoDB ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "nexuscrm_ai"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_retries: int = 3
    openai_timeout: int = 60

    # --- Vector Store ---
    vector_store: Literal["faiss", "chroma", "pinecone"] = "faiss"
    faiss_index_path: str = "./data/faiss_index"
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "nexuscrm_knowledge"
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "nexuscrm-knowledge"

    # --- Embeddings ---
    embedding_provider: Literal["openai", "huggingface"] = "openai"
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LangSmith ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "nexuscrm-ai"

    # --- AWS ---
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-south-1"
    aws_s3_bucket: str = "nexuscrm-documents"

    # --- Rate Limiting ---
    rate_limit_per_minute: int = 60
    rate_limit_per_tenant_per_minute: int = 200

    # --- Knowledge Base ---
    knowledge_base_dir: str = "./knowledge_base"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7

    # --- Agent ---
    agent_max_iterations: int = 10
    agent_verbose: bool = True

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))

    @property
    def langsmith_configured(self) -> bool:
        return bool(self.langchain_api_key)

    @property
    def pinecone_configured(self) -> bool:
        return bool(self.pinecone_api_key)


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — reads .env once."""
    return Settings()


settings = get_settings()
