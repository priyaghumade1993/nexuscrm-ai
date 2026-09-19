# NexusCRM AI — Technical Skill Matrix

Skills demonstrated in this portfolio project, with evidence.

| Skill | Level | Where Demonstrated |
|-------|-------|-------------------|
| **Python** | Advanced | Entire codebase — async/await, type hints, Pydantic v2, dataclasses |
| **FastAPI** | Advanced | `backend/app/api/v1/` — REST endpoints, dependency injection, middleware |
| **SQLAlchemy 2.0 Async** | Advanced | `backend/app/database/base.py`, `backend/app/services/` |
| **PostgreSQL** | Intermediate | Multi-table schema, indexes, Alembic migrations, complex queries |
| **Alembic** | Intermediate | `migrations/env.py` — schema versioning, sync/async URL handling |
| **MongoDB / Motor** | Intermediate | `backend/app/database/mongodb.py` — async document store for AI metadata |
| **JWT / Auth** | Advanced | `backend/app/auth/` — bcrypt, JWT, RBAC, dependency injection |
| **LangChain** | Advanced | `backend/app/tools/crm_tools.py` — 10 @tool-decorated functions with Pydantic schemas |
| **LangGraph** | Advanced | `backend/app/graph/` — stateful graph, conditional routing, human-in-the-loop |
| **LangSmith** | Intermediate | Auto-tracing via LANGCHAIN_TRACING_V2; evaluation experiments |
| **OpenAI API** | Advanced | `backend/app/agents/llm_provider.py` — ChatOpenAI, embeddings, temperature control |
| **RAG Pipeline** | Advanced | `backend/app/rag/` — embedding, chunking, FAISS/Chroma/Pinecone adapters |
| **Vector Databases** | Intermediate | FAISS (default), ChromaDB, Pinecone — swappable via env var |
| **Multi-Agent Systems** | Advanced | Supervisor → 5 specialist agents; intent routing, guardrails |
| **Prompt Engineering** | Advanced | `backend/app/graph/prompts.py` — structured prompts with few-shot examples |
| **Prompt Injection Defense** | Advanced | `backend/app/guardrails/` — 2-layer: pattern matching + LLM semantic check |
| **Docker** | Intermediate | `Dockerfile` (multi-stage), `docker-compose.yml` (4 services) |
| **GitHub Actions CI/CD** | Intermediate | `.github/workflows/ci.yml` — lint, unit tests, integration tests, Docker build, ECS deploy |
| **Evaluation / Testing** | Advanced | 30-case dataset, LLM-as-judge, pytest unit tests, coverage |
| **Streamlit** | Intermediate | `frontend/streamlit/app.py` — multi-page chat UI with real API integration |
| **Pydantic v2** | Advanced | Settings, request/response schemas, tool arg schemas |
| **RBAC** | Advanced | Role enum, permission matrix, dependency injection enforcement |
| **Multi-tenancy** | Advanced | `tenant_id` on every table, JWT-claim enforcement, service-layer scoping |
| **Async Python** | Advanced | asyncpg, Motor, async SQLAlchemy, async FastAPI handlers |
| **Software Architecture** | Advanced | ADR documentation, separation of concerns, provider abstraction |

## Technologies NOT in this project (by design)

Per the project specification:
- ❌ Google Gemini API
- ❌ Anthropic Claude API
- ❌ Ollama

These are excluded intentionally to keep the scope focused. See `labs/` folder for supplementary explorations of additional technologies.
