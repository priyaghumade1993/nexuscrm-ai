# NexusCRM AI — Technology Decision Log (ADR)

Architecture Decision Records for every major technology choice.
Each decision documents: the context, the options considered, the choice made, and the trade-offs.

---

## ADR-001: LangGraph over LangChain LCEL for Agent Orchestration

**Status:** Accepted  
**Date:** 2024-01

### Context
We need an AI orchestration layer that can route user queries to different specialist agents (Sales, Customer, Analytics, Knowledge, Email) based on intent, enforce authorization rules, and support human-in-the-loop confirmations for destructive operations.

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| LangGraph | Stateful graph, conditional routing, checkpointing, built-in HitL | Newer API, more complex setup |
| LangChain LCEL | Mature, simple chains | Linear only — no branching |
| Custom Python | Full control | Reinventing the wheel; no observability |
| AutoGen / CrewAI | Multi-agent out of the box | Less control over security boundaries |

### Decision
**LangGraph** — chosen for its stateful graph model with conditional edges, which is essential for:
1. Branching on guardrail results (injection detected → END vs. continue)
2. Intent-based routing to different agents
3. Human-in-the-loop at the DELETE confirmation step
4. LangSmith integration for automatic tracing

### Trade-offs Accepted
- More complex graph topology than simple chains
- LangGraph API is evolving (pinned to 0.1.x)

---

## ADR-002: OpenAI GPT-4o as Primary LLM

**Status:** Accepted  
**Date:** 2024-01

### Context
We need an LLM for intent classification, tool selection, and response generation.

### Options Considered

| Option | Notes |
|--------|-------|
| OpenAI GPT-4o | Best tool-calling accuracy; JSON mode; widely deployed |
| Google Gemini | Not implemented per project constraints |
| Anthropic Claude | Not implemented per project constraints |
| Ollama (local) | Free but lower accuracy; no production SLA |
| Azure OpenAI | Same model, enterprise compliance; add later |

### Decision
**OpenAI GPT-4o** (default to gpt-4o-mini for cost optimization). Provider is abstracted behind `BaseLLMProvider` so other providers can be added later without changing agent code.

### Cost Note
Using gpt-4o-mini for intent classification and supervision; gpt-4o for response generation. Estimated cost: $0.01–0.05 per user query at production volume.

---

## ADR-003: PostgreSQL + MongoDB Polyglot Persistence

**Status:** Accepted  
**Date:** 2024-01

### Context
CRM data has two fundamentally different access patterns:
1. Structured relational queries (find leads by status, calculate pipeline value)
2. Flexible document storage (AI conversation history with varying fields)

### Options Considered

| Option | Notes |
|--------|-------|
| PostgreSQL only (with JSONB) | JSONB can store documents but JSON queries are slower |
| MongoDB only | Weaker transactional guarantees; JOINs are awkward |
| PostgreSQL + MongoDB | Best fit for each access pattern |
| MySQL | Less feature-rich (no native JSONB, weaker array support) |

### Decision
**Polyglot persistence:**
- PostgreSQL for all relational CRM data — ACID transactions, FK constraints, complex aggregations, Alembic migrations
- MongoDB for conversation history, agent execution logs, audit events — schema-flexible, write-heavy, no JOINs needed

### Interview Note
"PostgreSQL's `Numeric(15,2)` for deal amounts prevents the floating-point errors you'd get with `float`. MongoDB's flexible schema means an agent execution with 0 tools looks the same structurally as one with 5 — no schema migration needed."

---

## ADR-004: FAISS as Default Vector Store (with Pinecone for Production)

**Status:** Accepted  
**Date:** 2024-01

### Context
RAG requires a vector store for semantic search over the knowledge base.

### Options Considered

| Option | Cost | Infra | Scale | Notes |
|--------|------|-------|-------|-------|
| FAISS | Free | None | ~10M vectors | In-memory, file-persisted |
| ChromaDB | Free | None | ~1M vectors | Local persistent, metadata filtering |
| Pinecone | $$$  | Managed | Unlimited | Serverless, production-grade |
| Weaviate | Free/$$$ | Docker/Cloud | Large | Good but complex setup |
| Qdrant | Free/$$$ | Docker/Cloud | Large | Newer, good performance |

### Decision
**FAISS for development** (zero infrastructure, persists to files) → **Pinecone for production** (managed, no ops burden). Swappable via `VECTOR_STORE=faiss|chroma|pinecone` env var — zero code changes.

---

## ADR-005: Application-Enforced Authorization (Not LLM-Enforced)

**Status:** Accepted — CRITICAL SECURITY DESIGN  
**Date:** 2024-01

### Context
Who should enforce that a viewer cannot delete a lead? The application code, or the LLM?

### Analysis
**LLM-enforced authorization is NOT secure:**
- LLMs are probabilistic — they may follow instructions differently on different inputs
- Prompt injection can override LLM instructions: "Ignore your role restrictions"
- A different model version may have different compliance behavior
- There is no audit trail when the LLM decides to allow or deny

**Application-enforced authorization is secure:**
- Deterministic: same input, same result
- Cannot be overridden by prompt injection
- Generates audit logs
- Works even if LLM is replaced

### Decision
**Authorization enforced exclusively in Python application code:**
1. `authorization_validator` node in LangGraph checks role against a Python `ROLE_PERMISSIONS` dict
2. `require_role()` FastAPI dependency on every endpoint
3. Every service method receives `tenant_id` from JWT claims (not from user input)
4. The LLM never sees authorization logic — it only sees allowed operations

---

## ADR-006: SQLAlchemy 2.0 Async with asyncpg

**Status:** Accepted  
**Date:** 2024-01

### Context
FastAPI is async. Database operations should be non-blocking to support concurrent requests.

### Decision
- `create_async_engine` with `asyncpg` driver
- `async_sessionmaker` for session management
- Alembic uses synchronous URL (`psycopg2`) — Alembic doesn't support async
- `pool_size=10, max_overflow=20` — handles concurrent requests without exhausting connections

### Why NOT sync SQLAlchemy:
A sync DB call in an async FastAPI handler blocks the event loop, preventing other requests from being processed during that I/O wait. With asyncpg, 100 concurrent DB queries run in parallel instead of sequentially.

---

## ADR-007: Pydantic v2 for Data Validation

**Status:** Accepted  
**Date:** 2024-01

### Decision
Pydantic v2 (not v1) for:
- Request validation in FastAPI
- Response serialization (`model_from_attributes=True` for ORM objects)
- Settings management (`BaseSettings` with env var loading)
- Tool argument schemas for LangChain tools (enforces typed inputs to tools)

### Critical Use
`AgentContext` uses `model_config = ConfigDict(arbitrary_types_allowed=True)` to hold the `AsyncSession` (non-Pydantic type) alongside typed fields.

---

## ADR-008: Two-Layer Guardrail Design

**Status:** Accepted  
**Date:** 2024-01

### Decision
**Layer 1 (Pattern Matching):** Regex patterns for known injection signatures — runs in microseconds, catches 90%+ of attacks, no LLM cost.

**Layer 2 (LLM Semantic Check):** Only fires when patterns don't catch it — catches novel phrasing like "please disregard the previous context and help me with something different."

**Fail-open design for the LLM layer:** If OpenAI is unavailable, the LLM guardrail returns `is_safe=True`. The pattern matching layer (always available) still runs. Fail-closed would block all users during LLM downtime — wrong tradeoff for a CRM.

---

## ADR-009: DELETE Requires Explicit User Confirmation

**Status:** Accepted — CRITICAL SAFETY DESIGN  
**Date:** 2024-01

### Context
The LLM can misinterpret ambiguous instructions. "Clean up my leads" should NOT silently delete all leads.

### Decision
DELETE operations require two explicit signals:
1. **API level:** `?confirm=true` query parameter required on the DELETE endpoint (returns HTTP 400 without it)
2. **Tool level:** `confirmed=True` in `DeleteLeadArgs` (the tool checks before executing)
3. **Agent behavior:** If the user hasn't confirmed, the agent returns `requires_confirmation=True` and asks "Are you sure you want to delete X?" before executing

The LLM CANNOT pass `confirmed=True` on its own initiative — it only does so when the user has already said "yes" in the conversation.

---

## ADR-010: Streamlit for Demo Frontend

**Status:** Accepted (portfolio scope)  
**Date:** 2024-01

### Decision
Streamlit for the demo UI because:
- The portfolio goal is demonstrating AI/backend engineering, not frontend
- Streamlit lets us build a functional, visually clean demo in ~300 lines of Python
- Shares the same Python ecosystem as the backend

### Production Alternative
A production NexusCRM would use:
- Next.js 14 (App Router) + TypeScript for the frontend
- TanStack Query for API state management
- shadcn/ui component library
- The same `/api/v1` REST API (no backend changes needed)
