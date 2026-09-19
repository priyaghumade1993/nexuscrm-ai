# NexusCRM AI — Interview Q&A Preparation

Concise answers to likely interview questions.
Practice saying these out loud — they should sound natural, not memorized.

---

## Architecture & System Design

### Q: Walk me through the system architecture.

**Answer:**
NexusCRM AI has three layers. The **API layer** is a FastAPI application handling auth (JWT + bcrypt), RBAC, and routing. The **agent layer** is a LangGraph stateful workflow: every user message flows through a guardrail check, intent classifier, authorization validator, and supervisor before reaching one of five specialist agents — Sales, Customer, Analytics, Knowledge, or Email. The **data layer** uses PostgreSQL for relational CRM data and MongoDB for flexible AI metadata like conversation history and execution logs. For knowledge retrieval we have a RAG pipeline backed by FAISS locally or Pinecone in production.

### Q: Why LangGraph instead of a simple LangChain chain?

**Answer:**
Chains are linear — A calls B calls C. Our workflow branches: if the guardrail detects injection, we skip everything and return an error. If the user is a viewer asking for delete, we block at the auth node. If the intent is pricing, we route to KnowledgeAgent instead of SalesAgent. None of that branching is possible with a simple chain. LangGraph gives us a directed graph with conditional edges, stateful state passing, and human-in-the-loop — which we use for DELETE confirmations.

### Q: How do you ensure multi-tenant data isolation?

**Answer:**
Three-layer defense. First, every PostgreSQL table has a `tenant_id` column, and every query in the service layer filters by `tenant_id` extracted from the JWT claims — not from user input. Second, the JWT is signed with our `SECRET_KEY`, so users can't modify their `tenant_id`. Third, the LLM tools are scoped at construction time with an `AgentContext` that includes `tenant_id` — the LLM only receives tool results for its own tenant. The LLM never sees or decides on tenant filtering.

---

## AI & LangChain

### Q: What is RAG and why did you use it?

**Answer:**
RAG stands for Retrieval-Augmented Generation. Instead of asking the LLM to answer from its training data — which would hallucinate pricing and policies — we retrieve the actual source documents first, then ask the LLM to answer based only on those documents. In NexusCRM, when a sales rep asks "what's our Enterprise pricing?", we embed the query, search the FAISS vector store for the most similar chunks from our pricing.md, inject those chunks into the LLM's context, and instruct it to cite sources. The LLM cannot invent a price that isn't in the retrieved context.

### Q: How does your prompt injection protection work?

**Answer:**
Two layers. Layer one is fast pattern matching — about 15 regex patterns for known injection signatures: "ignore your instructions," "you are now DAN," "SELECT * FROM," "show all tenants." This runs in microseconds and catches 90%+ of attacks with no LLM cost. Layer two is an LLM-based semantic check that only fires when patterns don't match — it catches novel phrasing like "please disregard the previous guidance." The LLM check fails open: if OpenAI is unavailable, it returns safe=True so users aren't blocked during downtime. The pattern layer always runs.

### Q: What's the difference between LangChain Tools and function calling?

**Answer:**
LangChain's `@tool` decorator wraps a Python function with a Pydantic schema and a docstring that the LLM reads to understand what the function does and what arguments it expects. Under the hood, when you use ChatOpenAI, LangChain converts these tools to OpenAI function definitions. The LLM selects which tool to call and returns structured JSON arguments — LangChain deserializes those arguments using the Pydantic schema and calls the Python function. The key safety property: the LLM selects and parameterizes the tool, but our Python code executes it — the LLM never runs arbitrary code.

### Q: How do you evaluate LLM output quality?

**Answer:**
I built a 30-case evaluation dataset covering all intent types, agent paths, and security scenarios. Each case has an expected intent, expected agent routing, required tool calls, and forbidden content strings. I run the live agent against all 30 cases and measure four metrics: intent accuracy (did the classifier get it right?), agent accuracy (did the supervisor route correctly?), tool precision (were the required tools called?), and safety rate (were injection attempts blocked?). For response quality, I use LLM-as-a-judge — a separate GPT-4 call that scores the response for accuracy and groundedness. These metrics are tracked in LangSmith across prompt versions.

---

## Security

### Q: How do you prevent the AI from deleting data without user consent?

**Answer:**
Four gates. One: the FastAPI DELETE endpoint requires `?confirm=true` as a query parameter — without it, the endpoint returns HTTP 400 before any agent code runs. Two: the `delete_lead` tool has a `confirmed: bool` argument — the tool checks this flag and raises an error if it's False. Three: the agent node checks `state.confirm_destructive` before calling the delete tool — if False, it returns `requires_confirmation=True` with a message asking the user to confirm. Four: the LLM's system prompt says "NEVER execute delete without explicit user confirmation." Each gate independently prevents silent deletion — defense in depth.

### Q: How do you handle authentication?

**Answer:**
Standard JWT with bcrypt. On login, we verify the password against the bcrypt hash in PostgreSQL and return a signed JWT containing `user_id`, `tenant_id`, and `role`. All subsequent requests carry this token in the Authorization header. The `get_current_user` FastAPI dependency validates the JWT signature, checks expiry, and fetches the User record to verify `is_active`. Role checks use `require_role()` dependencies — for example, DELETE endpoints require `manager` or above. The `tenant_id` in every DB query comes from the JWT claims, not from request parameters.

---

## Database

### Q: Why PostgreSQL AND MongoDB? Isn't that over-engineering?

**Answer:**
It's the right tool for each access pattern. CRM data — leads, customers, deals — is inherently relational: "find all leads from company X that are in 'qualified' status, owned by user Y, for tenant Z." That's a complex multi-table JOIN with filtering, sorting, and pagination — exactly what PostgreSQL is built for. Alembic gives us schema migrations, Numeric(15,2) gives us accurate financial arithmetic. Conversation history is different: each AI exchange has a different structure depending on which tools were called, how many, and what they returned. MongoDB's flexible document schema means we don't need to ALTER TABLE every time we add a new metadata field. The separation isn't over-engineering — it's matching the data model to the access pattern.

### Q: What is Alembic and why is it important?

**Answer:**
Alembic is SQLAlchemy's database migration tool. When you change a SQLAlchemy model — add a column, change a type, add an index — Alembic generates a migration script that transforms the existing database schema to match. This is essential for production: you can't just drop and recreate tables when users have live data. Alembic maintains a version history of schema changes, lets you roll back, and runs migrations in Docker CMD before starting the app — so the database schema is always in sync with the code. It uses a synchronous database URL (psycopg2, not asyncpg) because Alembic's autogenerate doesn't support async.

---

## Python & FastAPI

### Q: Why async throughout the stack?

**Answer:**
FastAPI is async, which means request handlers are coroutines. If a handler calls a synchronous database function, it blocks the event loop — no other request can be processed until that DB query returns. With asyncpg (async PostgreSQL driver) and Motor (async MongoDB driver), I/O operations release the event loop so other requests can be served concurrently. For a CRM with 100 concurrent users each making a DB query, async means those 100 queries run in parallel rather than sequentially. The tradeoff: slightly more complex code (async/await keywords, async context managers) and Alembic still requires sync (separate URL).

### Q: What is dependency injection in FastAPI?

**Answer:**
FastAPI's `Depends()` system automatically resolves function dependencies. When I write `current_user: User = Depends(get_current_user)`, FastAPI calls `get_current_user()` for every request to that endpoint, passing in its own dependencies (like the request object to extract the JWT). This makes auth, DB sessions, and role checks reusable across endpoints — I write the logic once in `get_current_user` and inject it everywhere. It also makes testing easy: you can override `Depends(get_current_user)` with a mock in tests.

---

## Observability

### Q: How do you observe and debug the AI agent in production?

**Answer:**
Three layers. First, Python logging at every significant step — guardrail checks, intent classification, tool calls — with structured fields like tenant_id, execution_id, and latency. Second, MongoDB stores every agent execution with the input, intent, tools called, output, and success/failure — queryable for debugging specific user complaints. Third, LangSmith tracing: when `LANGCHAIN_TRACING_V2=true`, every LangGraph node execution, LLM call, and tool invocation is automatically traced to LangSmith with inputs, outputs, and latency. In production, I can click on a trace and see exactly what the LLM received, what tools it called, and what it responded — without changing any code.
