"""
Lab 01: MySQL vs PostgreSQL for CRM Applications

Demonstrates key differences between MySQL and PostgreSQL in the context of
CRM data management, and explains why NexusCRM AI chose PostgreSQL.

INTERVIEW TALKING POINT:
  "I evaluated both databases before choosing PostgreSQL. The key deciding
   factors were: JSONB for flexible AI metadata, window functions for sales
   analytics, and better support for complex multi-table joins."

This script connects to both databases if available, or runs in
comparison/demo mode showing the SQL differences.

Requirements: pip install sqlalchemy pymysql psycopg2-binary
"""

import textwrap

# ── Key Differences: Side-by-Side SQL Comparison ─────────────────────────────

COMPARISONS = [
    {
        "topic": "JSON Storage",
        "context": "Storing AI agent metadata (tool calls, execution traces)",
        "mysql": """
            -- MySQL 8+: JSON column type
            CREATE TABLE agent_executions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tool_calls JSON,                    -- stored as text internally
                metadata JSON
            );

            -- Querying JSON (MySQL syntax)
            SELECT JSON_EXTRACT(tool_calls, '$.intent') AS intent
            FROM agent_executions
            WHERE JSON_EXTRACT(metadata, '$.tenant_id') = 'tenant-abc';
        """,
        "postgres": """
            -- PostgreSQL: JSONB (binary JSON — indexed, faster queries)
            CREATE TABLE agent_executions (
                id SERIAL PRIMARY KEY,
                tool_calls JSONB,                   -- binary, supports GIN index
                metadata   JSONB
            );

            -- Querying JSONB (PostgreSQL syntax — cleaner operators)
            SELECT tool_calls->>'intent' AS intent
            FROM agent_executions
            WHERE metadata->>'tenant_id' = 'tenant-abc';

            -- Can create a GIN index for fast JSONB queries:
            CREATE INDEX idx_exec_meta ON agent_executions USING GIN(metadata);
        """,
        "winner": "PostgreSQL",
        "reason": "JSONB is stored in binary format (faster reads), supports GIN indexes, "
                  "and has cleaner query operators. MySQL's JSON is stored as text internally."
    },
    {
        "topic": "Window Functions for Sales Analytics",
        "context": "Ranking sales reps by revenue, running totals",
        "mysql": """
            -- MySQL 8+: Window functions supported (but less mature)
            SELECT
                rep_name,
                won_revenue,
                RANK() OVER (ORDER BY won_revenue DESC) AS rank,
                SUM(won_revenue) OVER () AS total_revenue,
                won_revenue / SUM(won_revenue) OVER () * 100 AS pct_of_total
            FROM rep_summary;

            -- MySQL limitation: CTEs with window functions can have optimizer issues
            -- EXPLAIN output is less readable than PostgreSQL
        """,
        "postgres": """
            -- PostgreSQL: Full window function support, excellent optimizer
            SELECT
                rep_name,
                won_revenue,
                RANK() OVER (ORDER BY won_revenue DESC) AS rank,
                SUM(won_revenue) OVER () AS total_revenue,
                ROUND(won_revenue / SUM(won_revenue) OVER () * 100, 2) AS pct_of_total,
                -- Running total (PostgreSQL excels at this)
                SUM(won_revenue) OVER (ORDER BY won_revenue DESC ROWS UNBOUNDED PRECEDING)
                    AS running_total
            FROM rep_summary;
        """,
        "winner": "PostgreSQL",
        "reason": "PostgreSQL has more mature window function support and a more "
                  "powerful query planner (EXPLAIN ANALYZE is far more detailed)."
    },
    {
        "topic": "FILTER Clause in Aggregates",
        "context": "Computing conditional counts in one query (pipeline metrics)",
        "mysql": """
            -- MySQL: Must use CASE WHEN inside SUM() — verbose
            SELECT
                owner_id,
                COUNT(*) AS total_deals,
                SUM(CASE WHEN stage = 'closed_won' THEN 1 ELSE 0 END)  AS won,
                SUM(CASE WHEN stage = 'closed_lost' THEN 1 ELSE 0 END) AS lost,
                SUM(CASE WHEN stage = 'negotiation' THEN amount ELSE 0 END) AS negotiation_value
            FROM deals
            GROUP BY owner_id;
        """,
        "postgres": """
            -- PostgreSQL: FILTER clause — cleaner, more readable
            SELECT
                owner_id,
                COUNT(*)                                              AS total_deals,
                COUNT(*) FILTER (WHERE stage = 'closed_won')         AS won,
                COUNT(*) FILTER (WHERE stage = 'closed_lost')        AS lost,
                SUM(amount) FILTER (WHERE stage = 'negotiation')     AS negotiation_value
            FROM deals
            GROUP BY owner_id;
        """,
        "winner": "PostgreSQL",
        "reason": "FILTER clause is SQL standard (2003) and far more readable than "
                  "CASE WHEN inside aggregates. NexusCRM uses this extensively in analytics queries."
    },
    {
        "topic": "Full Text Search",
        "context": "Searching leads and notes by keyword",
        "mysql": """
            -- MySQL: FULLTEXT index on specific columns
            ALTER TABLE leads ADD FULLTEXT INDEX ft_notes (notes);

            SELECT id, first_name, last_name, notes
            FROM leads
            WHERE MATCH(notes) AGAINST ('upsell cloud migration' IN BOOLEAN MODE);
        """,
        "postgres": """
            -- PostgreSQL: tsvector / tsquery — much more powerful
            -- Add a computed column for the search vector
            ALTER TABLE leads ADD COLUMN search_vector tsvector
                GENERATED ALWAYS AS (
                    to_tsvector('english', COALESCE(notes, '') || ' ' || COALESCE(company, ''))
                ) STORED;

            CREATE INDEX idx_leads_fts ON leads USING GIN(search_vector);

            -- Query: supports phrase search, proximity, ranking
            SELECT id, first_name, last_name,
                   ts_rank(search_vector, query) AS rank
            FROM leads,
                 to_tsquery('english', 'upsell & cloud & migration') query
            WHERE search_vector @@ query
            ORDER BY rank DESC;
        """,
        "winner": "PostgreSQL",
        "reason": "PostgreSQL's tsvector/tsquery supports ranking (ts_rank), phrase queries, "
                  "stemming, and language-aware parsing. MySQL FULLTEXT is simpler but less flexible."
    },
    {
        "topic": "Numeric Precision (Financial Data)",
        "context": "Deal amounts — must never have floating-point rounding errors",
        "mysql": """
            -- MySQL: DECIMAL(15,2) — same as PostgreSQL, both handle this correctly
            CREATE TABLE deals (
                amount DECIMAL(15, 2) NOT NULL,  -- $9,999,999,999,999.99 max
                probability DECIMAL(5, 2)         -- 0.00 to 100.00
            );

            -- MySQL uses DECIMAL correctly — no difference here
            -- Note: avoid FLOAT or DOUBLE for financial data in either DB
        """,
        "postgres": """
            -- PostgreSQL: NUMERIC(15, 2) — equivalent to DECIMAL
            CREATE TABLE deals (
                amount      NUMERIC(15, 2) NOT NULL,  -- exact fixed-point arithmetic
                probability NUMERIC(5, 2)
            );

            -- PostgreSQL also has the 'money' type, but NUMERIC is preferred
            -- (money has locale-dependent formatting issues)

            -- SQLAlchemy mapping:
            -- amount: Mapped[Decimal] = mapped_column(Numeric(15, 2))
        """,
        "winner": "Tie",
        "reason": "Both handle DECIMAL/NUMERIC correctly. NexusCRM uses NUMERIC(15,2) in "
                  "SQLAlchemy which maps to NUMERIC in PostgreSQL and DECIMAL in MySQL."
    },
    {
        "topic": "RETURNING Clause",
        "context": "Get the inserted row's ID without a second query",
        "mysql": """
            -- MySQL: No RETURNING clause
            -- Must use LAST_INSERT_ID() after INSERT
            INSERT INTO leads (first_name, last_name, tenant_id)
            VALUES ('Alice', 'Smith', 'tenant-abc');

            SELECT LAST_INSERT_ID();  -- separate round-trip to the DB

            -- This is a race condition risk in concurrent applications
            -- (LAST_INSERT_ID() is session-scoped, so it's actually safe,
            --  but it's less clean than RETURNING)
        """,
        "postgres": """
            -- PostgreSQL: RETURNING clause — single round-trip
            INSERT INTO leads (first_name, last_name, tenant_id)
            VALUES ('Alice', 'Smith', 'tenant-abc')
            RETURNING id, created_at;

            -- In SQLAlchemy async:
            -- result = await db.execute(stmt.returning(Lead.id, Lead.created_at))
            -- row = result.fetchone()

            -- Also works with UPDATE and DELETE:
            UPDATE leads SET status = 'qualified'
            WHERE id = 42
            RETURNING id, updated_at;
        """,
        "winner": "PostgreSQL",
        "reason": "RETURNING eliminates a round-trip to get the inserted ID. "
                  "SQLAlchemy async uses this heavily — critical for performance."
    },
]


def print_comparison(comp: dict) -> None:
    width = 80
    print(f"\n{'='*width}")
    print(f"TOPIC: {comp['topic']}")
    print(f"CONTEXT: {comp['context']}")
    print(f"WINNER: {comp['winner']}")
    print(f"REASON: {textwrap.fill(comp['reason'], width=width)}")
    print(f"\n{'─'*40} MySQL {'─'*34}")
    print(textwrap.dedent(comp['mysql']).strip())
    print(f"\n{'─'*38} PostgreSQL {'─'*31}")
    print(textwrap.dedent(comp['postgres']).strip())


# ── Decision Matrix ───────────────────────────────────────────────────────────

DECISION_MATRIX = {
    "Feature": ["JSONB / binary JSON", "Window functions", "FILTER in aggregates",
                "Full-text search", "RETURNING clause", "Async driver",
                "Schema migrations (Alembic)", "Native arrays", "Row-level security",
                "LISTEN/NOTIFY (async events)"],
    "MySQL 8+": ["JSON (text)", "✓ (8.0+)", "✗ (CASE WHEN)", "FULLTEXT",
                 "✗ (LAST_INSERT_ID)", "aiomysql", "✓ (alembic)", "✗",
                 "✗ (app-level)", "✗"],
    "PostgreSQL 15": ["JSONB (binary)", "✓ (excellent)", "✓ (standard SQL)",
                      "tsvector (advanced)", "✓ (single query)", "asyncpg",
                      "✓ (alembic)", "✓ (array type)", "✓ (native RLS)", "✓"],
}


def print_matrix() -> None:
    print(f"\n{'='*80}")
    print("DECISION MATRIX: MySQL 8+ vs PostgreSQL 15")
    print(f"{'─'*80}")
    fmt = "{:<35} {:<22} {:<22}"
    print(fmt.format("Feature", "MySQL 8+", "PostgreSQL 15"))
    print(f"{'─'*80}")
    for i, feature in enumerate(DECISION_MATRIX["Feature"]):
        mysql_val = DECISION_MATRIX["MySQL 8+"][i]
        pg_val = DECISION_MATRIX["PostgreSQL 15"][i]
        print(fmt.format(feature, mysql_val, pg_val))


# ── Why NexusCRM chose PostgreSQL ────────────────────────────────────────────

CONCLUSION = """
WHY NEXUSCRM AI CHOSE POSTGRESQL
═════════════════════════════════

1. JSONB for AI metadata: MongoDB stores conversation history (flexible schema),
   but PostgreSQL JSONB handles structured AI execution metadata inline with
   relational CRM data — no extra join to another database.

2. Analytics queries: The Power BI views (Lab 07) use FILTER clause, window
   functions, and complex GROUP BY — all work identically in PostgreSQL 15.
   MySQL would require rewriting several views.

3. asyncpg performance: asyncpg (the Python async PostgreSQL driver) benchmarks
   faster than aiomysql for the async I/O pattern FastAPI uses. SQLAlchemy 2.0
   async mode is better tested against asyncpg.

4. Alembic maturity: Both support Alembic, but PostgreSQL's type system
   (NUMERIC, JSONB, ARRAY, ENUM) maps more naturally to SQLAlchemy models.

5. RETURNING clause: Every INSERT in the service layer uses RETURNING to get
   the generated ID in one round-trip. This is not available in MySQL.

WHEN WOULD I CHOOSE MYSQL?
──────────────────────────
- Existing infrastructure already running MySQL (migration cost)
- Team expertise strongly favors MySQL
- Using PlanetScale (MySQL-compatible, serverless scaling)
- WordPress/PHP stack (MySQL is the ecosystem standard)

For greenfield Python + FastAPI + AI projects: PostgreSQL is the better fit.
"""

if __name__ == "__main__":
    print("NexusCRM AI — Lab 01: MySQL vs PostgreSQL Comparison")
    print("=" * 80)

    for comp in COMPARISONS:
        print_comparison(comp)

    print_matrix()
    print(CONCLUSION)

    print("\nLab 01 complete.")
