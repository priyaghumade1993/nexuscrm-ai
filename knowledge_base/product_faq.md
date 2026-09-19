# NexusCRM Product FAQ

## General Questions

**Q: What is NexusCRM?**
A: NexusCRM is an AI-powered Customer Relationship Management platform designed for modern sales teams. It combines traditional CRM functionality (lead management, deal tracking, customer records) with a multi-agent AI system powered by LangGraph and OpenAI GPT-4.

**Q: What makes NexusCRM different from other CRMs?**
A: Three key differentiators:
1. **Built-in AI Agent**: A multi-agent system that understands natural language queries and performs CRM actions on your behalf — not a bolted-on chatbot.
2. **RAG Knowledge Base**: Sales reps can ask about pricing, policies, and playbooks and get answers grounded in your actual company documents — no hallucinations.
3. **Multi-tenant Security**: Built from the ground up for agencies and enterprises managing multiple clients, with strict data isolation between tenants.

**Q: Is NexusCRM cloud-only or can it be self-hosted?**
A: Both options are available. The standard offering is cloud-hosted (AWS). Enterprise customers can opt for self-hosted deployment on their own infrastructure using our Docker Compose or Kubernetes manifests.

---

## AI Features

**Q: What AI capabilities does NexusCRM include?**
A: NexusCRM AI includes:
- **Natural language CRM queries**: "Show me all open leads from last week" or "Create a lead for Rahul Mehta at Infosys"
- **Deal intelligence**: Automatic deal stage progression suggestions based on activity patterns
- **Email drafting**: AI-generated follow-up emails personalized to the deal context
- **Pipeline analytics**: Natural language questions like "What's my pipeline value for Q4?"
- **Knowledge base Q&A**: Ask about pricing, policies, and product information
- **Lead scoring**: AI-powered lead quality scores based on engagement signals

**Q: Which AI model powers NexusCRM?**
A: NexusCRM uses OpenAI GPT-4o (default) for agent reasoning and tool selection. The model is configurable — Enterprise customers can use GPT-4-turbo or connect their own Azure OpenAI deployment.

**Q: Can the AI make changes to my CRM data?**
A: Yes, with guardrails:
- The AI can CREATE leads, UPDATE deal stages, and SEARCH records immediately
- DELETE operations always require explicit user confirmation — the AI will ask "Are you sure you want to delete X?" before executing
- All AI actions are logged with full audit trail in MongoDB
- Tenant isolation is enforced by application code, not the AI — the AI cannot access another tenant's data even if instructed to

**Q: Does the AI have access to all my data?**
A: The AI only has access to data within your tenant. It uses controlled database tools (not raw SQL) and can only perform operations exposed through the defined tool set. It cannot run arbitrary database queries.

**Q: What happens when the AI doesn't know something?**
A: For knowledge base queries, the AI responds with "I don't have that information in the knowledge base" rather than hallucinating an answer. For CRM data, it always queries the database — it never invents records.

---

## Security & Compliance

**Q: How is multi-tenancy handled?**
A: Every database record includes a `tenant_id` column. All queries are filtered by the authenticated user's tenant_id, extracted from their JWT token. Application code enforces this isolation — it is not dependent on AI behavior.

**Q: What authentication methods are supported?**
- Username/password with bcrypt hashing (all plans)
- JWT-based stateless sessions (all plans)
- SAML/SSO (Enterprise only): Okta, Azure AD, Google Workspace
- 2FA via TOTP (Professional and Enterprise)

**Q: Is our data encrypted?**
A: Yes:
- **In transit**: TLS 1.3 for all API traffic
- **At rest**: AES-256 encryption for database storage (AWS RDS encrypted)
- **Backups**: Encrypted daily backups with 30-day retention
- **API keys**: Stored as environment variables, never in source code or database

**Q: What compliance certifications does NexusCRM hold?**
A: SOC 2 Type II (in progress, expected Q2 2025), GDPR-compliant data processing, ISO 27001 (Enterprise tier). Full compliance reports available on request for Enterprise customers.

**Q: Where is data stored?**
A: Default: US East (AWS us-east-1). GDPR customers: EU (AWS eu-west-1). APAC: Singapore (AWS ap-southeast-1). Enterprise customers can specify data residency in their contract.

---

## Integrations

**Q: What integrations are available?**
A:
- **Email**: Gmail, Outlook/Office 365 (bi-directional sync)
- **Calendar**: Google Calendar, Outlook Calendar
- **Communication**: Slack (deal notifications), Teams
- **Marketing**: Mailchimp, HubSpot Marketing Hub
- **Data enrichment**: Clearbit, ZoomInfo (Professional+)
- **Automation**: Zapier (1000+ apps), Make (Integromat)
- **Analytics**: Power BI, Tableau, Google Data Studio
- **Custom**: REST API + Webhooks (Professional+)

**Q: Can we import data from Salesforce or HubSpot?**
A: Yes. We provide a migration wizard that imports:
- Contacts/Leads
- Accounts/Companies
- Deals/Opportunities
- Activity history
- Custom fields (mapped manually or auto-mapped)
Free migration service included for deals over $5,000/year.

**Q: Is there a REST API?**
A: Yes. Full REST API documentation available at `/docs` when running the application. Key endpoints: `/api/v1/leads`, `/api/v1/deals`, `/api/v1/customers`, `/api/v1/agent/chat`. API authentication uses the same JWT tokens as the UI.

---

## Technical Questions

**Q: What are the system requirements for self-hosted deployment?**
A:
- Docker Engine 24.x + Docker Compose 2.x
- 4 CPU cores, 8GB RAM minimum (16GB recommended for AI features)
- 50GB SSD storage (more for vector indexes)
- PostgreSQL 15+ (included in Docker Compose)
- MongoDB 7.0+ (included in Docker Compose)
- Internet access for OpenAI API calls (or Azure OpenAI for air-gapped)

**Q: What databases does NexusCRM use?**
A:
- **PostgreSQL**: Primary relational data (leads, customers, deals, users) — chosen for ACID compliance, complex queries, and Alembic migrations
- **MongoDB**: Conversation history, agent execution logs, audit events — chosen for flexible document schema as AI outputs vary in structure
- **Vector Database**: FAISS (default/dev), ChromaDB (local persistent), or Pinecone (production) — for RAG knowledge retrieval

**Q: How do I add documents to the knowledge base?**
A: Place markdown, PDF, or text files in the `knowledge_base/` directory, then run:
```bash
python scripts/ingest_knowledge_base.py
```
The script loads, chunks, embeds, and stores the documents in the vector database. The AI will immediately use the new documents for knowledge queries.

**Q: How do I scale NexusCRM for high traffic?**
A:
- The FastAPI backend is stateless — scale horizontally with multiple replicas
- Use AWS ALB or nginx as a load balancer
- PostgreSQL scales vertically (larger RDS instance) or with read replicas for reporting
- For AI throughput: increase `MAX_CONCURRENT_REQUESTS` in settings and use async throughout
- Production recommendation: minimum 3 FastAPI replicas behind a load balancer
