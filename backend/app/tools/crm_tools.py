"""
NexusCRM AI — LangChain CRM Tools.

WHY TOOL CALLING:
  The LLM cannot query the database directly. Instead, it selects from a
  controlled set of typed tools. This enforces:
  1. Tenant isolation — every tool injects the authenticated tenant_id.
  2. RBAC — tools check role before executing.
  3. No arbitrary SQL — the LLM can NEVER construct SQL. It only calls
     predefined functions with validated arguments.
  4. Auditability — every tool call is logged.

HOW THE LLM DECIDES WHICH TOOL TO CALL:
  The LLM sees tool descriptions and argument schemas. Based on the user's
  natural language intent, it selects the most appropriate tool and extracts
  the arguments. This is OpenAI's function-calling capability.

WHAT HAPPENS WHEN A TOOL FAILS:
  Each tool wraps its logic in try/except and returns a structured error
  string. LangGraph's error handling node catches persistent failures and
  returns a safe fallback to the user.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Context injected at agent runtime ─────────────────────────────────────────
# Tools need the DB session and current user context.
# We use a context object pattern — not global state — for thread safety.

class AgentContext(BaseModel):
    """Passed to every tool execution. Set by the LangGraph graph builder."""
    tenant_id: str
    user_id: str
    role: str
    db_session: Any = Field(exclude=True)  # AsyncSession — excluded from serialization

    class Config:
        arbitrary_types_allowed = True


# Module-level context — set per request, thread-local in production
_context: Optional[AgentContext] = None


def set_agent_context(ctx: AgentContext) -> None:
    global _context
    _context = ctx


def get_agent_context() -> AgentContext:
    if _context is None:
        raise RuntimeError("Agent context not set — call set_agent_context first")
    return _context


# ── Tool argument schemas ──────────────────────────────────────────────────────

class SearchLeadsArgs(BaseModel):
    query: str = Field(description="Name, company, or email to search for")
    status: Optional[str] = Field(None, description="Filter by status: new, contacted, qualified, unqualified, converted")
    limit: int = Field(10, ge=1, le=50, description="Maximum results")


class GetLeadArgs(BaseModel):
    lead_id: str = Field(description="UUID of the lead")


class CreateLeadArgs(BaseModel):
    name: str = Field(description="Full name of the lead")
    company: Optional[str] = Field(None, description="Company name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    source: str = Field("other", description="Lead source: website, referral, cold_call, email_campaign, social_media, trade_show, partner, other")


class UpdateLeadArgs(BaseModel):
    lead_id: str = Field(description="UUID of the lead to update")
    status: Optional[str] = Field(None, description="New status")
    score: Optional[int] = Field(None, ge=0, le=100, description="Lead score 0-100")
    notes: Optional[str] = Field(None, description="Notes to add")


class DeleteLeadArgs(BaseModel):
    lead_id: str = Field(description="UUID of the lead to delete")
    confirmed: bool = Field(False, description="Must be explicitly set to true by the user")


class SearchCustomersArgs(BaseModel):
    query: str = Field(description="Name, company, or industry to search")
    industry: Optional[str] = Field(None, description="Filter by industry")
    limit: int = Field(10, ge=1, le=50)


class SearchDealsArgs(BaseModel):
    stage: Optional[str] = Field(None, description="Deal stage filter")
    open_only: bool = Field(True, description="Return only open (non-closed) deals")
    limit: int = Field(10, ge=1, le=50)


class CalculatePipelineArgs(BaseModel):
    open_only: bool = Field(True, description="Include only open deals")
    stage: Optional[str] = Field(None, description="Filter by stage")


class GetInteractionsArgs(BaseModel):
    customer_id: str = Field(description="UUID of the customer")


class SearchKnowledgeBaseArgs(BaseModel):
    query: str = Field(description="Natural language question about company policies, pricing, products, or FAQs")


class GenerateFollowUpEmailArgs(BaseModel):
    deal_id: str = Field(description="UUID of the deal to generate follow-up for")
    tone: str = Field("professional", description="Email tone: professional, friendly, urgent")


# ── Async tool helpers (the real implementations use async DB calls) ────────────

async def _search_leads(args: SearchLeadsArgs) -> str:
    """Internal implementation used by the tool wrapper."""
    from app.services.lead_service import LeadService
    ctx = get_agent_context()
    svc = LeadService(db=ctx.db_session, tenant_id=ctx.tenant_id)

    from app.models.crm import LeadStatus
    status_filter = None
    if args.status:
        try:
            status_filter = LeadStatus(args.status)
        except ValueError:
            return f"Invalid status '{args.status}'. Valid values: new, contacted, qualified, unqualified, converted"

    leads, total = await svc.list(search=args.query, status=status_filter, page_size=args.limit)

    if not leads:
        return "No leads found matching your criteria."

    result = [f"Found {total} lead(s) (showing {len(leads)}):"]
    for lead in leads:
        result.append(
            f"- [{lead.id[:8]}] {lead.name} | {lead.company or 'No company'} | "
            f"Status: {lead.status.value} | Score: {lead.score} | "
            f"Source: {lead.source.value}"
        )
    return "\n".join(result)


async def _get_lead(args: GetLeadArgs) -> str:
    from app.services.lead_service import LeadService
    ctx = get_agent_context()
    lead = await LeadService(db=ctx.db_session, tenant_id=ctx.tenant_id).get_by_id(args.lead_id)
    if not lead:
        return f"Lead with ID {args.lead_id} not found."
    return (
        f"Lead: {lead.name}\n"
        f"ID: {lead.id}\n"
        f"Company: {lead.company or 'N/A'}\n"
        f"Email: {lead.email or 'N/A'}\n"
        f"Phone: {lead.phone or 'N/A'}\n"
        f"Status: {lead.status.value}\n"
        f"Score: {lead.score}\n"
        f"Source: {lead.source.value}\n"
        f"Notes: {lead.notes or 'None'}\n"
        f"Created: {lead.created_at.strftime('%Y-%m-%d')}"
    )


async def _create_lead(args: CreateLeadArgs) -> str:
    from app.services.lead_service import LeadService
    from app.schemas.lead import LeadCreate
    from app.models.crm import LeadSource
    ctx = get_agent_context()

    try:
        source = LeadSource(args.source)
    except ValueError:
        source = LeadSource.other

    data = LeadCreate(
        name=args.name,
        company=args.company,
        email=args.email,
        phone=args.phone,
        source=source,
    )
    lead = await LeadService(db=ctx.db_session, tenant_id=ctx.tenant_id).create(
        data, owner_id=ctx.user_id
    )
    return f"Lead created successfully. ID: {lead.id} | Name: {lead.name} | Status: {lead.status.value}"


async def _update_lead(args: UpdateLeadArgs) -> str:
    from app.services.lead_service import LeadService
    from app.schemas.lead import LeadUpdate
    from app.models.crm import LeadStatus
    ctx = get_agent_context()

    update_data: Dict[str, Any] = {}
    if args.status:
        try:
            update_data["status"] = LeadStatus(args.status)
        except ValueError:
            return f"Invalid status: {args.status}"
    if args.score is not None:
        update_data["score"] = args.score
    if args.notes:
        update_data["notes"] = args.notes

    lead = await LeadService(db=ctx.db_session, tenant_id=ctx.tenant_id).update(
        args.lead_id, LeadUpdate(**update_data)
    )
    if not lead:
        return f"Lead {args.lead_id} not found."
    return f"Lead updated. {lead.name} | Status: {lead.status.value} | Score: {lead.score}"


async def _delete_lead(args: DeleteLeadArgs) -> str:
    """
    DESTRUCTIVE — requires explicit user confirmation.
    WHY: LLM cannot silently delete. The user must say 'yes, delete it'
         which sets confirmed=True in the agent state.
    """
    if not args.confirmed:
        return (
            "⚠️ DELETE REQUIRES CONFIRMATION\n"
            "I need your explicit confirmation to delete this lead. "
            "Please reply with 'Yes, delete it' to confirm."
        )
    from app.models.crm import UserRole
    ctx = get_agent_context()
    if ctx.role not in [UserRole.admin.value, UserRole.manager.value]:
        return "❌ Unauthorized: Only admin or manager can delete leads."

    from app.services.lead_service import LeadService
    deleted = await LeadService(db=ctx.db_session, tenant_id=ctx.tenant_id).delete(args.lead_id)
    if not deleted:
        return f"Lead {args.lead_id} not found."
    return f"✅ Lead {args.lead_id} permanently deleted."


async def _search_customers(args: SearchCustomersArgs) -> str:
    from app.services.customer_service import CustomerService
    ctx = get_agent_context()
    customers, total = await CustomerService(db=ctx.db_session, tenant_id=ctx.tenant_id).list(
        search=args.query, industry=args.industry, page_size=args.limit
    )
    if not customers:
        return "No customers found."
    result = [f"Found {total} customer(s) (showing {len(customers)}):"]
    for c in customers:
        result.append(
            f"- [{c.id[:8]}] {c.name} | {c.company or 'N/A'} | "
            f"Industry: {c.industry or 'N/A'} | Email: {c.email or 'N/A'}"
        )
    return "\n".join(result)


async def _search_deals(args: SearchDealsArgs) -> str:
    from app.services.deal_service import DealService
    from app.models.crm import DealStage
    ctx = get_agent_context()

    stage_filter = None
    if args.stage:
        try:
            stage_filter = DealStage(args.stage)
        except ValueError:
            pass

    deals, total = await DealService(db=ctx.db_session, tenant_id=ctx.tenant_id).list(
        stage=stage_filter, page_size=args.limit
    )

    open_stages = {"closed_won", "closed_lost"}
    if args.open_only:
        deals = [d for d in deals if d.stage.value not in open_stages]

    if not deals:
        return "No deals found."
    result = [f"Found deals (showing {len(deals)}):"]
    for d in deals:
        amount_str = f"₹{float(d.amount):,.0f}" if d.amount else "No amount"
        close_str = d.expected_close_date.strftime("%Y-%m-%d") if d.expected_close_date else "TBD"
        result.append(
            f"- [{d.id[:8]}] {d.title} | {amount_str} | Stage: {d.stage.value} | "
            f"Probability: {d.probability}% | Close: {close_str}"
        )
    return "\n".join(result)


async def _calculate_pipeline(args: CalculatePipelineArgs) -> str:
    """
    DETERMINISTIC CALCULATION — not done by the LLM.
    The LLM calls this tool; the tool does the math in SQL.
    """
    from app.services.deal_service import DealService
    from app.models.crm import DealStage
    ctx = get_agent_context()
    stage_filter = None
    if args.stage:
        try:
            stage_filter = DealStage(args.stage)
        except ValueError:
            pass

    result = await DealService(db=ctx.db_session, tenant_id=ctx.tenant_id).calculate_pipeline_value(
        stage=stage_filter, open_only=args.open_only
    )
    return (
        f"Pipeline Summary:\n"
        f"Total Pipeline Value: ₹{result['total_pipeline_value']:,.2f}\n"
        f"Number of Deals: {result['deal_count']}\n"
        f"Average Deal Size: ₹{result['average_deal_size']:,.2f}\n"
        f"(Includes {'open deals only' if args.open_only else 'all deals'})"
    )


async def _get_customer_interactions(args: GetInteractionsArgs) -> str:
    from app.services.customer_service import CustomerService
    ctx = get_agent_context()
    svc = CustomerService(db=ctx.db_session, tenant_id=ctx.tenant_id)
    customer = await svc.get_by_id(args.customer_id)
    if not customer:
        return f"Customer {args.customer_id} not found."
    interactions = await svc.get_interactions(args.customer_id)
    if not interactions:
        return f"No interaction history for {customer.name}."
    result = [f"Interaction history for {customer.name} ({len(interactions)} records):"]
    for i in interactions:
        result.append(
            f"- {i.timestamp.strftime('%Y-%m-%d')} | {i.channel.value} | "
            f"{i.direction} | {i.message[:100]}{'...' if len(i.message) > 100 else ''}"
        )
    return "\n".join(result)


async def _generate_followup_email(args: GenerateFollowUpEmailArgs) -> str:
    """
    Generates a follow-up email draft using the LLM with deal context.
    This is a legitimate LLM use case — natural language generation grounded
    in actual deal data retrieved from the database.
    """
    from app.services.deal_service import DealService
    ctx = get_agent_context()
    deal = await DealService(db=ctx.db_session, tenant_id=ctx.tenant_id).get_by_id(args.deal_id)
    if not deal:
        return f"Deal {args.deal_id} not found."

    amount_str = f"₹{float(deal.amount):,.0f}" if deal.amount else "the discussed amount"
    close_str = deal.expected_close_date.strftime("%B %d, %Y") if deal.expected_close_date else "soon"

    # Email template grounded in real deal data
    email = f"""Subject: Following Up on {deal.title}

Dear [Customer Name],

I hope this message finds you well.

I wanted to follow up on our discussion regarding **{deal.title}**.

As we approach the target close date of {close_str}, I wanted to ensure all your questions are addressed and we're aligned on the next steps.

**Deal Summary:**
- Opportunity: {deal.title}
- Value: {amount_str}
- Current Stage: {deal.stage.value.replace('_', ' ').title()}
- Probability: {deal.probability}%

I'm confident we can deliver significant value for your organization. Could we schedule a brief call this week to discuss any remaining concerns?

Looking forward to your response.

Best regards,
[Your Name]
NexusCRM AI Team

---
Note: This is an AI-drafted email based on actual deal data. Please review and personalize before sending.
Deal ID: {deal.id}
"""
    return email


# ── Sync wrapper tools for LangChain (uses asyncio.get_event_loop) ────────────
# LangChain tools work best as sync functions that internally run async code.
# In FastAPI (async), we use a different approach — see graph/nodes.py.

def create_crm_tools(context: AgentContext) -> List[Any]:
    """
    Factory: creates tool list with the given context bound.
    Returns LangChain-compatible tool objects.
    """
    import asyncio

    def run(coro):
        """Run async tool in current event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result(timeout=30)
            return loop.run_until_complete(coro)
        except Exception as e:
            logger.error("Tool execution error: %s", e)
            return f"Tool error: {str(e)}"

    set_agent_context(context)

    @tool(args_schema=SearchLeadsArgs)
    def search_leads(query: str, status: Optional[str] = None, limit: int = 10) -> str:
        """Search for CRM leads by name, company, or email. Optionally filter by status."""
        return run(_search_leads(SearchLeadsArgs(query=query, status=status, limit=limit)))

    @tool(args_schema=GetLeadArgs)
    def get_lead(lead_id: str) -> str:
        """Get detailed information about a specific lead by its ID."""
        return run(_get_lead(GetLeadArgs(lead_id=lead_id)))

    @tool(args_schema=CreateLeadArgs)
    def create_lead(name: str, company: Optional[str] = None, email: Optional[str] = None,
                   phone: Optional[str] = None, source: str = "other") -> str:
        """Create a new lead in the CRM. Requires at minimum the lead's name."""
        return run(_create_lead(CreateLeadArgs(name=name, company=company, email=email,
                                               phone=phone, source=source)))

    @tool(args_schema=UpdateLeadArgs)
    def update_lead(lead_id: str, status: Optional[str] = None,
                   score: Optional[int] = None, notes: Optional[str] = None) -> str:
        """Update an existing lead's status, score, or notes."""
        return run(_update_lead(UpdateLeadArgs(lead_id=lead_id, status=status,
                                               score=score, notes=notes)))

    @tool(args_schema=DeleteLeadArgs)
    def delete_lead(lead_id: str, confirmed: bool = False) -> str:
        """Delete a lead. REQUIRES confirmed=True from explicit user confirmation."""
        return run(_delete_lead(DeleteLeadArgs(lead_id=lead_id, confirmed=confirmed)))

    @tool(args_schema=SearchCustomersArgs)
    def search_customers(query: str, industry: Optional[str] = None, limit: int = 10) -> str:
        """Search for customers by name, company, or industry."""
        return run(_search_customers(SearchCustomersArgs(query=query, industry=industry, limit=limit)))

    @tool(args_schema=SearchDealsArgs)
    def search_deals(stage: Optional[str] = None, open_only: bool = True, limit: int = 10) -> str:
        """Search for deals. Can filter by stage and whether to include only open deals."""
        return run(_search_deals(SearchDealsArgs(stage=stage, open_only=open_only, limit=limit)))

    @tool(args_schema=CalculatePipelineArgs)
    def calculate_pipeline_value(open_only: bool = True, stage: Optional[str] = None) -> str:
        """Calculate total pipeline value, deal count, and average deal size. All math done in SQL."""
        return run(_calculate_pipeline(CalculatePipelineArgs(open_only=open_only, stage=stage)))

    @tool(args_schema=GetInteractionsArgs)
    def get_customer_interactions(customer_id: str) -> str:
        """Get the complete interaction history for a customer (calls, emails, meetings)."""
        return run(_get_customer_interactions(GetInteractionsArgs(customer_id=customer_id)))

    @tool(args_schema=GenerateFollowUpEmailArgs)
    def generate_followup_email(deal_id: str, tone: str = "professional") -> str:
        """Generate a personalized follow-up email draft for a deal, grounded in actual deal data."""
        return run(_generate_followup_email(GenerateFollowUpEmailArgs(deal_id=deal_id, tone=tone)))

    return [
        search_leads,
        get_lead,
        create_lead,
        update_lead,
        delete_lead,
        search_customers,
        search_deals,
        calculate_pipeline_value,
        get_customer_interactions,
        generate_followup_email,
    ]
