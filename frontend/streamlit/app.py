"""
NexusCRM AI — Streamlit Frontend.

A portfolio-quality demo UI showcasing the NexusCRM AI agent.

PAGES:
  1. 🤖 AI Chat      — Main agent interface; conversational CRM
  2. 📊 Dashboard    — Pipeline metrics, lead status charts
  3. 👥 Leads        — Lead list with search/filter
  4. 💼 Deals        — Deal pipeline board
  5. 🧪 Evaluation   — Run the evaluation dataset, view results
  6. ⚙️  Settings    — API endpoint, auth token

HOW TO RUN:
  streamlit run frontend/streamlit/app.py

  Or with a custom API backend:
  API_BASE_URL=http://localhost:8000 streamlit run frontend/streamlit/app.py

INTERVIEW TALKING POINT:
  "I built the frontend in Streamlit rather than React because the portfolio
   goal is demonstrating AI/backend engineering, not frontend work. Streamlit
   lets me build a functional, good-looking demo in ~300 lines of Python.
   A production version would use Next.js + the REST API."
"""
import os
import json
import time
from datetime import datetime

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1 = f"{API_BASE}/api/v1"

st.set_page_config(
    page_title="NexusCRM AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State ─────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None


# ── API Helpers ───────────────────────────────────────────────────────────────
def api_headers() -> dict:
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_V1}{path}", headers=api_headers(), timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Is the backend running?")
    return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_V1}{path}", json=payload, headers=api_headers(), timeout=30)
        if r.status_code in (200, 201):
            return r.json()
        st.error(f"API error {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Is the backend running?")
    return None


# ── Login Page ────────────────────────────────────────────────────────────────
def login_page():
    st.title("🤖 NexusCRM AI")
    st.subheader("Sign in to continue")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("Email", value="priya@geekhub.io")
            password = st.text_input("Password", type="password", value="Password123!")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                result = api_post("/auth/login", {"email": email, "password": password})
                if result and "access_token" in result:
                    st.session_state.token = result["access_token"]
                    st.session_state.user_email = email
                    st.success("Signed in!")
                    time.sleep(0.5)
                    st.rerun()
                elif result:
                    st.error("Invalid credentials")

        st.divider()
        st.caption("**Demo credentials** (all password: `Password123!`)")
        st.code(
            "Admin:   admin@geekhub.io\n"
            "Manager: manager@geekhub.io\n"
            "Sales:   priya@geekhub.io\n"
            "Viewer:  viewer@geekhub.io"
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.title("🤖 NexusCRM AI")
        st.caption(f"Signed in as **{st.session_state.user_email}**")

        # Health check
        try:
            health = requests.get(f"{API_BASE}/health", timeout=3).json()
            status_color = "🟢" if health.get("status") == "healthy" else "🟡"
            st.caption(f"{status_color} Backend: {health.get('status', 'unknown')}")
            if health.get("openai") == "configured":
                st.caption("🤖 OpenAI: configured")
            else:
                st.caption("⚠️ OpenAI: not configured (demo mode)")
        except Exception:
            st.caption("🔴 Backend: offline")

        st.divider()

        page = st.radio(
            "Navigation",
            ["🤖 AI Chat", "📊 Dashboard", "👥 Leads", "💼 Deals", "🧪 Evaluation"],
            label_visibility="collapsed",
        )

        st.divider()
        if st.button("Sign Out", use_container_width=True):
            st.session_state.token = None
            st.session_state.chat_history = []
            st.session_state.session_id = None
            st.rerun()

    return page


# ── AI Chat Page ──────────────────────────────────────────────────────────────
def chat_page():
    st.title("🤖 NexusCRM AI Assistant")
    st.caption("Ask me anything about your leads, deals, customers, or company knowledge.")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("metadata"):
                meta = msg["metadata"]
                cols = st.columns(4)
                if meta.get("intent"):
                    cols[0].caption(f"🎯 Intent: `{meta['intent']}`")
                if meta.get("agent"):
                    cols[1].caption(f"🤖 Agent: `{meta['agent']}`")
                if meta.get("latency_ms"):
                    cols[2].caption(f"⚡ {meta['latency_ms']}ms")
                if meta.get("tools"):
                    with st.expander(f"🔧 Tools used ({len(meta['tools'])})"):
                        for tool in meta["tools"]:
                            st.code(f"[{tool.get('tool', 'unknown')}] {tool.get('result', '')[:200]}")

    # Confirmation pending
    if st.session_state.get("pending_confirmation"):
        conf = st.session_state.pending_confirmation
        st.warning(f"⚠️ {conf['message']}")
        col1, col2 = st.columns(2)
        if col1.button("✅ Yes, proceed", type="primary"):
            _send_chat(conf["original_message"], confirm=True)
            del st.session_state.pending_confirmation
            st.rerun()
        if col2.button("❌ Cancel"):
            del st.session_state.pending_confirmation
            st.info("Action cancelled.")

    # Chat input
    if prompt := st.chat_input("Ask NexusCRM AI..."):
        _send_chat(prompt)
        st.rerun()

    # Example queries
    with st.expander("💡 Example queries"):
        examples = [
            "Show me all my open leads",
            "What is the total pipeline value?",
            "Find customer TCS and show their interaction history",
            "What is our Enterprise plan pricing?",
            "Create a lead for Raj Sharma at Infosys, email raj@infosys.com",
            "Write a follow-up email for the TCS deal",
            "What deals are closing this month?",
            "How should I handle a price objection?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}"):
                _send_chat(ex)
                st.rerun()


def _send_chat(message: str, confirm: bool = False):
    st.session_state.chat_history.append({"role": "user", "content": message})

    with st.spinner("🤔 Thinking..."):
        payload = {
            "message": message,
            "session_id": st.session_state.session_id,
            "confirm_destructive": confirm,
        }
        result = api_post("/agent/chat", payload)

    if result:
        st.session_state.session_id = result.get("session_id")

        if result.get("requires_confirmation"):
            st.session_state.pending_confirmation = {
                "message": result.get("confirmation_message", "Are you sure?"),
                "original_message": message,
            }

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.get("response", "No response"),
            "metadata": {
                "intent": result.get("intent"),
                "agent": result.get("agent"),
                "latency_ms": result.get("latency_ms"),
                "tools": result.get("tools_used", []),
            }
        })
    else:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "⚠️ Could not reach the AI agent. Please check the backend.",
        })


# ── Dashboard Page ────────────────────────────────────────────────────────────
def dashboard_page():
    st.title("📊 Pipeline Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    leads_data = api_get("/leads?page_size=100")
    deals_data = api_get("/deals?page_size=100")

    if leads_data:
        total_leads = leads_data.get("total", 0)
        col1.metric("Total Leads", total_leads)

        hot_leads = sum(1 for l in leads_data.get("items", []) if l.get("status") == "hot")
        col2.metric("🔥 Hot Leads", hot_leads)

    if deals_data:
        open_deals = [d for d in deals_data.get("items", [])
                      if d.get("stage") not in ("closed_won", "closed_lost")]
        total_pipeline = sum(float(d.get("amount", 0)) for d in open_deals)
        col3.metric("Pipeline Value", f"${total_pipeline:,.0f}")
        col4.metric("Open Deals", len(open_deals))

    st.divider()

    if leads_data:
        import pandas as pd
        leads = leads_data.get("items", [])
        if leads:
            df = pd.DataFrame(leads)
            st.subheader("Lead Status Distribution")
            status_counts = df["status"].value_counts()
            st.bar_chart(status_counts)

    if deals_data:
        deals = deals_data.get("items", [])
        if deals:
            st.subheader("Recent Deals")
            df = pd.DataFrame(deals)[["name", "stage", "amount", "expected_close_date"]]
            df["amount"] = df["amount"].apply(lambda x: f"${float(x):,.0f}")
            st.dataframe(df, use_container_width=True)


# ── Leads Page ────────────────────────────────────────────────────────────────
def leads_page():
    st.title("👥 Leads")

    search = st.text_input("Search leads", placeholder="Name, email, or company...")

    params = f"?page_size=50"
    if search:
        params += f"&search={search}"

    data = api_get(f"/leads{params}")
    if data:
        leads = data.get("items", [])
        st.caption(f"Showing {len(leads)} of {data.get('total', 0)} leads")
        if leads:
            import pandas as pd
            df = pd.DataFrame(leads)
            display_cols = ["name", "email", "company", "status", "lead_score", "source"]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available], use_container_width=True)
        else:
            st.info("No leads found.")


# ── Deals Page ────────────────────────────────────────────────────────────────
def deals_page():
    st.title("💼 Deal Pipeline")

    data = api_get("/deals?page_size=50")
    if data:
        deals = data.get("items", [])
        stages = ["prospect", "qualified", "demo_scheduled", "demo_completed",
                  "proposal_sent", "negotiation", "closed_won", "closed_lost"]

        cols = st.columns(len(stages))
        for i, stage in enumerate(stages):
            stage_deals = [d for d in deals if d.get("stage") == stage]
            stage_value = sum(float(d.get("amount", 0)) for d in stage_deals)
            with cols[i]:
                st.caption(f"**{stage.replace('_', ' ').title()}**")
                st.caption(f"${stage_value:,.0f} ({len(stage_deals)})")
                for deal in stage_deals:
                    st.info(f"**{deal['name'][:20]}**\n${float(deal.get('amount', 0)):,.0f}")


# ── Evaluation Page ───────────────────────────────────────────────────────────
def evaluation_page():
    st.title("🧪 AI Agent Evaluation")
    st.caption("Run the 30-case evaluation suite to measure agent accuracy.")

    st.info(
        "**Evaluation runs the live agent** against 30 test cases covering "
        "all intents, agents, and safety scenarios. Requires OpenAI API key."
    )

    col1, col2 = st.columns(2)
    tag_filter = col1.multiselect(
        "Filter by tag",
        ["happy_path", "safety", "security", "rbac", "analytics", "knowledge", "email"],
    )

    if st.button("▶️ Run Evaluation", type="primary"):
        with st.spinner("Running evaluation (this may take 2–5 minutes)..."):
            import asyncio, sys
            sys.path.insert(0, "backend")
            try:
                from app.evaluation.runner import run_evaluation
                report = asyncio.run(run_evaluation(
                    tag_filter=tag_filter or None,
                    output_file="evaluation_results/latest.json",
                ))
                st.success(
                    f"✅ Evaluation complete — "
                    f"{report.passed}/{report.total_cases} passed "
                    f"({report.passed/report.total_cases:.0%})"
                )
                col1, col2, col3 = st.columns(3)
                col1.metric("Intent Accuracy", f"{report.intent_accuracy:.0%}")
                col2.metric("Agent Accuracy", f"{report.agent_accuracy:.0%}")
                col3.metric("Avg Latency", f"{report.avg_latency_ms:.0f}ms")

                st.subheader("Results")
                import pandas as pd
                df = pd.DataFrame(report.results)
                df["status"] = df["passed"].apply(lambda x: "✓ PASS" if x else "✗ FAIL")
                display_cols = ["case_id", "input", "expected_intent", "actual_intent",
                                "latency_ms", "status"]
                available = [c for c in display_cols if c in df.columns]
                st.dataframe(df[available], use_container_width=True)

            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.caption("Make sure the backend dependencies are installed and OPENAI_API_KEY is set.")


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.token:
        login_page()
        return

    page = sidebar()

    if page == "🤖 AI Chat":
        chat_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "👥 Leads":
        leads_page()
    elif page == "💼 Deals":
        deals_page()
    elif page == "🧪 Evaluation":
        evaluation_page()


if __name__ == "__main__":
    main()
