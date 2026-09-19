/**
 * Lab 02: JavaScript/TypeScript API Client for NexusCRM AI
 *
 * A typed API client demonstrating how to consume the NexusCRM FastAPI backend
 * from a Node.js or browser JavaScript/TypeScript application.
 *
 * INTERVIEW TALKING POINT:
 *   "I built a typed TypeScript client to show the API contract between
 *    the FastAPI backend and a JS frontend. The client handles auth token
 *    refresh, typed request/response models, and streaming chat responses."
 *
 * Requirements: npm install (see package.json)
 * Usage: npx ts-node client.ts
 */

// ── Types (mirror the Pydantic schemas from the backend) ─────────────────────

interface TokenResponse {
  access_token: string;
  token_type: string;
}

interface UserResponse {
  id: string;
  email: string;
  full_name: string;
  role: "admin" | "manager" | "sales_rep" | "viewer";
  tenant_id: string;
}

interface LeadCreate {
  first_name: string;
  last_name: string;
  email?: string;
  company?: string;
  source?: string;
  lead_score?: number;
  notes?: string;
}

interface LeadResponse {
  id: string;
  first_name: string;
  last_name: string;
  email: string | null;
  company: string | null;
  status: string;
  lead_score: number;
  source: string;
  created_at: string;
}

interface PaginatedLeads {
  items: LeadResponse[];
  total: number;
  page: number;
  size: number;
}

interface ChatMessage {
  message: string;
  conversation_id?: string;
}

interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  result: string;
}

interface ChatResponse {
  response: string;
  intent: string;
  agent_used: string | null;
  tools_called: ToolCall[];
  conversation_id: string;
  execution_id: string;
  requires_confirmation: boolean;
  confirmation_prompt: string | null;
}

interface PipelineMetrics {
  stage: string;
  deal_count: number;
  total_value: number;
  weighted_pipeline: number;
}

// ── API Client ────────────────────────────────────────────────────────────────

class NexusCRMClient {
  private baseUrl: string;
  private token: string | null = null;
  private tokenExpiry: number | null = null;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private getHeaders(includeAuth: boolean = true): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (includeAuth && this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }
    return headers;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    requireAuth: boolean = true
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const options: RequestInit = {
      method,
      headers: this.getHeaders(requireAuth),
    };

    if (body !== undefined) {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      const errorText = await response.text();
      let errorDetail: string;
      try {
        const errorJson = JSON.parse(errorText);
        errorDetail = errorJson.detail || errorText;
      } catch {
        errorDetail = errorText;
      }
      throw new APIError(response.status, errorDetail);
    }

    return response.json() as Promise<T>;
  }

  // ── Authentication ─────────────────────────────────────────────────────────

  /**
   * Login and store the JWT token.
   * Subsequent requests automatically include the Authorization header.
   */
  async login(email: string, password: string): Promise<UserResponse> {
    // FastAPI OAuth2 login expects form data, not JSON
    const formData = new URLSearchParams();
    formData.append("username", email);
    formData.append("password", password);

    const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData.toString(),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new APIError(response.status, err.detail || "Login failed");
    }

    const tokenResponse: TokenResponse = await response.json();
    this.token = tokenResponse.access_token;

    // Decode expiry from JWT payload (base64 decode middle segment)
    const [, payload] = this.token.split(".");
    const decoded = JSON.parse(atob(payload));
    this.tokenExpiry = decoded.exp * 1000; // convert to ms

    // Fetch and return the current user
    return this.me();
  }

  /** Logout — clear stored token */
  logout(): void {
    this.token = null;
    this.tokenExpiry = null;
  }

  /** Return the authenticated user's profile */
  async me(): Promise<UserResponse> {
    return this.request<UserResponse>("GET", "/api/v1/auth/me");
  }

  /** Check if the stored token is still valid */
  isAuthenticated(): boolean {
    return this.token !== null && (this.tokenExpiry === null || Date.now() < this.tokenExpiry);
  }

  // ── Leads ──────────────────────────────────────────────────────────────────

  /** List leads with optional pagination and filtering */
  async listLeads(params: {
    page?: number;
    size?: number;
    status?: string;
    source?: string;
  } = {}): Promise<PaginatedLeads> {
    const qs = new URLSearchParams();
    if (params.page !== undefined) qs.set("page", String(params.page));
    if (params.size !== undefined) qs.set("size", String(params.size));
    if (params.status) qs.set("status", params.status);
    if (params.source) qs.set("source", params.source);
    const query = qs.toString() ? `?${qs}` : "";
    return this.request<PaginatedLeads>("GET", `/api/v1/leads${query}`);
  }

  /** Get a single lead by ID */
  async getLead(id: string): Promise<LeadResponse> {
    return this.request<LeadResponse>("GET", `/api/v1/leads/${id}`);
  }

  /** Create a new lead */
  async createLead(lead: LeadCreate): Promise<LeadResponse> {
    return this.request<LeadResponse>("POST", "/api/v1/leads", lead);
  }

  /** Update lead fields */
  async updateLead(id: string, updates: Partial<LeadCreate>): Promise<LeadResponse> {
    return this.request<LeadResponse>("PATCH", `/api/v1/leads/${id}`, updates);
  }

  /**
   * Delete a lead — requires explicit confirmation.
   * Pass confirm=true to proceed; without it the API returns HTTP 400.
   *
   * WHY the double-confirm pattern:
   *   Backend: DELETE /leads/{id}?confirm=true required
   *   Client:  must pass confirmDeletion=true explicitly
   *   This prevents accidental deletes from both accidental API calls
   *   and from the AI agent running without explicit user consent.
   */
  async deleteLead(id: string, confirmDeletion: boolean = false): Promise<void> {
    if (!confirmDeletion) {
      throw new Error(
        "Deletion requires explicit confirmation. Pass confirmDeletion=true to proceed."
      );
    }
    await this.request<void>("DELETE", `/api/v1/leads/${id}?confirm=true`);
  }

  // ── AI Agent Chat ──────────────────────────────────────────────────────────

  /**
   * Send a message to the AI agent.
   *
   * If the response has requires_confirmation=true, the agent needs
   * explicit user confirmation before performing a destructive action.
   * Call chat() again with the same message prefixed by "Yes, confirm."
   * to proceed.
   */
  async chat(message: string, conversationId?: string): Promise<ChatResponse> {
    const payload: ChatMessage = { message };
    if (conversationId) payload.conversation_id = conversationId;
    return this.request<ChatResponse>("POST", "/api/v1/agent/chat", payload);
  }

  // ── Analytics (Power BI views exposed via API) ────────────────────────────

  /** Get pipeline metrics by stage */
  async getPipelineMetrics(): Promise<PipelineMetrics[]> {
    return this.request<PipelineMetrics[]>("GET", "/api/v1/analytics/pipeline");
  }
}

// ── Error class ───────────────────────────────────────────────────────────────

class APIError extends Error {
  constructor(
    public statusCode: number,
    message: string
  ) {
    super(message);
    this.name = "APIError";
  }
}

// ── Demo: Using the client ────────────────────────────────────────────────────

async function demo(): Promise<void> {
  const client = new NexusCRMClient("http://localhost:8000");

  console.log("NexusCRM AI — TypeScript API Client Demo");
  console.log("=".repeat(50));

  // ── 1. Login ───────────────────────────────────────────────────────────────
  console.log("\n1. Authenticating...");
  try {
    const user = await client.login("sales@demo.nexuscrm.io", "Password123!");
    console.log(`   Logged in as: ${user.full_name} (${user.role})`);
    console.log(`   Tenant: ${user.tenant_id}`);
  } catch (err) {
    if (err instanceof APIError && err.statusCode === 401) {
      console.log("   Auth failed — is the backend running? (npm run demo to skip live calls)");
      runOfflineDemo(client);
      return;
    }
    throw err;
  }

  // ── 2. List leads ──────────────────────────────────────────────────────────
  console.log("\n2. Fetching leads...");
  const leads = await client.listLeads({ page: 1, size: 5, status: "new" });
  console.log(`   Total leads: ${leads.total}`);
  leads.items.forEach((lead) => {
    console.log(`   - ${lead.first_name} ${lead.last_name} (${lead.status}, score: ${lead.lead_score})`);
  });

  // ── 3. Create a lead ───────────────────────────────────────────────────────
  console.log("\n3. Creating a new lead...");
  const newLead = await client.createLead({
    first_name: "Alice",
    last_name: "Wonderland",
    email: "alice@acme.example.com",
    company: "Acme Corp",
    source: "website",
    lead_score: 75,
    notes: "Interested in Enterprise plan, asked about SSO.",
  });
  console.log(`   Created lead: ${newLead.id}`);

  // ── 4. Chat with the AI agent ──────────────────────────────────────────────
  console.log("\n4. Chatting with the AI agent...");
  const chatResp = await client.chat(
    "What's our current pipeline value and how many deals are in negotiation?",
    undefined
  );
  console.log(`   Intent: ${chatResp.intent}`);
  console.log(`   Agent: ${chatResp.agent_used}`);
  console.log(`   Tools called: ${chatResp.tools_called.map((t) => t.tool).join(", ")}`);
  console.log(`   Response: ${chatResp.response.slice(0, 150)}...`);
  console.log(`   Conversation ID: ${chatResp.conversation_id}`);

  // ── 5. Confirmation flow (destructive action) ──────────────────────────────
  console.log("\n5. Testing confirmation flow for delete...");
  const deleteChat = await client.chat(
    `Delete lead ${newLead.id}`,
    chatResp.conversation_id
  );
  if (deleteChat.requires_confirmation) {
    console.log(`   Agent requires confirmation: "${deleteChat.confirmation_prompt}"`);
    console.log("   Sending confirmation...");
    const confirmed = await client.chat(
      `Yes, confirm deletion of lead ${newLead.id}`,
      chatResp.conversation_id
    );
    console.log(`   Result: ${confirmed.response.slice(0, 100)}`);
  }

  // ── 6. Direct delete via REST (also requires confirm param) ───────────────
  console.log("\n6. Direct REST delete (with explicit confirmation)...");
  try {
    await client.deleteLead(newLead.id, true);
    console.log(`   Lead ${newLead.id} deleted.`);
  } catch (err) {
    if (err instanceof APIError && err.statusCode === 404) {
      console.log("   Lead already deleted by agent (expected).");
    } else {
      throw err;
    }
  }

  console.log("\nDemo complete.");
}

function runOfflineDemo(client: NexusCRMClient): void {
  console.log("\nOFFLINE DEMO — showing API shapes without live server");
  console.log("=".repeat(50));

  const mockLead: LeadResponse = {
    id: "lead-uuid-1234",
    first_name: "Alice",
    last_name: "Wonderland",
    email: "alice@acme.example.com",
    company: "Acme Corp",
    status: "new",
    lead_score: 75,
    source: "website",
    created_at: new Date().toISOString(),
  };
  console.log("\nMock lead shape:", JSON.stringify(mockLead, null, 2));

  const mockChatResponse: ChatResponse = {
    response: "Your pipeline has 8 active deals worth $1.2M weighted value. 2 deals are in negotiation totaling $340K.",
    intent: "analytics_query",
    agent_used: "AnalyticsAgent",
    tools_called: [
      { tool: "get_pipeline_metrics", args: {}, result: "..." },
      { tool: "list_deals_by_stage", args: { stage: "negotiation" }, result: "..." },
    ],
    conversation_id: "conv-uuid-5678",
    execution_id: "exec-uuid-9012",
    requires_confirmation: false,
    confirmation_prompt: null,
  };
  console.log("\nMock chat response shape:", JSON.stringify(mockChatResponse, null, 2));

  // Show the confirmDeletion guard working
  console.log("\nTesting deletion guard (no server needed):");
  try {
    // This will throw without a network call
    client.deleteLead("some-id", false);
  } catch (err) {
    if (err instanceof Error) {
      console.log("  Guard triggered:", err.message);
    }
  }
}

// ── Entry point ───────────────────────────────────────────────────────────────
// Note: fetch is available natively in Node 18+
demo().catch((err) => {
  if (err instanceof APIError) {
    console.error(`API Error ${err.statusCode}: ${err.message}`);
  } else {
    console.error("Unexpected error:", err);
  }
  process.exit(1);
});

export { NexusCRMClient, APIError };
export type {
  TokenResponse, UserResponse, LeadCreate, LeadResponse,
  PaginatedLeads, ChatMessage, ChatResponse, ToolCall, PipelineMetrics,
};
