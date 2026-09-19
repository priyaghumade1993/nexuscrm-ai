-- NexusCRM AI — Power BI SQL Views
--
-- These views expose pre-aggregated CRM metrics for Power BI dashboards.
-- Power BI connects directly to PostgreSQL using the ODBC connector.
--
-- WHY VIEWS INSTEAD OF DIRECT TABLE ACCESS:
--   1. Row-level security: views can enforce tenant_id filtering
--   2. Stable schema: underlying tables can change; views stay consistent
--   3. Performance: views can pre-join tables that Power BI would otherwise JOIN on every render
--   4. Computed metrics: Power BI's DAX can use raw numbers; SQL pre-computes the hard stuff
--
-- HOW TO USE:
--   Run this script against your PostgreSQL instance.
--   In Power BI Desktop: Get Data → PostgreSQL → connect to your host
--   Import the views as datasets.

-- ── 1. Pipeline Overview by Stage ────────────────────────────────────────────
-- Used for: funnel visualization, pipeline value bar chart
CREATE OR REPLACE VIEW vw_pipeline_by_stage AS
SELECT
    d.tenant_id,
    d.stage,
    COUNT(*)                        AS deal_count,
    SUM(d.amount)                   AS total_value,
    AVG(d.amount)                   AS avg_deal_value,
    AVG(d.probability)              AS avg_probability,
    SUM(d.amount * d.probability / 100) AS weighted_pipeline
FROM deals d
WHERE d.stage NOT IN ('closed_lost')
GROUP BY d.tenant_id, d.stage
ORDER BY
    CASE d.stage
        WHEN 'prospect'         THEN 1
        WHEN 'qualified'        THEN 2
        WHEN 'demo_scheduled'   THEN 3
        WHEN 'demo_completed'   THEN 4
        WHEN 'proposal_sent'    THEN 5
        WHEN 'negotiation'      THEN 6
        WHEN 'closed_won'       THEN 7
    END;

-- ── 2. Lead Source Performance ────────────────────────────────────────────────
-- Used for: which lead sources produce the most qualified leads
CREATE OR REPLACE VIEW vw_lead_source_performance AS
SELECT
    l.tenant_id,
    l.source,
    COUNT(*)                                                    AS total_leads,
    COUNT(*) FILTER (WHERE l.status IN ('qualified', 'hot'))   AS qualified_leads,
    AVG(l.lead_score)                                           AS avg_lead_score,
    COUNT(*) FILTER (WHERE l.status IN ('qualified', 'hot'))::FLOAT
        / NULLIF(COUNT(*), 0)                                   AS qualification_rate
FROM leads l
GROUP BY l.tenant_id, l.source;

-- ── 3. Monthly Deal Closures ──────────────────────────────────────────────────
-- Used for: monthly revenue trend line chart
CREATE OR REPLACE VIEW vw_monthly_closed_deals AS
SELECT
    d.tenant_id,
    DATE_TRUNC('month', d.updated_at)   AS close_month,
    COUNT(*)                             AS deals_closed,
    SUM(d.amount)                        AS revenue_closed,
    AVG(d.amount)                        AS avg_deal_size
FROM deals d
WHERE d.stage = 'closed_won'
GROUP BY d.tenant_id, DATE_TRUNC('month', d.updated_at)
ORDER BY close_month DESC;

-- ── 4. Sales Rep Performance ──────────────────────────────────────────────────
-- Used for: rep leaderboard table
CREATE OR REPLACE VIEW vw_rep_performance AS
SELECT
    d.tenant_id,
    u.full_name                          AS rep_name,
    u.email                              AS rep_email,
    COUNT(d.id)                          AS total_deals,
    COUNT(d.id) FILTER (WHERE d.stage = 'closed_won')  AS won_deals,
    COUNT(d.id) FILTER (WHERE d.stage = 'closed_lost') AS lost_deals,
    SUM(d.amount) FILTER (WHERE d.stage = 'closed_won') AS won_revenue,
    COUNT(d.id) FILTER (WHERE d.stage = 'closed_won')::FLOAT
        / NULLIF(COUNT(d.id) FILTER (WHERE d.stage IN ('closed_won', 'closed_lost')), 0)
                                         AS win_rate,
    COUNT(l.id)                          AS total_leads,
    AVG(l.lead_score)                    AS avg_lead_score
FROM users u
LEFT JOIN deals d  ON d.owner_id = u.id AND d.tenant_id = u.tenant_id
LEFT JOIN leads l  ON l.owner_id = u.id AND l.tenant_id = u.tenant_id
WHERE u.role IN ('sales_rep', 'manager')
GROUP BY d.tenant_id, u.id, u.full_name, u.email;

-- ── 5. Deals Closing This Month ───────────────────────────────────────────────
-- Used for: urgent deals card / alert
CREATE OR REPLACE VIEW vw_deals_closing_this_month AS
SELECT
    d.tenant_id,
    d.id                                 AS deal_id,
    d.name                               AS deal_name,
    c.company_name                       AS customer_name,
    u.full_name                          AS owner_name,
    d.stage,
    d.amount,
    d.probability,
    d.amount * d.probability / 100       AS weighted_value,
    d.expected_close_date,
    d.expected_close_date - CURRENT_DATE AS days_remaining
FROM deals d
LEFT JOIN customers c ON c.id = d.customer_id
LEFT JOIN users u ON u.id = d.owner_id
WHERE
    d.stage NOT IN ('closed_won', 'closed_lost')
    AND d.expected_close_date BETWEEN CURRENT_DATE
    AND DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month - 1 day'
ORDER BY d.expected_close_date ASC;

-- ── 6. Lead Funnel Summary ────────────────────────────────────────────────────
-- Used for: top-level KPI cards
CREATE OR REPLACE VIEW vw_lead_funnel_summary AS
SELECT
    tenant_id,
    COUNT(*)                                                    AS total_leads,
    COUNT(*) FILTER (WHERE status = 'new')                     AS new_leads,
    COUNT(*) FILTER (WHERE status IN ('contacted', 'warm'))    AS in_progress,
    COUNT(*) FILTER (WHERE status IN ('qualified', 'hot'))     AS qualified,
    COUNT(*) FILTER (WHERE status = 'cold')                    AS cold,
    AVG(lead_score)                                            AS avg_lead_score
FROM leads
GROUP BY tenant_id;

-- Grant read access to Power BI service account
-- GRANT SELECT ON vw_pipeline_by_stage TO powerbi_readonly;
-- GRANT SELECT ON vw_lead_source_performance TO powerbi_readonly;
-- GRANT SELECT ON vw_monthly_closed_deals TO powerbi_readonly;
-- GRANT SELECT ON vw_rep_performance TO powerbi_readonly;
-- GRANT SELECT ON vw_deals_closing_this_month TO powerbi_readonly;
-- GRANT SELECT ON vw_lead_funnel_summary TO powerbi_readonly;
