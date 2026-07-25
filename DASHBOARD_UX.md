# DASHBOARD_UX.md
# AI-Powered Behavioral Anomaly Detection — Dashboard UX Specification

> **Status:** Phase 10 — Frozen Dashboard UX Design  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** API_SPEC.md, DATA_SCHEMA.md, EXPLAINABILITY.md, TECH_STACK.md, EVAL_METRICS.md  
> **Scope:** Defines the visual layout, interaction flows, and API integrations for the frontend SOC analyst dashboard. No implementation code. 

---

## Table of Contents

1. [Architecture Consistency Check](#1-architecture-consistency-check)
2. [Analyst Persona & Primary User Journey](#2-analyst-persona--primary-user-journey)
3. [View-by-View Specification](#3-view-by-view-specification)
   - 3a. [Alert Queue View](#3a-alert-queue-view)
   - 3b. [Alert Detail View](#3b-alert-detail-view)
   - 3c. [Entity History / Timeline View](#3c-entity-history--timeline-view)
   - 3d. [Evaluation Metrics View](#3d-evaluation-metrics-view)
4. [Ambiguity Visual Treatment (Insider Drift)](#4-ambiguity-visual-treatment-insider-drift)
5. [Cold-Start & Drift Visibility](#5-cold-start--drift-visibility)
6. [Real-Time Behavior](#6-real-time-behavior)
7. [Alternatives Considered](#7-alternatives-considered)
8. [Judging-Criteria Traceability](#8-judging-criteria-traceability)

---

## 1. Architecture Consistency Check

A thorough review was conducted against `API_SPEC.md` to ensure every data element planned for the UX is supported by an existing endpoint.

| UI Element | Source Endpoint in `API_SPEC.md` | Result |
|------------|----------------------------------|--------|
| Alert Queue List | `GET /api/v1/alerts` (`RenderedAlertData`) | ✅ Pass |
| Single Alert Details (Attributions) | `GET /api/v1/alerts/{alert_id}` (`AlertDetail`) | ✅ Pass |
| Entity History | `GET /api/v1/entities/{entity_id}/history` | ✅ Pass |
| Evaluation Metrics / Charts | `GET /api/v1/metrics` (`SystemMetricsResponse`) | ✅ Pass |
| Cold-Start / Drift Indicators | `GET /api/v1/entities/{entity_id}/status` | ✅ Pass |
| Live Updates | `GET /api/v1/stream/alerts` (SSE) | ✅ Pass |

**Consistency verdict:** No gaps found. Every required data point, including specific requirements for ambiguity, cold-start, and concept drift, is already fully typed and exposed by the backend API. No schema or API extensions are required.

---

## 2. Analyst Persona & Primary User Journey

### Analyst Persona: "Tier 1 SOC Analyst"
- **Goal:** Quickly triage incoming alerts, distinguish false positives from true threats, and escalate genuine anomalies to incident response.
- **Pain Points:** Alert fatigue, lack of context on what an entity *usually* does, and opaque "black box" ML risk scores that provide no reasoning.
- **Needs:** Clear prioritization (ranked by severity), plain-English explanations of *why* an alert fired, and immediate access to the entity's history without opening a new tool.

### Primary User Journey
1. **Scan:** The analyst opens the **Alert Queue View**, which automatically ranks alerts by `risk_score` descending.
2. **Preview:** They scan the top alert's 150-character `human_readable_explanation` directly in the queue table.
3. **Investigate:** Clicking the alert slides open the **Alert Detail View** (side panel or expanded row) showing the full attribution breakdown.
4. **Contextualize:** Within the detail view, they click "View Entity Context", which loads the **Entity History / Timeline View** alongside the entity's **Cold-Start / Drift Status**.
5. **Decide:** Using the explanation and history, the analyst determines if this is a true threat, benign drift, or an ambiguous edge case.

---

## 3. View-by-View Specification

The dashboard is a Single Page Application (SPA) utilizing a multi-pane layout to avoid context switching. 

### 3a. Alert Queue View
- **Layout:** A dense, full-width data table occupying the main screen space.
- **Key Elements:**
  - **Severity Badge:** Color-coded `risk_tier` (Critical: Red, High: Orange, Medium: Yellow, Low: Gray).
  - **Risk Score:** Progress-bar style column for `risk_score` (0-100) for rapid visual scanning.
  - **Entity ID & Class:** The `entity_id` and predicted `attack_class`.
  - **Explanation Snippet:** The truncated `human_readable_explanation` from `AlertSummary`.
- **Interaction:** Clicking a row opens the Alert Detail View. A top-bar provides filters (`attack_class`, `risk_tier`, `time_range`).
- **Endpoint Consumed:** `GET /api/v1/alerts`

### 3b. Alert Detail View
- **Layout:** A slide-out right drawer (taking ~40% of screen width) that overlays the queue.
- **Key Elements:**
  - **Header:** Risk Score, Attack Class, Timestamp, and Entity ID.
  - **Narrative Box:** The full, unabridged `human_readable_explanation`. Highlighted with a light background to draw the eye first.
  - **Feature Attribution List:** A vertical list of `feature_attributions`. Each feature displays its `human_label`, actual `feature_value`, and a horizontal bar chart showing the `attribution_score` magnitude (red for "toward anomaly", green/gray for "toward normal").
  - **Raw Payload Toggle:** An accordion to view the `raw_event_snapshot` JSON for deep technical drill-down.
- **Interaction:** "View Entity History" button at the bottom of the drawer.
- **Endpoint Consumed:** `GET /api/v1/alerts/{alert_id}`

### 3c. Entity History / Timeline View
- **Layout:** A modal or dedicated sub-page showing a vertical chronological timeline.
- **Key Elements:**
  - **Timeline:** A vertical line charting the last 50 events (`EntityHistoryEntry`). Events that generated alerts are marked with red/orange nodes; normal events are gray nodes.
  - **Status Header:** Displays the output of `EntityStatusResponse`. Includes badges for "New Entity (Cold-Start)" or "Profile Recently Adapted (Concept Drift)".
- **Interaction:** Hovering over a timeline node reveals the resource accessed and auth outcome.
- **Endpoints Consumed:** `GET /api/v1/entities/{entity_id}/history` and `GET /api/v1/entities/{entity_id}/status`

### 3d. Evaluation Metrics View
- **Layout:** A dedicated "System Analytics" tab separating operational triage from system performance.
- **Key Elements (Powered by Chart.js):**
  - **Performance Header:** PR-AUC and Precision@1% headline numbers.
  - **Confusion Matrix:** Heatmap showing attack classifications.
  - **Drift Adaptation Chart:** A line chart plotting the `daily_fp_rate` for drifted entities vs. stable entities over the 30-day window.
- **Endpoint Consumed:** `GET /api/v1/metrics`

---

## 4. Ambiguity Visual Treatment (Insider Drift)

`EXPLAINABILITY.md` established that Insider Drift represents an intentionally ambiguous, medium-risk overlap with normal behavior. The UI must reflect this explicitly to prevent analysts from treating it as a hard false-positive.

- **Visual Badge:** Alerts classified as `insider_drift` receive a distinct "purple" (or neutral) styling for their `attack_class` badge, contrasting with the stark red/orange of brute force or lateral movement.
- **Explanation Styling:** The narrative box for these alerts prefixes the text with an explicit "⚠️ **Ambiguity Flag:**", mirroring the logic in `EXPLAINABILITY.md` (e.g., "Ambiguity Flag: Entity behavior has drifted significantly...").
- **Risk Score Warning:** If an Insider Drift alert has a lower confidence score (e.g., `< 0.6`), a tooltip on the risk score explicitly states: *"System has detected profile drift, but signals are below threshold for malicious lateral movement. Review recommended."*

---

## 5. Cold-Start & Drift Visibility

To visibly prove to judges that the system handles edge cases, the dashboard makes backend ML state highly visible in the UI.

- **Cold-Start Indicator:** 
  - In the Alert Queue and Alert Detail views, if `cold_start_flag == True`, an unmistakable "❄️ Cold-Start Entity" badge is rendered next to the `entity_id`. 
  - Tooltip: *"Entity has insufficient history (< 10 events). Scored against global population baseline."*
- **Concept Drift Indicator:**
  - In the Entity History view, the status header displays the `drift_severity` from `EntityStatusResponse`.
  - If `drift_severity` is "medium" or "high", an "🔄 Profile Adapted" badge appears.
  - Tooltip: *"System detected legitimate behavioral drift on [Date]. The baseline profile was updated (Version: [X]) without raising persistent alerts."*

---

## 6. Real-Time Behavior

The dashboard operates in a **Live Feed** paradigm, explicitly leveraging the streaming-readiness built into `API_SPEC.md`.

- **Mechanism:** The frontend uses the native browser `EventSource` API to connect to the `GET /api/v1/stream/alerts` Server-Sent Events (SSE) endpoint.
- **UX Behavior:** As new alerts are pushed from the backend, they are unshifted (inserted at the top) into the Alert Queue table in real-time. New rows receive a brief CSS flash animation (e.g., fading yellow background) to draw the analyst's attention to the live ingestion.
- **Why SSE over Polling:** SSE provides a continuous, low-latency push without the network overhead of `setInterval` polling, perfectly matching the "near real-time" problem statement requirement and demonstrating actual streaming capability to the judges.

---

## 7. Alternatives Considered

1. **Dashboard Layout: Single-Page Dense vs. Multi-Page Drill-Down**
   - *Considered:* Separating the Queue, Alert Details, and Entity History into distinct page URLs (`/queue`, `/alert/123`, `/entity/456`).
   - *Chosen:* Single-Page Dense (Slide-out drawer for details). Security analysts need rapid context without losing their place in the queue. Page navigation breaks triage flow. The slide-out drawer pattern (used by modern tools like Datadog and Splunk) is far superior for this persona.
2. **Alert Queue Format: Card Grid vs. Dense Table**
   - *Considered:* Rendering alerts as stylized cards in a grid.
   - *Chosen:* Dense Table. Cards waste vertical space. An analyst needs to scan 20-50 rows per screen to assess priority effectively. A dense table allows rapid sorting and scanning by risk score and category.
3. **Data Visualization: Heavy Custom D3.js vs. Chart.js**
   - *Considered:* Using D3.js for highly bespoke timeline and metric visualizations.
   - *Chosen:* Chart.js. `TECH_STACK.md` locked vanilla JS. Chart.js is easily imported via CDN, provides out-of-the-box responsive line/bar charts, and requires a fraction of the code compared to D3.js, fitting the hackathon time constraint while still satisfying the `EVAL_METRICS.md` chart requirements.

---

## 8. Judging-Criteria Traceability

| Hackathon Judging Criterion | UX Design Decision Providing Evidence |
|-----------------------------|----------------------------------------|
| **Explainability and analyst usability** | The slide-out Alert Detail view prioritizes the human-readable explanation and visualizes attribution magnitude, directly addressing the "black box" pain point. |
| **Handling cold-start entities** | Unmistakable "❄️ Cold-Start Entity" UI badge proves to judges that the system recognizes and correctly flags zero-history actors. |
| **Handling concept drift** | "🔄 Profile Adapted" badge in the Entity History view proves the backend is actively updating profiles without hiding the mechanics. |
| **System design and scalability** | Live-feed SSE integration with CSS flash animations provides a highly visible, visceral demonstration of "near real-time streaming capability." |
