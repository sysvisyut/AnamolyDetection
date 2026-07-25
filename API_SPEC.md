# API_SPEC.md
# AI-Powered Behavioral Anomaly Detection — API Specification

> **Status:** Phase 9 — Frozen API Design Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** ARCHITECTURE.md, DATA_SCHEMA.md, TECH_STACK.md, EVAL_METRICS.md  
> **Scope:** Defines REST endpoints, request/response payload structures, error handling, pagination, ingestion architecture, and streaming readiness. No implementation code. This is the contract for the dashboard (Phase 10) and backend implementation.

---

## Table of Contents

1. [Architecture Consistency Check](#1-architecture-consistency-check)
2. [Endpoint Inventory](#2-endpoint-inventory)
3. [Pydantic Model Definitions](#3-pydantic-model-definitions)
4. [Error Handling Conventions](#4-error-handling-conventions)
5. [Pagination, Filtering, and Sorting Conventions](#5-pagination-filtering-and-sorting-conventions)
6. [Ingestion Processing Model](#6-ingestion-processing-model)
7. [Streaming-Readiness Statement](#7-streaming-readiness-statement)
8. [Alternatives Considered](#8-alternatives-considered)
9. [Judging-Criteria Traceability](#9-judging-criteria-traceability)

---

## 1. Architecture Consistency Check

Prior to API design, `DATA_SCHEMA.md` and `TECH_STACK.md` were reviewed to ensure seamless integration:

- ✅ **Framework Versions:** The API uses FastAPI (0.111.x) and Pydantic (2.7.x) as required by `TECH_STACK.md` Decision 7.
- ✅ **Object Reuse:** The design strictly reuses the objects defined in `DATA_SCHEMA.md` (e.g., `AlertPayload`, `EntityHistoryEntry`, `FeatureAttribution`, `RenderedAlertData`).
- 🚨 **Flag - RiskScore Definition:** The prompt referenced a distinct `RiskScore` object. However, `DATA_SCHEMA.md` §5a defines `risk_score` as a primitive `int` (`[0, 100]`), and `risk_tier` as a primitive `str`. I will adhere to the `DATA_SCHEMA.md` definition and use primitives, avoiding the introduction of a new ad-hoc `RiskScore` struct.

---

## 2. Endpoint Inventory

All endpoints are mounted under the `/api/v1` prefix.

### 2.1 Ingestion
- **POST `/api/v1/events/ingest`**
  - **Purpose:** Receives new access-log events (single or batch) from external systems.
  - **Request Body:** `IngestRequest` (list of `RawAccessLog`).
  - **Response Body:** `{"status": "accepted", "count": int}`.
  - **Status Codes:** `202 Accepted` (success), `422 Unprocessable Entity` (validation error).
  - **Orchestration:** Validates payload synchronously, then hands off to `BackgroundTasks` to execute the ML pipeline (Feature Engineering → BPM/SDM → Classifier → Alert Store).

### 2.2 Alert Queue
- **GET `/api/v1/alerts`**
  - **Purpose:** Retrieves a ranked, paginated, and filterable queue of alerts for the SOC dashboard.
  - **Query Params:** `page`, `page_size`, `risk_tier`, `attack_class`, `entity_id`, `since`, `until`.
  - **Response Body:** `RenderedAlertData` (defined in `DATA_SCHEMA.md` §5e).
  - **Status Codes:** `200 OK`, `400 Bad Request` (invalid filter ranges).
  - **Orchestration:** Queries the Alert Store via SQLite backend.

### 2.3 Single Alert Detail
- **GET `/api/v1/alerts/{alert_id}`**
  - **Purpose:** Retrieves the full payload of a single alert, including feature attributions, raw event snapshot, and recent entity history.
  - **Response Body:** `AlertDetail` (defined in `DATA_SCHEMA.md` §5e).
  - **Status Codes:** `200 OK`, `404 Not Found`.
  - **Orchestration:** Queries the Alert Store for the alert, and the Profile Store for entity context.

### 2.4 Entity Context & History
- **GET `/api/v1/entities/{entity_id}/history`**
  - **Purpose:** Retrieves the chronological timeline of events for a specific entity.
  - **Query Params:** `limit` (default 50).
  - **Response Body:** `list[EntityHistoryEntry]`.
  - **Status Codes:** `200 OK`, `404 Not Found`.
  - **Orchestration:** Queries the Alert & Result Store for historical events tied to the entity.

- **GET `/api/v1/entities/{entity_id}/status`**
  - **Purpose:** Surfaces cold-start and drift status for the entity.
  - **Response Body:** `EntityStatusResponse` (derived from `EntityProfile` fields: `cold_start_flag`, `drift_metrics`, `profile_version`).
  - **Status Codes:** `200 OK`, `404 Not Found`.
  - **Orchestration:** Queries the Profile Store.

### 2.5 Evaluation Metrics
- **GET `/api/v1/metrics`**
  - **Purpose:** Feeds the dashboard's analytics view with data required by `EVAL_METRICS.md` (e.g., PR-AUC, Precision@1%, confusion matrix, daily drift FPR).
  - **Query Params:** `time_window` (default 30d).
  - **Response Body:** `SystemMetricsResponse`.
  - **Status Codes:** `200 OK`.
  - **Orchestration:** Invokes the Evaluation Module against the Alert Store and Ground-Truth labels.

### 2.6 Streaming Push (Reserved)
- **GET `/api/v1/stream/alerts`**
  - **Purpose:** Server-Sent Events (SSE) endpoint pushing new alerts to the dashboard live.
  - **Response Body:** `text/event-stream` stream of `AlertSummary` JSON objects.
  - **Status Codes:** `200 OK`.

---

## 3. Pydantic Model Definitions

All Pydantic models strictly reflect `DATA_SCHEMA.md`. No new fields are invented.

**Core Inputs:**
- `RawAccessLog`: Mirrors `DATA_SCHEMA.md` §2a. Includes nested `GeoLocation`, `DeviceFingerprint`, and `CommandEntry`. Used for ingestion payload validation.

**Core Outputs:**
- `FeatureAttribution`: Mirrors `DATA_SCHEMA.md` §5d.
- `EntityHistoryEntry`: Mirrors `DATA_SCHEMA.md` §5c.
- `AlertSummary`: Lightweight alert model matching `DATA_SCHEMA.md` §5e (alert queue table view).
- `AlertDetail`: Full payload matching `DATA_SCHEMA.md` §5e (drill-down view).
- `RenderedAlertData`: Paginated wrapper containing `list[AlertSummary]`, `total_count`, `page`, `page_size`.

**Specific Endpoint Response Models:**
- `EntityStatusResponse`: 
  - `entity_id` (str)
  - `is_cold_start` (bool)
  - `profile_version` (int)
  - `drift_severity` (str: "none", "low", "medium", "high")
  - `last_drift_check` (str ISO-8601)
- `SystemMetricsResponse`:
  - `pr_auc` (float)
  - `precision_at_1_pct` (float)
  - `confusion_matrix` (list[list[int]])
  - `daily_fp_rate` (list[float])

---

## 4. Error Handling Conventions

All API errors follow a standardized JSON structure to ensure predictable parsing by the dashboard UI.

**Standard Error Shape:**
```json
{
  "error_code": "ERR_VALIDATION_FAILED",
  "detail": "Input failed schema validation.",
  "context": {
    "field": "geo_location.latitude",
    "message": "Value must be between -90.0 and 90.0"
  }
}
```

**HTTP Status Code Conventions:**
- `200 OK`: Successful read operations.
- `202 Accepted`: Successful async ingestion (processing deferred).
- `400 Bad Request`: Business logic violations (e.g., `until` timestamp before `since` timestamp).
- `404 Not Found`: Resource (`alert_id` or `entity_id`) does not exist.
- `422 Unprocessable Entity`: Handled automatically by Pydantic; mapped to the standard error shape via a custom exception handler.
- `500 Internal Server Error`: Unhandled exceptions, mapped to a generic `ERR_INTERNAL` code to prevent leaking stack traces.

---

## 5. Pagination, Filtering, and Sorting Conventions

The `GET /api/v1/alerts` endpoint uses query parameters to manipulate the queue.

**Pagination:**
- Uses offset-based pagination: `page` (1-indexed, default 1) and `page_size` (default 50).
- Response includes `total_count` to allow the dashboard to render page numbers.

**Filtering:**
- `risk_tier`: Supports multiple values comma-separated (e.g., `?risk_tier=high,critical`).
- `attack_class`: Supports multiple values (e.g., `?attack_class=brute_force,credential_stuffing`).
- `entity_id`: Exact string match.
- `since` / `until`: ISO-8601 UTC strings.

**Sorting (Fixed):**
- As dictated by `DATA_SCHEMA.md` §5e, sorting is rigidly fixed to guarantee the operational alert budget is respected:
- **Primary Sort:** `risk_score` DESCENDING.
- **Secondary Sort:** `timestamp` DESCENDING.
- Custom sorting by users is disabled at the API level to enforce triage discipline (highest risk must be processed first).

---

## 6. Ingestion Processing Model

**Architecture:** Asynchronous Background Processing
- The `POST /api/v1/events/ingest` endpoint accepts the payload, validates it via Pydantic, and immediately returns a `202 Accepted` response.
- The actual inference pipeline (Feature Engineering → BPM/SDM → Classifier) is dispatched using FastAPI's native `BackgroundTasks`.

**Justification for Near Real-Time Requirement:**
A synchronous pipeline would block the API thread during inference (especially the ~20ms SDM Captum attribution step). For a batch of 100 events, this would cause a 2-second timeout. By decoupling ingestion from inference via background tasks, the API latency remains sub-10ms. Events are queued and processed asynchronously, appearing in the alert queue within milliseconds (meeting the "near real-time" definition) without crashing the ingest listener under high load.

---

## 7. Streaming-Readiness Statement

The architecture supports the Tier 2 streaming requirement (Phase 12) via **Server-Sent Events (SSE)**.
- **Mechanism:** The `GET /api/v1/stream/alerts` endpoint is reserved to return a `StreamingResponse` with media type `text/event-stream`.
- **Implementation:** `TECH_STACK.md` Decision 7 designated FastAPI + Uvicorn with the `standard` extras, natively enabling SSE support without WebSockets or third-party brokers.
- **Data Flow:** When a background task finishes processing an event and stores a new alert, it will emit a signal to an `asyncio.Queue` or `asyncio.Event` listener. The SSE endpoint consumes this queue and pushes lightweight `AlertSummary` JSON objects to the connected browser.
- **Redesign impact:** Zero. This is entirely additive and relies entirely on native FastAPI/Python async features.

---

## 8. Alternatives Considered

1. **Synchronous Inference vs. Background Tasks**
   - *Considered:* Running the pipeline inside the API route handler before returning a `200 OK`.
   - *Chosen:* `BackgroundTasks` with `202 Accepted`. Synchronous processing is easier to debug but fails scalability tests immediately. The background task approach provides the non-blocking behavior needed for realistic ingestion.
2. **WebSockets vs. Server-Sent Events (SSE)**
   - *Considered:* Implementing WebSockets for the live alert feed.
   - *Chosen:* SSE. The data flow for alerts is strictly unidirectional (Server → Dashboard). WebSockets introduce unnecessary complexity (two-way handshakes, custom reconnection logic). SSE is natively supported by the browser `EventSource` API and HTTP/1.1, keeping the frontend implementation purely vanilla JS.
3. **Cursor-based vs. Offset-based Pagination**
   - *Considered:* Cursor pagination for better performance on deep pages.
   - *Chosen:* Offset pagination (`page`, `page_size`). Analysts rarely paginate past page 2 or 3 of an alert queue (they filter instead). Offset pagination easily provides the `total_count` needed to show queue size on the dashboard, which is more valuable than deep-pagination performance.

---

## 9. Judging-Criteria Traceability

| Hackathon Judging Criterion | API Design Decision Providing Evidence |
|-----------------------------|----------------------------------------|
| **System design and scalability** | Async background task ingestion prevents thread blocking; SSE endpoint proves real-time streaming capability without heavy brokers. |
| **Modular, production-quality architecture** | Strict reliance on Pydantic validation; standardized error envelope; HTTP 202 async pattern; offset pagination; separation of concern between queue retrieval (summary) and drill-down (detail). |
| **Explainability and analyst usability** | The API is tailored to the dashboard's needs: separate endpoints for the ranked summary queue vs. heavy explanation payloads ensure the dashboard loads instantly. |
