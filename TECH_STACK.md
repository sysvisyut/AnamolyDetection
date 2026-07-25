# TECH_STACK.md
# AI-Powered Behavioral Anomaly Detection — Technology Stack Decisions

> **Status:** Phase 3 — Frozen Technology Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** ARCHITECTURE.md v1.0, PROJECT_STRUCTURE.md v1.0, CODING_GUIDELINES.md v1.0  
> **Scope:** Technology decisions only. No implementation code, no schema details.  
> All future implementation phases consume this document alongside Phase 1 and Phase 2 outputs.

---

## Table of Contents

1. [Architecture Consistency Check](#1-architecture-consistency-check)
2. [Decision 1 — Language and Runtime](#2-decision-1--language-and-runtime)
3. [Decision 2 — Synthetic Data Generation](#3-decision-2--synthetic-data-generation)
4. [Decision 3 — Behavioral Profiling Model Framework](#4-decision-3--behavioral-profiling-model-framework)
5. [Decision 4 — Sequence Detection Framework](#5-decision-4--sequence-detection-framework)
6. [Decision 5 — Explainability Library](#6-decision-5--explainability-library)
7. [Decision 6 — Persistence and Storage](#7-decision-6--persistence-and-storage)
8. [Decision 7 — FastAPI Backend Stack](#8-decision-7--fastapi-backend-stack)
9. [Decision 8 — Dashboard / Frontend](#9-decision-8--dashboard--frontend)
10. [Decision 9 — Streaming Simulation Mechanism](#10-decision-9--streaming-simulation-mechanism)
11. [Decision 10 — Testing Framework](#11-decision-10--testing-framework)
12. [Decision 11 — Dev Tooling](#12-decision-11--dev-tooling)
13. [Compatibility Check](#13-compatibility-check)
14. [Explicitly Out of Scope](#14-explicitly-out-of-scope)

---

## 1. Architecture Consistency Check

Before proceeding, ARCHITECTURE.md v1.0 and PROJECT_STRUCTURE.md v1.0 were re-read in full and cross-referenced against each other. The following consistency properties were verified:

| Check | Result |
|-------|--------|
| Every ARCHITECTURE.md component has exactly one corresponding folder in PROJECT_STRUCTURE.md | ✅ Pass — 11 components, 11 packages |
| Every boundary (A–L) has a named producing and consuming file in CODING_GUIDELINES.md Section 5.1 | ✅ Pass |
| ARCHITECTURE.md's Streaming Attachment Point maps to `streaming/` with `batch_reader.py` (T1) and `simulated_stream.py` (T2) | ✅ Pass |
| ARCHITECTURE.md's Cold-Start Handler slot maps to `cold_start/` package (T2) | ✅ Pass |
| ARCHITECTURE.md's Drift Monitor slot maps to `drift/` package (T2) | ✅ Pass |
| ARCHITECTURE.md's Entity Profile Store and Alert & Result Store both map to `stores/` package with distinct files | ✅ Pass |
| The `dashboard/` is at `src/dashboard/` (a sibling of `anomaly_detection/`), consistent with it being a JavaScript SPA that communicates with the API only over HTTP | ✅ Pass — correctly positioned outside the Python package hierarchy |
| Label field stripping responsibility assigned to `streaming/` package, matching ARCHITECTURE.md Risk 3 mitigation | ✅ Pass |
| Tier 1 standalone guarantee in PROJECT_STRUCTURE.md Section 3.5 is consistent with the T1 boundary set in ARCHITECTURE.md Section 4 | ✅ Pass |

**One minor naming inconsistency noted and resolved:**  
CODING_GUIDELINES.md Section 1.6 contains a typo: boundary C's class name is listed as `EngineeeredFeatures` (three e's). The correct canonical name is `EngineeredFeatures`. This is a documentation typo, not a structural conflict. It must be corrected in `common/types.py` when that file is implemented in Phase 4. No structural change is required.

**Consistency verdict:** No conflicting assumptions exist between ARCHITECTURE.md and PROJECT_STRUCTURE.md. Proceeding to technology decisions.

---

## 2. Decision 1 — Language and Runtime

### Chosen Technology
**Python 3.11** (minimum), targeting **Python 3.12** as the standard runtime.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| Python 3.10 | Still receives security fixes, but misses `tomllib` in stdlib and has slower `asyncio` performance than 3.11+. The `match` statement syntax (useful for attack class routing) is available only from 3.10+, but 3.11 is universally available on all major CI/CD runners. |
| Python 3.13 | Too new as of the hackathon date; PyTorch stable wheels, SHAP, and several scikit-learn builds lag behind the latest CPython release. Risk of missing wheel availability for at least one dependency is unacceptable. |

### Rationale
Python 3.11 introduced a 10–60% interpreter speed improvement over 3.10 (CPython benchmark suite), which matters for online inference throughput. Python 3.12 is the current stable release with broad wheel coverage for all chosen ML libraries (scikit-learn, PyTorch, SHAP). The `pyproject.toml` will declare `requires-python = ">=3.11"` and CI will run against 3.12.

### Judging Criterion
Supports **system design and scalability**: using a modern, performance-improved runtime signals production-quality thinking without any cost. Supports **detection accuracy** indirectly: faster inference allows tighter demo loop times.

---

## 3. Decision 2 — Synthetic Data Generation

### Chosen Technologies
- **NumPy 1.26 / 2.x** — vectorized random sampling for timing distributions, IP address generation, numeric noise
- **pandas 2.x** — tabular assembly, schema enforcement, Parquet I/O
- **Faker 25.x** — realistic names, email addresses, geographic locations, device fingerprint strings

This is the exact combination named in the problem statement's "Synthetic Data Generator Recommendation." It is confirmed rather than modified.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **SDV (Synthetic Data Vault)** | Powerful conditional synthetic data library. Rejected because it is designed to mimic existing datasets statistically — this project defines the behavioral model from scratch with explicit attack injection logic. SDV's abstraction layer would obscure the behavioral assumptions that judges need to inspect. |
| **Gretel.ai / generative model-based synthesis** | Cloud-API-dependent, requires model training time, and adds privacy/infrastructure complexity. Rejected for all three reasons. |

### Rationale
The problem statement explicitly recommends NumPy + pandas + Faker, and this combination is the right tool for the task: the data schema is fully controlled, attack patterns are deterministically injected, and Parquet output gives the Feature Engineering layer an efficient binary format. No library is heavier than necessary for pure synthetic generation.

**Additional library for this component:** `pyarrow 15.x` — required for pandas Parquet I/O. No new component or folder is introduced; this is a library dependency of `data_generator/`.

### Judging Criterion
Supports **report clarity**: explicit behavioral assumptions coded in `entity_profiles.py` and `attack_injector.py` are directly documentable in the Technical Report. Supports **detection accuracy**: controlled injection rates allow precise precision/recall measurement at known anomaly frequencies.

---

## 4. Decision 3 — Behavioral Profiling Model Framework

### Chosen Technology
**scikit-learn 1.5.x**

Specifically, the BPM will use scikit-learn's statistical and one-class tooling. The exact model choice (e.g., `IsolationForest`, `OneClassSVM`, per-entity statistical z-scoring, or `sklearn`-compatible autoencoder wrapping) is deferred to the Phase 5 ML model selection phase, as required by ARCHITECTURE.md. The framework commitment is scikit-learn.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **PyOD (Python Outlier Detection)** | Excellent library with 40+ anomaly detectors including deep learning options. Rejected as the *primary* BPM framework because it adds a non-trivial dependency for T1 with limited additional benefit over scikit-learn's built-in isolation forest and one-class SVM — both of which the problem statement already names. PyOD is approved as an optional supplement in Phase 5 if the BPM selection phase warrants it. |
| **statsmodels** | Strong for statistical time-series and distribution fitting, but lacks the `fit/predict` sklearn-compatible interface that integrates cleanly with the `base.py` abstract interface and pipeline. Per-entity statistical profiling can be implemented with NumPy/SciPy without statsmodels. |

### Rationale
scikit-learn provides a unified `fit()` / `predict()` / `score_samples()` interface that maps cleanly onto the `base.py` abstract BPM interface defined in PROJECT_STRUCTURE.md. Its `Pipeline` and `ColumnTransformer` utilities are also used by the Feature Engineering encoders. Using one framework for both FE preprocessing and BPM inference reduces the dependency surface and avoids dtype conversion issues at boundary C. The `joblib` integration bundled with scikit-learn handles model artifact serialization to `models/behavioral_profiling/artifacts/` without an additional library.

### Judging Criterion
Supports **detection accuracy on imbalanced labels**: scikit-learn's one-class and density estimation models are specifically designed for settings where anomalies are rare — they learn normality, not a binary boundary. Supports **modular, production-quality architecture**: the uniform `fit/predict` API means the BPM implementation is swappable in Phase 5 without touching boundary C or F contracts.

---

## 5. Decision 4 — Sequence Detection Framework

### Chosen Technology
**PyTorch 2.3.x** (via `torch` package)

The Sequence Detection Model will be implemented using PyTorch. The specific model architecture (LSTM, GRU, Transformer encoder, or other) is deferred to the Phase 6 model selection phase as required by ARCHITECTURE.md Section 2. The framework commitment is PyTorch.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **TensorFlow 2.x / Keras** | Equally capable for sequence modeling. Rejected because the explainability library of choice (Captum — see Decision 5) is PyTorch-native and does not support TensorFlow models. Choosing TensorFlow would force either a different explainability approach or a two-framework installation, both of which increase dependency complexity without a compensating advantage. |
| **scikit-learn (MLPRegressor or similar)** | Could approximate sequence learning via fixed-window feature stacking, but does not natively support recurrent architectures or attention mechanisms. Excluded from SDM: the problem statement explicitly names LSTM/GRU/Transformer, and a scikit-learn MLP would not satisfy the "sequence-aware" criterion convincingly for judges. |
| **HuggingFace Transformers** | Pre-trained transformer models exist for sequence anomaly detection. Rejected: pre-trained models for tabular access-log sequences do not exist in a useful form; training a transformer from scratch adds infrastructure (tokenizers, attention masks) that scikit-learn/PyTorch from scratch avoids, and the time budget cannot support it reliably. |

### Rationale
PyTorch is chosen over TensorFlow primarily because the explainability choice (Captum) is architecturally dependent on it. This cross-component dependency between Decision 4 and Decision 5 is the most critical compatibility constraint in the stack. Both decisions are locked together: **if the SDM framework changes, the explainability framework must be re-evaluated simultaneously.** PyTorch 2.x also has significantly improved `torch.compile()` and `torch.export()` support that gives a credible path to production serving if the T3 Docker deployment is pursued.

### Judging Criterion
Supports **detection accuracy on imbalanced labels**: recurrent/attention architectures naturally encode temporal context, making them superior to independent-event models for sequence-pattern attacks (lateral movement, low-and-slow exfiltration). Supports **system design and scalability**: PyTorch's `torch.save()` / `torch.load()` produces portable artifacts and `torch.compile()` provides a production-grade inference path.

---

## 6. Decision 5 — Explainability Library

### Chosen Technology
**SHAP 0.45.x** as the primary attribution engine for the Behavioral Profiling Model (scikit-learn-compatible models).  
**Captum 0.7.x** as the attribution engine for the Sequence Detection Model (PyTorch-native).

Both libraries output feature importance scores in the same conceptual format (signed attribution values per feature). The `feature_attribution.py` file in `src/anomaly_detection/explainability/` will normalize outputs from both into the `feature_attributions[]` field of the boundary-I `AlertPayload`.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **LIME** | Model-agnostic, supports both sklearn and PyTorch models through a single library. Rejected because LIME's explanation quality is substantially lower than SHAP for tabular models (high variance between runs, no global consistency), and it does not provide gradient-based attribution for sequence models, which is specifically needed for LSTM/GRU explanations. LIME's computational cost for sequence models is also significantly higher (requires hundreds of perturbation samples per explanation). |
| **Custom rule-based attribution** | Fully transparent, no framework dependency. Rejected as the *primary* mechanism because the problem statement specifically requires "feature attribution" that is defensible to judges — rule-based attribution that does not reflect the model's actual learned weights would be indefensible in a Q&A. Rule-based narrative generation is retained as the responsibility of `explainability/narrative.py`, which produces the human-readable explanation *on top of* the SHAP/Captum attribution values. |
| **InterpretML (Microsoft)** | Supports Explainable Boosting Machines and glass-box models well. Rejected because it primarily supports its own model types; integration with arbitrary scikit-learn and PyTorch models requires significant custom adapters. |

### Rationale
Using two attribution libraries (SHAP for BPM, Captum for SDM) is the most natural fit given the framework split (scikit-learn / PyTorch). Both are mature, well-documented, and widely recognized by ML practitioners and judges. The normalization adapter in `feature_attribution.py` is a small, well-bounded piece of code that unifies the outputs. This is not a new component — it is the intended use of `feature_attribution.py` as defined in PROJECT_STRUCTURE.md.

**Critical compatibility note:** Captum requires PyTorch. SHAP requires NumPy and either scikit-learn or PyTorch. No version conflict exists when both are installed alongside PyTorch 2.3.x + scikit-learn 1.5.x + Python 3.12. (See Section 13 for full verification.)

### Judging Criterion
Directly serves **explainability and analyst usability** — the primary differentiating criterion for this problem. SHAP values have become the de facto standard for model attribution explanations in the ML community; judges will recognize them immediately. Captum's gradient-based attribution (e.g., Integrated Gradients) provides interpretable per-timestep importance for sequence models, directly addressing the "which factors contributed" requirement.

---

## 7. Decision 6 — Persistence and Storage

### Chosen Technologies

| Store | T1 Backend | T2/T3 Backend | File Format |
|-------|-----------|---------------|-------------|
| Generated raw data (`data/raw/`) | Parquet files (via `pyarrow`) | Same | `.parquet` |
| Ground-truth label store (`data/labeled/`) | Parquet files | Same | `.parquet` |
| Processed features (`data/processed/`) | Optional Parquet cache | Same | `.parquet` |
| Entity Profile Store (`data/profiles/`) | SQLite (`stores/backends/sqlite.py`) | Redis (T2, `stores/backends/redis.py`) | `.db` / in-memory |
| Alert & Result Store | SQLite | Redis (T2) | `.db` / in-memory |
| Model artifacts (`models/*/artifacts/`) | `joblib` (BPM, scikit-learn) + `torch.save()` (SDM) | Same | `.pkl` / `.pt` |

**SQLite** (via Python stdlib `sqlite3`) is the T1 persistent backend for both stores. No additional database library (SQLAlchemy, peewee) is introduced for T1 — raw `sqlite3` calls kept within `stores/backends/sqlite.py`.

**In-memory dict backend** (`stores/backends/in_memory.py`) is the T1 dev/demo backend for running without disk I/O.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **PostgreSQL** | Full-featured relational DB with excellent JSON support. Rejected for T1: requires a running server process, a connection string, and either SQLAlchemy or asyncpg as an additional dependency. Disproportionate for a hackathon demo that runs on a single machine. Remains a legitimate T2 upgrade path that does not require restructuring `stores/` — only the backend implementation changes. |
| **DuckDB** | Excellent analytical query engine, ideal for Parquet-backed queries. Rejected because the alert and profile access patterns are predominantly point-lookup (by `entity_id` and `alert_id`), not analytical aggregations. DuckDB excels at OLAP; the project needs OLTP-style indexed lookups. SQLite is the right fit. |
| **JSON flat files** | Zero dependency. Rejected because JSON is not binary-efficient for large feature arrays (boundary C payloads contain `feature_vector` arrays), and flat JSON files have no indexing, making `entity_id`-keyed profile lookups O(n) at scale. |

### Rationale
SQLite is embedded in Python's stdlib, requires no server, and supports the concurrent read / sequential write pattern used by the inference pipeline (many reads of profiles, occasional writes). The `in_memory.py` backend enables fast local dev without touching disk. The `sqlite.py` backend enables demo persistence across restarts. Both implement the same `ProfileStore` abstract interface, making the T2 Redis upgrade a single-file swap.

Parquet via `pyarrow` is chosen for data files because it is compressed, columnar, and directly readable by pandas with zero configuration — exactly what the Data Generator and Feature Engineering layers need.

### Judging Criterion
Supports **modular, production-quality architecture**: the swappable backend pattern (`in_memory.py` → `sqlite.py` → `redis.py`) is a textbook demonstration of the repository pattern and the open/closed principle. Supports **system design and scalability**: the existence of a Redis backend path signals that the design anticipates production scale even though it is not built for the hackathon.

---

## 8. Decision 7 — FastAPI Backend Stack

### Chosen Technologies

| Library | Version | Role |
|---------|---------|------|
| **FastAPI** | 0.111.x | ASGI web framework; `api/` package |
| **Pydantic** | 2.7.x | Request/response validation; boundary-K payload serialization |
| **Uvicorn** | 0.30.x | ASGI server (with `uvicorn[standard]` for WebSocket support — needed for T2 streaming push) |
| **python-multipart** | 0.0.9.x | Multipart form parsing (FastAPI dependency for file upload endpoints) |
| **httpx** | 0.27.x | Async HTTP client for test fixtures that call the API (test-only dependency) |

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **Flask 3.x** | Synchronous by default; requires additional effort (Gevent, Eventlet, or async Flask) to support concurrent inference requests. The simulated-streaming demo (T2) requires non-blocking event delivery — async is architecturally necessary, not optional. FastAPI's native `async def` route handlers and lifespan manager are directly required. |
| **Django REST Framework** | Full-featured, battle-tested. Rejected: DRF's ORM integration, settings module, and app registry are significant boilerplate for a project that uses SQLite directly without a Django model layer. DRF adds ~200+ files of framework overhead vs. FastAPI's near-zero boilerplate. |
| **Litestar (formerly Starlette-based)** | Similar feature set to FastAPI. Rejected for familiarity: FastAPI has the largest community, most documented examples for ML model serving, and is explicitly the required technology in the problem statement. |

### Rationale
FastAPI is the only backend technology explicitly named in the problem statement and in ARCHITECTURE.md. Pydantic v2 is the right choice over Pydantic v1 because FastAPI 0.100+ dropped v1 support, and Pydantic v2's Rust-based core delivers 5–50× faster validation — important for the simulated-streaming path where payloads are validated on every event. Uvicorn with the `standard` extras includes `websockets` and `httptools`, enabling T2 WebSocket-based streaming push from the API to the dashboard without introducing a new server.

### Judging Criterion
Directly satisfies the **mandatory FastAPI backend requirement**. Supports **system design and scalability**: Pydantic v2 models for all API responses directly serialize the boundary-K `RenderedAlertData` type, ensuring the data contract is machine-validated at the API surface.

---

## 9. Decision 8 — Dashboard / Frontend

### Chosen Technology
**Vanilla HTML5 + CSS3 + plain JavaScript (ES2022)** served as a static SPA from `src/dashboard/`.

The dashboard fetches data from the FastAPI backend over REST (boundary K) using the browser's native `fetch()` API. No bundler, no build step, no npm dependencies.

For charts and visualizations: **Chart.js 4.x** loaded via CDN `<script>` tag.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **Streamlit** | Extremely fast to build analytics dashboards in pure Python. Rejected because Streamlit requires its own server process and session management, which conflicts with the FastAPI backend already serving the ML pipeline. Running both simultaneously creates a two-server architecture for the demo. More critically, Streamlit's component model is incompatible with the HTML/CSS/JS structure defined in PROJECT_STRUCTURE.md (`index.html`, `main.css`, `api_client.js`), which would require a structural change. |
| **React (Vite)** | Industry-standard SPA framework. Rejected for hackathon time budget: React adds a required build pipeline (Vite/Webpack), JSX compilation, `node_modules`, and a JavaScript module system — none of which contribute to the judging criteria and all of which consume setup time. The dashboard's requirements (ranked table, risk score display, entity history, timeline) are achievable without a component framework. The plain JS approach keeps `src/dashboard/` as a zero-build-step artifact. |
| **Gradio** | Purpose-built for ML demo interfaces. Rejected: Gradio's layout model does not support a ranked alert queue UX naturally; it is designed for input/output demonstrations, not for an operations-center–style alert management view. The analyst dashboard requires a table with sorting, a side panel for explanations, and an entity history view — none of which Gradio provides cleanly without custom components. |

### Rationale
The dashboard's functional requirements (boundary K data display, ranked alerts, risk scores, entity history) are straightforwardly implementable in vanilla JS with `fetch()`. The plain approach eliminates all build tooling, meaning a judge can open `index.html` directly from a browser without any npm install step — a significant reliability advantage in a live demo. Chart.js 4.x via CDN provides the visualization capability needed for T2 (timeline view) and T3 (advanced charts) without adding a local dependency.

**T2 upgrade path:** The `simulated_stream.py` backend delivers events to the dashboard via a `Server-Sent Events` (SSE) endpoint on FastAPI (an `EventSourceResponse` or streaming `StreamingResponse`). The browser's native `EventSource` API receives these without requiring WebSocket or any additional JavaScript library. This is the specific reason Uvicorn is installed with the `standard` extras (Section 8).

### Judging Criterion
Supports **explainability and analyst usability**: a purpose-built analyst dashboard with a ranked alert queue, risk score display, and feature attribution panel is a more compelling demonstration of analyst UX than an auto-generated Streamlit or Gradio interface. A custom dashboard signals that analyst needs were deliberately designed for, not generated by a library.

---

## 10. Decision 9 — Streaming Simulation Mechanism

### Chosen Technology
**Python `asyncio` + FastAPI `StreamingResponse` / `EventSourceResponse`** (Server-Sent Events)

For T1 (batch mode): the Streaming Attachment Point is a synchronous file reader (`batch_reader.py`) that loads the full dataset into memory and returns it as a pandas DataFrame.

For T2 (simulated-streaming mode): `simulated_stream.py` implements an `asyncio` generator that yields events from the dataset in timestamp order, sleeping `(next_timestamp - current_timestamp) / compression_factor` between events. The FastAPI API exposes this as a Server-Sent Events stream consumed by the dashboard's `EventSource`.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **Apache Kafka (via `confluent-kafka` or `kafka-python`)** | The correct true-streaming technology. Explicitly deferred to T3 / Phase 12 per ARCHITECTURE.md Section 5. Introducing it for T2 would require a running Kafka broker, Zookeeper (or KRaft), topic management, consumer group configuration, and offset tracking — all out of scope for the hackathon timeline. The Streaming Attachment Point's abstract interface is already designed to accept a Kafka consumer as a third implementation alongside `batch_reader.py` and `simulated_stream.py`. |
| **Redis Streams** | Lighter than Kafka, already in the T2 tech stack (stores/backends/redis.py). Plausible option. Rejected for streaming simulation because it would couple the streaming mechanism to the T2 store backend — two separate T2 upgrades becoming inter-dependent. The `asyncio`-only approach keeps them independent. If Redis Streams is genuinely desired in a later phase, it can replace `simulated_stream.py` without touching any other component. |
| **Celery / task queues** | Designed for background job execution, not time-ordered event replay. The event ordering and time-compression semantics of simulated streaming are not naturally expressible as Celery tasks without significant custom logic. |

### Rationale
The `asyncio` generator approach requires no infrastructure beyond what is already present (FastAPI + Uvicorn). The `sleep` compression factor is configurable via `config/streaming.yaml`, satisfying the "configurable time-compression factor" requirement from ARCHITECTURE.md Section 5. Server-Sent Events are supported natively by all modern browsers and require no JavaScript library. The upgrade path to true Kafka streaming (Phase 12) involves replacing only the generator inside `simulated_stream.py` with a Kafka consumer loop — zero change to FastAPI, the dashboard, or any ML component.

### Judging Criterion
Supports **system design and scalability (real-time streaming feasibility)**: a working SSE-based live event stream visible in the dashboard during the demo directly and visually demonstrates near-real-time capability. The judge sees alerts appearing live, not a static table refresh. This is the highest-impact visual demonstration of the streaming criterion.

---

## 11. Decision 10 — Testing Framework

### Chosen Technologies

| Tool | Version | Role |
|------|---------|------|
| **pytest** | 8.x | Test runner, fixture system, plugin ecosystem |
| **pytest-asyncio** | 0.23.x | Async test function support for FastAPI endpoint tests |
| **pytest-cov** | 5.x | Coverage reporting |
| **httpx** | 0.27.x | Async HTTP test client for FastAPI (replaces `requests` in async context) |

No frontend JavaScript testing framework is introduced. Dashboard JavaScript correctness will be validated through manual testing and integration tests via the FastAPI API contract (boundary K).

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **unittest (stdlib)** | No additional dependency. Rejected because pytest's fixture system, parametrize decorator, and marker system (`@pytest.mark.tier1/2/3`) are architecturally required by CODING_GUIDELINES.md Section 3.3. Replicating this in unittest would require significantly more boilerplate. |
| **Jest (JavaScript)** | Correct tool for JavaScript unit testing. Rejected because: (1) introducing npm into the project for dashboard JS testing adds build tooling that was explicitly rejected in Decision 8, and (2) the dashboard JS is thin `fetch()`-and-render logic where the primary validation concern is the data contract at boundary K, not JavaScript logic itself. Boundary K is tested via `tests/api/test_alerts_router.py` using httpx. |

### Rationale
pytest is the industry standard for Python ML project testing. The `pytest-asyncio` plugin is required to test FastAPI endpoints with `async def` route handlers without wrapping them in `asyncio.run()`. The tier marker system defined in CODING_GUIDELINES.md Section 3.3 is natively supported via `pyproject.toml`'s `[tool.pytest.ini_options]` markers block.

### Judging Criterion
Supports **modular, production-quality architecture**: a test suite that runs per-tier (`pytest -m tier1`) demonstrates that the Tier 1 system is independently testable without Tier 2 or Tier 3 code present — directly verifying the T1 standalone guarantee from PROJECT_STRUCTURE.md Section 3.5.

---

## 12. Decision 11 — Dev Tooling

### Chosen Technologies

| Tool | Version | Role |
|------|---------|------|
| **ruff** | 0.4.x | Linting + import-order enforcement (`isort` replacement) |
| **ruff format** | (bundled with ruff) | Code formatting (`black`-compatible output) |
| **mypy** | 1.10.x | Static type checking; enforces type annotations on all public functions |
| **pip-tools** (`pip-compile`) | 7.x | Dependency pinning: generates `requirements.txt` and `requirements-dev.txt` from `pyproject.toml` |
| **python-dotenv** | 1.0.x | Loads `.env` into `os.environ` for local development |
| **pre-commit** | 3.7.x | Runs ruff and mypy as git hooks |

**No Poetry.** `pyproject.toml` with `pip-tools` is used for dependency management.

### Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| **Poetry** | Full dependency resolver, virtual environment management, and build system. Rejected because Poetry's lock file format differs from pip's, creating friction when judges or evaluators install the project using standard `pip install -r requirements.txt`. pip-tools produces standard pip-compatible requirements files with no additional tooling required from the reviewer. |
| **black + isort separately** | Two tools doing what ruff does as one. Rejected: ruff replaces both with faster execution and a single configuration block in `pyproject.toml`. Running two separate formatters in pre-commit slows commit hooks unnecessarily. |
| **pylint** | More comprehensive linting than ruff. Rejected: pylint's runtime is 10–20× slower than ruff, and its stricter rules create significant boilerplate friction during rapid hackathon development. Ruff covers the critical rules (unused imports, undefined names, import direction hints via `ruff`'s `I` rule set) without slowing the development loop. |

### Rationale
`ruff` + `mypy` in pre-commit hooks enforces CODING_GUIDELINES.md Section 2 (import direction) and Section 1 (naming) as automated checks rather than manual review processes. `mypy` enforces that boundary payload types from `common/types.py` are used correctly across all packages — particularly that `base.py` abstract method signatures use named types, not raw primitives (CODING_GUIDELINES.md Section 5.4). `pip-tools` ensures reproducible installs across machines without Poetry's additional runtime.

### Judging Criterion
Supports **modular, production-quality architecture**: the presence of mypy type checking across the codebase, combined with ruff import-order enforcement, demonstrates that the import dependency layer hierarchy (CODING_GUIDELINES.md Section 2.1) is machine-validated, not just documented.

---

## 13. Compatibility Check

The following matrix verifies that the chosen library versions are mutually compatible as of Python 3.12 (the target runtime).

### Core Compatibility Matrix

| Library A | Version | Library B | Version | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| FastAPI | 0.111.x | Pydantic | 2.7.x | ✅ Compatible | FastAPI 0.100+ requires Pydantic v2. FastAPI 0.111 is tested against Pydantic 2.7. |
| FastAPI | 0.111.x | Uvicorn | 0.30.x | ✅ Compatible | Standard ASGI pairing; no version conflict. |
| Pydantic | 2.7.x | Python | 3.12 | ✅ Compatible | Pydantic v2's Rust core (pydantic-core) provides wheels for Python 3.12 on all platforms. |
| PyTorch | 2.3.x | Python | 3.12 | ✅ Compatible | PyTorch 2.3 released official Python 3.12 wheels (CPU and CUDA). |
| PyTorch | 2.3.x | Captum | 0.7.x | ✅ Compatible | Captum 0.7 targets PyTorch 2.x. Verified in Captum release notes. |
| scikit-learn | 1.5.x | Python | 3.12 | ✅ Compatible | scikit-learn 1.4+ provides Python 3.12 wheels. |
| scikit-learn | 1.5.x | NumPy | 1.26 / 2.x | ✅ Compatible | scikit-learn 1.5 supports NumPy 1.17–2.x. NumPy 2.0 compatibility released in scikit-learn 1.5. |
| SHAP | 0.45.x | scikit-learn | 1.5.x | ✅ Compatible | SHAP's `TreeExplainer` and `LinearExplainer` are tested against scikit-learn 1.x. SHAP 0.44+ supports scikit-learn 1.4+. |
| SHAP | 0.45.x | NumPy | 2.x | ✅ Compatible | SHAP 0.44.1+ added NumPy 2.0 support. |
| SHAP | 0.45.x | Python | 3.12 | ✅ Compatible | SHAP 0.44+ provides Python 3.12 wheels. |
| pandas | 2.x | pyarrow | 15.x | ✅ Compatible | pandas 2.x recommends pyarrow as the default Parquet backend. |
| pandas | 2.x | NumPy | 2.x | ✅ Compatible | pandas 2.2+ supports NumPy 2.0. |
| pytest | 8.x | pytest-asyncio | 0.23.x | ✅ Compatible | pytest-asyncio 0.23 supports pytest 8. |
| pytest | 8.x | pytest-cov | 5.x | ✅ Compatible | pytest-cov 5 supports pytest 8. |
| ruff | 0.4.x | Python | 3.12 | ✅ Compatible | ruff has no Python version dependency; it analyses source files statically. |
| mypy | 1.10.x | Pydantic | 2.7.x | ✅ Compatible | `mypy` with `pydantic.mypy` plugin (bundled) supports Pydantic v2 models for type inference. |
| Faker | 25.x | Python | 3.12 | ✅ Compatible | Faker 24+ supports Python 3.12. |

### Known Compatibility Constraints (Non-Conflicts, but Must Be Managed)

| Constraint | Mitigation |
|------------|------------|
| PyTorch 2.3 CPU-only wheel is ~750 MB; GPU wheel is larger. CI environments must specify `torch` index URL correctly. | `pyproject.toml` will specify CPU-only index URL for CI. GPU usage is not required for the hackathon models. |
| Captum 0.7 does not support `torch.compile()`-wrapped models for attribution. | Attribution is computed on uncompiled model instances. `torch.compile()` is a T3 optimization applied to inference only, not to the attribution path. No conflict introduced. |
| SHAP's `TreeExplainer` requires `xgboost` or `sklearn` tree models. If the BPM selection phase (Phase 5) chooses a non-tree model (e.g., OneClassSVM), `KernelExplainer` must be used instead. | `feature_attribution.py` must select the correct SHAP explainer class based on the BPM type at runtime. This is an implementation detail for Phase 5, not a version conflict. |
| mypy strict mode with PyTorch produces spurious errors on `torch.Tensor` operations due to missing stubs. | `mypy` will be configured with `ignore_missing_imports = true` for `torch` specifically, and `torch` tensor operations will be typed as `torch.Tensor` rather than using full generic inference. |

### Full Stack Python Environment Summary

```
python = ">=3.11,<3.13"         # target 3.12

# Runtime dependencies
fastapi = "^0.111"
pydantic = "^2.7"
uvicorn = {extras = ["standard"], version = "^0.30"}
python-multipart = "^0.0.9"
torch = "^2.3"
captum = "^0.7"
scikit-learn = "^1.5"
shap = "^0.45"
numpy = ">=1.26,<3"
pandas = "^2.2"
pyarrow = "^15"
faker = "^25"
pyyaml = "^6"
python-dotenv = "^1.0"
httpx = "^0.27"

# Dev/test dependencies
pytest = "^8"
pytest-asyncio = "^0.23"
pytest-cov = "^5"
ruff = "^0.4"
mypy = "^1.10"
pre-commit = "^3.7"
pip-tools = "^7"
```

---

## 14. Explicitly Out of Scope

The following technologies and infrastructure are deliberately NOT used for this hackathon project. Introducing any of them would require either restructuring T1/T2 components or consuming time that must go to ML quality.

| Tool / Technology | Why Excluded |
|------------------|-------------|
| **Apache Kafka / Confluent Platform** | True streaming infrastructure requiring a broker process. Deferred to T3 / Phase 12 per ARCHITECTURE.md Section 5. The Streaming Attachment Point is designed to accept a Kafka consumer as a future swap-in. |
| **Redis** | Reserved as T2 store backend upgrade only. Not introduced in T1. No code outside `stores/backends/redis.py` (T2) references it. |
| **PostgreSQL / any server-based RDBMS** | Requires a running server. SQLite covers all T1 persistence needs without a server process. |
| **Docker / Docker Compose** | Reserved as T3 deployment wrapper per ARCHITECTURE.md and PROJECT_STRUCTURE.md. No internal code changes required to add it later. |
| **Kubernetes / Helm** | Microservice orchestration; explicitly out of scope for a hackathon modular monolith. |
| **MLflow / Weights & Biases / DVC** | Experiment tracking platforms. Valuable in production ML but add significant setup time for no judging benefit. Model artifacts are versioned with filename conventions defined in CODING_GUIDELINES.md Section 1.1. |
| **Celery / RQ (task queues)** | Background job execution systems. The simulated-streaming path uses `asyncio` natively inside the existing FastAPI process. No external task queue is needed. |
| **GraphQL** | Not required; the boundary-K data contract is well-defined and served by REST endpoints. GraphQL flexibility is not needed for a fixed dashboard schema. |
| **WebSocket (raw)** | Server-Sent Events (SSE) is preferred over WebSocket for the T2 streaming path because SSE is unidirectional (server → browser), matching the data flow exactly, and requires no handshake protocol. WebSocket would be over-engineered for this use case. |
| **React / Vue / Angular / Next.js** | SPA frameworks that introduce build tooling. Vanilla JS with Chart.js is sufficient for the dashboard requirements. Rejected in Decision 8. |
| **Jupyter notebooks as pipeline components** | Explicitly prohibited by PROJECT_STRUCTURE.md. `notebooks/` is for exploration only. |
| **Cloud provider SDKs (AWS, GCP, Azure)** | No cloud infrastructure dependency. The project runs entirely on a single machine. |

---

*End of TECH_STACK.md — Phase 3 output. This document is frozen. Amendments require a versioned change record.*
