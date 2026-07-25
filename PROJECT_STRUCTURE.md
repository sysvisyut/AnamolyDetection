# PROJECT_STRUCTURE.md
# AI-Powered Behavioral Anomaly Detection — Repository Structure

> **Status:** Phase 2 — Frozen Repository Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** ARCHITECTURE.md v1.0  
> **Scope:** Directory layout, folder ownership, and structural conventions only.  
> No implementation code, no schema details. All future implementation phases  
> consume this document alongside ARCHITECTURE.md as their joint input.

---

## Table of Contents

1. [Repository Directory Tree](#1-repository-directory-tree)
2. [Folder Ownership Map](#2-folder-ownership-map)
3. [Tier Separation Conventions](#3-tier-separation-conventions)
4. [Alternatives Considered](#4-alternatives-considered)
5. [Judging-Criteria Traceability](#5-judging-criteria-traceability)

---

## 1. Repository Directory Tree

The following tree represents the **complete, intended final state** of the repository after all tiers are implemented. Files that belong exclusively to Tier 2 or Tier 3 are marked with `# [T2]` or `# [T3]` inline comments. Everything not marked is Tier 1.

```
anomaly_detection/                          ← repository root
│
├── README.md                               ← project overview, quickstart
├── ARCHITECTURE.md                         ← Phase 1 frozen design reference
├── PROJECT_STRUCTURE.md                    ← Phase 2 frozen structure reference (this file)
├── CODING_GUIDELINES.md                    ← Phase 2 frozen coding conventions
├── pyproject.toml                          ← build system, dependency groups, tool config
├── requirements.txt                        ← pinned runtime dependencies (generated)
├── requirements-dev.txt                    ← pinned dev/test dependencies (generated)
├── .env.example                            ← environment variable template (no secrets)
├── .gitignore
│
├── config/                                 ← all runtime configuration
│   ├── default.yaml                        ← baseline config (used by T1 batch mode)
│   ├── streaming.yaml                      # [T2] simulated-streaming overrides
│   ├── cold_start.yaml                     # [T2] cold-start handler tuning
│   ├── drift.yaml                          # [T2] drift monitor thresholds
│   └── docker-compose.yaml                 # [T3] container orchestration
│
├── data/                                   ← all data files; never imported as Python
│   ├── raw/                                ← boundary A output (SDG → SAP)
│   │   └── .gitkeep
│   ├── labeled/                            ← ground-truth label store (eval use only)
│   │   └── .gitkeep
│   ├── processed/                          ← boundary B/C outputs (FE → models)
│   │   └── .gitkeep
│   └── profiles/                           ← Entity Profile Store flat-file backend (T1)
│       └── .gitkeep
│
├── src/                                    ← all importable Python source code
│   │
│   ├── anomaly_detection/                  ← top-level package (matches project name)
│   │   ├── __init__.py
│   │   │
│   │   ├── common/                         ← shared utilities; no business logic
│   │   │   ├── __init__.py
│   │   │   ├── types.py                    ← all shared dataclasses / TypedDicts
│   │   │   ├── logging.py                  ← structured logging setup
│   │   │   ├── config.py                   ← config loader (reads config/*.yaml)
│   │   │   ├── exceptions.py               ← project-wide custom exception hierarchy
│   │   │   └── validators.py               ← boundary contract validators
│   │   │
│   │   ├── data_generator/                 ← ARCHITECTURE component: Synthetic Data Generator
│   │   │   ├── __init__.py
│   │   │   ├── generator.py                ← main entry point; produces boundary-A records
│   │   │   ├── entity_profiles.py          ← per-entity behavioral assumption definitions
│   │   │   ├── attack_injector.py          ← injects all 7 labeled attack patterns
│   │   │   ├── label_store.py              ← writes ground-truth labels to data/labeled/
│   │   │   └── schemas.py                  ← boundary-A record schema definition
│   │   │
│   │   ├── streaming/                      ← ARCHITECTURE component: Streaming Attachment Point
│   │   │   ├── __init__.py
│   │   │   ├── batch_reader.py             ← T1 batch delivery mode (boundary B, batch)
│   │   │   ├── simulated_stream.py         # [T2] time-paced event replay (boundary B, simulated_stream)
│   │   │   └── stream_interface.py         ← abstract base class both modes implement
│   │   │
│   │   ├── feature_engineering/            ← ARCHITECTURE component: Feature Engineering
│   │   │   ├── __init__.py
│   │   │   ├── session_builder.py          ← assembles per-session feature records
│   │   │   ├── sequence_builder.py         ← constructs ordered sequence windows (boundary C)
│   │   │   ├── encoders.py                 ← categorical encoding and normalization
│   │   │   ├── geo_velocity.py             ← geo-velocity delta and impossible-travel features
│   │   │   └── profile_updater.py          ← produces boundary-D profile update events
│   │   │
│   │   ├── stores/                         ← ARCHITECTURE components: Entity Profile Store + Alert & Result Store
│   │   │   ├── __init__.py
│   │   │   ├── profile_store.py            ← Entity Profile Store: read/write boundary-E contracts
│   │   │   ├── alert_store.py              ← Alert & Result Store: write boundary-I, read boundary-J
│   │   │   └── backends/                   ← storage backend implementations
│   │   │       ├── __init__.py
│   │   │       ├── in_memory.py            ← T1 default (dict-backed, for dev/demo)
│   │   │       ├── sqlite.py               ← T1 persistent option (file-backed SQLite)
│   │   │       └── redis.py                # [T2] production-grade backend (optional upgrade)
│   │   │
│   │   ├── models/                         ← ARCHITECTURE components: BPM + SDM + Score Fusion
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── behavioral_profiling/       ← ARCHITECTURE component: Behavioral Profiling Model
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py                 ← abstract BPM interface (boundary C in, boundary F out)
│   │   │   │   ├── trainer.py              ← offline training entry point
│   │   │   │   ├── inference.py            ← online scoring entry point
│   │   │   │   └── artifacts/              ← saved model files (not committed to git)
│   │   │   │       └── .gitkeep
│   │   │   │
│   │   │   ├── sequence_detection/         ← ARCHITECTURE component: Sequence Detection Model
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py                 ← abstract SDM interface (boundary C in, boundary F out)
│   │   │   │   ├── trainer.py              ← offline training entry point
│   │   │   │   ├── inference.py            ← online scoring entry point
│   │   │   │   └── artifacts/              ← saved model files (not committed to git)
│   │   │   │       └── .gitkeep
│   │   │   │
│   │   │   └── fusion/                     ← ARCHITECTURE component: Score Fusion
│   │   │       ├── __init__.py
│   │   │       └── score_fusion.py         ← combines boundary-F scores → boundary-G signal
│   │   │
│   │   ├── classifier/                     ← ARCHITECTURE component: Anomaly Classifier
│   │   │   ├── __init__.py
│   │   │   ├── base.py                     ← abstract classifier interface (boundary G in, boundary H out)
│   │   │   ├── trainer.py                  ← offline training entry point
│   │   │   ├── inference.py                ← online classification entry point
│   │   │   └── artifacts/                  ← saved model files (not committed to git)
│   │   │       └── .gitkeep
│   │   │
│   │   ├── explainability/                 ← ARCHITECTURE component: Explainability Layer
│   │   │   ├── __init__.py
│   │   │   ├── risk_scorer.py              ← converts fused score → 0-100 risk_score + risk_tier
│   │   │   ├── feature_attribution.py      ← produces feature_attributions[] (boundary I)
│   │   │   ├── narrative.py                ← generates human_readable_explanation string
│   │   │   ├── alert_builder.py            ← assembles complete boundary-I Alert Payload
│   │   │   ├── mitre_mapping.py            # [T2] MITRE ATT&CK category enrichment
│   │   │   └── calibration.py              # [T3] confidence calibration post-processing
│   │   │
│   │   ├── api/                            ← ARCHITECTURE component: FastAPI Backend
│   │   │   ├── __init__.py
│   │   │   ├── main.py                     ← FastAPI app factory and lifespan handler
│   │   │   ├── dependencies.py             ← dependency-injection wiring (stores, models)
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── inference.py            ← inference endpoints (boundary J query path)
│   │   │   │   ├── alerts.py               ← alert retrieval endpoints (boundary K)
│   │   │   │   ├── entities.py             ← entity timeline endpoints (boundary K)
│   │   │   │   ├── simulation.py           # [T3] interactive attack simulation endpoint
│   │   │   │   └── feedback.py             # [T3] analyst feedback endpoint (boundary L)
│   │   │   └── middleware.py               ← CORS, logging, error-handling middleware
│   │   │
│   │   ├── cold_start/                     # [T2] ARCHITECTURE component: Cold-Start Handler
│   │   │   ├── __init__.py
│   │   │   ├── handler.py                  ← intercepts cold_start_flag=true (boundary E path)
│   │   │   └── priors.py                   ← group-prior and heuristic profile generators
│   │   │
│   │   ├── drift/                          # [T2] ARCHITECTURE component: Drift Monitor
│   │   │   ├── __init__.py
│   │   │   ├── monitor.py                  ← passive observer of Entity Profile Store
│   │   │   ├── detector.py                 ← drift statistic computation
│   │   │   └── retraining_trigger.py       # [T3] automated retraining orchestration
│   │   │
│   │   └── evaluation/                     # [T2] ARCHITECTURE component: Model Evaluation Module
│   │       ├── __init__.py
│   │       ├── evaluator.py                ← consumes labeled data, calls classifier outputs
│   │       ├── metrics.py                  ← precision, recall, F1, AUROC computation
│   │       └── report_generator.py         ← produces evaluation report artifact
│   │
│   └── dashboard/                          ← ARCHITECTURE component: Analyst Dashboard
│       ├── index.html                      ← single-page app entry point
│       ├── styles/
│       │   └── main.css
│       ├── scripts/
│       │   ├── api_client.js               ← fetches boundary-K data from FastAPI
│       │   ├── alert_queue.js              ← ranked alert queue panel
│       │   ├── entity_view.js              ← entity history and basic timeline (T1)
│       │   ├── timeline_view.js            # [T2] full entity timeline panel
│       │   └── visualizations.js           # [T3] advanced chart panels
│       └── assets/
│           └── .gitkeep
│
├── tests/                                  ← mirrors src/anomaly_detection/ structure exactly
│   ├── conftest.py                         ← shared pytest fixtures
│   ├── common/
│   │   └── test_validators.py
│   ├── data_generator/
│   │   ├── test_generator.py
│   │   ├── test_attack_injector.py
│   │   └── test_label_store.py
│   ├── streaming/
│   │   ├── test_batch_reader.py
│   │   └── test_simulated_stream.py        # [T2]
│   ├── feature_engineering/
│   │   ├── test_session_builder.py
│   │   ├── test_sequence_builder.py
│   │   └── test_geo_velocity.py
│   ├── stores/
│   │   ├── test_profile_store.py
│   │   └── test_alert_store.py
│   ├── models/
│   │   ├── behavioral_profiling/
│   │   │   └── test_inference.py
│   │   ├── sequence_detection/
│   │   │   └── test_inference.py
│   │   └── fusion/
│   │       └── test_score_fusion.py
│   ├── classifier/
│   │   └── test_inference.py
│   ├── explainability/
│   │   ├── test_risk_scorer.py
│   │   ├── test_feature_attribution.py
│   │   ├── test_narrative.py
│   │   └── test_alert_builder.py
│   ├── api/
│   │   ├── test_inference_router.py
│   │   ├── test_alerts_router.py
│   │   └── test_entities_router.py
│   ├── cold_start/                         # [T2]
│   │   └── test_handler.py
│   ├── drift/                              # [T2]
│   │   └── test_monitor.py
│   └── evaluation/                         # [T2]
│       └── test_evaluator.py
│
├── scripts/                                ← CLI entry points; not importable as library code
│   ├── generate_data.py                    ← runs Synthetic Data Generator
│   ├── train_models.py                     ← runs all model training pipelines
│   ├── run_evaluation.py                   # [T2] runs Model Evaluation Module
│   └── seed_demo.py                        ← pre-loads stores for a live demo run
│
├── notebooks/                              ← throwaway experimentation only; not part of the pipeline
│   └── README.md                           ← "Notebooks are for exploration. No pipeline code lives here."
│
└── docs/                                   ← generated and authored documentation
    ├── report/
    │   └── TECHNICAL_REPORT.md             ← hackathon report (Phase-final output)
    └── presentation/
        └── PRESENTATION_OUTLINE.md         ← presentation slide outline
```

---

## 2. Folder Ownership Map

The table below maps every top-level folder and every package subfolder to the ARCHITECTURE.md component it owns. The "Boundary Responsibility" column cites the data contract boundary labels from ARCHITECTURE.md Section 3.

| Folder / Package | ARCHITECTURE.md Component | Boundary Responsibility | Tier |
|-----------------|--------------------------|------------------------|------|
| `src/anomaly_detection/common/` | Shared utilities — no architecture component | Provides types used by all boundaries | T1 |
| `src/anomaly_detection/data_generator/` | **Synthetic Data Generator** | Produces boundary **A** (Raw Access Logs); writes to `data/raw/` and `data/labeled/` | T1 |
| `src/anomaly_detection/streaming/` | **Streaming Attachment Point** | Consumes boundary **A**; produces boundary **B**; strips `label` field | T1 (batch) / T2 (simulated stream) |
| `src/anomaly_detection/feature_engineering/` | **Feature Engineering** | Consumes boundary **B**; produces boundaries **C** and **D** | T1 |
| `src/anomaly_detection/stores/profile_store.py` | **Entity Profile Store** | Consumes boundary **D**; serves boundary **E** | T1 |
| `src/anomaly_detection/stores/alert_store.py` | **Alert & Result Store** | Consumes boundary **I**; serves boundary **J** | T1 |
| `src/anomaly_detection/stores/backends/` | Storage backends for both stores | Internal to stores; no external boundary | T1 (in_memory, sqlite) / T2 (redis) |
| `src/anomaly_detection/models/behavioral_profiling/` | **Behavioral Profiling Model** | Consumes boundaries **C** and **E**; produces boundary **F** (`model_id=bpm`) | T1 |
| `src/anomaly_detection/models/sequence_detection/` | **Sequence Detection Model** | Consumes boundaries **C** and **E**; produces boundary **F** (`model_id=sdm`) | T1 |
| `src/anomaly_detection/models/fusion/` | **Score Fusion** | Consumes boundary **F** (both model scores); produces boundary **G** | T1 |
| `src/anomaly_detection/classifier/` | **Anomaly Classifier** | Consumes boundary **G**; produces boundary **H** | T1 |
| `src/anomaly_detection/explainability/` | **Explainability Layer** | Consumes boundaries **G** and **H**; produces boundary **I** | T1 (core) / T2 (mitre_mapping) / T3 (calibration) |
| `src/anomaly_detection/api/` | **FastAPI Backend** | Orchestrates inference pipeline; bidirectional boundary **J**; produces boundary **K**; consumes boundary **L** (T3) | T1 (core routers) / T3 (simulation, feedback routers) |
| `src/anomaly_detection/cold_start/` | **Cold-Start Handler** | Attaches between boundary **E** and **BPM**; produces synthetic profile | T2 |
| `src/anomaly_detection/drift/` | **Drift Monitor** | Passive reader of Entity Profile Store; emits drift events to BPM/SDM recalibration interface | T2 (monitor, detector) / T3 (retraining_trigger) |
| `src/anomaly_detection/evaluation/` | **Model Evaluation Module** | Offline consumer of `data/labeled/`; reads classifier outputs from Alert Store | T2 |
| `src/dashboard/` | **Analyst Dashboard** | Consumes boundary **K**; produces boundary **L** (T3) | T1 (alert_queue, entity_view) / T2 (timeline_view) / T3 (visualizations) |
| `data/raw/` | Data output of Synthetic Data Generator | Boundary **A** on-disk representation | T1 |
| `data/labeled/` | Ground-truth label store | Consumed only by Model Evaluation Module | T1 (written) / T2 (consumed) |
| `data/processed/` | Intermediate feature outputs | Boundaries **B** and **C** on-disk cache (optional) | T1 |
| `data/profiles/` | Entity Profile Store flat-file backend | Boundary **E** persistence | T1 |
| `tests/` | Test suite mirroring `src/anomaly_detection/` | No production boundary; validates all boundaries | T1–T3 (mirrors source tier) |
| `scripts/` | CLI wrappers for pipeline entry points | No production boundary; developer tooling | T1 |
| `config/` | Runtime configuration | Read by `common/config.py`; consumed by every component | T1 (default.yaml) / T2–T3 (other files) |
| `docs/` | Documentation deliverables | Not a pipeline component | T1–T3 |
| `notebooks/` | Exploration only | Explicitly prohibited from containing pipeline code | N/A |

---

## 3. Tier Separation Conventions

### 3.1 Directory-Level Separation

The primary mechanism for tier separation is **directory placement**:

- **Tier 1:** All code under `src/anomaly_detection/` that is NOT in `cold_start/`, `drift/`, or `evaluation/` is Tier 1. The T1 system is runnable without those folders existing.
- **Tier 2:** The `cold_start/`, `drift/`, and `evaluation/` packages are entirely Tier 2. Within mixed-tier files (e.g., `explainability/`, `streaming/`, `stores/backends/`, `dashboard/scripts/`), individual files are annotated with `# [T2]` in the directory tree above.
- **Tier 3:** Individual files within otherwise T1/T2 packages are annotated with `# [T3]`. The `api/routers/feedback.py` and `api/routers/simulation.py` files are Tier 3; the rest of `api/` is Tier 1.

### 3.2 Python-Level Separation — Optional Import Guards

For any file that is imported conditionally (i.e., a T2/T3 module imported by a T1 module), the import must be wrapped in a try/except guard at the call site — never at the top of the file. This ensures the T1 system does not crash if a T2/T3 file is absent.

Pattern (to be enforced during implementation phases, not coded here):

```
# In a T1 module that optionally uses a T2 capability:
# The guard lives in the function body, not at module top-level.
# T2 capability degrades gracefully to a no-op when absent.
```

### 3.3 Configuration-Level Separation

Tier 2 and Tier 3 features are enabled via configuration flags in `config/*.yaml` files. A T1-only run uses only `config/default.yaml`. T2 features require `config/streaming.yaml`, `config/cold_start.yaml`, and/or `config/drift.yaml` to be present and loaded. T3 features are enabled by additional flags within those files.

This means a reviewer or judge can audit exactly which capabilities are active by reading the loaded config, without inspecting source code.

### 3.4 Test-Level Separation

Test files annotated `# [T2]` or `# [T3]` in the directory tree above are collected and run as a separate pytest mark:

- `pytest -m tier1` — runs only tests for the T1 baseline system.
- `pytest -m tier2` — runs T1 + T2 tests.
- `pytest` (no marker) — runs the full suite.

Every test file must declare its tier mark in its module docstring (format defined in CODING_GUIDELINES.md).

### 3.5 Tier 1 Standalone Guarantee

The Tier 1 system is runnable if and only if the following are present:

1. `src/anomaly_detection/` (excluding `cold_start/`, `drift/`, `evaluation/`)
2. `src/dashboard/scripts/alert_queue.js` and `src/dashboard/scripts/entity_view.js`
3. `config/default.yaml`
4. `data/` directory (contents generated at runtime)
5. `scripts/generate_data.py` and `scripts/train_models.py`

No Tier 2 or Tier 3 file is a required import of any Tier 1 file. This invariant must be verified at the start of each implementation phase.

---

## 4. Alternatives Considered

### Option A: Flat src-layout (All Modules at One Level)

**Description:** Every component is a single Python file at the top level of `src/anomaly_detection/`: `generator.py`, `feature_engineering.py`, `profiling.py`, `sequence_model.py`, `classifier.py`, `explainability.py`, `api.py`. No sub-packages.

**Why Rejected:**
- A flat layout does not enforce module boundary discipline: any file can import from any other file without the structure making the dependency direction visible. This makes import-direction violations (e.g., a model importing from the API layer) undetectable without static analysis tooling.
- When each module grows to multiple files (trainer, inference, base interface, model artifacts), a flat layout forces everything into a single file or creates a chaotic set of top-level files with no grouping signal.
- Multiple AI-assisted coding sessions working in parallel on different components cannot be given clear "files you may modify / files you must not touch" instructions without sub-package boundaries as reference points. A flat layout makes every session aware of the entire codebase simultaneously.
- Tier separation cannot be expressed through directory placement; it would require file-naming conventions alone (e.g., `cold_start_handler.py` vs. `handler_t2.py`), which are fragile and non-obvious.

**What It Gets Right:** Minimal boilerplate for a very small project. Appropriate for a single-file proof-of-concept.

---

### Option B: Domain-Layered Clean Architecture (domain / application / infrastructure)

**Description:** Code is organized into horizontal layers that apply across all components:
- `domain/` — core business entities and logic (pure Python, no framework dependencies)
- `application/` — use cases and orchestration
- `infrastructure/` — database adapters, API framework, external services

**Why Rejected:**
- Clean Architecture layers are designed around a single bounded domain context. This project has 11 distinct architectural components crossing multiple ML model types, a streaming layer, two separate stores, and a web frontend. Forcing all of these into three horizontal layers obscures which layer belongs to which architectural component.
- For AI-assisted development across many coding sessions, clean-architecture layers multiply cognitive overhead: a session implementing the Behavioral Profiling Model would need to read and understand files spread across `domain/`, `application/`, and potentially `infrastructure/`, even for simple model inference logic.
- The "dependency rule" (inner layers must not depend on outer layers) conflicts with the ML pipeline's data-flow dependency structure (BPM and SDM must read from Entity Profile Store, which is an "infrastructure" concern, but their output flows into "application" Score Fusion).
- Hackathon judging does not reward layered architecture purity; it rewards clear component separation, which the chosen approach delivers more directly.

**What It Gets Right:** Strong enforcement of the dependency inversion principle and makes the system highly testable in isolation. These benefits are achievable in the chosen structure through explicit import-direction rules (see CODING_GUIDELINES.md) without the organizational overhead.

---

### Chosen Approach: Component-Per-Package (Modular Monolith)

**Description:** Each ARCHITECTURE.md component owns exactly one Python sub-package under `src/anomaly_detection/`. The package name maps 1:1 to the architecture component name. Within each package, files are split by responsibility (base interface, trainer, inference, schemas). The entire codebase is a single installable Python package.

**Why This Is Better for This Project:**

1. **Direct architecture traceability:** Every folder name corresponds to a named component in ARCHITECTURE.md. A judge, reviewer, or future developer can navigate the codebase using the architecture diagram as their map.

2. **AI-assisted parallel development:** Each coding session can be scoped to a single package (`feature_engineering/`, `classifier/`, etc.) with a clear "files you may modify" constraint. The sub-package boundary is the natural isolation unit for session scoping.

3. **Import direction enforcement:** Python's package system makes cross-boundary imports visible and auditable. The rule "common may import nothing; stores may import only common; models may import stores and common; api may import everything except nothing imports api" is naturally expressed as a package hierarchy.

4. **Tier separation through directory placement:** The `cold_start/`, `drift/`, and `evaluation/` packages being entirely Tier 2 is immediately visible without reading any code. Tier 3 files within T1 packages are annotated in the tree.

5. **Scalability to microservices:** Each sub-package is already structured to become an independent service — it has a defined interface (base.py), independent trainer and inference entry points, and its own test directory. The microservice upgrade path (Tier 3) requires wrapping each package in a FastAPI process, not restructuring its internals.

---

## 5. Judging-Criteria Traceability

| Judging Criterion | How the Structure Addresses It |
|------------------|-------------------------------|
| **Modular, production-quality architecture** | Each ARCHITECTURE.md component maps to exactly one sub-package with its own `base.py` interface, `trainer.py`, `inference.py`, and test directory. A judge reviewing the codebase sees clear module boundaries, not a monolithic script. |
| **Maintainability / scalability** | The component-per-package layout means any single component can be refactored, replaced, or promoted to a microservice without touching sibling packages. The `backends/` subdirectory under `stores/` demonstrates this: swapping `in_memory.py` for `redis.py` changes zero code outside the `stores/` package. |
| **System design and scalability (real-time streaming)** | The `streaming/` package's `stream_interface.py` abstract base class makes the streaming upgrade path structurally visible: `batch_reader.py` and `simulated_stream.py` are parallel implementations of the same interface. A true-streaming Kafka adapter would be a third file in the same package. |
| **Explainability and analyst usability** | The `explainability/` package being an independent module with its own files (`risk_scorer.py`, `feature_attribution.py`, `narrative.py`) demonstrates to judges that explainability is a first-class architectural concern, not logic embedded inside the detector. |
| **Report clarity** | The `docs/` folder and the `docs/report/TECHNICAL_REPORT.md` path make the documentation deliverable a named, findable artifact — not a random Markdown file at the repo root. |
| **Detection accuracy on imbalanced labels** | `models/behavioral_profiling/` and `models/sequence_detection/` being separate packages with independent `trainer.py` files signals that each model can be trained, evaluated, and tuned independently — directly supporting the independent calibration strategy from ARCHITECTURE.md Section 8. |

---

*End of PROJECT_STRUCTURE.md — Phase 2 output. This document is frozen. Amendments require a versioned change record.*
