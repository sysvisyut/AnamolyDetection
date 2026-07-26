# PROJECT_PROGRESS.md
# AI-Powered Behavioral Anomaly Detection — Progress Tracker

> **Last Updated:** 2026-07-25
> **Overall Completion %:** ~27%

## 1. Module Status

| Module ID | Module Name | Tier | Status | Completion % | Blocker | Notes |
|-----------|-------------|------|--------|--------------|---------|-------|
| M01 | Project Scaffolding & Configuration | T1 | Complete | 100% | None | Scaffolding complete; tests passing. |
| M02 | Core Data Contracts & Validators | T1 | Complete | 100% | None | Models implemented per M02 spec, 100% coverage. |
| M03 | Synthetic Data Gen (Core & Profiles) | T1 | Complete | 100% | None | Generator fully implemented; 32 tests pass, 97% coverage; 500-entity/59k-event run validated. |
| M04 | Synthetic Data Gen (Attack Injection) | T1 | Pending | 0% | None | |
| M05 | Entity Profile & Alert Stores | T1 | Pending | 50% | None | ProfileStore implemented in src/profiling; Alert Store pending. |
| M06 | Feature Engineering Pipeline | T1 | Complete | 100% | None | 9 modules; 97 tests pass; coverage: config=100%, encoders=97%, extractor=100%, pipeline=100%, geo=100%, interface=95%, seq_builder=98%, session_builder=96%, __init__=100%. All 8 acceptance criteria satisfied. |
| M07 | Behavioral Profiling Model (BPM) | T1 | Complete | 100% | None | Implemented statistical Z-score model, population priors, and ExtendedProfilingOutput. 7 tests pass; >90% coverage. |
| M08 | Sequence Detection Model (SDM) | T1 | Complete | 100% | None | Implemented GRU Autoencoder, dataset pipeline, inference engine, Captum hooks, and 7 passing tests. |
| M09 | Score Fusion & Classifier | T1 | Pending | 50% | None | Anomaly Classifier implemented with standalone LightGBM. Score Fusion pending. |
| M10 | Explainability Layer | T1 | Complete | 100% | None | Implemented attribution engine (SHAP + Captum IG), narrative translator with Insider Drift heuristics, and consistency validator. 5 passing tests with >80% coverage. |
| M11 | Orchestrator, Cold-Start & Drift Update | T1/T2 | Pending | 0% | M05, M06 | |
| M12 | FastAPI Core & Inference Endpoint | T1 | Pending | 0% | None | |
| M13 | Read Endpoints & Async Streaming | T1/T2 | Pending | 0% | None | |
| M14 | Dashboard UX | T1 | Pending | 0% | None | |
| M15 | Model Evaluation & T3 Polish | T2/T3 | Pending | 0% | None | |

## 2. Integration Status

- **Data Generator -> Feature Engineering:** Ready (M06 `ProfileStoreInterface` defined; concrete store pending M05)
- **Feature Engineering -> Models:** Not Started
- **Models -> Explainability:** Not Started
- **Explainability -> Alert Store:** Not Started
- **FastAPI -> Dashboard:** Not Started
- **Full End-to-End Streaming:** Not Started

## 3. Testing Status

- **Unit Tests:** Feature Engineering complete (97 tests, ≥95% coverage per module)
- **Contract Boundary Validation:** Not Started
- **Integration Tests:** Not Started

## 4. Known Technical Debt

- **M06 — encoders.py lines 424, 460, 493:** Three branches in `encode_resource_breadth`, `encode_entity_ip_ratio`, and `encode_session_event_count` representing alternative cold-start defaults are unreachable in current test setup due to input constraints. Low priority — these are safety guards.
- **M06 — session_builder.py lines 322–325:** The `get_recent_events()` method returning an empty list for an unseen entity is untested due to the mocked store always returning profiles in tests. Covered by contract but not by a direct unit test.
- **M05 (stores/):** `ProfileStoreInterface` is fully defined in M06. `ProfileStore` is now implemented in `src/profiling`.
- **Device Spoofing cold-start ambiguity:** Documented in M06 docstrings — fingerprint dims default to 0.5 (neutral/unknown) for cold-start entities rather than 0.0 (definite mismatch). This weakens Device Spoofing detection for brand-new entities but is the correct conservative behavior.
- **M07 (BPM) — Z-score ceiling:** Z-scores are capped at 5.0 to map to [0, 1] range. Extremely anomalous events (>5σ) are compressed to 1.0, losing relative magnitude in the primary `anomaly_score` field, though preserved in `per_feature_deviations`.
