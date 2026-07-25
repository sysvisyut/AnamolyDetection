# CODING_GUIDELINES.md
# AI-Powered Behavioral Anomaly Detection — Coding Conventions

> **Status:** Phase 2 — Frozen Coding Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** ARCHITECTURE.md v1.0, PROJECT_STRUCTURE.md v1.0  
> **Scope:** Naming, import rules, test conventions, documentation standards, and  
> module boundary definitions. No implementation code or business logic.  
> All future implementation phases consume this document alongside  
> ARCHITECTURE.md and PROJECT_STRUCTURE.md as their joint input.

---

## Table of Contents

1. [Naming Conventions](#1-naming-conventions)
2. [Import and Dependency Direction Rules](#2-import-and-dependency-direction-rules)
3. [Test File Naming and Placement](#3-test-file-naming-and-placement)
4. [Docstring and Comment Conventions](#4-docstring-and-comment-conventions)
5. [Module Boundary Rules](#5-module-boundary-rules)

---

## 1. Naming Conventions

### 1.1 Files and Directories

| Item | Convention | Example |
|------|-----------|---------|
| Python source files | `snake_case.py` | `session_builder.py`, `risk_scorer.py` |
| Python test files | `test_<source_name>.py` | `test_session_builder.py` |
| Python packages (directories) | `snake_case/` with `__init__.py` | `feature_engineering/`, `behavioral_profiling/` |
| Config files | `snake_case.yaml` | `default.yaml`, `cold_start.yaml` |
| Markdown documents | `SCREAMING_SNAKE_CASE.md` | `ARCHITECTURE.md`, `CODING_GUIDELINES.md` |
| Dashboard JavaScript files | `snake_case.js` | `alert_queue.js`, `api_client.js` |
| Dashboard CSS files | `snake_case.css` | `main.css` |
| Saved model artifact files | `<component>_<version>.pkl` or `<component>_<version>.pt` | `bpm_v1.pkl`, `sdm_v1.pt` |
| Data files | `snake_case.<ext>` | `synthetic_logs.parquet`, `entity_profiles.db` |

**Rules:**
- No hyphens in Python filenames. Hyphens break Python import resolution.
- No version numbers in source file names (only in saved artifact filenames).
- All Python files are lowercase. No CamelCase filenames.

---

### 1.2 Python Classes

| Item | Convention | Example |
|------|-----------|---------|
| All classes | `PascalCase` | `SessionBuilder`, `ScoreFusion`, `AlertPayload` |
| Abstract base classes | `PascalCase` with no special prefix/suffix | `BehavioralProfilingModel`, `StreamInterface` |
| Concrete implementations | `PascalCase` describing the implementation | `InMemoryProfileStore`, `SQLiteAlertStore` |
| Pydantic / dataclass models | `PascalCase` matching the boundary contract name | `RawAccessLog`, `FeatureVector`, `AlertPayload` |
| Custom exceptions | `PascalCase` ending in `Error` | `ProfileNotFoundError`, `BoundaryContractError` |
| Enum classes | `PascalCase` | `AttackClass`, `RiskTier`, `DeliveryMode` |
| Enum members | `SCREAMING_SNAKE_CASE` | `AttackClass.BRUTE_FORCE`, `RiskTier.CRITICAL` |

---

### 1.3 Functions and Methods

| Item | Convention | Example |
|------|-----------|---------|
| All functions and methods | `snake_case` | `build_sequence_window()`, `compute_risk_score()` |
| Boolean-returning functions | `is_*`, `has_*`, `can_*` | `is_cold_start()`, `has_profile()` |
| Factory functions | `create_*` | `create_alert_store()`, `create_bpm()` |
| Abstract methods | Same as regular methods; marked with `@abstractmethod` | `score()`, `classify()` |
| Async functions | Same as sync functions; must be named without `async_` prefix | `get_alert()` not `async_get_alert()` |

---

### 1.4 Variables and Parameters

| Item | Convention | Example |
|------|-----------|---------|
| Local variables | `snake_case` | `entity_id`, `fused_score`, `sequence_window` |
| Function parameters | `snake_case` | `event_id`, `anomaly_score` |
| Module-level constants | `SCREAMING_SNAKE_CASE` | `MAX_SEQUENCE_WINDOW`, `DEFAULT_RISK_THRESHOLD` |
| Private attributes | `_snake_case` (single leading underscore) | `_profile_cache`, `_model_artifact` |
| "Should not exist outside this file" names | `__snake_case` (double leading underscore) | `__internal_state` |
| Type alias names | `PascalCase` | `EntityId = str`, `FeatureVector = list[float]` |

**Rules:**
- Never use single-letter variable names outside of loop counters (`i`, `j`, `k`) and comprehension variables.
- Boolean variables must use `is_`, `has_`, or `can_` prefix: `is_anomaly = True`, not `anomaly = True`.
- Avoid abbreviations unless they are defined in ARCHITECTURE.md (e.g., `bpm`, `sdm`, `sap`, `sdg`, `el`, `ac`). These specific abbreviations are canonical and may be used exactly as written.

---

### 1.5 Configuration Keys

- Config keys in YAML files use `snake_case`.
- Multi-level config keys use nesting, not dot-notation in YAML: `score_fusion.threshold` becomes `score_fusion:` with a `threshold:` child.
- Config keys that toggle tier features use the prefix `enable_`: `enable_simulated_stream: true`.

---

### 1.6 Boundary-Derived Names

Every data contract payload defined in ARCHITECTURE.md Section 3 must be represented as a named Python class in `src/anomaly_detection/common/types.py`. The class name must match the boundary label's description, rendered in `PascalCase`:

| Boundary | Payload Class Name |
|----------|-------------------|
| A | `RawAccessLog` |
| B | `InboundEvent` |
| C | `EngineeeredFeatures` |
| D | `ProfileUpdateEvent` |
| E | `EntityProfile` |
| F | `ModelScore` |
| G | `UnifiedAnomalySignal` |
| H | `ClassificationResult` |
| I | `AlertPayload` |
| J | (internal store query — no named class required) |
| K | `RenderedAlertData` |
| L | `AnalystAction` |

These class names are canonical across the entire codebase. They must not be renamed in any implementation phase without a versioned change record in ARCHITECTURE.md.

---

## 2. Import and Dependency Direction Rules

### 2.1 The Dependency Layer Hierarchy

The allowed import direction is **strictly top-down** within the following hierarchy. A layer may import from any layer below it. A layer must **never** import from any layer above it.

```
Layer 6 (top):  api/
                dashboard/

Layer 5:        explainability/

Layer 4:        classifier/

Layer 3:        models/
                (behavioral_profiling/, sequence_detection/, fusion/)

Layer 2:        feature_engineering/
                streaming/
                data_generator/

Layer 1:        stores/
                cold_start/       [T2]
                drift/            [T2]
                evaluation/       [T2]

Layer 0 (base): common/
```

### 2.2 Explicit Permitted and Prohibited Directions

| Package | May Import From | Must Never Import From |
|---------|----------------|------------------------|
| `common/` | Python stdlib, approved third-party libs only | Everything in `src/anomaly_detection/` |
| `stores/` | `common/` | `feature_engineering/`, `models/`, `classifier/`, `explainability/`, `api/` |
| `data_generator/` | `common/` | `stores/`, `feature_engineering/`, `models/`, `classifier/`, `explainability/`, `api/` |
| `streaming/` | `common/`, `data_generator/` (schema only) | `stores/`, `models/`, `classifier/`, `explainability/`, `api/` |
| `feature_engineering/` | `common/`, `stores/` | `models/`, `classifier/`, `explainability/`, `api/` |
| `models/behavioral_profiling/` | `common/`, `stores/` | `classifier/`, `explainability/`, `api/` |
| `models/sequence_detection/` | `common/`, `stores/` | `classifier/`, `explainability/`, `api/` |
| `models/fusion/` | `common/`, `models/behavioral_profiling/`, `models/sequence_detection/` | `classifier/`, `explainability/`, `api/` |
| `classifier/` | `common/`, `stores/`, `models/fusion/` | `explainability/`, `api/` |
| `explainability/` | `common/`, `stores/`, `models/fusion/`, `classifier/` | `api/` |
| `cold_start/` [T2] | `common/`, `stores/` | `models/`, `classifier/`, `explainability/`, `api/` |
| `drift/` [T2] | `common/`, `stores/` | `models/`, `classifier/`, `explainability/`, `api/` |
| `evaluation/` [T2] | `common/`, `stores/`, `classifier/` | `api/` |
| `api/` | Everything in `src/anomaly_detection/` | Nothing (api is the top layer) |
| `dashboard/` | No Python imports (JavaScript only); fetches from `api/` over HTTP | N/A |

### 2.3 Third-Party Library Import Rules

- All third-party imports must be declared in `pyproject.toml` before being used in any source file.
- Third-party libraries may only be imported in the layer where they are needed. A library that is a detail of the `stores/` layer (e.g., SQLAlchemy) must not be imported in `models/` or any higher layer.
- The `common/` package may import only: Python stdlib, `pydantic`, `PyYAML`, and logging utilities. No ML framework imports in `common/`.
- The `models/` packages may import ML frameworks (e.g., scikit-learn, PyTorch, NumPy, pandas). No ML framework imports in `api/` or `explainability/` beyond what is needed for attribution methods.

### 2.4 Circular Import Rule

Circular imports are prohibited. The layered hierarchy above structurally prevents them when the rules are followed. If a circular import appears, it signals that a type belongs in `common/types.py` rather than in either of the importing modules.

### 2.5 Relative vs. Absolute Imports

- Within a sub-package, use relative imports: `from .schemas import RawAccessLog`.
- Across sub-packages, use absolute imports: `from anomaly_detection.common.types import EntityProfile`.
- Never use star imports (`from module import *`) anywhere in the codebase.

---

## 3. Test File Naming and Placement

### 3.1 Placement Rule

Every test file lives in `tests/` at the path that mirrors its source file's path within `src/anomaly_detection/`. The prefix `test_` is added to the filename.

| Source file | Test file |
|------------|-----------|
| `src/anomaly_detection/feature_engineering/session_builder.py` | `tests/feature_engineering/test_session_builder.py` |
| `src/anomaly_detection/models/fusion/score_fusion.py` | `tests/models/fusion/test_score_fusion.py` |
| `src/anomaly_detection/api/routers/alerts.py` | `tests/api/test_alerts_router.py` |

### 3.2 Test Function Naming

All test functions follow the pattern:

```
test_<function_or_class_name>_<scenario>()
```

Examples:
- `test_build_sequence_window_returns_correct_length()`
- `test_score_fusion_above_threshold_sets_is_anomaly_true()`
- `test_alert_store_persists_and_retrieves_by_alert_id()`
- `test_batch_reader_strips_label_field_from_output()`

Do not use generic names like `test_it_works()` or `test_main()`.

### 3.3 Pytest Mark Conventions

Every test file must declare its tier mark in its module-level docstring:

```
"""
Tests for feature_engineering.session_builder.

Tier: T1
Pytest mark: @pytest.mark.tier1
"""
```

The three valid marks are: `@pytest.mark.tier1`, `@pytest.mark.tier2`, `@pytest.mark.tier3`.

A test file's tier mark must match the tier of the source file it tests. A T2 source file cannot have T1-marked tests (that would imply T1 depends on it).

### 3.4 Test Class Usage

- Use test classes (`class TestSessionBuilder:`) only when grouping multiple tests that share a non-trivial setUp/tearDown fixture. For simple unit tests, use plain functions.
- If a test class is used, it must be named `Test<ClassName>` where `ClassName` matches the class under test exactly.

### 3.5 Fixture Rules

- Shared fixtures across the entire test suite live in `tests/conftest.py`.
- Fixtures shared within a sub-package live in that sub-package's `conftest.py`.
- Fixtures must be named in `snake_case` and named after what they produce, not how they produce it: `entity_profile_fixture` not `mock_profile_creator`.
- Fixtures that produce boundary contract objects (e.g., a sample `AlertPayload`) must be named `sample_<boundary_class_name>` in lowercase: `sample_alert_payload`, `sample_entity_profile`.
- No test may call into a live database, live API, or live ML model. All external dependencies must be mocked or provided via fixtures.

### 3.6 Test Running Commands

| Command | Runs |
|---------|------|
| `pytest -m tier1` | Tier 1 tests only |
| `pytest -m "tier1 or tier2"` | T1 + T2 tests |
| `pytest` | Full suite |
| `pytest tests/feature_engineering/` | Single package tests |
| `pytest -k "score_fusion"` | Tests matching a keyword |

---

## 4. Docstring and Comment Conventions

### 4.1 Module Header (Required for Every Source File)

Every Python source file must begin with a module-level docstring in the following format:

```python
"""
<One-sentence description of what this module does.>

ARCHITECTURE COMPONENT: <Component name from ARCHITECTURE.md>
BOUNDARY RESPONSIBILITY: <Boundary letter(s) this module produces or consumes, e.g., "Produces C, Consumes B">
TIER: <T1 | T2 | T3>

<Optional: additional context, constraints, or cross-references.>
"""
```

Example:

```python
"""
Assembles session-level feature vectors from raw inbound events.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Consumes B (InboundEvent); contributes to producing C (EngineeredFeatures)
TIER: T1
"""
```

This header is the primary mechanism by which AI coding sessions and human reviewers verify that a file belongs to the correct component and respects the correct boundary contracts.

### 4.2 Class Docstrings (Required)

Every class must have a docstring that includes:
1. A one-sentence description of the class's responsibility.
2. If the class is an abstract base: a statement of the contract it enforces.
3. If the class is a boundary payload: the boundary letter it represents.

```python
class ScoreFusion:
    """
    Combines BPM and SDM model scores into a unified anomaly signal.

    Consumes boundary F (two ModelScore instances, one per model).
    Produces boundary G (UnifiedAnomalySignal).

    The fusion strategy (weighted average, max, learned combiner) is a
    configurable implementation detail; this class enforces the interface.
    """
```

### 4.3 Function Docstrings (Required for Public Functions)

Every public function (not prefixed with `_`) must have a Google-style docstring:

```python
def compute_risk_score(fused_score: float, cold_start: bool) -> int:
    """
    Converts a fused anomaly score to a 0–100 integer risk score.

    Args:
        fused_score: The unified anomaly signal score in range [0.0, 1.0].
        cold_start: If True, applies a confidence penalty to the raw score.

    Returns:
        An integer risk score in range [0, 100].

    Raises:
        BoundaryContractError: If fused_score is outside [0.0, 1.0].
    """
```

Private functions (`_prefixed`) require at minimum a one-line docstring unless the function name is entirely self-explanatory.

### 4.4 Boundary Contract Comments

Any line of code that directly produces or consumes a boundary contract payload must be preceded by a single-line comment citing the boundary:

```python
# Produces boundary G: UnifiedAnomalySignal
signal = UnifiedAnomalySignal(
    entity_id=entity_id,
    fused_score=fused_score,
    ...
)
```

```python
# Consumes boundary F: ModelScore from BPM
bpm_score: ModelScore = self._bpm.score(features)
```

This convention makes boundary violations immediately visible during code review and AI-session audits.

### 4.5 Tier Annotation Comments

Any code block that implements a Tier 2 or Tier 3 capability within a file that is otherwise Tier 1 must be marked with a comment block:

```python
# ── [T2] COLD-START PATH ──────────────────────────────────────────────────────
# This block is only reached when profile.cold_start_flag is True.
# The cold_start module is optional; the guard below makes T1 runnable without it.
try:
    from anomaly_detection.cold_start.handler import ColdStartHandler
    _cold_start_available = True
except ImportError:
    _cold_start_available = False
# ─────────────────────────────────────────────────────────────────────────────
```

### 4.6 Comment Rules

- Comments explain **why**, not **what**. The code itself explains what; comments explain intent, constraints, and non-obvious tradeoffs.
- No commented-out code committed to the repository. Use `git stash` or branches for work in progress.
- `TODO` comments are permitted during active development phases but must include a phase number: `# TODO(Phase 5): implement SHAP attribution here`.
- `FIXME` comments are prohibited in merged code.

---

## 5. Module Boundary Rules

### 5.1 Definition of a Module Boundary

For the purposes of this project's development process, a **module boundary** is defined as the interface between two ARCHITECTURE.md components at one of the named data contract boundaries (A through L). A module boundary is crossed whenever one Python package calls a function defined in another Python package and passes or receives a boundary contract payload.

The **boundary-owning file** for each boundary is:

| Boundary | Producing File | Consuming File |
|----------|---------------|----------------|
| A | `data_generator/generator.py` | `streaming/batch_reader.py` or `streaming/simulated_stream.py` |
| B | `streaming/batch_reader.py` (or `simulated_stream.py`) | `feature_engineering/session_builder.py` |
| C | `feature_engineering/sequence_builder.py` | `models/behavioral_profiling/inference.py`, `models/sequence_detection/inference.py` |
| D | `feature_engineering/profile_updater.py` | `stores/profile_store.py` |
| E | `stores/profile_store.py` | `models/behavioral_profiling/inference.py`, `models/sequence_detection/inference.py` |
| F | `models/behavioral_profiling/inference.py`, `models/sequence_detection/inference.py` | `models/fusion/score_fusion.py` |
| G | `models/fusion/score_fusion.py` | `classifier/inference.py`, `explainability/alert_builder.py` |
| H | `classifier/inference.py` | `explainability/alert_builder.py` |
| I | `explainability/alert_builder.py` | `stores/alert_store.py` |
| J | `api/routers/alerts.py`, `api/routers/entities.py` | `stores/alert_store.py` |
| K | `api/routers/alerts.py`, `api/routers/entities.py` | `dashboard/scripts/api_client.js` |
| L | `dashboard/scripts/api_client.js` | `api/routers/feedback.py` |

### 5.2 The "Files You May Modify / Must Not Touch" Contract

Every future implementation phase prompt must include an explicit statement of this form:

```
FILES YOU MAY CREATE OR MODIFY:
  - src/anomaly_detection/<package>/[file list]
  - tests/<package>/[file list]
  - config/[file if applicable]

FILES YOU MUST NOT MODIFY:
  - src/anomaly_detection/common/types.py  (unless the phase explicitly extends a type)
  - Any file outside the package(s) listed above
  - ARCHITECTURE.md, PROJECT_STRUCTURE.md, CODING_GUIDELINES.md

BOUNDARY CONTRACTS YOU MUST RESPECT (do not change field names or types):
  - Boundary [letter]: [payload class name] — as defined in common/types.py
```

This contract is non-negotiable. Any implementation phase that modifies a file outside its listed scope has violated the module boundary and must be reviewed before integration.

### 5.3 What Constitutes a Boundary Violation

The following are boundary violations that must be caught in code review:

1. A package importing from a package in a higher layer (e.g., `stores/` importing from `api/`).
2. A boundary payload class being defined in any file other than `common/types.py`.
3. A boundary field being renamed, added, or removed in a consuming file without the corresponding change in the producing file and `common/types.py`.
4. The `label` field from boundary A appearing in any file other than `data_generator/label_store.py` and `evaluation/evaluator.py`.
5. A Tier 2 or Tier 3 file being imported at the module top-level (not inside a guarded try/except) by a Tier 1 file.
6. Any file in `notebooks/` being imported by any file in `src/`.

### 5.4 Interface Contract for Abstract Base Classes

Every component that may have multiple implementations (stores, models, streaming) must define its interface as a Python abstract base class in a `base.py` file within its package. The abstract base class is the authoritative definition of the module boundary for that component. All type annotations on abstract method signatures must use types from `common/types.py` — never raw primitives where a named type exists.

The rule: **if a boundary payload crosses a `base.py` method signature, the payload must be a named type from `common/types.py`.**

### 5.5 What Each Future Phase Prompt Must Cite

At the start of every future implementation phase prompt, the following documents must be listed as inputs and must have been read:

1. `ARCHITECTURE.md` — for component identity and boundary contract field names.
2. `PROJECT_STRUCTURE.md` — for the exact file paths to create or modify.
3. `CODING_GUIDELINES.md` — for naming, import direction, and docstring format.

Any implementation that cannot be traced back to a specific component in ARCHITECTURE.md and a specific file in PROJECT_STRUCTURE.md is out of scope for that phase.

---

*End of CODING_GUIDELINES.md — Phase 2 output. This document is frozen. Amendments require a versioned change record.*
