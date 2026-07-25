# COLDSTART_DRIFT_STRATEGY.md
# AI-Powered Behavioral Anomaly Detection — Cold-Start & Drift Strategy

> **Status:** Phase 11 — Frozen ML Edge-Case Strategy  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** DATA_SCHEMA.md, ML_PIPELINE.md, EVAL_METRICS.md, API_SPEC.md, DASHBOARD_UX.md, ATTACK_TAXONOMY.md  
> **Scope:** Defines the mathematical and logical mechanisms for scoring zero-history entities (Cold-Start) and adapting to legitimate behavioral evolution (Concept Drift) without learning away slow attacks.

---

## Table of Contents

1. [Architecture Consistency Check](#1-architecture-consistency-check)
2. [Cold-Start Strategy (Entity-Type Fallback)](#2-cold-start-strategy-entity-type-fallback)
3. [Concept Drift Strategy (Gated EWMA)](#3-concept-drift-strategy-gated-ewma)
4. [Interaction with Insider Drift](#4-interaction-with-insider-drift)
5. [Compatibility Mapping to EVAL_METRICS.md](#5-compatibility-mapping-to-eval_metricsmd)
6. [Compatibility Mapping to API_SPEC.md & DASHBOARD_UX.md](#6-compatibility-mapping-to-api_specmd--dashboard_uxmd)
7. [Alternatives Considered](#7-alternatives-considered)
8. [Judging-Criteria Traceability](#8-judging-criteria-traceability)

---

## 1. Architecture Consistency Check

Prior to designing these mechanisms, a strict consistency check was performed against previous phases:

- ✅ **Compatibility with EVAL_METRICS.md:** The Cold-Start evaluation protocol (comparing PR-AUC of `cold_start_flag == True` vs `False`) is perfectly supported by the `event_count`-based graduation rule defined below. The Concept Drift evaluation (daily FPR of drifted entities) is directly supported by the Gated EWMA strategy, which aims to keep FPR low for legitimate shifts.
- ✅ **Compatibility with API_SPEC.md / DASHBOARD_UX.md:** The UX assumes the existence of `is_cold_start`, `profile_version`, and `drift_severity`. These exact fields are populated by the logic defined in Sections 2 and 3 and conform exactly to `DATA_SCHEMA.md` §5b (`EntityProfile`).
- **Verdict:** No mismatches. The strategy integrates seamlessly with the existing schemas, APIs, and UI designs.

---

## 2. Cold-Start Strategy (Entity-Type Fallback)

When a brand-new entity begins generating events, it lacks the historical data necessary to compute a personalized `baseline_vector` (used by the BPM) or a `sequence_history` (used by the SDM).

### Mechanism: Entity-Type Fallback Profile
1. **Offline Computation:** During the training phase (Days 1–21), the system aggregates all normal events and computes a global baseline profile for each `entity_type` (i.e., one fallback profile for `user`, one for `service_account`, one for `edge_device`).
2. **Inference Substitution:** If an incoming event has `SessionMetadata.is_cold_start == True`, the Feature Engineering layer dynamically substitutes the missing personalized `baseline_vector` and `baseline_std` with the corresponding global fallback profile for that `entity_type`.
3. **Confidence Discounting:** Because the entity is being scored against a generic population rather than its own habits, the `classification_confidence` output from the Anomaly Classifier is capped at a maximum of `0.5`. This mathematically forces cold-start alerts into the "Ambiguity Zone" defined in `EXPLAINABILITY.md`, triggering the UI to warn the analyst that the score is based on a population prior.

### Explicit Graduation Rule
An entity is considered "Cold-Start" if `EntityProfile.event_count < MIN_PROFILE_EVENTS` (default: 10 events).
- **Transition:** On the ingestion of the 10th event, the entity "graduates." 
- **Action:** The fallback profile is permanently discarded for this entity. A brand-new personalized `baseline_vector` and `baseline_std` are calculated exclusively using the entity's 10 historical events. `cold_start_flag` is set to `False`. All subsequent events (11+) are scored against this personalized baseline.

---

## 3. Concept Drift Strategy (Gated EWMA)

Legitimate behavior evolves (e.g., an employee moves to a new time zone or gets promoted to a new role). The baseline profile must adapt to prevent a permanent spike in False Positives.

### Mechanism: Gated Exponentially-Weighted Moving Average (EWMA)
Instead of a periodic full-database retrain, entity profiles adapt continuously via an EWMA applied to their `baseline_vector` after every event.

**The Update Formula:**
`new_baseline_vector = (1 - α) * old_baseline_vector + (α * current_feature_vector)`
Where `α` (alpha) is the learning rate (default: `0.05`, meaning the new event contributes 5% to the baseline, retaining a 95% memory of the past).

**The Gating/Resistance Mechanism (Crucial):**
If the system unconditionally applies the EWMA formula to every event, it will "learn" active attacks, causing the anomaly score of an ongoing attack to gradually drop back to 0. To prevent this, updates are strictly **Gated by Anomaly Score**:
1. The incoming event is scored by the BPM/SDM.
2. If `fused_score >= 0.4` (the lower boundary of the Ambiguity Zone): **The EWMA update is REJECTED.** The baseline is frozen.
3. If `fused_score < 0.4`: The event is deemed normal enough to represent legitimate behavior, and the EWMA update is applied. `profile_version` is incremented.

---

## 4. Interaction with Insider Drift

**The Challenge:** `ATTACK_TAXONOMY.md` defines "Insider Drift" as a slow, gradual accumulation of anomalous behaviors (e.g., slowly downloading files from uncharacteristic directories over weeks). A naive drift-adaptation mechanism would see these small daily changes as legitimate drift, update the baseline, and the attack would never trigger an alert.

**How this Strategy Defeats Insider Drift:**
The Gated EWMA mechanism uses a strict cutoff of `0.4`. 
- Legitimate drift (e.g., a schedule shift from 9 AM to 7 AM) changes a few features like `hour_of_day_sin`. This yields a minor anomaly score bump (e.g., `0.1` → `0.25`). Because `0.25 < 0.4`, the baseline updates. Over a few days, the score settles back to `0.1`.
- **Insider Drift** inherently accesses rare resources and unusual commands (`resource_rarity_score` and `command_rarity_score` spike). By design, the ML Pipeline scores these specific feature deviations heavily. 
- When the Insider Drift attack begins, the `fused_score` quickly reaches `0.4` (the Ambiguity Zone). 
- **The Trap:** Because the score hits `0.4`, the Gating Mechanism **freezes the profile**. The EWMA update is rejected. 
- As the insider continues their slow exfiltration over the next week, they are continually scored against the frozen baseline. The anomaly score creeps higher (`0.42`, `0.45`, `0.48`) until it finally breaches the hard alert threshold (`0.50`), triggering a High-severity alert. 
- **Defensible Conclusion:** By freezing the baseline the moment behavior enters the Ambiguity Zone, we mathematically guarantee that slow, stealthy attacks accumulate anomaly momentum until they are caught, while truly benign shifts adapt smoothly.

---

## 5. Compatibility Mapping to EVAL_METRICS.md

| Metric in `EVAL_METRICS.md` | How This Strategy Supports It |
|-----------------------------|-------------------------------|
| **Cold-Start vs Warm-Start PR-AUC** | By explicitly maintaining `cold_start_flag` up to the 10-event graduation point, evaluation scripts can cleanly partition the test set into Cold and Warm groups to compute these metrics. |
| **Drift Adaptation Timeline Plot** | Legitimate drifted entities (defined in the data generator) will score `< 0.4`, meaning their EWMA will update. Because their baseline adapts, their `fused_score` will stay below `0.5`, resulting in the required stable, low Daily False Positive Rate. |
| **Insider Drift Calibration** | Because the baseline freezes at `0.4`, Insider Drift events will cluster in the `0.4` to `0.6` range, perfectly satisfying the requirement that they land in the "Medium Risk / Ambiguity" calibration bucket rather than being learned as normal. |

---

## 6. Compatibility Mapping to API_SPEC.md & DASHBOARD_UX.md

| UX / API Expectation | How This Strategy Provides It |
|----------------------|-------------------------------|
| **"is cold-start" boolean** | Directly mapped to `EntityProfile.cold_start_flag`, which is driven by the `< 10` event graduation rule. |
| **"profile recently adapted" UI badge** | If the EWMA gating mechanism accepts an update, it increments `EntityProfile.profile_version`. The API compares the current `profile_version` to an older snapshot to populate `drift_severity = "low/medium"`, which triggers the UI badge. |
| **Ambiguity Flag UI logic** | The Cold-Start fallback explicitly forces a `max(0.5)` confidence penalty, passing the ambiguity requirement up through the API to the Dashboard detail view. |

---

## 7. Alternatives Considered

### Cold-Start
- **Alternative 1: Nearest-Neighbor Fallback.** (Find the closest existing entity profile using k-NN on static HR/asset data). *Rejected:* Running a k-NN search on every new entity creation adds severe latency to the real-time inference path and requires a new data index. Entity-Type Fallback is O(1) time complexity.
- **Alternative 2: Score as Zero/Normal unconditionally.** *Rejected:* Blindly trusting new entities creates a massive security hole. Using a population prior is standard practice and scientifically defensible.

### Concept Drift
- **Alternative 1: Periodic Full Retrain.** (Rebuild all baselines every Sunday night using the last 30 days of data). *Rejected:* If an attack happens on Monday, and the retrain runs on Sunday, the system learns the attack into the baseline by the next week. Also, batch retraining violates the near-real-time streaming ethos (Tier 2/3) of the project. Gated EWMA is computationally cheap and real-time.
- **Alternative 2: Unconditional EWMA.** (Update baseline on every event). *Rejected:* As explained in Section 4, this learns away Insider Drift. The Anomaly-Gated threshold is strictly necessary.

---

## 8. Judging-Criteria Traceability

| Hackathon Judging Criterion | Strategy Alignment |
|-----------------------------|--------------------|
| **Handling cold-start entities** | Addressed via the `Entity-Type Fallback` and the strict 10-event `Graduation Rule`, ensuring the system doesn't break or silently ignore brand-new users. |
| **Handling concept drift** | Addressed via `Gated EWMA`, allowing fluid legitimate changes to adapt in real-time without generating false positives. |
| **Explainability and analyst usability** | By forcing cold-start scores to carry a confidence penalty, we ensure the analyst is explicitly warned when a score is based on a population average rather than a personalized baseline. |
