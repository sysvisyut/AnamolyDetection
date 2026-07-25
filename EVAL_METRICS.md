# EVAL_METRICS.md
# AI-Powered Behavioral Anomaly Detection — Evaluation Methodology

> **Status:** Phase 8 — Frozen Evaluation Metrics Design  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** ARCHITECTURE.md, ML_PIPELINE.md, DATA_SCHEMA.md, EXPLAINABILITY.md, SYNTHETIC_DATA_GENERATOR_DESIGN.md  
> **Scope:** Defines the exact, unambiguous metrics, datasets, and protocols used to evaluate the ML pipeline against the hackathon judging criteria.

---

## Table of Contents

1. [Architecture Consistency Check & Generator Flag](#1-architecture-consistency-check--generator-flag)
2. [Core Detection Metrics (Imbalanced Binary)](#2-core-detection-metrics-imbalanced-binary)
3. [Alert-Budget Metric (Precision@1%)](#3-alert-budget-metric-precision1)
4. [Multi-Class Classification Evaluation](#4-multi-class-classification-evaluation)
5. [Cold-Start Evaluation Protocol](#5-cold-start-evaluation-protocol)
6. [Concept Drift Evaluation Protocol](#6-concept-drift-evaluation-protocol)
7. [Explainability Evaluation Approach](#7-explainability-evaluation-approach)
8. [Required Charts & Tables](#8-required-charts--tables)
9. [Alternatives Considered](#9-alternatives-considered)
10. [Judging-Criteria Traceability](#10-judging-criteria-traceability)

---

## 1. Architecture Consistency Check & Generator Flag

A thorough review of `DATA_SCHEMA.md`, `ML_PIPELINE.md`, and `SYNTHETIC_DATA_GENERATOR_DESIGN.md` was performed to ensure evaluation protocols can be seamlessly implemented. 

### Consistency Validations
- ✅ **Classification Output Match:** The evaluation metrics will consume `ClassificationResult` and `AlertPayload` as defined in `DATA_SCHEMA.md` §5a and `ML_PIPELINE.md`. The fields `fused_score`, `predicted_class`, `classification_confidence`, and `risk_score` exactly match.
- ✅ **Drift Support:** `SYNTHETIC_DATA_GENERATOR_DESIGN.md` §2e explicitly models legitimate gradual schedule shifts and role expansions, providing exactly the dataset required to evaluate concept drift handling (false positives on evolving normal behavior).

### 🚨 Incompatibility Flag: Zero-History Entity Generation
**Issue:** `SYNTHETIC_DATA_GENERATOR_DESIGN.md` §2a specifies that every entity generates events based on a Poisson distribution with `λ = 120` events per 30 days. Statistically, the probability of an entity generating fewer than 10 events (the `MIN_PROFILE_EVENTS` threshold for `cold_start_flag = True`) is effectively zero. The generator does not natively produce a true "Cold-Start" holdout group.

**Resolution / Proposed Addition to Phase 5:**
To evaluate cold-start handling, the generator implementation **must** partition a subset of entities (e.g., 5% of the population, designated as "Late Joiners"). These entities are hardcoded to generate **zero** events during the Days 1–21 training split, and only begin sampling from their Poisson distribution starting on Day 26 (the test split). This guarantees they trigger the population-prior fallback during evaluation. We will proceed with the cold-start protocol design (Section 5) relying on this addition.

---

## 2. Core Detection Metrics (Imbalanced Binary)

The first evaluation layer assesses the binary `is_anomaly` detection (normal vs. any attack).

### Why PR-AUC over ROC-AUC
At the default 1.5% attack injection rate, the dataset is highly imbalanced (98.5% normal).
- **ROC-AUC (Receiver Operating Characteristic Area Under Curve):** Plots True Positive Rate vs. False Positive Rate (FPR). Because the negative class (normal events) is massive, a large absolute number of false positives results in a tiny FPR change. ROC-AUC remains artificially high and paints an overly optimistic picture.
- **PR-AUC (Precision-Recall Area Under Curve):** Plots Precision vs. Recall. Precision (True Positives / (True Positives + False Positives)) explicitly penalizes false positives in the denominator. PR-AUC heavily punishes models that generate too many false alerts to achieve high recall, making it the definitive metric for imbalanced anomaly detection.

### Explicit Warning on Accuracy
Raw accuracy is banned as a headline metric for this project. 
**Numeric Proof:** With a 1.5% injection rate, a trivial classifier that hardcodes `return "normal"` for every event will achieve **98.5% accuracy**, despite detecting exactly zero attacks. Reporting 98.5% accuracy is deeply misleading to judges. Evaluation scripts must output a warning if raw accuracy is printed.

---

## 3. Alert-Budget Metric (Precision@1%)

SOC analysts cannot investigate thousands of alerts; they have a daily capacity limit. We operationalize this constraint as a top-K ranking problem, where `K = 1%` of all test events.

### Exact Computation Procedure

1. **Define the Budget (K):** Over the Days 26–30 test split, count total events `N_test`. The budget is `K = floor(0.01 * N_test)`.
2. **Rank:** Sort all test events descending by `AlertPayload.fused_score`.
3. **Tie-Breaking:** If multiple events have the exact same `fused_score` at the rank-K boundary:
   - First, sort by `classification_confidence` (descending).
   - Second, sort by a deterministic hash of the `event_id` to ensure reproducible sorting.
4. **Select:** The top `K` events constitute the "Alert Set".
5. **Compute Metrics:**
   - **Precision@1%:** (Count of true anomalies in Alert Set) / K
   - **Recall@1%:** (Count of true anomalies in Alert Set) / (Total true anomalies in test set)

### Fully Worked Numeric Example
- **Total Test Events:** 10,000
- **Total True Anomalies:** 150 (1.5%)
- **Alert Budget (K):** 1% of 10,000 = **100 events**
- The system ranks all 10,000 events by `fused_score` and takes the top 100.
- Upon checking labels, 85 of these 100 events are true anomalies (15 are false positives).
- **Precision@1%:** 85 / 100 = **85.0%** (85% of analyst time is spent on real threats).
- **Recall@1%:** 85 / 150 = **56.7%** (The top 1% queue catches over half the attacks).

---

## 4. Multi-Class Classification Evaluation

The second layer evaluates the classifier's ability to identify the *type* of attack (boundary H).

### Standard Methodology
- **Confusion Matrix:** An 8x8 matrix (7 attacks + normal). Row = True Label, Column = Predicted Class.
- **Per-Class Metrics:** Precision, Recall, and F1-Score computed individually for Brute Force, Impossible Travel, Credential Stuffing, Lateral Movement, Device Spoofing, and Low-and-Slow.
- **Macro-F1:** The unweighted average of the F1 scores for the classes (penalizes the model heavily if it fails entirely on a rare class like low-and-slow).

### Distinct Treatment for Insider Drift
Per `SYNTHETIC_DATA_GENERATOR_DESIGN.md` §3.7, Insider Drift is explicitly designed to be ambiguous, simulating legitimate role evolution mixed with potential insider threat. 
**Evaluation Rule:** Insider Drift is evaluated as a **Calibration Metric**, not a hard-accuracy metric.

1. **Exclusion from Binary Metrics:** Insider drift events are removed from the denominator when calculating the hard "Recall@1%" metric, because an ambiguous event *should not* rank in the absolute top 1% of highest-risk alerts.
2. **Calibration Goal:** The ideal system scores Insider Drift near the decision boundary. We measure what percentage of true Insider Drift events land in the `medium` risk tier (risk score 25–49) and measure their mean `classification_confidence`.
3. **Why this is honest:** Forcing Insider Drift into a binary "attack vs normal" rubric forces the model to confidently classify ambiguity. Treating it as a calibration target proves to judges that the system understands nuanced, medium-risk situations without spamming the high-priority queue.

---

## 5. Cold-Start Evaluation Protocol

**Objective:** Prove that the population-prior fallback works for new entities that lack a trained baseline profile.

### Test Design
Utilizing the "Late Joiner" entities defined in Section 1 (entities with no events in the training split). During the test split, these entities will have `cold_start_flag = True`.

### Metrics
Segment the test set into two groups: 
1. **Cold-Start Group:** Events where `cold_start_flag == True`.
2. **Warm-Start Group:** Events where `cold_start_flag == False`.

Compute **PR-AUC** and **Precision@1%** separately for both groups.
- **Success Criteria:** The Cold-Start PR-AUC will naturally be lower than the Warm-Start PR-AUC (proving that learning a specific baseline matters), but the absolute Cold-Start PR-AUC should remain significantly better than random guessing (proving the population-prior fallback is functional).

---

## 6. Concept Drift Evaluation Protocol

**Objective:** Prove the system dynamically updates its definition of "normal" without indefinitely raising false positive alerts when legitimate behavior changes.

### Test Design
`SYNTHETIC_DATA_GENERATOR_DESIGN.md` §2e generates normal entities undergoing "Gradual schedule shift" and "Role expansion". We will track false positive rates specifically on these entities.

### Metric: Daily False Positive Rate Timeline
Over the 30-day simulation window:
1. Isolate the subset of normal entities undergoing legitimate drift.
2. Compute the daily False Positive Rate (FPR) for this subset.
3. **Success Criteria:** If the system fails to adapt, the FPR will spike on Day ~15 as the drift becomes pronounced and stay high. A successful adaptive system will see the FPR remain stable and low (< 5%) over the entire 30 days, as the `profile_version` baselines seamlessly incorporate the new behavior.

---

## 7. Explainability Evaluation Approach

Explainability quality is qualitative by nature, but we will measure it using a hybrid automated/manual approach.

### Automated: Consistency Pass Rate
Phase 7 (`EXPLAINABILITY.md` §6) defined the `validate_explanation_consistency` function, which maps the top cited features to the predicted class.
- **Metric:** Run this validation over all True Positive alerts in the Alert Set. Measure the percentage that achieve `is_consistent == True`.
- **Target:** > 95% pass rate.

### Semi-Quantitative: Manual Spot-Review Rubric
Randomly sample 3 True Positive alerts from each of the 6 core attack types (18 alerts total). A human reviewer (or LLM-as-a-judge prompt during development) scores the `human_readable_explanation` on a 0–3 scale:
- **3 (Perfect):** Correct driver features cited, grammatically sound sentence, denormalized numbers make sense.
- **2 (Acceptable):** Correct driver features cited, but formatting is clunky or secondary features are noisy.
- **1 (Poor):** Missing the most critical feature (e.g., missed geo-velocity on impossible travel).
- **0 (Fail):** Incoherent sentence or attributes blame to completely irrelevant features.
- **Target:** Average score > 2.5.

---

## 8. Required Charts & Tables

This exact list will be handed to the Report and Presentation phases (Phases 19-20) to generate the final deliverables.

1. **Precision-Recall Curve (Overall):** Standard PR curve for the binary detector.
2. **Precision@k vs. k Plot:** X-axis is alert budget `k` (from 0.1% to 5.0%), Y-axis is Precision. Shows how precision degrades as analysts dig deeper into the queue.
3. **Multi-Class Confusion Matrix Heatmap:** 8x8 grid showing classification accuracy.
4. **Per-Class Performance Table:** Columns for Class, Precision, Recall, and F1.
5. **Cold-Start vs. Warm-Start PR-AUC Bar Chart:** Side-by-side comparison proving cold-start resilience.
6. **Drift Adaptation Timeline Plot:** Line chart showing daily FPR for drifted vs. stable entities over 30 days.
7. **Insider Drift Calibration Histogram:** Bar chart showing the distribution of risk tiers (Low, Medium, High) specifically for true Insider Drift events, demonstrating they land in the target 'Medium' bucket.

---

## 9. Alternatives Considered

1. **Evaluation Split: Chronological vs. K-Fold Cross-Validation**
   - *Considered:* 5-Fold CV to maximize use of data.
   - *Chosen:* Chronological (70% Train, 15% Val, 15% Test). Behavioral anomaly detection is a strict time-series problem. K-fold would cause data leakage by allowing the model to train on future events (e.g., training on post-drift data to predict pre-drift data). Since synthetic data is cheap to generate, we do not need K-fold to overcome small sample sizes.
2. **Metric: Fixed-Threshold (e.g., 0.5) vs. Budget-Ranked (Top-1%)**
   - *Considered:* Evaluating Precision/Recall strictly at `fused_score >= 0.5`.
   - *Chosen:* Budget-Ranked (Precision@1%). Fixed thresholds ignore the reality of SOC capacity. If a system generates 5,000 alerts > 0.5, the SOC will ignore most of them. Precision@1% reflects true operational value.
3. **Explainability: SHAP Force Plots vs. Narrative Consistency**
   - *Considered:* Outputting raw SHAP/Captum visualizations for the report.
   - *Chosen:* Narrative Consistency Pass Rate. The judging criteria specified "analyst usability." Raw force plots are harder for Tier 1 analysts to read than English narratives.

---

## 10. Judging-Criteria Traceability

| Hackathon Judging Criterion | Corresponding Metric / Protocol in this Document |
|-----------------------------|--------------------------------------------------|
| **Detection accuracy on imbalanced labels** | PR-AUC (Overall); Explicit ban on raw accuracy. |
| **Correct anomaly-type classification** | Multi-Class Confusion Matrix; Per-Class F1-Scores. |
| **FPR at realistic analyst alert budget** | Precision@1% and Recall@1% calculations. |
| **Explainability and analyst usability** | Consistency Pass Rate (>95%); Manual 0-3 Rubric. |
| **Handling cold-start entities** | Cold-Start vs Warm-Start PR-AUC comparison. |
| **Handling concept drift** | Drift Adaptation Timeline Plot (Stable FPR on drifted users). |
| **System design and scalability** | Chronological evaluation split (mimics scalable streaming). |
| **Report clarity** | Required Charts & Tables checklist (directly feeds final report). |
