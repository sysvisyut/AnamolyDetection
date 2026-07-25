# ML_PIPELINE.md
# AI-Powered Behavioral Anomaly Detection — ML Pipeline Design

> **Status:** Phase 6 — Frozen ML Pipeline Design Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Changelog:** 2026-07-25 — Added Step 5b (Gated EWMA ProfileStore.upsert()) per CONSISTENCY_REVIEW.md Blocking Issue #2.
> **Reads From:** ARCHITECTURE.md v1.0, DATA_SCHEMA.md v1.0, TECH_STACK.md v1.0, ATTACK_TAXONOMY.md v1.0  
> **Scope:** Concrete model design, training procedure, and pipeline wiring for BPM, SDM, Anomaly  
> Classifier, Score Fusion, and all extension points. No implementation code. Every decision here  
> must be implementable directly in the `models/` package without further design choices during coding.

---

## Table of Contents

1. [Architecture Consistency Check](#1-architecture-consistency-check)
2. [Behavioral Profiling Model](#2-behavioral-profiling-model-bpm)
3. [Sequence Detection Model](#3-sequence-detection-model-sdm)
4. [Anomaly Classifier](#4-anomaly-classifier)
5. [Class Imbalance Strategy](#5-class-imbalance-strategy)
6. [Cold-Start Extension Point](#6-cold-start-extension-point)
7. [Concept Drift Extension Point](#7-concept-drift-extension-point)
8. [Explainability Extension Point](#8-explainability-extension-point)
9. [Training and Evaluation Procedure](#9-training-and-evaluation-procedure)
10. [Alternatives Considered](#10-alternatives-considered)
11. [Judging-Criteria Traceability](#11-judging-criteria-traceability)
12. [Inference Pipeline Orchestrator Flow](#12-inference-pipeline-orchestrator-flow)

---

## 1. Architecture Consistency Check

DATA_SCHEMA.md Section 4 (Model I/O Contracts) and ATTACK_TAXONOMY.md were re-read in full before designing this pipeline. The following conformance properties are verified:

### 1a. Model I/O Contract Conformance

| Contract | Requirement (DATA_SCHEMA.md §4) | Pipeline Design Conformance |
|----------|--------------------------------|----------------------------|
| **BPM input** | `feature_vector`: `list[float]` length 24; `entity_profile`: `EntityProfile` (non-null) | BPM receives exactly a length-24 NumPy array derived from `feature_vector`, and reads the `baseline_vector` and `baseline_std` from the `EntityProfile`. No null profile is ever passed (Cold-Start Handler guarantee from ARCHITECTURE.md §6.1). |
| **BPM output (boundary F)** | `entity_id`, `event_id`, `model_id="bpm"`, `anomaly_score` ∈ [0,1], `confidence` ∈ [0,1], `cold_start_flag: bool`, `top_contributing_features: list[str]` max 5 | All 7 fields produced; anomaly score normalized to [0,1] via calibrated sigmoid; confidence derived from calibration certainty; features populated by SHAP. |
| **SDM input** | `sequence_window: torch.Tensor` shape (1, W, 24), W=20; `sequence_mask: torch.Tensor` shape (1, W) bool | GRU autoencoder receives exactly shape (1, 20, 24) float32; mask is passed to the encoder for padding exclusion from loss. |
| **SDM output (boundary F)** | Same 7 fields as BPM output with `model_id="sdm"` | All 7 fields produced; anomaly score derived from masked reconstruction error; confidence lower when padding fraction > 0.5; features from Captum Integrated Gradients. |
| **Score Fusion input** | Two `ModelScore` objects; `fusion_threshold` from config; `fusion_weights` summing to 1.0 | Score Fusion reads both ModelScore objects and config; produces `UnifiedAnomalySignal` (boundary G). |
| **Score Fusion output (boundary G)** | `entity_id`, `event_id`, `fused_score` ∈ [0,1], `is_anomaly: bool`, `bpm_score`, `sdm_score`, `cold_start_flag`, `contributing_features[]` | All fields produced; `is_anomaly = fused_score ≥ fusion_threshold`. |
| **Classifier input** | `fused_score`, `bpm_score`, `sdm_score` from boundary G + full `feature_vector` (24 dims) via pipeline context (§4d note) | Classifier receives a 27-dimensional input vector: [fused_score, bpm_score, sdm_score] + feature_vector[0..23]. |
| **Classifier output (boundary H)** | `entity_id`, `event_id`, `predicted_class` (one of 8 values), `class_probabilities{}` summing to 1.0, `classification_confidence` ∈ [0,1], `is_anomaly: bool` | All 6 fields produced; `class_probabilities` keys are all 8 label taxonomy values; `classification_confidence = max(class_probabilities.values())`. |

> **No contract changes required.** All model designs below conform to DATA_SCHEMA.md v1.0 without modification.

### 1b. Attack Signal Coverage Check

Verifying that the 24-dimensional feature vector (DATA_SCHEMA.md §3.2) carries sufficient signal for each attack type, as documented in ATTACK_TAXONOMY.md:

| Attack | Required Dims | BPM Can Carry Signal | SDM Required | Coverage Verdict |
|--------|--------------|---------------------|-------------|-----------------|
| Brute Force | 5, 11, 20 | ✅ Strong (burst failure count) | Helpful | ✅ Covered |
| Impossible Travel | 6, 7 | ✅ Strong (deterministic geo-velocity) | Helpful | ✅ Covered |
| Credential Stuffing | 5, 11, 22 | ⚠️ Moderate (needs cross-entity `ip_entity_ratio`) | Helpful | ✅ Covered (dim 22 is in feature vector) |
| Lateral Movement | 9, 12, 14, 21 | ✅ Moderate | Strengthens | ✅ Covered |
| Device Spoofing | 15, 16, 17 | ✅ Moderate (binary match dims) | Strengthens | ✅ Covered |
| Low-and-Slow | 0–3, 14, 19 | ❌ Weak individually | **Essential** | ✅ Covered by SDM sequence memory |
| Insider Drift | 9, 13, 21 | ❌ Very weak individually | **Essential** | ✅ Covered by SDM slow-gradient detection |

**Consistency verdict:** All model I/O contracts are satisfied. All 7 attack types have sufficient signal coverage across the 24 feature dimensions. No schema changes are proposed.

---

## 2. Behavioral Profiling Model (BPM)

### 2.1 Design Choice: Isolation Forest (Primary) + Z-Score Baseline (Secondary)

**Chosen model:** `sklearn.ensemble.IsolationForest`  
**Role:** Per-entity normality model. Learns what normal feature vectors look like for a given entity and assigns an anomaly score to each incoming event's feature vector.

**Tradeoff note (hackathon-aware):** A one-class autoencoder would learn richer non-linear normality representations, but requires a per-entity training loop, adding substantial implementation time for 500+ entities. Isolation Forest is trained in a single batch per entity, is deterministic given a random seed, requires no hyperparameter tuning per entity, and performs well on tabular high-dimensional data out of the box. The Z-score baseline provides a lightweight fallback and a secondary signal for SHAP attribution.

### 2.2 What It Models

For each entity, the BPM learns the distribution of normal `feature_vector` (24-dimensional) observations over the entity's historical event window. It assigns an outlier score to new events based on how isolated they are from the entity's learned normal region.

The BPM is **not** a global model — it is a **per-entity model** (or, for cold-start entities, a per-peer-group model). The Entity Profile Store (boundary E) stores the serialized model artifact for each entity alongside the profile.

### 2.3 Architecture / Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `n_estimators` | 200 | Sufficient for 24-dimensional space; diminishing returns above 200 |
| `max_samples` | `min(256, n_training_events)` | Capped at 256 for entities with limited history; uses all available events if <256 |
| `contamination` | 0.01 | Informed by problem statement's 0.5%–3% injection rate; 1% is the midpoint estimate for initial threshold calibration. This parameter only affects the `predict()` threshold, not the anomaly score — the raw `score_samples()` output is always used. |
| `max_features` | 1.0 | Use all 24 features per split |
| `random_state` | Per-entity seed from `entity_seed` in config | Reproducible per-entity model artifacts |

**Z-Score Secondary Signal:**  
Alongside Isolation Forest, maintain a per-entity **Z-score anomaly score** computed as:  
`z_score = max over all 24 dims of |feature_vector[d] - baseline_vector[d]| / max(baseline_std[d], ε)`  
where ε = 0.01 (minimum std to avoid division by zero).  
Normalize: `z_score_norm = min(z_score / 5.0, 1.0)` (capped at 5σ → 1.0).  
This secondary score is used by the SHAP attribution (Section 8) as a reference and is available to the Score Fusion layer as an observability signal. It is **not** part of the primary `anomaly_score` field in boundary F.

### 2.4 Anomaly Score Normalization

Raw Isolation Forest `score_samples()` output is in the range approximately [-0.5, 0.5] (more negative = more anomalous). Normalize to [0, 1] as follows:

1. During training, collect the distribution of raw scores over the training set: compute the 1st and 99th percentiles as `score_min` and `score_max`.
2. Apply: `anomaly_score = clip((score_max - raw_score) / (score_max - score_min), 0.0, 1.0)`
   - This maps: highest raw score (most normal) → 0.0; lowest raw score (most anomalous) → 1.0.
3. Store `score_min` and `score_max` in the model artifact alongside the IsolationForest object.
4. At inference, apply the same calibration: if `raw_score < score_min`, clip to 1.0; if `raw_score > score_max`, clip to 0.0.

**Confidence field:** Set `confidence = 1.0 - cold_start_flag × 0.4`. For non-cold-start entities, `confidence = 1.0`. For cold-start entities, `confidence = 0.6`. This communicates reduced reliability to the Score Fusion and Explainability layers.

### 2.5 Training Procedure

**Training granularity:** One IsolationForest model artifact per entity.

**Training data:** All `normal`-labeled `feature_vector` records for the entity in the training split. The label `normal` is used here **only to select which events to train on** — the model itself is unsupervised (one-class). No anomaly events are included in BPM training.

**Minimum training events:** An entity must have at least `MIN_PROFILE_EVENTS = 10` normal events in the training split to receive a dedicated model. Entities below this threshold have `cold_start_flag = True` and receive the population-prior model (Section 6).

**Procedure:**
1. For each entity with ≥ 10 normal training events:
   a. Assemble a 2D NumPy array of shape `(N_events, 24)` from all normal `feature_vector` records.
   b. Fit `IsolationForest` with the parameters above.
   c. Compute calibration percentiles (`score_min`, `score_max`) on the training set.
   d. Serialize: `{model: IsolationForest, score_min: float, score_max: float}` → pickle to `models/behavioral_profiling/artifacts/bpm_<entity_id>.pkl`.
2. For entities with < 10 events: set `cold_start_flag = True` in profile; Phase 11 Cold-Start Handler provides the prior model.
3. One **population-level** IsolationForest per entity type (`user`, `service_account`, `edge_device`) is also trained on all normal events for that type. These serve as the cold-start fallback models.

**Training time estimate:** Fitting one IsolationForest(n_estimators=200) on 100 samples in 24 dimensions takes ~0.05 seconds on a modern CPU. For 500 entities: 500 × 0.05 = ~25 seconds total. Well within hackathon constraints.

### 2.6 Inference Procedure

For each incoming `EngineeredFeatures` object at boundary C:
1. Look up the entity's `EntityProfile` (boundary E) → check `cold_start_flag`.
2. If not cold-start: load `bpm_<entity_id>.pkl`; compute `raw_score = model.score_samples([feature_vector])[0]`.
3. If cold-start: load `bpm_population_<entity_type>.pkl` (Section 6).
4. Normalize `raw_score` to `anomaly_score` using stored `score_min`/`score_max`.
5. Run SHAP TreeExplainer on the same input → extract top-5 feature names by absolute SHAP value.
6. Produce `ModelScore` (boundary F): `{entity_id, event_id, model_id="bpm", anomaly_score, confidence, cold_start_flag, top_contributing_features}`.

**Model loading strategy:** Model artifacts are loaded once and cached in memory at server startup using a Python dict keyed by `entity_id`. On entity profile update (new events processed), the cache is invalidated and the model is re-fitted. This avoids per-request disk I/O.

---

## 3. Sequence Detection Model (SDM)

### 3.1 Design Choice: GRU Autoencoder

**Chosen architecture:** GRU-based sequence autoencoder  
**Framework:** PyTorch (locked in TECH_STACK.md Decision 3)  
**Role:** Learns normal event sequences for each entity type (not per-entity — see Section 3.4 for justification) and assigns a reconstruction-error-based anomaly score to each incoming sequence window.

**Tradeoff note (hackathon-aware):** A per-entity GRU autoencoder would be most precise but requires training 500+ individual models. A single global GRU autoencoder trained on all entities conflates different behavioral profiles. The chosen design trains **one GRU autoencoder per entity type** (3 models total), giving meaningful separation (user vs. service_account vs. edge_device behavioral patterns are very different) without the cost of 500 individual training runs.

### 3.2 Input Specification

**Exactly as specified in DATA_SCHEMA.md §4b and §3.3:**
- Input tensor: `torch.Tensor` of shape `(B, W, F)` = `(batch_size, 20, 24)`, dtype=float32
- Mask tensor: `torch.Tensor` of shape `(B, W)`, dtype=bool; `True` = real event, `False` = zero-padding
- W = 20 (sequence window length, locked in DATA_SCHEMA.md §3.3)
- F = 24 (feature vector dimensions, locked in DATA_SCHEMA.md §3.2)
- Zero-padding: left-padded for entities with fewer than W historical events (mask is False for padding positions)

### 3.3 Architecture

```
GRU Autoencoder
=================

ENCODER:
  Layer 1: GRU(input_size=24, hidden_size=64, num_layers=2, batch_first=True,
                dropout=0.2 between layers)
  → Output: hidden state h_enc of shape (B, 64) [take final non-padded step's hidden state]
  → Padding exclusion: use the mask to select the last real (non-padded) hidden state via
    index: h_enc = encoder_output[arange(B), last_real_idx, :]
    where last_real_idx = mask.sum(dim=1) - 1 (0-indexed position of last real event)

BOTTLENECK:
  h_enc shape: (B, 64) — this is the learned behavioral latent representation

DECODER:
  Layer 1: Repeat h_enc W=20 times → shape (B, 20, 64)
  Layer 2: GRU(input_size=64, hidden_size=64, num_layers=2, batch_first=True,
                dropout=0.2 between layers)
  Layer 3: Linear(in_features=64, out_features=24)  ← reconstructs each timestep
  → Output: reconstructed_sequence of shape (B, W, F) = (B, 20, 24)

TOTAL PARAMETERS:
  Encoder GRU (2 layers): approximately 24×4×64 + 64×4×64 + (GRU layer 2: 64×4×64×2) ≈ 56,832
  Decoder GRU (2 layers): approximately 64×4×64 + 64×4×64 + (GRU layer 2) ≈ 49,152
  Linear: 64×24 + 24 = 1,560
  Total: approximately 107,544 parameters — easily trainable in <10 minutes on CPU for 60,000 sequences
```

### 3.4 Why Per-Entity-Type Rather Than Per-Entity or Global

| Scope | Models Trained | Training Time | Behavioral Precision | Verdict |
|-------|---------------|---------------|---------------------|---------|
| Per-entity | 500 | ~500 × minutes = impractical | Highest | Rejected (hackathon constraint) |
| Global | 1 | ~5 minutes | Lowest (mixes all behaviors) | Rejected (too coarse) |
| Per-entity-type | 3 | ~15 minutes total | Good (user/svc/device separated) | **Chosen** |

The three entity types have fundamentally different behavioral patterns:
- `user`: variable timing, multiple geos, diverse resources, command sequences
- `service_account`: near-deterministic timing, single geo, narrow resources, no commands
- `edge_device`: near-deterministic, fixed geo, single resource, no commands

Training one model per type means the GRU autoencoder learns the specific temporal regularity of each type, making deviations from that regularity more detectable.

### 3.5 Loss Function

**Primary loss:** Masked Mean Squared Error (MSE) over real (non-padded) timesteps only:

```
masked_mse = sum over t where mask[b,t]=True of MSE(reconstructed[b,t,:], input[b,t,:])
             / count of True positions in mask
```

Using the mask in the loss is critical: if padding positions contributed to the loss, the model would learn to reconstruct zeros, which would incorrectly reward padding positions and bias reconstruction errors toward the real events.

**No separate regularization loss.** The GRU dropout (0.2) provides sufficient implicit regularization for the sequence lengths and dataset size involved.

### 3.6 Anomaly Score Derivation

At inference, the SDM computes the **per-event reconstruction error** for each non-padded position:
```
per_event_error[t] = MSE(reconstructed[t,:], input[t,:])  for all t where mask[t]=True
```

Aggregate to a **session-level anomaly score**:
```
raw_sdm_score = mean(per_event_error over non-padded positions)
              + max(per_event_error over non-padded positions) × 0.3
```
The 30% max-error component ensures that a single highly anomalous event in a window (e.g., a sudden brute-force event in an otherwise normal sequence) is not fully averaged away by the surrounding normal events.

**Normalization to [0, 1]:** Same percentile calibration approach as the BPM:
1. Compute the 1st and 99th percentiles of `raw_sdm_score` over the training set: `err_min`, `err_max`.
2. `anomaly_score = clip((raw_sdm_score - err_min) / (err_max - err_min), 0.0, 1.0)`.
3. Store `{err_min, err_max}` in the model artifact.

**Confidence:** `confidence = real_event_count / W`. If the window is fully padded (0 real events), confidence = 0.0. If `real_event_count / W < 0.5`, `cold_start_flag = True` for the SDM.

### 3.7 Training Procedure

**Data preparation:**
1. For each entity in the training split, construct all possible sliding windows of length W=20 from the chronological sequence of `feature_vector` records, stepping by 1 event at a time.
2. Left-pad windows for entities with fewer than W historical events.
3. Label for training: **normal events only**. All sequences containing at least one event from the anomaly-labeled training set are excluded from SDM training.
4. Assemble per-entity-type DataLoaders.

**Training:**
- Optimizer: Adam, learning rate = 1e-3, weight_decay = 1e-5
- Batch size: 64 (sequences, not events)
- Epochs: 50 (with early stopping: patience = 5 epochs; monitor: validation masked MSE)
- Learning rate scheduler: ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
- Validation set: the chronological 15% validation split (Section 9); sequences from validation entities only.

**Training time estimate:**  
~60,000 events / 500 entities = ~120 events per entity average.  
Sliding windows: ~100 windows per entity (stepping by 1) × 350 user entities = ~35,000 user sequences + ~12,000 service + ~6,000 device = ~53,000 total sequences.  
GRU forward + backward pass on 64-sequence batches: ~830 batches per epoch × 50 epochs × ~0.3s per batch on CPU = ~12 minutes per entity type × 3 = ~36 minutes total. Acceptable for hackathon training.

**Model artifact:** Saved as `models/sequence_detection/artifacts/sdm_<entity_type>.pt` using `torch.save({model_state_dict, err_min, err_max, W, F})`.

### 3.8 Inference Procedure

For each incoming `EngineeredFeatures` at boundary C:
1. Look up entity type from `EntityProfile` (boundary E).
2. Load `sdm_<entity_type>.pt` (pre-loaded in memory cache at startup).
3. Retrieve `sequence_window` from `EngineeredFeatures` → convert to `torch.Tensor` shape (1, 20, 24), float32.
4. Construct `sequence_mask` from the Entity Profile's `sequence_history` length: True for the last N positions (where N = min(len(sequence_history), W)), False for the first (W - N) positions.
5. Run forward pass (encoder → decoder) in `torch.no_grad()` context.
6. Compute masked reconstruction error → `raw_sdm_score` → normalize to `anomaly_score`.
7. Run Captum Integrated Gradients on the encoder path → top-5 feature names by mean absolute gradient across sequence positions.
8. Produce `ModelScore` (boundary F): `{entity_id, event_id, model_id="sdm", anomaly_score, confidence, cold_start_flag, top_contributing_features}`.

---

## 4. Anomaly Classifier

### 4.1 Design Choice: Separate LightGBM Multi-Class Classifier

**Chosen design:** A standalone LightGBM (`lightgbm.LGBMClassifier`) multi-class classifier trained on the labeled training dataset.

**Why separate (not multi-head, not hybrid):** See Section 10.2 for full alternatives analysis. The key reasons:
- The classifier receives a richer input than BPM/SDM alone: the full 24-dim feature vector plus fused/model scores (27 total features). A multi-head extension of the GRU autoencoder would only have access to the sequence reconstruction path, not the static feature vector.
- The classifier is independently retrain-able. If the attack taxonomy changes, the classifier is retrained without touching BPM or SDM artifacts.
- LightGBM trains to convergence in seconds on the dataset sizes involved, requires no GPU, and natively handles class imbalance via `class_weight`.
- LightGBM's native feature importance provides a second attribution signal alongside SHAP.

**Justification of LightGBM over scikit-learn alternatives:**  
- XGBoost: comparable accuracy but slower training and no native histogram-based binning for large feature sets.
- RandomForest: less sensitive to rare classes; lower precision at high thresholds.
- Logistic Regression: insufficient for the non-linear feature interactions (e.g., `resource_rarity_score` × `has_exfil_command` interaction for lateral movement vs. low-and-slow).

### 4.2 Classifier Input (27 Features)

The classifier receives a concatenated 27-dimensional vector at inference time:

| Index | Feature | Source |
|-------|---------|--------|
| 0 | `fused_score` | Boundary G (`UnifiedAnomalySignal`) |
| 1 | `bpm_score` | Boundary G |
| 2 | `sdm_score` | Boundary G |
| 3–26 | `feature_vector[0..23]` | Full 24-dim feature vector from boundary C (passed via pipeline context as specified in DATA_SCHEMA.md §4d) |

This 27-feature design allows the classifier to use both the detector's combined judgment (`fused_score`) and the raw feature evidence for each attack type. Without the 24 raw dimensions, the classifier would need to infer `geo_velocity_kmph` (dim 6) from the fused score alone — which is insufficient for distinguishing impossible travel from brute force when both produce high fused scores.

### 4.3 Target Classes

The classifier predicts one of **8 classes** (the full label taxonomy from DATA_SCHEMA.md §2a):

```
0: normal
1: brute_force
2: impossible_travel
3: credential_stuffing
4: lateral_movement
5: device_spoofing
6: low_and_slow
7: insider_drift
```

**Important note:** The classifier is trained on **all events** (both normal and anomaly), not just on events flagged as anomalous by the detector. This is intentional: the classifier must learn to say `normal` for events the detector scores below threshold. This avoids a two-stage failure mode where the detector lets a false positive through and the classifier then assigns it a high-confidence attack class.

### 4.4 Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `n_estimators` | 500 | Standard for gradient boosting on tabular data |
| `learning_rate` | 0.05 | Conservative; combined with 500 estimators gives good convergence |
| `max_depth` | 6 | Sufficient for 27-feature interactions; prevents overfitting on small attack classes |
| `num_leaves` | 31 | LightGBM default; appropriate for depth=6 |
| `class_weight` | Inverse frequency (see Section 5) | Critical for 0.5%–3% positive rate |
| `objective` | `multiclass` | Outputs softmax probabilities over all 8 classes |
| `metric` | `multi_logloss` | Calibrated probability output |
| `num_class` | 8 | |
| `random_state` | `random_seed` from config | Reproducibility |
| `n_jobs` | -1 | Use all available CPU cores |

### 4.5 Output Contract (Boundary H)

Exactly as specified in DATA_SCHEMA.md §4d:

```
ClassificationResult:
  entity_id:                  str    — propagated from boundary G
  event_id:                   str    — propagated from boundary G
  predicted_class:            str    — argmax(class_probabilities)
  class_probabilities:        dict[str, float]  — all 8 label values; sum = 1.0
  classification_confidence:  float  — max(class_probabilities.values())
  is_anomaly:                 bool   — propagated from UnifiedAnomalySignal.is_anomaly
```

**Handling the `is_anomaly` flag with the `normal` class:**  
If `is_anomaly = True` (fused_score ≥ threshold) but the classifier's highest-probability class is `normal`, the classifier decision is overridden: `predicted_class` is set to the second-highest probability class. This resolves a potential conflict where the detector says "anomalous" but the classifier says "normal" — the detector's binary decision takes precedence, and the classifier provides the attack type. This override logic is implemented in `models/anomaly_classifier/inference.py` and is documented in the docstring.

Conversely, if `is_anomaly = False`, the classification may still be run (for observability and evaluation purposes), but `predicted_class = "normal"` is enforced regardless of the classifier output, and the result is not surfaced as an alert.

### 4.6 Training Procedure

1. Assemble training data:
   - For every event in the training split, construct the 27-feature input: [fused_score, bpm_score, sdm_score, feature_vector[0..23]].
   - Note: `fused_score`, `bpm_score`, and `sdm_score` in the training set are the **outputs of the trained BPM and SDM run in inference mode on the training data**. This means the classifier is trained on the BPM/SDM's actual outputs, not on ideal scores. The order of training is therefore: (1) train BPM, (2) train SDM, (3) run BPM+SDM on training data to get scores, (4) train classifier.
   - Target: `label` from the training schema (this is the only component where the `label` field is used in training).
2. Apply class weights (Section 5).
3. Fit LGBMClassifier on the assembled (X, y) arrays.
4. Evaluate on the validation split using micro/macro F1, per-class precision/recall, and AUROC (one-vs-rest).
5. Save artifact: `models/anomaly_classifier/artifacts/classifier.pkl`.

---

## 5. Class Imbalance Strategy

### 5.1 The Imbalance Problem

At a 1.5% injection rate (default) with 8 classes:
- `normal`: ~98.5% of events
- Each attack class: ~0.21% average (7 classes × 0.21% = 1.47% total anomalies)
- `insider_drift` and `low_and_slow`: ~0.10% each (lower share per SYNTHETIC_DATA_GENERATOR_DESIGN.md §4.1)

This is a severely imbalanced multi-class problem. A naive classifier trained without correction will predict `normal` for nearly all events.

### 5.2 Strategy: Dual-Layer Approach

**Layer 1 — BPM (Unsupervised, No Label Required):**  
Isolation Forest is inherently suited to imbalanced data because it is trained **only on normal events**. It does not see anomaly labels during training. The contamination parameter (0.01) is used only to calibrate the prediction threshold, not to train the model.

**Layer 2 — Classifier (Supervised, Label Required):**  
Two complementary techniques applied to the classifier:

**Technique A: Inverse-Frequency Class Weights**  
Compute class weight for class c as:  
`weight[c] = total_events / (n_classes × count_events_with_label_c)`  
Pass these weights to LGBMClassifier via `class_weight` parameter (as a dict keyed by class index). This mathematically equalizes the contribution of each class to the gradient during training, regardless of its frequency.

**Technique B: SMOTE on Minority Attack Classes (Training Only)**  
Before fitting the classifier, apply SMOTE (`imblearn.over_sampling.SMOTE`) to the training feature matrix to oversample minority attack classes to 10× their natural frequency (or to the next most frequent class's count, whichever is smaller).  
**SMOTE is applied only to the 7 attack classes** (not to `normal`). The `normal` class is already the majority and must not be oversampled.  
SMOTE random seed: `random_seed` from config.

**Why both techniques together?**  
- Class weights alone penalize misclassifying rare attacks but do not solve the training data sparsity problem (the classifier never sees many examples of e.g. `low_and_slow`).
- SMOTE alone without class weights can create synthetic examples in feature regions that are ambiguous, potentially increasing false positives.
- Combined: SMOTE increases the effective training sample for rare classes; class weights then ensure the loss function treats the expanded rare-class samples with appropriate weight.

### 5.3 Connection to the "Top 1% Alert Budget" Operating Point

The problem statement and judging criteria require evaluating the system at a realistic analyst alert budget — meaning: rank all events by `fused_score` and evaluate precision/recall for the top-scoring 1% of events.

This evaluation mode requires a well-calibrated **continuous** anomaly score, not a binary prediction. The design choices that ensure this:

1. **BPM produces continuous scores:** The IsolationForest's `score_samples()` output is a real-valued anomaly score before any threshold is applied. The percentile normalization (Section 2.4) ensures the score is well-distributed across [0, 1] for the training population.

2. **SDM produces continuous scores:** The masked reconstruction error is similarly normalized to [0, 1].

3. **Score Fusion preserves continuous signal:** The weighted average of BPM and SDM scores (default 0.5/0.5) produces a continuous `fused_score`. The `fusion_threshold` is applied to produce the binary `is_anomaly` flag, but the continuous `fused_score` is always stored in `AlertPayload` and used for ranking.

4. **Ranking at the top 1% operating point:**  
   At evaluation time, sort all events by `fused_score` descending. Select the top 1% (by count) as the "alert set." Compute:
   - Precision@1%: what fraction of the top-1% events are true anomalies?
   - Recall@1%: what fraction of all true anomalies are in the top-1%?
   
   At 1.5% injection rate, a perfect system would have ~1.5 true anomalies per 100 events. A top-1% alert budget captures 1% of events. For precision@1% = 100%, the system would need all top-1% alerts to be true anomalies — this is achievable for easily separable attacks (Brute Force, Impossible Travel) but not for Insider Drift (which overlaps with normal behavior by design).

5. **Threshold tuning from the continuous score:** The default `fusion_threshold = 0.5` is a starting point. The Model Evaluation Module (T2) produces a precision-recall curve over all threshold values, allowing the analyst to select the threshold that achieves their desired operating point. The `risk_tier` bucketing in `AlertPayload` (0–24/25–49/50–74/75–100) provides a coarser pre-computed version of this threshold sweep that the dashboard can use directly.

---

## 6. Cold-Start Extension Point

**Full design:** Deferred to Phase 11.  
**This section documents the pipeline's interface requirements for the cold-start handler.**

### 6.1 Trigger

The cold-start condition is signaled by `EntityProfile.cold_start_flag = True`, which is set when `EntityProfile.event_count < MIN_PROFILE_EVENTS` (default: 10, configurable in `config/default.yaml`).

### 6.2 Pipeline Attachment Point

The Cold-Start Handler attaches at boundary E (Entity Profile Store → BPM). The BPM's inference procedure (Section 2.6, step 1) checks `cold_start_flag` before loading the per-entity model. If `cold_start_flag = True`:

1. The Cold-Start Handler is invoked: it must produce a **valid EntityProfile** and load the corresponding **population-level BPM** (`bpm_population_<entity_type>.pkl`, which is trained as part of the standard BPM training procedure in Section 2.5).
2. The population-level BPM is then used exactly as the per-entity BPM would be — the inference path is identical.
3. `cold_start_flag = True` is propagated to boundary F (`ModelScore`), then to boundary G, then to boundary I (`AlertPayload`).

### 6.3 Phase 11 Decision Points

Phase 11 will design and choose between:
- **Population-prior profile:** Aggregate behavioral baseline over all entities of the same `entity_type`. Simple, low-bias, but high-variance for entities with unusual behavior.
- **Peer-group profile:** Cluster entities by persona variant (`executive`, `developer`, etc.) and use cluster-level baselines. More precise but requires clustering.
- **Zero-shot heuristic:** Rule-based thresholds derived directly from the config's distribution parameters (Section 2 of SYNTHETIC_DATA_GENERATOR_DESIGN.md). Directly interpretable but not learned from data.

**Interface invariant guaranteed now (Phase 6):** The BPM will never receive a null or missing EntityProfile. The cold-start handler is the guard. This invariant must be enforced in `models/behavioral_profiling/inference.py` with an assertion at the top of the inference function.

---

## 7. Concept Drift Extension Point

**Full design:** Deferred to Phase 11.  
**This section documents the pipeline's interface requirements for the drift monitor.**

### 7.1 Attachment Point

The Drift Monitor attaches to the Entity Profile Store as a **passive read-only observer** (ARCHITECTURE.md §6.2 constraint). It cannot modify profiles. It reads rolling snapshots of each entity's `baseline_vector` history (stored in `EntityProfile.drift_metrics.feature_means_history`) and detects distributional shift.

### 7.2 How the Pipeline Supports Drift Detection

The following pipeline elements are designed to enable drift detection without Phase 11 implementing anything new:

1. **`EntityProfile.profile_version`** (DATA_SCHEMA.md §5b): incremented on every profile upsert by `feature_engineering/session_builder.py`. The Drift Monitor uses this to track when baselines change.

2. **`EntityProfile.drift_metrics.feature_means_history`**: a list of the last K `baseline_vector` snapshots. K is configurable (default 10). Feature Engineering appends a snapshot every time the baseline is updated. The Drift Monitor computes population-level distribution divergence from this history.

3. **`ModelScore.confidence`** (boundary F): when drift is suspected (Phase 11 decision), the Drift Monitor can signal the BPM/SDM to lower their confidence output for the affected entity, propagating the uncertainty downstream to `AlertPayload.risk_score`.

4. **Profile versioning for retrain isolation:** A drift-triggered retraining event creates a new BPM artifact with a new filename suffix (`bpm_<entity_id>_v<profile_version>.pkl`), not overwriting the previous version. This preserves the previous baseline for post-hoc comparison.

### 7.3 Phase 11 Decision Points

Phase 11 will choose and implement the drift detection algorithm (e.g., ADWIN, Page-Hinkley on reconstruction error, or Population Stability Index on feature distributions) and the retrain trigger policy. The current pipeline provides all necessary data structures.

---

## 8. Explainability Extension Point

**Full design:** Phase 7.  
**This section documents where and how attribution attaches to this pipeline.**

### 8.1 BPM Attribution: SHAP TreeExplainer

**Attachment point:** Inside `models/behavioral_profiling/inference.py`, immediately after computing `anomaly_score` (Section 2.6, step 5).

**Method:** `shap.TreeExplainer(isolation_forest_model).shap_values(feature_vector_array)`.

SHAP TreeExplainer is directly compatible with IsolationForest and runs in O(T × D) time where T = number of trees (200) and D = depth (typically log2(max_samples)). For a single event (1 × 24 array), SHAP inference takes ~0.5ms on CPU — not a bottleneck.

**Output:** SHAP values of shape `(1, 24)` — one signed attribution value per feature dimension. The BPM passes the top-5 absolute-value feature names as `top_contributing_features` in boundary F.

**Phase 7 responsibility:** Constructing the `FeatureAttribution` objects (DATA_SCHEMA.md §5d) from the raw SHAP values, including the `human_label` mapping and `direction` field. This is handled by `explainability/feature_attribution.py`.

### 8.2 SDM Attribution: Captum Integrated Gradients

**Attachment point:** Inside `models/sequence_detection/inference.py`, after the forward pass (Section 3.8, step 7).

**Method:** `captum.attr.IntegratedGradients(forward_func=encoder_forward_function).attribute(inputs=sequence_tensor, baselines=torch.zeros_like(sequence_tensor), target=None)`.

The encoder is defined as the attribution target (the bottleneck representation `h_enc`). Integrated Gradients computes the average gradient of `h_enc` norm with respect to each input position × feature dimension over 50 interpolation steps.

**Attribution aggregation:** The raw IG attribution tensor has shape `(1, 20, 24)`. Aggregate to feature-level by taking the **mean absolute gradient across all real (non-padded) sequence positions**:  
`feature_importance[d] = mean over real positions t of |attribution[0, t, d]|`  
→ yields shape `(24,)` — one importance value per feature dimension.

Top-5 feature names by `feature_importance` populate `top_contributing_features` in boundary F.

**Phase 7 responsibility:** Same as BPM — constructing `FeatureAttribution` objects and the narrative template logic.

### 8.3 Classifier Attribution

**Not computed in the ML pipeline.** The LightGBM classifier's native feature importances are computed at training time (gain-based) and stored in the model artifact. These are used by Phase 7 to augment the `human_readable_explanation` for the predicted attack class. Per-inference LightGBM SHAP attribution is available but deferred to Phase 7's performance evaluation.

---

## 9. Training and Evaluation Procedure

### 9.1 Data Split Strategy

The synthetic dataset is split **chronologically** (not randomly). Random shuffling would allow future events to inform past models, which is train/test leakage for time-series behavioral data.

| Split | Proportion | Simulation Days | Purpose |
|-------|-----------|----------------|---------|
| Training | 70% | Days 1–21 | BPM fitting, SDM fitting, Classifier fitting |
| Validation | 15% | Days 22–25 | Early stopping, threshold calibration, hyperparameter selection |
| Test | 15% | Days 26–30 | Final evaluation (held out until all model decisions are frozen) |

**Day boundary selection:** With the default 30-day simulation window:
- Days 1–21 → Training (events with `timestamp` in the first 70% of the simulation period)
- Days 22–25 → Validation (next 15%)
- Days 26–30 → Test (last 15%)

**Why chronological split?** An entity's profile evolves over time. Randomly sampling future events into the training set would give the BPM knowledge of the entity's behavior after attack injection, and would allow the SDM to see post-attack sequences before seeing the attack itself. Chronological splitting ensures the training set always precedes the test set in time.

### 9.2 Training Order

The three models must be trained in order (due to the classifier's dependency on BPM/SDM output scores):

1. **Train BPM** (per-entity IsolationForest, Section 2.5) on training split normal events.
2. **Train SDM** (per-entity-type GRU autoencoder, Section 3.7) on training split normal sequences.
3. **Run BPM + SDM in inference mode on the full training split** to generate `bpm_score` and `sdm_score` for every training event.
4. **Train Classifier** (Section 4.6) on the 27-feature training set produced in step 3.

### 9.3 Evaluation Schema: Inference Schema, No Label Leakage

**Evaluation procedure uses the inference schema** (DATA_SCHEMA.md §2b), not the training schema. Specifically:

1. The test split events are processed through the pipeline using the inference schema (no `label` field present in the data).
2. The pipeline produces `ClassificationResult` objects (boundary H) and `AlertPayload` objects (boundary I) for test events.
3. The `evaluation/evaluator.py` module (T2) joins the pipeline's output `event_id`s against the separate `data/labeled/labels_<run_id>.parquet` file (which contains `event_id` + `label`) to compute metrics.
4. The label file is **never read by any inference-path component**. Only `evaluation/evaluator.py` reads it.

This guarantees DATA_SCHEMA.md §2c's label stripping policy: the evaluation appears from the pipeline's perspective exactly as production inference would.

### 9.4 Evaluation Metrics

| Metric | Scope | Why |
|--------|-------|-----|
| Per-class Precision, Recall, F1 | 8-class classifier | Captures per-attack-type detection quality; reveals which attacks are hardest |
| Macro-F1 | All classes | Overall balanced performance; penalizes poor rare-class recall |
| AUROC (one-vs-rest per class) | Detection threshold independence | Evaluates ranking quality independent of the chosen threshold |
| AUPRC (precision-recall AUC) | Binary anomaly detection (anomaly vs. normal) | More informative than AUROC for heavily imbalanced datasets |
| Precision@1% (binary) | Top 1% of events by `fused_score` | The "alert budget" metric; directly tied to the analyst operating point |
| FPR@TPR=0.9 | Binary | Measures false positive rate at 90% true positive rate |
| Risk tier distribution of true positives | Alert queue | Confirms that true anomalies land in expected risk tiers (Section 3.7 of ATTACK_TAXONOMY.md) |

---

## 10. Alternatives Considered

### 10.1 Sequence Detection Model Architecture Alternatives

#### Alternative A: LSTM Autoencoder

**Description:** Replace GRU with LSTM cells. Architecture otherwise identical.

**Honest tradeoffs:**
- LSTM has 4 gates vs. GRU's 3 gates → ~33% more parameters for the same hidden size
- LSTM is generally slightly more expressive for long-range dependencies
- GRU is faster to train and has fewer parameters → better for hackathon time budget
- For sequence lengths of W=20, long-range dependency learning is not a critical differentiator (both LSTM and GRU saturate on W=20 within a few epochs)
- Multiple published benchmarks on anomaly detection in tabular time-series show GRU and LSTM performing within 1–2% AUROC of each other on sequences of this length

**Why Rejected:** Not rejected on accuracy grounds — rejected on implementation time. The GRU autoencoder is simpler to debug, trains slightly faster, and produces equivalent results for W=20. The Tier 1 priority is a working end-to-end system, not maximum model sophistication.

#### Alternative B: Transformer Encoder (Self-Attention)

**Description:** Replace the GRU with a Transformer encoder (multi-head self-attention over the W=20 sequence), followed by a pooled representation and a symmetric Transformer decoder for reconstruction.

**Honest tradeoffs:**
- Transformers are generally more powerful for capturing long-range dependencies, but with W=20, the self-attention has only 400 attention pairs — the quadratic attention cost is trivial
- Transformers typically require more training data than GRUs for equivalent sequence-level performance; our training set of ~53,000 sequences is on the smaller side for a Transformer
- Transformers are less interpretable with Captum Integrated Gradients: the self-attention mechanism introduces attribution mixing across timesteps, making it harder to attribute anomaly scores to specific feature dimensions at specific positions
- Positional encoding must be explicitly designed for aperiodic behavioral sequences; sinusoidal encoding is the standard but may not capture the entity's irregular event timing
- Implementation complexity is significantly higher: multi-head attention, LayerNorm, positional encoding, etc.

**Why Rejected:** The GRU autoencoder meets the detection requirements for all 7 attack types (verified in Section 1b). The Transformer's advantages (longer-range dependencies, richer representations) are not needed for W=20 sequences with our attack taxonomy. The Explainability requirement (Captum IG is cleaner for GRU than Transformer) and the hackathon time constraint are the decisive factors.

#### Alternative C: Graph-Based Model (GNN on Entity-Resource Graph)

**Description:** Already rejected at the architecture level in DATA_SCHEMA.md §7.1. Rejected here for the same reasons: PyG dependency, GNN interpretability research gap, structural departure from the ARCHITECTURE.md component diagram.

**Verdict:** GRU Autoencoder is the correct choice for hackathon constraints, attack taxonomy coverage, and explainability requirements.

---

### 10.2 Anomaly Classifier Design Alternatives

#### Alternative A: Multi-Head Extension of the GRU Autoencoder

**Description:** Add a classification head to the GRU encoder's bottleneck representation `h_enc`. The encoder is trained with two loss terms simultaneously: (1) reconstruction loss (unsupervised, for anomaly detection) and (2) classification cross-entropy (supervised, for attack-type prediction). The classification head would be a Linear(64, 8) layer with softmax.

**Honest tradeoffs:**
- Advantages: single model, shared representation, no separate inference step for classification
- Disadvantages:
  - The classification head only has access to `h_enc` (shape 64), not the full 24-dim feature vector. Critical attack discriminators like `geo_velocity_kmph` (dim 6) are entangled in the bottleneck representation, not directly accessible.
  - Joint training with two loss terms requires careful balancing of loss weights (`α × reconstruction + β × classification`). Miscalibration causes one loss to dominate, harming either detection or classification.
  - Multi-head training requires labeled training data in the sequence DataLoader — but the SDM is designed to train on normal events only. Adding attack labels to the SDM training loop changes the fundamental training approach and must be justified.
  - If the classifier head underperforms, fixing it requires retraining the entire GRU autoencoder — losing the independently-retrain-able classifier benefit.

**Why Rejected:** The full 24-dim feature vector contains the most discriminating attack-class features, and the classifier must have access to all of them. The multi-head design structurally limits the classifier to the bottleneck, which is a lossy compression of the sequence. The separate LightGBM classifier has access to all 27 features (24 + 3 model scores) and is independently retrain-able.

#### Alternative B: Rule-Assisted Hybrid (Deterministic Rules + Model)

**Description:** Apply a set of deterministic classification rules first (e.g., `if geo_velocity_kmph > 0.4: predicted_class = impossible_travel`). If no rule fires with high confidence, fall back to the LightGBM classifier.

**Honest tradeoffs:**
- Advantages: rules are perfectly interpretable; brute force and impossible travel are cleanly rule-separable (nearly 100% rule precision)
- Disadvantages:
  - Rules are brittle: adding a new attack type or changing feature engineering requires updating the rules
  - Rules require hand-crafting thresholds (e.g., what `geo_velocity_kmph` threshold distinguishes travel from noise?); this is essentially doing feature engineering twice
  - The hybrid introduces a coverage gap: events that trigger the rule path don't go through LightGBM, losing the probability calibration for those events. `AlertPayload.classification_confidence` would be undefined for rule-classified events.
  - Rules produce hard binary classifications, not probability distributions. The `class_probabilities{}` field in boundary H would need to be synthesized (e.g., `{predicted_class: 1.0, all others: 0.0}`), which is misleading and breaks the `classification_confidence` interpretation.

**Why Rejected:** The `class_probabilities{}` contract (boundary H) requires a probability distribution over all 8 classes. Rules cannot produce this. The LightGBM classifier produces calibrated probabilities natively via `predict_proba()`. The hybrid design would require special-casing the probability output for rule-classified events, adding implementation complexity with no accuracy benefit (LightGBM already learns the "easy" rules automatically from data).

**Verdict:** Standalone LightGBM multi-class classifier with the 27-feature input is the correct choice.

---

## 11. Judging-Criteria Traceability

| Evaluation Criterion | Design Decision | How It Serves the Criterion |
|---------------------|----------------|----------------------------|
| **Detection accuracy on imbalanced labels (0.5%–3%)** | IsolationForest BPM (unsupervised, trained on normal events only) | Inherently handles class imbalance — never sees anomaly labels, so class ratio does not affect training. Score is derived from statistical isolation depth, which is not affected by class frequency. |
| **Detection accuracy on imbalanced labels** | Dual-layer imbalance strategy: SMOTE + class weights for classifier (Section 5) | SMOTE increases minority attack class sample counts; class weights equalize loss contribution. Together they ensure the classifier does not collapse to predicting `normal` for all events. |
| **Detection accuracy on imbalanced labels** | AUPRC as primary metric (Section 9.4) | AUPRC is more informative than AUROC for imbalanced datasets. A system with AUROC = 0.95 on a 1.5% positive rate dataset may have AUPRC = 0.40 (poor precision at high recall). AUPRC is the correct primary metric and is explicitly included in the evaluation procedure. |
| **Correct anomaly-type classification** | 27-feature classifier input (24 dims + 3 model scores) | Attack-discriminating features (e.g., `geo_velocity_kmph` for impossible travel, `fingerprint_mac_match` for device spoofing) are directly accessible to the classifier. The classifier does not need to infer these from model scores alone. |
| **Correct anomaly-type classification** | Separate LightGBM classifier (independently retrain-able) | If the attack taxonomy evolves or if per-class accuracy is unacceptable, the classifier is retrained without touching BPM/SDM. This is a direct service to classification accuracy without regression risk on the detection components. |
| **Correct anomaly-type classification** | `is_anomaly` override for `normal` prediction (Section 4.5) | Prevents the failure mode where the detector flags an event as anomalous but the classifier returns `normal`. The override ensures the alert queue always shows a specific attack class for flagged events. |
| **False positive rate at realistic analyst alert budget** | Continuous `fused_score` preserved in `AlertPayload` (Section 5.3) | The alert queue is sorted by `fused_score` descending. Analysts can review alerts in risk order. At the top-1% budget operating point, the system surfaces the highest-confidence anomalies first, minimizing analyst time wasted on marginal events. |
| **False positive rate at realistic analyst alert budget** | Calibrated normalization to [0,1] using training percentiles (Sections 2.4, 3.6) | Without normalization, the raw IsolationForest or reconstruction error scores have undefined range. The percentile normalization ensures that "0.7 score" has a consistent meaning across entities: 70th percentile of anomalousness relative to the training distribution. This makes the threshold and risk tier calibration meaningful. |
| **Insider Drift correctly scored as medium, not high** | GRU SDM detects slow-gradient sequence change, not abrupt changes | The GRU autoencoder learns normal sequence patterns. Insider drift's slow expansion produces a gradual increase in reconstruction error across the window — not a spike. The mean+0.3×max aggregation (Section 3.6) does not over-amplify gradual increases. Expected fused score: 0.28–0.52 → medium tier. |
| **System design and scalability** | Per-entity-type SDM (3 models, not 500) | Demonstrates architectural awareness of training scalability. The design explicitly acknowledges the tradeoff (Section 3.4) and chooses the level of granularity that balances model precision against training time. This reasoning is documentable in the Technical Report as evidence of engineering judgment. |
| **System design and scalability** | Model artifact caching at startup (Sections 2.6, 3.8) | Per-request disk I/O for model loading would be the primary inference bottleneck. Caching ensures O(1) model lookup at inference time. This is noted here explicitly because it is a design decision, not an implementation detail. |
| **Explainability and analyst usability** | SHAP on BPM + Captum IG on SDM, both producing `top_contributing_features` at boundary F (Section 8) | Both models produce human-readable feature names (not index numbers) at the earliest possible boundary (F, before fusion). Phase 7 constructs the full `FeatureAttribution` objects and narrative. The feature names at boundary F are the raw input to that narrative construction — no backward lookup required. |

---

## 12. Inference Pipeline Orchestrator Flow

### Step 5b: Profile Store Update (Gated EWMA)

After anomaly score fusion (Step 5) and before alert persistence (Step 6),
the orchestrator executes a gated profile update:

1. If the fused anomaly score is BELOW the drift-gating threshold
   (defined in COLDSTART_DRIFT_STRATEGY.md), the session is treated as
   legitimate behavioral evidence and the entity's baseline profile is
   updated via ProfileStore.upsert() using the Gated EWMA mechanism:
     updated_profile = (1 - α) * current_profile + α * session_features
   where α is the EWMA decay parameter defined in
   COLDSTART_DRIFT_STRATEGY.md.

2. If the fused anomaly score is AT OR ABOVE the drift-gating threshold,
   the profile is NOT updated — the session is treated as potentially
   anomalous and excluded from the baseline to prevent attack behavior
   from being learned as normal.

3. For cold-start entities (no existing profile in ProfileStore), this
   step initializes a new profile entry from the current session's
   features, simultaneously graduating the entity from cold-start status.

4. ProfileStore.upsert() must complete before alert persistence (Step 6)
   to ensure the persisted alert reflects the post-update profile state
   for any subsequent queries.

---

*End of ML_PIPELINE.md — Phase 6 output. This document is frozen. Amendments require a versioned change record and must not violate any contract defined in DATA_SCHEMA.md v1.0.*
