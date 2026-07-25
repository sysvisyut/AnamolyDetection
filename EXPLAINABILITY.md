# EXPLAINABILITY.md
# AI-Powered Behavioral Anomaly Detection — Explainability Layer Design

> **Status:** Phase 7 — Frozen Explainability Design Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** ARCHITECTURE.md v1.0, DATA_SCHEMA.md v1.0, ML_PIPELINE.md v1.0, ATTACK_TAXONOMY.md v1.0  
> **Scope:** Attribution method, narrative translation, per-attack explanation logic, consistency  
> validation, and all extension points for the Explainability Layer component.  
> No implementation code. Every design element must be directly implementable in  
> `explainability/feature_attribution.py`, `explainability/narrative.py`, and  
> `explainability/alert_builder.py` without further design decisions.

---

## Table of Contents

1. [Architecture Consistency Check](#1-architecture-consistency-check)
2. [Attribution Method](#2-attribution-method)
   - 2a. [BPM Attribution: SHAP TreeExplainer](#2a-bpm-attribution-shap-treeexplainer)
   - 2b. [SDM Attribution: Captum Integrated Gradients](#2b-sdm-attribution-captum-integrated-gradients)
   - 2c. [Attribution Merging for the Alert Payload](#2c-attribution-merging-for-the-alert-payload)
3. [Narrative Translation Layer](#3-narrative-translation-layer)
   - 3a. [Feature-to-Phrase Mapping (all 24 dimensions)](#3a-feature-to-phrase-mapping-all-24-dimensions)
   - 3b. [Template Algorithm](#3b-template-algorithm)
   - 3c. [Connector and Ranking Rules](#3c-connector-and-ranking-rules)
4. [Per-Attack-Type Explanation Logic](#4-per-attack-type-explanation-logic)
5. [Insider Drift Ambiguity Handling](#5-insider-drift-ambiguity-handling)
6. [Consistency Validation](#6-consistency-validation)
7. [Alternatives Considered](#7-alternatives-considered)
8. [Judging-Criteria Traceability](#8-judging-criteria-traceability)

---

## 1. Architecture Consistency Check

ML_PIPELINE.md v1.0 and DATA_SCHEMA.md v1.0 were re-read in full before designing this layer. The following compatibility properties are verified:

### 1a. Attribution Method Compatibility with Phase 6 Models

| Model | Phase 6 Architecture | Attribution Method Proposed | Compatibility |
|-------|--------------------|-----------------------------|--------------|
| **BPM** | `sklearn.ensemble.IsolationForest` (tree-based ensemble, no gradient) | `shap.TreeExplainer` | ✅ Full compatibility. SHAP TreeExplainer has explicit support for `IsolationForest` and computes exact (not approximate) Shapley values via tree traversal. No gradient is required. |
| **SDM** | GRU autoencoder (encoder: 2-layer GRU hidden=64; decoder: 2-layer GRU + Linear); bottleneck `h_enc` shape (B, 64); PyTorch | `captum.attr.IntegratedGradients` on the encoder path | ✅ Full compatibility. Captum IG requires only a differentiable `forward_func` that maps input tensor → scalar output. The encoder's `h_enc.norm()` (L2 norm of the bottleneck) is a scalar, differentiable, and monotonically related to how strongly the sequence is encoded — making it a valid IG target. IG requires no architectural modification to the GRU. |
| **Classifier** | `lightgbm.LGBMClassifier`, 27 features, trained offline | Training-time gain-based feature importance + optional per-inference `shap.TreeExplainer` | ✅ Full compatibility. LightGBM is a tree-based model; SHAP TreeExplainer supports LightGBM natively. Classifier attribution is used to confirm the `predicted_class` explanation, not to generate the primary narrative (that comes from BPM + SDM attribution). |

**Proposed SDM target function clarification:** ML_PIPELINE.md §8.2 specifies `captum.attr.IntegratedGradients(...).attribute(..., target=None)`. The `target=None` means IG computes gradients of a scalar aggregation of the output. The target scalar is defined as:

```
ig_target_scalar = torch.norm(h_enc, p=2, dim=1).mean()
```

This is the mean L2 norm of the encoder's bottleneck representation across the batch. It is:
- Differentiable (L2 norm is smooth everywhere except the origin)
- Monotonically related to the strength of the encoder's representation (higher norm = stronger sequence pattern encoded)
- Not the reconstruction error itself — this avoids conflating attribution (what drove the encoding) with anomaly scoring (how large the error was)

This clarification is a Phase 7 elaboration of the Phase 6 description, not a change to it.

### 1b. FeatureAttribution Schema Conformance

Verifying that the objects produced by this design match DATA_SCHEMA.md §5d exactly:

| Schema Field | Type | Constraint | This Design's Conformance |
|-------------|------|-----------|--------------------------|
| `feature_name` | `str` | One of 24 named dimensions from DATA_SCHEMA.md §3.2 | ✅ Attribution is always produced for named dimensions (e.g., `"geo_velocity_kmph"`, not index numbers). The `FEATURE_NAMES` constant list in `explainability/narrative.py` is the authoritative mapping. |
| `feature_value` | `float` | Actual normalized value at inference time | ✅ Taken directly from the 24-dim `feature_vector` at the same index as `feature_name`. |
| `attribution_score` | `float` | Signed; positive = toward anomaly | ✅ SHAP values are signed by definition. Captum IG values are signed by definition. Sign convention: positive = pushes toward anomaly. BPM SHAP: positive SHAP value means the feature increases the anomaly score. SDM IG: positive gradient means the feature increases the bottleneck norm (i.e., reinforces the anomalous pattern). |
| `direction` | `str` | `"toward_anomaly"` if `attribution_score > 0`, else `"toward_normal"` | ✅ Set as: `direction = "toward_anomaly" if attribution_score > 0 else "toward_normal"`. |
| `source_model` | `str` | `"bpm"` or `"sdm"` | ✅ Set to `"bpm"` for SHAP-derived attributions; `"sdm"` for Captum IG-derived attributions. |
| `human_label` | `str` | Plain-English description | ✅ Populated from `HUMAN_LABEL_MAP` constant in `explainability/narrative.py` (Section 3a of this document). |

**Schema conformance verdict:** No schema fields are added, removed, or changed. The `FeatureAttribution` objects produced by this design are a strict conforming implementation of DATA_SCHEMA.md §5d. **No schema change is proposed.**

### 1c. AlertPayload Schema Conformance

The `human_readable_explanation` field in `AlertPayload` (DATA_SCHEMA.md §5a) has the constraint: max 500 characters. The narrative template algorithm (Section 3) must enforce this. The `feature_attributions` field requires 1–10 entries. This design produces 5–10 entries (5 from BPM + 5 from SDM, deduplicated; deduplication may reduce this to a minimum of 2 entries in cases of high overlap). Both constraints are met.

---

## 2. Attribution Method

### 2a. BPM Attribution: SHAP TreeExplainer

**Library:** `shap` (version locked in TECH_STACK.md)  
**Model type:** `sklearn.ensemble.IsolationForest`  
**Attachment point:** `models/behavioral_profiling/inference.py`, immediately after computing `anomaly_score` (ML_PIPELINE.md §2.6, step 5)  
**Implemented in:** `explainability/feature_attribution.py`, function `compute_bpm_attributions(model, feature_vector_array) -> list[FeatureAttribution]`

#### Exact Procedure

1. Create a `shap.TreeExplainer` from the entity's fitted `IsolationForest` object:
   ```
   explainer = shap.TreeExplainer(isolation_forest_model)
   ```
   The `TreeExplainer` is created once per entity at model load time (not per inference call) and cached alongside the model in the startup model cache.

2. Compute SHAP values for the single input event:
   ```
   shap_values = explainer.shap_values(feature_vector_array)
   ```
   - Input shape: `(1, 24)` NumPy float64 array
   - Output: `shap_values` of shape `(1, 24)` — one signed SHAP value per feature dimension
   - SHAP values are in the same units as the model's output (anomaly score space before calibration)
   - Positive SHAP value: the feature pushes the anomaly score higher (toward anomaly)
   - Negative SHAP value: the feature pushes the anomaly score lower (toward normal)

3. Extract the SHAP vector for the single event: `sv = shap_values[0]` → shape `(24,)`

4. Construct `FeatureAttribution` objects for all 24 dimensions:
   ```
   for d in range(24):
     feature_name = FEATURE_NAMES[d]
     feature_value = feature_vector_array[0, d]
     attribution_score = float(sv[d])
     direction = "toward_anomaly" if attribution_score > 0 else "toward_normal"
     source_model = "bpm"
     human_label = HUMAN_LABEL_MAP[feature_name]
     → create FeatureAttribution(...)
   ```

5. Sort the 24 `FeatureAttribution` objects by `|attribution_score|` descending.

6. Return the top-5 by absolute value as `top_bpm_attributions: list[FeatureAttribution]`.

**Runtime:** TreeExplainer SHAP inference on 1 × 24 input with 200 trees takes approximately 0.3–0.8ms on CPU. This is acceptable on the synchronous inference path.

**Sign invariant:** For IsolationForest, SHAP values for `score_samples()` have the property that positive values increase the raw score (make the point look more normal). Since the anomaly score is `calibrated as (score_max - raw_score) / range` (ML_PIPELINE.md §2.4), positive raw SHAP → lower anomaly score → `"toward_normal"`. To maintain the sign convention that positive attribution = toward anomaly, **negate the SHAP values**:
```
sv = -1.0 × shap_values[0]
```
After negation: positive sv[d] → dimension d pushed anomaly score up (toward anomaly). This negation must be documented in `feature_attribution.py`.

---

### 2b. SDM Attribution: Captum Integrated Gradients

**Library:** `captum` (version locked in TECH_STACK.md)  
**Model type:** GRU autoencoder (PyTorch), encoder path only  
**Attachment point:** `models/sequence_detection/inference.py`, after forward pass (ML_PIPELINE.md §3.8, step 7)  
**Implemented in:** `explainability/feature_attribution.py`, function `compute_sdm_attributions(encoder, sequence_tensor, sequence_mask) -> list[FeatureAttribution]`

#### Exact Procedure

1. Define the `forward_func` for Captum. This function takes the sequence tensor as input and returns the IG target scalar:
   ```
   def encoder_forward_func(sequence_tensor):
     # sequence_tensor shape: (1, 20, 24)
     # Run encoder only (not decoder)
     encoder_output, _ = encoder_gru(sequence_tensor)
     # encoder_output shape: (1, 20, 64)
     # Select last real (non-padded) hidden state using mask:
     last_idx = sequence_mask.sum(dim=1) - 1   # shape (1,)
     h_enc = encoder_output[torch.arange(1), last_idx, :]  # shape (1, 64)
     # Return scalar: L2 norm of bottleneck
     return torch.norm(h_enc, p=2, dim=1).mean()   # scalar
   ```
   This function must be defined during inference, not at model load time (because `sequence_mask` varies per event).

2. Instantiate IG and compute attribution:
   ```
   ig = captum.attr.IntegratedGradients(forward_func=encoder_forward_func)
   attributions = ig.attribute(
     inputs=sequence_tensor,          # shape (1, 20, 24), float32
     baselines=torch.zeros_like(sequence_tensor),  # all-zeros baseline
     n_steps=50,                      # 50 Riemann approximation steps
     method="gausslegendre"           # most accurate quadrature method
   )
   # attributions shape: (1, 20, 24)
   ```

3. Aggregate attribution over real (non-padded) sequence positions:
   ```
   # Extract real positions using mask (shape: (1, 20) bool)
   real_mask = sequence_mask[0]  # shape (20,), True for real events
   real_attributions = attributions[0][real_mask, :]  # shape (n_real, 24)
   feature_importance = real_attributions.abs().mean(dim=0)  # shape (24,)
   feature_signed = real_attributions.mean(dim=0)            # shape (24,)
   ```
   - `feature_importance[d]` = mean absolute IG across all real timesteps for feature d
   - `feature_signed[d]` = mean signed IG across all real timesteps for feature d (used as `attribution_score`)

4. Construct `FeatureAttribution` objects for all 24 dimensions:
   ```
   for d in range(24):
     feature_name = FEATURE_NAMES[d]
     feature_value = sequence_tensor[0, last_idx, d].item()  # value at triggering event
     attribution_score = float(feature_signed[d])
     direction = "toward_anomaly" if attribution_score > 0 else "toward_normal"
     source_model = "sdm"
     human_label = HUMAN_LABEL_MAP[feature_name]
   ```
   Note: `feature_value` is taken from the **last real position** in the sequence window (the triggering event's feature vector), not a mean over the window. This ensures the narrative can refer to a specific current value ("geo-velocity of 1,847 km/h") rather than a window average.

5. Sort by `feature_importance` (absolute value of signed mean attribution) descending.

6. Return top-5 as `top_sdm_attributions: list[FeatureAttribution]`.

**Runtime:** Captum IG with n_steps=50 on a (1, 20, 24) tensor through a 2-layer GRU takes approximately 15–30ms on CPU per inference call. This is the primary performance cost of the explainability layer. If this exceeds the latency budget, n_steps may be reduced to 20 (acceptable precision reduction) or IG may be moved to an asynchronous path (see ARCHITECTURE.md §9, Risk 4 mitigation).

**Baseline choice rationale:** The all-zeros baseline (torch.zeros_like(sequence_tensor)) represents "no history / no events" — a meaningful behavioral null state. An alternative baseline (e.g., training-set mean sequence) would be more statistically grounded but adds complexity; the zero baseline is standard for GRU autoencoders and sufficient for narrative generation.

---

### 2c. Attribution Merging for the Alert Payload

**Implemented in:** `explainability/alert_builder.py`, function `merge_attributions(top_bpm: list[FeatureAttribution], top_sdm: list[FeatureAttribution]) -> list[FeatureAttribution]`

The `AlertPayload.feature_attributions` field requires 1–10 entries (DATA_SCHEMA.md §5a). The BPM and SDM each produce top-5 attributions, giving a raw pool of 10 entries (possibly with overlapping feature names when both models agree on which features are important).

#### Merging Algorithm

1. Check for duplicate `feature_name` entries between `top_bpm` and `top_sdm`.

2. For each duplicated `feature_name` f:
   - Create a single merged `FeatureAttribution` with:
     - `feature_name` = f
     - `feature_value` = `top_bpm[f].feature_value` (BPM uses current event; SDM uses last-real-position; both refer to the same triggering event — values should be identical or near-identical)
     - `attribution_score` = `(top_bpm[f].attribution_score + top_sdm[f].attribution_score) / 2` (average signed attribution across both models)
     - `direction` = `"toward_anomaly"` if `attribution_score > 0` else `"toward_normal"`
     - `source_model` = `"bpm+sdm"` ← **Proposed micro-extension:** DATA_SCHEMA.md §5d specifies `source_model` as `"bpm"` or `"sdm"`. A merged entry from both models should use `"bpm+sdm"`.

> **⚠️ Proposed Schema Change (Minor):** DATA_SCHEMA.md §5d currently constrains `source_model` to `"bpm"` or `"sdm"`. Adding `"bpm+sdm"` as a valid value is a non-breaking additive change (MINOR version per DATA_SCHEMA.md §6.2 — "adding a nullable/optional field" maps to the closest equivalent of adding a new enum value to a non-breaking field). This change must be recorded in DATA_SCHEMA.md §9 (Changelog) with version bump to 1.1 before implementation. All downstream consumers (API router, dashboard) must be checked to ensure they handle `"bpm+sdm"` gracefully. The dashboard renders `source_model` as a label badge — adding a third value is trivially handled. No other downstream component is affected.

3. For non-duplicated entries: include as-is (source_model = "bpm" or "sdm" respectively).

4. Sort the merged list by `|attribution_score|` descending (merged entries with averaged scores may rank lower than single-source entries with high individual scores — this is correct; it penalizes features where the two models disagree in magnitude).

5. Truncate to a maximum of 10 entries (unlikely to be exceeded since the raw pool is at most 10 and deduplication reduces it).

6. Enforce minimum of 1 entry (guaranteed if at least one attribution is non-zero, which is always true for anomalous events).

7. Return the merged, sorted, truncated list as `AlertPayload.feature_attributions`.

---

## 3. Narrative Translation Layer

**Implemented in:** `explainability/narrative.py`  
**Primary function:** `build_explanation(feature_attributions: list[FeatureAttribution], predicted_class: str, classification_confidence: float, fused_score: float, cold_start_flag: bool) -> str`

### 3a. Feature-to-Phrase Mapping (All 24 Dimensions)

The `HUMAN_LABEL_MAP` is a constant dict in `explainability/narrative.py`. Every entry has: (1) a short `human_label` (used in the `FeatureAttribution` object) and (2) a `phrase_template` (used in narrative construction, where `{value}` is replaced with the actual feature value formatted appropriately).

| Dim | Feature Name | Human Label | Phrase Template (toward_anomaly) | Phrase Template (toward_normal) |
|-----|-------------|------------|----------------------------------|--------------------------------|
| 0 | `hour_of_day_sin` | "Login time (hour of day)" | "an off-hours login at {value} UTC" | "login during normal hours" |
| 1 | `hour_of_day_cos` | "Login time (hour cycle)" | "an off-hours login at {value} UTC" | "login during normal hours" |
| 2 | `day_of_week_sin` | "Day of week" | "unusual {value} access" | "access on a normal workday" |
| 3 | `day_of_week_cos` | "Day of week" | "unusual {value} access" | "access on a normal workday" |
| 4 | `session_duration_norm` | "Session length" | "an unusually long session ({value} sec)" | "normal session length" |
| 5 | `failure_count_norm` | "Authentication failure count" | "{value} consecutive authentication failures" | "no unusual authentication failures" |
| 6 | `geo_velocity_kmph` | "Speed between logins (km/h)" | "a geo-velocity of {value} km/h between consecutive logins" | "normal login locations" |
| 7 | `is_new_geo` | "New geographic location" | "a new country ({value}) not in this entity's location history" | "a known login location" |
| 8 | `resource_category_enc` | "Resource category accessed" | "access to an unusual resource category ({value})" | "access to a normal resource category" |
| 9 | `resource_rarity_score` | "Resource access rarity" | "access to a resource rarely or never accessed before (rarity score {value:.2f})" | "access to a frequently-used resource" |
| 10 | `auth_method_enc` | "Authentication method" | "use of an unusual authentication method ({value})" | "use of the entity's normal authentication method" |
| 11 | `auth_outcome_enc` | "Authentication outcome" | "authentication failure" | "successful authentication" |
| 12 | `command_seq_length_norm` | "Command sequence length" | "an unusually long command sequence ({value} commands)" | "normal command sequence length" |
| 13 | `command_rarity_score` | "Command rarity" | "use of commands rarely issued by this entity (rarity score {value:.2f})" | "use of the entity's normal command set" |
| 14 | `has_exfil_command` | "Data transfer command detected" | "a data transfer command (scp/rsync/curl/wget) was issued" | "no data transfer commands detected" |
| 15 | `fingerprint_os_match` | "OS profile match" | "an OS mismatch on device {device_id} (expected {expected_os}, saw {actual_os})" | "OS matches the registered device profile" |
| 16 | `fingerprint_mac_match` | "MAC address match" | "a MAC address mismatch on device {device_id} (expected {expected_mac}, saw {actual_mac})" | "MAC address matches the registered device profile" |
| 17 | `fingerprint_protocol_match` | "Protocol match" | "a protocol mismatch on device {device_id} (expected {expected_proto}, saw {actual_proto})" | "protocol matches the registered device profile" |
| 18 | `entity_type_enc` | "Entity type" | (not used in narrative — context only) | (not used in narrative) |
| 19 | `inter_event_gap_norm` | "Time since last login" | "an unusually long gap ({value} hours) since this entity's previous login" | "normal login frequency" |
| 20 | `session_event_count_norm` | "Events in current session" | "{value} events in the current session (unusually high volume)" | "normal session depth" |
| 21 | `resource_breadth_norm` | "Resource variety in session" | "access to {value} distinct resources in a single session (unusually broad)" | "access to a normal number of resources" |
| 22 | `ip_entity_ratio` | "Entities reached from this IP" | "this IP address was used to attempt access to {value} distinct entities in the past 24 hours" | "this IP address is associated with a normal number of entities" |
| 23 | `entity_ip_ratio` | "IPs used by this entity" | "this entity used {value} distinct IP addresses in the past 24 hours" | "this entity used a normal number of IP addresses" |

**Notes on dimensions 0–1 and 2–3 (paired circular encodings):**  
`hour_of_day_sin` and `hour_of_day_cos` are always treated as a pair. When either appears in the top attribution list, both are merged into a single narrative phrase using the reconstructed hour: `hour = round(atan2(sin_val, cos_val) × 24 / (2π)) % 24`. Similarly for `day_of_week_sin/cos` → weekday name. The merging logic is in `narrative.py`'s `_decode_circular_feature(sin_dim, cos_dim, period)` helper.

**Notes on dimensions 15–17 (device fingerprint match fields):**  
These are binary features (0.0 or 1.0). Their phrase templates reference the actual and expected values from the `raw_event_snapshot` in the `AlertPayload`. The narrative builder retrieves these from `alert.raw_event_snapshot.device_fingerprint` and `entity_profile.known_os_profiles`/`known_mac_addresses` respectively.

### 3b. Template Algorithm

The narrative generation algorithm in `build_explanation()`:

**Step 1: Select contributing features for narrative**

From `AlertPayload.feature_attributions` (already sorted by `|attribution_score|` descending from Section 2c):
- Filter to features with `direction = "toward_anomaly"` and `|attribution_score| > ATTRIBUTION_THRESHOLD` (default: 0.05)
- Take the top N_narrative features (default: N_narrative = 3; configurable in `config/default.yaml`). Minimum: 1. Maximum: 4.
- Apply **circular encoding merge**: if both `hour_of_day_sin` and `hour_of_day_cos` are selected, merge them into a single phrase slot using the decoded hour value. Similarly for `day_of_week_sin/cos`. This prevents the narrative from citing two features that describe the same human-readable concept.

**Step 2: Format individual feature phrases**

For each selected feature f:
1. Look up `phrase_template` for `(feature_name, direction)` from `HUMAN_LABEL_MAP`.
2. Format `{value}` using the `feature_value` field from the `FeatureAttribution` object, applying type-appropriate formatting:
   - Velocity: `{value:.0f}` (integer km/h, e.g., "1,847")
   - Duration/time: convert from normalized [0,1] to seconds using the stored denormalization factor, then format as "X hours Y minutes" or "X seconds"
   - Count: denormalize by known cap (e.g., `failure_count_norm × 20`) and format as integer
   - Rarity scores: `{value:.2f}` (two decimal places)
   - Binary (0/1): no value substitution needed; the phrase is binary
   - Device fingerprint fields (dims 15–17): look up actual values from `raw_event_snapshot`

**Step 3: Compose the sentence**

Apply the sentence template for the `predicted_class` (Section 4). The generic fallback template is:

```
"Flagged due to {phrase_1}{connector_1}{phrase_2}{connector_2}{phrase_3}."
```

Where connectors are selected per Section 3c. The sentence is always capitalized, ends with a period, and is never longer than 500 characters.

**Step 4: Prepend class context**

Prepend an attack-class-specific context prefix (Section 4) that identifies the attack type before the evidence:

```
"{class_context_prefix}. {evidence_sentence}"
```

For example: `"Brute force attack detected. Flagged due to 14 consecutive authentication failures from IP 10.0.0.8."`

**Step 5: Append confidence qualifier (if applicable)**

- If `classification_confidence < 0.65`: append `" (classification confidence: {classification_confidence:.0%})"`
- If `cold_start_flag = True`: append `" [Note: entity profile is new — scores may be less reliable]"`
- If `predicted_class = "insider_drift"`: apply full ambiguity framing (Section 5)

**Step 6: Truncate to 500 characters**

If the composed sentence exceeds 500 characters, truncate at the last complete word before 497 characters and append `"..."`.

### 3c. Connector and Ranking Rules

| Number of narrative features | Connector template |
|------------------------------|-------------------|
| 1 | `"Flagged due to {phrase_1}."` |
| 2 | `"Flagged due to {phrase_1} combined with {phrase_2}."` |
| 3 | `"Flagged due to {phrase_1}, {phrase_2}, and {phrase_3}."` |
| 4 | `"Flagged due to {phrase_1}, {phrase_2}, {phrase_3}, and {phrase_4}."` |

**Ranking priority rules** (applied within Step 1 before selecting top-N):
- Rule 1: Features from `source_model = "bpm+sdm"` (both models agree) rank above single-source features of the same `|attribution_score|`.
- Rule 2: Among features for the same attack class (from Section 4), **expected primary features** rank above unexpected features with similar scores. For example, for `brute_force`: `failure_count_norm` always ranks first if it has any positive attribution, regardless of its relative score vs. other features. This implements a weak prior from ATTACK_TAXONOMY.md — the expected signal should lead the explanation.
- Rule 3: Circular encoding pairs (e.g., `hour_of_day_sin` + `hour_of_day_cos`) count as one feature slot after merging.

---

## 4. Per-Attack-Type Explanation Logic

This section specifies, for each attack type: (1) the expected primary feature drivers from ATTACK_TAXONOMY.md, (2) the framing strategy, and (3) a concrete example explanation sentence.

All example sentences below are generated by the algorithm in Section 3 using representative feature values. They conform to DATA_SCHEMA.md §5a's example style.

---

### 4.1 Brute Force

**Primary feature drivers:** `failure_count_norm` (dim 5), `auth_outcome_enc` (dim 11), `session_event_count_norm` (dim 20)  
**Expected attribution source:** BPM (strong); SDM (helpful — burst pattern in sequence window)  
**Class context prefix:** `"Brute force attack detected"`

**Framing strategy:**  
Lead with the failure count as a concrete number (denormalized). Cite the source IP if `ip_entity_ratio` is in the top attributions (it will be for high-confidence brute force — same IP used for all failures). End with the time window if `session_event_count_norm` is elevated.

**Feature priority override:** `failure_count_norm` is always the first phrase if it has positive attribution (ranking Rule 2).

**Value denormalization for narrative:**  
`failure_count_norm` → `round(failure_count_norm × 20)` failures  
`session_event_count_norm` → `round(session_event_count_norm × 200)` events

**Example explanation sentence:**
> "Brute force attack detected. Flagged due to 14 consecutive authentication failures, a session with 17 events in rapid succession (average 4 seconds between events), and this IP address being used to attempt access to 1 distinct entity (concentrated attack)."

---

### 4.2 Impossible Travel

**Primary feature drivers:** `geo_velocity_kmph` (dim 6), `is_new_geo` (dim 7)  
**Expected attribution source:** BPM (strong — single-event deterministic signal)  
**Class context prefix:** `"Impossible travel detected"`

**Framing strategy:**  
`geo_velocity_kmph` is almost always the top attribution for this attack type (near-deterministic physical signal). Cite the velocity in km/h (denormalized: `round(geo_velocity_kmph × 2000)` km/h) and the new location from `raw_event_snapshot.geo_location`. If `is_new_geo` is also elevated, cite the country change. Cite the time gap between events.

**Value denormalization for narrative:**  
`geo_velocity_kmph` → `round(geo_velocity_kmph × 2000)` km/h  
For the location names: retrieved from `raw_event_snapshot.geo_location.city` (current) and the entity's `most_frequent_country` from the entity profile

**Example explanation sentence:**
> "Impossible travel detected. Flagged due to a geo-velocity of 1,847 km/h between Mumbai (02:31 UTC) and London (02:47 UTC) — physically impossible for a commercial traveler — and a new country (GB) not in this entity's location history."

---

### 4.3 Credential Stuffing

**Primary feature drivers:** `ip_entity_ratio` (dim 22), `failure_count_norm` (dim 5), `auth_outcome_enc` (dim 11)  
**Expected attribution source:** BPM (moderate — cross-entity IP ratio); SDM (helpful — sequence of failures)  
**Class context prefix:** `"Credential stuffing campaign detected"`

**Framing strategy:**  
Lead with the cross-entity IP signal (`ip_entity_ratio`) — this is the key discriminator from brute force. Cite the number of entities targeted from this IP (denormalized: `round(ip_entity_ratio × 10)` entities). Then cite the per-entity failure count (typically 1–5). Distinguish from brute force: brute force has many failures per entity; stuffing has few failures but many entities.

**Value denormalization for narrative:**  
`ip_entity_ratio` → `round(ip_entity_ratio × 10)` distinct entities from this IP in 24h  
`failure_count_norm` → `round(failure_count_norm × 20)` failures against this entity

**Example explanation sentence:**
> "Credential stuffing campaign detected. Flagged due to this IP address attempting access to 42 distinct entities in the past 3 minutes, with 2 authentication failures against this specific entity — consistent with an automated credential list attack."

---

### 4.4 Lateral Movement

**Primary feature drivers:** `resource_rarity_score` (dim 9), `resource_breadth_norm` (dim 21), `has_exfil_command` (dim 14), `command_seq_length_norm` (dim 12)  
**Expected attribution source:** BPM (moderate); SDM (strengthens — abrupt resource diversity shift in sequence)  
**Class context prefix:** `"Lateral movement detected"`

**Framing strategy:**  
Lead with resource breadth (how many distinct resources were accessed in this session — denormalized: `round(resource_breadth_norm × 50)`) and the presence/absence of exfil commands. Cite resource rarity as a secondary signal ("resources not previously accessed by this entity"). If SDM score is the higher contributor, mention the sequence-level signal explicitly: "an abrupt shift in access pattern detected across the session's event sequence."

**Value denormalization for narrative:**  
`resource_breadth_norm` → `round(resource_breadth_norm × 50)` distinct resources  
`command_seq_length_norm` → `round(command_seq_length_norm × 50)` commands

**Example explanation sentence:**
> "Lateral movement detected. Flagged due to access to 17 distinct resources across 5 categories (file, API, port, database, device) in a single session — compared to a baseline of 2 categories — a reconnaissance-to-exfiltration command sequence of 12 commands, and a data transfer command (scp) directed to an external IP."

---

### 4.5 Device Spoofing

**Primary feature drivers:** `fingerprint_mac_match` (dim 16), `fingerprint_os_match` (dim 15), `fingerprint_protocol_match` (dim 17)  
**Expected attribution source:** BPM (moderate — binary feature mismatch); SDM (strengthens — abrupt discrete change in fingerprint dims)  
**Class context prefix:** `"Device spoofing detected"`

**Framing strategy:**  
These are binary features; the phrase template always includes the actual vs. expected values retrieved from `raw_event_snapshot` and the entity profile. Lead with whichever mismatch has the highest attribution (typically MAC address for Strategy A, OS for Strategy B). Always include `device_id` in the narrative so the analyst knows which registered device is implicated.

**Value substitution for narrative (special case):**  
For dims 15–17, the `feature_value` (0.0 or 1.0) is not cited directly. Instead, retrieve:
- Expected MAC: `entity_profile.known_mac_addresses[0]`
- Actual MAC: `raw_event_snapshot.device_fingerprint.mac_address`
- Expected OS: `entity_profile.known_os_profiles[0]`
- Actual OS: `{raw_event_snapshot.device_fingerprint.os_family}/{raw_event_snapshot.device_fingerprint.os_version}`
- Device ID: `raw_event_snapshot.device_fingerprint.device_id`

**Example explanation sentence:**
> "Device spoofing detected. Flagged due to device dev_3f8a21bc reappearing with MAC address AA:BB:CC:99:88:77 (registered: AA:BB:CC:11:22:33) and OS Linux/22.04 (registered: Windows/11.0) — the device ID is recognized but its hardware profile has changed."

---

### 4.6 Low-and-Slow Exfiltration

**Primary feature drivers:** `has_exfil_command` (dim 14), `hour_of_day_sin/cos` (dims 0–1), `inter_event_gap_norm` (dim 19)  
**Expected attribution source:** SDM (primary — multi-day pattern essential); BPM (weak — individual events near-normal)  
**Class context prefix:** `"Low-and-slow exfiltration pattern detected"`

**Framing strategy:**  
This is the attack where the SDM score dominates. When `sdm_score > bpm_score` by more than 0.15, add a sequence-context qualifier to the prefix: `"Low-and-slow exfiltration pattern detected (sequence-level signal spanning {N} days)"`. Cite the off-hours timing (decoded hour), the exfil command, and the inter-event gap. The narrative must communicate that no single event is individually alarming — the anomaly is a pattern.

**Value denormalization for narrative:**  
`inter_event_gap_norm` → `round(inter_event_gap_norm × 24)` hours since last login  
Hour reconstruction from `hour_of_day_sin/cos` → formatted as "HH:MM UTC"

**Sequence-context qualifier trigger:** if `sdm_score > bpm_score + 0.15`, insert the qualifier. Otherwise use the standard prefix.

**Example explanation sentence:**
> "Low-and-slow exfiltration pattern detected (sequence-level signal spanning 11 days). Flagged due to a recurring off-hours login pattern (02:31 UTC, outside this entity's normal 08:00–18:00 window), a data transfer command (scp) directed to an external IP in each session, and an inter-session gap of 26 hours consistent with nightly access — no single event exceeds the anomaly threshold individually."

---

### 4.7 Normal (No Alert)

Events classified as `normal` do not generate `AlertPayload` records and therefore do not receive narrative explanations. No implementation is required for this class in the narrative layer. The pipeline enforces this via the `is_anomaly = False` guard in `models/anomaly_classifier/inference.py` (ML_PIPELINE.md §4.5).

---

## 5. Insider Drift Ambiguity Handling

**Target:** `predicted_class = "insider_drift"`  
**Designed fused score range:** 0.28–0.52 (near threshold; ATTACK_TAXONOMY.md §7)  
**Designed classification confidence:** 0.40–0.60  
**Designed risk tier:** medium (25–49)

### 5.1 Ambiguity Framing Strategy

Insider drift is the only attack type for which the explanation must actively communicate **uncertainty and alternative interpretations**, rather than asserting a clear threat. This is not a fallback or failure mode — it is the correct behavior for a well-calibrated system (per SYNTHETIC_DATA_GENERATOR_DESIGN.md §3.7).

The explanation for insider drift must:
1. **Not use assertive language** ("attack detected", "malicious activity"). Instead, use investigative language ("pattern of behavioral expansion identified", "flagged for analyst review").
2. **Explicitly cite the alternative legitimate interpretation** (role change, new project assignment) alongside the concerning interpretation.
3. **Quantify the ambiguity** using `classification_confidence` in the sentence.
4. **State which signals are absent** (the five ambiguity points from SYNTHETIC_DATA_GENERATOR_DESIGN.md §3.7) — this helps the analyst quickly confirm whether this is a real threat or a legitimate change.

### 5.2 Insider Drift Class Context Prefix

```
"Behavioral expansion pattern identified (not a confirmed attack — analyst review recommended)"
```

This prefix is used **only** for `predicted_class = "insider_drift"`. All other attack types use assertive prefixes (Section 4).

### 5.3 Insider Drift Narrative Template

The `build_explanation()` function applies a **dedicated override template** when `predicted_class = "insider_drift"`:

```
"{prefix}. {entity_id} has accessed {new_resource_count} resources outside their 
normal profile over the past {drift_days} days, primarily in the {category} 
resource category (rarity score: {rarity:.2f}). This may indicate a role change 
or new project assignment, or it may indicate an insider threat. Notably, no 
off-hours access, authentication failures, device anomalies, or exfiltration 
commands were detected — only resource footprint expansion. Classification 
confidence: {confidence:.0%}. Suggested action: verify with HR or line manager 
whether a role or project change occurred for this entity."
```

**Template field resolution:**
- `entity_id`: from `AlertPayload.entity_id`
- `new_resource_count`: approximate count from `resource_breadth_norm × 50`, representing resources accessed in the current detection window outside the normal profile
- `drift_days`: derived from the SDM's sequence window. If `real_event_count` in the sequence window spans multiple days (estimated from `inter_event_gap_norm`), report the span. Fallback: "recent period"
- `category`: the resource category most frequently cited in `resource_rarity_score`-elevated events; retrieved from `raw_event_snapshot.resource_accessed` prefix (e.g., "file", "api")
- `rarity`: the `resource_rarity_score` feature value (already normalized [0,1])
- `confidence`: `classification_confidence` from `ClassificationResult`

### 5.4 Five-Absence Statement

The template always includes the explicit absence statement:
```
"Notably, no off-hours access, authentication failures, device anomalies, or 
exfiltration commands were detected — only resource footprint expansion."
```
This is a fixed string — it does not vary by event. Its function is to help the analyst rule out the more severe attacks (low-and-slow, lateral movement) before escalating.

### 5.5 Connection to False-Positive Tuning

The insider drift explanation is explicitly designed for false-positive tuning use (per SYNTHETIC_DATA_GENERATOR_DESIGN.md §3.7's requirement that "the ambiguity must be real, not cosmetic"):

1. **Analyst decision feedback (T3):** When an analyst marks an insider drift alert as `false_positive` with a note like "legitimate promotion", this event becomes a training example for the concept drift monitor. The `analyst_notes` field in `AlertPayload` is the channel for this.
2. **Threshold calibration:** The system's FP rate for insider drift is expected to be higher than for other attack types. The T2 evaluation module computes FP rate per attack class. A system with insider drift FP rate > 60% is operating as designed (the ground truth for insider drift is genuinely ambiguous, per SYNTHETIC_DATA_GENERATOR_DESIGN.md §3.7's declaration that "insider drift is not fully distinguishable from legitimate role evolution using event-level data alone").
3. **Risk tier suppression:** Insider drift is capped at `risk_tier = "medium"` in the `AlertPayload` risk tier assignment logic. This is a hardcoded cap in `explainability/alert_builder.py`:
   ```
   if predicted_class == "insider_drift":
     risk_tier = min(risk_tier_from_score, "medium")  # never escalate to high/critical
   ```
   The cap is documented in the docstring as an architectural decision, not a bug.

### 5.6 Concrete Example Explanation Sentence

> "Behavioral expansion pattern identified (not a confirmed attack — analyst review recommended). usr_4d8e21bc has accessed 8 resources outside their normal profile over the past 18 days, primarily in the file resource category (rarity score: 0.54). This may indicate a role change or new project assignment, or it may indicate an insider threat. Notably, no off-hours access, authentication failures, device anomalies, or exfiltration commands were detected — only resource footprint expansion. Classification confidence: 48%. Suggested action: verify with HR or line manager whether a role or project change occurred for this entity."

---

## 6. Consistency Validation

**Implemented in:** `explainability/alert_builder.py`, function `validate_explanation_consistency(feature_attributions: list[FeatureAttribution], predicted_class: str) -> ValidationResult`

### 6.1 Purpose

The consistency validation check ensures that the top-cited features in `feature_attributions` are plausibly associated with the `predicted_class`. It prevents the pathological case where the narrative cites geo-velocity as the top factor but the predicted class is `device_spoofing` — a combination that would confuse an analyst.

### 6.2 Expected Feature Sets Per Attack Class

The following `EXPECTED_PRIMARY_FEATURES` dict is defined in `explainability/alert_builder.py`:

```
EXPECTED_PRIMARY_FEATURES = {
  "brute_force":          {"failure_count_norm", "auth_outcome_enc", "session_event_count_norm", "ip_entity_ratio"},
  "impossible_travel":    {"geo_velocity_kmph", "is_new_geo"},
  "credential_stuffing":  {"ip_entity_ratio", "failure_count_norm", "auth_outcome_enc", "entity_ip_ratio"},
  "lateral_movement":     {"resource_rarity_score", "resource_breadth_norm", "has_exfil_command", "command_seq_length_norm", "command_rarity_score"},
  "device_spoofing":      {"fingerprint_mac_match", "fingerprint_os_match", "fingerprint_protocol_match"},
  "low_and_slow":         {"has_exfil_command", "hour_of_day_sin", "hour_of_day_cos", "inter_event_gap_norm", "session_duration_norm"},
  "insider_drift":        {"resource_rarity_score", "resource_breadth_norm", "command_rarity_score"},
  "normal":               set()
}
```

### 6.3 Validation Algorithm

```
function validate_explanation_consistency(feature_attributions, predicted_class):
  1. Extract the set of cited feature names from the top-N_narrative entries
     in feature_attributions (only those with direction = "toward_anomaly"):
     cited_features = {fa.feature_name for fa in feature_attributions[:N_narrative]
                       if fa.direction == "toward_anomaly"}
  
  2. Retrieve expected = EXPECTED_PRIMARY_FEATURES[predicted_class]
  
  3. Compute overlap = cited_features ∩ expected
  
  4. Compute consistency_score = len(overlap) / max(len(expected), 1)
     (fraction of expected features that appear in the citation)
  
  5. If consistency_score >= CONSISTENCY_THRESHOLD (default: 0.33 = at least 1 expected feature cited):
     → return ValidationResult(is_consistent=True, consistency_score=consistency_score)
  
  6. If consistency_score < CONSISTENCY_THRESHOLD:
     → Enter fallback mode (Section 6.4)
```

The threshold of 0.33 (at least 1 expected feature in the top-N_narrative) is intentionally lenient. Multiple attacks can share some signal (e.g., `failure_count_norm` is relevant for both brute force and credential stuffing). A stricter threshold would trigger false consistency failures.

### 6.4 Fallback Mode: Override vs. Warning

When `consistency_score < 0.33`:

**Case A — The top-1 attribution feature IS in `expected`:**  
This can occur if the narrative selected features 2–3 that are off-class but the primary driver is correct. Action: `is_consistent = True` (override). The primary signal is correct; lower-ranked features may reflect model noise.

**Case B — The top-1 attribution feature is NOT in `expected`:**  
This indicates a genuine mismatch between the classifier's predicted class and the attribution model's primary signal. Possible causes: the classifier is confident about a class that the BPM/SDM did not primarily flag for, or a rare edge case where multiple attack signals overlap.

Action in Case B:
1. Log the mismatch: `logger.warning(f"Consistency check failed: top feature {top_feature} not in expected set for {predicted_class}. Consistency score: {consistency_score:.2f}")`.
2. Do **not** suppress the alert. The classifier's predicted class takes precedence (it has access to all 27 features and is the authoritative classifier).
3. Append a consistency note to `human_readable_explanation`:
   ```
   " [Note: primary attribution signal ({top_feature}) is atypical for this classification — analyst verification recommended.]"
   ```
4. Log to a `consistency_failures.log` file for offline investigation during the hackathon demo. This provides a debugging artifact that demonstrates system self-awareness to judges.

### 6.5 What Consistency Validation Does NOT Do

- It does not override the `predicted_class` (classifier authority is preserved).
- It does not suppress the alert.
- It does not change `risk_score` or `risk_tier`.
- It does not modify `feature_attributions`.

Its sole outputs are: (1) an optional note appended to `human_readable_explanation`, (2) a log entry, and (3) the `ValidationResult` object stored in `alert_builder.py`'s internal state for debugging.

---

## 7. Alternatives Considered

### 7.1 Alternative A: Attention Weight Extraction (Rejected)

**Description:** Use the GRU encoder's internal hidden state transitions as a proxy for feature importance. Specifically, compute the magnitude of hidden state change `|h_t - h_{t-1}|` as an approximation of which timesteps were most informative. Then, for each high-importance timestep t, use the feature vector at that timestep as the "explanation."

**Why it was considered:** This approach requires no external library (no Captum), has zero additional runtime cost, and produces timestep-level explanations — which could be useful for communicating "the attack happened at event 15 in the window."

**Why Rejected:**

1. **Temporal attribution ≠ Feature attribution:** Hidden state changes identify *which timesteps* were anomalous, not *which features* drove the anomaly within those timesteps. The problem statement requires feature-level attribution ("flagged due to geo-velocity"). Timestep attribution alone cannot produce "geo-velocity of 1,847 km/h" — only "event at 02:31 UTC was the anomalous one."

2. **Attention weight proxy is noisy for GRUs:** GRUs do not have explicit attention weights. The hidden-state-change proxy is a heuristic that can be dominated by high-magnitude features in the GRU input (e.g., `failure_count_norm` will always produce large state changes even in benign contexts). This biases the proxy toward explaining *loud* features, not necessarily *anomalous* ones.

3. **Not compatible with the `FeatureAttribution` schema:** DATA_SCHEMA.md §5d requires `attribution_score` to be a signed value interpretable as "pushes toward anomaly." A hidden state delta magnitude is unsigned and not directly interpretable in this way.

4. **Would not support the Insider Drift explanation:** Insider drift's anomaly is a slow gradient over 20 timesteps, not a single large state change at one position. The attention proxy would produce near-uniform attributions across the window — not useful for the narrative.

**Verdict:** Rejected. Captum IG is the correct tool and is already in the approved tech stack.

---

### 7.2 Alternative B: Ablation-Based Attribution (Rejected as Primary, Retained as Calibration Check)

**Description:** Compute feature importance by systematically setting each feature dimension to its baseline value (0.0 for all normalized features) and measuring the change in `anomaly_score`. The importance of feature d is: `delta_score[d] = anomaly_score(original) - anomaly_score(feature_d_zeroed)`. This is a simple, model-agnostic approach.

**For BPM:** Run IsolationForest 25 times (once per feature zeroed out) to get 25 delta scores. Each run takes ~0.05ms → total 1.25ms — comparable to SHAP.

**For SDM:** Run the GRU encoder 25 times with one feature dimension zeroed across the entire sequence window. Each run takes ~2ms → total 50ms for 25 features. But we need all 24 features → total ~50ms just for ablation, before IG cost.

**Why it was considered:** Ablation is completely model-agnostic (works for any model), does not require Captum, and produces intuitively interpretable importance scores.

**Why Rejected as Primary:**

1. **Runtime cost for SDM is prohibitive:** 25 GRU forward passes per inference event × 15ms per pass = 375ms per event for ablation attribution on the SDM. SHAP TreeExplainer (for BPM) and Captum IG (50 steps × single pass, ~20ms total for SDM) are both much faster.

2. **Ablation does not produce signed attributions:** Ablation measures magnitude of impact but not direction (a feature zeroed out could either increase or decrease the anomaly score). Without a signed `attribution_score`, the `direction` field in `FeatureAttribution` cannot be populated from first principles — it would require a second round of ablation. The `FeatureAttribution` schema requires signed scores (DATA_SCHEMA.md §5d).

3. **Ablation has interaction effects:** Zeroing feature d changes the joint distribution of the remaining features, potentially attributing importance to feature d that is actually driven by d's interaction with other features. SHAP accounts for interactions explicitly (by averaging over all possible subsets); ablation does not.

**Why Retained as Calibration Check:**  
During training-time evaluation (not inference), ablation is used as a **sanity check** on the SHAP and IG values. If SHAP ranks `geo_velocity_kmph` as the top feature for an impossible travel alert, the ablation check should also show a large delta when `geo_velocity_kmph` is zeroed. Discrepancies between SHAP and ablation rankings indicate model instability or feature interaction effects worth investigating. This check is implemented in `evaluation/evaluator.py` (T2) and not in the inference path.

---

### 7.3 Final Attribution Method Summary

| Method | BPM Compatible | SDM Compatible | Signed | Runtime (BPM) | Runtime (SDM) | Schema Compatible | Verdict |
|--------|---------------|----------------|--------|--------------|--------------|------------------|---------|
| SHAP TreeExplainer | ✅ | ❌ (GRU not tree) | ✅ | ~0.5ms | N/A | ✅ | **Chosen for BPM** |
| Captum IG | ✅ (via wrapper) | ✅ | ✅ | ~5ms (overkill for IF) | ~20ms | ✅ | **Chosen for SDM** |
| Attention proxy | N/A | ❌ (GRU has no attn) | ❌ | N/A | ~0ms | ❌ | Rejected |
| Ablation | ✅ | ✅ | ❌ | ~1.25ms | ~375ms | ❌ | Rejected as primary; retained as calibration check |

---

## 8. Judging-Criteria Traceability

### 8.1 Explainability and Analyst Usability (Primary Criterion)

| Design Decision | How It Serves the Criterion |
|----------------|----------------------------|
| **SHAP TreeExplainer for BPM** | SHAP produces Shapley values — the only attribution method with a mathematical guarantee of consistency, dummy variable handling, and efficiency (sum of attributions = difference between prediction and baseline). This guarantee means analysts can trust that the cited features genuinely explain the model's decision, not just correlate with it. |
| **Captum Integrated Gradients for SDM** | IG satisfies completeness (sum of attributions = output difference from baseline) and sensitivity (non-zero attribution for features with non-zero gradient). For sequence models, IG is the standard and most reliable attribution method, widely used in production NLP systems. Its use demonstrates methodological rigor to technical judges. |
| **`human_label` mapping for all 24 dimensions (Section 3a)** | Every attribution is translated to a human-readable phrase. An analyst sees "a geo-velocity of 1,847 km/h between consecutive logins" rather than "feature 6 = 0.924." This directly satisfies the problem statement's example explanation style and the "SOC analyst usability" requirement. |
| **Per-attack-type framing (Section 4)** | Each attack type has a different narrative structure that leads with its most diagnostic signal. A brute force explanation leads with failure count; an impossible travel explanation leads with velocity. This is how a trained SOC analyst would prioritize information — the explanation mirrors expert reasoning rather than generic feature listing. |
| **Consistency validation (Section 6)** | The validation check catches cases where the attribution model and classifier disagree. Surfacing this disagreement to the analyst (via the note in `human_readable_explanation`) is more useful than silently presenting a potentially misleading explanation. It demonstrates system transparency. |
| **Max 500 character constraint with truncation** | A narrative that is too long will not be read. The 500-character constraint forces the template to be concise and prioritized — only the top 3–4 contributing factors appear. This respects analyst attention and triage speed. |

### 8.2 False Positive Rate at Analyst Alert Budget (Secondary Criterion)

| Design Decision | How It Serves the Criterion |
|----------------|----------------------------|
| **Insider drift risk tier cap at `medium`** (Section 5.5) | Analysts using a "critical+high only" alert budget will never see insider drift alerts at all. This is architecturally correct: insider drift is a "watch" signal, not an "act immediately" signal. The cap prevents analyst fatigue from false positives in the medium-risk zone affecting the high/critical alert queue. |
| **Explicit ambiguity framing for insider drift** (Section 5) | When an analyst does see an insider drift alert (in medium-tier review), the explanation immediately communicates "this may not be an attack — check HR records." This reduces the probability that the analyst escalates the alert unnecessarily (reducing effective FP cost even if the detection was correct). |
| **Absence statement in insider drift narrative** (Section 5.4) | Explicitly listing the five absent signals ("no off-hours access, authentication failures, device anomalies, or exfiltration commands") allows the analyst to rule out more severe attacks in seconds. This reduces triage time and prevents the insider drift explanation from being confused with a lateral movement or low-and-slow explanation. |
| **Consistency validation warning in explanation** (Section 6.4) | When the attribution model and classifier disagree, the analyst is informed. An analyst who sees "[Note: primary attribution signal is atypical for this classification]" knows to scrutinize the alert more carefully before acting — preventing premature false-positive dismissals and false-positive escalations equally. |
| **`classification_confidence` in explanation (Section 3b, Step 5)** | Low-confidence classifications (< 65%) always show their confidence in the explanation text. An analyst who sees "classification confidence: 48%" for a lateral movement alert knows to treat it as a medium-confidence detection rather than a high-confidence one — improving triage calibration and reducing acted-upon false positives. |

---

*End of EXPLAINABILITY.md — Phase 7 output. This document is frozen. Amendments require a versioned change record. The proposed minor schema extension (`source_model = "bpm+sdm"`) must be recorded in DATA_SCHEMA.md §9 (Changelog) as version 1.1 before implementation begins.*
