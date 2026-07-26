# DATA_SCHEMA.md
# AI-Powered Behavioral Anomaly Detection — Data Schema Reference

> **Status:** Phase 4 — Frozen Data Contract Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Reads From:** ARCHITECTURE.md v1.0, PROJECT_STRUCTURE.md v1.0, CODING_GUIDELINES.md v1.0, TECH_STACK.md v1.0  
> **Scope:** Data shapes, field definitions, and type contracts for every boundary  
> named in ARCHITECTURE.md. No implementation code. This document is the single  
> source of truth for all future implementation phases.

---

## Table of Contents

1. [Architecture Consistency Check](#1-architecture-consistency-check)
2. [Raw Access Log Schema](#2-raw-access-log-schema)
   - 2a. [Training Schema](#2a-training-schema-boundary-a-full)
   - 2b. [Inference Schema](#2b-inference-schema-boundary-b-label-stripped)
   - 2c. [Label Stripping Policy](#2c-label-stripping-policy)
3. [Feature-Engineered Representation Schema](#3-feature-engineered-representation-schema-boundary-c)
4. [Model I/O Contracts](#4-model-io-contracts)
   - 4a. [Behavioral Profiling Model](#4a-behavioral-profiling-model-bpm)
   - 4b. [Sequence Detection Model](#4b-sequence-detection-model-sdm)
   - 4c. [Score Fusion](#4c-score-fusion)
   - 4d. [Anomaly Classifier](#4d-anomaly-classifier)
5. [Core Shared Objects](#5-core-shared-objects)
   - 5a. [AlertPayload](#5a-alertpayload-boundary-i)
   - 5b. [EntityProfile](#5b-entityprofile-boundary-e)
   - 5c. [EntityHistoryEntry](#5c-entityhistoryentry)
   - 5d. [FeatureAttribution](#5d-featureattribution)
   - 5e. [RenderedAlertData](#5e-renderedalertdata-boundary-k)
6. [Versioning Policy](#6-versioning-policy)
7. [Alternatives Considered](#7-alternatives-considered)
8. [Judging-Criteria Traceability](#8-judging-criteria-traceability)
9. [Changelog](#9-changelog)

---

## 1. Architecture Consistency Check

All prior-phase documents (ARCHITECTURE.md v1.0, PROJECT_STRUCTURE.md v1.0, CODING_GUIDELINES.md v1.0, TECH_STACK.md v1.0) and the problem statement were re-read before schema design began. The following cross-document consistency properties relevant to schema design were verified:

| Check | Result | Note |
|-------|--------|------|
| Boundary A (Raw Access Logs) field list in ARCHITECTURE.md matches problem statement's suggested schema | ✅ Pass | Architecture extends the problem statement with `event_id` and `session_id` (added in this document) |
| Label field stripping assigned to `streaming/` package in both ARCHITECTURE.md (Risk 3) and CODING_GUIDELINES.md (Section 5.3, item 4) | ✅ Pass | Enforced at boundary B; no downstream component ever holds a `label` field |
| Boundary C (Feature Vectors & Sequences) described as a `feature_vector` (normalized numeric array) + `sequence_window` in ARCHITECTURE.md — consistent with TECH_STACK.md's scikit-learn / PyTorch split | ✅ Pass | Feature vector feeds BPM (scikit-learn); sequence window feeds SDM (PyTorch) |
| CODING_GUIDELINES.md typo `EngineeeredFeatures` → correct canonical name is `EngineeredFeatures` | ✅ Noted | Correct name used throughout this document; must be fixed in `common/types.py` |
| `command_sequence` and `device_fingerprint` are listed as raw fields in boundary A but their internal structure is not defined in ARCHITECTURE.md — schema definition is correctly deferred to this phase | ✅ Pass | Concrete sub-structures defined in Section 2 |
| Alert payload boundary I fields (`risk_score`, `feature_attributions[]`, `human_readable_explanation`) are consistent with SHAP+Captum output structure chosen in TECH_STACK.md Decision 5 | ✅ Pass | |
| SQLite / in-memory stores chosen in TECH_STACK.md accept JSON-serialisable types; all schema types here are JSON-safe | ✅ Pass | No raw NumPy arrays stored in SQLite; feature vectors serialised as JSON arrays |
| Parquet chosen for `data/raw/` and `data/labeled/` — all field types here are Parquet-compatible | ✅ Pass | `list[str]` maps to Parquet `list<string>`; nested structs map to `struct` type |

**Consistency verdict:** No conflicting assumptions. Two schema extensions beyond the problem statement's suggested fields are introduced and flagged below: `event_id` (global unique event identifier required by every downstream boundary) and `session_id` (required to group events into sessions for sequence window construction). Neither implies a new architectural component.

---

## 2. Raw Access Log Schema

### 2a. Training Schema (Boundary A, Full)

**Produced by:** `data_generator/generator.py`  
**Consumed by:** `streaming/batch_reader.py` (or `simulated_stream.py`)  
**On-disk format:** Apache Parquet, one file per generation run, path `data/raw/synthetic_logs_<run_id>.parquet`  
**Label file:** Parallel file at `data/labeled/labels_<run_id>.parquet` containing only `event_id` + `label`  
**Python class:** `RawAccessLog` in `common/types.py`

#### Field Definitions

| # | Field Name | Python Type | Parquet Type | Nullable | Constraints | Purpose / Attack Relevance |
|---|-----------|-------------|--------------|----------|-------------|---------------------------|
| 1 | `event_id` | `str` | `string` | No | UUID v4; globally unique across all runs | Primary key for joining label file; referenced in every downstream boundary |
| 2 | `session_id` | `str` | `string` | No | UUID v4; groups related events; same session = same `entity_id` within one authenticated window | Required for sequence window construction at boundary C |
| 3 | `entity_id` | `str` | `string` | No | Format: `usr_<8hex>`, `svc_<8hex>`, or `dev_<8hex>` | The entity whose behavior is being modeled |
| 4 | `entity_type` | `str` | `string` | No | Enum: `user`, `service_account`, `edge_device` | Determines which behavioral profile peer group applies (cold-start T2) |
| 5 | `timestamp` | `str` | `timestamp(tz=UTC)` | No | ISO-8601 with UTC timezone; microsecond precision | Event ordering; geo-velocity calculation; hour-of-day features |
| 6 | `source_ip` | `str` | `string` | No | IPv4 dotted-decimal | Brute force detection (repeated IP); impossible travel (IP geolocation) |
| 7 | `geo_location` | `GeoLocation` | `struct` | No | See sub-structure below | Impossible travel detection; entity location baseline |
| 8 | `resource_accessed` | `str` | `string` | No | Format: `<category>/<identifier>`, e.g. `file/reports/q1.xlsx`, `port/22`, `api/admin/users` | Lateral movement (unusual resource set); low-and-slow (accumulating resource access) |
| 9 | `auth_method` | `str` | `string` | No | Enum: `password`, `token`, `certificate`, `biometric`, `none` | Credential misuse detection; auth failure rate |
| 10 | `auth_outcome` | `str` | `string` | No | Enum: `success`, `failure`, `mfa_required` | Brute force (high failure rate); credential stuffing (many entities, few IPs, high failure) |
| 11 | `session_duration` | `float` | `float` | No | Seconds; `0.0` for failed auth events (no session established) | Low-and-slow exfiltration (off-hours, long sessions); baseline session normality |
| 12 | `command_sequence` | `list[CommandEntry]` | `list<struct>` | No | Empty list `[]` for non-privileged sessions | Lateral movement (unusual command order); insider drift (expanding command repertoire) |
| 13 | `device_fingerprint` | `DeviceFingerprint` | `struct` | No | See sub-structure below | Device spoofing (fingerprint mismatch on known device ID) |
| 14 | `failure_count` | `int` | `int32` | No | Count of consecutive auth failures immediately preceding this event; `0` for success events | Brute force and credential stuffing: rapid repeated failures |
| 15 | `label` | `str` | `string` | **Training only** | Enum: see label taxonomy below; absent in inference schema | Ground-truth class; never crosses boundary B |

> **Schema Extensions vs. Problem Statement:**  
> `event_id` (field 1) and `session_id` (field 2) are added beyond the problem statement's suggested fields. `auth_outcome` (field 10) and `failure_count` (field 14) are also added. All four are required by specific attack detection logic: `event_id` is the universal join key across all 12 boundaries; `session_id` is the grouping key for sequence construction; `auth_outcome` and `failure_count` are the primary signal fields for brute force and credential stuffing, which cannot be detected without them from the raw event alone.

#### Label Taxonomy (Training Schema Only)

| Label Value | Attack Category | Maps to `AttackClass` enum |
|-------------|----------------|---------------------------|
| `normal` | No anomaly | `AttackClass.NORMAL` |
| `brute_force` | Rapid repeated auth failures | `AttackClass.BRUTE_FORCE` |
| `impossible_travel` | Geo-velocity violation | `AttackClass.IMPOSSIBLE_TRAVEL` |
| `credential_stuffing` | Many entities, few IPs, high failure rate | `AttackClass.CREDENTIAL_STUFFING` |
| `lateral_movement` | Unusual, broadening resource sequence | `AttackClass.LATERAL_MOVEMENT` |
| `device_spoofing` | Device ID reappears with mismatched fingerprint | `AttackClass.DEVICE_SPOOFING` |
| `low_and_slow` | Gradual off-hours resource accumulation | `AttackClass.LOW_AND_SLOW` |
| `insider_drift` | Legitimate entity gradually expanding footprint | `AttackClass.INSIDER_DRIFT` |

#### GeoLocation Sub-Structure

```
GeoLocation:
  city:        str     — e.g. "Mumbai" (Faker-generated)
  country:     str     — ISO 3166-1 alpha-2, e.g. "IN"
  latitude:    float   — WGS84, range [-90.0, 90.0]
  longitude:   float   — WGS84, range [-180.0, 180.0]
```

**Why separate lat/lon rather than encoding geohash?** Geo-velocity calculation (speed = distance / time between consecutive events) requires Haversine formula on raw lat/lon. A geohash encoding would require decoding first. Raw coordinates are kept for Feature Engineering transparency.

#### CommandEntry Sub-Structure

```
CommandEntry:
  sequence_position:  int     — 0-indexed position within the session
  command:            str     — e.g. "sudo", "ssh", "curl", "scp", "grep"
  target:             str     — resource or host the command acted on; empty string if N/A
  outcome:            str     — enum: "success", "failure", "denied"
  elapsed_seconds:    float   — seconds since session start at time of command
```

**Why this structure?** Lateral movement is detected by the sequence of (command, target) pairs, not by individual commands. The `sequence_position` enables the SDM to learn ordered patterns. `elapsed_seconds` enables low-and-slow detection at the command level.

**For non-privileged sessions:** `command_sequence` is an empty list `[]`. Feature Engineering maps this to a zero-length sequence, which the SDM treats as a baseline non-privileged session type.

#### DeviceFingerprint Sub-Structure

```
DeviceFingerprint:
  device_id:        str     — stable device identifier, e.g. "dev_3f8a21bc"
  os_family:        str     — e.g. "Windows", "Linux", "iOS", "Embedded/RTU"
  os_version:       str     — e.g. "11.0", "22.04", "16.3"
  mac_address:      str     — format: "AA:BB:CC:DD:EE:FF" (Faker-generated, not real)
  protocol:         str     — primary protocol used, e.g. "HTTPS", "Modbus", "MQTT", "RDP"
  user_agent:       str     — for HTTP-based access; empty string for non-HTTP protocols
  firmware_version: str     — for edge devices; empty string for standard OS endpoints
```

**Why this structure?** Device spoofing is detected by comparing the current `DeviceFingerprint` against the entity's stored historical fingerprint. A mismatch on `mac_address` or (`os_family`, `os_version`) pair for a known `device_id` triggers the spoofing signal. The `protocol` field is also part of the fingerprint baseline — a device that switches from `Modbus` to `HTTPS` is anomalous in an industrial context.

#### Example Row (Single Event)

```json
{
  "event_id":        "a3f7c821-dead-4be2-beef-000000000001",
  "session_id":      "sess_f2a1b3c4",
  "entity_id":       "usr_4d8e21bc",
  "entity_type":     "user",
  "timestamp":       "2026-07-15T02:31:45.123456Z",
  "source_ip":       "192.168.4.101",
  "geo_location": {
    "city":      "Kolkata",
    "country":   "IN",
    "latitude":  22.5726,
    "longitude": 88.3639
  },
  "resource_accessed": "file/finance/payroll_2026.xlsx",
  "auth_method":       "password",
  "auth_outcome":      "success",
  "session_duration":  1847.3,
  "command_sequence": [
    {"sequence_position": 0, "command": "ls",   "target": "/finance/", "outcome": "success", "elapsed_seconds": 12.1},
    {"sequence_position": 1, "command": "cat",  "target": "payroll_2026.xlsx", "outcome": "success", "elapsed_seconds": 45.8},
    {"sequence_position": 2, "command": "scp",  "target": "192.168.99.5:/tmp/", "outcome": "success", "elapsed_seconds": 120.3}
  ],
  "device_fingerprint": {
    "device_id":       "dev_3f8a21bc",
    "os_family":       "Windows",
    "os_version":      "11.0",
    "mac_address":     "AA:BB:CC:11:22:33",
    "protocol":        "HTTPS",
    "user_agent":      "Mozilla/5.0 (Windows NT 11.0)",
    "firmware_version": ""
  },
  "failure_count": 0,
  "label":         "low_and_slow"
}
```

---

### 2b. Inference Schema (Boundary B, Label Stripped)

**Produced by:** `streaming/batch_reader.py` (T1) or `streaming/simulated_stream.py` (T2)  
**Consumed by:** `feature_engineering/session_builder.py`  
**Python class:** `InboundEvent` in `common/types.py`

The `InboundEvent` is identical to `RawAccessLog` **minus the `label` field**, plus one metadata field:

| Field | Type | Added / Removed | Purpose |
|-------|------|----------------|---------|
| All fields from `RawAccessLog` except `label` | — | Retained | Normal pipeline use |
| `label` | — | **Removed** | Never crosses boundary B |
| `delivery_mode` | `str` | **Added** | Enum: `batch` or `simulated_stream`; used by Feature Engineering to apply batch-vs-stream specific windowing logic |

The `delivery_mode` field does not affect any ML computation. It is a routing hint used only within `feature_engineering/session_builder.py` to determine how to flush partial sequence windows. It is **not** propagated past boundary C.

---

### 2c. Label Stripping Policy

**Where stripping occurs:** Inside `streaming/batch_reader.py` (and `streaming/simulated_stream.py`) at the point where a `RawAccessLog` is converted into an `InboundEvent`. The `label` field is extracted from the `RawAccessLog` object and discarded. It is never written to the `InboundEvent` dataclass.

**Parallel label retention:** Ground-truth labels are written by `data_generator/label_store.py` to `data/labeled/labels_<run_id>.parquet` containing only (`event_id`, `label`) pairs. This file is accessed exclusively by `evaluation/evaluator.py` (T2) and never by any inference-path component.

**Enforcement contract:** Per CODING_GUIDELINES.md Section 5.3, item 4: "The `label` field from boundary A may appear in any file other than `data_generator/label_store.py` and `evaluation/evaluator.py`." Any code review or static analysis that finds `label` in any other file is a boundary violation.

**Testing:** `tests/streaming/test_batch_reader.py` must contain a test named `test_batch_reader_strips_label_field_from_output()` that asserts the `InboundEvent` produced by the batch reader does not have a `label` attribute.

---

## 3. Feature-Engineered Representation Schema (Boundary C)

**Produced by:** `feature_engineering/sequence_builder.py` (final assembly of boundary C)  
**Consumed by:** `models/behavioral_profiling/inference.py` (uses `feature_vector`) and `models/sequence_detection/inference.py` (uses `sequence_window`)  
**Python class:** `EngineeredFeatures` in `common/types.py`

### 3.1 Top-Level Structure

```
EngineeredFeatures:
  entity_id:         str              — propagated from InboundEvent; identifies the entity
  event_id:          str              — propagated from InboundEvent; the triggering event's ID
  session_id:        str              — propagated from InboundEvent; sequence grouping key
  feature_vector:    list[float]      — fixed-length normalized vector; see Section 3.2
  sequence_window:   list[list[float]]— sliding window of recent feature vectors; see Section 3.3
  session_metadata:  SessionMetadata  — non-numeric contextual flags; see Section 3.4
```

### 3.2 Feature Vector — 24 Named Dimensions (Fixed-Length)

The `feature_vector` is always length 24. Each dimension has a name, source field, transformation, and value range. The BPM receives this fixed-length vector as its primary input.

| Dim | Name | Source Field(s) | Transformation | Range | Attack Relevance |
|-----|------|----------------|----------------|-------|-----------------|
| 0 | `hour_of_day_sin` | `timestamp` | `sin(2π × hour / 24)` | [-1, 1] | Off-hours access baseline |
| 1 | `hour_of_day_cos` | `timestamp` | `cos(2π × hour / 24)` | [-1, 1] | Circular hour encoding (avoids 23→0 discontinuity) |
| 2 | `day_of_week_sin` | `timestamp` | `sin(2π × weekday / 7)` | [-1, 1] | Weekend vs. weekday baseline |
| 3 | `day_of_week_cos` | `timestamp` | `cos(2π × weekday / 7)` | [-1, 1] | Circular day encoding |
| 4 | `session_duration_norm` | `session_duration` | Min-max normalized over entity's historical distribution | [0, 1] | Low-and-slow (unusually long sessions) |
| 5 | `failure_count_norm` | `failure_count` | `min(failure_count / 20, 1.0)` — capped at 20 | [0, 1] | Brute force; credential stuffing |
| 6 | `geo_velocity_kmph` | `geo_location`, `timestamp` (vs. previous event) | Haversine distance / elapsed time; capped at 2000 km/h | [0, 1] after `/2000` | Impossible travel |
| 7 | `is_new_geo` | `geo_location` vs. entity profile | 1.0 if country differs from entity's most frequent country, else 0.0 | {0, 1} | Impossible travel; general anomaly |
| 8 | `resource_category_enc` | `resource_accessed` | Label-encoded category prefix (`file`=0, `port`=1, `api`=2, `db`=3, `device`=4, `other`=5), divided by 5 | [0, 1] | Lateral movement (shifting resource categories) |
| 9 | `resource_rarity_score` | `resource_accessed` vs. entity profile | Fraction of entity's historical events accessing this resource (inverted: 1 − freq); 1.0 for never-seen | [0, 1] | Lateral movement; low-and-slow |
| 10 | `auth_method_enc` | `auth_method` | One-hot position: password=0, token=1, cert=2, biometric=3, none=4; divided by 4 | [0, 1] | Credential misuse (method switch) |
| 11 | `auth_outcome_enc` | `auth_outcome` | success=0.0, mfa_required=0.5, failure=1.0 | {0, 0.5, 1} | Brute force; credential stuffing |
| 12 | `command_seq_length_norm` | `command_sequence` | `min(len(command_sequence) / 50, 1.0)` | [0, 1] | Lateral movement (long unusual command chains) |
| 13 | `command_rarity_score` | `command_sequence` vs. entity profile | Mean per-command rarity over sequence; 1.0 if empty sequence | [0, 1] | Insider drift; lateral movement |
| 14 | `has_exfil_command` | `command_sequence` | 1.0 if any command is in `{scp, rsync, ftp, curl, wget, nc}`; else 0.0 | {0, 1} | Low-and-slow; lateral movement |
| 15 | `fingerprint_os_match` | `device_fingerprint` vs. entity profile | 1.0 if `os_family` + `os_version` match stored baseline; 0.0 otherwise | {0, 1} | Device spoofing |
| 16 | `fingerprint_mac_match` | `device_fingerprint` vs. entity profile | 1.0 if `mac_address` matches stored baseline; 0.0 otherwise | {0, 1} | Device spoofing (primary signal) |
| 17 | `fingerprint_protocol_match` | `device_fingerprint` vs. entity profile | 1.0 if `protocol` matches stored baseline; 0.0 otherwise | {0, 1} | Device spoofing |
| 18 | `entity_type_enc` | `entity_type` | user=0.0, service_account=0.5, edge_device=1.0 | {0, 0.5, 1} | Peer group differentiation |
| 19 | `inter_event_gap_norm` | `timestamp` vs. previous event for this entity | `min(gap_seconds / 86400, 1.0)` — normalized to 1 day | [0, 1] | Low-and-slow (long silent gaps before access) |
| 20 | `session_event_count_norm` | Count of events in this session so far | `min(count / 200, 1.0)` | [0, 1] | Session depth; brute force (many events per session) |
| 21 | `resource_breadth_norm` | Count of distinct resources accessed in this session | `min(distinct / 50, 1.0)` | [0, 1] | Lateral movement (many distinct resources) |
| 22 | `ip_entity_ratio` | Events from this IP / events for this entity (sliding 24h window) | `min(ratio / 10, 1.0)` | [0, 1] | Credential stuffing (1 IP, many entities) |
| 23 | `entity_ip_ratio` | Distinct IPs used by this entity / events for this entity (24h window) | `min(ratio / 5, 1.0)` | [0, 1] | Compromised credentials (many IPs for 1 entity) |

> **Normalization note:** All normalizations are computed against entity-level rolling statistics stored in the Entity Profile Store (boundary E). For cold-start entities (T2), the global population mean and standard deviation are used as fallback normalization parameters (Cold-Start Handler slot, Phase 11).

### 3.3 Sequence Window (SDM Input)

```
sequence_window: list[list[float]]
  — Shape: (W, 24) where W = window length (default 20, configurable)
  — Each element is a feature_vector for one past event of this entity
  — Ordered chronologically: index 0 = oldest event in window, index W-1 = current event
  — If entity has fewer than W events, the window is left-padded with zero vectors
  — PyTorch tensor shape at SDM input: torch.Tensor of shape (1, W, 24)
    (batch=1, sequence=W, features=24)
```

**Why fixed-length W with zero-padding rather than variable-length?** PyTorch recurrent models require a fixed batch tensor shape for efficient training. Variable-length sequences require `pack_padded_sequence`, which adds implementation complexity with no benefit for hackathon time budget. Zero-padding with a mask is the standard approach and is transparent to judges.

### 3.4 SessionMetadata (Contextual Flags, Not Fed to Models)

```
SessionMetadata:
  is_cold_start:       bool   — True if entity has fewer than MIN_PROFILE_EVENTS historical events
  delivery_mode_hint:  str    — "batch" or "simulated_stream"; for FE internal use only
  profile_event_count: int    — How many historical events exist for this entity at inference time
```

`SessionMetadata` is **not** included in the `feature_vector` or `sequence_window`. It is passed alongside the `EngineeredFeatures` object within the pipeline but is not fed to any ML model. It is used by: (1) the BPM to attach `cold_start_flag` to its output (boundary F), and (2) the Cold-Start Handler (T2) to decide whether to substitute a prior profile.

---

## 4. Model I/O Contracts

### 4a. Behavioral Profiling Model (BPM)

**Boundary consumed:** C (EngineeredFeatures → specifically `feature_vector`)  
**Boundary consumed:** E (EntityProfile → provides baseline for normality scoring)  
**Boundary produced:** F (ModelScore, `model_id="bpm"`)

#### Input

| Field | Type | Shape/Constraint |
|-------|------|-----------------|
| `feature_vector` | `list[float]` or `np.ndarray` | Length 24; all values in [0, 1] or [-1, 1] depending on dimension |
| `entity_profile` | `EntityProfile` | See Section 5b; must not be null (Cold-Start Handler guarantee) |

#### Output — `ModelScore` with `model_id="bpm"`

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `entity_id` | `str` | Non-empty | Propagated from input |
| `event_id` | `str` | Non-empty | Propagated from input |
| `model_id` | `str` | `"bpm"` | Identifies the producing model |
| `anomaly_score` | `float` | [0.0, 1.0] | 0.0 = perfectly normal; 1.0 = maximally anomalous |
| `confidence` | `float` | [0.0, 1.0] | Model's confidence in the score; lower for cold-start entities |
| `cold_start_flag` | `bool` | — | Propagated from `SessionMetadata.is_cold_start` |
| `top_contributing_features` | `list[str]` | Max 5 feature names | Feature dimension names (from Section 3.2) with highest absolute attribution; populated by SHAP at inference time |

**Anomaly score derivation:** The BPM transforms the model's raw output (e.g., `decision_function` for OneClassSVM, `score_samples` for IsolationForest) into [0, 1] using a monotonic normalization calibrated on training data. The normalization function is stored alongside the model artifact.

---

### 4b. Sequence Detection Model (SDM)

**Boundary consumed:** C (EngineeredFeatures → specifically `sequence_window`)  
**Boundary consumed:** E (EntityProfile → `sequence_history` for comparison baseline)  
**Boundary produced:** F (ModelScore, `model_id="sdm"`)

#### Input

| Field | Type | PyTorch shape | Constraint |
|-------|------|--------------|------------|
| `sequence_window` | `torch.Tensor` | `(1, W, 24)` | W=20 by default; float32; left-padded with zeros if entity has < W events |
| `sequence_mask` | `torch.Tensor` | `(1, W)` | bool; True for real events, False for padding positions |

#### Output — `ModelScore` with `model_id="sdm"`

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `entity_id` | `str` | Non-empty | Propagated from input |
| `event_id` | `str` | Non-empty | Propagated from input |
| `model_id` | `str` | `"sdm"` | Identifies the producing model |
| `anomaly_score` | `float` | [0.0, 1.0] | 0.0 = sequence is normal; 1.0 = maximally anomalous sequence |
| `confidence` | `float` | [0.0, 1.0] | Lower when window is mostly zero-padded (short entity history) |
| `cold_start_flag` | `bool` | — | True if padding fraction > 0.5 (fewer than W/2 real events) |
| `top_contributing_features` | `list[str]` | Max 5 feature names | Feature names with highest Captum Integrated Gradients attribution across the window |

**Sequence anomaly score derivation:** The SDM outputs a reconstruction error (for autoencoder-style) or a sequence likelihood (for predictive models). This raw output is normalized to [0, 1] using the same monotonic normalization approach as the BPM, calibrated on training data. The specific normalization is a Phase 6 implementation decision.

---

### 4c. Score Fusion

**Boundary consumed:** F (two `ModelScore` objects — one BPM, one SDM)  
**Boundary produced:** G (`UnifiedAnomalySignal`)

#### Input

| Field | Type | Constraint |
|-------|------|-----------|
| `bpm_score` | `ModelScore` | `model_id` must be `"bpm"` |
| `sdm_score` | `ModelScore` | `model_id` must be `"sdm"` |
| `fusion_threshold` | `float` | [0.0, 1.0]; read from `config/default.yaml`; default 0.5 |
| `fusion_weights` | `tuple[float, float]` | (bpm_weight, sdm_weight); must sum to 1.0; default (0.5, 0.5) |

#### Output — `UnifiedAnomalySignal` (Boundary G)

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `entity_id` | `str` | Non-empty | Propagated |
| `event_id` | `str` | Non-empty | Propagated |
| `fused_score` | `float` | [0.0, 1.0] | `bpm_weight × bpm_score.anomaly_score + sdm_weight × sdm_score.anomaly_score` |
| `is_anomaly` | `bool` | — | `fused_score >= fusion_threshold` |
| `bpm_score` | `float` | [0.0, 1.0] | Preserved for observability (Risk 2 mitigation in ARCHITECTURE.md) |
| `sdm_score` | `float` | [0.0, 1.0] | Preserved for observability |
| `cold_start_flag` | `bool` | — | `bpm_score.cold_start_flag OR sdm_score.cold_start_flag` |
| `contributing_features` | `list[str]` | Union of top features from both models; deduplicated | Used by Explainability Layer |

---

### 4d. Anomaly Classifier

**Boundary consumed:** G (`UnifiedAnomalySignal`)  
**Boundary produced:** H (`ClassificationResult`)

#### Input

| Field | Type | Constraint | Note |
|-------|------|-----------|------|
| `fused_score` | `float` | [0.0, 1.0] | Primary signal |
| `bpm_score` | `float` | [0.0, 1.0] | Secondary signal |
| `sdm_score` | `float` | [0.0, 1.0] | Secondary signal |
| `contributing_features` | `list[str]` | Max 10 feature names | Used as input features to the classifier |
| `feature_vector` | `list[float]` | Length 24 | Full feature vector passed through from boundary C via the pipeline context; required for classification |

> **Note:** The Anomaly Classifier receives the full `feature_vector` in addition to the `UnifiedAnomalySignal` fields. The full vector is required because attack class discrimination depends on features (e.g., `geo_velocity_kmph` for impossible travel, `fingerprint_mac_match` for device spoofing) that may not appear in `contributing_features` from BPM/SDM attribution. The full vector is passed through the pipeline context (managed by the API orchestration layer), not embedded in the boundary G object itself. This avoids enlarging boundary G unnecessarily.

#### Output — `ClassificationResult` (Boundary H)

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `entity_id` | `str` | Non-empty | Propagated |
| `event_id` | `str` | Non-empty | Propagated |
| `predicted_class` | `str` | Must be one of the 8 label values in the label taxonomy | The most likely attack category |
| `class_probabilities` | `dict[str, float]` | Keys = all 8 label values; values sum to 1.0 | Full posterior distribution over attack classes |
| `classification_confidence` | `float` | [0.0, 1.0] | `max(class_probabilities.values())` — the winning class's probability |
| `is_anomaly` | `bool` | — | Propagated from `UnifiedAnomalySignal.is_anomaly` |

---

## 5. Core Shared Objects

All objects in this section are defined in `common/types.py` and are used across the API, stores, and dashboard boundaries.

### 5a. AlertPayload (Boundary I)

**Produced by:** `explainability/alert_builder.py`  
**Consumed by:** `stores/alert_store.py` (write), `api/routers/alerts.py` (read)  
**Persisted as:** JSON-serialized row in the Alert & Result Store (SQLite)

| Field | Type | Nullable | Constraint | Description |
|-------|------|----------|-----------|-------------|
| `alert_id` | `str` | No | UUID v4; generated by `alert_builder.py` | Primary key for alert retrieval |
| `entity_id` | `str` | No | Non-empty | Entity that triggered the alert |
| `event_id` | `str` | No | Non-empty | The specific event that triggered scoring |
| `session_id` | `str` | No | Non-empty | Session containing the triggering event |
| `timestamp` | `str` | No | ISO-8601 UTC; copied from the raw event | When the event occurred |
| `detected_at` | `str` | No | ISO-8601 UTC; set by `alert_builder.py` | When the alert was generated (may differ from event time in batch mode) |
| `risk_score` | `int` | No | [0, 100] | Composite risk score; see `RiskScore` breakdown in Section 5d |
| `risk_tier` | `str` | No | Enum: `low` (0–24), `medium` (25–49), `high` (50–74), `critical` (75–100) | Analyst routing tier |
| `attack_class` | `str` | No | One of the 8 label taxonomy values | Predicted attack category |
| `classification_confidence` | `float` | No | [0.0, 1.0] | Classifier confidence in `attack_class` |
| `fused_score` | `float` | No | [0.0, 1.0] | Raw fused anomaly score |
| `bpm_score` | `float` | No | [0.0, 1.0] | BPM component score (for observability) |
| `sdm_score` | `float` | No | [0.0, 1.0] | SDM component score (for observability) |
| `cold_start_flag` | `bool` | No | — | True if entity had insufficient history at inference time |
| `human_readable_explanation` | `str` | No | Max 500 characters | Natural language explanation, e.g. "Flagged due to geo-velocity of 1,847 km/h combined with a new device fingerprint (OS mismatch) and an off-hours login at 02:31 UTC." |
| `feature_attributions` | `list[FeatureAttribution]` | No | 1–10 entries | Ordered by descending absolute attribution magnitude |
| `raw_event_snapshot` | `dict` | No | Serialized `InboundEvent` minus `delivery_mode` | Preserved for analyst drill-down; does not contain `label` |
| `analyst_decision` | `str` | Yes (T3) | Enum: `true_positive`, `false_positive`, `needs_review`; null until analyst acts | Analyst feedback (T3 feature, written by `api/routers/feedback.py`) |
| `analyst_notes` | `str` | Yes (T3) | Max 2000 characters; null until analyst acts | Analyst annotation (T3) |

**Example `human_readable_explanation` for each attack type:**

| Attack Class | Example Explanation |
|-------------|-------------------|
| `brute_force` | "Flagged due to 14 consecutive authentication failures within 45 seconds from IP 10.0.0.8." |
| `impossible_travel` | "Flagged due to geo-velocity of 1,847 km/h between Mumbai (02:31 UTC) and London (02:47 UTC) — physically impossible for this entity." |
| `credential_stuffing` | "Flagged due to high authentication failure rate (78%) across 42 distinct entity IDs from a single source IP within 3 minutes." |
| `lateral_movement` | "Flagged due to access to 17 resources across 5 distinct categories in one session, compared to a historical baseline of 2 resource categories." |
| `device_spoofing` | "Flagged due to device ID dev_3f8a21bc reappearing with a different MAC address (AA:BB:CC:99:88:77 vs. stored AA:BB:CC:11:22:33) and OS family switch from Windows to Linux." |
| `low_and_slow` | "Flagged due to 23 off-hours file access events accumulating over 11 days, each individually below the anomaly threshold but collectively indicating progressive exfiltration." |
| `insider_drift` | "Flagged due to a gradual 340% expansion in resource footprint over 30 days. Current access pattern overlaps with the lateral_movement profile but below decision threshold — classified as edge case." |
| `normal` | Not applicable; normal events do not generate alerts. |

---

### 5b. EntityProfile (Boundary E)

**Produced and served by:** `stores/profile_store.py`  
**Consumed by:** `models/behavioral_profiling/inference.py`, `models/sequence_detection/inference.py`, `cold_start/handler.py` (T2), `drift/monitor.py` (T2)

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `entity_id` | `str` | No | Primary key |
| `entity_type` | `str` | No | `user`, `service_account`, or `edge_device` |
| `baseline_vector` | `list[float]` | No | Rolling mean of `feature_vector` over last N events; length 24; populated after MIN_PROFILE_EVENTS events |
| `baseline_std` | `list[float]` | No | Rolling standard deviation; length 24; used for z-score normalization within BPM |
| `sequence_history` | `list[list[float]]` | No | Last W (default 20) `feature_vector` entries, chronologically ordered; used to populate `sequence_window` for SDM |
| `most_frequent_country` | `str` | No | ISO 3166-1 alpha-2; used for `is_new_geo` feature computation |
| `known_mac_addresses` | `list[str]` | No | Set of MAC addresses seen for this entity; used for `fingerprint_mac_match` |
| `known_os_profiles` | `list[dict]` | No | Set of `{os_family, os_version}` dicts seen for this entity |
| `known_protocols` | `list[str]` | No | Protocols seen for this entity |
| `resource_access_counts` | `dict[str, int]` | No | `{resource_identifier: count}`; used for `resource_rarity_score` |
| `command_frequency` | `dict[str, int]` | No | `{command: count}`; used for `command_rarity_score` |
| `event_count` | `int` | No | Total events seen for this entity; determines `cold_start_flag` |
| `cold_start_flag` | `bool` | No | `event_count < MIN_PROFILE_EVENTS` (default: 10, configurable) |
| `drift_metrics` | `DriftMetrics` | No (T2) | Rolling distribution statistics for drift detection; empty dict in T1 |
| `last_updated` | `str` | No | ISO-8601 UTC timestamp of last profile update |
| `profile_version` | `int` | No | Monotonically increasing; incremented on every upsert; used for drift-triggered retraining isolation (ARCHITECTURE.md Risk 6) |

#### DriftMetrics Sub-Structure (T2 field, empty dict in T1)

```
DriftMetrics:
  feature_means_history:   list[list[float]]  — last K baseline_vector snapshots (K configurable)
  last_drift_check:        str                — ISO-8601 UTC
  drift_severity:          str                — enum: "none", "low", "medium", "high"
  drift_detected_at:       str | null         — ISO-8601 UTC of most recent detected drift event
```

---

### 5c. EntityHistoryEntry

**Produced by:** `stores/alert_store.py` (query)  
**Consumed by:** `api/routers/entities.py` (assembles list for boundary K)  
**Displayed by:** `dashboard/scripts/entity_view.js` (T1), `dashboard/scripts/timeline_view.js` (T2)

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | `str` | Event identifier |
| `timestamp` | `str` | ISO-8601 UTC |
| `resource_accessed` | `str` | Resource for this event |
| `auth_outcome` | `str` | `success`, `failure`, `mfa_required` |
| `risk_score` | `int` | Alert risk score if an alert was generated; `null` if event was normal (no alert) |
| `attack_class` | `str` | Alert attack class if an alert was generated; `"normal"` if no alert |
| `has_alert` | `bool` | True if this event generated an alert in the Alert Store |

---

### 5d. FeatureAttribution

**Produced by:** `explainability/feature_attribution.py`  
**Embedded in:** `AlertPayload.feature_attributions` (list)

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `feature_name` | `str` | One of the 24 named dimensions from Section 3.2 | Which feature contributed |
| `feature_value` | `float` | Actual normalized value at inference time | What the feature value was |
| `attribution_score` | `float` | Signed; magnitude indicates importance | SHAP value (BPM) or Captum Integrated Gradient (SDM); positive = pushes toward anomalous |
| `direction` | `str` | `"toward_anomaly"` if `attribution_score > 0`, else `"toward_normal"` | Human-readable direction |
| `source_model` | `str` | `"bpm"`, `"sdm"`, or `"bpm+sdm"` | Which model this attribution came from |
| `human_label` | `str` | Plain-English feature description | e.g. `"geo_velocity_kmph"` → "Speed between consecutive logins (km/h)" |

The `human_label` mapping for all 24 dimensions is maintained as a constant dict in `explainability/narrative.py`.

---

### 5e. RenderedAlertData (Boundary K)

**Produced by:** `api/routers/alerts.py` and `api/routers/entities.py`  
**Consumed by:** `dashboard/scripts/api_client.js`  
**Format:** JSON over REST (HTTP GET responses)

#### Alerts List Response

```
RenderedAlertData:
  alerts:       list[AlertSummary]   — paginated; sorted by risk_score DESC, then timestamp DESC
  total_count:  int                  — total alerts matching the query (for pagination)
  page:         int                  — current page number (1-indexed)
  page_size:    int                  — number of alerts per page; default 50
  filters:      AlertFilters         — echo of applied filters
```

#### AlertSummary (lightweight, for the alert queue table)

```
AlertSummary:
  alert_id:                 str
  entity_id:                str
  timestamp:                str     — ISO-8601 UTC
  risk_score:               int     — [0, 100]
  risk_tier:                str     — "low" | "medium" | "high" | "critical"
  attack_class:             str
  classification_confidence: float
  cold_start_flag:          bool
  human_readable_explanation: str   — truncated to 150 chars for queue display
```

#### Alert Detail Response (single alert drill-down)

```
AlertDetail:
  (all AlertSummary fields, full human_readable_explanation)
  fused_score:          float
  bpm_score:            float
  sdm_score:            float
  feature_attributions: list[FeatureAttribution]
  raw_event_snapshot:   dict
  entity_history:       list[EntityHistoryEntry]  — last 50 events for this entity
```

#### AlertFilters

```
AlertFilters:
  risk_tier:    list[str] | null  — filter by tier(s)
  attack_class: list[str] | null  — filter by class(es)
  entity_id:    str | null        — filter to single entity
  since:        str | null        — ISO-8601 UTC lower bound on timestamp
  until:        str | null        — ISO-8601 UTC upper bound on timestamp
```

---

## 6. Versioning Policy

### 6.1 Schema Version Field

The `DATA_SCHEMA.md` document carries a version in the header (`Version: 1.0`). Any change to a field's name, type, constraint, or to the addition/removal of a field in any of the schemas defined above constitutes a schema change and requires:

1. Incrementing the version number (semantic versioning: `MAJOR.MINOR`; MAJOR for breaking changes, MINOR for additive-only changes).
2. Adding an entry to the Changelog (Section 9) with: date, version, affected schema(s), description of change, and which pipeline components must be updated.
3. Updating `common/types.py` to match.
4. A corresponding entry in `ARCHITECTURE.md`'s versioned change record if the change affects a boundary contract.

### 6.2 Breaking vs. Non-Breaking Changes

| Change Type | Classification | MAJOR / MINOR |
|-------------|---------------|--------------|
| Removing a field from any boundary contract | Breaking | MAJOR |
| Renaming a field | Breaking | MAJOR |
| Changing a field's type | Breaking | MAJOR |
| Adding a required field | Breaking (consumers must update) | MAJOR |
| Adding a nullable / optional field | Non-breaking | MINOR |
| Changing a constraint (e.g., widening range) | Non-breaking | MINOR |
| Changing a constraint (e.g., narrowing range) | Breaking | MAJOR |
| Adding a new label taxonomy value | Breaking (classifier must retrain) | MAJOR |

### 6.3 Inference Schema Isolation

Any MAJOR change to the Training Schema (Section 2a) that also changes the Inference Schema (Section 2b) requires re-evaluating the label stripping policy in Section 2c and updating `common/types.py`'s `RawAccessLog` and `InboundEvent` classes simultaneously. Both classes must remain in sync in the same `types.py` commit.

### 6.4 Feature Vector Dimensionality Lock

The feature vector length of **24** is locked at Phase 4. Changing it is a MAJOR breaking change requiring retraining of BPM, SDM, and Anomaly Classifier. Phase 5 (BPM design) and Phase 6 (SDM design) must accept this length as fixed. If Phase 5 or 6 determines that a different length is required, that is a proposed architecture change that must be flagged, reviewed, and versioned before implementation.

---

## 7. Alternatives Considered

### 7.1 Feature Representation Format: Fixed Vector vs. Learned Embedding vs. Graph

**Decision:** Fixed-length 24-dimensional normalized vector (chosen).

**Alternative A: Variable-Length Sequence with No Fixed Vector**  
Description: Do not construct a fixed feature vector. Instead, feed raw event fields directly to the SDM as a tokenized sequence (similar to BERT for tabular data), with the BPM operating on event-level embeddings rather than hand-engineered features.

Why Rejected:
- Requires a learned tokenization/embedding layer that adds a third model training pipeline (tokenizer → embedding → BPM/SDM), significantly exceeding the hackathon time budget.
- SHAP attribution cannot easily be applied to a learned embedding layer: the attributions would explain embedding dimensions, not human-interpretable feature names. This directly undermines the "explainability and analyst usability" judging criterion.
- The problem statement's suggested synthetic data schema maps directly to hand-engineered features with clear semantic meaning; hand engineering is not a limitation here — it is an asset for explainability.

**Alternative B: Graph Structure (Entity-Resource Bipartite Graph)**  
Description: Represent each event as an edge in a bipartite graph (entity nodes ↔ resource nodes), and use a Graph Neural Network (GNN) for anomaly scoring.

Why Rejected:
- PyTorch Geometric (PyG) adds a heavy additional dependency not in the current TECH_STACK.md.
- GNN training requires substantially more data and longer training time than a recurrent sequence model on the same hardware.
- Explainability of GNN outputs is an active research problem; no mature library provides reliable GNN attribution comparable to Captum for RNNs.
- The graph approach would require a structural change to the feature engineering pipeline (graph construction module) — a new component not in ARCHITECTURE.md. This must be flagged as a proposed change rather than silently introduced.

**Why the Chosen Approach Wins:**  
The 24-dimensional fixed vector is directly explainable (each dimension has a human-readable name), directly supports SHAP attribution without adapter layers, supports both the BPM (scikit-learn, expects 2D array inputs) and SDM (PyTorch, expects 3D tensors via windowing), and requires no additional libraries beyond those in TECH_STACK.md.

---

### 7.2 Alert Object Nesting Strategy: Flat vs. Nested

**Decision:** Nested with sub-objects (`feature_attributions: list[FeatureAttribution]`, `raw_event_snapshot: dict`) (chosen).

**Alternative A: Fully Flat Alert (All Fields at Top Level)**  
Description: Expand all nested objects into the `AlertPayload` top level. `feature_attributions` becomes `attr_feature_0`, `attr_score_0`, ..., `attr_feature_9`, `attr_score_9`. `raw_event_snapshot` fields become `ev_entity_id`, `ev_timestamp`, etc.

Why Rejected:
- A flat alert with up to 10 attribution slots + all raw event fields would exceed 60 top-level fields, making the object unreadable and serialization schema fragile.
- Flat structures cannot represent variable-length attribution lists cleanly (the number of attributions per event varies from 1 to 10).
- SQLite storage of a flat object with 60+ columns is correct but wasteful; JSON column storage of the nested alert is simpler to query and extend.
- Dashboard rendering of attribution data from a flat structure requires index-based parsing (`attr_feature_0` through `attr_feature_9`), which is more brittle than iterating over `feature_attributions`.

**Alternative B: Fully Normalized Relational (Separate DB Tables per Sub-Object)**  
Description: Store `AlertPayload`, `FeatureAttribution`, and `EntityHistoryEntry` in separate SQLite tables with foreign keys. The API assembles the nested object at read time via JOIN.

Why Rejected:
- Requires a SQL query at read time that JOINs 3–4 tables per alert retrieval, adding query complexity for a hackathon demo where SQLite performance is not a bottleneck.
- The Alert & Result Store's interface (boundary J) is defined as reading alert records — not as a relational query layer. Adding JOINs to the store would leak SQL logic into the `alert_store.py` interface in a way that makes the T2 Redis upgrade harder (Redis does not support JOINs).
- For the Tier 1 demo, the number of alerts is small (controlled injection rate of 0.5%–3% of events). Full normalization is premature optimization.

**Why the Chosen Approach Wins:**  
Nested JSON stored in a single SQLite column is the right fit for the T1 scope. The `AlertPayload` is a read-once, write-once object (written when detected, read by API). Sub-objects are always retrieved together, never queried independently. Nesting matches this access pattern. The API serializes the nested object directly via Pydantic v2, which supports nested model serialization natively.

---

## 8. Judging-Criteria Traceability

| Judging Criterion | Schema Decision | How It Supports the Criterion |
|------------------|----------------|-------------------------------|
| **Detection accuracy on imbalanced labels** | `failure_count` and `auth_outcome` as explicit raw fields; `failure_count_norm` and `auth_outcome_enc` as dedicated feature vector dimensions | Brute force and credential stuffing are the two attack types most dependent on failure rate signals. Making these first-class fields rather than deriving them from the command sequence ensures they are never lost during feature engineering. |
| **Correct anomaly-type classification** | Full `feature_vector` passed through pipeline context to the Anomaly Classifier (Section 4d) | The Classifier receives the full 24-dimensional vector, not just the top contributing features from BPM/SDM. This ensures attack-discriminating features (e.g., `fingerprint_mac_match` for device spoofing, `geo_velocity_kmph` for impossible travel) are always available for classification even if BPM/SDM did not rank them highest. |
| **False positive rate at analyst alert budget** | `risk_tier` enum on `AlertPayload`; `AlertSummary` sorted by `risk_score` DESC in boundary K | Judges can be shown the alert queue sorted by risk score. An analyst budget of "top 1% of events" maps directly to showing only `critical` and `high` tier alerts. The risk tier thresholds (0–24 low, 25–49 medium, 50–74 high, 75–100 critical) are calibrated against this. |
| **Explainability and analyst usability** | `FeatureAttribution` with `feature_name`, `attribution_score`, `direction`, `human_label`; `human_readable_explanation` in `AlertPayload` | The `human_label` field provides a plain-English name for every feature (e.g., "Speed between consecutive logins (km/h)" for `geo_velocity_kmph`). The narrative template in `explainability/narrative.py` uses these labels to construct the human-readable explanation that matches the problem statement's example format ("Flagged due to geo-velocity combined with a new device fingerprint"). |
| **Handling cold-start entities** | `cold_start_flag` propagated through boundaries F, G, and I; visible in `AlertSummary` in boundary K | Analysts see the `cold_start_flag` in the alert queue, enabling them to appropriately weight alerts from new entities. The flag is an architectural first-class field — not a comment or log message — ensuring the dashboard can render a visual warning for cold-start alerts. |
| **Handling concept drift** | `profile_version` on `EntityProfile`; `DriftMetrics` sub-structure as a T2 slot | The `profile_version` ensures that if a drift-triggered retrain creates a new profile, the old profile version is preserved. The `drift_detected_at` field in `DriftMetrics` provides a timestamped record that can be displayed in the entity timeline (T2 dashboard panel). |
| **System design and scalability** | Strict Training/Inference schema split (Section 2a vs. 2b); label stripping at a single enforced point | The enforcement of label separation at boundary B — not scattered through every downstream component — directly demonstrates the clean architectural separation that production ML systems require to prevent evaluation leakage. This is directly inspectable by judges in the codebase. |
| **Report clarity** | `event_id` and `session_id` as universal join keys; every field traced to a specific attack or pipeline requirement | Every field in this document has a stated purpose in the "Purpose / Attack Relevance" column. The Technical Report (Phase-final) can directly cite this document as evidence that no field is speculative. |

---

## 9. Changelog

| Version | Date | Author | Change Summary | Affected Schemas | Components to Update |
|---------|------|--------|----------------|-----------------|---------------------|
| 1.0 | 2026-07-24 | Initial version | | | |
| 1.1 | 2026-07-25 | Refined `ClassificationOutput` signature (M08) | Boundary H | M08 Classifier |
| 1.2 | 2026-07-26 | Added `"bpm+sdm"` to `FeatureAttribution.source_model` (M09) | FeatureAttribution | API/Dashboard |

---

*End of DATA_SCHEMA.md — Phase 4 output. This document is frozen. Amendments require a versioned change record in Section 9 and a corresponding entry in ARCHITECTURE.md if any boundary contract field is affected.*
