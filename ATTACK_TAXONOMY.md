# ATTACK_TAXONOMY.md
# AI-Powered Behavioral Anomaly Detection — Attack Taxonomy Reference

> **Status:** Phase 5 — Frozen Taxonomy Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Source:** SYNTHETIC_DATA_GENERATOR_DESIGN.md v1.0, DATA_SCHEMA.md v1.0  
> **Scope:** Standalone summary table for use in Technical Report and presentation.  
> Every row corresponds to a label value in DATA_SCHEMA.md Section 2a Label Taxonomy.  
> **Reuse note:** This document is designed to be included directly in `docs/report/TECHNICAL_REPORT.md`  
> and presentation slides without modification.

---

## Summary Table

| # | Attack Type | Label Value | Signal Type | Detection Difficulty | Expected Risk Tier |
|---|-------------|-------------|-------------|--------------------|--------------------|
| — | Normal Baseline | `normal` | Benign | N/A | N/A |
| 1 | Brute Force | `brute_force` | Anomaly | Low | Critical (75–100) |
| 2 | Impossible Travel | `impossible_travel` | Anomaly | Low | Critical (75–100) |
| 3 | Credential Stuffing | `credential_stuffing` | Anomaly | Medium | High–Critical (50–100) |
| 4 | Lateral Movement | `lateral_movement` | Anomaly | Medium | High (50–74) |
| 5 | Device Spoofing | `device_spoofing` | Anomaly | Medium | High (50–74) |
| 6 | Low-and-Slow Exfiltration | `low_and_slow` | Anomaly | High | Medium–High (40–74) |
| 7 | Insider Drift | `insider_drift` | **Edge Case** | **Very High** | **Medium (25–49)** |

---

## 0. Normal Baseline

| Attribute | Detail |
|-----------|--------|
| **Simulation Approach** | Events sampled from parameterized entity-type behavioral profiles (user, service_account, edge_device). Each entity has a stable active-hour window, fixed geo-location set, authorized resource pool, and registered device set. Three tiers of realistic noise are layered: field-level Gaussian jitter, per-event rare-deviation probability, and slow temporal behavioral shift. |
| **Key Schema Fields** | All 15 fields; no field is structurally anomalous. |
| **Signal Type** | Benign |
| **Primary Detection Signal** | None (defines the normality baseline for BPM and SDM training). |
| **BPM Expected Score** | 0.0 – 0.30 |
| **SDM Expected Score** | 0.0 – 0.25 |
| **Expected Fused Score** | 0.0 – 0.28 |
| **Risk Tier** | N/A (no alert generated) |
| **Detection Difficulty** | N/A |
| **Notes** | Tier 2 noise (1% foreign IP, 3% rare resource access) will occasionally push individual normal events into the 0.20–0.40 score range. Threshold calibration must account for this noise floor to achieve acceptable false positive rates. |

---

## 1. Brute Force

| Attribute | Detail |
|-----------|--------|
| **Simulation Approach** | Inject N=10–50 rapid authentication failure events against a single target entity from a single source IP, spaced 1–8 seconds apart. With probability 0.4, follow with a successful compromise event. Injection occurs during the entity's active hours to blend with legitimate traffic. |
| **Key Schema Fields** | `auth_outcome` (failure), `failure_count` (1–50 ramping), `source_ip` (fixed foreign IP), `timestamp` (burst spacing), `session_duration` (0.0 for failed events), `label` |
| **Signal Type** | Anomaly |
| **Primary Detection Signal (BPM)** | `failure_count_norm` (dim 5) → spikes to 0.5–1.0; `auth_outcome_enc` (dim 11) → 1.0; `session_event_count_norm` (dim 20) → elevated by burst |
| **Primary Detection Signal (SDM)** | Sequence of repeated failure vectors in the window; `failure_count_norm` monotonically increasing across sequence positions |
| **BPM Expected Score** | 0.75 – 0.98 |
| **SDM Expected Score** | 0.70 – 0.95 |
| **Expected Fused Score** | 0.72 – 0.97 |
| **Risk Tier** | Critical (75–100) |
| **Detection Difficulty** | **Low.** The burst of high `failure_count` events is structurally distinct from normal noise (max normal `failure_count` = 1–3 with 0.01 probability per event). No realistic normal behavior produces 10+ consecutive failures from a single IP. |
| **Distinguishability Basis** | `failure_count` burst + single source IP + rapid inter-event timing (1–8 sec) — all three must coincide. Single-feature signal: `failure_count_norm` alone has near-zero false positive rate. |

---

## 2. Impossible Travel

| Attribute | Detail |
|-----------|--------|
| **Simulation Approach** | Insert a successful authentication event for a target entity from a geographically remote location (>500 km, producing a geo-velocity >800 km/h) within 5–30 minutes of a legitimate event in the entity's home location. All other fields (resource, command, device) match the entity's normal profile — only geo-location and source IP are anomalous. |
| **Key Schema Fields** | `geo_location` (city, country, latitude, longitude — remote location), `source_ip` (foreign country IP), `timestamp` (within 5–30 min of anchor event), `label` |
| **Signal Type** | Anomaly |
| **Primary Detection Signal (BPM)** | `geo_velocity_kmph` (dim 6) → 0.4–1.0; `is_new_geo` (dim 7) → 1.0; compound signal strongly anomalous |
| **Primary Detection Signal (SDM)** | `is_new_geo` and `geo_velocity_kmph` appear as outliers in the sequence window compared to the entity's consistently near-zero geo-velocity history |
| **BPM Expected Score** | 0.80 – 0.99 |
| **SDM Expected Score** | 0.75 – 0.95 |
| **Expected Fused Score** | 0.77 – 0.97 |
| **Risk Tier** | Critical (75–100) |
| **Detection Difficulty** | **Low.** Geo-velocity is a deterministic, physical constraint. Any event with `geo_velocity_kmph > 800` is physically impossible without supersonic transport, making this the most unambiguous signal in the taxonomy. |
| **Distinguishability Basis** | `geo_velocity_kmph` (dim 6) alone provides a near-perfect decision rule. Tier 2 noise (2% foreign IP) is constrained to stay within 500 km of a home city, never producing velocity > 200 km/h. No threshold calibration challenge. |

---

## 3. Credential Stuffing

| Attribute | Detail |
|-----------|--------|
| **Simulation Approach** | Inject a campaign of 1–5 authentication failure events against each of 15–60 target entities, all originating from the same single source IP, interleaved within a 30-minute window. With probability 0.10 per entity, inject a follow-on successful compromise. Unlike brute force, failures are spread thin across many entities rather than concentrated on one. |
| **Key Schema Fields** | `source_ip` (single foreign IP across all entities), `auth_outcome` (failure), `failure_count` (1–5 per entity), `entity_id` (multiple targets), `timestamp` (interleaved across entities), `label` |
| **Signal Type** | Anomaly |
| **Primary Detection Signal (BPM)** | `failure_count_norm` (dim 5) → moderate (0.05–0.25, lower than brute force); `auth_outcome_enc` (dim 11) → 1.0; `ip_entity_ratio` (dim 22) → 1.0 (single IP, many entities) |
| **Primary Detection Signal (SDM)** | Per-entity: low anomaly from sequence perspective. **Stronger signal: population-level** — `ip_entity_ratio` will be elevated for all targeted entities simultaneously. The API orchestration layer can optionally implement a population-level correlation check (not part of T1, but the `ip_entity_ratio` feature is in the feature vector). |
| **BPM Expected Score** | 0.55 – 0.85 |
| **SDM Expected Score** | 0.30 – 0.60 (lower than BPM; per-entity sequence is not strongly anomalous) |
| **Expected Fused Score** | 0.42 – 0.72 |
| **Risk Tier** | High–Critical (50–100) |
| **Detection Difficulty** | **Medium.** Per-entity, the signal is moderate (few failures per entity). The strong signal is cross-entity (single IP, many entities) captured by `ip_entity_ratio`. A BPM-only system with per-entity scoping will underperform; the feature vector's cross-entity IP ratio features are the discriminating factor. |
| **Distinguishability Basis** | `ip_entity_ratio` (dim 22) is the primary discriminator from brute force. Brute force: high `failure_count` per entity, single entity per IP. Credential stuffing: low `failure_count` per entity, many entities per IP. |

---

## 4. Lateral Movement

| Attribute | Detail |
|-----------|--------|
| **Simulation Approach** | In a single compromised session, inject 10–30 events accessing resources entirely outside the entity's normal `ResourceSet`, spanning ≥3 distinct resource categories (`file/`, `api/`, `port/`, `db/`). Include a command sequence of 8–20 commands with reconnaissance-to-exfil progression (`ls` → `find` → `ssh` → `scp`). All events are fully authenticated, use the entity's registered device and home geo-location. |
| **Key Schema Fields** | `resource_accessed` (outside ResourceSet, multi-category), `command_sequence` (recon-to-exfil progression with exfil commands), `session_duration` (900–7200 sec, elevated), `label` |
| **Signal Type** | Anomaly |
| **Primary Detection Signal (BPM)** | `resource_rarity_score` (dim 9) → 0.8–1.0 (all resources novel); `resource_breadth_norm` (dim 21) → 0.20–0.60; `command_seq_length_norm` (dim 12) → elevated; `has_exfil_command` (dim 14) → 1.0 |
| **Primary Detection Signal (SDM)** | Sequence window shows abrupt transition from entity's normal resource access pattern to a diverse, multi-category burst. Command sequence pattern (recon → access → exfil) is the primary SDM signature. |
| **BPM Expected Score** | 0.65 – 0.90 |
| **SDM Expected Score** | 0.60 – 0.88 |
| **Expected Fused Score** | 0.62 – 0.89 |
| **Risk Tier** | High (50–74) |
| **Detection Difficulty** | **Medium.** The resource rarity and exfil command signals are strong. However, `developer` persona entities have a wider legitimate resource footprint and occasionally issue network commands, which narrows the margin. The multi-category breadth criterion (≥3 categories) is what separates lateral movement from a developer doing legitimate cross-service work. |
| **Distinguishability Basis** | `resource_rarity_score` + `resource_breadth_norm` + `has_exfil_command` + `command_seq_length_norm` acting jointly. Lateral movement vs. Insider Drift: lateral movement is session-level (1 session, many resources) and includes exfil commands; insider drift is multi-week and never includes exfil commands. |

---

## 5. Device Spoofing

| Attribute | Detail |
|-----------|--------|
| **Simulation Approach** | Insert 1–5 events where the entity's registered `device_id` appears with a modified fingerprint: either (A) changed MAC address only, (B) changed OS family + version + user_agent, or (C) changed protocol (especially for edge devices). All other event fields match the entity's normal profile. The attacker is using the entity's stolen credentials and session token from a different device. |
| **Key Schema Fields** | `device_fingerprint.device_id` (same as registered), `device_fingerprint.mac_address` (changed — strategy A), `device_fingerprint.os_family`/`os_version` (changed — strategy B), `device_fingerprint.protocol` (changed — strategy C), `label` |
| **Signal Type** | Anomaly |
| **Primary Detection Signal (BPM)** | `fingerprint_mac_match` (dim 16) → 0.0 (strategy A/B); `fingerprint_os_match` (dim 15) → 0.0 (strategy B); `fingerprint_protocol_match` (dim 17) → 0.0 (strategy C) |
| **Primary Detection Signal (SDM)** | Sequence window shows transition from all-1.0 fingerprint match dimensions to all-0.0, an abrupt discrete change that the SDM learns as anomalous |
| **BPM Expected Score** | 0.60 – 0.88 |
| **SDM Expected Score** | 0.55 – 0.85 |
| **Expected Fused Score** | 0.57 – 0.86 |
| **Risk Tier** | High (50–74) |
| **Detection Difficulty** | **Medium.** The fingerprint match features (dims 15–17) are binary; a mismatch is unambiguous for a known device. However, a user with 2 registered devices will regularly switch between them (device 2 with probability 0.15 per event), which means `fingerprint_mac_match = 0.0` occurs legitimately for dual-device users. The key discriminator is that device spoofing uses an **unknown** MAC/OS combination with a **known** `device_id` — the `device_id` is the same but the fingerprint changed. |
| **Distinguishability Basis** | `fingerprint_mac_match = 0.0` for a `device_id` that has been seen before with a different MAC. Tier 2 unknown-device noise generates a new `device_id`, which Feature Engineering treats as a new unknown device (all fingerprint match dimensions = 0.0, but the cause is different). |

---

## 6. Low-and-Slow Exfiltration

| Attribute | Detail |
|-----------|--------|
| **Simulation Approach** | Inject 1–3 events per day over 5–20 days, each in the off-hours window (01:00–04:30 UTC), each accessing a resource within the entity's authorized `ResourceSet`, each including an exfil command (`scp`, `rsync`, `wget`, or `curl` with an external target). The campaign is designed so that any single event is individually near-normal: off-hours access is in the entity's 3% Tier 2 noise baseline. Only the cumulative multi-day pattern reveals the attack. |
| **Key Schema Fields** | `timestamp` (off-hours, recurring), `command_sequence` (exfil command present), `session_duration` (elevated but not extreme), `inter_event_gap_norm` (derived — long gaps between sessions), `resource_accessed` (within authorized set), `label` |
| **Signal Type** | Anomaly |
| **Primary Detection Signal (BPM)** | `hour_of_day_sin/cos` (dims 0–1) deviation from entity's normal active-hour baseline; `has_exfil_command` (dim 14) → 1.0; `inter_event_gap_norm` (dim 19) → elevated (long quiet periods between access events) |
| **Primary Detection Signal (SDM)** | **The SDM is the primary detector.** The sequence window across the 30-day simulation period shows a recurring pattern of off-hours events with exfil commands, interleaved with long inter-event gaps. The BPM scores individual events as near-normal; the SDM detects the temporal pattern. This is the primary design justification for including the SDM in the architecture. |
| **BPM Expected Score** | 0.30 – 0.55 (near-normal individual events) |
| **SDM Expected Score** | 0.55 – 0.82 (pattern-level detection) |
| **Expected Fused Score** | 0.42 – 0.68 |
| **Risk Tier** | Medium–High (40–74) |
| **Detection Difficulty** | **High.** The hardest attack for a BPM-only system. The deliberate design (authorized resources, home geo, registered device, no IP anomaly) removes all fast signals. Only `has_exfil_command` and off-hours timing are individually anomalous, but at 3% normal off-hours noise, this alone has a high false positive rate. The SDM's sequence-level detection of the multi-day recurring pattern is essential. |
| **Distinguishability Basis** | `has_exfil_command = 1.0` (dims 14) — present in every low-and-slow event. Combined with `hour_of_day` deviation and multi-day sequence pattern in the SDM window. Vs. lateral movement: low-and-slow uses authorized resources (low `resource_rarity_score`) and occurs over days (long `inter_event_gap`); lateral movement uses unauthorized resources in a single session. |

---

## 7. Insider Drift (Edge Case)

| Attribute | Detail |
|-----------|--------|
| **Simulation Approach** | Beginning 5–20 days into the simulation, gradually introduce 1–3 new resource accesses per week from outside the entity's `ResourceSet` but within the same resource category family. All events occur during the entity's **normal active hours**, using **authorized access** (full authentication success), the **registered device**, and the **home geo-location**. No exfil commands. The expansion pace mimics legitimate role evolution (promotion, new project assignment). After 2 weeks, some drift resources begin appearing multiple times per week (resource rarity score naturally decays), further blurring the anomaly signal. |
| **Key Schema Fields** | `resource_accessed` (gradually expanding to outside ResourceSet), `timestamp` (normal business hours — **not** off-hours), `auth_outcome` (success — **not** failure), `command_sequence` (normal for entity persona — **no** exfil commands), `device_fingerprint` (registered device — **no** fingerprint change), `geo_location` (home location — **no** geo anomaly), `label` |
| **Signal Type** | **Edge Case** — Genuinely ambiguous. Not all insider drift is malicious. |
| **Primary Detection Signal (BPM)** | `resource_rarity_score` (dim 9) → gradually elevated (0.4–0.7); `resource_breadth_norm` (dim 21) → slowly increasing over weeks; `command_rarity_score` (dim 13) → slightly elevated if new resource categories require new command types |
| **Primary Detection Signal (SDM)** | Slow monotonic increase in `resource_rarity_score` and `resource_breadth_norm` across the sequence window over the 30-day period. No abrupt transitions (unlike lateral movement). The SDM must distinguish slow-gradient expansion from random walk noise. |
| **BPM Expected Score** | 0.30 – 0.55 (individual events near-normal; gradually increasing over time) |
| **SDM Expected Score** | 0.25 – 0.50 (pattern is gradual; window shows slow gradient, not abrupt shift) |
| **Expected Fused Score** | 0.28 – 0.52 (near the 0.5 threshold — may or may not trigger `is_anomaly = True`) |
| **Risk Tier** | **Medium (25–49)** — "Watch and investigate" classification |
| **Detection Difficulty** | **Very High.** All five major fast signals are absent: no off-hours timing, no authentication failure, no device change, no geo anomaly, no exfil commands. The only signals are the slow drift in `resource_rarity_score` and `resource_breadth_norm` — both of which are also produced by Tier 3 normal noise (legitimate role expansion in 5% of entities). |
| **Distinguishability Basis** | Insider drift is **not fully distinguishable** from legitimate role evolution using event-level data alone. The model is expected to produce `classification_confidence` of 0.40–0.60 and `risk_tier = medium`. A well-calibrated system will flag insider drift as "needs investigation" rather than "confirmed threat." This is the correct behavior: human analyst review is the appropriate response. |
| **Ambiguity Design (Five Overlapping Properties)** | (1) Normal business hours → no timing anomaly; (2) Authorized access → no auth anomaly; (3) No exfil commands → no exfil signal; (4) Registered device → no device anomaly; (5) Home geo-location → no travel anomaly. All five signals are designed to be absent simultaneously. |
| **Expected Classifier Output** | `predicted_class = "insider_drift"` with probability 0.40–0.60; `predicted_class = "lateral_movement"` with probability 0.20–0.35 (resource expansion is the shared feature); `predicted_class = "normal"` with probability 0.10–0.25. |
| **False Positive Implication** | A system that flags insider drift as `high` risk (score 50–74) or `critical` (75–100) has been overfit or miscalibrated. The expected production behavior is medium-tier alert with `cold_start_flag = False` (the entity is well-profiled) and `classification_confidence < 0.65`. |

---

## Detection Difficulty Matrix

| Attack | BPM Can Detect Alone | SDM Required | Cross-Entity Signal | Primary Feature Dimensions |
|--------|---------------------|-------------|---------------------|---------------------------|
| Normal | N/A | N/A | N/A | — |
| Brute Force | ✅ Strong | Helpful | No | dims 5, 11, 20 |
| Impossible Travel | ✅ Strong | Helpful | No | dims 6, 7 |
| Credential Stuffing | ⚠️ Moderate | Helpful | **Yes** | dims 5, 11, 22 |
| Lateral Movement | ✅ Moderate | Strengthens | No | dims 9, 12, 14, 21 |
| Device Spoofing | ✅ Moderate | Strengthens | No | dims 15, 16, 17 |
| Low-and-Slow | ❌ Weak (individual events) | **Essential** | No | dims 0–3, 14, 19 |
| Insider Drift | ❌ Very Weak | **Essential** | No | dims 9, 13, 21 |

---

## Feature Dimension Quick Reference

| Dim | Name | Primary Attack Signal |
|-----|------|-----------------------|
| 0–1 | `hour_of_day_sin/cos` | Off-hours access (Low-and-Slow) |
| 2–3 | `day_of_week_sin/cos` | Weekend access baseline |
| 4 | `session_duration_norm` | Long sessions (Low-and-Slow) |
| 5 | `failure_count_norm` | Brute Force, Credential Stuffing |
| 6 | `geo_velocity_kmph` | Impossible Travel |
| 7 | `is_new_geo` | Impossible Travel |
| 8 | `resource_category_enc` | Lateral Movement (category shift) |
| 9 | `resource_rarity_score` | Lateral Movement, Insider Drift |
| 10 | `auth_method_enc` | Credential misuse |
| 11 | `auth_outcome_enc` | Brute Force, Credential Stuffing |
| 12 | `command_seq_length_norm` | Lateral Movement |
| 13 | `command_rarity_score` | Insider Drift, Lateral Movement |
| 14 | `has_exfil_command` | Low-and-Slow, Lateral Movement |
| 15 | `fingerprint_os_match` | Device Spoofing |
| 16 | `fingerprint_mac_match` | Device Spoofing (primary) |
| 17 | `fingerprint_protocol_match` | Device Spoofing (edge devices) |
| 18 | `entity_type_enc` | Peer group routing |
| 19 | `inter_event_gap_norm` | Low-and-Slow (multi-day gaps) |
| 20 | `session_event_count_norm` | Brute Force (burst depth) |
| 21 | `resource_breadth_norm` | Lateral Movement, Insider Drift |
| 22 | `ip_entity_ratio` | Credential Stuffing (cross-entity) |
| 23 | `entity_ip_ratio` | Compromised credentials (multi-IP) |

---

*End of ATTACK_TAXONOMY.md — Phase 5 output (Document 2 of 2). This document is frozen. It is ready for direct inclusion in `docs/report/TECHNICAL_REPORT.md` and the hackathon presentation.*
