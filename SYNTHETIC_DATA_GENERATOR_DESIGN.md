# SYNTHETIC_DATA_GENERATOR_DESIGN.md
# AI-Powered Behavioral Anomaly Detection — Synthetic Data Generator Design

> **Status:** Phase 5 — Generator Design Reference  
> **Version:** 1.0  
> **Last Updated:** 2026-07-25  
> **Changelog:** 2026-07-25 — Added Late Joiner cold-start test set generation requirement per CONSISTENCY_REVIEW.md Blocking Issue #1.

---

## Table of Contents

1. [Architecture Consistency Check](#1-architecture-consistency-check)
2. [Per-Entity Behavioral Profile Model](#2-per-entity-behavioral-profile-model)
   - 2a. [Profile Architecture Overview](#2a-profile-architecture-overview)
   - 2b. [User Profile](#2b-user-profile)
   - 2c. [Service Account Profile](#2c-service-account-profile)
   - 2d. [Edge Device Profile](#2d-edge-device-profile)
   - 2e. [Noise Layering Strategy](#2e-noise-layering-strategy)
   - 2f. [Cold-Start Holdout Group (Late Joiners)](#2f-cold-start-holdout-group-late-joiners)
3. [Attack Taxonomy with Simulation Algorithms](#3-attack-taxonomy-with-simulation-algorithms)
   - 3.1 [Brute Force](#31-brute-force)
   - 3.2 [Impossible Travel](#32-impossible-travel)
   - 3.3 [Credential Stuffing](#33-credential-stuffing)
   - 3.4 [Lateral Movement](#34-lateral-movement)
   - 3.5 [Device Spoofing](#35-device-spoofing)
   - 3.6 [Low-and-Slow Exfiltration](#36-low-and-slow-exfiltration)
   - 3.7 [Insider Drift (Edge Case)](#37-insider-drift-edge-case)
4. [Configuration Surface](#4-configuration-surface)
5. [Alternatives Considered](#5-alternatives-considered)
6. [Judging-Criteria Traceability](#6-judging-criteria-traceability)

---

## 1. Architecture Consistency Check

DATA_SCHEMA.md v1.0 was re-read in full before beginning this design. The following cross-checks between the generator design and the Training Schema are verified:

| Check | Result | Note |
|-------|--------|------|
| Generator produces all 15 fields of `RawAccessLog` (including `event_id`, `session_id`, `auth_outcome`, `failure_count`) | ✅ Pass | All 15 schema fields are explicitly targeted in Section 2 profile definitions and Section 3 attack algorithms |
| `command_sequence` is a `list[CommandEntry]` with sub-fields: `sequence_position`, `command`, `target`, `outcome`, `elapsed_seconds` | ✅ Pass | All 5 `CommandEntry` sub-fields used in Sections 3.4 (lateral movement) and 3.6 (low-and-slow) |
| `device_fingerprint` is a struct with: `device_id`, `os_family`, `os_version`, `mac_address`, `protocol`, `user_agent`, `firmware_version` | ✅ Pass | All 7 sub-fields used in Section 2 profiles and Section 3.5 (device spoofing) |
| `geo_location` struct has: `city`, `country`, `latitude`, `longitude` | ✅ Pass | All 4 sub-fields used in Sections 2b (user profile) and 3.2 (impossible travel) |
| Label taxonomy matches DATA_SCHEMA.md: `normal`, `brute_force`, `impossible_travel`, `credential_stuffing`, `lateral_movement`, `device_spoofing`, `low_and_slow`, `insider_drift` | ✅ Pass | All 8 values are used; `normal` is the base-rate label for all non-attacked events |
| `label` field exists only in Training Schema; generator writes labels to `data/labeled/labels_<run_id>.parquet` via `label_store.py` | ✅ Pass | Label separation mechanism is part of the configuration surface in Section 4 |
| `entity_type` enum values: `user`, `service_account`, `edge_device` | ✅ Pass | Three profile types defined in Section 2 |
| `auth_method` enum values: `password`, `token`, `certificate`, `biometric`, `none` | ✅ Pass | Assigned per entity type in Section 2 |
| `auth_outcome` enum values: `success`, `failure`, `mfa_required` | ✅ Pass | Used in normal profiles and in brute force / credential stuffing injection |
| No new schema fields are required by any attack design in Section 3 | ✅ Pass | All attacks use only the 15 existing fields and their sub-structures |

**Consistency verdict:** The generator design is fully compatible with DATA_SCHEMA.md v1.0's Training Schema. No proposed schema changes are required.

---

## 2. Per-Entity Behavioral Profile Model

### 2a. Profile Architecture Overview

Each entity in the synthetic dataset is assigned a **static behavioral profile** at generation time. This profile is a parameter set, not data — it is the source of truth for what "normal" looks like for that entity. The generator then samples events from each profile's distributions to produce the baseline dataset.

**Profile hierarchy:**

```
EntityProfile (static, per-entity, set at generation time)
  └── EntityType    → selects the base distribution family (Section 2b/2c/2d)
  └── PersonaVariant → one of 3–5 named sub-types within each entity type
                       (e.g., "executive", "developer", "analyst" for users)
  └── HomeGeoSet   → 1–3 named city/country pairs the entity legitimately uses
  └── ResourceSet  → the entity's legitimate resource access pool (size and category)
  └── DeviceSet    → 1–2 registered device fingerprints
  └── WorkSchedule → active hours distribution parameters
```

**Three entity types and their approximate population shares (configurable):**

| Entity Type | Default Population Share | Behavioral Complexity |
|-------------|--------------------------|----------------------|
| `user` | 70% | High (temporal, geo, command variation) |
| `service_account` | 20% | Low (highly regular, narrow resource set) |
| `edge_device` | 10% | Very low (near-deterministic, single-protocol) |

**Events per entity (default):** Sample from a Poisson distribution with λ = 120 events per entity per 30-day simulation window. This gives approximately 60–200 events per entity, providing sufficient sequence history for model training while keeping the dataset tractable.

**Session grouping:** Group consecutive events by the rule: events within 30 minutes of the prior event for the same entity share a `session_id`. A new session begins after a gap > 30 minutes or after an `auth_outcome=failure` event that is not followed by a success (i.e., the session never authenticated).

---

### 2b. User Profile

#### Persona Variants

| Persona | Description | Active Hours Center | Geo Variety | Resource Set Size |
|---------|-------------|--------------------|-----------|--------------------|
| `executive` | C-suite / senior management | Business hours (08:00–18:00 local) | 2–3 cities (travel expected) | Small, high-privilege (20–30 resources) |
| `developer` | Engineering roles | Flexible (07:00–23:00), peak 09:00–17:00 | 1–2 cities | Medium, technical (40–60 resources: `port/`, `api/`, `file/code/`) |
| `analyst` | Data/security analyst | Business hours (08:00–18:00) | 1 city | Medium, data-heavy (40–60 resources: `file/`, `db/`) |
| `support` | Helpdesk / IT operations | Shift-based (three 8-hour shift centers) | 1 city | Medium, broad-but-shallow (50–80 resources across all categories) |
| `remote_worker` | Fully remote employee | Business hours in any single timezone | 1 fixed city, 1 VPN IP pool | Same as `analyst` |

#### Login Timing Distribution

For each user:

1. Sample an **active hour center** μ_h from the persona's active hour range (e.g., for `executive`: Uniform[8, 18]).
2. Sample an **active hour spread** σ_h ~ Uniform[1.5, 3.0] hours.
3. For each event:
   - Sample hour-of-day from N(μ_h, σ_h²), truncated to [0, 24).
   - Apply day-of-week weighting: weight Mon–Fri by 0.9 each, Sat–Sun by (1 − 5×0.9)/2 = 0.025 each (rare weekend access). For `remote_worker`, uniform over all 7 days.
   - Noise: with probability 0.05, sample hour from Uniform[0, 24) unconditionally (models ad-hoc late/early events).

#### Geo-Location Distribution

1. Each user has a `HomeGeoSet` of 1–3 city/country pairs, assigned at entity creation time using Faker.
2. For each event, sample a home city from the `HomeGeoSet` with weights proportional to [0.8, 0.15, 0.05] (for 3-city sets).
3. For the sampled city, set `latitude` and `longitude` by adding Gaussian noise: N(0, 0.05²) degrees to each coordinate (models within-city variation).
4. Noise: with probability 0.02, sample a completely foreign city from a global city pool (models legitimate business travel before it becomes an attack signal).

#### Resource Access Distribution

1. Each user has a `ResourceSet` of size R (from persona table above), created by sampling resource identifiers from a type-appropriate pool. Pools:
   - `file/<department>/`: 200 pre-defined file paths per department (Finance, HR, Engineering, Legal)
   - `api/<service>/`: 100 pre-defined API endpoints
   - `port/<number>`: ports from {22, 80, 443, 3306, 5432, 8080, 8443}
   - `db/<schema>/<table>`: 60 pre-defined database table references
2. Within the `ResourceSet`, assign each resource a **relative access weight** sampled from a Dirichlet distribution with α = 0.5 (produces a heavy-tailed distribution — a few resources are accessed very frequently, most rarely). This models the realistic Pareto distribution of resource access.
3. For each event, sample one resource from the `ResourceSet` according to these weights.
4. Noise: with probability 0.03, sample from outside the entity's `ResourceSet` (models occasional legitimate cross-department access).

#### Authentication

- `auth_method`: Each user is assigned a primary auth method at entity creation (sample: 60% `password`, 25% `token`, 10% `certificate`, 5% `biometric`). With probability 0.05, use a secondary method per event.
- `auth_outcome`: `success` with probability 0.96, `mfa_required` with probability 0.03, `failure` with probability 0.01 (models fat-finger password mistakes). `failure_count`: 0 for success events; sample from Geometric(p=0.7) for failure events (most failed attempts stop after 1–2 tries).

#### Session Duration

Sample from a log-normal distribution: `session_duration ~ exp(N(μ_s, σ_s²))` where:
- μ_s = 7.0 (≈ 1,100 seconds ≈ 18 minutes) for standard sessions
- σ_s = 0.8
- Cap at 28,800 seconds (8 hours) for `session_duration > 28800`
- Set `session_duration = 0.0` for any event where `auth_outcome = failure`

#### Command Sequence

- Probability of a privileged session (non-empty `command_sequence`): 0.30 for `developer`, 0.15 for `analyst`/`support`, 0.05 for `executive`/`remote_worker`.
- For privileged sessions, sample command sequence length from Poisson(λ=5), minimum 1.
- Each `command` is drawn from the entity's **command pool**: a fixed set of 5–15 commands assigned at entity creation time from a vocabulary of {`ls`, `cat`, `grep`, `sudo`, `ssh`, `scp`, `rsync`, `curl`, `wget`, `ps`, `netstat`, `chmod`, `find`, `tar`, `vim`}.
- `target`: for each command, sample a target resource from the entity's `ResourceSet` (or a host for network commands).
- `outcome`: `success` with probability 0.93, `failure` 0.05, `denied` 0.02.
- `elapsed_seconds`: `sequence_position × gamma_draw` where `gamma_draw ~ Gamma(shape=2, scale=15)` seconds per command step.
- `sequence_position`: set to 0-indexed position in the list.

#### Device Fingerprint

Each user has `DeviceSet` of 1–2 devices, created at entity creation time:
- `device_id`: assigned at entity creation (format: `dev_<8hex>`).
- `os_family`: `Windows` (60%), `Linux` (20%), `macOS` (15%), `iOS` (5%).
- `os_version`: sampled per os_family from a small set of 3 realistic versions.
- `mac_address`: Faker-generated, fixed per `device_id`.
- `protocol`: `HTTPS` for most users; `RDP` for `support` persona.
- `user_agent`: Faker-generated Mozilla string, fixed per `device_id`.
- `firmware_version`: empty string for users.
- Per event: sample a device from the `DeviceSet` with weights [0.85, 0.15] (primary and secondary device).

---

### 2c. Service Account Profile

Service accounts model automated processes (CI/CD pipelines, scheduled jobs, API integrations). Their behavior is strongly regular with very low variance.

#### Persona Variants

| Persona | Description | Active Hours | Resource Set | Command Sequences |
|---------|-------------|-------------|-------------|------------------|
| `cicd_pipeline` | CI/CD build and deploy runner | Business hours + occasional nights (deploy events) | 10–20 resources (specific repos, artifact stores, deployment targets) | None (API-based) |
| `monitoring_agent` | System health monitor | 24/7, every N minutes (N ~ Uniform[1, 15]) | 5–10 read-only system resources | None |
| `etl_job` | Batch data extraction | Nightly (00:00–05:00), tight schedule | 5–15 database tables and file paths | None |
| `api_integration` | Third-party API bridge | Business hours | 3–8 external-facing API endpoints | None |

#### Login Timing Distribution

Service accounts use a **near-deterministic** schedule:
- Sample an **interval** I ~ Uniform[60, 900] seconds per persona (fixed per entity).
- Each event timestamp = prior timestamp + I + ε, where ε ~ N(0, 5²) seconds (models clock jitter).
- `etl_job` persona: shift start time to Uniform[00:00, 02:00], interval = Uniform[30, 120] seconds.

#### Geo-Location

Single fixed geo-location per entity (the data center or cloud region). `latitude` / `longitude` noise: N(0, 0.001²) (sub-block variation only).

#### Authentication

- `auth_method`: `token` (70%) or `certificate` (30%).
- `auth_outcome`: `success` with probability 0.995; `failure` 0.005 (models token expiry). `failure_count` = 0 for success events.
- No `mfa_required` for service accounts.

#### Session Duration

Service accounts have very short sessions: `session_duration ~ exp(N(4.0, 0.3²))` ≈ 50–100 seconds. `cicd_pipeline` may have longer deploys: `session_duration ~ exp(N(5.5, 0.4²))` ≈ 200–600 seconds.

#### Command Sequence and Device Fingerprint

- `command_sequence`: empty list `[]` for all service account events (API-based interaction, not shell-based).
- `device_fingerprint`: fixed per entity at creation; `os_family = Linux`, `protocol` per persona (`HTTPS` for `cicd_pipeline`/`api_integration`, `Modbus` for `monitoring_agent` in OT contexts), `firmware_version = ""`.

---

### 2d. Edge Device Profile

Edge devices model IoT and OT assets (sensors, PLCs, RTUs, industrial controllers). They are the most regular of the three entity types.

#### Persona Variants

| Persona | Description | Protocol | Active Hours | Resource Set |
|---------|-------------|---------|-------------|-------------|
| `iot_sensor` | Environmental / telemetry sensor | `MQTT` | 24/7, every 30–60 seconds | 1–2 ingest endpoints |
| `plc_controller` | Industrial PLC sending status | `Modbus` | 24/7, every 1–10 seconds | 1 SCADA endpoint |
| `security_camera` | Video stream / alert sender | `RTSP`/`HTTPS` | 24/7, every 30 seconds | 1 recording server |
| `rtu_device` | Remote Terminal Unit | `DNP3` | 24/7, every 5–30 seconds | 1 control system endpoint |

#### Login Timing Distribution

Nearly deterministic: `interval = fixed_polling_interval + ε` where ε ~ N(0, 1²) seconds. Jitter models physical-layer variation.

#### Geo-Location

Single fixed geo-location, no variance (device is physically fixed). Noise: ε = 0.

#### Authentication

- `auth_method`: `certificate` (80%), `none` (20% for low-security IoT).
- `auth_outcome`: `success` with probability 0.999; `failure` 0.001.
- `failure_count` = 0 always in the normal profile.

#### Session Duration

Extremely short: `session_duration ~ exp(N(2.0, 0.2²))` ≈ 5–15 seconds (models a single telemetry push).

#### Command Sequence

Empty list `[]`. Edge devices do not issue shell commands.

#### Device Fingerprint

Fully fixed per entity:
- `os_family`: `Embedded/RTU`
- `os_version`: specific firmware version string (e.g., `"FW-2.3.1"`)
- `mac_address`: fixed
- `protocol`: per persona
- `user_agent`: empty string
- `firmware_version`: a specific version string (e.g., `"2.3.1"`)

---

### 2e. Noise Layering Strategy

Noise is layered on top of base profiles in three tiers, applied in order during event generation:

#### Tier 1: Field-Level Gaussian Noise (Every Event)

Applied to every event independently:
- Timestamp: add ε ~ N(0, 30²) seconds (models clock drift, network latency in log ingestion).
- Session duration: multiply by exp(N(0, 0.1²)) (lognormal multiplicative noise).
- Geo coordinates: add ε ~ N(0, 0.05²) degrees to lat/lon.

#### Tier 2: Rare One-Off Deviations (Per-Event Probability)

Applied with low probability to individual events:
- **Foreign IP:** With probability 0.01, replace `source_ip` with a random IPv4 address outside the entity's normal subnet. This models a VPN endpoint, corporate travel network, or hotel Wi-Fi.
- **Off-hours access:** With probability 0.03, override the sampled event timestamp to a random hour outside the entity's active window. This models an infrequent genuine late-night access.
- **Rare resource:** With probability 0.03, access a resource from outside the entity's `ResourceSet` (legitimate cross-department one-off). This is the primary source of moderate `resource_rarity_score` values in normal events.
- **Secondary device:** For users with a `DeviceSet` of size 2, the secondary device is used with probability 0.15 (as defined in Section 2b). Noise: with probability 0.005, generate a new unknown device not in the entity's `DeviceSet` (models a shared corporate laptop — *this is legitimate, not spoofing*).

#### Tier 3: Temporal Behavioral Shift (Per-Entity, Slow)

Applied at entity creation time for a subset of entities:
- **Gradual schedule shift:** 10% of user entities are assigned a **slow drift rate** ρ_h ~ Uniform[0.1, 0.5] hours per 7 days. Over the 30-day simulation window, their `active hour center` μ_h shifts by at most ρ_h × 4 ≈ 0.4–2.0 hours. This models a changing work schedule and is **not labeled as an attack** — it is normal behavioral evolution.
- **Role expansion:** 5% of user entities gradually expand their `ResourceSet` by adding 1–3 new resources per 10 days. Again, **not labeled as an attack** unless the expansion rate is combined with the attack-level parameters defined in Section 3.7 (Insider Drift).

This noise design is critical: Tier 3 normal noise overlaps with early-stage Insider Drift, which is what makes Insider Drift genuinely ambiguous (see Section 3.7).

---

### 2f. Cold-Start Holdout Group (Late Joiners)

To satisfy the evaluation protocols defined in EVAL_METRICS.md, exactly 5% of all generated entities must be designated as 'Late Joiners'. For these entities, the generator must enforce that no events are emitted between Days 1–21 (the training window). Their Poisson sampling timeline must be restricted exclusively to Days 26–30, ensuring they trigger the cold-start fallback logic during testing.

- Exactly 5% of all generated entities are designated "Late Joiners"
- Late Joiner events begin exclusively on Day 26 (the first day of the evaluation window), with zero events in Days 1-25 (the training window)
- Late Joiners are distributed across all three entity_types (user, service_account, edge_device) proportionally
- Late Joiners receive the same attack injection rate as normal entities (0.5%-3% of their sessions), so cold-start evaluation includes both normal and anomalous cold-start scoring
- The Late Joiner flag is tracked internally for evaluation purposes (`cold_start_flag: true` in the training schema) but is not exposed at inference time
- Random seed handling applies equally to Late Joiner assignment, ensuring reproducibility

---

## 3. Attack Taxonomy with Simulation Algorithms

### Global Injection Framework

Attacks are injected by the `attack_injector.py` module after the base normal dataset has been generated for all entities. The injection framework works as follows:

1. The base dataset is generated completely first (all normal events, all entities).
2. For each attack type, a subset of entities is selected as **attack targets** based on the injection rate and entity type constraints.
3. Attack events are inserted into the selected entity's timeline by either: (a) modifying existing events, or (b) inserting new synthetic events at specific timeline positions.
4. All modified or inserted events receive the appropriate non-`normal` label in `label_store.py`.
5. Surrounding normal events that are part of the same session as the attack are also relabeled if they are contextually required to complete the attack signature.

**Injection rate:** Configurable as a percentage of total events. Default: 1.5% of all events labeled as non-normal. Minimum: 0.5%. Maximum: 3.0%.

---

### 3.1 Brute Force

**Label:** `brute_force`  
**Target entity types:** `user` (80%), `service_account` (20%)  
**Typical signal:** Rapid successive authentication failures from a single IP against a single entity.

#### Trigger / Injection Logic

1. Select a target entity from the user/service_account pool. The entity must have at least 20 prior normal events (ensures a well-formed profile exists).
2. Select an **injection timestamp** T_inject: sample from the entity's normal active hours (brute force during active hours is more realistic than purely off-hours, since attackers time attacks to blend with legitimate traffic peaks).
3. Insert N_fail events immediately before T_inject where:
   - N_fail ~ Uniform[10, 50] (number of failed attempts)
   - Each event uses the same `source_ip` (different from the entity's normal IP pool)
   - `auth_outcome = failure` for all N_fail events
   - `auth_method = password` for all N_fail events
   - `failure_count` = running count (1, 2, 3, ..., N_fail)
   - `session_duration = 0.0` for all N_fail events (no session established)
   - `command_sequence = []`
   - `timestamp`: space events T_fail seconds apart where T_fail ~ Uniform[1, 8] seconds (rapid fire)
   - `resource_accessed`: same resource for all events (the login endpoint)
   - Labels: all N_fail events labeled `brute_force`
4. Optionally (with probability 0.4), append a **successful access** event immediately after the N_fail events:
   - `auth_outcome = success`, `failure_count = 0`
   - `session_duration` sampled from the entity's normal distribution
   - This event is labeled `brute_force` (the successful compromise is part of the attack)

#### Controlling Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `bf_n_fail_min` | 10 | [5, 20] | Minimum failure events; lower = harder to detect |
| `bf_n_fail_max` | 50 | [20, 100] | Maximum failure events |
| `bf_inter_event_sec_min` | 1 | [0.5, 5] | Minimum seconds between failures |
| `bf_inter_event_sec_max` | 8 | [2, 30] | Maximum seconds between failures |
| `bf_success_probability` | 0.4 | [0, 1] | Probability of a successful compromise following the failures |
| `bf_inject_during_active_hours` | True | bool | If False, inject at random hour (easier to detect) |

#### Distinguishability from Noisy Normal

Normal behavior includes `failure_count = 1–3` from Tier 1 noise with probability 0.01 per event. A brute force burst has:
- N_fail ≥ 10 failures in a single burst (no normal entity produces this)
- All failures from the **same source IP**, which differs from the entity's normal IP set
- Inter-event gap of 1–8 seconds (normal failures are isolated events, not burst sequences)

The critical distinguishing feature is the **combination** of high failure count + single source IP + burst timing. A normal entity might occasionally have 1–2 failures from an unfamiliar IP (Tier 2 noise), but never 10+ consecutive failures from a single non-normal IP.

#### Schema Fields Primarily Manipulated

`auth_outcome`, `failure_count`, `source_ip`, `timestamp` (spacing), `session_duration`, `label`

---

### 3.2 Impossible Travel

**Label:** `impossible_travel`  
**Target entity types:** `user` only (service accounts and edge devices have fixed geo-locations)  
**Typical signal:** Two authenticated events for the same entity from geographically distant locations, separated by a time interval physically impossible to traverse.

#### Trigger / Injection Logic

1. Select a target user entity. Select an existing **anchor event** E_anchor from the entity's normal timeline. E_anchor must have `auth_outcome = success`.
2. Select a **displacement time** Δt ~ Uniform[5, 30] minutes (how long after the anchor event the impossible location appears).
3. Select a **remote location** L_remote: a city/country pair from a pool of locations guaranteed to be > 500 km from E_anchor's location. Compute the Haversine distance D km between L_remote and E_anchor's `geo_location`. Ensure that D / (Δt / 60) > 800 km/h (physically impossible without aircraft, and even then, > 900 km/h is impossible for commercial aircraft with boarding time).
4. Insert an **impossible event** E_impossible at timestamp E_anchor.timestamp + Δt:
   - `geo_location` = L_remote (city, country, lat/lon)
   - `source_ip`: a new IP consistent with L_remote's country (Faker-generated in the L_remote country's IP range)
   - `auth_outcome = success`, `failure_count = 0`
   - `auth_method`: same as the entity's normal auth method
   - `session_duration`: sampled from entity's normal distribution
   - `resource_accessed`: sampled from entity's normal `ResourceSet`
   - `command_sequence`: sampled from entity's normal command behavior
   - `device_fingerprint`: same device as E_anchor (the attacker has the entity's device — or equivalently, the session token is stolen)
   - Label: `impossible_travel`

#### Controlling Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `it_delta_t_min_min` | 5 | [2, 15] | Minimum minutes between anchor and impossible event |
| `it_delta_t_min_max` | 30 | [15, 60] | Maximum minutes |
| `it_min_distance_km` | 500 | [200, 2000] | Minimum geographic distance |
| `it_min_velocity_kmph` | 800 | [400, 3000] | Minimum velocity to qualify as impossible |
| `it_use_same_device` | True | bool | If False, use a different device (adds device change signal) |

#### Distinguishability from Noisy Normal

Tier 2 noise allows a foreign IP with probability 0.01 per event. However:
- Tier 2 foreign IPs are still from the entity's normal time zone range and do not produce geo-velocities > 800 km/h
- The generator enforces that all Tier 2 noise events are geographically consistent with the entity's travel profile (within 500 km of a home city)
- The impossible travel event is guaranteed to produce `geo_velocity_kmph` (feature dim 6) > 0.4 (normalized; corresponds to 800 km/h)

Normal behavioral evolution (the gradual schedule shift in Tier 3) never changes `geo_location` — it only changes login timing. This keeps impossible travel unambiguous.

#### Schema Fields Primarily Manipulated

`geo_location` (city, country, latitude, longitude), `source_ip`, `timestamp`, `label`

---

### 3.3 Credential Stuffing

**Label:** `credential_stuffing`  
**Target entity types:** Multiple user entities simultaneously (this is a **population-level** attack)  
**Typical signal:** A single IP produces authentication failures against many distinct entity IDs in a short window. Unlike brute force (many failures against one entity), credential stuffing spreads failures across many entities.

#### Trigger / Injection Logic

Credential stuffing is injected as a **campaign** targeting N_entities entities:

1. Select N_entities ~ Uniform[15, 60] user entities from the population.
2. Select a **campaign source IP** IP_attack (a single foreign IP, not in any entity's normal IP pool).
3. Select a **campaign start time** T_start: sample from business hours (credential stuffers use credential-list dumps that they execute in bulk).
4. For each target entity E_i:
   a. Insert N_fail_i ~ Uniform[1, 5] failure events (most stuffing attempts are just 1–2 per target before moving on):
      - `source_ip = IP_attack` (same IP for all entities in the campaign)
      - `auth_outcome = failure`, `auth_method = password`
      - `failure_count` = 1 through N_fail_i
      - `session_duration = 0.0`
      - `timestamp`: stagger between entities by Δ_stagger ~ Uniform[2, 15] seconds per event, so events across all entities are interleaved rapidly
      - `command_sequence = []`
      - `resource_accessed`: the login endpoint (same for all)
      - Label: `credential_stuffing`
   b. With probability 0.10 (compromise probability — lower than brute force because stuffed credentials are stale), insert one success event for that entity:
      - `auth_outcome = success`, `failure_count = 0`
      - Normal session behavior follows
      - Label: `credential_stuffing`

5. Total campaign duration: N_entities × Δ_stagger per entity × mean N_fail ≈ 60 × 8 seconds × 3 = 24 minutes. This compresses the entire campaign into less than 30 minutes.

#### Controlling Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `cs_n_entities_min` | 15 | [5, 30] | Minimum entities targeted |
| `cs_n_entities_max` | 60 | [30, 200] | Maximum entities targeted |
| `cs_n_fail_per_entity_min` | 1 | [1, 3] | Minimum failures per entity |
| `cs_n_fail_per_entity_max` | 5 | [2, 15] | Maximum failures per entity |
| `cs_compromise_probability` | 0.10 | [0, 0.3] | Per-entity probability of successful compromise |
| `cs_stagger_sec_min` | 2 | [0.5, 5] | Minimum stagger between inter-entity events |
| `cs_stagger_sec_max` | 15 | [5, 60] | Maximum stagger |

#### Distinguishability from Noisy Normal

Normal behavior includes isolated failure events with probability 0.01 per event. The distinguishing characteristic of credential stuffing is **population-level correlation**: many entities share the same `source_ip` at the same time. No normal behavior generates this pattern. The feature `ip_entity_ratio` (dim 22) and `entity_ip_ratio` (dim 23) in the feature vector specifically capture this cross-entity IP correlation.

Credential stuffing vs. brute force discrimination: brute force has many failures against one entity from one IP; credential stuffing has few failures per entity but against many entities from one IP. The classifier (boundary H) uses `session_event_count_norm` (dim 20) + `failure_count_norm` (dim 5) + `ip_entity_ratio` (dim 22) to discriminate.

#### Schema Fields Primarily Manipulated

`source_ip`, `auth_outcome`, `failure_count`, `entity_id` (multiple entities), `timestamp`, `label`

---

### 3.4 Lateral Movement

**Label:** `lateral_movement`  
**Target entity types:** `user` (90%), `service_account` (10%)  
**Typical signal:** A compromised entity rapidly accesses resources outside its normal `ResourceSet`, broadening across multiple resource categories in a single session.

#### Trigger / Injection Logic

1. Select a target user entity with a normal `ResourceSet` of size R.
2. Select an **injection session**: either an existing session or create a new one starting at T_inject.
3. Build an **expansion resource list** L_expand of size N_expand ~ Uniform[10, 30] resources, drawn **exclusively from outside** the entity's `ResourceSet`. The resources must span at least 3 distinct resource categories (e.g., `file/`, `api/`, `port/`, `db/`).
4. Build a **command sequence** C_lateral modeling a typical lateral movement pattern:
   - Start with reconnaissance commands: `ls`, `find`, `grep` targeting the expansion resources
   - Progress to access commands: `cat`, `ssh`, `curl`
   - End with exfil-adjacent commands: `scp`, `rsync` or `wget`
   - Sequence length: Uniform[8, 20] commands
   - All commands have `outcome = success` (attacker has compromised credentials)
   - `elapsed_seconds`: model realistic command execution with cumulative Gamma(shape=2, scale=30) increments
5. Generate N_expand events in the injection session, each accessing one resource from L_expand:
   - `timestamp`: space events 30–120 seconds apart (methodical access, not frantic)
   - `auth_outcome = success` for all (session already authenticated)
   - `session_duration`: total session accumulated; typically 900–7200 seconds
   - Spread `command_sequence` entries across events (each event gets a subset of C_lateral)
   - `device_fingerprint`: same as entity's registered device (attacker uses the compromised session)
   - Labels: all events in the injection session labeled `lateral_movement`

#### Controlling Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `lm_n_expand_min` | 10 | [5, 20] | Minimum expansion resources |
| `lm_n_expand_max` | 30 | [15, 60] | Maximum expansion resources |
| `lm_min_categories` | 3 | [2, 5] | Minimum distinct resource categories in expansion set |
| `lm_cmd_length_min` | 8 | [4, 15] | Minimum command sequence length |
| `lm_cmd_length_max` | 20 | [10, 40] | Maximum command sequence length |
| `lm_inter_event_sec_min` | 30 | [10, 60] | Minimum seconds between events (pace of movement) |
| `lm_inter_event_sec_max` | 120 | [60, 300] | Maximum seconds between events |
| `lm_include_exfil_command` | True | bool | Whether to include `scp`/`rsync` in command sequence |

#### Distinguishability from Noisy Normal

Tier 2 noise allows a rare resource access with probability 0.03 per event. But:
- Tier 2 rare resources are never in a burst (each event is independent)
- Tier 2 rare resources span at most 1 additional category (not 3+)
- Tier 2 events never include exfil commands (`scp`, `rsync`)
- `resource_breadth_norm` (dim 21) during a lateral movement session will be 10–30 / 50 = 0.2–0.6, compared to a normal session's 1–5 / 50 = 0.02–0.1

#### Schema Fields Primarily Manipulated

`resource_accessed`, `command_sequence`, `session_duration`, `resource_breadth_norm` (derived), `timestamp`, `label`

---

### 3.5 Device Spoofing

**Label:** `device_spoofing`  
**Target entity types:** `user` (60%), `edge_device` (40%)  
**Typical signal:** A known device ID (`device_fingerprint.device_id`) reappears with a different MAC address, OS family, or protocol than previously recorded.

#### Trigger / Injection Logic

1. Select a target entity. The entity must have at least 10 prior events (ensures the device fingerprint is established in the profile store).
2. Retrieve the entity's registered `DeviceFingerprint` (the primary device from `DeviceSet`).
3. Create a **spoofed fingerprint** by modifying one or more fields:
   - Strategy A (MAC spoof): keep `device_id`, `os_family`, `os_version`, `protocol` identical; change `mac_address` to a Faker-generated value not in the entity's `known_mac_addresses`.
   - Strategy B (OS spoof): keep `device_id` and `mac_address`; change `os_family` and `os_version` (e.g., from `Windows/11.0` to `Linux/22.04`). Also change `user_agent` to match.
   - Strategy C (Protocol spoof): keep `device_id`; change `protocol` (e.g., from `HTTPS` to `Modbus`). Relevant for edge devices accessed by a compromised controller.
   - Select strategy: A with probability 0.5, B with probability 0.3, C with probability 0.2.
4. Insert N_spoof ~ Uniform[1, 5] events using the spoofed fingerprint:
   - All other fields (geo, resource, auth) match the entity's normal profile (the attacker is using the entity's stolen credentials and mimicking normal access patterns)
   - `auth_outcome = success`, `failure_count = 0`
   - `label = device_spoofing`

#### Controlling Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `ds_n_spoof_events_min` | 1 | [1, 3] | Minimum events with spoofed fingerprint |
| `ds_n_spoof_events_max` | 5 | [3, 15] | Maximum events with spoofed fingerprint |
| `ds_strategy_weights` | (0.5, 0.3, 0.2) | — | Weight for MAC/OS/Protocol spoof strategy |
| `ds_min_prior_events` | 10 | [5, 30] | Minimum prior events before injection (ensures profile exists) |

#### Distinguishability from Noisy Normal

Tier 2 noise allows an **unknown device** (not in the `DeviceSet`) with probability 0.005 per event. Device spoofing is different:
- The spoofed event uses the entity's **registered `device_id`** but with a different MAC/OS — this is the key signal. A truly unknown device would have an unknown `device_id`.
- `fingerprint_mac_match` (dim 16) and `fingerprint_os_match` (dim 15) will both be 0.0 for spoofed events against a known `device_id`.
- Tier 2 unknown-device noise generates a new `device_id` entirely — the feature engineering treats it as an unknown device (no profile match), not a known device with a changed fingerprint.

#### Schema Fields Primarily Manipulated

`device_fingerprint` (device_id, os_family, os_version, mac_address, protocol, user_agent, firmware_version), `label`

---

### 3.6 Low-and-Slow Exfiltration

**Label:** `low_and_slow`  
**Target entity types:** `user` only  
**Typical signal:** A compromised entity accesses sensitive resources gradually, over many days, during off-hours, with each individual event appearing near-normal but the cumulative pattern indicating systematic data gathering.

#### Trigger / Injection Logic

Low-and-slow is a **longitudinal** attack injected across multiple days:

1. Select a target user entity with a normal active window that does **not** include late-night hours (00:00–05:00 UTC). Select an entity with a `ResourceSet` that includes `file/` resources (most common exfiltration target).
2. Select a **campaign duration** D_days ~ Uniform[5, 20] days.
3. Select a **target resource set** L_exfil of 5–15 sensitive resources from **within** the entity's `ResourceSet` (the attacker accesses resources the entity is legitimately authorized for — low-and-slow is about data aggregation, not unauthorized resource access).
4. For each day d in [1, D_days]:
   a. With probability p_daily = Uniform[0.4, 0.8] (attacker doesn't operate every day — models operational security), inject N_daily ~ Uniform[1, 3] events.
   b. Each injected event:
      - `timestamp`: sample from off-hours window Uniform[01:00, 04:30] UTC (slightly different each day to avoid exact pattern matching)
      - `resource_accessed`: sample from L_exfil (rotating through the target resources)
      - `command_sequence`: include at least one exfil command (`scp`, `rsync`, `wget`, or `curl`) with `target` pointing to an external IP not in the entity's normal resource pool
      - `session_duration`: sample from Uniform[300, 3600] seconds (brief but purposeful sessions)
      - `auth_outcome = success`, `failure_count = 0`
      - `device_fingerprint`: entity's registered primary device (no fingerprint anomaly — the attacker is persistent)
      - `geo_location`: entity's primary home location (no geo anomaly — local network access)
      - `source_ip`: within the entity's normal IP subnet (VPN-based access, no IP anomaly)
      - Label: `low_and_slow`

Note: The individual events are deliberately designed to have near-normal values on most features. The anomalous signal is primarily:
- Off-hours timestamp (dims 0–3 will show deviation from entity's active-hour baseline)
- Presence of exfil commands (dim 14 = 1.0)
- `inter_event_gap_norm` (dim 19) will show long gaps (multi-hour/multi-day between events)
- Cumulative `resource_rarity_score` (dim 9) will increase as the attacker cycles through less-frequently-accessed resources in L_exfil

#### Controlling Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `las_duration_days_min` | 5 | [3, 10] | Minimum campaign duration |
| `las_duration_days_max` | 20 | [10, 45] | Maximum campaign duration |
| `las_p_daily_min` | 0.4 | [0.2, 0.6] | Minimum daily operation probability |
| `las_p_daily_max` | 0.8 | [0.5, 1.0] | Maximum daily operation probability |
| `las_off_hours_start_utc` | 01:00 | [00:00, 03:00] | Start of off-hours window |
| `las_off_hours_end_utc` | 04:30 | [03:00, 06:00] | End of off-hours window |
| `las_n_exfil_resources_min` | 5 | [3, 10] | Minimum resources in exfil target set |
| `las_n_exfil_resources_max` | 15 | [8, 30] | Maximum resources in exfil target set |

#### Distinguishability from Noisy Normal

The challenge: Tier 2 noise allows off-hours access with probability 0.03 per event. The distinguishing signals are:
1. Exfil commands (`has_exfil_command = 1.0`, dim 14) — normal entities never issue `scp/rsync` except `developer` persona with probability 0.05 per session
2. The pattern is persistent across **multiple sessions over multiple days** — the SDM's sequence window (boundary C) captures this temporal pattern even when individual events look near-normal
3. `inter_event_gap_norm` values in the sequence window will show alternating long-gap / short-session patterns across days

This is intentionally the **hardest** attack to detect for the BPM (individual events are near-normal) but more detectable by the SDM (temporal pattern across events).

#### Schema Fields Primarily Manipulated

`timestamp` (off-hours), `command_sequence` (exfil commands), `session_duration`, `inter_event_gap_norm` (derived), `label`

---

### 3.7 Insider Drift (Edge Case)

**Label:** `insider_drift`  
**Target entity types:** `user` only  
**Design intent:** Insider drift is the **genuinely ambiguous** case. It is designed so that a well-calibrated model will reasonably score it as **medium-risk** (25–49 range) rather than clearly high-risk or clearly normal. The ambiguity is not cosmetic — it arises from real structural overlap with legitimate behavioral evolution.

#### Design Philosophy: Real Ambiguity, Not Cosmetic Ambiguity

Insider drift models a **legitimate employee** whose behavior has genuinely changed over time — a promotion, a new project assignment, or a role change that has not yet been formally reflected in their access control profile. The key property: **every individual event is fully authorized and individually explainable**. No single event is unambiguously malicious. The anomaly is a **pattern of gradual expansion** that may or may not indicate malicious intent.

This is distinct from lateral movement (which is fast, unauthorized, and session-level) and from low-and-slow (which involves off-hours access and exfil commands). Insider drift occurs:
- **During business hours** (the employee is legitimately working)
- **Using authorized resources** (the access control system permits access)
- **Without exfil commands** (no explicit exfiltration)
- **At a pace that mimics normal role evolution** (weeks, not days)

#### Trigger / Injection Logic

1. Select a target user entity. Assign a **drift persona** that determines the nature of the expansion:
   - `promotion`: entity gains access to higher-privilege resources over time (e.g., `analyst` gaining access to `executive` resource categories)
   - `project_handoff`: entity starts accessing a new project's resources due to team membership change
   - `role_overlap`: entity's responsibilities temporarily expand into a colleague's domain
2. Select a **drift start day** D_start ~ Uniform[5, 20] days into the simulation window (allows the normal profile to establish first).
3. Select a **drift rate** ρ_r ~ Uniform[1, 3] new resources per week (slow, deliberate expansion).
4. Build a **drift resource pool** L_drift: resources from outside the entity's normal `ResourceSet` but within the same general category family (e.g., more `file/finance/` resources for an `analyst`, not completely different categories like `port/22`).
5. For each day d from D_start to end of simulation:
   a. With probability p_drift ~ Uniform[0.2, 0.5] (not every day; models occasional new-project meetings):
      - Select 1–2 resources from L_drift that have not yet been accessed (rotating through the new resource pool as the weeks progress)
      - Insert 1–2 events accessing these resources:
        - `timestamp`: during entity's **normal active hours** (not off-hours — this is key ambiguity point 1)
        - `auth_outcome = success` (fully authorized access — key ambiguity point 2)
        - `session_duration`: within entity's normal distribution
        - `command_sequence`: normal to the entity's persona (no exfil commands — key ambiguity point 3)
        - `device_fingerprint`: entity's registered device (no device anomaly — key ambiguity point 4)
        - `geo_location`: entity's home city (no geo anomaly — key ambiguity point 5)
        - `source_ip`: entity's normal IP
        - Label: `insider_drift`
6. After D_start + 14 days, add **resource rarity score decay**: the first 2–3 drift resources start appearing multiple times per week, reducing their `resource_rarity_score` toward the normal range. This simulates the employee "settling in" to new responsibilities.

#### The Five Ambiguity Points (and Why They Matter for False-Positive Tuning)

| Ambiguity Point | Why It Matters |
|----------------|----------------|
| Normal business hours | The `hour_of_day_sin/cos` features (dims 0–3) do not flag this event. A system that flags time-of-day anomalies will not catch insider drift. |
| Fully authorized access | `auth_outcome = success`, `failure_count = 0`. The authentication stack says nothing is wrong. |
| No exfil commands | `has_exfil_command = 0.0` (dim 14). The command-sequence signal that catches low-and-slow is absent. |
| Registered device | `fingerprint_mac_match = 1.0` (dim 16). Device spoofing detectors see no issue. |
| Home geo-location | `geo_velocity_kmph = 0.0` (dim 6), `is_new_geo = 0.0` (dim 7). Impossible-travel detectors see no issue. |

#### What the Models Will See

- **BPM:** Individual events will have elevated `resource_rarity_score` (dim 9) and slightly elevated `resource_breadth_norm` (dim 21), but not above the normal-entity Tier 2 noise threshold (Tier 2 noise generates `resource_rarity_score` anomalies at probability 0.03). Expected BPM score: 0.3–0.55 (medium range).
- **SDM:** The sequence window across the 30-day period will show a slowly increasing diversity of resources accessed — but the sequence patterns themselves (command types, session durations, timing) remain normal. Expected SDM score: 0.25–0.50.
- **Fused score:** Expected range 0.28–0.52 (near the default threshold of 0.5).
- **Expected classification:** The classifier will often assign `insider_drift` but with moderate confidence (0.4–0.6) — it may also assign `lateral_movement` at lower probability, since the expanding resource footprint is the shared feature.

This means a well-tuned system will classify these events correctly but with a `classification_confidence` of 0.4–0.6 and a `risk_tier` of `medium` rather than `high`. This is the intended behavior: **insider drift is a "watch and investigate" signal, not an "immediately block" signal**.

#### Legitimate vs. Malicious Ambiguity

The generator does not inject any events that would unambiguously distinguish legitimate role expansion from insider threat. Both look identical in the data. In a real system, the resolution would come from HRMS data (role change notifications), but the generator does not include this. This is a deliberate design choice: it tests whether the model produces the correct *level* of uncertainty, not just the correct label.

#### Controlling Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `id_drift_start_day_min` | 5 | [3, 15] | Minimum day offset before drift begins |
| `id_drift_start_day_max` | 20 | [10, 25] | Maximum day offset |
| `id_drift_rate_min` | 1 | [1, 2] | Minimum new resources per week |
| `id_drift_rate_max` | 3 | [2, 5] | Maximum new resources per week |
| `id_p_daily_min` | 0.2 | [0.1, 0.4] | Minimum daily operation probability |
| `id_p_daily_max` | 0.5 | [0.3, 0.7] | Maximum daily operation probability |
| `id_resource_category_constrained` | True | bool | If True, drift resources stay within same category family (harder to detect) |

---

## 4. Configuration Surface

### 4.1 Main Configuration Keys (in `config/default.yaml`)

```
data_generator:
  run_id:                 "auto"      # "auto" = UUID v4 generated at runtime; or set explicitly for reproducibility
  random_seed:            42          # NumPy and Faker global random seed
  simulation_days:        30          # Length of simulated timeline in days
  simulation_start_date:  "2026-07-01T00:00:00Z"  # ISO-8601 start of simulated period

  entity_population:
    total_entities:       500         # Total number of entities
    user_fraction:        0.70        # Fraction of entities of type "user"
    service_account_fraction: 0.20   # Fraction of type "service_account"
    edge_device_fraction: 0.10       # Fraction of type "edge_device"

  events_per_entity:
    lambda_poisson:       120         # Average events per entity per 30 days

  injection:
    target_anomaly_rate:  0.015       # 1.5% of all events labeled non-normal
    anomaly_rate_min:     0.005       # Hard floor (from problem statement)
    anomaly_rate_max:     0.030       # Hard ceiling (from problem statement)
    # Per-attack-type shares of the total anomaly budget:
    brute_force_share:    0.20
    impossible_travel_share: 0.15
    credential_stuffing_share: 0.15
    lateral_movement_share: 0.15
    device_spoofing_share: 0.15
    low_and_slow_share:   0.10
    insider_drift_share:  0.10
    # Shares must sum to 1.0

  output:
    raw_parquet_path:     "data/raw/synthetic_logs_{run_id}.parquet"
    labels_parquet_path:  "data/labeled/labels_{run_id}.parquet"
    # Inference-schema parquet (label field absent):
    inference_parquet_path: "data/processed/inference_logs_{run_id}.parquet"
```

### 4.2 Injection Rate Enforcement

The generator computes the total target anomaly event count as:  
`N_anomaly = round(N_total_events × target_anomaly_rate)`

It then allocates anomaly events per attack type based on the `*_share` parameters. Shares are normalized to sum to 1.0 if they do not (defensive normalization). The generator does not inject more than `N_anomaly` events regardless of per-attack-type parameter ranges.

### 4.3 Random Seed / Reproducibility Strategy

1. A global `random_seed` is set in `config/default.yaml`.
2. At generator startup, `numpy.random.default_rng(seed=random_seed)` creates the primary RNG.
3. `Faker.seed_instance(random_seed)` is called to seed all Faker calls.
4. Each entity is assigned a **derived per-entity seed**: `entity_seed = random_seed + entity_index × 1000`. This ensures that entity-level generation is reproducible even if the entity population size changes (adding entities at the end does not change existing entity generation).
5. The `run_id` is stored in both output files' Parquet metadata. If `run_id = "auto"`, the generated UUID is logged to stdout so it can be used to reproduce the exact run.
6. All attack injection uses the same primary RNG, seeded before injection begins. Attack injection is always applied after the full normal dataset is generated, ensuring the same normal dataset regardless of which attacks are injected.

### 4.4 Dual-Output: Training Schema vs. Inference Schema

A single generator run produces both outputs simultaneously:

**Step 1:** Generate all events with labels → `RawAccessLog` objects with `label` field populated.  
**Step 2:** Write Training Schema output:
  - `data/raw/synthetic_logs_{run_id}.parquet` — all 15 fields including `label`.
  - `data/labeled/labels_{run_id}.parquet` — only `event_id` + `label`.

**Step 3:** Write Inference Schema output:
  - `data/processed/inference_logs_{run_id}.parquet` — all fields from Training Schema **except `label`**.
  - This is a direct projection of the Training Schema with the `label` column dropped.

**Step 4:** Verify label separation:
  - Assert that `inference_parquet_path` Parquet file does not contain a column named `label`.
  - Assert that `labels_parquet_path` contains exactly the two columns `event_id` and `label` and no others.

These assertions run as part of the generator's output validation before the run is considered complete.

**Note:** The inference Parquet file at `data/processed/inference_logs_{run_id}.parquet` is the on-disk representation of the Inference Schema. The `streaming/batch_reader.py` consumes this file (or the raw training file if running in evaluation mode where the streaming layer performs the stripping). This is consistent with DATA_SCHEMA.md Section 2c.

---

## 5. Alternatives Considered

### 5.1 Purely Procedural vs. Statistical/Generative Approach

**Chosen: Purely Procedural / Rule-Based Generation**

Each event is generated by sampling from explicitly parameterized distributions (Poisson, log-normal, Dirichlet, Gamma, etc.). There is no generative model (GAN, VAE) involved.

**Alternative A: Statistical/Generative Model (e.g., a lightweight VAE trained on a real dataset)**  
Description: Train a Variational Autoencoder on a publicly available network access log dataset (e.g., LANL or CERT). Use the trained VAE's latent space to generate synthetic events by sampling from learned distributions.

Why Rejected:
- The problem statement specifies that the dataset is synthetic and ground-truth labels are provided. Using a VAE trained on LANL/CERT would produce a distribution that reflects LANL/CERT's specific behavioral patterns, not a generalizable behavioral model. The resulting synthetic data would implicitly encode dataset-specific artifacts.
- A VAE requires a pre-training phase that itself needs labeled or clean data — a circular dependency for a generator designed to create labeled training data from scratch.
- SHAP and CAPTUM attributions require that feature importances can be traced back to interpretable input features. If the generator used a learned embedding, the ground-truth "reason" for an attack would be buried in latent dimensions — making the `human_readable_explanation` in `AlertPayload` meaningless as a ground-truth reference.
- Hackathon time budget: VAE pre-training, hyperparameter tuning, and generation-quality validation would consume 1–2 days of implementation time that must go to the ML detection pipeline.

**Alternative B: Library-Based Generation (SDV / Gretel.ai)**  
Already rejected in TECH_STACK.md Decision 2. The same reasoning applies: SDV generates data that mimics a source dataset's statistical distribution, not a behavioral profile specification.

**Why Procedural Wins:**  
Procedural generation produces data with **explicit, documented, auditable assumptions**. Every parameter in this document is a testable claim about entity behavior. The Technical Report can cite exact distribution parameters as behavioral model assumptions. Judges can verify that the attack injection is meaningful (not trivially separable) by reading the parameter ranges.

---

### 5.2 Independent Per-Entity Generation vs. Population-Level Generation with Cross-Entity Correlation

**Chosen: Primarily Per-Entity, with One Population-Level Attack (Credential Stuffing)**

For all attacks except credential stuffing, each entity's attack timeline is generated independently. For credential stuffing, the attack is generated as a population-level campaign that simultaneously targets multiple entities from a shared IP.

**Alternative A: Full Population-Level Generation (All Events Co-Generated)**  
Description: Generate the entire population's event stream as a single time-ordered sequence. Cross-entity correlations (shared IPs, correlated access patterns, population-wide behavioral changes) are introduced at the population level.

Why Rejected:
- Full population-level generation requires holding all entity timelines in memory simultaneously. For 500 entities × 120 events = 60,000 events, this is tractable, but the implementation complexity scales with the number of cross-entity correlation rules. For a hackathon, the complexity is disproportionate to the benefit.
- The only attack type that genuinely requires cross-entity correlation is credential stuffing (shared source IP across entities). All other attacks are single-entity phenomena. Introducing a full population-level generator for one attack type is over-engineering.
- The per-entity approach with a post-hoc credential stuffing campaign (Section 3.3) achieves the same cross-entity correlation for the only attack that needs it.

**Alternative B: Cross-Entity Behavioral Correlation (Peer-Group Norms)**  
Description: Generate entity profiles such that entities in the same peer group (e.g., all `analyst` users) share correlated behavioral baselines (similar active hours, overlapping resource sets). This would make the BPM's per-entity baseline more realistic (entities don't exist in isolation — they share organizational norms).

Why Rejected for T1 (but noted for T2/T3):
- Peer-group correlations would complicate the cold-start handler (T2) implementation: if the cold-start prior is group-level, the generator must produce group-level statistics, not just entity-level ones.
- The added realism is a T2 concern. For T1, per-entity independent profiles with the persona variant system (Section 2a) provide sufficient behavioral diversity without cross-entity dependencies.
- **Flagged for T2:** The Cold-Start Handler in `cold_start/priors.py` should use entity-type-level behavioral statistics derived from the generator's persona parameters — this information is available in the config but does not require cross-entity generation-time correlation.

**Why the Chosen Approach Wins:**  
Per-entity independent generation is parallelizable (each entity's timeline can be generated independently), reproducible (per-entity seeds isolate generation), and auditable (an entity's behavioral profile is self-contained in its parameter set). The single population-level attack (credential stuffing) is implemented as a post-processing step after all entity timelines are complete, preserving these properties.

---

## 6. Judging-Criteria Traceability

### 6.1 Detection Accuracy on Imbalanced Labels

| Design Choice | How It Supports Detection Accuracy |
|--------------|-----------------------------------|
| **Injection rate 0.5%–3%** | The problem statement requires testing on imbalanced data. The default 1.5% rate produces a dataset where anomalies are genuinely rare — exactly the regime where one-class and isolation-forest BPM models are appropriate. Training the classifier on this distribution forces it to handle class imbalance explicitly (oversampling, class weights) rather than cheating on balanced data. |
| **Non-trivially separable attack designs** | Each attack design specifies what makes it hard to distinguish from normal noise. Low-and-slow is designed to be individually near-normal at the event level. Insider drift is designed to overlap with legitimate behavioral evolution. A model that achieves high precision/recall on this data has genuinely learned the detection task. |
| **Longitudinal attacks (low-and-slow, insider drift)** | These attacks require the SDM to exploit its sequence-level memory. A model that only uses per-event features (BPM alone) will miss them. This design decision forces the score fusion (boundary G) to meaningfully combine both models — if the SDM were not genuinely needed, the architecture would be over-engineered. |

### 6.2 Report Clarity

| Design Choice | How It Supports Report Clarity |
|--------------|-------------------------------|
| **Fully documented distribution parameters** | Every parameter in this document (e.g., `bf_n_fail_min = 10`) is a citable claim in the Technical Report. The report can include a table of "Behavioral Profile Assumptions" drawn directly from Section 2 and a table of "Attack Injection Parameters" drawn from Section 3. |
| **Named persona variants** | The `executive`, `developer`, `analyst`, `support`, `remote_worker` personas give the Technical Report concrete behavioral archetypes to describe. A judge reading the report can immediately understand what "normal user behavior" looks like. |
| **Explicit noise tier design** | The three-tier noise design (field-level Gaussian, rare one-off deviations, temporal behavioral shift) is directly documentable as the "how we ensured the data was not trivially separable" section of the report. |

### 6.3 False Positive Rate at Realistic Alert Budget

| Design Choice | How It Supports FP Rate Calibration |
|--------------|-------------------------------------|
| **Tier 2 noise (3% rare resource access, 1% foreign IP)** | These noise events will inevitably trigger low-level anomaly scores in the BPM. The BPM and fusion threshold must be calibrated to not alert on these — they represent the realistic false-positive pressure from normal user behavior. The evaluation module (T2) can measure false positive rate specifically on Tier 2 noise events. |
| **Insider drift medium-risk design** | Insider drift is explicitly designed to land in the 25–49 `medium` risk tier. The `risk_tier` bucketing in `AlertPayload` means that a system configured for a "critical+high only" alert budget will correctly suppress insider drift alerts. The FP rate metric can be evaluated separately for each tier. |
| **Per-attack distinguishability analysis** | Each attack design in Section 3 documents exactly which feature dimensions separate it from normal noise. A well-implemented system can be tested with ablation: disabling each feature dimension and measuring the resulting FP rate increase. This provides a rigorous basis for the Technical Report's feature importance analysis. |

---

*End of SYNTHETIC_DATA_GENERATOR_DESIGN.md — Phase 5 output (Document 1 of 2). This document is frozen. Amendments require a versioned change record.*
