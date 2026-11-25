# Study protocol (DRAFT): graph memory vs flat memory in the GOA — Stress workload

Status: **approved (Option B), committed before any Stress run with the LLM.**
Only dry runs (no LLM) were used while building, to check the workload
stresses what it claims to. Changes after LLM results exist are reported as
deviations.

## Why a second workload

W-ref (STUDY.md) showed that graph memory is safe (acceptance non-inferior)
and wastes far fewer placement attempts (442 memory attempts for 395
successes, against 950 attempts for 392 with flat memory), but it could not
change outcomes:

- Inter-zone RTTs are 4.3–9.4 ms. Only services with `data_locality: any`
  can go cross-zone, and their budgets are 200–800 ms, so no placement can
  break its latency budget (SLA violations were 0 in every run).
- There is one node failure, so failure history barely matters (the failure
  penalty changed a graph decision 6 times in 10 seeds).
- The phase C hotspot stays on one zone, so route history never goes stale.
- The three phase B types are new to everyone, but similarity to known types
  is not tested on its own terms.

Stress turns each of these on at once (the combined variant, "S5"). It tests
whether the relationships the graph stores (RTT links, failures, similar
types, time-decayed routes) change decisions when they matter.

## Workload: Stress

The same generator, topology, catalog and calibration as W-ref (same
10 zones, 50 nodes, 1,000 requests, three 1-hour phases, K = 48), with four
changes.

| # | Change | W-ref | Stress | Source / reason |
|---|---|---|---|---|
| 1 | Inter-zone RTT | 4.3–9.4 ms (RIPE Atlas, metro) | 5–80 ms. The zones become a regional edge, with a near cluster and far zones; drawn once per topology from a fixed seed | Regional range so that cross-zone choices can break tight budgets. The exact source for the regional range is still to be named (RIPE Atlas regional anchors) |
| 2 | Latency-tight types that may cross zones | none (the cross-zone types have 200–800 ms budgets) | 2 new `any`-locality types with budgets of 25 and 40 ms, 1 type similar to a cross-zone type, 1 dissimilar control | Without these, RTT can never cause an SLA violation (see above) |
| 3 | Failures | 1 node, t = 9,720 s | 5 failures, correlated: 2 events take down most nodes of one zone together, 3 are single nodes; repair time drawn from the Google-cluster downtime distribution (`failure_model.json`, p25–p75 = 6.6–118 min) clipped to 10–60 min | Repeated failures on zones that were good routes, to test failure memory and recovery |
| 4 | Phase C hotspot | one zone (the trace's surge zone) for the whole hour | moves across 3 zones, 20 min each (the trace's surge zone, then 2 others) | Route history goes stale at each move, to test time decay |

### New service types (phase B onwards)

| Type | Similar to | Locality | Budget | Priority | Purpose |
|---|---|---|---|---|---|
| `vehicle_teleop` | `drone_control` (0.70), `vr_training_session` (0.72) | any | 25 ms | high | tight budget, similar type |
| `vr_training_session` | `ar_session` (0.73), `vehicle_teleop` (0.72) | any | 40 ms | high | tight budget, similar type |
| `parking_occupancy` | `traffic_monitor` (0.87) | any | 200 ms | normal | similar-type transfer to a type that crosses zones |
| `log_archive_batch` | none (highest 0.55, below 0.6) | any | 2,000 ms | low | **negative control**: dissimilar, latency-insensitive |

Similarity values are `type_similarity()` on the final sentences (MiniLM),
checked by the generator, which stops if any designed link is missing.
**Changed from the first draft** (before any run): the draft's
`crowd_counting` (60 ms), `vehicle_teleop` and `remote_inspection_stream`
did not come out similar to their intended pairs. A latency-tight type that
may leave its zone is "realtime + any", a profile no existing type has, so
the tight types end up similar to `ar_session` / `drone_control` (which never
leave their zone) and to each other. Similar-type transfer to a type that
*does* go cross-zone is therefore tested with `parking_occupancy`. Mix: new
types are 30% of phase B and C requests (teleop 10%, VR 10%, parking 7%,
archive 3%); the W-ref mix is scaled to the other 70%.

Each type gets 3 seed intent sentences paraphrased like the existing ones.
Similarity is whatever `type_similarity()` computes (profile and embedding);
it is checked and reported before any run. The plan expects the first three
to come out SIMILAR_TO their intended pair and the control to come out
similar to none.

Adding types changes the SLM schema prompt for all conditions alike, so
Stress numbers are not comparable one-for-one with W-ref. That is expected.

### Simulation setting (proposed values)

Topology: the W-ref zones and nodes, grouped into three regions. RTT per
zone pair is drawn once (fixed seed) from the band of that pair.

| Region | Zones (nodes) | RTT inside |
|---|---|---|
| Metro core | z1 (11), z0 (3), z2 (3), z3 (4) | 5–10 ms |
| East | z6 (8), z4 (3), z5 (4) | 10–20 ms |
| West | z7 (7), z8 (4), z9 (3) | 10–20 ms |

Between regions: core to East or West 30–50 ms, East to West 60–80 ms. Intra-zone stays 1 ms.
So `vehicle_teleop` (25 ms) can only leave its zone within its own
region (5–20 ms). `vr_training_session` (40 ms) also reaches the
neighbouring regions when that pair's RTT is under 39 ms, but never the opposite
region.

Timeline (3 h, as W-ref):

| Time | Event |
|---|---|
| 0:00–1:00 | Phase A: 4 known types |
| 1:00–2:00 | Phase B: + 3 W-ref new types + 4 Stress new types (new types about 30% of B and C requests) |
| 0:45 | single node fails, z3 |
| 1:30 | zone-wide failure, z6 (6 of 8 nodes) |
| 2:00–2:20 | hotspot on z1 (the trace's surge zone, as W-ref) |
| 2:05 | single node fails, z1 |
| 2:20–2:40 | hotspot on z6 |
| 2:30 | zone-wide failure, z7 (5 of 7 nodes) |
| 2:40–3:00 | hotspot on z7 |
| 2:50 | single node fails, z4 |

Failure times and zones are fixed across seeds. The failed node(s) and repair
times (10–60 min, from `failure_model.json`) are drawn per seed.

### Seeds

- Evaluation seeds 0–9, development seeds 100–104 (new files in `workloads/stress/` and
  `workloads/stress_dev/`).
- A generator script (`src/make_stress_workloads.py`) writes them and a
  report: per seed, failures (time, zone, nodes, repair), hotspot schedule,
  RTT distribution, per-type counts, and the similarity matrix.

## Conditions

Same three as W-ref: `graph_memory`, `flat_memory`, `no_memory`. Same LLM
(Qwen2.5-7B-Instruct Q4_K_M, temperature 0), one LLM cache per condition.

### Who sees RTT (decided: Option B)

In the current code, only graph memory sees RTT: its `graph_memory` hints in
the LLM prompt carry `rtt_ms`, and its zone score subtracts an RTT penalty.
Flat memory, the digest scan and the plain LLM payload are blind to RTT.
On Stress, a graph-memory win on SLA violations could then come from extra
*information* rather than from better *memory*.

- **Option A, keep as is.** Flat memory is the earlier study's GOA unchanged.
  The NEAR edges are part of what graph memory is. The claim would be
  "graph memory brings topology into the decision", and a reviewer can
  object that flat memory plus an RTT table would do the same.
- **Option B (recommended), give every condition the RTT row.** The LLM
  payload for all three conditions includes the origin zone's RTT to each
  zone (`rtt_ms_from_origin`), and the system prompt tells the LLM to respect
  `latency_ms`. Graph memory keeps its RTT term in the fast-path score,
  because that is how a graph uses an edge. Any remaining graph-memory gain
  then comes from memory structure (which past placements, from which
  origins, how recently, after which failures), not from access to RTT.
  This is a recorded change from the earlier study's flat GOA; W-ref already
  establishes the reproduction.

**Digest scan, decided while building (dry runs only).** The cross-zone
digest scan (memory conditions only) originally ignored RTT. In dry runs it
caused half or more of all SLA violations (Stress seed 0: 24 of 48 for flat,
30 of 52 for graph), identically for both memories, which would hide any
difference between them. In line with Option B (same information for every
condition), the digest scan now skips zones whose `rtt + 1 ms` exceeds the
request's budget, in every condition (`digest_rtt_aware`, on for Stress, off
for W-ref). Dry runs, seeds 0–2, mean SLA-violation rate:

| | flat | graph | none |
|---|---|---|---|
| digest RTT-blind | 0.047 | 0.043 | 0.044 |
| digest RTT-aware (used) | 0.033 | 0.030 | 0.044 |

With it, the violations left on seed 0 come from the memory fast path (flat
9 of 46 placements, graph 3 of 34), the LLM fallback, preemption and
replanning. The memory difference is now visible rather than masked. The no-memory condition has no
digest scan, so it is unaffected. The preemption and replanning orders stay
as in the earlier study for all conditions.

What the LLM sees under Option B (all three conditions get the same fields,
except `graph_memory`):

```json
{"request": {"cpu_req": 12, "latency_ms": 25, "priority": "high", "origin_zone": "z9"},
 "profile": {...},
 "zone_digests": {"z1": {"top_cpu_free": 38}, "z8": {"top_cpu_free": 22}},
 "rtt_ms_from_origin": {"z8": 12, "z7": 18, "z1": 45, "z6": 72},
 "graph_memory": [{"zone": "z8", "success_rate": 0.96, "evidence": 5.2, "rtt_ms": 12}]}
```

The system prompt adds, for all conditions: "rtt_ms_from_origin is the delay from
the request's origin to each zone; a zone with rtt + 1 ms above
request.latency_ms breaks the SLA." The LLM cache key gains the origin zone,
because the same digests now mean different answers from different origins.

## Graph memory on Stress

The graph schema is unchanged from W-ref (`src/graph_memory.py`):

| Element | Stores | Expected size |
|---|---|---|
| `zone` | failure times (decaying failure risk) | 10 |
| `type` | service type | 11 |
| `ctx` (type, origin) | "type X requested from zone Y" | up to 110 |
| NEAR zone→zone | RTT ms | 90 |
| SIMILAR_TO type→type | similarity ≥ 0.6, top 2 per type | ~6–12 |
| HAS_CTX type→ctx | membership | = ctx |
| ROUTED_TO ctx→zone | decayed successes / failures, last time | ~150–250 |
| PREEMPTED type→type | evictions, migrated / displaced | ~10–20 |
| PREFERS rule→zone | confidence, valid (invalidated by failure) | a few |

Worked example: the first `vehicle_teleop` request from z9 has no direct
history. The graph pools evidence from teleop requests from z8 (12 ms away,
weighted by proximity) and from the similar type `vr_training_session` from
z9 (weighted by similarity 0.72), which was placed in z1. The RTT penalty
then rules out z1 (45 ms against a 25 ms budget), so z8 ranks first. (Its
other similar type, `drone_control`, never leaves its zone, so its routes
only point back at the origin.) Flat memory, keyed by type only,
would offer the zone where teleop worked most often, from any origin.

## Tuning

Re-run before evaluation, because the RTT and failure weights now matter.
Use the same three settings (`history`, `balanced`, `risk`) on Stress dev seeds
100–104, with the same rule: highest mean acceptance, tie-break fewer LLM
reaches. Proposed addition, decided here before any run: when two settings
are within 0.005 acceptance, prefer the lower SLA-violation rate before
looking at LLM reaches. Output goes to `results_stress/tune/chosen_weights.json`.

## Metrics

All W-ref metrics (A1–A4, B1–B4, C1–C5), plus:

| ID | Metric | Why |
|---|---|---|
| A2 | SLA-violation rate (already defined; now non-zero) | primary outcome on Stress |
| A5 | SLA-violation rate on the 3 latency-tight new types alone | where RTT can bite |
| B5 | Wasted memory attempts per 100 requests (memory-suggested zone tried and refused) | W-ref's clearest effect, now pre-registered |
| B6 | Acceptance drop after each failure (15 min before minus 15 min after; mean over 5), and for the 2 zone-wide failures alone | failure memory |
| B7 | Acceptance in the 10 min after each hotspot move, compared with the 10 min before | stale-route handling (time decay) |
| B8 | Memory precision (first try) on the 3 similar new types, pooled, and on the control type, over the whole run. Each new type's acceptance in its first 30 min is reported but not tested: in dry runs it was 1.0 for every condition (phase B starts lightly loaded), so it cannot separate them | SIMILAR_TO; the control type should show no graph-memory advantage |
| B9 | Share of placements that are LLM, digest scan, memory | breakdown behind G4 (digest scan vs memory) |
| B10 | Preemption events and preempted services (both) | as fixed after W-ref |
| — | Graph decision trace: direct / nearby origin / similar type / rule; RTT and failure penalty changed the choice | how the graph decided |

## Analysis

Same as W-ref: paired by seed, mean difference, 95% t-interval, Wilcoxon
signed-rank p, Holm across the metrics of each comparison, α = 0.05.
`analyze_study.py` is run on one machine only (the GPU machine), and its scipy
version is recorded, because Wilcoxon p-values with ties differ between scipy
versions (seen on W-ref).

## Pre-registered checks (`results_stress/gates.json`)

| Check | Pass condition | Type |
|---|---|---|
| S1 SLA | graph < flat on SLA-violation rate, Holm p < 0.05 | primary |
| S2 acceptance | graph − flat acceptance: lower 95% bound > −0.01 (non-inferior) | primary |
| S3 acceptance gain | graph > flat on acceptance, Holm p < 0.05 | secondary |
| S4 wasted attempts | graph < flat on B5, Holm p < 0.05 | secondary |
| S5 failures | graph < flat on B6 (mean acceptance drop after failures), Holm p < 0.05 | secondary |
| S6 similar types | graph > flat on B8 memory precision for the 3 similar new types, Holm p < 0.05. The control type is reported, not tested: it is small low-priority batch work that almost always fits in its home zone, so memory is rarely asked about it (in dry runs graph memory had no first tries on it at all). Graph memory is more precise in general (W-ref), so S6 is read together with overall precision (B1) | secondary |
| S7 LLM use | graph < none and flat < none on LLM reach, reported together with the B9 breakdown | secondary |
| S8 cost | graph prompt tokens/100 requests at most 25% above flat | cost |

A sanity check runs before the study: on one Stress seed, the dry run (no
LLM) must show non-zero SLA violations for at least one condition. If not,
the workload does not stress what it claims to, and the RTT range or budgets
are revised before any LLM run, with the change recorded here.

## What it costs to run

Estimated on the RTX 4070 SUPER from W-ref timings (graph and flat ~2.4 min,
none ~4.3 min per run). Stress will likely be somewhat slower, because more
escalations mean more LLM calls.

| Stage | Runs | Estimate |
|---|---|---|
| tune | 3 settings × 5 dev seeds = 15 | ~40–50 min |
| eval | 3 conditions × 10 seeds = 30 | ~1.5–2 h |

## Files

```
src/make_stress_workloads.py   Stress topology RTT, failures, hotspot, new types; writes workloads + report;
                               stops if a designed SIMILAR_TO link is missing
scenario/stress/               topology_stress.json, service_catalog_stress.json, intent_sentences_stress.json, stress_report.json
workloads/stress/, workloads/stress_dev/
src/run_study.py               --workload wref|stress (default wref; W-ref behaviour unchanged)
src/analyze_study.py           --workload wref|stress; Stress metrics and gates S1–S8, path_breakdown.csv
src/evaluator.py               SLA over all placements, by type and by path; per-failure-event and
                               hotspot-move windows; new-type first-30-min acceptance
src/goa_agent.py               rtt_in_prompt (Option B) and digest_rtt_aware, both on for Stress only;
                               memory precision per service type
results_stress/                outputs (same layout as results/)
```

W-ref behaviour is unchanged: with both switches off, the W-ref dry run on seed 0
gives the same numbers as before (flat 0.86 / 31.3 reaches / precision 0.432,
graph 0.865 / 31.4 / 0.943), and re-analysing the committed W-ref results gives
the same gates.
