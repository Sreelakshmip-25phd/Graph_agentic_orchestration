# Study protocol: graph memory vs flat memory in the GOA (W-ref)

Written and committed **before** any evaluation run. Changing anything below after
results exist must be reported as a deviation.

## Question

Does replacing the Global Orchestration Agent's flat memory with a graph memory
improve orchestration decisions on the workload the flat-memory study was
evaluated on, and at what cost?

## Workload: W-ref

- The earlier study's workload generator (`src/phase1_scenario.py`), topology
  (`scenario/topology_eval.json`: 10 zones, 50 nodes), service catalog (7 types)
  and lifetime calibration (K = 48, `scenario/calibration_report.json`).
- Per seed: 1,000 requests in three phases of 1 h (A: 4 familiar types; B: + 3
  new types; C: surge on the zone with the highest trace activity), one node
  failure at t = 9,720 s.
- **Evaluation seeds 0–9.** Seeds 0–2 are the earlier study's files, byte-identical
  (`src/make_wref_workloads.py` regenerates them and checks). Seeds 3–9 are new
  draws from the same generator.
- **Dev seeds 100–104**: same generator, used only for the tuning step, never
  reported.

## Conditions

One agent (`GlobalOrchestrationAgent`), one LLM, one escalation chain. Only the
memory differs.

| Condition | Memory |
|---|---|
| `graph_memory` | `GraphMemory`: routes per (service type, origin zone), zone RTT links, similar types, zone failures, preemption outcomes, rules. Used in the fast path, as `graph_memory` context in the LLM prompt, and to rank preemption victims. |
| `flat_memory` | Episodic + procedural memory keyed by service type. Code path unchanged from the earlier study. |
| `no_memory` | No memory fast path, no digest scan (the earlier `memory_ablated`). |

LLM: Qwen2.5-7B-Instruct Q4_K_M via the local llama.cpp server, temperature 0.
The served model id is recorded in every result file. Each condition has its own
LLM answer cache, shared across that condition's seeds only.

Known, intended differences that come with graph memory: the GOA prompt gains a
`graph_memory` block (top 3 zones) and the preemption prompt gains each
candidate's `past_migration_rate`. Their token cost is measured (metric C3).

## Tuning (before evaluation)

The graph zone score has five weights. Three settings are fixed in
`src/graph_memory.py` (`history`, `balanced`, `risk`). Each is run with
`graph_memory` on dev seeds 100–104. **Chosen**: highest mean acceptance;
tie-break fewer LLM reaches per 100 requests; then the order listed. The choice is
written to `results/tune/chosen_weights.json` and used unchanged for evaluation.

## Metrics

**A. Task outcome by memory type**
- A1 acceptance (overall; per phase A, B, C)
- A2 SLA-violation rate: intra-zone 1 ms + inter-zone RTT (if cross-zone) above
  the request's `latency_ms`
- A3 replan success rate: (replanned + migrated) / (replanned + migrated + displaced)
- A4 displaced services at end of run

**B. How the memory helped**
- B1 memory precision (first try): share of escalations where the first
  memory-suggested zone accepted the request
- B2 memory hit rate: share of escalations resolved by the memory fast path
- B3 preemption regret: share of evicted services that ended displaced
- B4 acceptance drop after the failure: acceptance in the 30 min before minus
  the 30 min after the node failure

**C. Memory cost**
- C1 LLM reach per 100 requests (decisions reaching the LLM block, cached or not)
- C2 LLM consultations per 100 requests (zone choice + preemption + rule authoring)
- C3 LLM prompt tokens per 100 requests (estimated as characters / 4, cache-independent)
- C4 decision latency, mean and p95 (ms)
- C5 graph lookup time (ms) and graph size

## Analysis

Paired by seed. For each comparison (graph − flat, graph − none, flat − none) and
metric: mean difference, 95% t-interval, Wilcoxon signed-rank p, Holm correction
across the metrics of that comparison. α = 0.05.

## Pre-registered checks (`results/gates.json`)

| Check | Pass condition |
|---|---|
| G1 reproduction | flat memory on seeds 0–2 within 0.02 acceptance and 5 LLM reaches/100 of the earlier study |
| G2 non-inferiority | graph − flat acceptance: lower 95% bound > −0.01 |
| G3 memory precision | graph > flat, Holm p < 0.05 |
| G4 memory reduces LLM use | graph < none and flat < none on LLM reach, Holm p < 0.05 |
| G5 prompt overhead | graph prompt tokens/100 req at most 25% above flat |
| secondary | graph > flat on acceptance, Holm p < 0.05 (not expected on W-ref) |

## What W-ref can and cannot show

From dry runs (no LLM): most LLM reaches on W-ref come from capacity exhaustion
(`llm_unresolved`) and zone-local preemption, which no memory can avoid, and
inter-zone RTTs are 4–9 ms against a tightest budget of 30 ms. So W-ref is expected
to show **decision precision and non-inferiority**, not large acceptance gains.
The Stress workload (next step, separate protocol) is where relational memory is
expected to change outcomes.

## Deviations (recorded after results existed)

- **Preemption count (reporting only, no gate affected).** The W-ref run's
  `preemptions` field counted distinct evicted services, while the earlier
  study counted eviction events (a service evicted twice counts twice). From
  the Stress workload on, `audit()` reports both, as `preemption_events` and
  `preempted_services`; preemption regret keeps distinct services as its
  denominator. W-ref result files were not re-run: the reproduction check
  reports their distinct-service count and the GOA's preemption-decision
  counter, and leaves the event count empty.
