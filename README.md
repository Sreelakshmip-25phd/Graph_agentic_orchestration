# Graph-memory agentic edge orchestration

The Global Orchestration Agent (GOA) from the agentic edge-orchestration study,
with its flat memory replaceable by a **graph memory** of zones, service types,
routes per origin zone, RTT links, failures, preemption outcomes and rules.
This repository runs the comparison **graph vs flat vs no memory** on the
earlier study's workload (W-ref). The protocol is in [STUDY.md](STUDY.md).

## Layout

```
src/
  graph_memory.py        graph memory (networkx) and the zone score
  goa_agent.py           GlobalOrchestrationAgent(memory="graph"|"flat"|"none")
  evaluator.py           discrete-event simulation + per-run metrics
  run_study.py           tune / eval runner (checkpointed, resumable)
  analyze_study.py       summary, paired statistics, gates, figures
  make_wref_workloads.py regenerates W-ref seeds (checks seeds 0-2 are identical)
  make_stress_workloads.py builds the Stress scenario and workloads
  sim_engine.py, zoa_agent.py, llm_client.py, phase1_*.py   from the earlier study
scenario/                topology, service catalog, intent sentences, calibration
workloads/wref/          evaluation workloads, seeds 0-9
workloads/dev/           tuning workloads, seeds 100-104 (never reported)
workloads/stress{,_dev}/ Stress workloads (STUDY_STRESS.md)
scenario/stress/         Stress topology, catalog, intent sentences, report
reference_results/       the earlier study's flat-memory results (reproduction check)
local_llm/               llama.cpp server helpers (same as the earlier study)
```

## Running it (GPU machine)

Same environment as the earlier study; two new packages:

```bash
pip install -r requirements.txt
```

Start the same model as before (Qwen2.5-7B-Instruct Q4_K_M) in one terminal:

```bash
cd local_llm && python3 start_server.py medium
```

In a second terminal, from the repository root:

```bash
export LOCAL_LLM_URL=http://127.0.0.1:8080
python3 src/run_study.py --stage tune
python3 src/run_study.py --stage eval
python3 src/analyze_study.py
```

On Windows PowerShell use `$env:LOCAL_LLM_URL="http://127.0.0.1:8080"` and `python`.

- **tune**: graph memory, 3 weight settings × 5 dev seeds = 15 runs.
- **eval**: 3 memory conditions × 10 seeds = 30 runs.
- Every run is saved as soon as it finishes; re-running a command skips finished
  runs, so an interrupted run resumes where it stopped.
- Expected time on an RTX 4070 SUPER: about 2–3 min per run (the earlier study's
  flat-memory runs took ~2 min of decision time, no-memory ~3 min), so roughly
  30–45 min for tune and 1–1.5 h for eval.
- Check the code path without an LLM: `python3 src/run_study.py --stage eval --dry-run --seeds 0`
  (results go to `results/dryrun/`, which is not committed).

## Stress workload

Protocol: [STUDY_STRESS.md](STUDY_STRESS.md). Regional RTT (5–80 ms), five
failures (two zone-wide), a moving phase-C hotspot and four new service
types; every condition's LLM prompt carries the origin's RTT row, and the
digest scan skips zones that would break the latency budget. Workloads are
committed (`workloads/stress/`, `workloads/stress_dev/`, built by
`src/make_stress_workloads.py`; scenario files in `scenario/stress/`).

```bash
python3 src/run_study.py --workload stress --stage tune
python3 src/run_study.py --workload stress --stage eval
python3 src/analyze_study.py --workload stress
```

- **tune**: 3 weight settings × 5 Stress dev seeds = 15 runs; **eval**: 3 conditions × 10 seeds = 30 runs.
- Results go to `results_stress/` (same layout as `results/`, plus
  `path_breakdown.csv`; gates S1–S8 in `gates.json`). LLM caches are separate
  (`llm_cache/stress_*`).
- Run `analyze_study.py` on one machine only: Wilcoxon p-values with ties
  differ between scipy versions; `gates.json` records the version.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `tune/chosen_weights.json` | weight setting chosen on the dev seeds |
| `eval/<system>_seed<n>.json` | per-run metrics, decision trace, model id |
| `reproduction_check.csv` | flat memory seeds 0–2 vs the earlier study |
| `summary.csv` | system × metric, mean and 95% CI |
| `paired_stats.csv` | graph−flat, graph−none, flat−none: mean difference, CI, Wilcoxon, Holm |
| `gates.json` | pre-registered checks G1–G5 |
| `fig1_paired_differences.png` | paired differences per metric |
| `fig2_acceptance_over_time.png` | acceptance over time, phases and failure marked |
| `fig3_decision_trace.png` | which graph relationship decided each graph-memory placement |

Commit `results/` (except `dryrun/`) and push it back for analysis. The LLM
answer caches (`llm_cache/`) stay local.
