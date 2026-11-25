#!/usr/bin/env python3
"""Run the W-ref study: graph vs flat vs no memory in the same GOA.

Stages
  tune  graph memory only, each weight setting in graph_memory.WEIGHT_SETTINGS
        on the dev seeds (workloads/dev). Picks the setting with the highest
        mean acceptance (tie-break: fewer LLM reaches) and writes
        results/tune/chosen_weights.json. Dev seeds are never reported.
  eval  flat_memory, graph_memory, no_memory on the W-ref seeds
        (workloads/wref), graph memory using the chosen weights.

Every run is checkpointed: an existing result file is skipped, so an
interrupted run resumes where it stopped. LLM answers are cached per
system (llm_cache/), shared across that system's seeds and never across
systems, as in the earlier study.

Workloads (--workload)
  wref    the earlier study's workload (STUDY.md); results/ (default)
  stress  STUDY_STRESS.md: regional RTT, 5 failures, moving hotspot, new
          types; RTT row in every condition's prompt; results_stress/

Usage
  python src/run_study.py --stage tune
  python src/run_study.py --stage eval
  python src/run_study.py --stage eval --systems graph_memory --seeds 0 1
  python src/run_study.py --stage eval --dry-run --seeds 0   # no LLM, code check only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CACHE_DIR, DEV_SEEDS, EVAL_SEEDS, RESULTS, SCEN, WORKLOADS, ZOA_DIR

# per workload: scenario files, eval / dev workload dirs, results root,
# cache prefix, RTT row in the prompt (Option B), failure window (s)
WORKLOAD = {
    "wref": dict(scen=SCEN, topo="topology_eval.json",
                 catalog="service_catalog.json", sentences="intent_sentences.json",
                 eval="wref", dev="dev", results=RESULTS, cache="",
                 rtt_in_prompt=False, digest_rtt_aware=False, adapt_window_s=1800.0),
    "stress": dict(scen=os.path.join(SCEN, "stress"), topo="topology_stress.json",
                   catalog="service_catalog_stress.json",
                   sentences="intent_sentences_stress.json",
                   eval="stress", dev="stress_dev", results=RESULTS + "_stress",
                   cache="stress_", rtt_in_prompt=True,
                   digest_rtt_aware=os.environ.get("STRESS_DIGEST_RTT", "1") == "1",
                   adapt_window_s=900.0),
}
from evaluator import Simulation, load_workload
from goa_agent import GlobalOrchestrationAgent
from graph_memory import WEIGHT_SETTINGS, type_similarity
from llm_client import MultiLLM, PROVIDERS, discover_providers
from sim_engine import solve_placement
from zoa_agent import Embedder, IntentLibrary, build_schema, ground_truth, \
    slm_system_prompt

SYSTEMS = {"flat_memory": "flat", "graph_memory": "graph", "no_memory": "none"}
CACHE_PREFIX = ""


class StubLLM:
    """--dry-run only: answers nothing, so every LLM decision falls back to
    the GOA's deterministic fallback. For checking the code path, not for
    results."""
    def __init__(s, *a, **k):
        s.calls = s.tokens = s.cache_hits = 0

    def ask(s, system, user, cache_key=None):
        return {}

    def flush(s):
        pass


def model_name():
    """Ask the local server which model it serves, so every result file
    records it (the earlier study's results did not)."""
    if os.environ.get("STUDY_MODEL_NAME"):
        return os.environ["STUDY_MODEL_NAME"]
    for p in PROVIDERS:
        if p.get("local"):
            base = p["url"].rsplit("/v1/", 1)[0]
            try:
                j = requests.get(base + "/v1/models", timeout=5).json()
                return ", ".join(m.get("id", "?") for m in j.get("data", []))
            except Exception as e:
                return f"unknown ({e})"
    return "hosted: " + ", ".join(p["name"] for p in PROVIDERS)


def setup(W):
    catalog = json.load(open(os.path.join(W["scen"], W["catalog"])))["services"]
    sentences = json.load(open(os.path.join(W["scen"], W["sentences"])))
    topo = json.load(open(os.path.join(W["scen"], W["topo"])))
    _, required_fields, semantic = build_schema(catalog)
    slm_system = slm_system_prompt(build_schema(catalog)[0])

    print("loading embedder...")
    embedder = Embedder([s["text"] for s in sentences])
    print(f"  backend: {embedder.backend}")
    texts = {s["text"] for s in sentences}
    for sub in (W["eval"], W["dev"]):
        d = os.path.join(WORKLOADS, sub)
        for f in os.listdir(d) if os.path.isdir(d) else []:
            reqs, _ = load_workload(os.path.join(d, f))
            texts |= {r["intent_text"] for r in reqs}
    texts = sorted(texts)
    print(f"pre-encoding {len(texts)} unique texts...")
    vec_cache = dict(zip(texts, embedder.encode(texts)))

    thr = json.load(open(os.path.join(ZOA_DIR, "threshold_tuning.json")))["tuning"]["thr"]
    print(f"intent-cache threshold: {thr}")

    centroids = {}
    for st in catalog:
        v = np.mean([vec_cache[s["text"]] for s in sentences
                     if s["service_type"] == st], axis=0)
        centroids[st] = v / (np.linalg.norm(v) + 1e-12)
    type_sims = type_similarity(semantic, centroids)

    def fresh_lib():
        return IntentLibrary([(s["text"], ground_truth(catalog, s["service_type"]))
                              for s in sentences], thr, embedder, vec_cache)

    return dict(topo=topo, required_fields=required_fields, semantic=semantic,
                slm_system=slm_system, fresh_lib=fresh_lib, type_sims=type_sims,
                W=W)


def run_one(ctx, memory, weights, slm, goa, workload_path, out_path, meta):
    reqs, events = load_workload(workload_path)
    orch = GlobalOrchestrationAgent(ctx["fresh_lib"](), slm, goa, ctx["topo"],
                                    ctx["semantic"], solve_placement,
                                    memory=memory, graph_weights=weights,
                                    type_sims=ctx["type_sims"],
                                    rtt_in_prompt=ctx["W"]["rtt_in_prompt"],
                                    digest_rtt_aware=ctx["W"]["digest_rtt_aware"])
    c0, tok0 = goa.calls, goa.tokens
    t0 = time.time()
    r = Simulation(ctx["topo"], reqs, events, orch, ctx["required_fields"],
                   ctx["slm_system"],
                   adapt_window_s=ctx["W"]["adapt_window_s"]).run()
    wall = time.time() - t0
    m, n = orch.metrics, len(reqs)

    def ratio(a, b):
        return round(m[a] / m[b], 3) if m[b] else None

    mem_resolved = m["esc_graph"] if memory == "graph" else \
        m["esc_episodic"] + m["esc_procedural"]
    r.update(meta)
    r.update({
        "memory": memory,
        "graph_weights": weights,
        "n_requests": n,
        "wall_s": round(wall, 1),
        # B. how the memory helped
        "memory_precision_first": ratio("mem_first_success", "mem_first_attempts"),
        "memory_precision_all": ratio("mem_success", "mem_attempts"),
        "memory_hit_rate": (round(mem_resolved / m["escalations"], 3)
                            if m["escalations"] and memory != "none" else None),
        "wasted_memory_attempts_per_100req": (
            round(100 * (m["mem_attempts"] - m["mem_success"]) / n, 2)
            if memory != "none" else None),
        "memory_precision_by_type": {
            st: {"first_tries": a, "ok": b, "precision": round(b / a, 3) if a else None}
            for st, (a, b) in sorted(orch.mem_first_by_type.items())},
        # C. memory cost
        "llm_reach_per_100req": round(100 * m["reached_llm_block"] / n, 2),
        "llm_asks_per_100req": round(100 * m["llm_asks"] / n, 2),
        "llm_prompt_tokens_per_100req": round(100 * m["llm_prompt_chars"] / 4 / n, 1),
        "fresh_llm_calls": goa.calls - c0,
        "fresh_llm_tokens": goa.tokens - tok0,
        "graph_lookup_ms_mean": (round(float(np.mean(orch.gm.lookup_ms)), 3)
                                 if orch.gm and orch.gm.lookup_ms else None),
        "graph_size": orch.gm.size() if orch.gm else None,
        "decision_trace": dict(orch.trace),
        "paths": {k: v for k, v in m.items() if k.startswith("esc_")},
        "counters": dict(m),
    })
    with open(out_path, "w") as f:
        json.dump(r, f, indent=1, default=str)
    slm.flush()
    goa.flush()
    return r


def make_llms(dry, tag):
    tag = CACHE_PREFIX + tag
    if dry:
        return StubLLM(), StubLLM()
    return (MultiLLM(os.path.join(CACHE_DIR, f"{tag}_slm.json"), role="slm"),
            MultiLLM(os.path.join(CACHE_DIR, f"{tag}_goa.json"), role="goa"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "eval"], required=True)
    ap.add_argument("--workload", choices=list(WORKLOAD), default="wref")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="no LLM; checks the code path only (results go to results/dryrun)")
    ap.add_argument("--force", action="store_true", help="re-run existing results")
    args = ap.parse_args()

    if args.dry_run:
        model = "dry-run (no LLM)"
    else:
        discover_providers()
        if not PROVIDERS:
            sys.exit("No LLM provider configured. Start the local server and set "
                     "LOCAL_LLM_URL (see local_llm/README.md).")
        model = model_name()
    print(f"model: {model}")

    global CACHE_PREFIX
    W = WORKLOAD[args.workload]
    CACHE_PREFIX = W["cache"]
    print(f"workload: {args.workload} | RTT in prompt: {W['rtt_in_prompt']}")
    ctx = setup(W)
    res = W["results"]
    root = os.path.join(res, "dryrun" if args.dry_run else "", args.stage)
    os.makedirs(root, exist_ok=True)
    meta = {"model": model, "workload_set": args.workload,
            "rtt_in_prompt": W["rtt_in_prompt"],
            "digest_rtt_aware": W["digest_rtt_aware"]}

    if args.stage == "tune":
        seeds = args.seeds or DEV_SEEDS
        jobs = [(name, w) for name, w in WEIGHT_SETTINGS.items()]
        total, done = len(jobs) * len(seeds), 0
        summary = {}
        for name, w in jobs:
            slm, goa = make_llms(args.dry_run, f"tune_{name}")
            ars, reach, sla = [], [], []
            for seed in seeds:
                done += 1
                out = os.path.join(root, f"graph_{name}_seed{seed}.json")
                if os.path.exists(out) and not args.force:
                    r = json.load(open(out))
                    print(f"[{done}/{total}] skip (exists): {name} seed {seed}")
                else:
                    r = run_one(ctx, "graph", w, slm, goa,
                                os.path.join(WORKLOADS, W["dev"], f"workload_seed{seed}.jsonl"),
                                out, {**meta, "stage": "tune", "weight_setting": name,
                                      "seed": seed, "workload": W["dev"]})
                    print(f"[{done}/{total}] {name} seed {seed}: "
                          f"AR={r['acceptance_overall']} "
                          f"reach/100={r['llm_reach_per_100req']} "
                          f"SLA={r['sla_violation_rate']} ({r['wall_s']}s)")
                ars.append(r["acceptance_overall"])
                reach.append(r["llm_reach_per_100req"])
                sla.append(r["sla_violation_rate"])
            summary[name] = {"mean_acceptance": round(float(np.mean(ars)), 4),
                             "mean_llm_reach_per_100req": round(float(np.mean(reach)), 2),
                             "mean_sla_violation_rate": round(float(np.mean(sla)), 4)}
        if args.workload == "stress":
            # STUDY_STRESS.md: within 0.005 acceptance of the best, prefer the
            # lower SLA-violation rate, then fewer LLM reaches
            top = max(v["mean_acceptance"] for v in summary.values())
            near = [k for k in summary if summary[k]["mean_acceptance"] >= top - 0.005]
            best = min(near, key=lambda k: (summary[k]["mean_sla_violation_rate"],
                                            summary[k]["mean_llm_reach_per_100req"]))
            criterion = ("within 0.005 of the highest mean acceptance on dev "
                         "seeds: lowest SLA-violation rate, then fewer LLM reaches")
        else:
            best = max(summary, key=lambda k: (summary[k]["mean_acceptance"],
                                               -summary[k]["mean_llm_reach_per_100req"]))
            criterion = ("highest mean acceptance on dev seeds; "
                         "tie-break fewer LLM reaches")
        out = {"chosen": best, "weights": WEIGHT_SETTINGS[best],
               "criterion": criterion,
               "dev_seeds": seeds, "per_setting": summary, "model": model}
        json.dump(out, open(os.path.join(root, "chosen_weights.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        return

    # eval
    cw = os.path.join(res, "dryrun" if args.dry_run else "", "tune",
                      "chosen_weights.json")
    if "graph_memory" in args.systems:
        if os.path.exists(cw):
            weights = json.load(open(cw))["weights"]
        elif args.dry_run:
            weights = WEIGHT_SETTINGS["balanced"]
        else:
            sys.exit(f"{cw} missing - run --stage tune first")
    else:
        weights = None
    seeds = args.seeds if args.seeds is not None else EVAL_SEEDS
    total, done = len(args.systems) * len(seeds), 0
    for system in args.systems:
        memory = SYSTEMS[system]
        slm, goa = make_llms(args.dry_run, f"eval_{system}")
        for seed in seeds:
            done += 1
            out = os.path.join(root, f"{system}_seed{seed}.json")
            if os.path.exists(out) and not args.force:
                print(f"[{done}/{total}] skip (exists): {system} seed {seed}")
                continue
            r = run_one(ctx, memory, weights if memory == "graph" else None, slm, goa,
                        os.path.join(WORKLOADS, W["eval"], f"workload_seed{seed}.jsonl"),
                        out, {**meta, "stage": "eval", "system": system,
                              "seed": seed, "workload": W["eval"]})
            print(f"[{done}/{total}] {system} seed {seed}: "
                  f"AR={r['acceptance_overall']} "
                  f"C={r['acceptance_per_phase'].get('C')} "
                  f"reach/100={r['llm_reach_per_100req']} "
                  f"mem_prec={r['memory_precision_first']} "
                  f"SLA={r['sla_violation_rate']} ({r['wall_s']}s)")
    print("done")


if __name__ == "__main__":
    main()
