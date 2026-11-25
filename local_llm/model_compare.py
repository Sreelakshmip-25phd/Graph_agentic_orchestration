#!/usr/bin/env python3
"""Standalone model-tier comparison harness (Feature 3b).

Compares locally-hosted model tiers (local_llm/models.py) on the two
things this project actually asks a model to do:

  1. SLM/translation quality — replay intent_data/intent_ood_probe.json
     (the existing generalization probe: paraphrased/unseen phrasings of
     known service types) through zoa_agent.slm_system_prompt() and score
     each response against zoa_agent.ground_truth(). No new sentences are
     generated here — reuses the probe file exactly as-is.
  2. GOA/reasoning quality — replay a small, fixed set of representative
     zone-selection (goa_decide_sys) and preemption (PREEMPT_SYS) prompts
     against simple, well-defined heuristics (see REASONING_CASES below).

Also records latency (min/mean/p95) and token usage per model, reusing
MultiLLM's own tracking (latencies_ms / tokens / calls) rather than a
separate stopwatch.

Each target model must already be running its own OpenAI-compatible
server (see start_server.py --also for running two at once). This
script only calls out to servers you tell it about — it doesn't launch
anything itself.

Usage:
    export LOCAL_LLM_URL_SLM=http://127.0.0.1:8080   # small tier
    export LOCAL_LLM_URL_GOA=http://127.0.0.1:8081   # large tier
    python3 model_compare.py \\
        --target llama32_1b@http://127.0.0.1:8080 \\
        --target mistral7b@http://127.0.0.1:8081 \\
        --target qwen_medium@http://127.0.0.1:8082

Each --target is "<label>@<base_url>" — label is free-form (used only
for the report), so any downloaded tier/server combination can be
compared without editing this file. Outputs:
    harness_final/model_comparison.csv
    harness_final/model_comparison_figure.png
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SRC = os.path.join(BASE, "src")
sys.path.insert(0, SRC)

from config import HARN, INTENT_DATA, SCEN  # noqa: E402
from goa_agent import PREEMPT_SYS, goa_decide_sys  # noqa: E402
from llm_client import MultiLLM  # noqa: E402
from zoa_agent import build_schema, ground_truth, slm_system_prompt  # noqa: E402

from models import MODELS  # noqa: E402


def display_name(label):
    """Human-readable "Name (XB)" for a --target label that matches a
    models.py tier key (e.g. "medium" -> "Qwen2.5-7B-Instruct (7B)");
    falls back to the raw label unchanged for a custom/external label
    that isn't one of this project's known tiers."""
    spec = MODELS.get(label)
    if not spec:
        return label
    params = spec["params_b"]
    params_str = f"{params:g}B"
    return f"{spec['display_name']} ({params_str})"


def params_b(label):
    """Sort key: known tier's parameter count, or +inf (sorts last,
    after every known tier) for a custom label model_compare.py has no
    size info for."""
    spec = MODELS.get(label)
    return spec["params_b"] if spec else float("inf")

FIELDS = ["service_type", "latency_class", "data_locality", "priority"]

# --- representative GOA decision prompts (fixed, documented set) --------
# Kept intentionally small: these are meant to be read and understood,
# not exhaustive. Each case is checked against a well-defined heuristic
# rather than an exact-match ground truth, since GOA's job is reasoning
# under ambiguity, not recall.
ZONE_SELECTION_CASES = [
    {
        "name": "clear_winner",
        "request": {"cpu_req": 4.0, "mem_gb": 2.0, "latency_ms": 300,
                    "priority": "normal", "origin_zone": "z1"},
        "profile": {"service_type": "traffic_monitor",
                    "latency_class": "interactive",
                    "data_locality": "any", "priority": "normal"},
        "zone_digests": {
            "z1": {"zone": "z1", "top_cpu_free": 1.0, "t": 0.0},
            "z2": {"zone": "z2", "top_cpu_free": 30.0, "t": 0.0},
            "z3": {"zone": "z3", "top_cpu_free": 2.0, "t": 0.0}},
        # heuristic: the only zone that can actually fit the request
        "acceptable_zones": {"z2"},
    },
    {
        "name": "two_viable",
        "request": {"cpu_req": 5.0, "mem_gb": 2.0, "latency_ms": 500,
                    "priority": "high", "origin_zone": "z1"},
        "profile": {"service_type": "video_analytics",
                    "latency_class": "interactive",
                    "data_locality": "any", "priority": "high"},
        "zone_digests": {
            "z1": {"zone": "z1", "top_cpu_free": 0.5, "t": 0.0},
            "z2": {"zone": "z2", "top_cpu_free": 20.0, "t": 0.0},
            "z3": {"zone": "z3", "top_cpu_free": 18.0, "t": 0.0}},
        # heuristic: either zone that can fit is a justified pick
        "acceptable_zones": {"z2", "z3"},
    },
]

PREEMPT_CASES = [
    {
        "name": "justified_eviction",
        "incoming": {"priority": "critical", "service_type": "video_analytics"},
        # sorted by remaining_s ascending, as the deterministic fallback
        # in goa_agent._try_preempt does
        "candidates": [
            {"req_id": "r1", "priority": "low", "remaining_s": 5.0,
             "service_type": "batch_job"},
            {"req_id": "r2", "priority": "normal", "remaining_s": 120.0,
             "service_type": "traffic_monitor"},
        ],
        # heuristic: any candidate is a "justified" pick (both are
        # strictly lower priority than the incoming critical request);
        # the deterministic fallback (nearest-to-completion) is r1.
        "acceptable_ids": {"r1", "r2"},
        "preferred_id": "r1",
    },
    {
        "name": "no_good_candidate",
        "incoming": {"priority": "high", "service_type": "traffic_monitor"},
        "candidates": [
            {"req_id": "r3", "priority": "normal", "remaining_s": 2.0,
             "service_type": "video_analytics"},
        ],
        # still technically evictable (normal < high) - a null response
        # here is overly conservative but not wrong; r3 is the only
        # justified pick.
        "acceptable_ids": {"r3", None},
        "preferred_id": "r3",
    },
]


def load_probe():
    p = os.path.join(INTENT_DATA, "intent_ood_probe.json")
    return json.load(open(p))


def scoped_multillm(base_url, cache_path, no_system_role=False):
    """A MultiLLM instance whose provider list is exactly one server —
    lets model_compare.py score models one at a time without touching
    the shared discover_providers()/PROVIDERS machinery evaluator.py
    uses for the real pipeline.

    no_system_role: set True for a model whose chat template has no
    system role (confirmed for Gemma via llama-cpp-python's
    format_gemma(), which hardcodes system_message="" and silently
    drops it - see test_prompt_variant_d-style diagnostics). MultiLLM
    then folds system+user into one user turn instead of a system
    message the model would never actually see."""
    import llm_client
    llm_client.PROVIDERS.clear()
    llm_client.PROVIDERS.append({
        "key": "local", "roles": None, "local": True,
        "no_system_role": no_system_role,
        "url": f"{base_url.rstrip('/')}/v1/chat/completions",
        "models": ["local"], "name": "compare_target",
    })
    return MultiLLM(cache_path, role="compare")


def score_translation(model, catalog, intent_schema, probe):
    sys_prompt = slm_system_prompt(intent_schema)
    field_correct = {f: 0 for f in FIELDS}
    exact = 0
    n = 0
    for item in probe:
        text, stype = item["text"], item["service_type"]
        if stype not in catalog:
            continue
        expected = ground_truth(catalog, stype)
        try:
            out = model.ask(sys_prompt, "INPUT: " + text,
                            cache_key=f"cmp|slm|{text}")
        except Exception as e:
            print(f"    [translation] {text[:40]!r}... failed: {e}")
            out = {}
        n += 1
        ok_all = True
        for f in FIELDS:
            if out.get(f) == expected[f]:
                field_correct[f] += 1
            else:
                ok_all = False
        if ok_all:
            exact += 1
    return {
        "n": n,
        "field_accuracy": {f: round(field_correct[f] / n, 3) if n else 0
                          for f in FIELDS},
        "translation_accuracy": round(
            sum(field_correct.values()) / (n * len(FIELDS)), 3) if n else 0,
        "full_schema_exact_match": round(exact / n, 3) if n else 0,
    }


def score_reasoning(model, semantic):
    goa_sys = goa_decide_sys(semantic)
    total = justified = 0

    for case in ZONE_SELECTION_CASES:
        user = json.dumps({"request": case["request"],
                           "profile": case["profile"],
                           "zone_digests": case["zone_digests"]})
        total += 1
        try:
            out = model.ask(goa_sys, user,
                            cache_key=f"cmp|zone|{case['name']}")
            tz = out.get("target_zone")
        except Exception as e:
            print(f"    [reasoning:{case['name']}] failed: {e}")
            tz = None
        if tz in case["acceptable_zones"]:
            justified += 1
        else:
            print(f"    [zone:{case['name']}] picked {tz!r}, "
                  f"acceptable={case['acceptable_zones']}")

    for case in PREEMPT_CASES:
        user = json.dumps({"incoming_request": case["incoming"],
                           "eviction_candidates": case["candidates"]})
        total += 1
        try:
            out = model.ask(PREEMPT_SYS, user,
                            cache_key=f"cmp|preempt|{case['name']}")
            evict_id = out.get("evict_req_id")
        except Exception as e:
            print(f"    [reasoning:{case['name']}] failed: {e}")
            evict_id = "<error>"
        valid_ids = {c["req_id"] for c in case["candidates"]}
        # explicitly flag an out-of-list id (a real failure mode - the
        # model inventing a req_id that was never offered as a candidate)
        out_of_list = evict_id not in valid_ids and evict_id is not None \
            and evict_id != "<error>"
        if out_of_list:
            print(f"    [preempt:{case['name']}] OUT-OF-LIST evict_req_id "
                  f"{evict_id!r} (valid: {valid_ids})")
        if evict_id in case["acceptable_ids"]:
            justified += 1

    return {"n": total, "reasoning_score": round(justified / total, 3)
                                          if total else 0}


def summarize_latency_tokens(model):
    import numpy as np
    lat = model.latencies_ms
    return {
        "calls": model.calls,
        "cache_hits": model.cache_hits,
        "min_latency_ms": round(float(np.min(lat)), 1) if lat else 0,
        "mean_latency_ms": round(float(np.mean(lat)), 1) if lat else 0,
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 1) if lat else 0,
        "mean_tokens": round(model.tokens / model.calls, 1) if model.calls else 0,
    }


def run_target(label, url, catalog, intent_schema, semantic, probe):
    no_sys = MODELS.get(label, {}).get("no_system_role", False)
    print(f"\n=== {label} ({url}) ==="
         + (" [no system role - folding into user turn]" if no_sys else ""))
    cache_path = os.path.join(HERE, f"compare_cache_{label}.json")
    model = scoped_multillm(url, cache_path, no_system_role=no_sys)
    t0 = time.time()
    translation = score_translation(model, catalog, intent_schema, probe)
    reasoning = score_reasoning(model, semantic)
    model.flush()
    wall = time.time() - t0
    row = {"model": label, "url": url,
          "translation_accuracy": translation["translation_accuracy"],
          "full_schema_exact_match": translation["full_schema_exact_match"],
          "reasoning_score": reasoning["reasoning_score"],
          **summarize_latency_tokens(model),
          "wall_s": round(wall, 1)}
    print(f"  translation_accuracy={row['translation_accuracy']} "
          f"exact_match={row['full_schema_exact_match']} "
          f"reasoning_score={row['reasoning_score']} "
          f"mean_latency_ms={row['mean_latency_ms']} ({wall:.0f}s)")
    return row, translation


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r.get("display_name", r["model"]) for r in rows]
    x = np.arange(len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Local model tier comparison: translation/reasoning "
                 "quality vs. latency", fontsize=13, fontweight="bold")

    ax = axes[0]
    w = 0.25
    ax.bar(x - w, [r["translation_accuracy"] for r in rows], w,
          label="Translation accuracy (per-field)", color=colors, edgecolor="black", lw=0.4)
    ax.bar(x, [r["full_schema_exact_match"] for r in rows], w,
          label="Full-schema exact match", color=colors, alpha=0.6,
          hatch="//", edgecolor="black", lw=0.4)
    ax.bar(x + w, [r["reasoning_score"] for r in rows], w,
          label="Reasoning score", color=colors, alpha=0.35,
          hatch="xx", edgecolor="black", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Quality"); ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(x, [r["mean_latency_ms"] for r in rows], 0.5, color=colors,
          edgecolor="black", lw=0.4, yerr=[
              max(r["p95_latency_ms"] - r["mean_latency_ms"], 0) for r in rows],
          capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Mean latency, ms (error bar to p95)")
    ax.set_title("Latency")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(HARN, "model_comparison_figure.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("saved:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", default=[],
                    help="label@base_url, e.g. "
                         "llama32_1b@http://127.0.0.1:8080 — repeatable. "
                         "Each target must already be running its own "
                         "OpenAI-compatible server.")
    ap.add_argument("--drop", action="append", default=[],
                    help="Model label to remove from the merged "
                         "comparison (e.g. a model that turned out not "
                         "to comply with the JSON-only prompts and isn't "
                         "worth keeping in the table/figure) - repeatable. "
                         "Applied after merging with any existing CSV and "
                         "before adding this run's --target results.")
    ap.add_argument("--fresh", action="store_true",
                    help="Start model_comparison.csv over instead of "
                         "merging into it. Default behavior appends/"
                         "updates rows so you can compare more models "
                         "than your hardware can run concurrently by "
                         "invoking this script once per batch of "
                         "servers you actually start.")
    args = ap.parse_args()
    if not args.target and not args.drop:
        raise SystemExit("nothing to do: pass --target and/or --drop")

    catalog = json.load(open(os.path.join(SCEN, "service_catalog.json")))["services"]
    intent_schema, required_fields, semantic = build_schema(catalog)
    probe = load_probe()
    print(f"probe: {len(probe)} OOD sentences, "
          f"{len(ZONE_SELECTION_CASES)} zone-selection + "
          f"{len(PREEMPT_CASES)} preemption reasoning cases")

    rows = []
    for spec in args.target:
        if "@" not in spec:
            raise SystemExit(f"--target must be label@url, got: {spec!r}")
        label, url = spec.split("@", 1)
        row, _ = run_target(label, url, catalog, intent_schema, semantic, probe)
        row["display_name"] = display_name(label)
        row["params_b"] = params_b(label)
        rows.append(row)

    import pandas as pd
    out_csv = os.path.join(HARN, "model_comparison.csv")
    new_by_label = {r["model"]: r for r in rows}
    merged = list(new_by_label.values())
    if not args.fresh and os.path.exists(out_csv):
        prior = pd.read_csv(out_csv).to_dict("records")
        # keep prior rows for labels this run didn't re-test; a label
        # tested again this run replaces its old row (last write wins,
        # same convention as the per-model on-disk ask() cache).
        merged = [r for r in prior if r["model"] not in new_by_label] + merged
        print(f"merging with {len(prior)} row(s) from existing {out_csv} "
              f"(pass --fresh to start over instead)")

    if args.drop:
        before = len(merged)
        merged = [r for r in merged if r["model"] not in args.drop]
        dropped = before - len(merged)
        print(f"dropped {dropped} row(s) for: {', '.join(args.drop)}"
              + ("" if dropped == len(args.drop) else
                 f" (note: {len(args.drop) - dropped} label(s) in --drop "
                 f"weren't present)"))

    if not merged:
        raise SystemExit("no rows left after --drop and no --target given "
                         "- nothing to write")

    # Backfill display_name/params_b for rows loaded from a CSV written
    # before this feature existed, and sort smallest -> largest so the
    # table/figure always read as a size progression rather than
    # whatever order --target happened to be invoked in.
    for r in merged:
        if not r.get("display_name"):
            r["display_name"] = display_name(r["model"])
        if r.get("params_b") is None or (isinstance(r.get("params_b"), float)
                                         and pd.isna(r["params_b"])):
            r["params_b"] = params_b(r["model"])
    merged.sort(key=lambda r: r["params_b"])

    df = pd.DataFrame(merged).set_index("model")
    df.to_csv(out_csv)
    print("\n" + df.to_string())
    print("saved:", out_csv)
    make_figure(merged)


if __name__ == "__main__":
    main()
