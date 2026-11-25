#!/usr/bin/env python3
"""Analyse a study run (W-ref: results/, Stress: results_stress/).

  results/reproduction_check.csv  flat memory seeds 0-2 vs the earlier study
  results/summary.csv             system x metric, mean and 95% CI over seeds
  results/paired_stats.csv        paired differences per seed, Wilcoxon, Holm
  results/gates.json              the pre-registered checks in STUDY.md
  results/fig1_paired_differences.png
  results/fig2_acceptance_over_time.png
  results/fig3_decision_trace.png

Stress (--workload stress) writes the same files into results_stress/,
with the Stress metrics and gates S1-S8 (STUDY_STRESS.md) instead of
G1-G5, plus path_breakdown.csv.

Usage: python src/analyze_study.py [--workload wref|stress] [--dry-run]
"""
import argparse
import csv
import glob
import json
import os
import sys
import warnings

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REFERENCE, RESULTS, SCEN

SYSTEMS = ["graph_memory", "flat_memory", "no_memory"]
LABEL = {"graph_memory": "Graph memory", "flat_memory": "Flat memory",
         "no_memory": "No memory"}
COLOR = {"graph_memory": "#2a78d6", "flat_memory": "#eb6834", "no_memory": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# (key, label, group, higher_is_better)
METRICS = [
    ("acceptance_overall", "Acceptance (overall)", "A", True),
    ("ar_A", "Acceptance, phase A", "A", True),
    ("ar_B", "Acceptance, phase B", "A", True),
    ("ar_C", "Acceptance, phase C", "A", True),
    ("sla_violation_rate", "SLA-violation rate", "A", False),
    ("replan_success_rate", "Replan success rate", "A", True),
    ("displaced_final", "Displaced services", "A", False),
    ("memory_precision_first", "Memory precision (first try)", "B", True),
    ("memory_hit_rate", "Memory hit rate", "B", True),
    ("preemption_regret", "Preemption regret", "B", False),
    ("ar_drop_after_failure", "Acceptance drop after failure", "B", False),
    ("llm_reach_per_100req", "LLM reach / 100 req", "C", False),
    ("llm_asks_per_100req", "LLM consultations / 100 req", "C", False),
    ("llm_prompt_tokens_per_100req", "LLM prompt tokens / 100 req", "C", False),
    ("decision_ms_mean", "Decision latency mean (ms)", "C", False),
    ("decision_ms_p95", "Decision latency p95 (ms)", "C", False),
    ("graph_lookup_ms_mean", "Graph lookup (ms)", "C", False),
]
# Stress: W-ref metrics, the failure metric replaced by the per-event one,
# plus the STUDY_STRESS.md metrics
STRESS_METRICS = [m for m in METRICS if m[0] != "ar_drop_after_failure"] + [
    ("sla_violation_rate_all_placements", "SLA-violation rate, incl. replans", "A", False),
    ("sla_tight_types", "SLA-violation rate, tight new types", "A", False),
    ("wasted_memory_attempts_per_100req", "Wasted memory attempts / 100 req", "B", False),
    ("ar_drop_failures", "Acceptance drop after failures (mean)", "B", False),
    ("ar_drop_zone_wide", "Acceptance drop after zone-wide failures", "B", False),
    ("ar_change_hotspot", "Acceptance change after hotspot moves", "B", True),
    ("prec_similar_types", "Memory precision, similar new types", "B", True),
    ("prec_control_type", "Memory precision, control type", "B", True),
]
PATHS = ["local", "graph", "episodic", "procedural", "digest", "llm",
         "preempt", "preempt_local", "llm_degraded", "llm_unresolved"]
GROUP = {"A": "A. Task outcome", "B": "B. How memory helped", "C": "C. Memory cost"}
COMPARISONS = [("graph_memory", "flat_memory"), ("graph_memory", "no_memory"),
               ("flat_memory", "no_memory")]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def flatten(r, metrics=METRICS, rep=None):
    out = {k: r.get(k) for k, *_ in metrics}
    for ph in "ABC":
        out[f"ar_{ph}"] = r["acceptance_per_phase"].get(ph)
    ad = r.get("adaptation_after_failure") or []
    out["ar_drop_after_failure"] = _mean([a.get("ar_drop") for a in ad])
    if rep is None:
        return out
    # Stress metrics (rep = scenario/stress/stress_report.json)
    sbt = r.get("sla_by_type", {})
    pl = sum(sbt.get(t, {}).get("placements", 0) for t in rep["tight_types"])
    vi = sum(sbt.get(t, {}).get("violations", 0) for t in rep["tight_types"])
    out["sla_tight_types"] = vi / pl if pl else None
    fe = r.get("failure_events", [])
    out["ar_drop_failures"] = _mean([e["ar_drop"] for e in fe])
    out["ar_drop_zone_wide"] = _mean([e["ar_drop"] for e in fe if e["zone_wide"]])
    out["ar_change_hotspot"] = _mean([h["ar_change"] for h in r.get("hotspot_moves", [])])
    fw = r.get("first_window_acceptance_by_type", {})
    out["new_similar_first30"] = _mean([fw.get(t, {}).get("acceptance")
                                        for t in rep["similar_types"]])
    out["new_control_first30"] = fw.get(rep["control_type"], {}).get("acceptance")
    pbt = r.get("memory_precision_by_type", {})
    tries = sum(pbt.get(t, {}).get("first_tries", 0) for t in rep["similar_types"])
    ok = sum(pbt.get(t, {}).get("ok", 0) for t in rep["similar_types"])
    out["prec_similar_types"] = ok / tries if tries else None
    c = pbt.get(rep["control_type"], {})
    out["prec_control_type"] = (c["ok"] / c["first_tries"]
                                if c.get("first_tries") else None)
    return out


def load(root):
    runs = {}
    for p in glob.glob(os.path.join(root, "*_seed*.json")):
        r = json.load(open(p))
        runs[(r["system"], r["seed"])] = r
    return runs


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else None, None
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(h)


def holm(pvals):
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    adj, running = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvals[i]))
        adj[i] = running
    return adj


def fmt(x, d=4):
    return None if x is None else round(x, d)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workload", choices=["wref", "stress"], default="wref")
    args = ap.parse_args()
    stress = args.workload == "stress"
    res = RESULTS + "_stress" if stress else RESULTS
    base = os.path.join(res, "dryrun") if args.dry_run else res
    runs = load(os.path.join(base, "eval"))
    if not runs:
        sys.exit(f"no results in {base}/eval")
    seeds = sorted({s for _, s in runs})
    global METRICS
    srep = None
    if stress:
        srep = json.load(open(os.path.join(SCEN, "stress", "stress_report.json")))
        srep["similar_types"] = list(srep["similar_types"])
        METRICS = STRESS_METRICS
    flat = {k: flatten(v, METRICS, srep) for k, v in runs.items()}
    model = sorted({v.get("model") for v in runs.values()})
    print(f"runs: {len(runs)} | seeds: {seeds} | model: {model}")

    # ---------------------------------------------- reproduction check
    rep_rows = []
    for s in ((0, 1, 2) if not stress else ()):
        ref_p = os.path.join(REFERENCE, f"flat_seed{s}.json")
        if ("flat_memory", s) not in runs or not os.path.exists(ref_p):
            continue
        ref, new = json.load(open(ref_p)), runs[("flat_memory", s)]
        for k in ("acceptance_overall", "llm_reach_per_100req",
                  "migrations", "replan_success_rate"):
            rep_rows.append({"seed": s, "metric": k, "earlier_study": ref.get(k),
                             "this_run": new.get(k)})
        # The earlier study's "preemptions" counted eviction events. Results
        # written before both counts existed have "preemptions" = distinct
        # services and no event count (the event log is not saved).
        events = new.get("preemption_events")
        services = new.get("preempted_services", new.get("preemptions")
                           if events is None else None)
        rep_rows += [
            {"seed": s, "metric": "preemption_events",
             "earlier_study": ref.get("preemptions"), "this_run": events},
            {"seed": s, "metric": "preempted_services",
             "earlier_study": None, "this_run": services},
            {"seed": s, "metric": "goa_preemption_decisions",
             "earlier_study": None,
             "this_run": new.get("counters", {}).get("preemptions")}]
    rep_ok = None
    if rep_rows:
        d_ar = [abs(r["this_run"] - r["earlier_study"]) for r in rep_rows
                if r["metric"] == "acceptance_overall"]
        d_re = [abs(r["this_run"] - r["earlier_study"]) for r in rep_rows
                if r["metric"] == "llm_reach_per_100req"]
        rep_ok = max(d_ar) <= 0.02 and max(d_re) <= 5.0
        with open(os.path.join(base, "reproduction_check.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rep_rows[0]))
            w.writeheader()
            w.writerows(rep_rows)

    # ---------------------------------------------------------- summary
    with open(os.path.join(base, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "metric", "system", "n_seeds", "mean", "ci95"])
        for k, label, g, _ in METRICS:
            for sysn in SYSTEMS:
                v = [flat[(sysn, s)][k] for s in seeds
                     if (sysn, s) in flat and flat[(sysn, s)][k] is not None]
                if not v:
                    continue
                m, h = ci95(v)
                w.writerow([GROUP[g], label, sysn, len(v), fmt(m), fmt(h)])

    # ----------------------------------------------------- paired stats
    paired = {}
    rows = []
    for a, b in COMPARISONS:
        res = []
        for k, label, g, hib in METRICS:
            d = [flat[(a, s)][k] - flat[(b, s)][k] for s in seeds
                 if (a, s) in flat and (b, s) in flat
                 and flat[(a, s)][k] is not None and flat[(b, s)][k] is not None]
            if not d:
                continue
            m, h = ci95(d)
            if len(d) >= 5 and any(x != 0 for x in d):
                p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
            else:
                p = None
            res.append({"comparison": f"{a} - {b}", "group": GROUP[g], "metric": label,
                        "key": k, "n_pairs": len(d), "mean_diff": m, "ci95": h,
                        "p_wilcoxon": p, "higher_is_better": hib})
        for r, ph in zip(res, holm([r["p_wilcoxon"] for r in res])):
            r["p_holm"] = ph
            r["significant"] = ph is not None and ph < 0.05
            r["favours"] = (None if not r["significant"] else
                            r["comparison"].split(" - ")[0]
                            if (r["mean_diff"] > 0) == r["higher_is_better"]
                            else r["comparison"].split(" - ")[1])
        paired[(a, b)] = {r["key"]: r for r in res}
        rows += res
    with open(os.path.join(base, "paired_stats.csv"), "w", newline="") as f:
        cols = ["comparison", "group", "metric", "n_pairs", "mean_diff", "ci95",
                "p_wilcoxon", "p_holm", "significant", "favours"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: fmt(r[k]) for k in
                                ("mean_diff", "ci95", "p_wilcoxon", "p_holm")}})

    # ------------------------------------------------------------ gates
    def pr(a, b, k):
        return paired.get((a, b), {}).get(k)

    def mean(sysn, k):
        v = [flat[(sysn, s)][k] for s in seeds if (sysn, s) in flat
             and flat[(sysn, s)][k] is not None]
        return float(np.mean(v)) if v else None

    if stress:
        g = stress_gates(pr, mean)
        path_breakdown(base, runs, seeds)
        json.dump({"model": model, "seeds": seeds, "gates": g,
                   "scipy": __import__("scipy").__version__,
                   "definitions": "see STUDY_STRESS.md"},
                  open(os.path.join(base, "gates.json"), "w"), indent=1)
        print(json.dumps(g, indent=1))
        figures(base, runs, flat, seeds, paired)
        print(f"outputs written to {base}")
        return

    g = {}
    g["G1_flat_reproduces_earlier_study"] = rep_ok
    r = pr("graph_memory", "flat_memory", "acceptance_overall")
    g["G2_graph_acceptance_not_worse_than_flat"] = (
        None if r is None or r["ci95"] is None else r["mean_diff"] - r["ci95"] > -0.01)
    r = pr("graph_memory", "flat_memory", "memory_precision_first")
    g["G3_graph_memory_precision_higher"] = (
        None if r is None else bool(r["significant"] and r["mean_diff"] > 0))
    ok = []
    for a in ("graph_memory", "flat_memory"):
        r = pr(a, "no_memory", "llm_reach_per_100req")
        ok.append(None if r is None else bool(r["significant"] and r["mean_diff"] < 0))
    g["G4_memory_reduces_llm_reach"] = None if None in ok else all(ok)
    tg, tf = mean("graph_memory", "llm_prompt_tokens_per_100req"), \
        mean("flat_memory", "llm_prompt_tokens_per_100req")
    g["G5_graph_prompt_overhead_at_most_25pct"] = (
        None if not tg or not tf else tg / tf - 1 <= 0.25)
    r = pr("graph_memory", "flat_memory", "acceptance_overall")
    g["secondary_graph_acceptance_higher"] = (
        None if r is None else bool(r["significant"] and r["mean_diff"] > 0))
    json.dump({"model": model, "seeds": seeds, "gates": g,
               "definitions": "see STUDY.md"},
              open(os.path.join(base, "gates.json"), "w"), indent=1)
    print(json.dumps(g, indent=1))

    figures(base, runs, flat, seeds, paired)
    print(f"outputs written to {base}")


def stress_gates(pr, mean):
    """STUDY_STRESS.md, pre-registered checks S1-S8."""
    def sig(a, b, k, sign):
        r = pr(a, b, k)
        return None if r is None else bool(r["significant"] and
                                           np.sign(r["mean_diff"]) == sign)
    g = {}
    g["S1_graph_fewer_sla_violations_than_flat"] = sig(
        "graph_memory", "flat_memory", "sla_violation_rate", -1)
    r = pr("graph_memory", "flat_memory", "acceptance_overall")
    g["S2_graph_acceptance_not_worse_than_flat"] = (
        None if r is None or r["ci95"] is None else r["mean_diff"] - r["ci95"] > -0.01)
    g["S3_graph_acceptance_higher"] = sig("graph_memory", "flat_memory",
                                          "acceptance_overall", 1)
    g["S4_graph_fewer_wasted_attempts"] = sig("graph_memory", "flat_memory",
                                              "wasted_memory_attempts_per_100req", -1)
    g["S5_graph_smaller_drop_after_failures"] = sig("graph_memory", "flat_memory",
                                                    "ar_drop_failures", -1)
    # the control type rarely escalates, so its precision is reported in
    # summary.csv when it exists, not tested (STUDY_STRESS.md, S6)
    g["S6_graph_more_precise_on_similar_new_types"] = sig(
        "graph_memory", "flat_memory", "prec_similar_types", 1)
    ok = [sig(a, "no_memory", "llm_reach_per_100req", -1)
          for a in ("graph_memory", "flat_memory")]
    g["S7_memory_reduces_llm_reach"] = None if None in ok else all(ok)
    tg, tf = mean("graph_memory", "llm_prompt_tokens_per_100req"), \
        mean("flat_memory", "llm_prompt_tokens_per_100req")
    g["S8_graph_prompt_overhead_at_most_25pct"] = (
        None if not tg or not tf else tg / tf - 1 <= 0.25)
    return g


def path_breakdown(base, runs, seeds):
    """B9: where escalations were resolved, per 100 requests, per system."""
    with open(os.path.join(base, "path_breakdown.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "path", "per_100_requests_mean", "sla_violations_mean"])
        for sysn in SYSTEMS:
            rs = [runs[(sysn, s)] for s in seeds if (sysn, s) in runs]
            if not rs:
                continue
            for p in PATHS:
                v = [100 * r["paths"].get(f"esc_{p}", 0) / r["n_requests"] for r in rs]
                sv = [r.get("sla_violations_by_path", {}).get(p, {}).get("violations", 0)
                      for r in rs]
                if any(v) or any(sv):
                    w.writerow([sysn, p, round(float(np.mean(v)), 2),
                                round(float(np.mean(sv)), 2)])


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def figures(base, runs, flat, seeds, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- fig 1: paired differences, one small panel per metric
    keys = [m for m in METRICS
            if any(k in paired.get(c, {}) for c in COMPARISONS[:2] for k in [m[0]])]
    fig, axes = plt.subplots(len(keys), 1, figsize=(7, 0.62 * len(keys) + 1.2))
    axes = np.atleast_1d(axes)
    marks = [(("graph_memory", "flat_memory"), "o", "Graph − Flat"),
             (("graph_memory", "no_memory"), "s", "Graph − None")]
    for ax, (k, label, g, hib) in zip(axes, keys):
        _style(ax)
        ax.axvline(0, color=INK2, lw=0.8)
        for yi, (cmp_, mk, _) in enumerate(marks):
            r = paired.get(cmp_, {}).get(k)
            if not r:
                continue
            y = 0.25 - 0.5 * yi
            ax.errorbar(r["mean_diff"], y, xerr=r["ci95"] or 0, fmt=mk,
                        color=COLOR["graph_memory"], ms=6, lw=1.6, capsize=0,
                        mfc=COLOR["graph_memory"] if mk == "o" else "white")
            if r["significant"]:
                ax.annotate("*", (r["mean_diff"], y), xytext=(6, -3),
                            textcoords="offset points", color=INK, fontsize=10)
        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([])
        ax.set_ylabel(f"{label}\n({'higher' if hib else 'lower'} is better)",
                      rotation=0, ha="right", va="center", fontsize=8, color=INK)
    handles = [plt.Line2D([], [], marker=mk, ls="", color=COLOR["graph_memory"],
                          mfc=COLOR["graph_memory"] if mk == "o" else "white",
                          label=lab) for _, mk, lab in marks]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.985),
               frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Paired differences per seed (mean, 95% CI); * = Holm-adjusted p < 0.05",
                 fontsize=9, color=INK, x=0.02, y=0.998, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(base, "fig1_paired_differences.png"), dpi=200)
    plt.close(fig)

    # ---- fig 2: acceptance over time
    fig, ax = plt.subplots(figsize=(7, 3.2))
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sysn in SYSTEMS:
        series = [runs[(sysn, s)]["acceptance_over_time"] for s in seeds
                  if (sysn, s) in runs]
        if not series:
            continue
        nb = min(len(x) for x in series)
        t = [series[0][i]["t"] / 3600 for i in range(nb)]
        y = [np.nanmean([x[i]["acceptance"] if x[i]["acceptance"] is not None
                         else np.nan for x in series]) for i in range(nb)]
        ax.plot(t, y, color=COLOR[sysn], lw=2, label=LABEL[sysn])
    for tb in (1, 2):
        ax.axvline(tb, color=INK2, lw=0.6, ls=":")
    r0 = next(iter(runs.values()))
    if r0.get("failure_events") and len(r0["failure_events"]) > 1:
        for e in r0["failure_events"]:
            x = e["t_fail"] / 3600
            ax.axvline(x, color=INK2, lw=1.0 if e["zone_wide"] else 0.6, ls="--")
            ax.text(x, 0.02, f" {e['zone']}" + (" zone" if e["zone_wide"] else ""),
                    color=INK2, fontsize=6, transform=ax.get_xaxis_transform())
        for h in r0.get("hotspot_moves", []):
            ax.text(h["t"] / 3600, 0.93, f"→{h['zone']}", color=INK2, fontsize=6,
                    ha="center", transform=ax.get_xaxis_transform())
    else:
        tf = [a["t_fail"] / 3600 for r in runs.values()
              for a in r.get("adaptation_after_failure", [])]
        if tf:
            ax.axvline(np.mean(tf), color=INK2, lw=0.8, ls="--")
            ax.text(np.mean(tf), 0.02, " node failure", color=INK2, fontsize=7,
                    transform=ax.get_xaxis_transform())
    for i, ph in enumerate("ABC"):
        ax.text(i + 0.5, 1.01, f"phase {ph}", ha="center", color=INK2, fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 3)
    ax.set_xlabel("Arrival time (h)", fontsize=8, color=INK2)
    ax.set_ylabel("Acceptance (5-min bins,\nmean over seeds)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(base, "fig2_acceptance_over_time.png"), dpi=200)
    plt.close(fig)

    # ---- fig 3: decision trace (graph memory)
    tr = [runs[("graph_memory", s)].get("decision_trace", {}) for s in seeds
          if ("graph_memory", s) in runs]
    if tr:
        cats = [("basis_direct", "Direct route (same type, same origin)"),
                ("basis_nearby_origin", "Same type, other origins"),
                ("basis_similar_type", "Similar service type"),
                ("basis_rule", "Learned rule"),
                ("basis_capacity", "Capacity only"),
                ("rtt_penalty_changed_choice", "RTT penalty changed the choice"),
                ("fail_penalty_changed_choice", "Failure penalty changed the choice")]
        vals = [np.mean([d.get(k, 0) for d in tr]) for k, _ in cats]
        fig, ax = plt.subplots(figsize=(7, 2.8))
        _style(ax)
        y = np.arange(len(cats))[::-1]
        ax.barh(y, vals, color=COLOR["graph_memory"], height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.1f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([c[1] for c in cats], fontsize=8, color=INK)
        ax.axhline(1.5, color=GRID, lw=0.8)
        ax.set_xlabel("Graph-memory placements per run (mean over seeds)",
                      fontsize=8, color=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(base, "fig3_decision_trace.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
