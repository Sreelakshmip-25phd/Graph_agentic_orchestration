#!/usr/bin/env python3
"""Generate the W-ref (reference) workloads.

W-ref is the workload the earlier flat-memory study was evaluated on.
Seeds 0-2 are copied unchanged from that study. This script re-creates
them with the original generator (phase1_scenario.py) and the original
lifetime calibration factor K, checks they are identical to the copied
files, and then generates:

  - evaluation seeds 3-9  -> workloads/wref/workload_seed{n}.jsonl
  - dev seeds 100-104     -> workloads/dev/workload_seed{n}.jsonl
    (used only to tune graph-memory weights; never reported)

Nothing about the generator is changed: same topology, catalog, traces,
phase mix, failure model and K, only different random seeds.

Run: python src/make_wref_workloads.py [--check-only]
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_scenario as p1
from config import DEV_SEEDS, EVAL_SEEDS, SCEN, WORKLOADS


def _context():
    """Everything phase1_scenario.main() derives before its seed loop."""
    grid = pd.read_csv(p1.need("milan_grid_centroids.csv"))
    act = pd.read_parquet(p1.need("milan_activity.parquet"))
    surge = pd.read_csv(p1.need("milan_surge_candidates.csv"))
    prof = json.load(open(p1.need("service_resource_profiles.json")))
    fail = json.load(open(p1.need("failure_model.json")))
    rttm = json.load(open(p1.need("rtt_model.json")))
    act["ts"] = pd.to_datetime(act["ts"])
    cell_w = act.groupby("cell_id")["activity"].mean().rename("w").reset_index()
    cells = grid.merge(cell_w, on="cell_id", how="left").fillna({"w": 0.0})

    # build_topology() writes its JSON into p1.SCEN; point it at a temp
    # dir so the committed topology is never overwritten, then check the
    # rebuilt topology matches the committed one.
    with tempfile.TemporaryDirectory() as tmp:
        p1.SCEN = tmp
        topo = p1.build_topology(cells, rttm, 10, "eval",
                                 servers_total=14, nodes_total=36)
    ref = json.load(open(os.path.join(SCEN, "topology_eval.json")))
    assert json.dumps(topo, sort_keys=True) == json.dumps(ref, sort_keys=True), \
        "rebuilt topology differs from scenario/topology_eval.json"

    catalog = json.load(open(os.path.join(SCEN, "service_catalog.json")))["services"]
    surge_day = surge.day.mode().iat[0]
    normal_days = sorted(d for d in act.day.unique() if d != surge_day)
    base_day = normal_days[len(normal_days) // 2]
    actz = act.merge(cells[["cell_id", "zone_eval"]], on="cell_id")

    def slot_zone_matrix(day):
        d = actz[actz.day == day]
        return (d.pivot_table(index="ts", columns="zone_eval", values="activity",
                              aggfunc="sum").fillna(0.0).sort_index())

    M_base, M_surge = slot_zone_matrix(base_day), slot_zone_matrix(surge_day)
    return dict(
        catalog=catalog, topo=topo, fail=fail,
        M_base=M_base, M_surge=M_surge,
        surge_zone=int(M_surge.sum(axis=0).idxmax()),
        cpu_icdf=p1.inv_cdf(prof["cpu_cores_pct"]),
        mem_icdf=p1.inv_cdf(prof["mem_norm_pct"]),
        life_icdf=p1.inv_cdf(prof["lifetime_s_pct"]))


def generate(seed, c, K):
    """Identical to the seed loop in phase1_scenario.main(), followed by
    calibrate_workload.scaled(K)."""
    catalog = c["catalog"]
    fam = [s for s, v in catalog.items() if v["phase"] == "A"]
    nov = [s for s, v in catalog.items() if v["phase"] == "B"]
    rng = np.random.default_rng(seed)
    a = (rng, catalog, c["cpu_icdf"], c["mem_icdf"], c["life_icdf"])
    reqs = []
    reqs += p1.sample_requests(*a, c["M_base"], p1.N_REQ["A"], "A", 0.0,
                               p1.PH_SPAN, fam, [0.25, 0.35, 0.25, 0.15])
    reqs += p1.sample_requests(*a, c["M_base"], p1.N_REQ["B"], "B", p1.PH_SPAN,
                               p1.PH_SPAN, fam + nov,
                               [0.15, 0.2, 0.15, 0.1, 0.15, 0.15, 0.10])
    reqs += p1.sample_requests(*a, c["M_surge"], p1.N_REQ["C"], "C",
                               2 * p1.PH_SPAN, p1.PH_SPAN, fam + nov,
                               [0.15, 0.3, 0.1, 0.15, 0.1, 0.1, 0.10])
    reqs.sort(key=lambda r: r["t_s"])
    for i, r in enumerate(reqs):
        r["req_id"] = f"r{seed}_{i:04d}"
        r["lifetime_s"] = round(r["lifetime_s"] * K, 0)

    fail_t = 2 * p1.PH_SPAN + 0.7 * p1.PH_SPAN
    mttr_s = float(c["fail"]["mttr_minutes"]) * 60
    zone_nodes = [n["node_id"] for z in c["topo"]["zones"]
                  if z["zone_id"] == f"z{c['surge_zone']}" for n in z["nodes"]]
    fail_node = str(np.random.default_rng(seed + 100).choice(zone_nodes))
    events = [{"type": "node_failure", "t_s": round(fail_t, 1),
               "node_id": fail_node, "recover_t_s": round(fail_t + mttr_s, 1),
               "source": "Google cluster MTTR"}]
    return reqs, events


def write(path, reqs, events):
    with open(path, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
        for e in events:
            f.write(json.dumps({"event": e}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="only verify seeds 0-2 reproduce the original files")
    args = ap.parse_args()

    K = json.load(open(os.path.join(SCEN, "calibration_report.json")))["K"]
    c = _context()
    print(f"topology matches committed file | surge zone z{c['surge_zone']} | K={K}")

    wref = os.path.join(WORKLOADS, "wref")
    for s in (0, 1, 2):
        reqs, events = generate(s, c, K)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as t:
            tmp = t.name
        write(tmp, reqs, events)
        same = open(tmp).read() == open(os.path.join(wref, f"workload_seed{s}.jsonl")).read()
        os.unlink(tmp)
        print(f"seed {s}: regenerated == original file: {same}")
        if not same:
            sys.exit("generator does not reproduce the original workload - stopping")
    if args.check_only:
        return

    for s in EVAL_SEEDS:
        if s in (0, 1, 2):
            continue
        reqs, events = generate(s, c, K)
        write(os.path.join(wref, f"workload_seed{s}.jsonl"), reqs, events)
        print(f"wref seed {s}: {len(reqs)} requests, failure on {events[0]['node_id']}")
    os.makedirs(os.path.join(WORKLOADS, "dev"), exist_ok=True)
    for s in DEV_SEEDS:
        reqs, events = generate(s, c, K)
        write(os.path.join(WORKLOADS, "dev", f"workload_seed{s}.jsonl"), reqs, events)
        print(f"dev seed {s}: {len(reqs)} requests, failure on {events[0]['node_id']}")


if __name__ == "__main__":
    main()
