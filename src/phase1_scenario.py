#!/usr/bin/env python3
"""Phase 1.2 — zone topology, service catalog, intent sentences, and
trace-driven workload generation. Requires Phase 1.1 outputs to exist
(validated/*.csv, *.parquet, *.json).

Run: python3 phase1_scenario.py
"""
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INTENT_DATA, SCEN, VALIDATED


def need(name):
    p = os.path.join(VALIDATED, name)
    assert os.path.exists(p) and os.path.getsize(p) > 0, \
        f"missing Phase 1.1 output: {name} - run phase1_data_foundation.py first"
    return p


HW = {"edge_server": {"cpu": 100, "mem_gb": 32, "disk_gb": 500},
      "edge_node": {"cpu": 40, "mem_gb": 8, "disk_gb": 100}}

SERVICES = {
    # name: (phase, cpu_scale, mem_gb_base, latency_ms, locality, priority)
    "traffic_monitor": ("A", 0.6, 0.5, 200, "any", "normal"),
    "video_analytics": ("A", 1.6, 2.0, 100, "zone_local", "high"),
    "iot_aggregator": ("A", 0.4, 0.5, 500, "any", "low"),
    "ar_session": ("A", 1.8, 1.5, 50, "zone_local", "high"),
    "federated_ml": ("B", 2.4, 4.0, 800, "any", "normal"),
    "digital_twin": ("B", 2.0, 3.0, 250, "any", "normal"),
    "drone_control": ("B", 1.0, 1.0, 30, "zone_local", "critical"),
}

# Original 3 hand-written templates per class - the seeds paraphrased by
# scripts/generate_intent_paraphrases.py. Kept separate from the runtime
# INTENT_TEMPLATES below so reruns of that script always expand from the
# same 21 seeds, never from an already-expanded pool.
SEED_INTENT_TEMPLATES = {
    "traffic_monitor": [
        "Monitor traffic flow at the {place} intersections and report congestion",
        "Track vehicle counts around {place} continuously",
        "Watch the {place} junctions for unusual traffic build-up"],
    "video_analytics": [
        "Run video analytics on the {place} cameras, results within 100 ms, footage stays in the zone",
        "Analyse the {place} camera feeds in real time without sending video out of the area",
        "Detect incidents on the {place} surveillance streams with low latency"],
    "iot_aggregator": [
        "Aggregate the {place} air-quality sensor readings every few minutes",
        "Collect and summarise environmental sensor data from {place}",
        "Batch the {place} IoT telemetry and forward hourly summaries"],
    "ar_session": [
        "Host an AR tourism session for visitors near {place} with under 50 ms response",
        "Serve augmented-reality overlays for the {place} area, ultra low latency",
        "Start an interactive AR guide around {place} for a walking tour"],
    "federated_ml": [
        "Train the shared mobility model on local data from {place} without moving raw data",
        "Run a federated learning round for the {place} district models",
        "Contribute the {place} data partition to this week's federated training job"],
    "digital_twin": [
        "Update the digital twin of the {place} energy grid with live readings",
        "Simulate load scenarios on the {place} district twin",
        "Keep the {place} infrastructure twin in sync with sensor state"],
    "drone_control": [
        "Coordinate the inspection drone over {place}, control loop under 30 ms, keep control local",
        "Fly the survey drone across {place} with real-time obstacle avoidance",
        "Run the drone path planner for the {place} corridor with hard latency limits"],
}

# Runtime templates used for catalog/sentence generation: the LLM-expanded
# pool (21 seeds -> 84 templates, see intent_data/) if it has been
# generated, else falls back to the seeds alone.
_EXPANDED_PATH = os.path.join(INTENT_DATA, "intent_templates_expanded.json")
if os.path.exists(_EXPANDED_PATH):
    INTENT_TEMPLATES = json.load(open(_EXPANDED_PATH))
else:
    INTENT_TEMPLATES = SEED_INTENT_TEMPLATES

PLACES = ["stadium", "central station", "old town", "market square",
          "university campus", "harbour district", "tech park",
          "exhibition centre", "north ring", "riverside"]

PCT_X = [1, 5, 25, 50, 75, 95, 99]
N_REQ = {"A": 400, "B": 250, "C": 350}
SEEDS = [0, 1, 2]
PH_SPAN = 3600.0
RNG_TOPO = 42


def inv_cdf(pct_dict):
    ys = [pct_dict[str(p)] if str(p) in pct_dict else pct_dict[p]
          for p in PCT_X]
    return lambda u: float(np.interp(u, PCT_X, ys))


def build_topology(cells, rttm, n_zones, name, servers_total, nodes_total,
                    seed=RNG_TOPO):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_zones, n_init=10, random_state=seed)
    lab = km.fit_predict(cells[["lon", "lat"]], sample_weight=cells.w + 1e-9)
    cells[f"zone_{name}"] = lab
    zshare = (cells.groupby(f"zone_{name}")["w"].sum()
              / max(cells.w.sum(), 1e-9))
    nid = 0
    srv_alloc = np.maximum(1, np.round(zshare.values * servers_total)).astype(int)
    nod_alloc = np.maximum(2, np.round(zshare.values * nodes_total)).astype(int)
    zones = []
    for z in range(n_zones):
        nodes = []
        for _ in range(srv_alloc[z]):
            nodes.append({"node_id": f"n{nid}", "type": "edge_server",
                          **HW["edge_server"]}); nid += 1
        for _ in range(nod_alloc[z]):
            nodes.append({"node_id": f"n{nid}", "type": "edge_node",
                          **HW["edge_node"]}); nid += 1
        c = km.cluster_centers_[z]
        zones.append({"zone_id": f"z{z}", "centroid_lon": float(c[0]),
                      "centroid_lat": float(c[1]),
                      "activity_share": round(float(zshare.iloc[z]), 4),
                      "nodes": nodes})
    lo, hi = rttm["min_ms"], rttm["max_ms"]
    dists = {}
    for a, b in itertools.combinations(range(n_zones), 2):
        ca, cb = km.cluster_centers_[a], km.cluster_centers_[b]
        dists[(a, b)] = math.dist(ca, cb)
    dmin, dmax = min(dists.values()), max(dists.values())
    rtt = {f"z{a}-z{b}": round(lo + (hi - lo) *
                               ((d - dmin) / max(dmax - dmin, 1e-12)), 2)
           for (a, b), d in dists.items()}
    topo = {"name": name, "n_zones": n_zones,
            "n_nodes": sum(len(z["nodes"]) for z in zones),
            "zones": zones, "inter_zone_rtt_ms": rtt,
            "intra_zone_ms": rttm["intra_zone_ms"],
            "zone_to_cloud_ms": rttm["zone_to_cloud_ms"],
            "hw_profiles": HW,
            "provenance": {"cells": "OpenData Milano grid (Telecom Italia)",
                           "weights": "Milan activity trace (O-RAN parsed)",
                           "rtt": rttm["source"]}}
    with open(os.path.join(SCEN, f"topology_{name}.json"), "w") as f:
        json.dump(topo, f, indent=1)
    return topo


def build_catalog(prof):
    catalog = {}
    for name, (phase, cs, mb, lat, loc, pri) in SERVICES.items():
        catalog[name] = {"phase": phase, "cpu_scale": cs, "mem_gb_base": mb,
                         "latency_ms": lat, "data_locality": loc,
                         "priority": pri, "templates": INTENT_TEMPLATES[name]}
    json.dump({"services": catalog, "resource_source": prof["source"],
              "places": PLACES},
              open(os.path.join(SCEN, "service_catalog.json"), "w"), indent=1)
    sentences = [{"service_type": s, "text": t.format(place=p)}
                 for s, c in catalog.items()
                 for t in c["templates"] for p in PLACES]
    json.dump(sentences, open(os.path.join(SCEN, "intent_sentences.json"), "w"),
              indent=1)
    return catalog, sentences


def sample_requests(rng, catalog, cpu_icdf, mem_icdf, life_icdf, M, n, phase,
                     t0, span_s, types, type_p):
    slot_w = M.sum(axis=1).values
    slot_p = slot_w / slot_w.sum()
    zone_p = (M.values / np.maximum(M.values.sum(axis=1, keepdims=True), 1e-12))
    slots = rng.choice(len(slot_p), size=n, p=slot_p)
    out = []
    for sl in sorted(slots):
        z = int(rng.choice(M.shape[1], p=zone_p[sl]))
        stype = str(rng.choice(types, p=type_p))
        c = catalog[stype]
        u = rng.uniform(1, 99)
        cpu = max(2.0, cpu_icdf(u) * c["cpu_scale"] * 10)
        mem = max(0.25, c["mem_gb_base"] + mem_icdf(rng.uniform(1, 99)) / 100 * 4)
        life = float(np.clip(life_icdf(rng.uniform(1, 99)), 60, span_s * 2))
        tmpl = str(rng.choice(c["templates"]))
        place = str(rng.choice(PLACES))
        t = t0 + (sl / len(slot_p)) * span_s + rng.uniform(0, span_s / len(slot_p))
        out.append({
            "req_id": None, "t_s": round(t, 1), "phase": phase,
            "origin_zone": f"z{z}", "service_type": stype,
            "intent_text": tmpl.format(place=place),
            "cpu_req": round(cpu, 1), "mem_gb": round(mem, 2),
            "lifetime_s": round(life, 0), "latency_ms": c["latency_ms"],
            "data_locality": c["data_locality"], "priority": c["priority"]})
    return out


def main():
    grid = pd.read_csv(need("milan_grid_centroids.csv"))
    act = pd.read_parquet(need("milan_activity.parquet"))
    surge = pd.read_csv(need("milan_surge_candidates.csv"))
    prof = json.load(open(need("service_resource_profiles.json")))
    fail = json.load(open(need("failure_model.json")))
    rttm = json.load(open(need("rtt_model.json")))
    act["ts"] = pd.to_datetime(act["ts"])
    print(f"grid cells: {len(grid):,} | activity rows: {len(act):,} "
          f"| days: {act.day.nunique()} | surge day: {surge.day.mode().iat[0]}")

    cell_w = act.groupby("cell_id")["activity"].mean().rename("w").reset_index()
    cells = grid.merge(cell_w, on="cell_id", how="left").fillna({"w": 0.0})

    topo_dev = build_topology(cells, rttm, 5, "dev", servers_total=7, nodes_total=18)
    topo_eval = build_topology(cells, rttm, 10, "eval", servers_total=14, nodes_total=36)
    for t in (topo_dev, topo_eval):
        print(f"topology_{t['name']}: {t['n_zones']} zones, {t['n_nodes']} nodes")

    catalog, sentences = build_catalog(prof)
    print(f"service catalog: {len(catalog)} types | "
          f"intent sentences: {len(sentences)}")

    cpu_icdf = inv_cdf(prof["cpu_cores_pct"])
    mem_icdf = inv_cdf(prof["mem_norm_pct"])
    life_icdf = inv_cdf(prof["lifetime_s_pct"])

    surge_day = surge.day.mode().iat[0]
    normal_days = sorted(d for d in act.day.unique() if d != surge_day)
    base_day = normal_days[len(normal_days) // 2]
    print(f"base day: {base_day} | surge day: {surge_day}")

    cz = cells[["cell_id", "zone_eval"]]
    actz = act.merge(cz, on="cell_id")

    def slot_zone_matrix(day):
        d = actz[actz.day == day]
        return (d.pivot_table(index="ts", columns="zone_eval", values="activity",
                              aggfunc="sum").fillna(0.0).sort_index())

    M_base, M_surge = slot_zone_matrix(base_day), slot_zone_matrix(surge_day)
    surge_zone = int(M_surge.sum(axis=0).idxmax())
    print(f"surge zone (from real trace): z{surge_zone}")

    FAM = [s for s, c in catalog.items() if c["phase"] == "A"]
    NOV = [s for s, c in catalog.items() if c["phase"] == "B"]

    report_runs = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        reqs = []
        reqs += sample_requests(rng, catalog, cpu_icdf, mem_icdf, life_icdf,
                                M_base, N_REQ["A"], "A", 0.0, PH_SPAN, FAM,
                                [0.25, 0.35, 0.25, 0.15])
        reqs += sample_requests(rng, catalog, cpu_icdf, mem_icdf, life_icdf,
                                M_base, N_REQ["B"], "B", PH_SPAN, PH_SPAN,
                                FAM + NOV,
                                [0.15, 0.2, 0.15, 0.1, 0.15, 0.15, 0.10])
        reqs += sample_requests(rng, catalog, cpu_icdf, mem_icdf, life_icdf,
                                M_surge, N_REQ["C"], "C", 2 * PH_SPAN, PH_SPAN,
                                FAM + NOV,
                                [0.15, 0.3, 0.1, 0.15, 0.1, 0.1, 0.10])
        reqs.sort(key=lambda r: r["t_s"])
        for i, r in enumerate(reqs):
            r["req_id"] = f"r{seed}_{i:04d}"

        fail_t = 2 * PH_SPAN + 0.7 * PH_SPAN
        mttr_s = float(fail["mttr_minutes"]) * 60
        zone_nodes = [n["node_id"] for z in topo_eval["zones"]
                      if z["zone_id"] == f"z{surge_zone}" for n in z["nodes"]]
        fail_node = str(np.random.default_rng(seed + 100).choice(zone_nodes))
        events = [{"type": "node_failure", "t_s": round(fail_t, 1),
                  "node_id": fail_node, "recover_t_s": round(fail_t + mttr_s, 1),
                  "source": "Google cluster MTTR"}]
        path = os.path.join(SCEN, f"workload_seed{seed}.jsonl")
        with open(path, "w") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
            for e in events:
                f.write(json.dumps({"event": e}) + "\n")

        surge_share_C = (pd.Series([r["origin_zone"] for r in reqs
                                    if r["phase"] == "C"]).value_counts(
                                    normalize=True).get(f"z{surge_zone}", 0))
        report_runs[f"seed{seed}"] = {
            "n_requests": len(reqs),
            "per_phase": {p: sum(r["phase"] == p for r in reqs) for p in "ABC"},
            "novel_in_A": sum(r["phase"] == "A" and r["service_type"] in NOV
                              for r in reqs),
            "surge_zone_share_in_C": round(float(surge_share_C), 3),
            "failure": events[0]}
        print(f"seed{seed}: {len(reqs)} requests | phase C surge-zone share: "
              f"{surge_share_C:.0%} | failure on {fail_node} @t={fail_t:.0f}s")

    checks = {
        "topologies": {t["name"]: {"zones": t["n_zones"], "nodes": t["n_nodes"]}
                       for t in (topo_dev, topo_eval)},
        "surge_zone": f"z{surge_zone}", "base_day": base_day,
        "surge_day": surge_day, "catalog_types": list(catalog.keys()),
        "intent_sentences": len(sentences), "runs": report_runs}
    ok = True
    for seed in SEEDS:
        r = report_runs[f"seed{seed}"]
        ok &= r["n_requests"] == sum(N_REQ.values())
        ok &= r["per_phase"] == N_REQ
        ok &= r["novel_in_A"] == 0
    for t in (topo_dev, topo_eval):
        caps = [n["cpu"] for z in t["zones"] for n in z["nodes"]]
        ok &= len(caps) == t["n_nodes"] and min(caps) > 0
    checks["all_checks_pass"] = bool(ok)
    json.dump(checks, open(os.path.join(SCEN, "scenario_report.json"), "w"),
              indent=1, default=str)
    print("\n" + "=" * 60)
    print("PHASE 1.2 SUMMARY")
    print("=" * 60)
    print(json.dumps({k: v for k, v in checks.items() if k != "runs"},
                     indent=1, default=str))
    print("Phase 2 (simulation core) viability:",
          "YES - green light" if ok else "NO - see checks above")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
