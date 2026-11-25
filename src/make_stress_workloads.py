#!/usr/bin/env python3
"""Generate the Stress scenario and workloads (STUDY_STRESS.md).

Built on the W-ref generator (same topology nodes, traces, resource
profiles, phase sizes and lifetime calibration K), with four changes:

  1. RTT     zones grouped into three regions; RTT per zone pair drawn once
             (fixed seed) from the band of that pair (5-80 ms)
  2. types   4 new service types from phase B: two latency-tight types that
             may leave their zone, one type similar to traffic_monitor, one
             dissimilar control
  3. failures 5 failures: 2 zone-wide (most nodes of a zone at once), 3 single
             nodes; repair time drawn from the Google-cluster downtime
             distribution, clipped to 10-60 min
  4. hotspot phase C's surge moves z1 -> z6 -> z7, 20 min each (the real
             surge-day activity pattern, moved onto the next zone)

Writes
  scenario/stress/topology_stress.json, service_catalog_stress.json,
  intent_sentences_stress.json, stress_report.json
  workloads/stress/workload_seed{0..9}.jsonl
  workloads/stress_dev/workload_seed{100..104}.jsonl

Run: python src/make_stress_workloads.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_scenario as p1
from config import DEV_SEEDS, EVAL_SEEDS, PLACES, SCEN, WORKLOADS
from make_wref_workloads import _context, write

STRESS_SCEN = os.path.join(SCEN, "stress")
RTT_SEED = 2025

REGIONS = {"core": ["z1", "z0", "z2", "z3"],
           "east": ["z6", "z4", "z5"],
           "west": ["z7", "z8", "z9"]}
RTT_BANDS = {("core", "core"): (5, 10), ("east", "east"): (10, 20),
             ("west", "west"): (10, 20), ("core", "east"): (30, 50),
             ("core", "west"): (30, 50), ("east", "west"): (60, 80)}

# name: (cpu_scale, mem_gb_base, latency_ms, data_locality, priority, templates)
NEW_TYPES = {
    "vehicle_teleop": (1.0, 1.0, 25, "any", "high", [
        "Remotely drive the delivery robot through {place}, control loop under 25 ms",
        "Teleoperate the autonomous shuttle around {place} with real-time video and steering",
        "Take remote control of the maintenance vehicle in {place} with hard latency limits"]),
    "vr_training_session": (1.8, 1.5, 40, "any", "high", [
        "Host a VR training session for the crews near {place} with under 40 ms response",
        "Stream an immersive virtual-reality walkthrough of {place}, ultra low latency",
        "Run an interactive mixed-reality rehearsal for the {place} event staff"]),
    "parking_occupancy": (0.6, 0.5, 200, "any", "normal", [
        "Track free parking spaces around {place} and report occupancy",
        "Monitor the {place} car parks and count vehicles entering and leaving",
        "Watch kerbside parking near {place} for congestion and overstays"]),
    "log_archive_batch": (0.3, 0.5, 2000, "any", "low", [
        "Archive yesterday's application logs from {place} to cold storage",
        "Compress and back up the {place} audit logs overnight",
        "Move the old {place} log files into the long-term archive"]),
}
TIGHT_TYPES = ["vehicle_teleop", "vr_training_session"]
SIMILAR_TYPES = {"vehicle_teleop": "drone_control",
                 "vr_training_session": "ar_session",
                 "parking_occupancy": "traffic_monitor"}
CONTROL_TYPE = "log_archive_batch"

# type mix: W-ref's phase B / C mix scaled to 70%, new types 30%
NEW_SHARE = {"vehicle_teleop": 0.10, "vr_training_session": 0.10,
             "parking_occupancy": 0.07, "log_archive_batch": 0.03}
WREF_B = [0.15, 0.2, 0.15, 0.1, 0.15, 0.15, 0.10]
WREF_C = [0.15, 0.3, 0.1, 0.15, 0.1, 0.1, 0.10]

HOTSPOT = [("z1", 7200.0), ("z6", 8400.0), ("z7", 9600.0)]  # 20 min each
# (time s, zone, nodes failing; None = one node)
FAILURES = [(2700.0, "z3", None), (5400.0, "z6", 6), (7500.0, "z1", None),
            (9000.0, "z7", 5), (10200.0, "z4", None)]
REPAIR_MIN = (10.0, 60.0)


def region_of(z):
    return next(r for r, zs in REGIONS.items() if z in zs)


def stress_topology(topo):
    rng = np.random.default_rng(RTT_SEED)
    out = json.loads(json.dumps(topo))
    out["name"] = "stress"
    rtt = {}
    for key in sorted(topo["inter_zone_rtt_ms"]):
        a, b = key.split("-")
        ra, rb = region_of(a), region_of(b)
        lo, hi = RTT_BANDS.get((ra, rb)) or RTT_BANDS[(rb, ra)]
        rtt[key] = round(float(rng.uniform(lo, hi)), 1)
    out["inter_zone_rtt_ms"] = rtt
    out["regions"] = REGIONS
    out["provenance"] = dict(topo.get("provenance", {}))
    out["provenance"]["rtt"] = (
        f"Stress: regional bands {json.dumps({f'{a}-{b}': v for (a, b), v in RTT_BANDS.items()})} ms, "
        f"uniform per pair, seed {RTT_SEED}")
    return out


def stress_catalog(catalog):
    cat = json.loads(json.dumps(catalog))
    for name, (cs, mb, lat, loc, pri, tm) in NEW_TYPES.items():
        cat[name] = {"phase": "B", "cpu_scale": cs, "mem_gb_base": mb,
                     "latency_ms": lat, "data_locality": loc, "priority": pri,
                     "templates": tm}
    return cat


def stress_sentences(sentences, cat):
    out = list(sentences)
    for name in NEW_TYPES:
        out += [{"service_type": name, "text": t.format(place=p)}
                for t in cat[name]["templates"] for p in PLACES[:5]]
    return out


def moved(M, src, dst):
    """The surge-day activity matrix with zone src's and dst's columns
    swapped, so the surge lands on dst."""
    if src == dst:
        return M
    M = M.copy()
    M[[src, dst]] = M[[dst, src]].values
    return M


def generate(seed, c, cat, topo, K):
    fam = [s for s, v in cat.items() if v["phase"] == "A"]
    nov = [s for s in cat if cat[s]["phase"] == "B" and s not in NEW_TYPES]
    new = list(NEW_TYPES)
    pb = [0.7 * p for p in WREF_B] + [NEW_SHARE[s] for s in new]
    pc = [0.7 * p for p in WREF_C] + [NEW_SHARE[s] for s in new]
    rng = np.random.default_rng(seed)
    a = (rng, cat, c["cpu_icdf"], c["mem_icdf"], c["life_icdf"])
    reqs = []
    reqs += p1.sample_requests(*a, c["M_base"], p1.N_REQ["A"], "A", 0.0,
                               p1.PH_SPAN, fam, [0.25, 0.35, 0.25, 0.15])
    reqs += p1.sample_requests(*a, c["M_base"], p1.N_REQ["B"], "B", p1.PH_SPAN,
                               p1.PH_SPAN, fam + nov + new, pb)
    n_c = p1.N_REQ["C"]
    sizes = [n_c // 3 + (1 if i < n_c % 3 else 0) for i in range(3)]
    span = p1.PH_SPAN / 3
    src = c["surge_zone"]
    for (zone, t0), n in zip(HOTSPOT, sizes):
        M = moved(c["M_surge"], src, int(zone[1:]))
        reqs += p1.sample_requests(*a, M, n, "C", t0, span, fam + nov + new, pc)
    reqs.sort(key=lambda r: r["t_s"])
    for i, r in enumerate(reqs):
        r["req_id"] = f"s{seed}_{i:04d}"
        r["lifetime_s"] = round(r["lifetime_s"] * K, 0)

    frng = np.random.default_rng(seed + 100)
    pct = c["fail"]["downtime_min_pct"]            # percentiles 5..95 (minutes)
    px = sorted(int(k) for k in pct)
    py = np.log([pct[str(k)] for k in px])

    def down(u):                                   # log-linear inverse CDF
        return float(np.exp(np.interp(u, px, py)))
    events = []
    for t, zone, k in FAILURES:
        nodes = [n["node_id"] for z in topo["zones"] if z["zone_id"] == zone
                 for n in z["nodes"]]
        chosen = [str(x) for x in frng.choice(nodes, size=k or 1, replace=False)]
        repair = float(np.clip(down(frng.uniform(5, 95)), *REPAIR_MIN)) * 60
        for nid in sorted(chosen):
            events.append({"type": "node_failure", "t_s": t, "node_id": nid,
                           "zone": zone, "zone_wide": bool(k),
                           "recover_t_s": round(t + repair, 1),
                           "source": "Google cluster downtime, clipped 10-60 min"})
    for zone, t in HOTSPOT:
        events.append({"type": "hotspot_move", "t_s": t, "zone": zone})
    return reqs, events


def similarity_report(cat, sentences):
    from graph_memory import SIM_THRESHOLD, SIM_TOP_K, type_similarity
    from zoa_agent import Embedder, build_schema
    _, _, semantic = build_schema(cat)
    emb = Embedder([s["text"] for s in sentences])
    cent = {}
    for st in cat:
        v = np.mean(emb.encode([s["text"] for s in sentences
                                if s["service_type"] == st]), axis=0)
        cent[st] = v / (np.linalg.norm(v) + 1e-12)
    sims = type_similarity(semantic, cent)
    edges = {}
    for st in sorted(cat):
        top = sorted(((b, v) for (a, b), v in sims.items() if a == st),
                     key=lambda x: -x[1])[:SIM_TOP_K]
        edges[st] = [(b, v) for b, v in top if v >= SIM_THRESHOLD]
    checks = {f"{a}->{b}": any(x == b for x, _ in edges[a])
              for a, b in SIMILAR_TYPES.items()}
    checks[f"{CONTROL_TYPE} has no SIMILAR_TO edge"] = not edges[CONTROL_TYPE]
    return {"backend": emb.backend, "similar_to": edges, "checks": checks}


def main():
    K = json.load(open(os.path.join(SCEN, "calibration_report.json")))["K"]
    c = _context()
    os.makedirs(STRESS_SCEN, exist_ok=True)
    topo = stress_topology(c["topo"])
    cat = stress_catalog(c["catalog"])
    sentences = stress_sentences(
        json.load(open(os.path.join(SCEN, "intent_sentences.json"))), cat)
    json.dump(topo, open(os.path.join(STRESS_SCEN, "topology_stress.json"), "w"), indent=1)
    json.dump({"services": cat}, open(os.path.join(STRESS_SCEN, "service_catalog_stress.json"), "w"), indent=1)
    json.dump(sentences, open(os.path.join(STRESS_SCEN, "intent_sentences_stress.json"), "w"), indent=1)

    rtts = np.array(list(topo["inter_zone_rtt_ms"].values()))
    sim = similarity_report(cat, sentences)
    print(f"RTT {rtts.min():.1f}-{rtts.max():.1f} ms | similarity checks: {sim['checks']}")
    if not all(sim["checks"].values()):
        sys.exit("similarity check failed - new types are not related as designed")

    runs = {}
    for sub, seeds in (("stress", EVAL_SEEDS), ("stress_dev", DEV_SEEDS)):
        d = os.path.join(WORKLOADS, sub)
        os.makedirs(d, exist_ok=True)
        for s in seeds:
            reqs, events = generate(s, c, cat, topo, K)
            write(os.path.join(d, f"workload_seed{s}.jsonl"), reqs, events)
            fails = [{"t_s": e["t_s"], "zone": e["zone"], "node": e["node_id"],
                      "repair_min": round((e["recover_t_s"] - e["t_s"]) / 60, 1)}
                     for e in events if e["type"] == "node_failure"]
            per_type = {}
            for r in reqs:
                per_type[r["service_type"]] = per_type.get(r["service_type"], 0) + 1
            hot = {z: round(float(np.mean([r["origin_zone"] == z for r in reqs
                                           if r["phase"] == "C" and t <= r["t_s"] < t + 1200])), 3)
                   for z, t in HOTSPOT}
            runs[f"{sub}/{s}"] = {"n": len(reqs),
                                  "per_phase": {p: sum(r["phase"] == p for r in reqs) for p in "ABC"},
                                  "per_type": per_type, "failures": fails,
                                  "hotspot_share_of_origin": hot}
            print(f"{sub} seed {s}: {len(reqs)} requests | {len(fails)} failed nodes "
                  f"| hotspot share {hot}")

    report = {
        "regions": REGIONS, "rtt_bands_ms": {f"{a}-{b}": v for (a, b), v in RTT_BANDS.items()},
        "rtt_ms": {"min": float(rtts.min()), "max": float(rtts.max()),
                   "mean": round(float(rtts.mean()), 2)},
        "new_types": {k: {"latency_ms": v[2], "data_locality": v[3], "priority": v[4]}
                      for k, v in NEW_TYPES.items()},
        "tight_types": TIGHT_TYPES, "similar_types": SIMILAR_TYPES,
        "control_type": CONTROL_TYPE, "type_mix_new_share": NEW_SHARE,
        "hotspot": HOTSPOT, "failures": FAILURES, "repair_min_clip": REPAIR_MIN,
        "similarity": sim, "K": K, "runs": runs}
    json.dump(report, open(os.path.join(STRESS_SCEN, "stress_report.json"), "w"), indent=1)
    print("wrote scenario/stress/ and workloads/stress{,_dev}/")


if __name__ == "__main__":
    main()
