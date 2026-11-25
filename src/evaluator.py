"""Discrete-event simulation of one (workload, orchestrator) run, plus
the per-run metrics.

Simulation is the earlier study's simulator. Its placement, preemption,
migration and failure handling are unchanged. Additions, none of which
change what gets placed where:
  - outcome hooks (on_outcome / on_preempt_outcome) so graph memory can
    learn from what happened after a placement; flat memory ignores them
  - arrival time, zone and latency budget logged per decision, for the
    SLA-violation and adaptation metrics
  - audit() reports only the metrics this study uses
"""
import json
import time
from collections import defaultdict

import numpy as np

from sim_engine import Node, Sim, ZoneRuntime

DIGEST_S = 5.0
POLICY_TICK_S = 120.0
ROLLING_BIN_S = 300.0      # bin width for the acceptance-over-time series
ADAPT_WINDOW_S = 1800.0    # window before/after a failure for adaptation
HOTSPOT_WINDOW_S = 600.0   # window before/after a hotspot move (Stress)
NEW_TYPE_WINDOW_S = 1800.0  # a type's first 30 min (Stress)


def load_workload(path):
    reqs, events = [], []
    for line in open(path):
        o = json.loads(line)
        (events if "event" in o else reqs).append(o.get("event", o))
    return reqs, events


class Simulation:
    def __init__(s, topo, reqs, events, orch, required_fields, slm_system,
                 adapt_window_s=None):
        s.sim, s.topo, s.orch = Sim(), topo, orch
        s.required_fields, s.slm_system = required_fields, slm_system
        s.nodes, s.zones = {}, {}
        s.event_log, s.services, s.lat_log = [], {}, []
        for z in topo["zones"]:
            zr = ZoneRuntime(z["zone_id"])
            s.zones[z["zone_id"]] = zr
            for spec in z["nodes"]:
                n = Node(spec, z["zone_id"])
                s.nodes[n.node_id] = n
                zr.register(n)
        s.rtt = {}
        for key, ms in topo["inter_zone_rtt_ms"].items():
            a, b = key.split("-")
            s.rtt[(a, b)] = s.rtt[(b, a)] = float(ms)
        s.intra_ms = float(topo.get("intra_zone_ms", 1.0))
        s.reqs, s.events = reqs, events
        s._peak, s._pending = defaultdict(float), 0
        s.failure_times = {}
        s.adapt_window_s = adapt_window_s or ADAPT_WINDOW_S
        s.placements = []  # every placement: fresh, replan, migration

    def _hook(s, name, *a):
        fn = getattr(s.orch, name, None)
        if fn:
            fn(*a)

    def _at_real(s, t, fn, *a):
        s._pending += 1

        def w(*args):
            s._pending -= 1
            fn(*args)
        s.sim.at(t, w, *a)

    def _tables(s):
        """Zone -> node -> state snapshot, with a 'running' list so the
        GOA can reason about preemption candidates."""
        out = {}
        for z, zr in s.zones.items():
            snap = zr.snapshot()
            for nid, row in snap.items():
                node = s.nodes[nid]
                running = []
                for rid, (cpu, mem) in node.services.items():
                    svc = s.services.get(rid)
                    if not svc or svc["state"] != "PLACED":
                        continue
                    req = svc["req"]
                    remaining = max((svc["t_end"] or s.sim.t) - s.sim.t, 0.0)
                    running.append({"req_id": rid,
                                    "priority": req.get("priority", "normal"),
                                    "remaining_s": remaining,
                                    "service_type": req.get("service_type"),
                                    "cpu": cpu, "mem": mem})
                row["running"] = running
            out[z] = snap
        return out

    def _place_on(s, t, req, nid, remaining=None):
        n = s.nodes[nid]
        deg = getattr(s.orch, "degraded", {}).get(req["req_id"])
        cpu, mem = deg if deg else (req["cpu_req"], req["mem_gb"])
        n.alloc(req["req_id"], cpu, mem)
        s.zones[n.zone_id].push(t, n)
        s._peak[nid] = max(s._peak[nid], n.cpu_used)
        svc = s.services[req["req_id"]]
        life = remaining if remaining is not None else req["lifetime_s"]
        svc.update(state="PLACED", node=nid, placed=True, t_end=t + life)
        s._at_real(t + life, s._complete, req["req_id"])
        return n.zone_id

    def _evict_victim(s, t, req):
        """Evict the victim the orchestrator flagged for req (release only).
        Migration happens after the incoming request has claimed the freed
        capacity, so the victim cannot land back on the node just freed."""
        victim_rid = getattr(s.orch, "preempted", {}).pop(req["req_id"], None)
        if victim_rid is None:
            return None
        victim_svc = s.services.get(victim_rid)
        if victim_svc is None or victim_svc["state"] != "PLACED":
            return None
        n = s.nodes[victim_svc["node"]]
        if victim_rid not in n.services:
            return None
        n.release(victim_rid)
        s.zones[n.zone_id].push(t, n)
        remaining = max((victim_svc["t_end"] or t) - t, 1.0)
        s.event_log.append({"t": t, "ev": "preempted", "req": victim_rid,
                            "by": req["req_id"], "node": n.node_id})
        return {"rid": victim_rid, "req": victim_svc["req"], "remaining": remaining,
                "svc": victim_svc, "by_req": req}

    def _commit_placement(s, t, req, nid, remaining=None, depth=0):
        """Single choke point for placing a request: evict any flagged
        victim first, place, then try to migrate the victim."""
        victim = s._evict_victim(t, req) if depth < 6 else None
        zid = s._place_on(t, req, nid, remaining=remaining)
        s.placements.append({"t": t, "origin": req["origin_zone"], "zone": zid,
                             "budget": req.get("latency_ms"),
                             "st": req.get("service_type"),
                             "fresh": remaining is None})
        if victim is not None:
            s._migrate_victim(t, victim, depth=depth + 1)
        return zid

    def _migrate_victim(s, t, victim, depth=0):
        new_nid, path = (s.orch.replan(t, victim["req"], s._tables(),
                                       s.required_fields, s.slm_system)
                         if hasattr(s.orch, "replan") else (None, None))
        if new_nid is None:
            victim["svc"]["state"] = "DISPLACED"
            s._hook("on_preempt_outcome", t, victim["req"], victim["by_req"], False)
            s._hook("on_outcome", t, victim["rid"], "displaced")
            return
        s._commit_placement(t, victim["req"], new_nid,
                            remaining=victim["remaining"], depth=depth)
        s.event_log.append({"t": t, "ev": "migrated", "req": victim["rid"],
                            "to_node": new_nid, "path": path})
        s._hook("on_preempt_outcome", t, victim["req"], victim["by_req"], True)

    def _arrival(s, req):
        t = s.sim.t
        t0 = time.time()
        nid, path = s.orch.place(t, req, s._tables(), s.topo,
                                 s.required_fields, s.slm_system)
        dms = (time.time() - t0) * 1000
        s.lat_log.append({"t": t, "phase": req["phase"], "ms": dms,
                          "st": req.get("service_type"),
                          "placed": nid is not None, "path": path})
        if nid is None:
            s.services[req["req_id"]]["state"] = "REJECTED"
            s.event_log.append({"t": t, "ev": "reject", "phase": req["phase"]})
            return
        k = len(s.placements)
        zid = s._commit_placement(t, req, nid)
        s.placements[k]["path"] = path
        s.event_log.append({"t": t, "ev": "place", "phase": req["phase"],
                            "zone": zid, "origin": req["origin_zone"],
                            "latency_budget_ms": req.get("latency_ms"),
                            "cross_zone": zid != req["origin_zone"]})

    def _complete(s, rid):
        svc = s.services[rid]
        if svc["state"] != "PLACED" or svc["t_end"] is None or \
                abs(svc["t_end"] - s.sim.t) > 1e-6:
            return
        n = s.nodes[svc["node"]]
        if rid in n.services:
            n.release(rid)
            s.zones[n.zone_id].push(s.sim.t, n)
        svc["state"] = "COMPLETED"
        s._hook("on_outcome", s.sim.t, rid, "completed")

    def _fail(s, ev):
        t = s.sim.t
        n = s.nodes[ev["node_id"]]
        displaced = n.fail()
        s.zones[n.zone_id].on_hard_failure(t, n.node_id)
        s.failure_times[ev["node_id"]] = t
        s.event_log.append({"t": t, "ev": "node_failure", "node": n.node_id,
                            "n_displaced": len(displaced)})
        s._hook("on_failure", t, n.zone_id)
        for rid in displaced:
            svc = s.services[rid]
            req = svc["req"]
            remaining = max((svc["t_end"] or t) - t, 1.0)
            new, path = (s.orch.replan(t, req, s._tables(), s.required_fields,
                                       s.slm_system)
                         if hasattr(s.orch, "replan") else (None, None))
            if new is not None:
                s._commit_placement(t, req, new, remaining=remaining)
                s.event_log.append({"t": t, "ev": "replanned", "req": rid,
                                    "path": path})
            else:
                svc["state"] = "DISPLACED"
                s._hook("on_outcome", t, rid, "displaced")
        s._at_real(ev["recover_t_s"], s._recover, ev["node_id"])

    def _recover(s, nid):
        n = s.nodes[nid]
        n.recover()
        s.zones[n.zone_id].push(s.sim.t, n)

    def _digest_tick(s, t):
        if hasattr(s.orch, "on_digest_tick"):
            s.orch.on_digest_tick(t, s.zones)
        if s._pending > 0:
            s.sim.at(t + DIGEST_S, s._digest_tick, t + DIGEST_S)

    def _policy_tick(s, t):
        if hasattr(s.orch, "on_policy_tick"):
            s.orch.on_policy_tick(t)
        if s._pending > 0:
            s.sim.at(t + POLICY_TICK_S, s._policy_tick, t + POLICY_TICK_S)

    def run(s):
        for r in s.reqs:
            s.services[r["req_id"]] = {"state": "REQUESTED", "node": None,
                                       "placed": False, "req": r, "t_end": None}
            s._at_real(r["t_s"], s._arrival, r)
        for e in s.events:
            if e["type"] == "node_failure":
                s._at_real(e["t_s"], s._fail, e)
        s.sim.at(0, s._digest_tick, 0)
        s.sim.at(POLICY_TICK_S, s._policy_tick, POLICY_TICK_S)
        s.sim.run()
        return s.audit()

    # ------------------------------------------------------------ metrics
    def audit(s):
        states = defaultdict(int)
        for v in s.services.values():
            states[v["state"]] += 1
        n = len(s.services)
        placed = sum(1 for v in s.services.values() if v["placed"])

        per_phase = defaultdict(lambda: [0, 0])
        for e in s.event_log:
            if e["ev"] in ("place", "reject"):
                per_phase[e["phase"]][0 if e["ev"] == "place" else 1] += 1

        # SLA: network latency of the placement vs the request's budget
        sla_viol = sla_n = 0
        for e in s.event_log:
            if e["ev"] != "place" or e.get("latency_budget_ms") is None:
                continue
            lat = s.intra_ms + (s.rtt.get((e["origin"], e["zone"]), 0.0)
                                if e["cross_zone"] else 0.0)
            sla_n += 1
            sla_viol += int(lat > e["latency_budget_ms"])

        replanned = sum(1 for e in s.event_log if e["ev"] == "replanned")
        migrations = sum(1 for e in s.event_log if e["ev"] == "migrated")
        displaced_final = states.get("DISPLACED", 0)
        attempts = replanned + migrations + displaced_final
        # Two preemption counts. Events = every eviction (a service evicted,
        # migrated and evicted again counts twice), as the earlier study
        # counted. Services = distinct evicted services, the denominator of
        # preemption regret.
        preemption_events = sum(1 for e in s.event_log if e["ev"] == "preempted")
        preempted = {e["req"] for e in s.event_log if e["ev"] == "preempted"}
        victims_displaced = sum(1 for r in preempted
                                if s.services[r]["state"] == "DISPLACED")

        # acceptance over time (arrival-time bins) and around failures
        t_end = max((x["t"] for x in s.lat_log), default=0.0)
        bins = int(np.ceil(t_end / ROLLING_BIN_S)) + 1
        off, acc = np.zeros(bins), np.zeros(bins)
        for x in s.lat_log:
            b = int(x["t"] // ROLLING_BIN_S)
            off[b] += 1
            acc[b] += int(x["placed"])
        rolling = [{"t": round(i * ROLLING_BIN_S, 1), "offered": int(off[i]),
                    "acceptance": round(acc[i] / off[i], 3) if off[i] else None}
                   for i in range(bins)]

        def window_ar(a, b):
            xs = [x["placed"] for x in s.lat_log if a <= x["t"] < b]
            return (round(float(np.mean(xs)), 3) if xs else None), len(xs)

        adaptation = []
        for nid, tf in s.failure_times.items():
            pre, npre = window_ar(tf - ADAPT_WINDOW_S, tf)
            post, npost = window_ar(tf, tf + ADAPT_WINDOW_S)
            adaptation.append({"node": nid, "t_fail": tf,
                               "ar_before": pre, "n_before": npre,
                               "ar_after": post, "n_after": npost,
                               "ar_drop": (round(pre - post, 3)
                                           if pre is not None and post is not None
                                           else None)})

        # ---- Stress metrics (also computed for W-ref; not used there) ----
        def net_ms(p):
            return s.intra_ms + (s.rtt.get((p["origin"], p["zone"]), 0.0)
                                 if p["zone"] != p["origin"] else 0.0)

        by_type = defaultdict(lambda: [0, 0])
        all_viol = 0
        for p in s.placements:
            if p["budget"] is None:
                continue
            v = int(net_ms(p) > p["budget"])
            all_viol += v
            by_type[p["st"]][0] += 1
            by_type[p["st"]][1] += v
        n_pl = sum(v[0] for v in by_type.values())
        by_path = defaultdict(lambda: [0, 0])
        for p in s.placements:
            if p["budget"] is not None:
                k = p.get("path") or ("fresh" if p["fresh"] else "replan_or_migration")
                by_path[k][0] += 1
                by_path[k][1] += int(net_ms(p) > p["budget"])
        sla_by_path = {k: {"placements": a, "violations": b}
                       for k, (a, b) in sorted(by_path.items()) if b}
        sla_by_type = {st: {"placements": a, "violations": b,
                            "rate": round(b / a, 4) if a else None}
                       for st, (a, b) in sorted(by_type.items())}

        W = s.adapt_window_s
        groups = defaultdict(list)
        for nid, tf in s.failure_times.items():
            groups[tf].append(nid)
        failure_events = []
        for tf in sorted(groups):
            nodes = sorted(groups[tf])
            pre, npre = window_ar(tf - W, tf)
            post, npost = window_ar(tf, tf + W)
            failure_events.append({
                "t_fail": tf, "zone": s.nodes[nodes[0]].zone_id, "nodes": nodes,
                "zone_wide": len(nodes) > 1, "window_s": W,
                "ar_before": pre, "n_before": npre, "ar_after": post,
                "n_after": npost,
                "ar_drop": (round(pre - post, 3) if pre is not None
                            and post is not None else None)})

        hotspot_moves = []
        for e in s.events:
            if e.get("type") != "hotspot_move" or e["t_s"] <= 2 * 3600.0:
                continue
            tm = e["t_s"]
            pre, npre = window_ar(tm - HOTSPOT_WINDOW_S, tm)
            post, npost = window_ar(tm, tm + HOTSPOT_WINDOW_S)
            hotspot_moves.append({
                "t": tm, "zone": e["zone"], "ar_before": pre, "ar_after": post,
                "n_before": npre, "n_after": npost,
                "ar_change": (round(post - pre, 3) if pre is not None
                              and post is not None else None)})

        first_t = {}
        for x in s.lat_log:
            first_t.setdefault(x["st"], x["t"])
        first_window = {}
        for st, t0 in sorted(first_t.items()):
            xs = [x["placed"] for x in s.lat_log
                  if x["st"] == st and t0 <= x["t"] < t0 + NEW_TYPE_WINDOW_S]
            first_window[st] = {"t_first": t0, "n": len(xs),
                                "acceptance": round(float(np.mean(xs)), 3) if xs else None}

        lm = [x["ms"] for x in s.lat_log]
        return {
            "acceptance_overall": round(placed / n, 3),
            "acceptance_per_phase": {p: round(v[0] / max(sum(v), 1), 3)
                                     for p, v in sorted(per_phase.items())},
            "sla_violation_rate": round(sla_viol / sla_n, 4) if sla_n else 0.0,
            "replanned": replanned,
            "migrations": migrations,
            "preemption_events": preemption_events,
            "preempted_services": len(preempted),
            "replan_success_rate": (round((replanned + migrations) / attempts, 3)
                                    if attempts else None),
            "displaced_final": displaced_final,
            "preemption_regret": (round(victims_displaced / len(preempted), 3)
                                  if preempted else None),
            "decision_ms_mean": round(float(np.mean(lm)), 2) if lm else 0,
            "decision_ms_p95": round(float(np.percentile(lm, 95)), 2) if lm else 0,
            "acceptance_over_time": rolling,
            "adaptation_after_failure": adaptation,
            "sla_violation_rate_all_placements": (round(all_viol / n_pl, 4)
                                                  if n_pl else 0.0),
            "sla_by_type": sla_by_type,
            "sla_violations_by_path": sla_by_path,
            "failure_events": failure_events,
            "hotspot_moves": hotspot_moves,
            "first_window_acceptance_by_type": first_window,
            "states": dict(states),
            "invariants": {"terminal": states.get("REQUESTED", 0) == 0,
                           "closed": states.get("REJECTED", 0) + placed == n,
                           "no_breach": all(p <= s.nodes[i].cpu_cap + 1e-9
                                            for i, p in s._peak.items())}}
