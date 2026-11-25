"""Graph memory for the Global Orchestration Agent (GOA).

Replaces the flat memory (EpisodicMemory + ProceduralStore, both keyed
by service_type only) with a typed graph of the entities the GOA reasons
about and the relationships between them.

Graph nodes
  ("zone", z)           a zone; keeps its failure history
  ("type", st)          a service type
  ("ctx", st, origin)   "service type st requested from zone origin"
  ("rule", st)          a procedural rule authored on a policy tick

Graph edges
  zone  -NEAR->        zone   rtt_ms                      (topology, static)
  type  -SIMILAR_TO->  type   sim                         (catalog + intent embeddings)
  type  -HAS_CTX->     ctx
  ctx   -ROUTED_TO->   zone   s, f (time-decayed successes/failures), last_t
  type  -PREEMPTED->   type   n, migrated, displaced     (victim -> incoming type)
  rule  -PREFERS->     zone   confidence, valid

Zone score used by the GOA (see rank()):
  score(z) = w_hist * p(z) * n/(n+2)       learned success, shrunk when evidence is thin
           + w_cap  * top_cpu_free(z)/100  capacity from the zone digest
           + w_rule * rule_confidence(z)   procedural rule for this type, if valid
           - w_rtt  * rtt(origin, z)/max_rtt
           - w_fail * failure_risk(z)      decays with FAIL_HALF_LIFE_S

p(z) pools direct evidence for (type, origin) with down-weighted
evidence from the same type at other origins (weighted by proximity)
and from similar types (weighted by similarity). That pooling is what
lets the graph answer for a context it has never seen.
"""
import math
import time

import networkx as nx
import numpy as np

HALF_LIFE_S = 3600.0        # decay of route evidence (sim seconds)
FAIL_HALF_LIFE_S = 1800.0   # decay of zone failure risk
OTHER_ORIGIN_W = 0.5        # weight of same-type evidence from other origins
SIMILAR_TYPE_W = 0.5        # weight of similar-type evidence
SIM_THRESHOLD = 0.6         # minimum similarity for a SIMILAR_TO edge
SIM_TOP_K = 2               # SIMILAR_TO edges kept per type
CONFIDENT_N, CONFIDENT_P = 3.0, 0.6   # same bar as flat memory (cnt >= 3)
RULE_MIN_N, RULE_MIN_P = 5.0, 0.6     # same bar as flat rule authoring

# Candidate weight settings; one is chosen on the dev seeds (tune step).
WEIGHT_SETTINGS = {
    "history": {"w_hist": 1.0, "w_cap": 0.2, "w_rule": 0.2, "w_rtt": 0.1, "w_fail": 0.3},
    "balanced": {"w_hist": 0.6, "w_cap": 0.4, "w_rule": 0.2, "w_rtt": 0.3, "w_fail": 0.5},
    "risk": {"w_hist": 0.5, "w_cap": 0.3, "w_rule": 0.1, "w_rtt": 0.5, "w_fail": 1.0},
}

LATENCY_CLASSES = ["realtime", "interactive", "batch"]
PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2, "critical": 3}


def _decay(t_now, t_then, half_life):
    return 0.5 ** (max(t_now - t_then, 0.0) / half_life)


def type_similarity(semantic, intent_centroids=None):
    """Similarity between service types: mean of profile similarity
    (latency class, locality, priority, resource scale) and, when given,
    cosine similarity of the types' mean intent embeddings."""
    types = sorted(semantic)

    def vec(st):
        p = semantic[st]
        lc = [1.0 if p["latency_class"] == c else 0.0 for c in LATENCY_CLASSES]
        return np.array(lc + [1.0 if p["data_locality"] == "any" else 0.0,
                              PRIORITY_RANK.get(p["priority"], 1) / 3,
                              p["cpu_scale"] / 2.5], float)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    sims = {}
    for a in types:
        for b in types:
            if a == b:
                continue
            s = cos(vec(a), vec(b))
            if intent_centroids is not None:
                s = 0.5 * s + 0.5 * cos(intent_centroids[a], intent_centroids[b])
            sims[(a, b)] = round(s, 3)
    return sims


class GraphMemory:
    def __init__(s, topo, semantic, weights, type_sims):
        s.w = dict(weights)
        s.g = nx.MultiDiGraph()   # parallel edges keyed by relation
        s.zones = [z["zone_id"] for z in topo["zones"]]
        s.rtt = {}
        for z in s.zones:
            s.g.add_node(("zone", z), kind="zone", failures=[])
        for key, ms in topo["inter_zone_rtt_ms"].items():
            a, b = key.split("-")
            s.rtt[(a, b)] = s.rtt[(b, a)] = float(ms)
            s.g.add_edge(("zone", a), ("zone", b), key="NEAR", rel="NEAR", rtt_ms=float(ms))
            s.g.add_edge(("zone", b), ("zone", a), key="NEAR", rel="NEAR", rtt_ms=float(ms))
        s.max_rtt = max(s.rtt.values()) if s.rtt else 1.0
        for st in semantic:
            s.g.add_node(("type", st), kind="type")
        for st in semantic:
            ranked = sorted(((b, v) for (a, b), v in type_sims.items() if a == st),
                            key=lambda x: -x[1])
            for b, v in ranked[:SIM_TOP_K]:
                if v >= SIM_THRESHOLD:
                    s.g.add_edge(("type", st), ("type", b), key="SIMILAR_TO", rel="SIMILAR_TO", sim=v)
        s.placed = {}        # req_id -> (st, origin, zone) for outcome hooks
        s.lookup_ms = []

    # ---------------------------------------------------------- writes
    def _ctx(s, st, origin):
        n = ("ctx", st, origin)
        if n not in s.g:
            s.g.add_node(n, kind="ctx")
            if ("type", st) not in s.g:
                s.g.add_node(("type", st), kind="type")
            s.g.add_edge(("type", st), n, key="HAS_CTX", rel="HAS_CTX")
        return n

    def record(s, st, origin, zone, ok, t):
        """A placement attempt for (st, origin) in zone succeeded/failed."""
        if st is None or zone is None:
            return
        c = s._ctx(st, origin)
        e = s.g.get_edge_data(c, ("zone", zone), key="ROUTED_TO")
        if e is None:
            s.g.add_edge(c, ("zone", zone), key="ROUTED_TO", rel="ROUTED_TO", s=0.0, f=0.0, last_t=t)
            e = s.g.get_edge_data(c, ("zone", zone), key="ROUTED_TO")
        d = _decay(t, e["last_t"], HALF_LIFE_S)
        e["s"], e["f"], e["last_t"] = e["s"] * d, e["f"] * d, t
        e["s" if ok else "f"] += 1.0

    def note_placed(s, req_id, st, origin, zone):
        s.placed[req_id] = (st, origin, zone)

    def on_outcome(s, t, req_id, outcome):
        """Service outcome after placement: 'completed' or 'displaced'.
        A displacement counts as a failure of that route."""
        info = s.placed.pop(req_id, None)
        if info and outcome == "displaced":
            s.record(info[0], info[1], info[2], False, t)

    def on_zone_failure(s, t, zone):
        s.g.nodes[("zone", zone)]["failures"].append(t)
        for r in [n for n in s.g if n[0] == "rule"]:
            e = s.g.get_edge_data(r, ("zone", zone), key="PREFERS")
            if e:
                e["confidence"] *= 0.5
                e["valid"] = e["confidence"] >= RULE_MIN_P

    def on_preempt_outcome(s, victim_type, incoming_type, migrated):
        if victim_type is None or incoming_type is None:
            return
        a, b = ("type", victim_type), ("type", incoming_type)
        for n in (a, b):
            if n not in s.g:
                s.g.add_node(n, kind="type")
        e = s.g.get_edge_data(a, b, key="PREEMPTED")
        if e is None:
            s.g.add_edge(a, b, key="PREEMPTED", rel="PREEMPTED", n=0, migrated=0, displaced=0)
            e = s.g.get_edge_data(a, b, key="PREEMPTED")
        e["n"] += 1
        e["migrated" if migrated else "displaced"] += 1

    def add_rule(s, st, zone, confidence, t):
        r = ("rule", st)
        if r in s.g:
            s.g.remove_node(r)
        s.g.add_node(r, kind="rule", t=t)
        s.g.add_edge(r, ("zone", zone), key="PREFERS", rel="PREFERS",
                     confidence=float(confidence), valid=True)

    # ----------------------------------------------------------- reads
    def failure_risk(s, zone, t):
        return min(1.0, sum(_decay(t, tf, FAIL_HALF_LIFE_S)
                            for tf in s.g.nodes[("zone", zone)]["failures"] if tf <= t))

    def _route(s, st, origin, zone, t):
        e = s.g.get_edge_data(("ctx", st, origin), ("zone", zone), key="ROUTED_TO")
        if not e:
            return 0.0, 0.0
        d = _decay(t, e["last_t"], HALF_LIFE_S)
        return e["s"] * d, e["f"] * d

    def evidence(s, st, origin, zone, t):
        """Pooled (successes, failures) for st from origin into zone,
        split by source so the decision trace can say what decided."""
        parts = {}
        parts["direct"] = s._route(st, origin, zone, t)
        so = fo = 0.0
        for o in s.zones:
            if o == origin:
                continue
            w = OTHER_ORIGIN_W * (1.0 - s.rtt.get((o, origin), s.max_rtt) / (2 * s.max_rtt))
            a, b = s._route(st, o, zone, t)
            so, fo = so + w * a, fo + w * b
        parts["nearby_origin"] = (so, fo)
        ss = fs = 0.0
        if ("type", st) in s.g:
            for _, nb, ed in s.g.out_edges(("type", st), data=True):
                if ed.get("rel") != "SIMILAR_TO":
                    continue
                w = SIMILAR_TYPE_W * ed["sim"]
                for o in s.zones:
                    a, b = s._route(nb[1], o, zone, t)
                    po = 1.0 if o == origin else OTHER_ORIGIN_W
                    ss, fs = ss + w * po * a, fs + w * po * b
        parts["similar_type"] = (ss, fs)
        S = sum(v[0] for v in parts.values())
        F = sum(v[1] for v in parts.values())
        return S, F, parts

    def rule_conf(s, st, zone):
        e = s.g.get_edge_data(("rule", st), ("zone", zone), key="PREFERS")
        return e["confidence"] if e and e.get("valid") else 0.0

    def rank(s, st, origin, t, digests, cpu_req, stale_s=15.0, exclude_origin=True):
        """Score every candidate zone. Returns a list of dicts sorted by
        score, best first. Zones whose fresh digest shows no node with
        enough free CPU are left out (same filter flat memory applies)."""
        t0 = time.perf_counter()
        out = []
        for z in s.zones:
            if exclude_origin and z == origin:
                continue
            d = digests.get(z)
            if d is not None and t - d["t"] <= stale_s and d["top_cpu_free"] < cpu_req:
                continue
            S, F, parts = s.evidence(st, origin, z, t)
            n = S + F
            p = (S + 1.0) / (n + 2.0)
            cap = min((d or {}).get("top_cpu_free", 0.0) / 100.0, 1.0)
            rtt = s.rtt.get((origin, z), 0.0)
            risk = s.failure_risk(z, t)
            rule = s.rule_conf(st, z)
            terms = {"hist": s.w["w_hist"] * p * n / (n + 2.0),
                     "cap": s.w["w_cap"] * cap,
                     "rule": s.w["w_rule"] * rule,
                     "rtt": -s.w["w_rtt"] * rtt / s.max_rtt,
                     "fail": -s.w["w_fail"] * risk}
            out.append({"zone": z, "score": sum(terms.values()), "p": p, "n": n,
                        "rtt_ms": rtt, "risk": risk, "rule": rule,
                        "terms": terms, "parts": parts})
        out.sort(key=lambda c: -c["score"])
        s.lookup_ms.append((time.perf_counter() - t0) * 1000)
        return out

    @staticmethod
    def confident(c):
        return c["n"] >= CONFIDENT_N and c["p"] >= CONFIDENT_P

    @staticmethod
    def basis(c, ranked):
        """Which relationship decided this choice (for the decision trace):
        the evidence source that contributed most, plus whether the RTT or
        failure penalty changed the ranking."""
        src = max(c["parts"], key=lambda k: sum(c["parts"][k]))
        if sum(c["parts"][src]) == 0:
            src = "rule" if c["rule"] > 0 else "capacity"
        flags = []
        for term in ("rtt", "fail"):
            alt = max(ranked, key=lambda x: x["score"] - x["terms"][term])
            if alt["zone"] != ranked[0]["zone"]:
                flags.append(f"{term}_penalty_changed_choice")
        return src, flags

    def llm_hints(s, ranked, k=3):
        """Compact summary of the top-k zones for the GOA prompt."""
        return [{"zone": c["zone"], "success_rate": round(c["p"], 2),
                 "evidence": round(c["n"], 1), "rtt_ms": round(c["rtt_ms"], 1),
                 "recent_failure": c["risk"] > 0.2, "rule": c["rule"] > 0}
                for c in ranked[:k]]

    def victim_migrate_rate(s, victim_type):
        n = m = 0
        if ("type", victim_type) in s.g:
            for _, _, e in s.g.out_edges(("type", victim_type), data=True):
                if e.get("rel") == "PREEMPTED":
                    n += e["n"]
                    m += e["migrated"]
        return (m + 1.0) / (n + 2.0), n

    def type_totals(s, st, t):
        """(successes, failures) per zone for st over all origins, for
        rule authoring on a policy tick."""
        tot = {}
        for o in s.zones:
            for z in s.zones:
                a, b = s._route(st, o, z, t)
                if a or b:
                    x = tot.setdefault(z, [0.0, 0.0])
                    x[0] += a
                    x[1] += b
        return tot

    def size(s):
        rels = {}
        for _, _, e in s.g.edges(data=True):
            rels[e["rel"]] = rels.get(e["rel"], 0) + 1
        return {"nodes": s.g.number_of_nodes(), "edges": s.g.number_of_edges(),
                "edges_by_relation": rels}
