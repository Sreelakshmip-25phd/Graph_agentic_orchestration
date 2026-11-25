"""Phase 2 — discrete-event simulation core.

Min-heap event scheduler, node runtime state, zone capacity tables, and
the deterministic placement solver. No LLM/agent logic lives here — see
zoa_agent.py (local placement) and goa_agent.py (global orchestration,
memory, LLM escalation) for that.
"""
import heapq

DEFAULT_POLICY = {"w_fit": 0.45, "w_balance": 0.35, "w_server_pref": 0.20,
                  "constraints": {}}

# Floating-point tolerance for capacity comparisons. cpu_used/mem_used
# accumulate through many small += / -= operations across a run (each
# preemption+migration is 2 more), so exact >= comparisons occasionally
# reject an allocation that's actually a perfect fit (e.g. free=12.7999...
# vs need=12.8) purely from rounding drift, not real over-commitment.
EPS = 1e-6


def solve_placement(req, profile, table, policy=None):
    """3-stage solver: hard-constraint filter -> weighted scoring ->
    policy-modulated weights (weights come from the zone's current
    GOA-issued policy, or DEFAULT_POLICY)."""
    p = {**DEFAULT_POLICY, **(policy or {})}
    cpu, mem = req["cpu_req"], req["mem_gb"]
    cands = {n: s for n, s in table.items()
             if s["healthy"] and s["cpu_free"] >= cpu - EPS
             and s["mem_free"] >= mem - EPS
             and s["type"] not in p["constraints"].get("forbid_types", [])}
    if profile.get("latency_class") == "realtime":
        srv = {n: s for n, s in cands.items() if s["type"] == "edge_server"}
        if srv:
            cands = srv
    if not cands:
        return None
    best, bs = None, -1e18
    for nid, s in sorted(cands.items()):
        cap = 100.0 if s["type"] == "edge_server" else 40.0
        sc = (p["w_fit"] * (1 - (s["cpu_free"] - cpu) / cap)
              + p["w_balance"] * (s["cpu_free"] / cap)
              + p["w_server_pref"] * (s["type"] == "edge_server"))
        if sc > bs:
            best, bs = nid, sc
    return best


class Sim:
    """Min-heap discrete-event scheduler."""
    def __init__(self):
        self.t, self._q, self._seq = 0.0, [], 0

    def at(self, t, fn, *a):
        heapq.heappush(self._q, (float(t), self._seq, fn, a))
        self._seq += 1

    def run(self):
        while self._q:
            t, _, fn, a = heapq.heappop(self._q)
            self.t = t
            fn(*a)


class Node:
    """A physical node (edge_server or edge_node) inside a zone."""
    def __init__(s, spec, zid):
        s.node_id, s.type, s.zone_id = spec["node_id"], spec["type"], zid
        s.cpu_cap, s.mem_cap = float(spec["cpu"]), float(spec["mem_gb"])
        s.cpu_used = s.mem_used = 0.0
        s.healthy = True
        s.services = {}  # req_id -> (cpu, mem)

    cpu_free = property(lambda s: s.cpu_cap - s.cpu_used)
    mem_free = property(lambda s: s.mem_cap - s.mem_used)

    def can_fit(s, c, m):
        return s.healthy and s.cpu_free >= c - EPS and s.mem_free >= m - EPS

    def alloc(s, i, c, m):
        assert s.can_fit(c, m), (
            f"over-alloc {s.node_id}: req={i} needs cpu={c} mem={m}, "
            f"free cpu={s.cpu_free} mem={s.mem_free}, healthy={s.healthy}, "
            f"running={list(s.services.keys())}")
        s.cpu_used += c
        s.mem_used += m
        s.services[i] = (c, m)

    def release(s, i):
        c, m = s.services.pop(i)
        s.cpu_used -= c
        s.mem_used -= m
        return c, m

    def fail(s):
        s.healthy = False
        d = list(s.services)
        s.services.clear()
        s.cpu_used = s.mem_used = 0.0
        return d

    def recover(s):
        s.healthy = True


class ZoneRuntime:
    """Per-zone capacity table + digest generation for cross-zone
    exchange (<500 bytes, broadcast every DIGEST_S seconds)."""
    def __init__(s, zid):
        s.zone_id = zid
        s.table = {}

    @staticmethod
    def _row(t, n, soft=False):
        return {"cpu_free": n.cpu_free, "mem_free": n.mem_free,
                "healthy": n.healthy, "soft_anomaly": soft,
                "type": n.type, "t": t}

    def register(s, n):
        s.table[n.node_id] = s._row(0, n)

    def push(s, t, n):
        soft = s.table[n.node_id]["soft_anomaly"]
        s.table[n.node_id] = s._row(t, n, soft)

    def on_hard_failure(s, t, nid):
        s.table[nid].update({"healthy": False, "cpu_free": 0,
                             "mem_free": 0, "t": t})

    def snapshot(s):
        return {k: dict(v) for k, v in s.table.items()}

    def digest(s, t, prev):
        h = [v for v in s.table.values() if v["healthy"]]
        cur = sum(v["cpu_free"] for v in h)
        return {"zone": s.zone_id, "t": t, "cpu_free": round(cur, 1),
                "mem_free": round(sum(v["mem_free"] for v in h), 1),
                "top_cpu_free": round(max((v["cpu_free"] for v in h),
                                          default=0), 1),
                "trend": round(cur - prev, 1),
                "healthy_frac": round(len(h) / max(len(s.table), 1), 2)}
