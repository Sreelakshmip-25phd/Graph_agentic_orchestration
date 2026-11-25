"""Global Orchestration Agent (GOA) with a selectable memory.

Escalation chain (unchanged from the flat-memory study):
  memory fast path -> cross-zone digest scan -> LLM zone selection
  -> preemption (LLM picks the victim) -> degradation -> unresolved.

`memory` selects what the GOA remembers and consults:
  "flat"  - EpisodicMemory + ProceduralStore keyed by service type only.
            This code path is the earlier study's GOA, unchanged.
  "graph" - GraphMemory (graph_memory.py): routes per (type, origin zone),
            zone RTT links, similar types, zone failures, preemption
            outcomes and rules. Used for the fast path, as context in the
            LLM prompt, and to rank preemption victims.
  "none"  - no memory fast path and no digest scan; escalations go to the
            LLM (the earlier study's memory_ablated system, unchanged).

Why preemption exists: without it the LLM only picks a target zone from a
capacity digest, which a sort already solves. Choosing which running,
lower-priority service to evict (priority gap, remaining lifetime,
service type) is a reasoning task a deterministic solver cannot do. The
evicted service goes through replan() to migrate it elsewhere first.
"""
import json
from collections import defaultdict, deque

import pandas as pd

from graph_memory import GraphMemory, RULE_MIN_N, RULE_MIN_P

PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2, "critical": 3}

DIGEST_S, STALE_S, MAX_XZ_CAND = 5.0, 15.0, 3
POLICY_TICK_S, EPISODIC_WINDOW = 120.0, 20
RULE_MIN_EPS, RULE_MIN_FRAC = 5, 0.6
MAX_PREEMPT_CANDIDATES = 3


RTT_PROMPT_NOTE = """
"rtt_ms_from_origin" is the network delay from the request's origin zone to
each zone. A zone whose rtt_ms_from_origin + 1 ms exceeds the request's
latency_ms breaks its latency SLA; avoid such zones."""


def goa_decide_sys(semantic, graph=False, rtt=False):
    extra = ("""
"graph_memory" lists the top candidate zones from the coordinator's
learned memory: success_rate for this service type and origin, how much
evidence backs it, RTT from the origin, whether the zone failed recently,
and whether a learned rule prefers it. Use it together with the digests."""
             if graph else "")
    if rtt:
        extra += RTT_PROMPT_NOTE
    return f"""You are the global coordinator of an edge-cloud
orchestrator. Given a service request and per-zone capacity digests,
choose the best target zone.{extra} Respond ONLY with JSON:
{{"target_zone": "<zone_id>", "justification": "<one sentence>"}}
Known profiles: {json.dumps(semantic)}"""


GOA_RULE_SYS = """Author ONE rule from episodic evidence. Respond ONLY JSON:
{"condition": {"service_type": "<type>"},
 "action": {"prefer_zone": "<zone_id>"},
 "rationale": "<one sentence>"}
Reflect the majority successful zone."""

PREEMPT_SYS = """You are the global coordinator of an edge-cloud
orchestrator. A high-priority request cannot be placed because the
target node is full. You must decide whether to evict one currently
running, lower-priority service to make room, reasoning about:
  - priority gap (bigger gap = safer to evict)
  - remaining lifetime (evicting something almost finished wastes less
    work than evicting something that just started)
  - service type (some types tolerate interruption better than others)
Respond ONLY with JSON:
{"evict_req_id": "<req_id or null>", "justification": "<one sentence>"}
Only choose from the candidates given. Return null if none are worth
evicting relative to the incoming request's priority."""

PREEMPT_GRAPH_NOTE = """
Each candidate's past_migration_rate is how often services of that type
were successfully migrated elsewhere after being evicted before (higher
= less harm from evicting it)."""


class EpisodicMemory:
    def __init__(s, n=EPISODIC_WINDOW):
        s.store = defaultdict(lambda: deque(maxlen=n))

    def record(s, st, z, ok, t):
        s.store[st].append({"zone": z, "ok": bool(ok), "t": round(t, 1)})

    def best_zone(s, st):
        eps = [e for e in s.store[st] if e["ok"]]
        if not eps:
            return None, 0, 0.0
        zc = pd.Series([e["zone"] for e in eps]).value_counts()
        return zc.index[0], int(zc.iloc[0]), round(zc.iloc[0] / len(eps), 2)

    def dump(s):
        return {k: list(v) for k, v in s.store.items()}


class ProceduralStore:
    def __init__(s):
        s.rules = {}

    def match(s, st):
        return s.rules.get(st)

    def add(s, rule):
        s.rules[rule["condition"]["service_type"]] = rule

    def dump(s):
        return list(s.rules.values())


class GlobalOrchestrationAgent:
    def __init__(s, lib, slm, llm, topo, semantic, solve_placement,
                 memory="flat", graph_weights=None, type_sims=None,
                 rtt_in_prompt=False, digest_rtt_aware=False):
        assert memory in ("flat", "graph", "none"), memory
        s.lib, s.slm, s.llm, s.topo = lib, slm, llm, topo
        s.semantic = semantic
        s.solve_placement = solve_placement
        s.memory = memory
        s.use_memory = memory != "none"
        s.use_preemption = True
        s.policy = defaultdict(dict)
        s.epi, s.proc = EpisodicMemory(), ProceduralStore()
        s.gm = (GraphMemory(topo, semantic, graph_weights, type_sims)
                if memory == "graph" else None)
        s.digest_cache, s._prev = {}, defaultdict(float)
        s.degraded = {}
        s.preempted = {}  # req_id -> victim req_id, consumed by evaluator
        s.metrics = defaultdict(int)
        s.trace = defaultdict(int)  # graph decision trace: what decided
        s.mem_first_by_type = defaultdict(lambda: [0, 0])  # st -> [tries, ok]
        s.zone_ids = [z["zone_id"] for z in topo["zones"]]
        # Stress, Option B: every condition's zone-choice prompt carries the
        # origin's RTT row. Off for W-ref (the earlier study's prompt).
        s.rtt_in_prompt = rtt_in_prompt
        # Stress option: the cross-zone digest scan skips zones whose RTT
        # from the origin would break the request's latency budget (same
        # for every condition). Off for W-ref (the earlier study's scan).
        s.digest_rtt_aware = digest_rtt_aware
        s.intra_ms = float(topo.get("intra_zone_ms", 1.0))
        s.rtt = {}
        for key, ms in topo["inter_zone_rtt_ms"].items():
            a, b = key.split("-")
            s.rtt[(a, b)] = s.rtt[(b, a)] = float(ms)
        s.goa_decide_sys = goa_decide_sys(semantic, graph=memory == "graph",
                                          rtt=rtt_in_prompt)

    # ------------------------------------------------------------ memory
    def _remember(s, req, st, zone, ok, t):
        """Store a placement outcome in the active memory. For flat (and
        none) this is exactly the earlier study's epi.record() call."""
        if s.gm is None:
            s.epi.record(st, zone, ok, t)
            return
        s.gm.record(st, req["origin_zone"], zone, ok, t)
        if ok:
            s.gm.note_placed(req["req_id"], st, req["origin_zone"], zone)

    def _remember_unresolved(s, req, st, t):
        """Nothing worked. Flat memory records a failure against the
        origin zone (earlier behaviour, kept). Graph memory has already
        recorded a failure for each zone it actually tried."""
        if s.gm is None:
            s.epi.record(st, req["origin_zone"], False, t)

    def _breaks_budget(s, req, zone):
        if not s.digest_rtt_aware or req.get("latency_ms") is None:
            return False
        rtt = 0.0 if zone == req["origin_zone"] else s.rtt.get((req["origin_zone"], zone), 0.0)
        return s.intra_ms + rtt > req["latency_ms"]

    def _mem_try(s, first, ok, st=None):
        """Instrumentation only: memory precision / hit rate, also per
        service type (Stress B8)."""
        s.metrics["mem_attempts"] += 1
        s.metrics["mem_success"] += int(ok)
        if first:
            s.metrics["mem_first_attempts"] += 1
            s.metrics["mem_first_success"] += int(ok)
            if st is not None:
                s.mem_first_by_type[st][0] += 1
                s.mem_first_by_type[st][1] += int(ok)

    # --------------------------------------------------------- plumbing
    def on_digest_tick(s, t, zones):
        for zid, zr in zones.items():
            d = zr.digest(t, s._prev[zid])
            s._prev[zid] = d["cpu_free"]
            s.digest_cache[zid] = d

    def _ask_llm(s, t, kind, sys_prompt, user, cache_key):
        s.metrics["llm_asks"] += 1
        s.metrics[f"llm_asks_{kind}"] += 1
        s.metrics["llm_prompt_chars"] += len(sys_prompt) + len(user)
        c0 = s.llm.calls
        out = s.llm.ask(sys_prompt, user, cache_key=cache_key)
        if s.llm.calls > c0:
            s.metrics["fresh_llm_calls"] += 1
        return out

    def _profile(s, req, required_fields, slm_system, t=0.0):
        prof, _ = s.lib.lookup(req["intent_text"])
        if prof is None:
            c0 = s.slm.calls
            out = s.slm.ask(slm_system, "INPUT: " + req["intent_text"],
                            cache_key="slm|" + req["intent_text"])
            prof = {k: out.get(k) for k in required_fields}
            s.lib.add(req["intent_text"], prof)
            s.metrics["slm_compiles"] += 1
            if s.slm.calls > c0:
                s.metrics["fresh_slm_calls"] += 1
        else:
            s.metrics["cache_hits"] += 1
        return prof

    def _try_zone(s, req, prof, zid, tables):
        return s.solve_placement(req, prof, tables[zid],
                                 s.policy.get(zid)) if zid in tables else None

    # ------------------------------------------------------- preemption
    def _try_preempt(s, t, req, prof, zid, tables):
        """Look for a lower-priority running service on the candidate
        zone's nodes whose eviction would free enough capacity. Returns
        node_id if a preemption was chosen (recorded in s.preempted for
        the evaluator to carry out), else None."""
        if not s.use_preemption or zid not in tables:
            return None
        req_rank = PRIORITY_RANK.get(req.get("priority", "normal"), 1)
        if req_rank < PRIORITY_RANK["high"]:
            return None  # only high/critical requests may preempt

        cpu_need, mem_need = req["cpu_req"], req["mem_gb"]
        candidates = []  # (node_id, victim_req_id, priority, remaining_s, type)
        for nid, row in tables[zid].items():
            if not row.get("healthy"):
                continue
            for victim in row.get("running", []):
                v_rank = PRIORITY_RANK.get(victim["priority"], 1)
                if v_rank >= req_rank:
                    continue  # only evict strictly lower priority
                freed_cpu = row["cpu_free"] + victim["cpu"]
                freed_mem = row["mem_free"] + victim["mem"]
                if freed_cpu >= cpu_need and freed_mem >= mem_need:
                    candidates.append((nid, victim["req_id"], victim["priority"],
                                       victim["remaining_s"], victim["service_type"]))
        if not candidates:
            return None

        rate = {}
        if s.gm is not None:
            # graph: prefer victims whose type has usually migrated
            # successfully after eviction, then the one nearest completion
            rate = {c[4]: s.gm.victim_migrate_rate(c[4])[0] for c in candidates}
            candidates.sort(key=lambda c: (-rate[c[4]], c[3]))
        else:
            candidates.sort(key=lambda c: c[3])  # nearest completion first
        candidates = candidates[:MAX_PREEMPT_CANDIDATES]

        chosen = candidates[0]  # heuristic fallback
        if len(candidates) > 1:
            try:
                cands = [{"req_id": c[1], "priority": c[2],
                          "remaining_s": round(c[3], 1), "service_type": c[4]}
                         for c in candidates]
                sys_prompt, key_extra = PREEMPT_SYS, ""
                if s.gm is not None:
                    for c in cands:
                        c["past_migration_rate"] = round(rate[c["service_type"]], 2)
                    sys_prompt = PREEMPT_SYS + PREEMPT_GRAPH_NOTE
                    key_extra = "|g|" + "|".join(
                        str(c["past_migration_rate"]) for c in cands)
                user = json.dumps({
                    "incoming_request": {"priority": req["priority"],
                                         "service_type": prof.get("service_type")},
                    "eviction_candidates": cands})
                out = s._ask_llm(t, "preempt", sys_prompt, user,
                    cache_key=f"preempt|{req['priority']}|" +
                    "|".join(c[1] for c in candidates) + key_extra)
                evict_id = out.get("evict_req_id")
                match = next((c for c in candidates if c[1] == evict_id), None)
                if match:
                    chosen = match
            except Exception:
                pass  # keep heuristic fallback

        nid, victim_rid = chosen[0], chosen[1]
        s.preempted[req["req_id"]] = victim_rid
        s.metrics["preemptions"] += 1
        return nid

    # ------------------------------------------------------- escalation
    def _graph_fast_path(s, t, req, prof, st, tables):
        """Graph memory fast path (replaces the episodic + procedural
        lookup). Returns (node_id or None, ranked candidates)."""
        ranked = s.gm.rank(st, req["origin_zone"], t, s.digest_cache, req["cpu_req"])
        first = True
        for c in [c for c in ranked if s.gm.confident(c)][:MAX_XZ_CAND]:
            nid = s._try_zone(req, prof, c["zone"], tables)
            s._mem_try(first, nid is not None, st)
            first = False
            if nid:
                src, flags = s.gm.basis(c, ranked)
                s.trace[f"basis_{src}"] += 1
                for f in flags:
                    s.trace[f] += 1
                s._remember(req, st, c["zone"], True, t)
                return nid, ranked
            s.gm.record(st, req["origin_zone"], c["zone"], False, t)
        return None, ranked

    def _escalate(s, t, req, prof, tables):
        """Returns (node_id_or_None, path_label)."""
        st = prof.get("service_type") or req.get("service_type", "unknown")
        s.metrics["escalations"] += 1
        ranked = None
        if s.use_memory:
            if s.gm is not None:
                nid, ranked = s._graph_fast_path(t, req, prof, st, tables)
                if nid:
                    s._log(t, st, "graph")
                    return nid, "graph"
            else:
                first = True
                zone, cnt, frac = s.epi.best_zone(st)
                if zone and cnt >= 3:
                    d = s.digest_cache.get(zone)
                    if d and d["top_cpu_free"] >= req["cpu_req"]:
                        nid = s._try_zone(req, prof, zone, tables)
                        s._mem_try(first, nid is not None, st)
                        first = False
                        if nid:
                            s.epi.record(st, zone, True, t)
                            s._log(t, st, "episodic")
                            return nid, "episodic"
                rule = s.proc.match(st)
                if rule:
                    nid = s._try_zone(req, prof, rule["action"]["prefer_zone"], tables)
                    s._mem_try(first, nid is not None, st)
                    if nid:
                        s.epi.record(st, rule["action"]["prefer_zone"], True, t)
                        s._log(t, st, "procedural")
                        return nid, "procedural"
            # Cross-zone digest scan: part of the memory fast path and
            # gated the same way, so ablating memory sends these decisions
            # to the LLM instead of an equally capable substitute.
            cands = sorted((d for z, d in s.digest_cache.items()
                            if z != req["origin_zone"] and t - d["t"] <= STALE_S
                            and d["top_cpu_free"] >= req["cpu_req"]
                            and not s._breaks_budget(req, d["zone"])),
                           key=lambda d: -d["top_cpu_free"])
            for d in cands[:MAX_XZ_CAND]:
                nid = s._try_zone(req, prof, d["zone"], tables)
                if nid:
                    s._remember(req, st, d["zone"], True, t)
                    s._log(t, st, "digest")
                    return nid, "digest"

        # LLM block: every decision reaching here counts, cached or not.
        s.metrics["reached_llm_block"] += 1
        digests = dict(s.digest_cache)
        payload = {"request": {k: req.get(k) for k in
                               ("cpu_req", "mem_gb", "latency_ms",
                                "priority", "origin_zone")},
                   "profile": prof, "zone_digests": digests}
        key_extra = ""
        if s.rtt_in_prompt:
            o = req["origin_zone"]
            payload["rtt_ms_from_origin"] = {
                z: (0.0 if z == o else s.rtt.get((o, z))) for z in s.zone_ids}
            key_extra += f"|o:{o}|lat:{req.get('latency_ms')}"
        if s.gm is not None:
            if ranked is None:
                ranked = s.gm.rank(st, req["origin_zone"], t, s.digest_cache,
                                   req["cpu_req"])
            hints = s.gm.llm_hints(ranked)
            payload["graph_memory"] = hints
            key_extra += "|g|" + json.dumps(hints, sort_keys=True)
        try:
            out = s._ask_llm(t, "goa_decide", s.goa_decide_sys, json.dumps(payload),
                cache_key=f"goa|{st}|{req['cpu_req']:.0f}|" +
                "|".join(f"{z}:{d['top_cpu_free']:.0f}"
                         for z, d in sorted(digests.items())) + key_extra)
            tz = out.get("target_zone")
        except Exception:
            tz = None
        order = ([tz] if tz in s.zone_ids else []) + sorted(
            s.zone_ids, key=lambda z: -s.digest_cache.get(z, {})
            .get("top_cpu_free", 0))
        for zid in dict.fromkeys(order):
            nid = s._try_zone(req, prof, zid, tables)
            if nid:
                s._remember(req, st, zid, True, t)
                s._log(t, st, "llm")
                return nid, "llm"
        # preemption: evict a lower-priority running service instead of
        # just degrading the incoming request's own resources
        for zid in dict.fromkeys(order):
            nid = s._try_preempt(t, req, prof, zid, tables)
            if nid:
                s._remember(req, st, zid, True, t)
                s._log(t, st, "preempt")
                return nid, "preempt"
        if prof.get("latency_class") == "realtime" or \
                prof.get("priority") in ("high", "critical"):
            dreq = {**req, "cpu_req": round(req["cpu_req"] * 0.6, 1),
                    "mem_gb": round(req["mem_gb"] * 0.8, 2)}
            for zid in dict.fromkeys(order):
                nid = s._try_zone(dreq, prof, zid, tables)
                if nid:
                    s.degraded[req["req_id"]] = (dreq["cpu_req"], dreq["mem_gb"])
                    s._remember(req, st, zid, True, t)
                    s._log(t, st, "llm_degraded")
                    return nid, "llm_degraded"
        s._remember_unresolved(req, st, t)
        s._log(t, st, "llm_unresolved")
        return None, "llm_unresolved"

    def _degrade_local(s, t, req, prof, tables):
        """zone_local requests that don't fit at home: preempt or degrade
        locally. Returns (node_id_or_None, path_label)."""
        st = prof.get("service_type") or req.get("service_type", "unknown")
        if prof.get("latency_class") == "realtime" or \
                prof.get("priority") in ("high", "critical"):
            s.metrics["escalations"] += 1
            s.metrics["reached_llm_block"] += 1
            nid = s._try_preempt(t, req, prof, req["origin_zone"], tables)
            if nid:
                s._remember(req, st, req["origin_zone"], True, t)
                s._log(t, st, "preempt_local")
                return nid, "preempt_local"
            dreq = {**req, "cpu_req": round(req["cpu_req"] * 0.6, 1),
                    "mem_gb": round(req["mem_gb"] * 0.8, 2)}
            nid = s._try_zone(dreq, prof, req["origin_zone"], tables)
            if nid:
                s.degraded[req["req_id"]] = (dreq["cpu_req"], dreq["mem_gb"])
                s._remember(req, st, req["origin_zone"], True, t)
                s._log(t, st, "llm_degraded")
                return nid, "llm_degraded"
            s._remember_unresolved(req, st, t)
            s._log(t, st, "llm_unresolved")
            return None, "llm_unresolved"
        s.metrics["locality_bound_reject"] += 1
        return None, "locality_bound_reject"

    def _log(s, t, st, path):
        s.metrics[f"esc_{path}"] += 1

    def place(s, t, req, zone_tables, topo, required_fields, slm_system):
        """Returns (node_id_or_None, path_label). "local" = the first
        same-zone solve succeeded with no escalation."""
        prof = s._profile(req, required_fields, slm_system, t)
        nid = s._try_zone(req, prof, req["origin_zone"], zone_tables)
        if nid is not None:
            return nid, "local"
        if prof.get("data_locality") != "zone_local":
            nid, path = s._escalate(t, req, prof, zone_tables)
            if nid:
                s.metrics["cross_zone"] += 1
            return nid, path
        return s._degrade_local(t, req, prof, zone_tables)

    # ------------------------------------------------------ policy tick
    def on_policy_tick(s, t):
        if s.gm is not None:
            return s._graph_policy_tick(t)
        # flat and none: the earlier study's behaviour, unchanged (none
        # still authors rules from its episodic log, it just never
        # consults them)
        for st in list(s.epi.store):
            if s.proc.match(st):
                continue
            zone, cnt, frac = s.epi.best_zone(st)
            if zone and cnt >= RULE_MIN_EPS and frac >= RULE_MIN_FRAC:
                rule = None
                try:
                    rule = s._ask_llm(t, "rule_author", GOA_RULE_SYS,
                        json.dumps({"service_type": st,
                                    "evidence": s.epi.dump()[st]}),
                        cache_key=f"rule|{st}|{zone}|{cnt}")
                except Exception:
                    rule = None
                if not isinstance(rule, dict) or rule.get("action", {}).get(
                        "prefer_zone") != zone:
                    rule = {"condition": {"service_type": st},
                            "action": {"prefer_zone": zone},
                            "rationale": "episodic majority (validated)"}
                rule.setdefault("condition", {"service_type": st})
                rule.update(confidence=frac,
                            provenance={"episodes": cnt, "t": round(t, 1)})
                s.proc.add(rule)
                s.metrics["rules_authored"] += 1

    def _graph_policy_tick(s, t):
        """Same authoring bar as flat memory (>= 5 successes, >= 60% of a
        type's successes in one zone), with evidence read from the graph.
        A rule is re-authored only after a failure has invalidated it."""
        types = {n[1] for n in s.gm.g if n[0] == "ctx"}
        for st in sorted(types):
            r = ("rule", st)
            if r in s.gm.g and any(e.get("valid") for _, _, e in
                                   s.gm.g.out_edges(r, data=True)):
                continue
            tot = s.gm.type_totals(st, t)
            succ = sum(v[0] for v in tot.values())
            if not tot or succ <= 0:
                continue
            zone = max(tot, key=lambda z: tot[z][0])
            cnt, frac = tot[zone][0], tot[zone][0] / succ
            if cnt < RULE_MIN_N or frac < RULE_MIN_P:
                continue
            evidence = {z: {"successes": round(a, 1), "failures": round(b, 1)}
                        for z, (a, b) in sorted(tot.items())}
            try:
                rule = s._ask_llm(t, "rule_author", GOA_RULE_SYS,
                    json.dumps({"service_type": st, "evidence": evidence}),
                    cache_key=f"grule|{st}|{zone}|{cnt:.0f}")
            except Exception:
                rule = None
            if not isinstance(rule, dict) or rule.get("action", {}).get(
                    "prefer_zone") != zone:
                s.metrics["rules_llm_overridden"] += 1
            s.gm.add_rule(st, zone, frac, t)
            s.metrics["rules_authored"] += 1

    # ------------------------------------------------- failure / replan
    def replan(s, t, req, tables, required_fields, slm_system):
        """Re-place a displaced/evicted request through the same
        escalation chain. Used for node failures and preemption victims."""
        prof = s._profile(req, required_fields, slm_system, t)
        s.metrics["replans"] += 1
        nid = s._try_zone(req, prof, req["origin_zone"], tables)
        if nid is not None:
            return nid, "local"
        return s._escalate(t, req, prof, tables)

    def on_failure(s, t, zone_id):
        s.policy[zone_id] = {"w_fit": 0.2, "w_balance": 0.7, "w_server_pref": 0.1}
        s.metrics["surge_policies"] += 1
        if s.gm is not None:
            s.gm.on_zone_failure(t, zone_id)

    # -------------------------------------------- outcome hooks (graph)
    def on_outcome(s, t, req_id, outcome):
        """'completed' or 'displaced'. Only graph memory learns from it."""
        if s.gm is not None:
            s.gm.on_outcome(t, req_id, outcome)

    def on_preempt_outcome(s, t, victim_req, incoming_req, migrated):
        if s.gm is not None:
            s.gm.on_preempt_outcome(victim_req.get("service_type"),
                                    incoming_req.get("service_type"), migrated)
