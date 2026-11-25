"""Multi-provider, cached, rate-limit-aware LLM caller.

Provider preference order (first available wins per call, falls through
on failure/429 with NO blocking sleep — see MultiLLM.ask):

  1. LOCAL_LLM_URL_SLM / LOCAL_LLM_URL_GOA - self-hosted, role-specific
                        (local_llm/start_server.py --also). Lets
                        MultiLLM(role="slm") hit a small/fast model and
                        MultiLLM(role="goa") hit a larger/reasoning model
                        instead of both sharing one server.
  2. LOCAL_LLM_URL   - self-hosted (local_llm/), single server shared by
                        every role, no rate limit. Preferred ahead of
                        hosted tiers so a correctly-provisioned GPU
                        machine never touches them at all. Kept for
                        backward compatibility with the single-server
                        setup this project originally shipped with.
  3. GROQ_API_KEY     - hosted, free tier, fast, tightly rate-limited.
  4. CEREBRAS_API_KEY - hosted, free tier (some models need payment).
  5. OPENROUTER_API_KEY - hosted, free tier, last resort.

Set env vars to enable whichever you have; any subset works. With none
set, main.py / phase runners will refuse to start the LLM-dependent
phases (Phase 4/5) with a clear error, rather than hanging.

If a local server is running a model whose chat template has no system
role (confirmed for Gemma via llama-cpp-python's format_gemma(), which
hardcodes system_message="" and silently drops it - see
local_llm/model_compare.py's diagnostic scripts), set the matching
*_NO_SYSTEM_ROLE=1 env var (LOCAL_LLM_URL_NO_SYSTEM_ROLE /
LOCAL_LLM_URL_SLM_NO_SYSTEM_ROLE / LOCAL_LLM_URL_GOA_NO_SYSTEM_ROLE) so
MultiLLM.ask() folds the system prompt into the user turn instead of a
separate system message that would otherwise never reach the model.
"""
import hashlib
import json
import os
import time
from collections import defaultdict

import requests as http_requests

PROVIDERS = []

# Caps generation length across every provider/model. Without this, a
# model that doesn't tightly follow "respond with ONLY a JSON object"
# (verified against Mistral-7B-Instruct-v0.3 via local_llm/model_compare.py
# - Qwen2.5/Llama-3.2 stayed terse without it) can ramble for hundreds of
# extra tokens, which both wastes latency and makes it more likely the
# response contains more than one brace-delimited chunk of text.
MAX_TOKENS = 400


def extract_json_object(raw):
    """Extract the first balanced {...} object from a raw completion.

    Replaces a naive `re.search(r"\\{.*\\}", raw, re.S)`, which is greedy
    from the FIRST '{' to the LAST '}' in the whole string - fine for a
    model that outputs nothing but the JSON object, but it silently
    slurps up any later prose (or an echoed schema example) that happens
    to contain its own braces, producing a string that isn't valid JSON.
    This scans forward from the first '{', tracking nesting depth and
    skipping over quoted-string content (so a literal '{' or '}' inside
    a string value doesn't miscount), and returns as soon as the first
    object closes - i.e. the FIRST complete JSON object, ignoring
    whatever trailing text (or a second example object) follows it."""
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no '{{' found in response: {raw!r}")
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    raise ValueError(f"no balanced JSON object found in response: {raw!r}")


def _get_key(name):
    return os.environ.get(name, "")


def discover_providers():
    """Populate the module-level PROVIDERS list from environment
    variables. Call once at startup (main.py / evaluator.py do this).

    Each provider dict may carry a "roles" key: None/absent means it's
    available to every MultiLLM role (the original, backward-compatible
    behavior); a list like ["slm"] restricts it to that role. See
    MultiLLM.ask() for how roles affect provider ordering."""
    PROVIDERS.clear()

    # Role-specific local servers (local_llm/start_server.py --also) take
    # priority over the single shared LOCAL_LLM_URL when both are set,
    # since they're the more specific configuration.
    for env_name, role in [("LOCAL_LLM_URL_SLM", "slm"),
                           ("LOCAL_LLM_URL_GOA", "goa")]:
        url = os.environ.get(env_name, "")
        if url:
            no_sys = os.environ.get(f"{env_name}_NO_SYSTEM_ROLE", "") not in (
                "", "0", "false", "False")
            PROVIDERS.append({
                "key": "local", "roles": [role], "local": True,
                "no_system_role": no_sys,
                "url": f"{url.rstrip('/')}/v1/chat/completions",
                "models": ["local"], "name": env_name,
            })
            print(f"  provider: {env_name} ({url}) [preferred for role={role}]"
                  + (" [no system role - e.g. Gemma]" if no_sys else ""))

    local_url = os.environ.get("LOCAL_LLM_URL", "")
    if local_url:
        no_sys = os.environ.get("LOCAL_LLM_URL_NO_SYSTEM_ROLE", "") not in (
            "", "0", "false", "False")
        PROVIDERS.append({
            "key": "local", "roles": None, "local": True,
            "no_system_role": no_sys,
            "url": f"{local_url.rstrip('/')}/v1/chat/completions",
            "models": ["local"],
            "name": "LOCAL_LLM_URL",
        })
        print(f"  provider: LOCAL_LLM_URL ({local_url}) [preferred]"
              + (" [no system role - e.g. Gemma]" if no_sys else ""))

    for name, url, models in [
        ("GROQ_API_KEY",
         "https://api.groq.com/openai/v1/chat/completions",
         ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]),
        ("CEREBRAS_API_KEY",
         "https://api.cerebras.ai/v1/chat/completions",
         ["llama-3.3-70b"]),
        ("OPENROUTER_API_KEY",
         "https://openrouter.ai/api/v1/chat/completions",
         ["openai/gpt-oss-20b:free", "google/gemma-4-31b-it:free"]),
    ]:
        k = _get_key(name)
        if k:
            PROVIDERS.append({"key": k, "url": url, "models": models,
                              "name": name})
            print(f"  provider: {name} ({len(models)} models)")

    return PROVIDERS


class MultiLLM:
    """Tries providers in order; caches on disk; on a 429 falls through
    to the next model/provider immediately (no blocking sleep) rather
    than waiting out the hosted tier's retry-after, which can be
    10+ minutes on a saturated free tier."""

    def __init__(self, cache_path, role="llm"):
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(
            cache_path) else {}
        self.calls = self.cache_hits = self.tokens = 0
        self.latencies_ms = []
        self.role = role

    def ask(self, system, user, cache_key=None):
        h = hashlib.md5((cache_key or (system + user)).encode()).hexdigest()
        if h in self.cache:
            self.cache_hits += 1
            return self.cache[h]
        # Role-specific local servers (roles=[...]) come first when they
        # match self.role; a generic provider (roles=None, e.g. the
        # single-server LOCAL_LLM_URL setup or any hosted API) is always
        # eligible and keeps its discover_providers() append order.
        # Stable sort preserves relative order within each group.
        provs = sorted(PROVIDERS, key=lambda p:
                       0 if (p.get("roles") is None or self.role in p["roles"])
                       else 1)
        for prov in provs:
            # A server whose chat template has no system role (confirmed
            # for Gemma - see module docstring) would otherwise silently
            # drop `system` entirely, so the model never sees it at all.
            # Folding both into one user turn is the standard workaround
            # and is what actually reaches the model in that case.
            if prov.get("no_system_role"):
                msgs = [{"role": "user", "content": system + "\n\n" + user}]
            else:
                msgs = [{"role": "system", "content": system},
                       {"role": "user", "content": user}]
            for model in prov["models"]:
                try:
                    t0 = time.time()
                    payload = {"model": model, "messages": msgs,
                              "temperature": 0, "max_tokens": MAX_TOKENS}
                    # local llama.cpp server's json_object grammar mode
                    # hangs rather than erroring - skip it there (any
                    # local provider, not just one literally named
                    # "LOCAL_LLM_URL" - see discover_providers()); the
                    # balanced-brace JSON extraction below handles
                    # providers that don't enforce a JSON response too.
                    if not prov.get("local"):
                        payload["response_format"] = {"type": "json_object"}
                    r = http_requests.post(prov["url"], timeout=90,
                        headers={"Authorization": f"Bearer {prov['key']}",
                                 "Content-Type": "application/json"},
                        json=payload)
                    if r.status_code == 429:
                        wait = float(r.headers.get("retry-after", 10))
                        print(f"  rate-limited on {prov['name']}/{model} "
                              f"(would need {wait}s) - trying next "
                              f"model/provider instead")
                        continue
                    r.raise_for_status()
                    j = r.json()
                    self.tokens += j.get("usage", {}).get("total_tokens", 0)
                    raw = j["choices"][0]["message"]["content"]
                    self.latencies_ms.append((time.time() - t0) * 1000)
                    self.calls += 1
                    out = json.loads(extract_json_object(raw))
                    self.cache[h] = out
                    self.flush()
                    return out
                except Exception as e:
                    print(f"  {prov['name']}/{model} failed: {e}")
                    continue
        raise RuntimeError(f"all providers failed for {self.role}")

    def flush(self):
        json.dump(self.cache, open(self.cache_path, "w"))
