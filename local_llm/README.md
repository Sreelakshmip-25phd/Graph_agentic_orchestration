# local_llm

Self-hosted, OpenAI-compatible LLM server for this project's evaluation
pipeline (`../main.py` / `../src/evaluator.py`) — **the preferred LLM
provider by default**, ahead of the hosted free tiers (Groq/Cerebras/
OpenRouter), because it has no rate limit.

Model weights are **not** committed to this repo (see `.gitignore`) —
`download_model.py` fetches them fresh on whichever machine runs this.

## Quick start

```bash
pip install -r requirements.txt
python3 start_server.py medium          # downloads model + starts server
# in another shell / tmux pane:
python3 test_server.py                  # smoke test + tokens/sec report
```

Then point the pipeline at it:

```bash
export LOCAL_LLM_URL=http://127.0.0.1:8080
cd .. && python3 main.py --from 5       # or `python3 main.py` for everything
```

`src/llm_client.py` picks this up automatically and puts it **first**
in the provider order (see `discover_providers()`) — no other code
changes needed. Hosted providers only get used as fallback if
`LOCAL_LLM_URL` is unset or the local server errors.

## Files

| File | Purpose |
|---|---|
| `models.py` | Model tier definitions — repo IDs, filenames, suggested role (slm/goa/both), chat_format, VRAM/RAM notes |
| `download_model.py` | Fetches a GGUF file for a given tier; skips if already present |
| `start_server.py` | Auto-detects GPU vs CPU-only and launches `llama_cpp.server` with sane defaults for either; `--also <tier>` runs a second server concurrently |
| `model_compare.py` | Standalone harness comparing model tiers on translation accuracy, reasoning quality, latency, and tokens (see "Comparing model tiers" below) |
| `client.py` | Minimal Python client, if you want to call the server directly outside the pipeline |
| `test_server.py` | Smoke test — verifies the server responds, parses as JSON, and reports tokens/sec |
| `requirements.txt` | Python deps (CPU build by default — see below for GPU) |

## Choosing a tier

Run `python3 start_server.py <tier>` with one of:

- `small` (Qwen2.5-1.5B, ~1GB) — runs almost anywhere, weak quality, last resort
- `medium` (Qwen2.5-7B, ~4.7GB) — **default**, matches this project's actual
  decision-making prompts well
- `large` (Qwen2.5-14B, ~9GB) — better quality if you have the VRAM to spare
- `llama32_1b` / `llama32_3b` (Llama-3.2, ~0.8-2GB) — smaller/faster SLM
  candidates from a different model family than Qwen2.5
- `phi35_mini` (Phi-3.5-mini, ~2.4GB) — strong reasoning-per-parameter,
  usable as either role
- `mistral7b` (Mistral-7B-Instruct-v0.3, ~4.4GB) / `gemma2_9b`
  (Gemma-2-9B-it, ~5.8GB) — non-Qwen GOA reasoning candidates

Each tier's `role` field in `models.py` (`"slm"`, `"goa"`, or `"both"`)
is only a suggestion for which MultiLLM role it suits — see
"Running separate SLM/GOA models" below for actually wiring that up,
and `model_compare.py` for measuring which tier is actually best for
which role on your hardware.

## Running separate SLM/GOA models

By default the whole pipeline (`MultiLLM(role="slm")` for ZOA intent
compilation, `MultiLLM(role="goa")` for zone selection/preemption
reasoning — see `../src/evaluator.py`) shares one `LOCAL_LLM_URL`
server. To actually give the SLM a small/fast model and the GOA a
larger/reasoning model, run two servers and point each role at its own:

```bash
python3 start_server.py llama32_3b --port 8080 \
    --also mistral7b --also-port 8081
```

```bash
export LOCAL_LLM_URL_SLM=http://127.0.0.1:8080   # small, for role="slm"
export LOCAL_LLM_URL_GOA=http://127.0.0.1:8081   # large, for role="goa"
cd .. && python3 main.py --from 5
```

`discover_providers()` (`src/llm_client.py`) registers each as
available only to its own role; `MultiLLM.ask()` prefers a role match
but still falls back to a generic `LOCAL_LLM_URL` (if also set) or a
hosted provider. Setting only `LOCAL_LLM_URL` (the original single-
server setup) keeps working unchanged.

## Comparing model tiers

`model_compare.py` scores any set of already-running local servers on:
translation accuracy + full-schema exact-match rate (replaying
`intent_data/intent_ood_probe.json` through the same SLM prompt the
pipeline uses), a small fixed set of representative GOA zone-selection
and preemption reasoning prompts checked against documented heuristics,
and latency/token usage (reusing MultiLLM's own tracking). It never
launches servers itself — start whichever tiers you want to compare
first:

```bash
python3 start_server.py llama32_1b --port 8080 --also mistral7b --also-port 8081
# in a third shell, once both are up:
python3 model_compare.py \
    --target llama32_1b@http://127.0.0.1:8080 \
    --target mistral7b@http://127.0.0.1:8081
```

Add as many `--target label@url` pairs as you have servers for —
nothing about the model list is hardcoded. Outputs
`../harness_final/model_comparison.csv` and
`../harness_final/model_comparison_figure.png`.

## Hardware guidance

RAM is essentially never the bottleneck (even the `large` tier needs
under 10GB of RAM/VRAM to load). The bottleneck is **compute
throughput**:

| Setup | Expected speed | Verdict |
|---|---|---|
| Any GPU with 8GB+ VRAM | Fast (real-time enough for a full evaluation run) | **Recommended** |
| Dedicated, uncontended 16+ CPU cores | Usable but slow | OK for small/overnight runs |
| Shared/throttled container (e.g. cloud sandbox with a cgroup CPU quota, contended by other tenants) | Impractically slow, even for the 1.5B tier | Not recommended — fall back to a hosted provider instead |

`start_server.py` reads the actual cgroup CPU quota (not just
`os.cpu_count()`, which can be misleading in containers) to size
`--n_threads` sensibly on CPU-only hosts, but no thread tuning fixes a
host that's fundamentally out of spare compute.

### GPU setup

The default `pip install -r requirements.txt` builds `llama-cpp-python`
CPU-only. For GPU offload (CUDA), install it with the CUDA build flag
instead:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir "llama-cpp-python[server]"
```

`start_server.py` will then detect `nvidia-smi` and pass
`--n_gpu_layers -1` (offload everything) automatically.

## Known issue: `response_format: json_object`

`llama_cpp.server`'s grammar-constrained JSON mode
(`response_format={"type": "json_object"}`) hangs rather than failing
cleanly on at least the version pinned here. `src/llm_client.py`
already special-cases this — it omits that parameter for the
`LOCAL_LLM_URL` provider and instead relies on prompting + regex
extraction (`re.search(r"\{.*\}", ...)`), same as `test_server.py` and
`client.py` do. If you call the server directly, do the same.

## Deploying to another machine

This folder is self-contained and git-friendly (it travels with the
rest of the project — no separate repo needed, just push the whole
`agentic_edge_orchestration/` folder):

```bash
git init && git add . && git commit -m "initial import"
git push ...

# on the target machine (ideally with a GPU):
git clone ...
cd agentic_edge_orchestration/local_llm
pip install -r requirements.txt   # or the CUDA variant above, if it has a GPU
python3 start_server.py medium
export LOCAL_LLM_URL=http://127.0.0.1:8080
cd .. && python3 main.py
```
