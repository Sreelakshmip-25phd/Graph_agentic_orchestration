"""Shared paths and constants.

BASE is always the project root (parent of src/), so scripts resolve the
same directories regardless of the current working directory.
"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOUNDATION = os.path.join(BASE, "data_foundation")
RAW = os.path.join(DATA_FOUNDATION, "raw")
VALIDATED = os.path.join(DATA_FOUNDATION, "validated")
REPORTS = os.path.join(DATA_FOUNDATION, "reports")

INTENT_DATA = os.path.join(BASE, "intent_data")
SCEN = os.path.join(BASE, "scenario")          # topology, catalog, intents
WORKLOADS = os.path.join(BASE, "workloads")    # wref/, dev/
ZOA_DIR = os.path.join(BASE, "zoa")            # intent-cache threshold
CACHE_DIR = os.path.join(BASE, "llm_cache")    # per-system LLM caches (not committed)
RESULTS = os.path.join(BASE, "results")
REFERENCE = os.path.join(BASE, "reference_results")

for _d in (SCEN, WORKLOADS, ZOA_DIR, CACHE_DIR, RESULTS):
    os.makedirs(_d, exist_ok=True)

# Evaluation seeds. 0-2 are the original flat-memory study's workloads,
# unchanged; 3-9 come from the same generator and calibration.
EVAL_SEEDS = list(range(10))
# Dev seeds: only for choosing graph-memory weights, never reported.
DEV_SEEDS = [100, 101, 102, 103, 104]

# The earlier pipeline's name for the evaluation seeds (kept so the
# copied phase scripts still import).
SEEDS = EVAL_SEEDS
HARN = RESULTS  # earlier name, used by local_llm/model_compare.py

PLACES = ["stadium", "central station", "old town", "market square",
          "university campus", "harbour district", "tech park",
          "exhibition centre", "north ring", "riverside"]
