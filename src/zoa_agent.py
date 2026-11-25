"""Phase 3R — Zonal Orchestration Agent building blocks.

Semantic intent cache (MiniLM embeddings + auto-tuned cosine threshold)
with fallback to SLM compilation on a cache miss. Placement itself is
sim_engine.solve_placement — this module is about turning free-text
intent into the structured profile the solver needs.
"""
import numpy as np


def latency_class_of(ms):
    return "realtime" if ms < 100 else ("interactive" if ms <= 500
                                        else "batch")


def ground_truth(services, stype):
    c = services[stype]
    return {"service_type": stype,
            "latency_class": latency_class_of(c["latency_ms"]),
            "data_locality": c["data_locality"],
            "priority": c["priority"]}


def build_schema(services):
    """INTENT_SCHEMA / REQUIRED_FIELDS / SEMANTIC depend on the loaded
    service catalog, so these are built at runtime rather than being
    module-level constants."""
    intent_schema = {"service_type": sorted(services.keys()),
                     "latency_class": ["realtime", "interactive", "batch"],
                     "data_locality": ["zone_local", "any"],
                     "priority": ["low", "normal", "high", "critical"]}
    required_fields = list(intent_schema.keys())
    semantic = {st: {**ground_truth(services, st),
                     "cpu_scale": services[st]["cpu_scale"],
                     "latency_ms": services[st]["latency_ms"]}
                for st in services}
    return intent_schema, required_fields, semantic


def slm_system_prompt(intent_schema):
    return f"""You translate a service request into a strict JSON object.
Respond with ONLY a JSON object. Fields:
  "service_type": EXACTLY one of {intent_schema['service_type']}
  "latency_class": "realtime"(<100ms)/"interactive"(100-500ms)/"batch"(>500ms)
  "data_locality": "zone_local" if data must stay local, else "any"
  "priority": "low"/"normal"/"high"/"critical"
Never invent new service_type names."""


class Embedder:
    def __init__(self, fit_texts):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend = "minilm"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(analyzer="char_wb",
                                       ngram_range=(3, 5)).fit(fit_texts)
            self.backend = "tfidf_char"

    def encode(self, texts):
        if self.backend == "minilm":
            return np.asarray(self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False))
        m = self.vec.transform(texts)
        return np.asarray((m / (np.sqrt(
            m.multiply(m).sum(1)) + 1e-12)).todense())


class IntentLibrary:
    """Semantic cache: cosine similarity against known (text, profile)
    pairs, above `threshold` = reuse; below = cache miss -> caller
    should invoke the SLM compiler and .add() the result."""

    def __init__(self, seed_entries, threshold, embedder, vec_cache=None):
        self.entries = list(seed_entries)
        self.emb = embedder
        self._vec_cache = vec_cache if vec_cache is not None else {}
        self.M = self.emb.encode([t for t, _ in self.entries])
        self.threshold = threshold

    def lookup(self, text):
        q = self._get_vec(text)
        sims = self.M @ q
        i = int(np.argmax(sims))
        if sims[i] >= self.threshold:
            return self.entries[i][1], float(sims[i])
        return None, float(sims[i])

    def _get_vec(self, text):
        if text in self._vec_cache:
            return self._vec_cache[text]
        v = self.emb.encode([text])[0]
        self._vec_cache[text] = v
        return v

    def add(self, text, profile):
        self.entries.append((text, profile))
        v = self._get_vec(text)
        self.M = np.vstack([self.M, v.reshape(1, -1)])
