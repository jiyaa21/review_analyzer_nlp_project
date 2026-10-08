"""Core pipeline: preprocessing -> prompt -> LLM -> validation -> aggregation."""
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import yaml

from src.baseline import predict as baseline_predict
from src.llm_client import LLMError
from src.preprocess import clean_text, corpus_keywords, extract_keywords

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


# ----------------------------------------------------------------- helpers
def load_prompts(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    for key in ("system_prompt", "analysis_prompt", "report_prompt"):
        if key not in prompts:
            raise KeyError(f"Prompt file is missing the '{key}' entry")
    return prompts


def parse_llm_json(text: str) -> dict:
    """Extract the first JSON object from the model output (handles code fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start:end + 1])


def validate_analysis(data: dict) -> dict:
    """Check the schema and normalise values. Raises ValueError on bad output."""
    sentiment = str(data.get("sentiment", "")).strip().lower()
    if sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"invalid sentiment: {sentiment!r}")
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    aspects = []
    for item in data.get("aspects") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("aspect", "")).strip().lower()
        sent = str(item.get("sentiment", "")).strip().lower()
        if name and sent in VALID_SENTIMENTS:
            aspects.append({"aspect": name, "sentiment": sent})
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "aspects": aspects,
        "summary": str(data.get("summary", "")).strip(),
    }


class Cache:
    """Tiny JSON file cache so the same review never costs two API calls."""

    def __init__(self, path: str, enabled: bool = True):
        self.path, self.enabled, self.data = path, enabled, {}
        if enabled and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except (OSError, ValueError):
                self.data = {}

    @staticmethod
    def key(model: str, review: str) -> str:
        return hashlib.sha256(f"{model}||{review}".encode("utf-8")).hexdigest()

    def get(self, key):
        return self.data.get(key) if self.enabled else None

    def set(self, key, value):
        if self.enabled:
            self.data[key] = value

    def save(self):
        if self.enabled:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)


# ---------------------------------------------------------------- analyzer
class ReviewAnalyzer:
    def __init__(self, client, prompts: dict, cfg: dict, use_cache: bool = True):
        self.client = client
        self.prompts = prompts
        self.cfg = cfg
        self.delay = cfg["llm"].get("request_delay", 0.0)
        self.top_n = cfg["preprocessing"]["top_keywords"]
        self.min_len = cfg["preprocessing"]["min_token_length"]
        self.cache = Cache(cfg["paths"]["cache_file"], enabled=use_cache)
        self.api_calls = 0

    def analyze_one(self, review: str) -> dict:
        """Analyse a single review. Never raises: falls back to the baseline."""
        cleaned = clean_text(review)
        result = {"keywords": extract_keywords(cleaned, self.top_n, self.min_len)}
        if not cleaned:
            return {**result, "sentiment": "neutral", "confidence": 0.0, "aspects": [],
                    "summary": "", "source": "empty"}

        # Light normalisation only: casing/punctuation help the LLM detect sarcasm & emphasis.
        text = " ".join(str(review).split())
        key = Cache.key(self.client.model, text)
        cached = self.cache.get(key)
        if cached:
            return {**result, **cached, "source": "cache"}

        # The prompt is built from the file; the cleaned review is injected into it.
        prompt = self.prompts["analysis_prompt"].replace("{review}", text.replace('"""', "'''"))
        error = ""
        for attempt in range(2):  # one extra try if the JSON is malformed
            try:
                raw = self.client.chat(self.prompts["system_prompt"], prompt)
                self.api_calls += 1
                analysis = validate_analysis(parse_llm_json(raw))
                self.cache.set(key, analysis)
                time.sleep(self.delay)
                return {**result, **analysis, "source": "llm"}
            except (ValueError, json.JSONDecodeError) as exc:
                error = f"bad model output: {exc}"
            except LLMError as exc:
                error = str(exc)
                break  # API-level failure: retries already happened inside the client

        # Graceful degradation so one failure never kills the whole batch.
        return {**result, "sentiment": baseline_predict(review), "confidence": 0.0,
                "aspects": [], "summary": "", "source": "fallback", "error": error}

    def analyze_batch(self, reviews: list, progress=None) -> list:
        results = []
        for i, item in enumerate(reviews, 1):
            res = {"id": item["id"], "review": item["review"], **self.analyze_one(item["review"])}
            results.append(res)
            if progress:
                progress(i, len(reviews), res)
        self.cache.save()
        return results

    # ------------------------------------------------------------ report
    def build_statistics(self, results: list) -> dict:
        sentiments = Counter(r["sentiment"] for r in results)
        aspect_stats = defaultdict(Counter)
        for r in results:
            for a in r["aspects"]:
                aspect_stats[a["aspect"]][a["sentiment"]] += 1
        liked = sorted(((n, c["positive"]) for n, c in aspect_stats.items() if c["positive"]),
                       key=lambda x: -x[1])[:5]
        disliked = sorted(((n, c["negative"]) for n, c in aspect_stats.items() if c["negative"]),
                          key=lambda x: -x[1])[:5]
        return {
            "total_reviews": len(results),
            "sentiment_counts": dict(sentiments),
            "top_liked_aspects": liked,
            "top_disliked_aspects": disliked,
            "top_keywords": corpus_keywords([r["review"] for r in results], 10, self.min_len),
        }

    def generate_report(self, stats: dict) -> str:
        """One extra LLM call turns the numbers into a readable business summary."""
        prompt = self.prompts["report_prompt"].replace("{stats}", json.dumps(stats, indent=2))
        try:
            return self.client.chat("You are a concise business analyst.", prompt, json_mode=False)
        except LLMError as exc:
            return f"(Report unavailable: {exc})"
