"""Thin wrapper around the Groq chat-completions REST API (OpenAI compatible).

Also contains MockLLMClient, an offline stand-in used for testing the pipeline
without an API key or internet connection.
"""
import os
import re
import time
import json

import requests

from src.baseline import predict as baseline_predict
from src.preprocess import tokenize


class LLMError(Exception):
    """Raised when the LLM API cannot return a usable answer."""


class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg["model"]
        self.api_key = os.getenv(cfg["api_key_env"], "").strip()
        if not self.api_key:
            raise LLMError(
                f"Environment variable {cfg['api_key_env']} is not set. "
                "Create a .env file (see .env.example) or run with --mock."
            )
        self.session = requests.Session()  # re-uses the TCP connection -> faster
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = None) -> str:
        """Send one chat request and return the assistant text. Retries on 429/5xx."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 400),
        }
        use_json = self.cfg.get("json_mode", False) if json_mode is None else json_mode
        if use_json:
            payload["response_format"] = {"type": "json_object"}

        retries = self.cfg.get("max_retries", 3)
        delay = self.cfg.get("retry_delay", 2)
        last_error = "unknown error"

        for attempt in range(1, retries + 1):
            try:
                resp = self.session.post(self.cfg["api_url"], json=payload,
                                         timeout=self.cfg.get("timeout", 30))
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()["choices"][0]["message"]["content"].strip()
                    except (KeyError, IndexError, ValueError) as exc:
                        raise LLMError(f"Unexpected API response format: {exc}")
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise LLMError(last_error)  # 400/401/404 ... retrying will not help
            if attempt < retries:
                time.sleep(delay * (2 ** (attempt - 1)))  # exponential back-off
        raise LLMError(f"LLM request failed after {retries} attempts ({last_error})")


class MockLLMClient:
    """Offline fake LLM: returns JSON in the same schema as the real model."""
    model = "mock-offline"

    ASPECTS = ["battery", "camera", "price", "service", "food", "delivery", "quality",
               "screen", "performance", "design", "sound", "support", "staff", "app"]

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = None) -> str:
        if "Statistics:" in user_prompt:  # report prompt
            return ("[MOCK REPORT] Customers are mixed overall. Run without --mock "
                    "to get a real LLM-written report.")
        matches = re.findall(r'Review: """(.*?)"""\s*Output:', user_prompt, re.S)
        review = matches[-1] if matches else user_prompt  # last one = the real review
        sentiment = baseline_predict(review)
        tokens = set(tokenize(review))
        aspects = []
        for asp in self.ASPECTS:
            if asp in tokens:
                aspects.append({"aspect": asp, "sentiment": sentiment})
        return json.dumps({
            "sentiment": sentiment, "confidence": 0.5, "aspects": aspects,
            "summary": "Mock summary: " + " ".join(review.split()[:8]),
        })
