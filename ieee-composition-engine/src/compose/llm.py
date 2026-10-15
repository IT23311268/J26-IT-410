"""
llm.py - the ONLY file in the project that talks to Gemini.

Everything else asks this file for help through one method:

    llm.ask_json(task, prompt)  ->  Python data (a list or a dict)

Keeping Gemini behind this one door gives us three things:

  1. TESTING WITHOUT THE INTERNET. The tests hand the engine a FakeLLM with
     ready-made answers, so all 80+ tests run offline, instantly, for free.
  2. A SAFE FALLBACK. If there is no API key, no internet, or Gemini says
     "too many requests", ask_json raises LLMUnavailable and the caller
     falls back to its rule-based method. The engine never just crashes.
  3. A CACHE. Every answer is saved in data/cache/llm/. Running the same
     paper twice costs zero API calls the second time - important on the
     free tier, which allows only a limited number of requests per minute.

Your API key is read from a file called .env in the project folder:

    GEMINI_API_KEY=your-key-here
    GEMINI_MODEL=gemini-3.5-flash-lite      (optional)

.env is listed in .gitignore, so the key never goes to GitHub.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

# Models to try, in order. Google recommends the 3.x models for new projects;
# 2.5 Flash still works for accounts that used it before. If a model name is
# rejected (retired or not available to this key), the next one is tried.
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache", "llm")


class LLMUnavailable(Exception):
    """Gemini could not be used right now. Callers fall back to rules."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_json_reply(text: str) -> Any:
    """Turn a model's reply into Python data.

    Even when asked for JSON only, models sometimes wrap it in ```json fences
    or add a sentence before it. We strip that and parse what is left.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=0)
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError as error:
        raise LLMUnavailable(f"Gemini's reply was not valid JSON: {error}") from error


class BaseLLM:
    """What every LLM client looks like to the rest of the engine."""

    name = "none"
    calls_made = 0
    cache_hits = 0

    def ask_json(self, task: str, prompt: str) -> Any:   # pragma: no cover
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class NoLLM(BaseLLM):
    """Used with --offline, or when no API key is set. Always says no."""

    name = "offline (rules only)"

    def __init__(self, reason: str = "offline mode"):
        self.reason = reason

    def ask_json(self, task: str, prompt: str) -> Any:
        raise LLMUnavailable(self.reason)

    @property
    def available(self) -> bool:
        return False


class FakeLLM(BaseLLM):
    """For tests: returns prepared answers instead of calling Gemini.

    answers maps a task name to either a fixed answer or a function that
    receives the prompt and returns an answer.
    """

    name = "fake (tests)"

    def __init__(self, answers: Dict[str, Any]):
        self.answers = answers
        self.prompts: List[tuple] = []

    def ask_json(self, task: str, prompt: str) -> Any:
        self.prompts.append((task, prompt))
        self.calls_made += 1
        if task not in self.answers:
            raise LLMUnavailable(f"FakeLLM has no answer for task '{task}'")
        answer = self.answers[task]
        return answer(prompt) if callable(answer) else answer


# ---------------------------------------------------------------------------
# The real Gemini client
# ---------------------------------------------------------------------------

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, models: Optional[List[str]] = None,
                 use_cache: bool = True, max_retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep):
        from google import genai            # imported here so tests never need it

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.models = models or DEFAULT_MODELS
        self.model = self.models[0]
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.sleep = sleep
        self.name = f"Gemini ({self.model})"
        self.calls_made = 0
        self.cache_hits = 0

    # -- cache --------------------------------------------------------------

    def _cache_path(self, prompt: str) -> str:
        digest = hashlib.sha256(f"{self.model}\n{prompt}".encode("utf-8")).hexdigest()[:32]
        return os.path.join(CACHE_DIR, f"{digest}.json")

    def _read_cache(self, prompt: str) -> Optional[str]:
        if not self.use_cache:
            return None
        path = self._cache_path(prompt)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)["reply"]
        return None

    def _write_cache(self, prompt: str, reply: str) -> None:
        if not self.use_cache:
            return
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self._cache_path(prompt), "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "reply": reply}, f, ensure_ascii=False)

    # -- the call -----------------------------------------------------------

    def ask_json(self, task: str, prompt: str) -> Any:
        cached = self._read_cache(prompt)
        if cached is not None:
            self.cache_hits += 1
            return parse_json_reply(cached)

        from google.genai import errors, types

        config = types.GenerateContentConfig(
            temperature=0.1,                      # low = consistent, factual answers
            response_mime_type="application/json",
        )
        attempt = 0
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config)
                self.calls_made += 1
                reply = response.text or ""
                data = parse_json_reply(reply)
                self._write_cache(prompt, reply)
                return data

            except errors.ClientError as error:
                code = getattr(error, "code", 0)
                if code == 404 and self._next_model():
                    continue                       # model retired: try the next one
                if code == 429 and attempt < self.max_retries:
                    attempt += 1
                    self.sleep(15 * attempt)       # free-tier limit: wait, then retry
                    continue
                if code in (400, 401, 403):
                    raise LLMUnavailable(
                        "Gemini rejected the request - check GEMINI_API_KEY in your .env file "
                        f"({code}: {str(error)[:120]})") from error
                raise LLMUnavailable(f"Gemini error {code}: {str(error)[:160]}") from error

            except errors.ServerError as error:
                if attempt < self.max_retries:
                    attempt += 1
                    self.sleep(5 * attempt)
                    continue
                raise LLMUnavailable(f"Gemini server error: {str(error)[:160]}") from error

            except LLMUnavailable:
                raise
            except Exception as error:             # no internet, DNS failure, proxy...
                raise LLMUnavailable(f"Could not reach Gemini: {str(error)[:160]}") from error

    def _next_model(self) -> bool:
        index = self.models.index(self.model)
        if index + 1 < len(self.models):
            self.model = self.models[index + 1]
            self.name = f"Gemini ({self.model})"
            return True
        return False


# ---------------------------------------------------------------------------
# Choosing a client
# ---------------------------------------------------------------------------

def make_llm(offline: bool = False) -> BaseLLM:
    """Return a Gemini client if a key is configured, otherwise NoLLM."""
    if offline:
        return NoLLM("offline mode (--offline)")

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    except ImportError:
        pass

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key == "your-key-here":
        return NoLLM("no GEMINI_API_KEY in .env - using rules only")

    models = list(DEFAULT_MODELS)
    chosen = os.environ.get("GEMINI_MODEL", "").strip()
    if chosen:
        models = [chosen] + [m for m in models if m != chosen]

    try:
        return GeminiLLM(key, models=models)
    except ImportError:
        return NoLLM("google-genai is not installed - run: pip install -r requirements.txt")
