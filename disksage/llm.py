"""LLM providers — one code path, any OpenAI-compatible backend.

Groq, Gemini, and every common *local* runner (Ollama, LM Studio, llama.cpp,
Jan, LocalAI, text-generation-webui, GPT4All, …) all speak the OpenAI chat API,
so we use a single client and just swap the base URL + model. "Supporting a
local LLM" is therefore just pointing at the right localhost port + model.

Provider is chosen by DISKSAGE_PROVIDER, else auto-detected in this order:
    local (any detected server)  ->  groq  ->  gemini

The tool works with NO provider at all — classify.py falls back to the
knowledge base and tags anything unknown as "review". The LLM only ever
improves labelling; it never gains delete authority (see safety.py).
"""

from __future__ import annotations

import json
import os
import urllib.request

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": os.environ.get("DISKSAGE_GROQ_MODEL", "openai/gpt-oss-20b"),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": os.environ.get("DISKSAGE_GEMINI_MODEL", "gemini-flash-latest"),
    },
}

# Common local OpenAI-compatible servers, by default port.
LOCAL_ENDPOINTS = [
    ("Ollama", "http://localhost:11434/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
    ("llama.cpp / LocalAI", "http://localhost:8080/v1"),
    ("Jan", "http://localhost:1337/v1"),
    ("text-generation-webui", "http://localhost:5000/v1"),
    ("GPT4All", "http://localhost:4891/v1"),
]


def list_models(base_url: str, timeout: float = 1.5) -> list[str]:
    """Return the model ids a local/remote OpenAI server advertises (or [])."""
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Authorization": "Bearer local"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        return [m.get("id") for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


def detect_local_servers(timeout: float = 1.0) -> list[dict]:
    """Probe the common local ports in parallel; return the ones that answer,
    each as {name, base_url, models}. Used by `setup` and `doctor`."""
    from concurrent.futures import ThreadPoolExecutor

    def probe(item):
        name, base = item
        models = list_models(base, timeout=timeout)
        return {"name": name, "base_url": base, "models": models} if models else None

    with ThreadPoolExecutor(max_workers=len(LOCAL_ENDPOINTS)) as ex:
        results = list(ex.map(probe, LOCAL_ENDPOINTS))
    return [r for r in results if r]


_LOCAL_CACHE: object = "__unset__"


def _detect_local_cached() -> list[dict]:
    global _LOCAL_CACHE
    if _LOCAL_CACHE == "__unset__":
        _LOCAL_CACHE = detect_local_servers()
    return _LOCAL_CACHE  # type: ignore[return-value]


def _resolve_local() -> dict | None:
    """Configured local server (fast path), else a cached auto-detect."""
    base = os.environ.get("DISKSAGE_LOCAL_BASE_URL") or os.environ.get("LMSTUDIO_BASE_URL")
    model = os.environ.get("DISKSAGE_LOCAL_MODEL")
    if base:
        if not model:
            found = list_models(base)
            model = found[0] if found else None
        if model:
            return {"name": "local", "base_url": base, "api_key": "local", "model": model}
        return None
    for s in _detect_local_cached():
        if s["models"]:
            return {"name": "local", "base_url": s["base_url"],
                    "api_key": "local", "model": s["models"][0]}
    return None


def resolve() -> dict | None:
    """Pick a usable provider config, or None if nothing is available."""
    forced = os.environ.get("DISKSAGE_PROVIDER", "").strip().lower()
    order = [forced] if forced else ["local", "groq", "gemini"]

    for name in order:
        if name == "local":
            cfg = _resolve_local()
            if cfg:
                return cfg
        elif name in PROVIDERS:
            p = PROVIDERS[name]
            key = os.environ.get(p["api_key_env"], "")
            if key:
                return {"name": name, "base_url": p["base_url"],
                        "api_key": key, "model": p["model"]}
    return None


def describe() -> str:
    cfg = resolve()
    return f"{cfg['name']} ({cfg['model']})" if cfg else "none (knowledge base only)"


from dataclasses import dataclass


@dataclass
class LLMResult:
    content: str | None = None
    error: str | None = None      # human-friendly reason when the call failed


def _friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    msg = str(exc)
    if "Connection" in name or "Connection" in msg or "getaddrinfo" in msg:
        return "can't reach the AI service — check your internet connection"
    if "Timeout" in name or "timed out" in msg.lower():
        return "the AI request timed out"
    if "Authentication" in name or "401" in msg:
        return "the API key was rejected — check it in your .env"
    if "RateLimit" in name or "429" in msg:
        return "rate limited by the provider — try again in a moment"
    if "NotFound" in name or "404" in msg or "model" in msg.lower():
        return "the configured model wasn't found for this provider"
    return f"AI request failed ({name})"


def chat_json(system: str, user: str) -> LLMResult:
    """Send one chat request. Returns an LLMResult with either content or a
    human-friendly error — never raises."""
    cfg = resolve()
    if not cfg:
        return LLMResult(error="no AI provider configured")
    try:
        from openai import OpenAI
    except ImportError:
        return LLMResult(error="the `openai` package isn't installed")

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                    timeout=30.0, max_retries=1)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        resp = client.chat.completions.create(
            model=cfg["model"], messages=messages, temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as first:
        # Some endpoints reject response_format — retry plain before giving up.
        try:
            resp = client.chat.completions.create(
                model=cfg["model"], messages=messages, temperature=0,
            )
        except Exception:
            return LLMResult(error=_friendly_error(first))
    try:
        return LLMResult(content=resp.choices[0].message.content)
    except (AttributeError, IndexError):
        return LLMResult(error="the AI returned an empty response")
