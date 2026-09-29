"""LLM providers — one code path, three backends.

LM Studio, Groq, and Gemini are all reachable through the OpenAI-compatible
chat API, so we use a single client and just swap the base URL + model.

Provider is chosen by DISKSAGE_PROVIDER, else auto-detected in this order:
    local (LM Studio, if running)  ->  groq  ->  gemini

The tool works with NO provider at all — classify.py falls back to the
knowledge base and tags anything unknown as "review". The LLM only ever
improves labelling; it never gains delete authority (see safety.py).
"""

from __future__ import annotations

import json
import os
import urllib.request

PROVIDERS = {
    "local": {
        "base_url": os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "api_key": "lm-studio",
        "model": os.environ.get("DISKSAGE_LOCAL_MODEL", ""),  # resolved live
    },
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


def _local_up() -> str | None:
    """Return the first loaded local model id, or None if LM Studio is down."""
    try:
        url = PROVIDERS["local"]["base_url"].rstrip("/") + "/models"
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.load(r)
        models = [m["id"] for m in data.get("data", [])]
        return models[0] if models else None
    except Exception:
        return None


def resolve() -> dict | None:
    """Pick a usable provider config, or None if nothing is available."""
    forced = os.environ.get("DISKSAGE_PROVIDER", "").strip().lower()
    order = [forced] if forced else ["local", "groq", "gemini"]

    for name in order:
        cfg = PROVIDERS.get(name)
        if not cfg:
            continue
        if name == "local":
            model = cfg["model"] or _local_up()
            if model:
                return {"name": name, "base_url": cfg["base_url"],
                        "api_key": cfg["api_key"], "model": model}
        else:
            key = os.environ.get(cfg["api_key_env"], "")
            if key:
                return {"name": name, "base_url": cfg["base_url"],
                        "api_key": key, "model": cfg["model"]}
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
