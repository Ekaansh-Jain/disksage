"""Turn raw scan items into tagged, explained items.

Strategy:
  1. Ask the knowledge base first (fast, offline, deterministic).
  2. For anything unknown, ask the LLM once, in a single batched call, sending
     ONLY paths + human-readable sizes (never file contents). Privacy by design.
  3. The LLM's answer is advisory. Its tier is clamped so it can never *upgrade*
     something into "safe" beyond what we trust; unknowns default to "review".
"""

from __future__ import annotations

import json
import os
import re

from . import dedup, knowledge, llm
from .scan import Item

VALID_TIERS = {"safe", "review", "keep"}

SYSTEM = (
    "You identify macOS disk items so a user can disksage space. "
    "You are given only folder paths and sizes — never file contents. "
    "For each path, return what it is and how safe it is to delete. "
    "Tier rules: 'safe' ONLY for caches/build artifacts that regenerate "
    "automatically; 'review' for things reclaimable but needing a "
    "redownload/reinstall, or anything you are unsure about; 'keep' for "
    "documents, source code, media, or app data that is not a cache. "
    "When unsure, use 'review'. Never guess 'safe'. "
    'Reply as JSON: {"items":[{"path":"...","label":"short name",'
    '"tier":"safe|review|keep","why":"under 12 words"}]}'
)


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f} {unit}" if unit != "B" else f"{int(x)} B"
        x /= 1024
    return f"{x:.1f} TB"


def _extract_items(raw: str) -> list[dict]:
    if not raw:
        return []
    try:
        return json.loads(raw).get("items", [])
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)  # find the JSON object
    if m:
        try:
            return json.loads(m.group(0)).get("items", [])
        except Exception:
            return []
    return []


from dataclasses import dataclass


@dataclass
class AiOutcome:
    ok: bool
    labeled: int = 0
    total: int = 0
    message: str = ""


def annotate_known(items: list[Item]) -> list[Item]:
    """Apply the knowledge base. Returns the items still unrecognized."""
    unknown: list[Item] = []
    for it in items:
        hit = knowledge.look_up(it.path)
        if hit:
            it.tier, it.label, it.why, it.source = hit.tier, hit.label, hit.why, "knowledge"
        else:
            it.tier, it.label, it.why, it.source = (
                "review", "unrecognized", "not in knowledge base", "unknown")
            unknown.append(it)
    return unknown


def annotate_with_ai(unknown: list[Item]) -> AiOutcome:
    """Ask the LLM to label the unknown items. Never raises; reports outcome."""
    total = len(unknown)
    payload = [{"path": u.path, "size": human(u.size)} for u in unknown]
    res = llm.chat_json(SYSTEM, json.dumps(payload))
    if res.error:
        return AiOutcome(ok=False, total=total, message=res.error)

    by_path = {u.path: u for u in unknown}
    labeled = 0
    for entry in _extract_items(res.content or ""):
        u = by_path.get(entry.get("path", ""))
        if not u:
            continue
        tier = str(entry.get("tier", "review")).lower()
        u.tier = tier if tier in VALID_TIERS else "review"
        u.label = str(entry.get("label", u.label))[:60] or u.label
        u.why = str(entry.get("why", u.why))[:120] or u.why
        u.source = "ai"
        labeled += 1
    return AiOutcome(ok=True, labeled=labeled, total=total)


def classify(items: list[Item], use_ai: bool = True) -> list[Item]:
    """Convenience: knowledge pass, then AI on the remainder. Used by tests and
    non-interactive callers that don't need progress/outcome reporting."""
    unknown = annotate_known(items)
    if use_ai and unknown and llm.resolve():
        annotate_with_ai(unknown)
    return items


def build_file_items(files: list[tuple[str, int]], big_min: int) -> list[Item]:
    """Turn the raw file list into duplicate items + big loose-file items.

    Duplicates take precedence: an original copy is never listed (so it can't be
    deleted through the tool), only the extra copies, tagged 'duplicate'."""
    items: list[Item] = []

    groups = dedup.find_duplicates(files)
    dup_paths: set[str] = set()
    for group in groups:
        keep = group[0]
        keep_dir = os.path.dirname(keep)
        try:
            gsize = os.path.getsize(keep)
        except OSError:
            gsize = 0
        for extra in group[1:]:
            dup_paths.add(extra)
            items.append(Item(
                path=extra, size=gsize, kind="duplicate", tier="duplicate",
                label="Duplicate copy",
                why=f"identical to {os.path.basename(keep)}; original kept in {keep_dir}",
                source="hash"))
        dup_paths.add(keep)  # never surface the kept original as a big file

    for path, size in files:
        if size >= big_min and path not in dup_paths:
            hit = knowledge.file_kind(path)
            items.append(Item(path=path, size=size, kind="file",
                              tier=hit.tier, label=hit.label, why=hit.why,
                              source=hit.source))
    return items
