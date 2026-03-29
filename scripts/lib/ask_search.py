"""ask-search wrapper for native web search."""

import json
import os
import shlex
import shutil
import signal
import subprocess
from typing import Any, Dict, List, Optional

from . import websearch
from .relevance import token_overlap_relevance as _compute_relevance

WEB_LIMITS = {"quick": 8, "default": 15, "deep": 25}


def _resolve_command_prefix(command: Optional[str] = None) -> Optional[List[str]]:
    raw = (
        command
        or os.environ.get("ASK_SEARCH_CMD")
        or os.environ.get("LAST30DAYS_ASK_SEARCH_CMD")
    )
    if raw:
        return shlex.split(raw)

    if shutil.which("ask-search"):
        return ["ask-search"]

    return None


def is_ask_search_available(command: Optional[str] = None) -> bool:
    return _resolve_command_prefix(command) is not None


def get_ask_search_status(command: Optional[str] = None) -> Dict[str, Any]:
    prefix = _resolve_command_prefix(command)
    return {
        "available": bool(prefix),
        "command": " ".join(prefix) if prefix else None,
    }


def _run_ask_search(args: List[str], timeout: int, command: Optional[str] = None) -> Dict[str, Any]:
    prefix = _resolve_command_prefix(command)
    if not prefix:
        raise RuntimeError("ask-search is not available")

    cmd = [*prefix, *args, "--json"]
    preexec = os.setsid if hasattr(os, "setsid") else None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=preexec,
        env=os.environ.copy(),
    )

    try:
        from last30days import register_child_pid, unregister_child_pid
        register_child_pid(proc.pid)
    except Exception:
        unregister_child_pid = None

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        proc.wait(timeout=5)
        raise RuntimeError(f"ask-search timed out after {timeout}s")
    finally:
        try:
            if unregister_child_pid:
                unregister_child_pid(proc.pid)
        except Exception:
            pass

    if proc.returncode != 0:
        detail = (stderr or stdout or "").strip()
        raise RuntimeError(detail or f"ask-search failed with exit code {proc.returncode}")

    data = json.loads((stdout or "").strip() or "{}")
    if not isinstance(data, dict):
        raise RuntimeError("ask-search returned invalid JSON")
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data


def search_web(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """Search the web via ask-search."""
    limit = WEB_LIMITS.get(depth, WEB_LIMITS["default"])
    query = (
        f"{topic} "
        f"-site:reddit.com -site:x.com -site:twitter.com "
        f"after:{from_date} before:{to_date}"
    )

    response = _run_ask_search(["--num", str(limit), query], timeout=30, command=command)
    raw_results = response.get("results", [])

    items = []
    if isinstance(raw_results, list):
        for i, raw in enumerate(raw_results):
            if not isinstance(raw, dict):
                continue

            url = str(raw.get("url", "")).strip()
            title = str(raw.get("title", "")).strip()
            snippet = str(raw.get("content", raw.get("snippet", ""))).strip()
            if not url or websearch.is_excluded_domain(url):
                continue
            if not title and not snippet:
                continue

            items.append({
                "id": f"W{i + 1}",
                "title": title[:200],
                "url": url,
                "source_domain": websearch.extract_domain(url),
                "snippet": snippet[:500],
                "date": None,
                "date_confidence": "low",
                "relevance": _compute_relevance(topic, f"{title} {snippet}"),
                "why_relevant": f"ask-search ({','.join(raw.get('engines', []))})" if raw.get("engines") else "ask-search",
            })

    return {"source": "ask-search", "items": items}
