"""Unified OpenCLI search wrapper for Reddit, X, and web search."""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import websearch
from .relevance import token_overlap_relevance as _compute_relevance

REDDIT_LIMITS = {"quick": 15, "default": 30, "deep": 50}
X_LIMITS = {"quick": 12, "default": 30, "deep": 60}
WEB_LIMITS = {"quick": 8, "default": 15, "deep": 25}


def _log(msg: str):
    sys.stderr.write(f"[opencli] {msg}\n")
    sys.stderr.flush()


def _resolve_command_prefix(command: Optional[str] = None) -> Optional[List[str]]:
    """Resolve the command used to launch opencli."""
    raw = (
        command
        or os.environ.get("OPENCLI_CMD")
        or os.environ.get("LAST30DAYS_OPENCLI_CMD")
    )
    if raw:
        return shlex.split(raw)

    if shutil.which("opencli"):
        return ["opencli"]

    if shutil.which("npx") and shutil.which("node"):
        return ["npx", "-y", "@jackwener/opencli"]

    return None


def is_opencli_available(command: Optional[str] = None) -> bool:
    """Return True when an opencli launcher is available."""
    return _resolve_command_prefix(command) is not None


def get_opencli_status(command: Optional[str] = None) -> Dict[str, Any]:
    """Return basic availability information for opencli."""
    prefix = _resolve_command_prefix(command)
    return {
        "available": bool(prefix),
        "command": " ".join(prefix) if prefix else None,
    }


def _extract_json(stdout: str) -> Any:
    """Extract a JSON object or array from stdout."""
    text = (stdout or "").strip()
    if not text:
        return []

    candidates = [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[-1])

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("opencli did not return valid JSON")


def _run_opencli(
    args: List[str],
    timeout: int,
    command: Optional[str] = None,
) -> Any:
    """Run opencli and parse JSON output."""
    prefix = _resolve_command_prefix(command)
    if not prefix:
        raise RuntimeError("opencli is not available")

    cmd = [*prefix, *args, "--format", "json"]
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")

    preexec = os.setsid if hasattr(os, "setsid") else None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=preexec,
        env=env,
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
        raise RuntimeError(f"opencli timed out after {timeout}s")
    finally:
        try:
            if unregister_child_pid:
                unregister_child_pid(proc.pid)
        except Exception:
            pass

    if proc.returncode != 0:
        detail = (stderr or stdout or "").strip()
        raise RuntimeError(detail or f"opencli failed with exit code {proc.returncode}")

    return _extract_json(stdout)


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> Optional[str]:
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text

    try:
        if len(text) > 10 and text[10] == "T":
            return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        pass

    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def _reddit_time_filter(from_date: str, to_date: str) -> str:
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return "month"

    days = max(1, (end - start).days)
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    if days <= 365:
        return "year"
    return "all"


def search_reddit(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """Search Reddit via opencli and normalize into internal dict shape."""
    from . import reddit as reddit_search

    limit = REDDIT_LIMITS.get(depth, REDDIT_LIMITS["default"])
    time_filter = _reddit_time_filter(from_date, to_date)
    queries = reddit_search.expand_reddit_queries(topic, depth)
    per_query = max(8, min(limit, (limit + len(queries) - 1) // len(queries) + 4))

    seen_urls = set()
    items: List[Dict[str, Any]] = []
    first_error = None

    for query in queries:
        _log(f"Reddit search: {query}")
        try:
            raw_items = _run_opencli(
                [
                    "reddit", "search", query,
                    "--sort", "relevance",
                    "--time", time_filter,
                    "--limit", str(per_query),
                ],
                timeout=45,
                command=command,
            )
        except Exception as e:
            if first_error is None:
                first_error = str(e)
            continue

        if not isinstance(raw_items, list):
            continue

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            url = str(raw.get("url", "")).strip()
            title = str(raw.get("title", "")).strip()
            if not url or not title or url in seen_urls:
                continue

            seen_urls.add(url)
            subreddit = str(raw.get("subreddit", "")).strip()
            if subreddit and not subreddit.lower().startswith("r/"):
                subreddit = f"r/{subreddit.lstrip('/')}"

            items.append({
                "id": f"R{len(items) + 1}",
                "title": title[:300],
                "url": url,
                "subreddit": subreddit,
                "date": None,
                "engagement": {
                    "score": _coerce_int(raw.get("score")),
                    "num_comments": _coerce_int(raw.get("comments")),
                },
                "relevance": reddit_search._compute_post_relevance(topic, title, ""),
                "why_relevant": "Reddit search via opencli",
                "selftext": "",
            })

            if len(items) >= limit:
                break

        if len(items) >= limit:
            break

    result: Dict[str, Any] = {"source": "opencli", "items": items[:limit]}
    if first_error and not items:
        result["error"] = first_error
    return result


def _x_query_variants(topic: str, from_date: str) -> List[str]:
    from .query import extract_core_subject, extract_compound_terms

    core = extract_core_subject(topic, max_words=5, strip_suffixes=True)
    queries = []
    if core:
        queries.append(f"{core} since:{from_date}")

    compounds = extract_compound_terms(topic)
    if compounds:
        quoted = " OR ".join(f'"{term}"' for term in compounds[:3])
        queries.append(f"({quoted}) since:{from_date}")

    core_words = core.split()
    if len(core_words) > 2:
        queries.append(f"{' '.join(core_words[:2])} since:{from_date}")

    strongest = ""
    if core_words:
        strongest = max(core_words, key=len)
    if strongest:
        queries.append(f"{strongest} since:{from_date}")

    deduped = []
    seen = set()
    for query in queries:
        if query not in seen:
            seen.add(query)
            deduped.append(query)
    return deduped or [f"{topic} since:{from_date}"]


def _normalize_x_items(
    raw_items: Any,
    query: str,
    from_date: str,
    to_date: Optional[str],
) -> List[Dict[str, Any]]:
    items = []
    if not isinstance(raw_items, list):
        return items

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue

        url = str(raw.get("url", "")).strip()
        text = str(raw.get("text", "")).strip()
        if not url or not text:
            continue

        date = _parse_date(raw.get("created_at") or raw.get("date"))
        if date and date < from_date:
            continue
        if date and to_date and date > to_date:
            continue

        likes = _coerce_int(raw.get("likes"))
        views = _coerce_int(raw.get("views"))

        engagement = {
            "likes": likes,
            "reposts": _coerce_int(raw.get("reposts") or raw.get("retweets")),
            "replies": _coerce_int(raw.get("replies")),
            "quotes": _coerce_int(raw.get("quotes")),
        }

        if views is not None and engagement["likes"] is None:
            engagement["likes"] = 0

        items.append({
            "id": f"X{len(items) + 1}",
            "text": text[:500],
            "url": url,
            "author_handle": str(raw.get("author_handle") or raw.get("author") or "").strip().lstrip("@"),
            "date": date,
            "engagement": engagement if any(v is not None for v in engagement.values()) else None,
            "why_relevant": "",
            "relevance": _compute_relevance(query, text),
        })

    return items


def search_x(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """Search X via opencli and normalize into internal dict shape."""
    limit = X_LIMITS.get(depth, X_LIMITS["default"])
    first_error = None

    for query in _x_query_variants(topic, from_date):
        _log(f"X search: {query}")
        try:
            raw_items = _run_opencli(
                ["twitter", "search", query, "--filter", "top", "--limit", str(limit)],
                timeout=60,
                command=command,
            )
        except Exception as e:
            if first_error is None:
                first_error = str(e)
            continue

        items = _normalize_x_items(raw_items, topic, from_date, to_date)
        if items:
            return {"source": "opencli", "items": items}

    result: Dict[str, Any] = {"source": "opencli", "items": []}
    if first_error:
        result["error"] = first_error
    return result


def search_x_handles(
    handles: List[str],
    topic: Optional[str],
    from_date: str,
    count_per: int = 3,
    command: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search recent posts from specific handles via opencli."""
    from .query import extract_core_subject

    core = extract_core_subject(topic, max_words=5, strip_suffixes=True) if topic else None
    all_items: List[Dict[str, Any]] = []
    seen_urls = set()

    for handle in handles:
        query = f"from:{handle}"
        if core:
            query = f"{query} {core}"
        query = f"{query} since:{from_date}"

        try:
            raw_items = _run_opencli(
                ["twitter", "search", query, "--filter", "top", "--limit", str(count_per)],
                timeout=45,
                command=command,
            )
        except Exception as e:
            _log(f"Handle search failed for @{handle}: {e}")
            continue

        for item in _normalize_x_items(raw_items, core or handle, from_date, None):
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_items.append(item)

    return all_items


def search_web(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """Search the web via opencli's Google adapter."""
    limit = WEB_LIMITS.get(depth, WEB_LIMITS["default"])
    query = (
        f"{topic} "
        f"-site:reddit.com -site:x.com -site:twitter.com "
        f"after:{from_date} before:{to_date}"
    )
    _log(f"Web search: {query}")

    raw_items = _run_opencli(
        ["google", "search", query, "--limit", str(limit), "--lang", "en"],
        timeout=45,
        command=command,
    )

    items = []
    if isinstance(raw_items, list):
        for i, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                continue

            url = str(raw.get("url", "")).strip()
            title = str(raw.get("title", "")).strip()
            snippet = str(raw.get("snippet", "")).strip()
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
                "why_relevant": f"Google search via opencli ({raw.get('type', 'result')})",
            })

    return {"source": "opencli", "items": items}
