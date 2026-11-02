"""Subreddit wikis via Arctic Shift (moderator-curated video lists).

``/api/subreddits/wikis?subreddit=X`` returns at most 100 pages, so the full
list of paths comes from ``/api/subreddits/wikis/list`` and the remaining
pages are fetched in batches with ``paths=a,b,c``.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterable

from .. import config
from ..extract import extract_videos
from ..fetch import ArcticShiftClient, FetchError

log = logging.getLogger(__name__)

_SKIP_RE = re.compile(config.WIKI_SKIP_PATH_RE, re.IGNORECASE)
_PATH_RE = re.compile(r"^/?r/([^/]+)(/wiki/.*)$", re.IGNORECASE)


def wiki_path(full_path: str) -> tuple[str, str] | None:
    """``/r/bwf/wiki/exercises/handstand`` -> ``("bwf", "/wiki/exercises/handstand")``."""
    m = _PATH_RE.match((full_path or "").strip())
    if not m:
        return None
    return m.group(1), m.group(2).rstrip("/")


def is_content_path(full_path: str) -> bool:
    return wiki_path(full_path) is not None and not _SKIP_RE.search(full_path)


def fetch_pages(client: ArcticShiftClient, subreddit: str) -> list[dict]:
    """All archived wiki pages of ``subreddit`` (``[{path, content, revision_date, ...}]``)."""
    # Wikis change rarely; cache them like an open window (refreshed after CACHE_TTL_DAYS).
    now = int(time.time())
    pages = client.get_json(config.ARCTIC_SHIFT_WIKIS_URL, {"subreddit": subreddit, "limit": 100}, window_end=now)
    by_path = {p["path"]: p for p in pages if p.get("path")}
    try:
        listed = client.get_json(config.ARCTIC_SHIFT_WIKIS_URL + "/list", {"subreddit": subreddit}, window_end=now)
    except FetchError as exc:
        log.warning("r/%s wiki: listing failed (%s); using the first %d pages", subreddit, exc, len(by_path))
        listed = []
    missing = sorted(p for p in listed if isinstance(p, str) and p not in by_path and is_content_path(p))
    step = config.WIKI_PATHS_PER_REQUEST
    for i in range(0, len(missing), step):
        batch = missing[i: i + step]
        try:
            for p in client.get_json(config.ARCTIC_SHIFT_WIKIS_URL, {"paths": ",".join(batch)}, window_end=now):
                if p.get("path"):
                    by_path.setdefault(p["path"], p)
        except FetchError as exc:
            log.warning("r/%s wiki: batch %d failed: %s", subreddit, i // step + 1, exc)
    # Some listed paths are not retrievable (the API returns nothing for them).
    log.info("r/%s wiki: %d pages listed, %d retrieved", subreddit, len(listed), len(by_path))
    return [p for p in by_path.values() if is_content_path(p["path"])]


def page_evidence(page: dict, subreddit: str | None = None) -> list[dict]:
    """Evidence dicts, one per line of the page that links YouTube.

    Every line keeps its own text (plus the nearest heading) as context, since
    it usually names the exercise; the page path is carried along as a tag hint.
    ``subreddit`` overrides the spelling taken from the path.
    """
    parsed = wiki_path(page.get("path", ""))
    if not parsed:
        return []
    subreddit = subreddit or parsed[0]
    path = parsed[1]
    created = int(float(page.get("revision_date") or 0))
    out = []
    heading = ""
    for n, line in enumerate((page.get("content") or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
        if not extract_videos(stripped):
            continue
        out.append({
            "id": f"wiki:{subreddit.lower()}{path}#{n}",
            "kind": "wiki",
            "author": f"wiki:{subreddit.lower()}",
            "body": f"{heading}\n{stripped}" if heading else stripped,
            "score": 0,
            "created_utc": created,
            "subreddit": subreddit,
            "path": path,
            "url": f"https://www.reddit.com/r/{subreddit}{path}",
        })
    return out


def collect(client: ArcticShiftClient, subreddits: Iterable[str] = config.WIKI_SUBREDDITS) -> list[dict]:
    out: list[dict] = []
    for sub in subreddits:
        try:
            pages = fetch_pages(client, sub)
        except FetchError as exc:
            log.error("r/%s wiki: %s (skipped)", sub, exc)
            continue
        ev = [e for page in pages for e in page_evidence(page, sub)]
        log.info("r/%s wiki: %d pages, %d lines linking YouTube", sub, len(pages), len(ev))
        out.extend(ev)
    return out
