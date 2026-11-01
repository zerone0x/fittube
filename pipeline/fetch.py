"""Fetch Reddit comments containing YouTube links from the Arctic Shift API.

Arctic Shift's full-text search is expensive server-side and the API enforces
a per-minute budget (HTTP 422 "Timeout. Maybe slow down a bit" plus an
``x-ratelimit-reset`` header). We therefore:

* make one request at a time with a minimum interval between requests,
* back off until the advertised reset on 422/429/5xx,
* cache every successful page under ``pipeline/.cache/arctic`` keyed by its
  query parameters, so reruns are cheap and incremental.

Each subreddit's history is split into fixed time windows (``WINDOW_YEARS``)
and the per-subreddit comment budget is spread across them. Fixed windows keep
cache keys stable between runs (pages of closed windows never change) and give
a sample from every era instead of only the newest comments, whose scores are
not yet meaningful.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import requests

from . import config

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


class ArcticShiftClient:
    def __init__(self, cache_dir: Path = config.CACHE_DIR / "arctic", session: requests.Session | None = None,
                 offline: bool = False, deadline: float | None = None):
        self.offline = offline
        # time.monotonic() after which no new requests are made (cache only),
        # so a time-boxed run still builds from everything fetched so far.
        self.deadline = deadline
        self._deadline_logged = False
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self._last_request = 0.0
        self.requests_made = 0
        self.cache_hits = 0

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, params: dict) -> Path:
        key = json.dumps(params, sort_keys=True)
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def _read_cache(self, params: dict, window_end: int) -> list | None:
        path = self._cache_path(params)
        if not path.exists():
            return None
        now = time.time()
        immutable = window_end < now - config.IMMUTABLE_AFTER_DAYS * 86400
        if not immutable and not self.cache_only() and path.stat().st_mtime < now - config.CACHE_TTL_DAYS * 86400:
            return None
        try:
            return json.loads(path.read_text())["data"]
        except (ValueError, KeyError):
            return None

    def _write_cache(self, params: dict, data: list) -> None:
        path = self._cache_path(params)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"params": params, "data": data}))
        tmp.replace(path)

    def cache_only(self) -> bool:
        """True when uncached pages must not be fetched (offline or out of time)."""
        if self.offline:
            return True
        if self.deadline is not None and time.monotonic() >= self.deadline:
            if not self._deadline_logged:
                log.warning("time budget exhausted: continuing from cached responses only")
                self._deadline_logged = True
            return True
        return False

    # -- http ----------------------------------------------------------------

    def _throttle(self) -> None:
        wait = self._last_request + config.MIN_REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _sleep(self, delay: float) -> None:
        if self.deadline is not None and time.monotonic() + delay > self.deadline:
            raise FetchError("time budget exhausted")
        time.sleep(delay)

    def _get(self, params: dict, url: str = config.ARCTIC_SHIFT_URL) -> tuple[list, int]:
        for attempt in range(1, config.MAX_RETRIES + 1):
            self._throttle()
            started = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                self._last_request = time.monotonic()
                delay = min(60, 2**attempt) + random.random()
                log.warning("request error (%s), retry %d in %.0fs", exc, attempt, delay)
                self._sleep(delay)
                continue
            elapsed = time.monotonic() - started
            self.requests_made += 1
            # Expensive queries eat into the per-minute budget; rest a multiple
            # of the time the server worked before sending the next request.
            self._last_request = time.monotonic() + max(0.0, elapsed * config.REST_FACTOR - config.MIN_REQUEST_INTERVAL)
            log.debug("GET %s r/%s before=%s -> %s in %.1fs", url.rsplit("/api/", 1)[-1], params.get("subreddit"),
                      params.get("before"), resp.status_code, elapsed)
            if resp.status_code == 200:
                body = resp.json()
                if body.get("error"):
                    raise FetchError(body["error"])
                return body.get("data") or [], int(params.get("limit") or 0)
            if resp.status_code in (422, 429) or resp.status_code >= 500:
                if elapsed > config.SLOW_QUERY_SECONDS and int(params.get("limit", 0)) > config.MIN_PAGE_SIZE:
                    # The query itself hit the server-side timeout: scanning for
                    # fewer matches is cheaper, so retry with a smaller page.
                    params = {**params, "limit": max(config.MIN_PAGE_SIZE, int(params["limit"]) // 2)}
                # Retrying every minute keeps the client in the penalty box;
                # back off exponentially (30s, 60s, 120s, ... capped).
                delay = min(config.MAX_BACKOFF_SECONDS, config.BASE_BACKOFF_SECONDS * 2 ** (attempt - 1))
                delay += random.random() * 5
                log.info("HTTP %s (%s), retry %d in %.0fs", resp.status_code, resp.text[:60].strip(), attempt, delay)
                self._sleep(delay)
                continue
            raise FetchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        raise FetchError(f"giving up after {config.MAX_RETRIES} attempts: {params}")

    def search(self, params: dict, window_end: int, url: str = config.ARCTIC_SHIFT_URL) -> tuple[list, int]:
        """Return ``(comments, page_size_used)``.

        The page size may be smaller than requested if the server timed out;
        a page shorter than ``page_size_used`` means the window is exhausted.
        ``url`` selects another search endpoint (e.g. posts) with the same
        paging semantics; it is part of the cache key.
        """
        key = _cache_key(params, url)
        cached = self._read_cache(key, window_end)
        if cached is not None:
            self.cache_hits += 1
            return cached, int(params["limit"])
        for limit in _smaller_limits(int(params["limit"])):
            cached = self._read_cache({**key, "limit": limit}, window_end)
            if cached is not None:
                self.cache_hits += 1
                return cached, limit
        if self.cache_only():
            return [], int(params["limit"])
        data, limit = self._get(params, url)
        self._write_cache({**key, "limit": limit}, data)
        return data, limit

    def get_json(self, url: str, params: dict, window_end: int) -> list:
        """Cached GET of a non-paged Arctic Shift endpoint (e.g. subreddit wikis)."""
        key = _cache_key(params, url)
        cached = self._read_cache(key, window_end)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.cache_only():
            return []
        data, _ = self._get(params, url)
        self._write_cache(key, data)
        return data


def _cache_key(params: dict, url: str) -> dict:
    # Comment-search keys stay as they were so existing caches remain valid.
    return params if url == config.ARCTIC_SHIFT_URL else {**params, "_url": url}


def _smaller_limits(limit: int) -> list[int]:
    out = []
    while limit // 2 >= config.MIN_PAGE_SIZE:
        limit //= 2
        out.append(limit)
    return out


def time_windows(start_year: int = config.START_YEAR, years: int = config.WINDOW_YEARS, now: float | None = None) -> list[tuple[int, int]]:
    """Return [(after, before)] epoch-second windows, newest first.

    The newest window is open-ended (``before`` = start of the next window) so
    its cache key stays stable while it is still filling up.
    """
    now = now if now is not None else time.time()
    current_year = datetime.fromtimestamp(now, timezone.utc).year
    windows = []
    year = start_year
    while year <= current_year:
        after = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
        before = int(datetime(year + years, 1, 1, tzinfo=timezone.utc).timestamp())
        windows.append((after, before))
        year += years
    return list(reversed(windows))


def fetch_subreddit(client: ArcticShiftClient, subreddit: str, max_comments: int) -> Iterator[dict]:
    """Yield up to ``max_comments`` comments for one subreddit.

    The budget is split evenly across time windows; budget left unused by
    sparse (old) windows rolls over to the remaining ones.
    """
    windows = time_windows()
    remaining = max_comments
    limit = config.PAGE_SIZE  # shrinks (sticky) if the server keeps timing out
    for i, (after, before) in enumerate(windows):
        if remaining <= 0:
            break
        pages_left = math.ceil(remaining / config.PAGE_SIZE)
        window_budget = max(1, math.ceil(pages_left / (len(windows) - i))) * config.PAGE_SIZE
        got = 0
        cursor = before
        while got < window_budget:
            params = {
                "subreddit": subreddit,
                "body": config.BODY_QUERY,
                "fields": config.COMMENT_FIELDS,
                "limit": limit,
                "after": after,
                "before": cursor,
            }
            try:
                data, limit = client.search(params, window_end=before)
            except FetchError as exc:
                log.error("r/%s: skipping rest of window %s: %s", subreddit,
                          datetime.fromtimestamp(after, timezone.utc).year, exc)
                break
            for c in data:
                yield c
            got += len(data)
            remaining -= len(data)
            if len(data) < limit:
                break
            oldest = min(int(c["created_utc"]) for c in data)
            if oldest >= cursor:  # defensive: avoid infinite loop
                break
            cursor = oldest


def fetch_all(subreddits: Iterable[str] = config.SUBREDDITS, max_comments: int = config.MAX_COMMENTS_PER_SUB,
              client: ArcticShiftClient | None = None) -> list[dict]:
    client = client or ArcticShiftClient()
    seen: set[str] = set()
    out: list[dict] = []
    for sub in subreddits:
        t0 = time.monotonic()
        n = 0
        try:
            for c in fetch_subreddit(client, sub, max_comments):
                if c["id"] in seen:
                    continue
                seen.add(c["id"])
                out.append(c)
                n += 1
        except FetchError as exc:
            log.error("r/%s: %s (keeping %d comments fetched so far)", sub, exc, n)
        log.info("r/%s: %d comments in %.0fs (requests=%d, cache hits=%d)",
                 sub, n, time.monotonic() - t0, client.requests_made, client.cache_hits)
    return out


_YT_LINK_RE = re.compile(r"youtu\.be/|youtube(?:-nocookie)?\.com/", re.IGNORECASE)


def _links_youtube(body: str) -> bool:
    return bool(_YT_LINK_RE.search(body or ""))


def pick_threads(comments: Iterable[dict], min_links: int = config.THREAD_MIN_LINKS,
                 max_threads: int = config.MAX_THREADS) -> list[tuple[str, int]]:
    """Threads worth expanding: those where several comments already link YouTube.

    Returns ``[(link_id, newest_seen_created_utc)]``, most linked threads first.
    """
    counts: dict[str, int] = {}
    newest: dict[str, int] = {}
    for c in comments:
        link = c.get("link_id")
        if not link or not _links_youtube(c.get("body", "")):
            continue
        counts[link] = counts.get(link, 0) + 1
        newest[link] = max(newest.get(link, 0), int(c.get("created_utc") or 0))
    ranked = sorted((l for l, n in counts.items() if n >= min_links), key=lambda l: (-counts[l], l))
    return [(l, newest[l]) for l in ranked[:max_threads]]


def fetch_threads(client: ArcticShiftClient, threads: list[tuple[str, int]],
                  max_pages: int = config.THREAD_MAX_PAGES) -> list[dict]:
    """Fetch the comments of whole threads (cheap ``link_id`` lookups).

    Full-text search is the API's bottleneck, but threads where people swap
    YouTube recommendations ("best mobility channels?") hold many more links
    than a time-sampled search finds. Only comments that link YouTube are kept.
    """
    out: list[dict] = []
    t0 = time.monotonic()
    for n, (link_id, newest) in enumerate(threads, 1):
        cursor = None
        for _ in range(max_pages):
            params = {"link_id": link_id, "fields": config.COMMENT_FIELDS, "limit": config.PAGE_SIZE}
            if cursor:
                params["before"] = cursor
            try:
                # Threads go quiet quickly: treat them as closed ~IMMUTABLE_AFTER_DAYS
                # after the newest comment we know of.
                data, limit = client.search(params, window_end=newest)
            except FetchError as exc:
                log.error("thread %s: %s", link_id, exc)
                break
            out.extend(c for c in data if _links_youtube(c.get("body", "")))
            if len(data) < limit:
                break
            cursor = min(int(c["created_utc"]) for c in data)
        if n % 50 == 0:
            log.info("threads: %d/%d (%d linking comments, %.0fs)", n, len(threads), len(out), time.monotonic() - t0)
    log.info("threads: %d expanded -> %d comments linking YouTube (requests=%d, cache hits=%d)",
             len(threads), len(out), client.requests_made, client.cache_hits)
    return out


def merge(*batches: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for batch in batches:
        for c in batch:
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
    return out
