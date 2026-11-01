"""Reddit submissions that link YouTube, and "recommend me videos" threads.

``/api/posts/search?subreddit=X&query=youtube`` (the ``url=`` filter times out
server-side). Each subreddit's post budget is spread over the same fixed time
windows as the comment search, so cache keys stay stable between runs.

* A post that links a YouTube video (as its URL or in its text) is evidence
  like a comment, its score being the post's karma -- unless it is a question
  or the author sharing their own content.
* Request posts ("best YouTube channels for mobility?") are not evidence
  themselves; their replies are fetched with cheap ``link_id`` lookups and
  processed like normal comments.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Iterable

from .. import config
from ..extract import extract_videos
from ..fetch import ArcticShiftClient, FetchError, time_windows

log = logging.getLogger(__name__)

_SELF_PROMO_RE = re.compile("|".join(config.SELF_PROMO_PATTERNS), re.IGNORECASE)
_ASK_RE = re.compile(
    r"\?|\b(?:looking for|recommend(?:ation)?s?|suggest(?:ion)?s?|any (?:good )?|help|advice|where (?:do|should|can) i"
    r"|what (?:is|are) (?:the|your|some)|need)\b",
    re.IGNORECASE,
)
_VIDEO_RE = re.compile(r"\b(?:youtube|youtubers?|videos?|vids?|channels?|tutorials?|follow[- ]?alongs?)\b",
                       re.IGNORECASE)
_MEDIA_RE = re.compile(_VIDEO_RE.pattern[:-3] + r"|routines?|programs?|resources?)\b", re.IGNORECASE)


def _text(post: dict) -> str:
    return f"{post.get('title') or ''}\n{post.get('url') or ''}\n{post.get('selftext') or ''}"


def is_question(post: dict) -> bool:
    return bool(_ASK_RE.search(post.get("title") or ""))


def is_request(post: dict) -> bool:
    """A post asking for (or collecting) video/channel recommendations, worth expanding.

    Its title mentions videos/channels ("Best YouTube channels for flexibility",
    "Sharing flexibility YouTubers!"), or it asks for help and the text mentions
    them, and it has replies.
    """
    if int(post.get("num_comments") or 0) < config.REQUEST_POST_MIN_COMMENTS:
        return False
    title = post.get("title") or ""
    head = f"{title}\n{(post.get('selftext') or '')[:300]}"
    return bool(_VIDEO_RE.search(title)) or (bool(_ASK_RE.search(title)) and bool(_MEDIA_RE.search(head)))


def is_self_promo(post: dict) -> bool:
    return bool(_SELF_PROMO_RE.search(f"{post.get('title') or ''}\n{post.get('selftext') or ''}"))


def post_evidence(post: dict) -> dict | None:
    """Comment-shaped evidence for a post recommending a video, else ``None``."""
    body = _text(post)
    if not extract_videos(body) or is_question(post) or is_self_promo(post):
        return None
    return {
        "id": f"t3_{post['id']}",
        "kind": "post",
        "author": post.get("author") or "",
        "body": body,
        "score": int(post.get("score") or 0),
        "created_utc": int(post.get("created_utc") or 0),
        "subreddit": post.get("subreddit") or "",
        "link_id": f"t3_{post['id']}",
    }


def fetch_posts(client: ArcticShiftClient, subreddit: str, max_posts: int) -> list[dict]:
    """Up to ``max_posts`` posts matching ``POST_QUERY``, sampled across time windows."""
    windows = time_windows()
    remaining = max_posts
    limit = config.PAGE_SIZE
    out: list[dict] = []
    for i, (after, before) in enumerate(windows):
        if remaining <= 0:
            break
        pages_left = math.ceil(remaining / config.PAGE_SIZE)
        window_budget = max(1, math.ceil(pages_left / (len(windows) - i))) * config.PAGE_SIZE
        got = 0
        cursor = before
        while got < window_budget:
            params = {"subreddit": subreddit, "query": config.POST_QUERY, "fields": config.POST_FIELDS,
                      "limit": limit, "after": after, "before": cursor}
            try:
                data, limit = client.search(params, window_end=before, url=config.ARCTIC_SHIFT_POSTS_URL,
                                            retries=config.POST_MAX_RETRIES)
            except FetchError as exc:
                # Usually the subreddit is too active for keyword search: the
                # other windows would time out the same way.
                log.error("r/%s posts: skipping the rest of this subreddit: %s", subreddit, exc)
                return out
            out.extend(data)
            got += len(data)
            remaining -= len(data)
            if len(data) < limit:
                break
            oldest = min(int(p["created_utc"]) for p in data)
            if oldest >= cursor:
                break
            cursor = oldest
    return out


def collect(client: ArcticShiftClient, subreddits: Iterable[str] = config.SUBREDDITS,
            max_posts: int = config.MAX_POSTS_PER_SUB) -> tuple[list[dict], list[tuple[str, int]]]:
    """Return ``(post_evidence, request_threads)``.

    ``request_threads`` is ``[(link_id, created_utc)]`` for ``fetch.fetch_threads``,
    most-commented first.
    """
    evidence: list[dict] = []
    requests_: list[tuple[int, str, int]] = []
    seen: set[str] = set()
    for sub in subreddits:
        try:
            posts = fetch_posts(client, sub, max_posts)
        except FetchError as exc:  # defensive: fetch_posts already handles per-window errors
            log.error("r/%s posts: %s (skipped)", sub, exc)
            continue
        n_ev = n_req = 0
        for post in posts:
            if not post.get("id") or post["id"] in seen:
                continue
            seen.add(post["id"])
            ev = post_evidence(post)
            if ev:
                evidence.append(ev)
                n_ev += 1
            if is_request(post):
                requests_.append((int(post.get("num_comments") or 0), f"t3_{post['id']}", int(post.get("created_utc") or 0)))
                n_req += 1
        log.info("r/%s posts: %d fetched, %d recommend a video, %d request threads", sub, len(posts), n_ev, n_req)
    requests_.sort(key=lambda r: (-r[0], r[1]))
    return evidence, [(link, created) for _, link, created in requests_]
