"""Hacker News comments linking YouTube fitness videos (Algolia HN Search API).

For every keyword in ``config.HN_KEYWORDS`` and link form (``youtube.com``,
``youtu.be``) we page through ``search_by_date`` newest-first, using a
``created_at_i<cursor`` numeric filter to go past Algolia's 1000-hit page.

HN is mostly not about fitness, so a comment only counts when a fitness term
appears within ``config.HN_NEAR_CHARS`` characters of the YouTube link. Every
mention is still flagged ``needs_llm_check`` for the LLM pass (#3).
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from .. import config
from ..build import is_negative
from ..extract import extract_videos
from . import FITNESS_RE, JsonClient, Mention, SourceError, html_to_text

log = logging.getLogger(__name__)

KIND = "hn"
SITE = "news.ycombinator.com"

def near_fitness_term(text: str, video_id: str, window: int = config.HN_NEAR_CHARS) -> bool:
    """True if a fitness term appears within ``window`` chars of a link to ``video_id``."""
    for m in re.finditer(re.escape(video_id), text):
        lo, hi = max(0, m.start() - window), min(len(text), m.end() + window)
        if FITNESS_RE.search(text[lo:hi]):
            return True
    return False


def parse_hit(hit: dict) -> list[Mention]:
    """Mentions for one Algolia comment hit (empty if irrelevant/unusable)."""
    author = (hit.get("author") or "").strip()
    markup = hit.get("comment_text") or ""
    if not author or not markup:
        return []
    text, _links = html_to_text(markup)
    refs = extract_videos(text)
    if not refs:
        return []
    relevant = [r for r in refs if near_fitness_term(text, r.id)]
    negative = is_negative(text)
    out = []
    for ref in relevant:
        out.append(Mention(
            video_id=ref.id,
            kind=KIND,
            site=SITE,
            author=f"hn:{author.lower()}",
            text=text,
            url=config.HN_ITEM_URL.format(id=hit.get("objectID")),
            created_utc=int(hit.get("created_at_i") or 0),
            score=hit.get("points") if isinstance(hit.get("points"), int) else None,
            negative=negative,
            n_videos=len(refs),
            timestamps=ref.timestamps,
        ))
    return out


def search(client: JsonClient, query: str, max_pages: int) -> list[dict]:
    """All comment hits for ``query``, newest first, up to ``max_pages`` pages."""
    hits: list[dict] = []
    cursor = None
    for page in range(max_pages):
        params = {"query": query, "tags": "comment", "hitsPerPage": config.HN_HITS_PER_PAGE}
        if cursor is not None:
            params["numericFilters"] = f"created_at_i<{cursor}"
        # The first page moves as new comments arrive; older pages are keyed
        # by their cursor and effectively immutable.
        data = client.get(config.HN_SEARCH_URL, params,
                          ttl_days=config.EXTERNAL_SEARCH_TTL_DAYS if cursor is None else None)
        batch = data.get("hits") or []
        hits.extend(batch)
        if len(batch) < config.HN_HITS_PER_PAGE:
            break
        oldest = min(int(h.get("created_at_i") or 0) for h in batch)
        if cursor is not None and oldest >= cursor:
            break
        cursor = oldest
    return hits


def collect(client: JsonClient | None = None, keywords: Iterable[str] = config.HN_KEYWORDS,
            max_pages: int = config.HN_MAX_PAGES, offline: bool = False) -> list[Mention]:
    client = client or JsonClient("hn", config.HN_MIN_REQUEST_INTERVAL, offline=offline)
    seen: set[str] = set()
    mentions: list[Mention] = []
    n_hits = 0
    for kw in keywords:
        for link in config.HN_LINK_QUERIES:
            query = f"{link} {kw}"
            try:
                hits = search(client, query, max_pages)
            except SourceError as exc:
                log.error("hn: query %r failed: %s", query, exc)
                continue
            for hit in hits:
                oid = str(hit.get("objectID"))
                if oid in seen:
                    continue
                seen.add(oid)
                n_hits += 1
                mentions.extend(parse_hit(hit))
    log.info("hn: %d comments -> %d relevant mentions of %d videos (requests=%d, cache hits=%d)",
             n_hits, len(mentions), len({m.video_id for m in mentions}), client.requests_made, client.cache_hits)
    return mentions
