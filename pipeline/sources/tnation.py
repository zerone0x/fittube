"""T-Nation forum (Discourse) posts linking YouTube videos.

``/search.json?q=youtube.com order:latest&page=N`` lists matching posts (with
their topic's category and ``like_count``), newest first; each kept post's full
HTML comes from ``/t/<topic_id>/<post_number>.json``. Requests are spaced
``config.TNATION_MIN_REQUEST_INTERVAL`` apart and every response is cached.

Many links are users' own training clips, so posts in training-log (and
off-topic) categories are dropped, as are posts whose text reads like
"my set / form check / here's me". Outside the training categories a post
must also mention a fitness term. The rest is flagged ``needs_llm_check``.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

from .. import config
from ..build import is_negative
from ..extract import extract_videos
from . import FITNESS_RE, JsonClient, Mention, SourceError, html_to_text

log = logging.getLogger(__name__)

KIND = "forum"
SITE = "t-nation.com"
SEARCH_PAGE_LIMIT = 10  # Discourse rejects page > 10

_MY = (r"(?:last |top |heavy |working |first |second |third |final |new |latest |current |old )?"
       r"(?:sets?|lifts?|attempts?|prs?|squats?|bench(?:es)?|deadlifts?|pulls?|press(?:es)?|cleans?|snatch(?:es)?"
       r"|jerks?|form|technique|video|vid|clip|footage|meet|channel)")
OWN_CLIP_RE = re.compile(
    r"\bmy " + _MY + r"\b"
    r"|\bform[ -]?checks?\b|\bcheck (?:out )?my\b|\bcritique\b"
    r"|\bhere'?s me\b|\bthis is me\b|\b(?:video|vid|clip|footage) of me\b"
    r"|\bme (?:doing|lifting|squatting|benching|pulling|pressing|hitting|attempting)\b"
    r"|\bi (?:filmed|recorded|uploaded)\b"
    r"|\bnew (?:pr|record)\b|\bpr'?d\b|\bugly attempt\b|\byours truly\b",
    re.IGNORECASE,
)
DROP_TOPIC_RE = re.compile(config.TNATION_DROP_TOPIC_PATTERN, re.IGNORECASE)


def is_own_clip(text: str) -> bool:
    return bool(OWN_CLIP_RE.search(text or ""))


def _epoch(iso: str) -> int:
    try:
        return int(datetime.fromisoformat((iso or "").replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def _likes(post: dict, fallback: int | None) -> int | None:
    if isinstance(post.get("like_count"), int):
        return post["like_count"]
    for a in post.get("actions_summary") or []:
        if a.get("id") == 2 and isinstance(a.get("count"), int):  # 2 = like
            return a["count"]
    if isinstance(post.get("reaction_users_count"), int):
        return max(post["reaction_users_count"], fallback or 0)
    return fallback


def parse_post(post: dict, topic: dict | None = None, like_count: int | None = None,
               trusted: bool = True) -> list[Mention]:
    """Mentions for one full Discourse post.

    Empty for own clips, posts in log/challenge topics, posts without a video
    and -- unless the post is in a trusted (training) category -- posts that
    never mention a fitness term.
    """
    username = (post.get("username") or "").strip()
    if not username or post.get("hidden") or post.get("user_deleted") or post.get("deleted_at"):
        return []
    text, links = html_to_text(post.get("cooked") or "")
    if is_own_clip(text) or DROP_TOPIC_RE.search((topic or {}).get("title") or ""):
        return []
    if not trusted and not FITNESS_RE.search(text + "\n" + ((topic or {}).get("title") or "")):
        return []
    refs = extract_videos(text + "\n" + "\n".join(links))
    if not refs:
        return []
    topic_id = post.get("topic_id") or (topic or {}).get("id")
    slug = post.get("topic_slug") or (topic or {}).get("slug") or "-"
    url = f"{config.TNATION_BASE_URL}/t/{slug}/{topic_id}/{post.get('post_number')}"
    negative = is_negative(text)
    return [Mention(
        video_id=ref.id,
        kind=KIND,
        site=SITE,
        author=f"tnation:{username.lower()}",
        text=text,
        url=url,
        created_utc=_epoch(post.get("created_at")),
        score=_likes(post, like_count),
        negative=negative,
        n_videos=len(refs),
        timestamps=ref.timestamps,
    ) for ref in refs]


def category_slugs(client: JsonClient) -> dict[int, str]:
    data = client.get(f"{config.TNATION_BASE_URL}/categories.json", {"include_subcategories": "true"},
                      ttl_days=config.EXTERNAL_POST_TTL_DAYS)
    out: dict[int, str] = {}
    for c in (data.get("category_list") or {}).get("categories") or []:
        out[c["id"]] = c.get("slug") or ""
        for s in c.get("subcategory_list") or []:
            out[s["id"]] = s.get("slug") or ""
    return out


def search_posts(client: JsonClient, max_pages: int, drop_category_ids: set[int]) -> list[tuple[dict, dict]]:
    """``[(search_post, topic)]`` for posts outside the dropped categories.

    Discourse serves at most ``SEARCH_PAGE_LIMIT`` pages per query, so deeper
    history is reached by re-querying with ``before:<day after oldest hit>``.
    """
    out: list[tuple[dict, dict]] = []
    seen: set[int] = set()
    dropped = 0
    before: str | None = None
    pages = 0
    while pages < max_pages:
        q = config.TNATION_QUERY + (f" before:{before}" if before else "")
        oldest = None
        more = False
        for page in range(1, SEARCH_PAGE_LIMIT + 1):
            if pages >= max_pages:
                break
            pages += 1
            try:
                data = client.get(f"{config.TNATION_BASE_URL}/search.json", {"q": q, "page": page},
                                  ttl_days=config.EXTERNAL_SEARCH_TTL_DAYS)
            except SourceError as exc:
                log.error("tnation: search %r page %d failed, stopping: %s", q, page, exc)
                return _done(out, dropped)
            posts = data.get("posts") or []
            topics = {t["id"]: t for t in data.get("topics") or []}
            for p in posts:
                created = (p.get("created_at") or "")[:10]
                if created and (oldest is None or created < oldest):
                    oldest = created
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                topic = topics.get(p.get("topic_id")) or {}
                if topic.get("category_id") in drop_category_ids:
                    dropped += 1
                    continue
                out.append((p, topic))
            more = bool(posts) and bool((data.get("grouped_search_result") or {}).get("more_full_page_results"))
            if not more:
                break
        if not more or not oldest:
            break
        # before: is exclusive; start at the day after the oldest hit (dupes are skipped).
        nxt = (date.fromisoformat(oldest) + timedelta(days=1)).isoformat()
        if nxt == before:
            break  # a single day holds more than a full query's worth of hits
        before = nxt
    return _done(out, dropped)


def _done(out: list, dropped: int) -> list:
    log.info("tnation: %d search hits kept, %d dropped by category", len(out), dropped)
    return out


def fetch_post(client: JsonClient, topic_id: int, post_number: int, known: dict[int, dict]) -> dict | None:
    """Full post JSON; a topic chunk also yields neighbouring posts, kept in ``known``."""
    data = client.get(f"{config.TNATION_BASE_URL}/t/{topic_id}/{post_number}.json",
                      ttl_days=config.EXTERNAL_POST_TTL_DAYS)
    found = None
    for p in (data.get("post_stream") or {}).get("posts") or []:
        p.setdefault("topic_slug", data.get("slug"))
        known[p["id"]] = p
        if p.get("post_number") == post_number:
            found = p
    return found


def collect(client: JsonClient | None = None, max_pages: int = config.TNATION_MAX_PAGES,
            offline: bool = False) -> list[Mention]:
    client = client or JsonClient("tnation", config.TNATION_MIN_REQUEST_INTERVAL, offline=offline)
    try:
        slugs = category_slugs(client)
    except SourceError as exc:
        log.error("tnation: cannot load categories (needed to drop training logs), skipping source: %s", exc)
        return []
    drop = {cid for cid, slug in slugs.items() if slug in config.TNATION_DROP_CATEGORIES}
    hits = search_posts(client, max_pages, drop)
    known: dict[int, dict] = {}
    mentions: list[Mention] = []
    for n, (hit, topic) in enumerate(hits, 1):
        post = known.get(hit["id"])
        if post is None:
            try:
                post = fetch_post(client, hit["topic_id"], hit["post_number"], known)
            except SourceError as exc:
                log.warning("tnation: post %s: %s", hit["id"], exc)
                continue
        if post is None:
            continue
        trusted = slugs.get(topic.get("category_id"), "") in config.TNATION_TRUSTED_CATEGORIES
        mentions.extend(parse_post(post, topic, hit.get("like_count"), trusted=trusted))
        if n % 100 == 0:
            log.info("tnation: %d/%d posts (%d mentions, requests=%d)", n, len(hits), len(mentions), client.requests_made)
    log.info("tnation: %d posts -> %d mentions of %d videos (requests=%d, cache hits=%d)",
             len(hits), len(mentions), len({m.video_id for m in mentions}), client.requests_made, client.cache_hits)
    return mentions
