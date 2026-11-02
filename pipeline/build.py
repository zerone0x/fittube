"""Turn raw comments into ``site/data/videos.json``.

Steps: extract video refs -> drop noise -> aggregate per video -> threshold ->
oEmbed metadata (drops deleted/private videos) -> tag -> score -> sort.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

from . import config
from .extract import clean_body, extract_videos, merge_timestamps
from .tagger import tag_video

log = logging.getLogger(__name__)

_NEGATIVE_RE = re.compile("|".join(config.NEGATIVE_PATTERNS), re.IGNORECASE)
_BOT_BODY_RE = re.compile("|".join(config.BOT_BODY_PATTERNS), re.IGNORECASE | re.DOTALL)


def is_bot(author: str) -> bool:
    a = (author or "").lower()
    return a in config.IGNORED_AUTHORS or a.endswith(config.BOT_SUFFIXES) or a.startswith("bot_")


def is_bot_body(body: str) -> bool:
    """Bot comments recognised by their text (e.g. "Videos in this thread: Watch Playlist")."""
    return bool(_BOT_BODY_RE.search(clean_body(body)))


def is_negative(text: str) -> bool:
    return bool(_NEGATIVE_RE.search(text or ""))


def permalink(comment: dict) -> str:
    if comment.get("url"):  # wiki pages carry their own URL
        return comment["url"]
    link = (comment.get("link_id") or "").removeprefix("t3_")
    if comment.get("kind") == "post":
        return f"https://www.reddit.com/r/{comment['subreddit']}/comments/{link}/"
    return f"https://www.reddit.com/r/{comment['subreddit']}/comments/{link}/_/{comment['id']}/"


def excerpt(text: str, limit: int = config.EXCERPT_CHARS) -> str:
    text = clean_body(text)
    # Replace markdown links by their text and collapse whitespace.
    text = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"^\s*>.*$", "", text, flags=re.MULTILINE)  # drop quoted parent text
    text = re.sub(r"[*_~`#]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return text


@dataclass
class VideoAgg:
    id: str
    authors: set[str] = field(default_factory=set)
    weighted_score: float = 0.0
    post_score: float = 0.0  # part of weighted_score that is post karma
    subreddits: dict[str, int] = field(default_factory=dict)
    first_seen: int | None = None
    comments: list[dict] = field(default_factory=list)
    timestamps: list[tuple[int, str]] = field(default_factory=list)
    wikis: dict[str, str] = field(default_factory=dict)  # subreddit -> wiki path
    wiki_texts: dict[str, list[str]] = field(default_factory=dict)  # wiki path -> lines
    posts: dict[str, dict] = field(default_factory=dict)  # permalink -> {subreddit, score}
    comment_counts: dict[str, int] = field(default_factory=dict)  # subreddit -> linking comments

    @property
    def recommenders(self) -> int:
        return len(self.authors)

    @property
    def score(self) -> float:
        return rank_score(self.recommenders, self.weighted_score) + wiki_boost(len(self.wikis))


def aggregate(comments: Iterable[dict]) -> dict[str, VideoAgg]:
    """Aggregate comments per video.

    A comment counts once per video it links. Each author counts once per
    video regardless of how many times they linked it. Negative-context
    comments contribute only ``NEGATIVE_WEIGHT`` of their score and do not
    count as recommenders. A comment linking several videos (e.g. a list of
    workouts) splits its score evenly between them.

    Evidence of ``kind="post"`` (a submission) counts like a comment. Evidence
    of ``kind="wiki"`` counts its subreddit's wiki as one recommender
    (``wiki:<subreddit>``), adds no karma and earns ``WIKI_BOOST`` in
    ``VideoAgg.score``; it is not shown as a comment excerpt.
    """
    videos: dict[str, VideoAgg] = {}
    wiki_created: dict[str, int] = {}
    for c in comments:
        kind = c.get("kind") or "comment"
        if kind == "wiki":
            for vid in _add_wiki(videos, c):
                created = int(c.get("created_utc") or 0)
                wiki_created[vid] = min(wiki_created.get(vid) or created, created)
            continue
        author = c.get("author") or ""
        score = int(c.get("score") or 0)
        body = c.get("body") or ""
        if score < config.MIN_COMMENT_SCORE or is_bot(author) or is_bot_body(body):
            continue
        refs = extract_videos(body)
        if not refs:
            continue
        negative = is_negative(body)
        weight = config.NEGATIVE_WEIGHT if negative else 1.0
        created = int(c.get("created_utc") or 0)
        for ref in refs:
            if ref.id in config.VIDEO_BLOCKLIST:
                continue
            v = videos.setdefault(ref.id, VideoAgg(ref.id))
            if kind == "post":
                v.posts[permalink(c)] = {"subreddit": c["subreddit"], "score": score}
                v.post_score += score * weight / len(refs)
            else:
                v.comment_counts[c["subreddit"]] = v.comment_counts.get(c["subreddit"], 0) + 1
            if not negative:
                v.authors.add(author.lower())
            v.weighted_score += score * weight / len(refs)
            v.subreddits[c["subreddit"]] = v.subreddits.get(c["subreddit"], 0) + 1
            if created and (v.first_seen is None or created < v.first_seen):
                v.first_seen = created
            v.comments.append({
                "text": body,
                "score": score,
                "permalink": permalink(c),
                "negative": negative,
                "n_videos": len(refs),
            })
            v.timestamps = merge_timestamps(v.timestamps + ref.timestamps)
    for vid, created in wiki_created.items():
        v = videos[vid]
        if v.first_seen is None and created:  # wiki-only: fall back to the revision date
            v.first_seen = created
    return videos


def _add_wiki(videos: dict[str, VideoAgg], c: dict) -> list[str]:
    sub = c["subreddit"]
    refs = extract_videos(c.get("body") or "")
    for ref in refs:
        v = videos.setdefault(ref.id, VideoAgg(ref.id))
        v.authors.add(f"wiki:{sub.lower()}")
        path = c.get("path") or "/wiki"
        # Prefer the canonical page over translations/variants (shortest path).
        if sub not in v.wikis or (len(path), path) < (len(v.wikis[sub]), v.wikis[sub]):
            v.wikis[sub] = path
        v.subreddits.setdefault(sub, 0)
        v.wiki_texts.setdefault(path, []).append(c.get("body") or "")
        v.timestamps = merge_timestamps(v.timestamps + ref.timestamps)
    return [r.id for r in refs]


def wiki_boost(n_wikis: int) -> float:
    """Extra ranking score for being listed in ``n_wikis`` subreddit wikis."""
    return config.WIKI_BOOST * n_wikis


def sources_boost(sources: Iterable[dict]) -> float:
    """``wiki_boost`` of a ``videos.json`` record's ``sources`` list (its non-community score)."""
    return wiki_boost(len({s.get("subreddit", "").lower() for s in sources or () if s.get("kind") == "wiki"}))


def rank_score(recommenders: int, sum_score: float) -> float:
    """Ranking score.

    ``recommenders * 10 + log1p(sum_score) * 5``

    Distinct recommenders dominate (independent people vouching for a video is
    the signal we care about); summed comment karma adds a log-damped bonus so a
    single viral comment cannot outrank a video many people recommend.
    """
    return round(recommenders * 10 + math.log1p(max(0.0, sum_score)) * 5, 1)


def passes_threshold(v: VideoAgg) -> bool:
    """Wiki-listed, or ``MIN_RECOMMENDERS`` people, or ``MIN_SUM_SCORE`` comment karma.

    Post karma ranks but does not qualify a video on its own: posts linking a
    video are mostly media shares (meet footage, films, news), not
    recommendations, so a lone post only counts as one recommender.
    """
    return (bool(v.wikis) or v.recommenders >= config.MIN_RECOMMENDERS
            or v.weighted_score - v.post_score >= config.MIN_SUM_SCORE)


# -- oEmbed -------------------------------------------------------------------

class OEmbed:
    """YouTube oEmbed lookups with a JSON cache (``None`` = unavailable video)."""

    def __init__(self, cache_path: Path = config.CACHE_DIR / "oembed.json"):
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict | None] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except ValueError:
                self.cache = {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.cache_path)

    def _fetch(self, vid: str) -> tuple[str, dict | None, bool]:
        url = f"https://www.youtube.com/watch?v={vid}"
        for attempt in range(4):
            try:
                r = self.session.get(config.OEMBED_URL, params={"url": url, "format": "json"}, timeout=20)
            except requests.RequestException:
                time.sleep(2**attempt)
                continue
            if r.status_code == 200:
                d = r.json()
                return vid, {"title": d.get("title", ""), "channel": d.get("author_name", "")}, True
            if r.status_code in (400, 401, 403, 404):
                return vid, None, True  # deleted, private or not embeddable
            time.sleep(2**attempt)
        return vid, None, False  # transient failure: don't cache

    def lookup(self, ids: list[str]) -> dict[str, dict | None]:
        todo = [i for i in ids if i not in self.cache]
        log.info("oEmbed: %d cached, %d to fetch", len(ids) - len(todo), len(todo))
        with ThreadPoolExecutor(config.OEMBED_WORKERS) as pool:
            for n, (vid, meta, cacheable) in enumerate(pool.map(self._fetch, todo), 1):
                if cacheable:
                    self.cache[vid] = meta
                if n % 200 == 0:
                    log.info("oEmbed: %d/%d", n, len(todo))
                    self.save()
        self.save()
        return {i: self.cache.get(i) for i in ids}


# -- output -------------------------------------------------------------------

def _top_comments(v: VideoAgg) -> list[dict]:
    # Prefer positive comments, then single-video comments (more specific), then karma.
    ranked = sorted(v.comments, key=lambda c: (c["negative"], -c["score"], c["n_videos"]))
    out, seen = [], set()
    for c in ranked:
        text = excerpt(c["text"])
        if not text or text in seen:
            continue
        seen.add(text)
        out.append({"text": text, "score": c["score"], "permalink": c["permalink"]})
        if len(out) == config.TOP_COMMENTS:
            break
    return out


MAX_POST_SOURCES = 3


def source_entries(v: VideoAgg) -> list[dict]:
    """``sources`` of a record: wikis first, then top posts, then comment counts per subreddit."""
    out: list[dict] = [
        {"kind": "wiki", "subreddit": sub, "path": path, "url": f"https://www.reddit.com/r/{sub}{path}"}
        for sub, path in sorted(v.wikis.items(), key=lambda kv: kv[0].lower())
    ]
    posts = sorted(v.posts.items(), key=lambda kv: (-kv[1]["score"], kv[0]))[:MAX_POST_SOURCES]
    out += [{"kind": "post", "subreddit": p["subreddit"], "url": url, "score": p["score"]} for url, p in posts]
    out += [{"kind": "comment", "subreddit": sub, "count": n}
            for sub, n in sorted(v.comment_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))]
    return out


def to_record(v: VideoAgg, meta: dict) -> dict:
    subreddits = sorted(v.subreddits, key=lambda s: (-v.subreddits[s], s.lower()))
    texts = [excerpt(c["text"], 2000) for c in v.comments if not c["negative"]]
    # Lines of the canonical wiki pages only (translations would repeat them).
    texts += [excerpt(t, 2000) for path in set(v.wikis.values()) for t in v.wiki_texts.get(path, [])]
    timestamps = [{"t": t, "label": label} for t, label in v.timestamps][: config.MAX_TIMESTAMPS]
    return {
        "id": v.id,
        "title": meta["title"],
        "channel": meta["channel"],
        "thumbnail": f"https://i.ytimg.com/vi/{v.id}/hqdefault.jpg",
        "recommenders": v.recommenders,
        "score": round(v.score, 1),
        "first_seen": datetime.fromtimestamp(v.first_seen or 0, timezone.utc).strftime("%Y-%m-%d"),
        "subreddits": subreddits,
        "comments": _top_comments(v),
        "timestamps": timestamps,
        "tags": tag_video(meta["title"], texts, subreddits, hints=list(v.wikis.values())),
        "sources": source_entries(v),
    }


def build(comments: list[dict], sources: list[str], oembed: OEmbed | None = None, limit: int | None = None) -> dict:
    videos = aggregate(comments)
    kept = [v for v in videos.values() if passes_threshold(v)]
    log.info("%d comments -> %d videos -> %d pass threshold", len(comments), len(videos), len(kept))
    kept.sort(key=lambda v: -v.score)
    if limit:
        kept = kept[:limit]
    oembed = oembed or OEmbed()
    meta = oembed.lookup([v.id for v in kept])
    records = [to_record(v, meta[v.id]) for v in kept if meta.get(v.id)]
    log.info("%d videos available on YouTube", len(records))
    records.sort(key=lambda r: (-r["score"], -r["recommenders"], r["id"]))
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": list(sources),
        "videos": records,
    }


def write(data: dict, path: Path = config.OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)
