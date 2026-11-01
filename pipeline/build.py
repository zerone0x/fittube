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


def is_bot(author: str) -> bool:
    a = (author or "").lower()
    return a in config.IGNORED_AUTHORS or a.endswith(config.BOT_SUFFIXES) or a.startswith("bot_")


def is_negative(text: str) -> bool:
    return bool(_NEGATIVE_RE.search(text or ""))


def permalink(comment: dict) -> str:
    link = (comment.get("link_id") or "").removeprefix("t3_")
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
    subreddits: dict[str, int] = field(default_factory=dict)
    first_seen: int | None = None
    comments: list[dict] = field(default_factory=list)
    timestamps: list[tuple[int, str]] = field(default_factory=list)

    @property
    def recommenders(self) -> int:
        return len(self.authors)


def aggregate(comments: Iterable[dict]) -> dict[str, VideoAgg]:
    """Aggregate comments per video.

    A comment counts once per video it links. Each author counts once per
    video regardless of how many times they linked it. Negative-context
    comments contribute only ``NEGATIVE_WEIGHT`` of their score and do not
    count as recommenders. A comment linking several videos (e.g. a list of
    workouts) splits its score evenly between them.
    """
    videos: dict[str, VideoAgg] = {}
    for c in comments:
        author = c.get("author") or ""
        score = int(c.get("score") or 0)
        if score < config.MIN_COMMENT_SCORE or is_bot(author):
            continue
        body = c.get("body") or ""
        refs = extract_videos(body)
        if not refs:
            continue
        negative = is_negative(body)
        weight = config.NEGATIVE_WEIGHT if negative else 1.0
        created = int(c.get("created_utc") or 0)
        for ref in refs:
            v = videos.setdefault(ref.id, VideoAgg(ref.id))
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
    return videos


def rank_score(recommenders: int, sum_score: float) -> float:
    """Ranking score.

    ``recommenders * 10 + log1p(sum_score) * 5``

    Distinct recommenders dominate (independent people vouching for a video is
    the signal we care about); summed comment karma adds a log-damped bonus so a
    single viral comment cannot outrank a video many people recommend.
    """
    return round(recommenders * 10 + math.log1p(max(0.0, sum_score)) * 5, 1)


def passes_threshold(v: VideoAgg) -> bool:
    return v.recommenders >= config.MIN_RECOMMENDERS or v.weighted_score >= config.MIN_SUM_SCORE


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


def to_record(v: VideoAgg, meta: dict) -> dict:
    subreddits = sorted(v.subreddits, key=lambda s: (-v.subreddits[s], s.lower()))
    sum_score = v.weighted_score
    texts = [excerpt(c["text"], 2000) for c in v.comments if not c["negative"]]
    timestamps = [{"t": t, "label": label} for t, label in v.timestamps][: config.MAX_TIMESTAMPS]
    return {
        "id": v.id,
        "title": meta["title"],
        "channel": meta["channel"],
        "thumbnail": f"https://i.ytimg.com/vi/{v.id}/hqdefault.jpg",
        "recommenders": v.recommenders,
        "score": rank_score(v.recommenders, sum_score),
        "first_seen": datetime.fromtimestamp(v.first_seen or 0, timezone.utc).strftime("%Y-%m-%d"),
        "subreddits": subreddits,
        "comments": _top_comments(v),
        "timestamps": timestamps,
        "tags": tag_video(meta["title"], texts, subreddits),
    }


def build(comments: list[dict], sources: list[str], oembed: OEmbed | None = None, limit: int | None = None) -> dict:
    videos = aggregate(comments)
    kept = [v for v in videos.values() if passes_threshold(v)]
    log.info("%d comments -> %d videos -> %d pass threshold", len(comments), len(videos), len(kept))
    kept.sort(key=lambda v: -rank_score(v.recommenders, v.weighted_score))
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
