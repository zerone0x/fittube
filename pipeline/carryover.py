"""Merge a fresh build with the previous ``videos.json``.

Arctic Shift is heavily rate-limited, so a run (especially a time-boxed CI run
with a cold cache) may see only part of the data a previous run saw. A video is
therefore never dropped just because a source was unavailable this run: every
previous record is carried over unless YouTube reports the video as gone, and
records seen in both are combined.

The previous file only stores aggregates, not the author sets, so combining is
conservative:

* ``recommenders`` = max of the community (non-wiki) recommenders of both,
  plus one per distinct subreddit wiki listing the video;
* ``score`` = max of the community scores of both (``score`` minus what its
  wiki listings earn: one recommender, i.e. 10 points, plus ``WIKI_BOOST``
  each), plus what the merged ``sources``' wiki listings earn;
* ``comments``/``timestamps``/``subreddits``/``sources``/list tags are unioned,
  ``first_seen`` is the earliest date; title/channel come from the fresh build.

Previous excerpts from bots caught by ``build.is_bot_body`` (added after some
data was built) are removed together with the recommender they counted as.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import config
from .build import OEmbed, is_bot_body, rank_score, sources_boost, wiki_boost

log = logging.getLogger(__name__)

LIST_TAGS = {"goal": 3, "body": 4, "equipment": 3}
_EPOCH = "1970-01-01"


def load(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        log.warning("no previous data at %s (%s); nothing to carry over", path, exc)
        return None
    return data if isinstance(data, dict) and isinstance(data.get("videos"), list) else None


def legacy_sources(v: dict) -> list[dict]:
    """``sources`` for records built before the field existed: Reddit comments."""
    return [{"kind": "comment", "subreddit": s} for s in v.get("subreddits", [])]


def _n_wikis(sources: list[dict]) -> int:
    return len({s.get("subreddit", "").lower() for s in sources if s.get("kind") == "wiki"})


def _wiki_score(sources: list[dict]) -> float:
    """Score a record owes to its wiki listings: each wiki is one recommender
    (``rank_score``: 10 points) plus ``WIKI_BOOST``."""
    n = _n_wikis(sources)
    return rank_score(n, 0) + wiki_boost(n)


def _community_score(v: dict, sources: list[dict]) -> float:
    """``score`` without the part owed to wiki listings."""
    return float(v.get("score") or 0) - _wiki_score(sources)


def _source_key(s: dict) -> tuple:
    kind = s.get("kind")
    if kind in ("wiki", "comment"):
        return kind, (s.get("subreddit") or "").lower()
    return kind, s.get("url") or json.dumps(s, sort_keys=True)


def union_sources(*lists: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for sources in lists:
        for s in sources or []:
            key = _source_key(s)
            old = merged.get(key)
            if old is None:
                merged[key] = dict(s)
            elif s.get("kind") == "comment":
                counts = [c for c in (old.get("count"), s.get("count")) if c is not None]
                if counts:
                    old["count"] = max(counts)
            elif s.get("kind") == "wiki" and len(s.get("path") or "") < len(old.get("path") or ""):
                merged[key] = dict(s)
            elif "score" in s and (old.get("score") or 0) < (s.get("score") or 0):
                old["score"] = s["score"]
    order = {"wiki": 0, "post": 1, "comment": 2}
    return sorted(merged.values(), key=lambda s: order.get(s.get("kind"), 3))


def clean_previous(v: dict) -> dict | None:
    """Drop bot excerpts (and the bot's recommender); ``None`` if nothing is left."""
    v = dict(v)
    comments = v.get("comments") or []
    kept = [c for c in comments if not is_bot_body(c.get("text") or "")]
    if len(kept) != len(comments):
        v["comments"] = kept
        rec = int(v.get("recommenders") or 0)
        v["recommenders"] = max(0, rec - 1)
        v["score"] = round(max(sources_boost(v.get("sources")), float(v.get("score") or 0) - (rank_score(rec, 0) - rank_score(rec - 1, 0))), 1)
        if v["recommenders"] == 0:
            return None
    if "sources" not in v:
        v["sources"] = legacy_sources(v)
    return v


_POST_PERMALINK_RE = re.compile(r"^https://(?:www\.)?reddit\.com/r/[^/]+/comments/([a-z0-9]+)/$")
_POST_EXCERPT_RE = re.compile(r"^(.+?) (https?://\S+)(?: (.*))?$", re.DOTALL)
# An id fragment ends at a non-id character, or directly at ``t=`` when the
# ``&`` before a timestamp was lost too (``watch?v=QnwAoesJvQt=475``).
_ID_FRAGMENT_RE = re.compile(r"(youtu\.be/|[?&]v=|/shorts/|/embed/|/live/)([A-Za-z0-9-]{8,10}?)(?=(t=)|[^A-Za-z0-9_-]|$)")
# A full 11-character id with the ``&`` before its timestamp lost
# (``watch?v=bt4m4zLioRwt=268``), whatever characters the id has.
_GLUED_TIMESTAMP_RE = re.compile(r"(youtu\.be/|[?&]v=|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})(?=t=\d)")


def repair_excerpts(videos: list[dict]) -> int:
    """Fix excerpts saved by older builds; returns the number of excerpts changed.

    * Underscores were stripped from URLs (``youtu.be/K4dmZ5n6uU`` for
      ``K4dmZ5_n6uU``): restored for ids of videos in ``videos``, with the
      ``&`` before a timestamp when it was lost as well (``v=IDt=475``).
    * The ``&`` / ``?`` before a timestamp was lost from any id
      (``v=bt4m4zLioRwt=268``): restored.
    * Post excerpts began with the title and the post's raw URL: the URL is
      replaced by an em dash, as ``build.post_text`` now does.

    Idempotent.
    """
    by_stripped: dict[str, str | None] = {}
    for v in videos:
        vid = v.get("id") or ""
        if "_" in vid:
            key = vid.replace("_", "")
            by_stripped[key] = None if key in by_stripped else vid  # ambiguous -> leave alone

    def fix_id(m: re.Match) -> str:
        vid = by_stripped.get(m.group(2))
        if not vid:
            return m.group(0)
        # Restore the separator before a glued-on timestamp: ``?v=ID&t=``, ``youtu.be/ID?t=``.
        sep = ("&" if m.group(1).endswith("v=") else "?") if m.group(3) else ""
        return m.group(1) + vid + sep

    changed = 0
    for v in videos:
        for c in v.get("comments") or []:
            text = c.get("text") or ""
            new = _ID_FRAGMENT_RE.sub(fix_id, text) if by_stripped else text
            new = _GLUED_TIMESTAMP_RE.sub(
                lambda m: m.group(1) + m.group(2) + ("&" if m.group(1).endswith("v=") else "?"), new)
            post = _POST_PERMALINK_RE.match(c.get("permalink") or "")
            m = _POST_EXCERPT_RE.match(new) if post else None
            if m and "\u2014" not in m.group(1) and (f"/comments/{post.group(1)}" in m.group(2) or "youtu" in m.group(2)):
                new = f"{m.group(1)} \u2014 {m.group(3)}" if m.group(3) else m.group(1)
            if new != text:
                c["text"] = new
                changed += 1
    return changed


def _union(*lists: list, key=lambda x: x) -> list:
    seen, out = set(), []
    for items in lists:
        for x in items or []:
            k = key(x)
            if k not in seen:
                seen.add(k)
                out.append(x)
    return out


def merge_records(prev: dict, fresh: dict) -> dict:
    prev_sources = prev.get("sources") or legacy_sources(prev)
    fresh_sources = fresh.get("sources") or []
    sources = union_sources(fresh_sources, prev_sources)

    community_rec = max(int(prev.get("recommenders") or 0) - _n_wikis(prev_sources),
                        int(fresh.get("recommenders") or 0) - _n_wikis(fresh_sources))
    community_score = max(_community_score(prev, prev_sources), _community_score(fresh, fresh_sources))

    comments = _union(fresh.get("comments"), prev.get("comments"), key=lambda c: c.get("permalink") or c.get("text"))
    comments = _union(comments, key=lambda c: c.get("text"))
    comments.sort(key=lambda c: -(c.get("score") or 0))

    timestamps = sorted(_union(fresh.get("timestamps"), prev.get("timestamps"), key=lambda t: t.get("t")),
                        key=lambda t: t.get("t") or 0)[: config.MAX_TIMESTAMPS]
    dates = [d for d in (prev.get("first_seen"), fresh.get("first_seen")) if d and d != _EPOCH]

    ptags, ftags = prev.get("tags") or {}, fresh.get("tags") or {}
    tags = {k: _union(ptags.get(k), ftags.get(k))[:n] for k, n in LIST_TAGS.items()}
    tags["level"] = ptags.get("level") if ptags.get("level") not in (None, "any") else ftags.get("level", "any")
    tags["type"] = ptags.get("type") or ftags.get("type") or "tutorial"

    out = dict(fresh)
    out.update({
        "recommenders": community_rec + _n_wikis(sources),
        "score": round(community_score + _wiki_score(sources), 1),
        "first_seen": min(dates) if dates else fresh.get("first_seen", _EPOCH),
        "subreddits": _union(fresh.get("subreddits"), prev.get("subreddits"), key=str.lower),
        "comments": comments[: config.TOP_COMMENTS],
        "timestamps": timestamps,
        "tags": tags,
        "sources": sources,
    })
    return out


def merge_previous(fresh: dict, previous: dict | None, oembed: OEmbed | None = None) -> dict:
    """Return ``fresh`` with every still-available previous video merged in."""
    if not previous:
        return fresh
    prev_by_id: dict[str, dict] = {}
    dropped_bot = 0
    for v in previous.get("videos", []):
        if not isinstance(v, dict) or not v.get("id") or v["id"] in config.VIDEO_BLOCKLIST:
            continue
        cleaned = clean_previous(v)
        if cleaned is None:
            dropped_bot += 1
            continue
        prev_by_id[v["id"]] = cleaned

    repair_excerpts(list(prev_by_id.values()) + list(fresh["videos"]))
    fresh_ids = {v["id"] for v in fresh["videos"]}
    carried_ids = [i for i in prev_by_id if i not in fresh_ids]
    oembed = oembed or OEmbed()
    meta = oembed.lookup(carried_ids)
    # Only drop videos YouTube says are gone (cached None), not transient failures.
    gone = {i for i in carried_ids if i in oembed.cache and meta.get(i) is None}

    videos = []
    merged = 0
    for v in fresh["videos"]:
        if v["id"] in prev_by_id:
            v = merge_records(prev_by_id[v["id"]], v)
            merged += 1
        videos.append(v)
    videos += [prev_by_id[i] for i in carried_ids if i not in gone]
    videos.sort(key=lambda r: (-r["score"], -r["recommenders"], r["id"]))
    log.info("carry-over: %d previous videos -> %d merged with fresh, %d carried, %d gone from YouTube, "
             "%d dropped (bot-only)", len(prev_by_id) + dropped_bot, merged, len(carried_ids) - len(gone),
             len(gone), dropped_bot)
    out = dict(fresh)
    out["sources"] = _union(fresh.get("sources"), previous.get("sources"), key=str.lower)
    out["videos"] = videos
    return out
