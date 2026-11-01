"""Merge non-Reddit mentions (``sources.Mention``) into a built ``videos.json``.

This runs after the Reddit build (``build.build``) and also works on an
existing ``videos.json`` (``python -m pipeline.run --merge-into``), so new
sources can be added without re-crawling Arctic Shift.

* Recommenders are namespaced per source (``hn:pg``, ``tnation:alice``), so
  the union with a video's Reddit recommenders is a plain sum.
* Videos already on the site take any relevant mention as corroboration; new
  videos need ``MIN_RECOMMENDERS`` distinct external recommenders (forum
  likes only affect ranking).
* Mentions by an author whose name matches the video's channel, or by a site
  linking its own channel (T-Nation articles), are dropped (self-promotion).
* New videos known only from HN also need a fitness term in their YouTube
  title (HN threads drift off-topic easily).
* Per-source counts are kept in ``source_stats`` so a rerun first strips the
  previous merge (``strip_external``) and the result is idempotent.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .. import config
from ..build import OEmbed, excerpt, rank_score
from ..extract import merge_timestamps
from ..tagger import tag_video
from . import TITLE_FITNESS_RE, Mention

log = logging.getLogger(__name__)

EXTERNAL_KINDS = ("hn", "forum")
SITES = {"hn": "news.ycombinator.com", "forum": "t-nation.com"}


@dataclass
class ExtAgg:
    id: str
    mentions: list[Mention] = field(default_factory=list)

    def authors(self, kind: str | None = None) -> set[str]:
        return {m.author for m in self.mentions if not m.negative and (kind is None or m.kind == kind)}

    def sum_score(self, kind: str | None = None) -> float:
        return sum((m.score or 0) * (config.NEGATIVE_WEIGHT if m.negative else 1.0) / m.n_videos
                   for m in self.mentions if kind is None or m.kind == kind)

    def passes_threshold(self) -> bool:
        # Forum likes are a much weaker signal than Reddit karma (and a single
        # poster can link their favourite clip in many threads), so new
        # videos need several distinct people; likes only affect ranking.
        return len(self.authors()) >= config.MIN_RECOMMENDERS


def aggregate(mentions: list[Mention]) -> dict[str, ExtAgg]:
    out: dict[str, ExtAgg] = {}
    for m in mentions:
        out.setdefault(m.video_id, ExtAgg(m.video_id)).mentions.append(m)
    return out


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def drop_self_promotion(agg: ExtAgg, channel: str) -> None:
    """Drop mentions by the channel itself, or by a site linking its own channel."""
    ch = _norm(channel)
    if not ch:
        return
    agg.mentions = [
        m for m in agg.mentions
        if _norm(m.author.split(":", 1)[-1]) != ch
        and ch not in {_norm(c) for c in config.EXTERNAL_SITE_CHANNELS.get(m.site, ())}
    ]


def reddit_sum_score(score: float, recommenders: int) -> float:
    """Invert ``build.rank_score`` (exact up to its 0.1 rounding)."""
    return max(0.0, round(math.expm1(max(0.0, (score - recommenders * 10) / 5)), 1))


def strip_external(data: dict) -> dict:
    """Undo a previous ``merge_external`` so merging again is idempotent."""
    kept = []
    for v in data.get("videos", []):
        stats = v.pop("source_stats", None)
        v.pop("needs_llm_check", None)
        if stats is not None:
            base = stats.get("reddit") or {"recommenders": 0, "sum_score": 0.0}
            if not base["recommenders"] and not base["sum_score"] and not v.get("subreddits"):
                continue  # video came from external sources only
            v["recommenders"] = base["recommenders"]
            v["score"] = rank_score(base["recommenders"], base["sum_score"])
            if "first_seen" in base:
                v["first_seen"] = base["first_seen"]
        v["comments"] = [c for c in v.get("comments", []) if c.get("site") in (None, "reddit.com")]
        v["timestamps"] = [t for t in v.get("timestamps", []) if not t.get("site")]
        if "sources" in v:
            v["sources"] = [s for s in v["sources"] if s.get("kind") not in EXTERNAL_KINDS]
            if not v["sources"]:
                del v["sources"]
        kept.append(v)
    kept.sort(key=lambda r: (-r["score"], -r["recommenders"], r["id"]))
    data["videos"] = kept
    data["sources"] = [s for s in data.get("sources", []) if s not in SITES.values()]
    return data


def _comment(m: Mention) -> dict:
    return {"text": excerpt(m.text), "score": m.score, "permalink": m.url, "site": m.site}


def _best_comments(mentions: list[Mention], per_site: int | None, total: int) -> list[dict]:
    ranked = sorted(mentions, key=lambda m: (m.negative, -(m.score or 0), m.n_videos, -m.created_utc))
    out, seen, by_site = [], set(), defaultdict(int)
    for m in ranked:
        c = _comment(m)
        if not c["text"] or c["text"] in seen or (per_site and by_site[m.site] >= per_site):
            continue
        seen.add(c["text"])
        by_site[m.site] += 1
        out.append(c)
        if len(out) == total:
            break
    return out


def _source_entries(mentions: list[Mention]) -> list[dict]:
    out, seen = [], set()
    for m in sorted(mentions, key=lambda m: (-(m.score or 0), -m.created_utc)):
        if m.url in seen:
            continue
        seen.add(m.url)
        out.append({"kind": m.kind, "site": m.site, "url": m.url, "score": m.score,
                    "needs_llm_check": m.needs_llm_check})
    return out


def _day(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _apply(v: dict, agg: ExtAgg, reddit: dict) -> dict:
    ext_authors = agg.authors()
    stats = {"reddit": reddit}
    for kind in EXTERNAL_KINDS:
        if any(m.kind == kind for m in agg.mentions):
            stats[kind] = {"recommenders": len(agg.authors(kind)), "sum_score": round(agg.sum_score(kind), 1)}
    v["recommenders"] = reddit["recommenders"] + len(ext_authors)
    v["score"] = rank_score(v["recommenders"], reddit["sum_score"] + agg.sum_score())
    dates = [d for d in [v.get("first_seen")] + [_day(m.created_utc) for m in agg.mentions if m.created_utc]
             if d and d > "1970-01-01"]
    if dates:
        v["first_seen"] = min(dates)
    # Keep existing timestamps verbatim; add external ones (tagged with their
    # site so strip_external can remove them) that are not near an existing one.
    have = v.get("timestamps", [])
    ext_ts: dict[int, tuple[str, str]] = {}
    for m in agg.mentions:
        if m.negative:
            continue
        for t, label in merge_timestamps(m.timestamps):
            if all(abs(t - h["t"]) > 3 for h in have) and (t not in ext_ts or (label and not ext_ts[t][0])):
                ext_ts[t] = (label, m.site)
    added = [(t, label) for t, label in merge_timestamps([(t, lab) for t, (lab, _) in ext_ts.items()])]
    extra = [{"t": t, "label": label, "site": ext_ts.get(t, ("", SITES["hn"]))[1]} for t, label in added]
    v["timestamps"] = sorted(have + extra, key=lambda x: x["t"])[: config.MAX_TIMESTAMPS]
    if v.get("comments"):
        v["comments"] = v["comments"] + _best_comments(agg.mentions, per_site=1, total=len(SITES))
    else:
        v["comments"] = _best_comments(agg.mentions, per_site=None, total=config.TOP_COMMENTS)
    v["sources"] = v.get("sources", []) + _source_entries(agg.mentions)
    v["source_stats"] = stats
    # Only external, keyword-filtered evidence: let the LLM pass (#3) vet it.
    v["needs_llm_check"] = not reddit["recommenders"] and not v.get("subreddits")
    return v


def merge_external(data: dict, mentions: list[Mention], oembed: OEmbed | None = None) -> dict:
    """Fold ``mentions`` into ``data`` (a ``videos.json`` dict) in place and return it."""
    data = strip_external(data)
    by_id = {v["id"]: v for v in data["videos"]}
    aggs = aggregate(mentions)

    new_ids = [vid for vid, a in aggs.items() if vid not in by_id and a.passes_threshold()]
    oembed = oembed or OEmbed()
    meta = oembed.lookup(new_ids) if new_ids else {}

    merged = added = 0
    for vid, agg in aggs.items():
        if vid in by_id:
            v = by_id[vid]
            drop_self_promotion(agg, v.get("channel", ""))
            if not agg.mentions:
                continue
            reddit = {"recommenders": v["recommenders"], "sum_score": reddit_sum_score(v["score"], v["recommenders"])}
            if v.get("first_seen"):
                reddit["first_seen"] = v["first_seen"]
            _apply(v, agg, reddit)
            merged += 1
        elif meta.get(vid):
            m = meta[vid]
            drop_self_promotion(agg, m["channel"])
            if not agg.mentions or not agg.passes_threshold():
                continue
            if all(x.kind == "hn" for x in agg.mentions) and not TITLE_FITNESS_RE.search(m["title"]):
                continue  # HN-only evidence: the title must look like fitness too
            texts = [excerpt(x.text, 2000) for x in agg.mentions if not x.negative]
            v = {
                "id": vid,
                "title": m["title"],
                "channel": m["channel"],
                "thumbnail": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
                "recommenders": 0,
                "score": 0.0,
                "first_seen": "",
                "subreddits": [],
                "comments": [],
                "timestamps": [],
                "tags": tag_video(m["title"], texts, []),
            }
            _apply(v, agg, {"recommenders": 0, "sum_score": 0.0})
            data["videos"].append(v)
            added += 1
    log.info("external sources: %d mentions of %d videos -> %d merged into existing, %d new",
             len(mentions), len(aggs), merged, added)

    data["videos"].sort(key=lambda r: (-r["score"], -r["recommenders"], r["id"]))
    kinds = {m.kind for m in mentions}
    data["sources"] = list(data.get("sources", [])) + [SITES[k] for k in EXTERNAL_KINDS if k in kinds]
    return data
