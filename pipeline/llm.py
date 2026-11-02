"""LLM pass (issue #3): vet recommendation evidence, drop non-recommended
videos, tag and summarise the rest.

It runs on a built ``videos.json`` dict -- after carry-over and the HN /
T-Nation merge -- because that is the only place every video (including
carried-over ones) is available. Two stages, both through the Anthropic
Message Batches API (or plain requests for a handful of items), with
structured output via forced tool use:

* **Stage A -- evidence** (``LLM_CLASSIFY_MODEL``, default Haiku): one request
  per video listing the comment/post excerpts saved for it. For each excerpt:
  ``is_recommendation``, ``sentiment``, ``quality_flags``, ``reason``.
* **Stage B -- vetting and tags** (``LLM_MODEL`` / ``FITTUBE_MODEL``, default
  Sonnet): one request per video that survived stage A: ``keep`` /
  ``reject_reason``, tags in the README vocabularies and a one-sentence
  ``summary``.

Results are cached in ``pipeline/llm_cache.json`` (committed), keyed by prompt
version + video id (+ comment id for stage A), so weekly refreshes only pay
for new items and runs without ``ANTHROPIC_API_KEY`` reproduce the same output
from the cache. Batches still running when the wait budget is spent are kept
in the cache (``pending``) and collected by the next run.

``apply_verdicts`` turns verdicts into data (see its docstring); every change
it makes to counts is recorded in the record's ``llm_rejected`` field so
``strip_llm`` can undo it before the next carry-over -- the pass is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import config
from .build import rank_score, sources_boost
from .tagger import BODY, EQUIPMENT, GOALS, LEVELS, TYPES

log = logging.getLogger(__name__)

# Bump when a prompt or schema changes in a way that invalidates cached answers.
PROMPT_A_VERSION = "a1"
PROMPT_B_VERSION = "b1"

SENTIMENTS = ["positive", "neutral", "negative"]
QUALITY_FLAGS = ["bro-science", "sales/affiliate", "unsafe-form", "self-promotion", "off-topic",
                 "personal-clip", "event-footage", "joke"]
REJECT_REASONS = ["none", "non-fitness", "personal-clip", "event-footage", "anti-recommendation", "spam"]
MAX_TAGS = {"goal": 3, "body": 4, "equipment": 3}
SUMMARY_CHARS = 220
REASON_CHARS = 160
COMMENT_CHARS = 600

# -- prompts --------------------------------------------------------------------

TOOL_A = {
    "name": "classify_evidence",
    "description": "Record a verdict for every numbered comment/post.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The [n] number of the comment."},
                        "is_recommendation": {"type": "boolean"},
                        "sentiment": {"type": "string", "enum": SENTIMENTS},
                        "quality_flags": {"type": "array", "items": {"type": "string", "enum": QUALITY_FLAGS}},
                        "reason": {"type": "string", "description": "At most 15 words."},
                    },
                    "required": ["index", "is_recommendation", "sentiment", "quality_flags", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["comments"],
        "additionalProperties": False,
    },
}

SYSTEM_A = """\
You review evidence for FitTube, a site that lists YouTube videos fitness communities \
(Reddit, Hacker News, the T-Nation forum) recommend. Every numbered comment or post below links \
the given video. For each one decide whether its author recommends the video to others as a \
worthwhile fitness resource: instruction, a follow-along routine, a program, an explainer or \
lecture, rehab advice, or a credible training documentary/motivation.

It is NOT a recommendation (is_recommendation=false) when the author:
- cites the video as wrong, bad, dangerous or something to avoid, or quotes it in order to disagree \
(e.g. "Robert Oberst says you shouldn't do deadlifts" used as a bad example);
- mocks it, or links it as a joke, meme, music, film clip or anything unrelated to training;
- shares their own (or a friend's) lift, attempt, PR, form check or physique, or asks people to \
guess their body fat / rate their physique;
- shares competition, meet, race or festival footage, highlights or news for entertainment;
- only asks a question about it without endorsing it;
- promotes their own channel or a product.
A recommendation can be brief ("this video fixed my squat") or part of a list of resources.

sentiment is the author's attitude toward the video. quality_flags lists concerns about the video \
or the link (empty if none). reason: at most 15 words. Return exactly one entry per [n] via the tool.\
"""

TOOL_B = {
    "name": "review_video",
    "description": "Record whether the video belongs on FitTube, its tags and a summary.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "keep": {"type": "boolean"},
            "reject_reason": {"type": "string", "enum": REJECT_REASONS},
            "goal": {"type": "array", "items": {"type": "string", "enum": GOALS}},
            "body": {"type": "array", "items": {"type": "string", "enum": BODY}},
            "equipment": {"type": "array", "items": {"type": "string", "enum": EQUIPMENT}},
            "level": {"type": "string", "enum": LEVELS},
            "type": {"type": "string", "enum": TYPES},
            "summary": {"type": "string"},
        },
        "required": ["keep", "reject_reason", "goal", "body", "equipment", "level", "type", "summary"],
        "additionalProperties": False,
    },
}

SYSTEM_B = f"""\
You curate FitTube, a site that lists YouTube videos fitness communities recommend. You get a \
video's title and channel, where it was recommended (subreddits, subreddit wiki pages, other sites) \
and the community comments that link it. Decide whether it belongs on the site, tag it and \
summarise it.

keep=true for instructional videos, follow-along workouts and routines, programs, exercise \
science explainers and lectures, rehab/physical-therapy content, sport technique for strength, \
mobility, running, climbing, swimming or martial arts training, and training documentaries or \
motivational talks about training.
keep=false (reject_reason) when the video is:
- non-fitness: music or album streams, films, games, comedy, politics, urbanism, cooking unrelated \
to training, general vlogs; a gym-sounding title is not enough (e.g. "Not Just Bikes - The Gym of \
Life" is about cities, "Electric Callboy - PUMP IT" is music);
- personal-clip: someone's own lift/attempt/PR/form check/physique (e.g. "405 x 20 Squat");
- event-footage: competition, meet, race, festival or comp coverage and highlights (e.g. "The \
North Face Climb Festival"), and climbing topics that are not training (comps, knots, gear);
- anti-recommendation: the comments cite the video as an example of what not to do;
- spam: ads or sales pitches.
Use reject_reason "none" when keep=true.

Tags (only these values; an empty list when nothing clearly applies):
- goal (max 3): strength, hypertrophy (muscle growth), fat-loss, mobility (flexibility/stretching), \
rehab (pain, injury, posture, prehab), endurance (cardio, running, conditioning), skill \
(gymnastics/calisthenics skills, handstands, sport or lift technique).
- body (max 4): full, upper, lower, core, back, chest, shoulders, arms, legs, glutes, neck, hips.
- equipment (max 3): none (bodyweight, no equipment), dumbbell, barbell, kettlebell, band, rings, \
bar (pull-up/dip bar, parallettes), machine (machines, cables, cardio machines).
- level: beginner, intermediate, advanced, or any when unclear or suitable for all.
- type: tutorial (how to do a movement), follow-along (do it with the video in real time), \
explainer (science, concepts, lecture, podcast), program (a training plan or routine design), \
motivation (story, documentary, speech).

summary: one plain sentence (max 25 words, max {SUMMARY_CHARS} characters) saying what the video \
teaches or offers, based only on the given information. No hype; do not invent details.\
"""


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _source_label(c: dict, v: dict) -> str:
    link = c.get("permalink") or ""
    site = c.get("site")
    if site == "news.ycombinator.com" or "ycombinator.com" in link:
        return "Hacker News comment"
    if site == "t-nation.com" or "t-nation.com" in link:
        return "T-Nation forum post"
    m = re.search(r"reddit\.com/r/([^/]+)/comments/[a-z0-9]+/(?:_/[a-z0-9]+/)?$", link)
    if m:
        return f"r/{m.group(1)} {'post' if '/_/' not in link else 'comment'}"
    return "comment"


def prompt_a(v: dict, comments: list[dict]) -> str:
    lines = [f'Video: "{v.get("title", "")}" by {v.get("channel") or "unknown channel"}', "",
             "Comments/posts linking this video:"]
    for i, c in enumerate(comments, 1):
        score = c.get("score")
        karma = f", score {score}" if score is not None else ""
        lines.append(f"[{i}] {_source_label(c, v)}{karma}: {_clip(c.get('text', ''), COMMENT_CHARS)}")
    return "\n".join(lines)


def prompt_b(v: dict, comments: list[dict]) -> str:
    wikis = [f"r/{s.get('subreddit')}{s.get('path') or '/wiki'}" for s in v.get("sources") or []
             if s.get("kind") == "wiki"]
    sites = sorted({s.get("site") for s in v.get("sources") or [] if s.get("site")})
    lines = [f'Title: "{v.get("title", "")}"', f"Channel: {v.get('channel') or 'unknown'}"]
    if v.get("subreddits"):
        lines.append("Recommended in: " + ", ".join(f"r/{s}" for s in v["subreddits"][:8]))
    if wikis:
        lines.append("Listed in subreddit wiki pages: " + ", ".join(wikis[:5]))
    if sites:
        lines.append("Also recommended on: " + ", ".join(sites))
    labels = [t.get("label") for t in v.get("timestamps") or [] if t.get("label")]
    if labels:
        lines.append("Timestamp labels: " + "; ".join(_clip(x, 60) for x in labels[:6]))
    if comments:
        lines += ["", "Comments:"] + [f"- {_clip(c.get('text', ''), COMMENT_CHARS)}" for c in comments[:4]]
    else:
        lines += ["", "No comment excerpts saved."]
    return "\n".join(lines)


def request_params(stage: str, model: str, user: str) -> dict:
    tool, system, max_tokens = (TOOL_A, SYSTEM_A, 1500) if stage == "A" else (TOOL_B, SYSTEM_B, 600)
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
        "messages": [{"role": "user", "content": user}],
    }


# -- validation -----------------------------------------------------------------

def validate_a(out: Any, n: int) -> dict[int, dict] | None:
    """``{index (1-based): verdict}`` from a stage-A tool input, ``None`` if unusable."""
    if not isinstance(out, dict) or not isinstance(out.get("comments"), list):
        return None
    verdicts: dict[int, dict] = {}
    for item in out["comments"]:
        if not isinstance(item, dict):
            continue
        i = item.get("index")
        if not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= n:
            continue
        if not isinstance(item.get("is_recommendation"), bool) or item.get("sentiment") not in SENTIMENTS:
            continue
        flags = item.get("quality_flags") if isinstance(item.get("quality_flags"), list) else []
        verdicts[i] = {
            "is_recommendation": item["is_recommendation"],
            "sentiment": item["sentiment"],
            "quality_flags": [f for f in dict.fromkeys(flags) if f in QUALITY_FLAGS],
            "reason": _clip(str(item.get("reason") or ""), REASON_CHARS),
        }
    return verdicts or None


def validate_b(out: Any) -> dict | None:
    """A normalised stage-B verdict, ``None`` if unusable (the keyword tags stay)."""
    if not isinstance(out, dict) or not isinstance(out.get("keep"), bool):
        return None
    tags: dict[str, Any] = {}
    for key, vocab in (("goal", GOALS), ("body", BODY), ("equipment", EQUIPMENT)):
        values = out.get(key)
        if not isinstance(values, list) or any(x not in vocab for x in values):
            return None
        tags[key] = list(dict.fromkeys(values))[: MAX_TAGS[key]]
    if out.get("level") not in LEVELS or out.get("type") not in TYPES:
        return None
    tags["level"], tags["type"] = out["level"], out["type"]
    if "none" in tags["equipment"] and len(tags["equipment"]) > 1:
        tags["equipment"] = [e for e in tags["equipment"] if e != "none"]
    reason = out.get("reject_reason") if out.get("reject_reason") in REJECT_REASONS else "none"
    if out["keep"]:
        reason = "none"
    elif reason == "none":
        reason = "non-fitness"
    summary = out.get("summary") if isinstance(out.get("summary"), str) else ""
    return {"keep": out["keep"], "reject_reason": reason, "tags": tags, "summary": _clip(summary, SUMMARY_CHARS)}


# -- cache ------------------------------------------------------------------------

def comment_key(c: dict) -> str:
    """Stable id of a saved excerpt: Reddit ``t1_``/``t3_`` ids, HN item ids, forum post ids."""
    link = c.get("permalink") or ""
    m = re.search(r"/comments/([a-z0-9]+)/[^/]*/([a-z0-9]+)/?$", link)
    if m:
        return f"t1_{m.group(2)}"
    m = re.search(r"reddit\.com/r/[^/]+/comments/([a-z0-9]+)/?$", link)
    if m:
        return f"t3_{m.group(1)}"
    m = re.search(r"ycombinator\.com/item\?id=(\d+)", link)
    if m:
        return f"hn_{m.group(1)}"
    m = re.search(r"t-nation\.com/t/(?:[^/]+/)?(\d+)/(\d+)", link)
    if m:
        return f"tn_{m.group(1)}_{m.group(2)}"
    basis = link or c.get("text") or ""
    return "x_" + hashlib.sha1(basis.encode()).hexdigest()[:12]


class LLMCache:
    """``pipeline/llm_cache.json``: verdicts, pending batches and a cost log."""

    def __init__(self, path: Path = config.LLM_CACHE_PATH):
        self.path = Path(path)
        self.data: dict[str, Any] = {"comments": {}, "videos": {}, "pending": [], "runs": []}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except ValueError:
                log.warning("unreadable %s; starting an empty LLM cache", self.path)

    def comment(self, vid: str, key: str) -> dict | None:
        return self.data["comments"].get(f"{PROMPT_A_VERSION}:{vid}:{key}")

    def set_comment(self, vid: str, key: str, verdict: dict) -> None:
        self.data["comments"][f"{PROMPT_A_VERSION}:{vid}:{key}"] = verdict

    def video(self, vid: str) -> dict | None:
        return self.data["videos"].get(f"{PROMPT_B_VERSION}:{vid}")

    def set_video(self, vid: str, verdict: dict) -> None:
        self.data["videos"][f"{PROMPT_B_VERSION}:{vid}"] = verdict

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.path)


# -- running requests -----------------------------------------------------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def tool_input(message: Any, tool_name: str) -> Any:
    for block in _get(message, "content") or []:
        if _get(block, "type") == "tool_use" and _get(block, "name") == tool_name:
            return _get(block, "input")
    return None


def price(model: str) -> tuple[float, float]:
    return config.LLM_PRICES.get(model, max(config.LLM_PRICES.values()))


def cost(model: str, input_tokens: float, output_tokens: float, batch: bool) -> float:
    pin, pout = price(model)
    return (input_tokens * pin + output_tokens * pout) / 1e6 * (config.LLM_BATCH_DISCOUNT if batch else 1.0)


# Rough expected output tokens per request (tool JSON), for the estimate only.
_OUT_TOKENS = {"A": lambda n: 40 + 45 * n, "B": lambda n: 160}


def estimate(stage: str, model: str, requests: dict[str, dict], meta: dict[str, Any], batch: bool) -> float:
    """Estimated USD for ``requests``: ~3.5 characters per input token, generous output."""
    chars = sum(len(json.dumps(p, ensure_ascii=False)) for p in requests.values())
    out = sum(_OUT_TOKENS[stage](len(meta[c]) if isinstance(meta.get(c), list) else 1) for c in requests)
    return cost(model, chars / 3.5, out, batch)


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    errors: int = 0
    invalid: int = 0

    def add(self, model: str, message: Any, batch: bool) -> None:
        u = _get(message, "usage")
        i, o = int(_get(u, "input_tokens", 0) or 0), int(_get(u, "output_tokens", 0) or 0)
        self.requests += 1
        self.input_tokens += i
        self.output_tokens += o
        self.cost += cost(model, i, o, batch)


Ingest = Callable[[str, Any, Any], bool]  # (custom_id, meta, message or None) -> valid?


class Runner:
    """Sends requests synchronously or as a Message Batch and feeds results to ``ingest``."""

    def __init__(self, client: Any, cache: LLMCache, *, deadline: float | None = None,
                 sync_max: int = config.LLM_SYNC_MAX, poll_seconds: float = config.LLM_POLL_SECONDS,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.client = client
        self.cache = cache
        self.deadline = deadline
        self.sync_max = sync_max
        self.poll_seconds = poll_seconds
        self.sleep = sleep
        self.clock = clock
        self.usage: dict[str, Usage] = {}

    def _usage(self, stage: str) -> Usage:
        return self.usage.setdefault(stage, Usage())

    def pending_ids(self, stage: str) -> set[str]:
        return {cid for p in self.cache.data["pending"] if p["stage"] == stage for cid in p["meta"]}

    def resume(self, stage: str, ingest: Ingest) -> None:
        """Collect batches submitted by earlier runs (if they have ended by the deadline)."""
        for p in [p for p in self.cache.data["pending"] if p["stage"] == stage]:
            log.info("LLM stage %s: resuming batch %s (%d requests, submitted %s)", stage, p["id"],
                     len(p["meta"]), p.get("submitted"))
            self._collect(p, ingest)

    def run(self, stage: str, model: str, requests: dict[str, dict], meta: dict[str, Any], ingest: Ingest,
            batch: bool) -> None:
        if not requests:
            return
        if not batch:
            self._run_sync(stage, model, requests, meta, ingest)
            return
        reqs = [{"custom_id": cid, "params": params} for cid, params in requests.items()]
        created = self.client.messages.batches.create(requests=reqs)
        entry = {"id": _get(created, "id"), "stage": stage, "model": model,
                 "submitted": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "meta": {cid: meta[cid] for cid in requests}}
        self.cache.data["pending"].append(entry)
        self.cache.save()  # a killed run can still resume this batch
        log.info("LLM stage %s: submitted batch %s with %d requests (%s)", stage, entry["id"], len(reqs), model)
        self._collect(entry, ingest)

    def _run_sync(self, stage: str, model: str, requests: dict[str, dict], meta: dict[str, Any],
                  ingest: Ingest) -> None:
        def call(item: tuple[str, dict]) -> tuple[str, Any]:
            cid, params = item
            try:
                return cid, self.client.messages.create(**params)
            except Exception as exc:  # noqa: BLE001 - one failing request falls back to keywords
                log.warning("LLM stage %s: request %s failed: %s", stage, cid, exc)
                return cid, None

        usage = self._usage(stage)
        with ThreadPoolExecutor(config.LLM_SYNC_WORKERS) as pool:
            for cid, message in pool.map(call, requests.items()):
                if message is None:
                    usage.errors += 1
                    continue
                usage.add(model, message, batch=False)
                if not ingest(cid, meta[cid], message):
                    usage.invalid += 1
        self.cache.save()

    def _collect(self, entry: dict, ingest: Ingest) -> bool:
        """Wait for ``entry``'s batch until the deadline; ingest and unlist it once ended."""
        bid, stage, model = entry["id"], entry["stage"], entry["model"]
        while True:
            batch = self.client.messages.batches.retrieve(bid)
            status = _get(batch, "processing_status")
            if status == "ended":
                break
            counts = _get(batch, "request_counts")
            left = None if self.deadline is None else self.deadline - self.clock()
            if left is not None and left < self.poll_seconds:
                log.warning("LLM stage %s: batch %s still %s (%s succeeded so far); leaving it pending "
                            "for the next run", stage, bid, status, _get(counts, "succeeded"))
                return False
            log.info("LLM stage %s: batch %s %s (%s processing, %s succeeded)", stage, bid, status,
                     _get(counts, "processing"), _get(counts, "succeeded"))
            self.sleep(self.poll_seconds)
        usage = self._usage(stage)
        for result in self.client.messages.batches.results(bid):
            cid = _get(result, "custom_id")
            res = _get(result, "result")
            if cid not in entry["meta"]:
                continue
            if _get(res, "type") != "succeeded":
                usage.errors += 1
                err = _get(res, "error")
                log.warning("LLM stage %s: %s %s %s", stage, cid, _get(res, "type"), _get(_get(err, "error"), "message", ""))
                continue
            message = _get(res, "message")
            usage.add(model, message, batch=True)
            if not ingest(cid, entry["meta"][cid], message):
                usage.invalid += 1
        self.cache.data["pending"] = [p for p in self.cache.data["pending"] if p["id"] != bid]
        self.cache.save()
        log.info("LLM stage %s: batch %s collected", stage, bid)
        return True


# -- applying verdicts -------------------------------------------------------------

def _n_wikis(v: dict) -> int:
    return len({(s.get("subreddit") or "").lower() for s in v.get("sources") or [] if s.get("kind") == "wiki"})


def _sum_score(v: dict) -> float:
    """Karma behind ``score`` (inverts ``rank_score`` after removing the wiki boost)."""
    rest = float(v.get("score") or 0) - sources_boost(v.get("sources")) - int(v.get("recommenders") or 0) * 10
    return max(0.0, math.expm1(max(0.0, rest / 5)))


def strip_llm(data: dict) -> dict:
    """Undo ``apply_verdicts``' count changes so carry-over sees the raw data; idempotent."""
    for v in data.get("videos", []):
        adj = v.pop("llm_rejected", None)
        if not adj:
            continue
        v["recommenders"] = int(v.get("recommenders") or 0) + int(adj.get("recommenders") or 0)
        v["score"] = round(float(v.get("score") or 0) + float(adj.get("score") or 0), 1)
        restored = list(adj.get("comments") or [])
        if restored:
            v["comments"] = sorted(list(v.get("comments") or []) + restored, key=lambda c: -(c.get("score") or 0))
    return data


def comment_verdict(cache: LLMCache, vid: str, c: dict) -> bool | None:
    """True = counts as a recommendation, False = rejected, None = not classified."""
    verdict = cache.comment(vid, comment_key(c))
    if not verdict or "is_recommendation" not in verdict:
        return None
    return bool(verdict["is_recommendation"]) and verdict.get("sentiment") != "negative"


def stage_a_drop(v: dict, cache: LLMCache) -> bool:
    """No wiki listing and every piece of community evidence we know of was rejected."""
    if _n_wikis(v):
        return False
    verdicts = [comment_verdict(cache, v["id"], c) for c in v.get("comments") or []]
    rejected = verdicts.count(False)
    if not rejected or True in verdicts or None in verdicts:
        return False
    community = int(v.get("recommenders") or 0)
    return community <= rejected


@dataclass
class Report:
    before: int = 0
    after: int = 0
    dropped: list[dict] = field(default_factory=list)
    adjusted: int = 0
    tagged: int = 0
    invalid_b: int = 0
    unverified: int = 0
    unverified_required: int = 0
    tags_before: dict = field(default_factory=dict)
    tags_after: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        reasons: dict[str, int] = {}
        for d in self.dropped:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        out = [f"videos: {self.before} -> {self.after} ({len(self.dropped)} dropped: "
               + (", ".join(f"{k} {n}" for k, n in sorted(reasons.items(), key=lambda kv: -kv[1])) or "none") + ")",
               f"recommender/score adjusted (rejected excerpts): {self.adjusted}",
               f"LLM-tagged: {self.tagged}; invalid stage-B output (keyword tags kept): {self.invalid_b}; "
               f"without a stage-B verdict: {self.unverified} (single-recommender / needs_llm_check: "
               f"{self.unverified_required})",
               f"tag coverage before: {self.tags_before}", f"tag coverage after:  {self.tags_after}"]
        for stage, u in sorted(self.usage.items()):
            out.append(f"stage {stage}: {u.requests} requests, {u.input_tokens} in / {u.output_tokens} out tokens, "
                       f"${u.cost:.2f}, {u.errors} errors, {u.invalid} invalid")
        out += [f"skipped: {s}" for s in self.skipped]
        out += [f"dropped {d['id']} [{d['reason']}] {d['title']}" for d in self.dropped]
        return out


def tag_coverage(videos: list[dict]) -> dict:
    n = len(videos) or 1
    pct = lambda k: round(100 * sum(1 for v in videos if (v.get("tags") or {}).get(k)) / n, 1)  # noqa: E731
    untagged = sum(1 for v in videos if not (v.get("tags") or {}).get("goal") and not (v.get("tags") or {}).get("body"))
    return {"goal%": pct("goal"), "body%": pct("body"), "equipment%": pct("equipment"),
            "no goal and no body": untagged}


def apply_verdicts(data: dict, cache: LLMCache, report: Report | None = None) -> dict:
    """Apply cached verdicts to ``data`` (call ``strip_llm`` first).

    * A video without a wiki listing is dropped when stage B rejects it, or
      when every saved excerpt was rejected by stage A and they account for
      all its community recommenders (e.g. a single-recommender video whose
      only comment is an anti-recommendation). Wiki-listed videos are
      moderator-curated and never dropped.
    * Rejected excerpts (not a recommendation, or negative) are hidden and no
      longer count: one recommender each (keeping at least one per accepted
      excerpt) and their karma; the change is stored in ``llm_rejected``.
    * Stage-B tags and ``summary`` replace the keyword tags; invalid output
      keeps the keyword tags. Accepted videos lose ``needs_llm_check``, as do
      external ``sources`` whose excerpt was accepted.
    * Videos without verdicts (no API key, budget cap, pending batch) are kept
      unchanged.
    """
    report = report or Report()
    report.before = len(data.get("videos", []))
    report.tags_before = tag_coverage(data.get("videos", []))
    kept = []
    for v in data.get("videos", []):
        vid = v["id"]
        wikis = _n_wikis(v)
        b = cache.video(vid)
        valid_b = b if b and "keep" in b else None
        if not wikis and valid_b and not valid_b["keep"]:
            report.dropped.append({"id": vid, "title": v.get("title", ""), "reason": valid_b["reject_reason"]})
            continue
        if stage_a_drop(v, cache):
            report.dropped.append({"id": vid, "title": v.get("title", ""), "reason": "no-recommendation"})
            continue

        comments = list(v.get("comments") or [])
        verdicts = [comment_verdict(cache, vid, c) for c in comments]
        rejected = [c for c, ok in zip(comments, verdicts) if ok is False]
        accepted = [c for c, ok in zip(comments, verdicts) if ok is True]
        if rejected:
            rec = int(v.get("recommenders") or 0)
            karma = sum(float(c.get("score") or 0) for c in rejected)
            new_rec = max(wikis + len(accepted), rec - len(rejected))
            new_score = round(rank_score(new_rec, max(0.0, _sum_score(v) - karma)) + sources_boost(v.get("sources")), 1)
            new_score = min(new_score, float(v.get("score") or 0))
            v["llm_rejected"] = {"recommenders": rec - new_rec, "score": round(float(v.get("score") or 0) - new_score, 1),
                                 "comments": rejected}
            v["recommenders"], v["score"] = new_rec, new_score
            v["comments"] = [c for c, ok in zip(comments, verdicts) if ok is not False]
            report.adjusted += 1

        if valid_b:
            v["tags"] = dict(valid_b["tags"])
            if valid_b.get("summary"):
                v["summary"] = valid_b["summary"]
            report.tagged += 1
        else:
            if b:
                report.invalid_b += 1
            report.unverified += 1
            if not wikis and (v.get("needs_llm_check") or int(v.get("recommenders") or 0) <= 1):
                report.unverified_required += 1

        if valid_b and (accepted or not comments):
            if "needs_llm_check" in v:
                v["needs_llm_check"] = False
        ok_links = {c.get("permalink") for c in accepted}
        for s in v.get("sources") or []:
            if s.get("needs_llm_check") and s.get("url") in ok_links:
                s["needs_llm_check"] = False
        kept.append(v)
    kept.sort(key=lambda r: (-r["score"], -r["recommenders"], r["id"]))
    data["videos"] = kept
    report.after = len(kept)
    report.tags_after = tag_coverage(kept)
    return data


# -- the pass ------------------------------------------------------------------------

def make_client() -> Any | None:
    """An Anthropic client, or ``None`` without ``ANTHROPIC_API_KEY`` / the SDK."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("ANTHROPIC_API_KEY is set but the anthropic package is missing; keyword tags only")
        return None
    return anthropic.Anthropic(max_retries=4)


@dataclass
class Options:
    model: str = config.LLM_MODEL
    classify_model: str = config.LLM_CLASSIFY_MODEL
    max_items: int = config.LLM_MAX_ITEMS
    max_cost: float = config.LLM_MAX_COST
    wait_minutes: float = config.LLM_WAIT_MINUTES
    force_sync: bool = False


def run_pass(data: dict, cache: LLMCache | None = None, client: Any = None, options: Options | None = None,
             runner: Runner | None = None) -> tuple[dict, Report]:
    """Classify/tag what is not cached yet (if ``client``), then ``apply_verdicts``."""
    opts = options or Options()
    cache = cache or LLMCache()
    report = Report()
    data = strip_llm(data)
    videos = data.get("videos", [])
    if client is None and runner is None:
        log.info("LLM pass: no API key; applying cached verdicts only")
    else:
        runner = runner or Runner(client, cache, deadline=time.monotonic() + opts.wait_minutes * 60)
        budget = opts.max_cost

        # Stage A: excerpts without a cached verdict, one request per video.
        def ingest_a(cid: str, keys: list[str], message: Any) -> bool:
            vid = cid[2:]
            verdicts = validate_a(tool_input(message, TOOL_A["name"]), len(keys))
            if verdicts is None:
                return False
            model = _get(message, "model") or opts.classify_model
            for i, key in enumerate(keys, 1):
                if i in verdicts:
                    cache.set_comment(vid, key, {**verdicts[i], "model": model})
            return len(verdicts) == len(keys)

        runner.resume("A", ingest_a)
        pending = runner.pending_ids("A")
        req_a, meta_a = {}, {}
        for v in videos:
            todo = [c for c in v.get("comments") or [] if cache.comment(v["id"], comment_key(c)) is None]
            cid = f"a-{v['id']}"
            if todo and cid not in pending and len(req_a) < opts.max_items:
                req_a[cid] = request_params("A", opts.classify_model, prompt_a(v, todo))
                meta_a[cid] = [comment_key(c) for c in todo]
        budget = _submit(runner, "A", opts.classify_model, req_a, meta_a, ingest_a, opts, budget, report)

        # Stage B: videos stage A did not already drop.
        def ingest_b(cid: str, vid: str, message: Any) -> bool:
            verdict = validate_b(tool_input(message, TOOL_B["name"]))
            model = _get(message, "model") or opts.model
            cache.set_video(vid, {**verdict, "model": model} if verdict else {"invalid": True, "model": model})
            return verdict is not None

        runner.resume("B", ingest_b)
        pending = runner.pending_ids("B") | {f"b-{cid[2:]}" for cid in runner.pending_ids("A")}
        req_b, meta_b = {}, {}
        for v in videos:
            cid = f"b-{v['id']}"
            if cache.video(v["id"]) is not None or cid in pending or stage_a_drop(v, cache):
                continue
            if len(req_b) >= opts.max_items:
                break
            ok = [c for c in v.get("comments") or [] if comment_verdict(cache, v["id"], c) is not False]
            req_b[cid] = request_params("B", opts.model, prompt_b(v, ok))
            meta_b[cid] = v["id"]
        _submit(runner, "B", opts.model, req_b, meta_b, ingest_b, opts, budget, report)
        report.usage = runner.usage
        _log_run(cache, runner, opts)
        cache.save()
    data = apply_verdicts(data, cache, report)
    return data, report


def _submit(runner: Runner, stage: str, model: str, requests: dict, meta: dict, ingest: Ingest, opts: Options,
            budget: float, report: Report) -> float:
    if not requests:
        log.info("LLM stage %s: nothing new to send", stage)
        return budget
    batch = not opts.force_sync and len(requests) > runner.sync_max
    est = estimate(stage, model, requests, meta, batch)
    log.info("LLM stage %s: %d requests to %s via %s, estimated cost $%.2f (budget left $%.2f)", stage,
             len(requests), model, "Message Batches" if batch else "Messages", est, budget)
    if est > budget:
        msg = f"stage {stage}: estimated ${est:.2f} exceeds the remaining budget ${budget:.2f}; not sent"
        log.error("LLM %s", msg)
        report.skipped.append(msg)
        return budget
    try:
        runner.run(stage, model, requests, meta, ingest, batch=batch)
    except Exception as exc:  # noqa: BLE001 - the pipeline must still produce data
        log.exception("LLM stage %s failed; continuing with cached verdicts", stage)
        report.skipped.append(f"stage {stage}: {type(exc).__name__}: {exc}")
    return budget - est


def _log_run(cache: LLMCache, runner: Runner, opts: Options) -> None:
    if not any(u.requests or u.errors for u in runner.usage.values()):
        return
    cache.data["runs"].append({
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {"A": opts.classify_model, "B": opts.model},
        **{f"stage_{s}": {"requests": u.requests, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                          "cost_usd": round(u.cost, 4), "errors": u.errors, "invalid": u.invalid}
           for s, u in sorted(runner.usage.items())},
    })
    cache.data["runs"] = cache.data["runs"][-50:]
