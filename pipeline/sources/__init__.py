"""Non-Reddit recommendation sources (Hacker News, T-Nation forum).

Each source module exposes ``collect(client=None, ...) -> list[Mention]``. A
``Mention`` is one post/comment linking one YouTube video; ``merge.py`` folds
mentions into the Reddit build. Sources share a small cached JSON client so
reruns are cheap and interrupted runs resume where they stopped.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import requests

from .. import config

log = logging.getLogger(__name__)


# Deliberately specific: generic words like "training", "running" or "form"
# match ML/ops talk on HN.
FITNESS_TERMS = [
    # "exercise your rights", "a stretch of road", "domain squatting", "rehab center"
    # are not about fitness.
    r"exercis(?:e|es|ed|ing)(?!\s+(?:your|their|his|her|my|our|its|this|that|the|caution|restraint"
    r"|judge?ment|discretion|control|options?|rights?|power|authority|f\W*\w*)\b)",
    r"stretch(?:es|ing)?(?!\s+(?:of|goals?|the truth)\b)",
    r"(?<![-/])(?<!domain )(?<!name )squat(?:s|ting)?",
    r"rehab(?:ilitation)?(?!\s+(?:centers?|centres?|facilit\w+|clinics?)\b)",
    r"workouts?", r"(?:hip|shoulder|ankle|thoracic|joint) mobility",
    r"mobility (?:work|drills?|routines?|exercises?|training)", r"deadlifts?",
    r"kettlebells?", r"barbells?", r"dumbbells?", r"physical therap\w*",
    r"physio(?:therap\w*)?", r"back pain", r"neck pain", r"knee pain", r"shoulder pain", r"sciatica", r"posture",
    r"zone ?2", r"vo2 ?max", r"running form", r"runner'?s knee", r"jogging", r"strength training", r"lifting",
    r"weight ?lifting", r"lift(?:ing)? weights", r"calisthenics", r"bodyweight", r"yoga", r"pilates",
    r"push-? ?ups?", r"pull-? ?ups?", r"chin-? ?ups?", r"planks?", r"lunges?",
    r"hamstrings?", r"glutes?", r"hip flexors?", r"core strength", r"foam roll\w*", r"hiit", r"cardio",
    r"aerobic", r"gym", r"fitness", r"muscles?(?! memory)", r"hypertrophy", r"powerlifting",
    r"starting strength", r"mark rippetoe", r"handstands?", r"tendons?", r"tendinitis", r"tendinopathy",
    r"injur(?:y|ies)",
]
FITNESS_RE = re.compile(r"\b(?:" + "|".join(FITNESS_TERMS) + r")\b", re.IGNORECASE)
# Video titles are short and on-topic, so broader words are safe there.
TITLE_FITNESS_RE = re.compile(r"\b(?:" + "|".join(FITNESS_TERMS + [
    r"flexibility", r"mobility", r"strength", r"bench press", r"running", r"stretch\w*", r"training",
    r"pain", r"neck", r"spine", r"knees?", r"shoulders?", r"hips?", r"drills?",
]) + r")\b", re.IGNORECASE)


@dataclass
class Mention:
    """One post on a non-Reddit site linking one YouTube video."""

    video_id: str
    kind: str  # "hn" | "forum"
    site: str  # "news.ycombinator.com" | "t-nation.com"
    author: str  # namespaced, e.g. "hn:pg"
    text: str  # plain-text post body
    url: str  # permalink to the post
    created_utc: int
    score: int | None = None  # likes/points when the site exposes them
    negative: bool = False
    n_videos: int = 1
    timestamps: list[tuple[int, str]] = field(default_factory=list)
    needs_llm_check: bool = True


class SourceError(RuntimeError):
    pass


class JsonClient:
    """GET JSON with a polite interval, retries and an on-disk cache.

    Cache entries are keyed by URL + params. ``ttl_days`` decides freshness
    per call; ``offline`` serves any cached entry and never hits the network.
    """

    def __init__(self, name: str, min_interval: float, cache_dir: Path | None = None,
                 session: requests.Session | None = None, offline: bool = False, max_retries: int = 4):
        self.name = name
        self.min_interval = min_interval
        self.offline = offline
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir or config.CACHE_DIR / name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.session.headers["Accept"] = "application/json"
        self._last = 0.0
        self.requests_made = 0
        self.cache_hits = 0

    def _path(self, url: str, params: dict | None) -> Path:
        key = json.dumps([url, params or {}], sort_keys=True)
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def get(self, url: str, params: dict | None = None, ttl_days: float | None = None) -> dict:
        path = self._path(url, params)
        if path.exists():
            fresh = ttl_days is None or path.stat().st_mtime >= time.time() - ttl_days * 86400
            if fresh or self.offline:
                try:
                    data = json.loads(path.read_text())["data"]
                    self.cache_hits += 1
                    return data
                except (ValueError, KeyError):
                    pass
        if self.offline:
            raise SourceError(f"{self.name}: not cached (offline): {url} {params or ''}")
        data = self._fetch(url, params)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"url": url, "params": params, "data": data}))
        tmp.replace(path)
        return data

    def _fetch(self, url: str, params: dict | None) -> dict:
        for attempt in range(1, self.max_retries + 1):
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self.session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                self._last = time.monotonic()
                delay = min(60, 2**attempt) + random.random()
                log.warning("%s: request error (%s), retry %d in %.0fs", self.name, exc, attempt, delay)
                time.sleep(delay)
                continue
            self._last = time.monotonic()
            self.requests_made += 1
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise SourceError(f"{self.name}: invalid JSON from {url}") from exc
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else min(300, 10 * 2 ** (attempt - 1))
                log.info("%s: HTTP %s, retry %d in %.0fs", self.name, resp.status_code, attempt, delay)
                time.sleep(delay + random.random())
                continue
            raise SourceError(f"{self.name}: HTTP {resp.status_code} for {url}")
        raise SourceError(f"{self.name}: giving up after {self.max_retries} attempts: {url}")


class _TextExtractor(HTMLParser):
    """HTML -> plain text, keeping link targets inline and dropping embeds/quotes."""

    _BLOCK = {"p", "br", "div", "li", "pre", "blockquote", "h1", "h2", "h3", "h4", "tr"}
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self._skip = 0  # depth inside a skipped subtree
        self._skip_keeps_links = False  # oneboxes: keep their link, quotes: drop
        self._href: str | None = None

    @staticmethod
    def _skippable(tag: str, cls: str) -> bool:
        # Discourse renders oneboxes as div.*onebox* / aside.onebox and quotes as aside.quote.
        return tag in ("script", "style", "aside", "blockquote") or (tag == "div" and "onebox" in cls)

    def _add_link(self, href: str | None) -> None:
        if href and (not self._skip or self._skip_keeps_links):
            self.links.append(href)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a":
            self._add_link(a.get("href"))
        if self._skip:
            if tag not in self._VOID:
                self._skip += 1
            return
        cls = a.get("class") or ""
        if self._skippable(tag, cls):
            self._skip = 1
            self._skip_keeps_links = "onebox" in cls and "quote" not in cls
            return
        if tag == "a":
            self._href = a.get("href") or None
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag == "a":
            self._add_link(dict(attrs).get("href"))
        elif not self._skip and tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self._skip:
            if tag not in self._VOID:
                self._skip -= 1
            return
        if tag == "a":
            if self._href:
                self.parts.append(f" {self._href} ")
            self._href = None
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip and not self._href:
            self.parts.append(data)


def html_to_text(markup: str) -> tuple[str, list[str]]:
    """Return ``(plain_text, hrefs)``.

    Anchors are replaced by their target URL so link positions are kept for
    proximity checks. Oneboxes (link previews) and quotes are dropped from the
    text; onebox links are still returned, quoted links are not (they are
    someone else's recommendation).
    """
    p = _TextExtractor()
    p.feed(markup or "")
    p.close()
    text = html.unescape("".join(p.parts))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return text, [html.unescape(h) for h in p.links if h]
