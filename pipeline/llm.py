"""LLM pass (issue #3): vet recommendation evidence, drop non-recommended
videos, tag and summarise the rest.

It runs on a built ``videos.json`` dict -- after carry-over and the HN /
T-Nation merge -- because that is the only place every video (including
carried-over ones) is available. Two stages, both through the Anthropic
Message Batches API (or plain requests for a handful of items), with
structured output via forced tool use:

* **Stage A -- evidence** (``LLM_CLASSIFY_MODEL``, default: the stage-B model): one request
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
from .build import is_content_wiki_source, rank_score, sources_boost
from .tagger import BODY, EQUIPMENT, GOALS, LEVELS, TYPES, tag_video

log = logging.getLogger(__name__)

# Bump when a prompt or schema changes in a way that invalidates cached answers.
PROMPT_A_VERSION = "a3"
PROMPT_B_VERSION = "b2"
# Earlier stage-B versions whose answers are still used: their keep verdict and
# tags stand, and only a video whose cached summary ``clean_summary`` cannot
# repair (it would be published without one) is asked again with the current
# prompt (``needs_reask``). Keeps a prompt fix from re-buying ~700 answers.
PROMPT_B_LEGACY = ("b1",)

SENTIMENTS = ["positive", "neutral", "negative"]
QUALITY_FLAGS = ["bro-science", "sales/affiliate", "unsafe-form", "self-promotion", "off-topic",
                 "personal-clip", "event-footage", "joke"]
REJECT_REASONS = ["none", "non-fitness", "personal-clip", "event-footage", "off-topic", "anti-recommendation", "spam"]
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
(Reddit, Hacker News, the T-Nation forum) recommend. Every numbered excerpt below comes from a \
comment or post that links the given video. The link itself is usually not visible: excerpts are \
cut to ~280 characters and links appear only as their anchor text. For each excerpt decide whether \
its author points people to the video as worth watching for training.

Default to is_recommendation=true. Linking a video while giving advice, answering a question, \
listing exercises or resources, or saying "here's a tutorial" counts -- even without praise, and \
even when the visible excerpt is general advice that never mentions the video. A video that \
teaches common mistakes ("how NOT to swing") is a recommendation too.

Set is_recommendation=false only when the excerpt itself shows that the author:
- presents the video as wrong, bad or dangerous, or quotes it in order to disagree (e.g. \
"Robert Oberst says you shouldn't do deadlifts" used as a bad example);
- links it as a joke, meme, music, film clip or something unrelated to training;
- shares their own or a friend's lift, attempt, PR, form check, physique or progress, or asks \
people to guess their body fat / rate their physique;
- shares competition, meet, race or festival footage or highlights for entertainment;
- promotes their own video or channel ("I made a video on this") or a product;
- is a bot (e.g. a "Video linked by /u/...: Title|Channel|Published" table).

sentiment is the author's attitude toward THIS video, not the tone of their advice: negative only \
when they criticise the video itself; matter-of-fact links are neutral. quality_flags lists \
concerns about the video or the link (empty if none). reason: at most 15 words. Return exactly one \
entry per [n].\
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
North Face Climb Festival");
- off-topic: sport topics that are not training, e.g. climbing knots, rappelling, belaying, rope \
and gear reviews or safety tests;
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

summary: one complete plain sentence (max 25 words, max {SUMMARY_CHARS} characters) describing \
the video itself: what it teaches, shows or offers, based only on the given information. No hype; \
do not invent details. Never mention where, how or by whom it was recommended or discussed: no \
subreddits (r/...), Reddit, wiki pages, forums, threads, communities, comments or commenters -- the \
site shows its sources already, and the comments are context for you, not content of the video. \
Never talk about the information you were given ("details are not provided", "based on the \
title") and do not guess ("likely shows ..."): if you cannot tell what the video covers, use an \
empty string. Plain text only: no quotes, braces, markup or JSON.\
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
        lines += ["", "Comments (context only; the summary must not mention them):"] + \
            [f"- {_clip(c.get('text', ''), COMMENT_CHARS)}" for c in comments[:4]]
    else:
        lines += ["", "No comment excerpts saved."]
    return "\n".join(lines)


# How structured output is requested. Models differ: some reject a forced
# tool_choice (e.g. when they always think), so ``Runner.probe`` tries these in
# order on one request and remembers what works per model in the cache.
# "tool": forced tool use; "tool-nothink": the same with thinking disabled;
# "json": JSON-schema structured output; "auto-tool": the tool offered with
# tool_choice auto and an instruction to call it.
STYLES = ["tool", "tool-nothink", "json", "auto-tool"]
THINKING_STYLES = {"json", "auto-tool"}  # may think first: more max_tokens, costlier estimate


def request_params(stage: str, model: str, user: str, style: str = "tool") -> dict:
    tool, system, max_tokens = (TOOL_A, SYSTEM_A, 1500) if stage == "A" else (TOOL_B, SYSTEM_B, 800)
    params: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "system": system,
                              "messages": [{"role": "user", "content": user}]}
    if style in ("tool", "tool-nothink"):
        params["tools"] = [tool]
        params["tool_choice"] = {"type": "tool", "name": tool["name"]}
        if style == "tool-nothink":
            params["thinking"] = {"type": "disabled"}
    elif style == "json":
        params["max_tokens"] = 8000
        params["system"] = system + "\nRespond with the JSON object only."
        params["output_config"] = {"format": {"type": "json_schema", "schema": tool["input_schema"]}}
    elif style == "auto-tool":
        params["max_tokens"] = 8000
        params["system"] = system + f"\nAlways answer by calling the {tool['name']} tool."
        params["tools"] = [tool]
        params["tool_choice"] = {"type": "auto"}
    else:
        raise ValueError(f"unknown output style {style!r}")
    return params


# -- validation -----------------------------------------------------------------

# Structured-output debris some answers leave in the summary string: a stray
# ``}`` / ``"}`` closing the JSON, ``</parameter>``, ``<br>```json``, followed
# by more junk ("Wait."). Everything from the first such character on is cut.
_SUMMARY_JUNK_RE = re.compile(r"[{}<>]|```")
# Markup or structured-output characters a published summary must not contain.
SUMMARY_MARKUP_RE = re.compile(r"[{}<>\[\]\\`|*_#]|```|https?://|&\w+;")
# Provenance: where or by whom the video was recommended. The site shows its
# sources already, so a summary describes the video only.
SUMMARY_PROVENANCE_RE = re.compile(
    r"\br/\w+|\bsubreddits?\b|\breddit(?:ors?)?\b|\bwikis?\b|\bFAQ\b|\bforums?\b|\bthreads?\b|"
    r"\bcomment(?:er)?s?\b|\bposters?\b|"
    r"\bcommunit(?:y|ies)\s+(?:members?|discussions?|recommend\w*|cite\w*|says?|suggest\w*|consider\w*|use\w*)\b|"
    r"\b(?:by|in|among|from|per|across)\s+(?:the\s+|several\s+|various\s+|many\s+)?(?:[\w'-]+\s+){0,3}"
    r"communit(?:y|ies)\b|"
    r"\b(?:in|from)\s+(?:a|the)\s+(?:[\w'-]+\s+){0,2}discussions?\b|"
    r"\b(?:recommended|listed|cited|featured|mentioned|linked|shared|praised|suggested|referenced|endorsed|"
    r"credited|described|discussed|noted|used)\s+(?:on|in|by|among|under)\s+(?:the\s+|a\s+|several\s+|many\s+)?"
    r"(?:[\w'-]+\s+){0,3}(?:lifters|users|members|people|runners|climbers|readers|viewers|redditors|"
    r"physical therapists|coaches)\b|"
    r"\brecommended\s+(?:on|in|by|among)\b|"
    r"\b(?:listed|cited|featured|mentioned|linked|shared|referenced)\s+(?:on|in)\s+(?:the|a|its)\b",
    re.IGNORECASE)
# Meta talk about the information the model had ("details beyond the title are
# not given") or guesses about what the video might show.
SUMMARY_META_RE = re.compile(
    r"\b(?:not|isn't|aren't|wasn't|never)\s+(?:(?:explicitly|clearly|further)\s+)?(?:stated|given|provided|described|"
    r"specified|shown|clear|known|indicated|detailed|named|identified|available)\b|"
    r"\bbeyond the title\b|\b(?:based on|from|going by|judging by|per)\s+(?:the|its)\s+title\b|\bthe title (?:suggests|"
    r"implies|indicates)\b|\b(?:available|given|provided|limited|saved)\s+(?:information|info|details|description|"
    r"context|excerpts?)\b|\binformation (?:given|provided|available)\b|\bunspecified\b|\bunclear\b|"
    r"\bno (?:further |other |additional |specific )?(?:details|information|description|instruction)\b|"
    r"\bnothing (?:in|shows|indicates|suggests)\b|\bappears to (?:be|show)\b|"
    r"[,;(]\s*(?:likely|presumably|probably|possibly|apparently)\b|\blikely (?:a|an|the|showing|covering|about|"
    r"demonstrating|for)\b|"
    # Summaries that only restate the title and channel ("Video titled \"Dog of
    # Wisdom\" from channel Joe.").
    r"\b(?:video|clip)\s+(?:titled|called|named)\b|\bfrom\s+(?:the\s+)?channel\b", re.IGNORECASE)
# Reception rather than content: how the video is regarded ("recommended as a
# brief, to-the-point technique guide", "as recommended for showing ...",
# "recommended as a film rather than ..."). Audience advice stays ("recommended
# for people with limited equipment", "recommended as a warm-up for ...").
SUMMARY_RECEPTION_RE = re.compile(
    r"\bas\s+recommended\s+(?:as|for)\b|"
    r"\brecommended\s+as\s+(?:[\w'-]+,?\s+){0,4}?(?:primary|definitive|best|go-to|classic|must-watch|brief|concise|"
    r"to-the-point|film|documentary|reference|resource|source)\b|"
    r"\brecommended\s+as\s+[^.;]*?\brather\s+than\b|"
    r"\brecommended\s+for\s+(?:its|their|showing|how\s+(?:it|well))\b", re.IGNORECASE)
# A personal recording rather than something that teaches (stage B sometimes
# keeps one and says so in the summary).
SUMMARY_PERSONAL_RE = re.compile(
    r"\bpersonal (?:recording|clip|video|footage)\b|\b(?:their|his|her|a lifter's|someone's) own (?:lift|attempt|set|"
    r"pr|clip|recording)\b|\bnothing shows it teaches\b|\brather than instruction\b", re.IGNORECASE)
_ENDS_OK_RE = re.compile(r"[\w)\"'”%][.!?]$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<![A-Z][a-z]\.)(?<!\b(?:vs|St|Mr|Ms)\.)(?<!\bProf\.)(?<=[.!?])\s+(?=[A-Z])")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[,;:]|\s[-–—])\s+")
# Where a clause can be cut before the provenance part ("a hip stretch that
# r/flexibility commenters recommend").
_CLAUSE_CUT_RE = re.compile(r"\s+(?:that|which|who|whom|where|with|per|as|and|or|after|while|when|alongside|"
                            r"recommended|listed|cited|featured|mentioned|praised|suggested|referenced|endorsed|"
                            r"described|discussed|credited|used|framed|shared|linked)\b", re.IGNORECASE)
# Inline provenance in the main clause ("A Turkish get-up tutorial recommended
# on r/kettlebell for learning ..."): the participle phrase goes.
_INLINE_PROVENANCE_RE = re.compile(
    r"(?:\s+(?:as|and|and\s+also|being|(?:(?:and|that|which)\s+)?(?:is|are|was|were)(?:\s+also)?))?"
    r"\s+(?:(?:often|widely|frequently|commonly|also|highly)\s+)?"
    r"(?:recommended|listed|cited|featured|suggested|referenced|endorsed|praised)\s+(?:on|in|among|under)\s+"
    r"(?:the\s+|a\s+|its\s+)?(?:(?:r/\w+|[\w'-]+)(?:'s)?\s+){0,4}?(?:r/\w+(?:\s+and\s+r/\w+)*|communit(?:y|ies))"
    r"(?:(?:'s)?\s+(?:[\w-]+\s+){0,3}?(?:wiki|FAQ|page|section|routine)"
    r"(?:\s+(?:[\w-]+\s+){0,3}?(?:page|section))?)?"
    r"(?:\s+and\s+(?:listed|cited|featured|mentioned)\s+(?:on|in)\s+(?:the\s+|its\s+)?(?:[\w/'-]+\s+){0,3}?"
    r"(?:wiki|FAQ)(?:\s+page)?)?"
    # ... and only where the sentence goes on grammatically without it.
    r"(?=\s*(?:[,;.]|$)|\s+for\b|\s+\w+ing\b)", re.IGNORECASE)
# A clause head must not end mid-phrase ("a shoulder exercise used").
_DANGLING_END_RE = re.compile(r"\b(?:used|was|is|are|were|been|being|and|or|as|the|a|an|its|their|his|her|by|in|on|"
                              r"of|to|from|for|with)$", re.IGNORECASE)
# Proper names that contain a subreddit ("the r/bodyweightfitness Recommended Routine").
_SUBREDDIT_NAMES = {"bodyweightfitness": "Bodyweight Fitness", "flexibility": "Flexibility", "Fitness": "Fitness"}
_NAMED_ROUTINE_RE = re.compile(r"\br/(\w+)(?=\s+(?:Recommended Routine|Recommended Routines|RR)\b)")
_MIN_SUMMARY_WORDS = 3


def summary_problems(text: Any) -> list[str]:
    """What makes ``text`` unfit to publish as a summary (``[]``: fine).

    Shared with ``check_data``: markup, provenance, meta talk, a personal-clip
    description, truncation, a missing or dangling ending, length.
    """
    if not isinstance(text, str) or not text.strip():
        return ["empty"]
    out = []
    if SUMMARY_MARKUP_RE.search(text):
        out.append("markup")
    if SUMMARY_PROVENANCE_RE.search(text):
        out.append("provenance")
    if SUMMARY_META_RE.search(text):
        out.append("meta")
    if SUMMARY_RECEPTION_RE.search(text):
        out.append("reception")
    if SUMMARY_PERSONAL_RE.search(text):
        out.append("personal-clip")
    if text.endswith(("…", "...")) or re.search(r"\s[^\w\s]+$", text) or not _ENDS_OK_RE.search(text):
        out.append("truncated")
    if len(text) > SUMMARY_CHARS:
        out.append("too-long")
    if len(text.split()) < _MIN_SUMMARY_WORDS:
        out.append("too-short")
    if text != text.strip() or "  " in text or "\n" in text:
        out.append("whitespace")
    return out


def _bad_part(text: str) -> bool:
    return bool(SUMMARY_PROVENANCE_RE.search(text) or SUMMARY_META_RE.search(text)
                or SUMMARY_RECEPTION_RE.search(text))


def _cut_clause(clause: str) -> str:
    """The longest clean head of ``clause`` before a cut point ('' if none).

    A ``with ...`` clause goes whole: what it lists usually comes from the
    comments ("with bridge work mentioned by commenters"), not the video.
    """
    if re.match(r"(?:with|including|and|plus)\b", clause, re.IGNORECASE):
        return ""
    for m in reversed(list(_CLAUSE_CUT_RE.finditer(clause))):
        head = clause[: m.start()].rstrip()
        if len(head.split()) >= 3 and not _bad_part(head) and not _CLAUSE_CUT_RE.match(" " + head) \
                and not _DANGLING_END_RE.search(head):
            return head
    return ""


def _strip_provenance(sentence: str) -> str:
    """``sentence`` up to its first provenance/meta clause ('' if the main clause is affected)."""
    # Only inside a clause: a clause that starts with the participle is dropped below.
    sentence = _INLINE_PROVENANCE_RE.sub(
        lambda m: m.group(0) if m.start() == 0 or sentence[m.start() - 1] in ",;:\u2013\u2014-" else "", sentence)
    if not _bad_part(sentence):
        return sentence
    parts = _CLAUSE_SPLIT_RE.split(sentence)
    seps = _CLAUSE_SPLIT_RE.findall(sentence)
    kept = ""
    for i, part in enumerate(parts):
        if _bad_part(part):
            head = _cut_clause(part)
            if i == 0:
                return head if len(head.split()) >= 4 else ""
            if head:
                return kept + seps[i - 1] + head
            # A short appositive left over ("toe walks, an exercise") goes too.
            if i >= 2 and len(parts[i - 1].split()) < 3 and re.match(r"(?:an?|the)\s", parts[i - 1], re.IGNORECASE):
                kept = kept[: len(kept) - len(parts[i - 1]) - len(seps[i - 2])]
            return kept
        kept += (seps[i - 1] if i else "") + part
    return kept


def _fit_length(text: str) -> str:
    """Shorten ``text`` to ``SUMMARY_CHARS`` at a sentence or clause boundary, never mid-word."""
    while len(text) > SUMMARY_CHARS:
        cuts = [m.start() for m in _CLAUSE_SPLIT_RE.finditer(text)] + \
               [m.start() for m in _SENTENCE_SPLIT_RE.finditer(text)]
        cuts = [c for c in cuts if c < len(text)]
        if not cuts:
            return ""
        text = text[: max(cuts)].rstrip(" ,;:-–—")
    return text


def clean_summary(text: Any) -> str:
    """A displayable summary, ``""`` when nothing usable is left (the video is
    then shown without one, and stage B is asked again).

    Removes structured-output debris (a stray ``}`` / ``</parameter>`` and
    anything after it), provenance (where or by whom the video was
    recommended: the site shows its sources), meta talk about missing
    information and guesses, a truncated last clause and dangling punctuation.
    A summary whose main clause is itself provenance or meta talk is dropped.
    Long summaries are shortened at a clause boundary, never mid-word.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    m = _SUMMARY_JUNK_RE.search(text)
    if m:
        text = text[: m.start()]
    text = text.rstrip(" \t,;:")
    # Quotes left unbalanced by the cut (``...in a car.”}``).
    while text and text[-1] in "\"”'" and (text.count("\"") % 2 or text.count("”") > text.count("“")
                                                 or (text[-1] == "'" and text.count("'") % 2)):
        text = text[:-1].rstrip()
    # Dangling punctuation after the sentence ("... in comments. ?").
    text = re.sub(r"(?:\s+[^\w\s\"'”)]+)+$", "", text).rstrip(" ,;:-–—")
    # A truncated last clause ("... specified in the given inf…"): drop it.
    if text.endswith(("…", "...")):
        parts = _CLAUSE_SPLIT_RE.split(text.rstrip(".…"))
        if len(parts) < 2:
            return ""
        text = text[: text.rstrip(".…").rfind(parts[-1])].rstrip(" ,;:-–—")
    text = _NAMED_ROUTINE_RE.sub(lambda m: _SUBREDDIT_NAMES.get(m.group(1), m.group(1)), text)
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept = []
    for i, sentence in enumerate(sentences):
        clean = _strip_provenance(sentence.rstrip(".!?").rstrip())
        if not clean:
            break
        end = sentence[-1] if sentence[-1:] in ".!?" else "."
        kept.append(clean.rstrip(" ,;:-–—") + end)
        if clean != sentence.rstrip(".!?").rstrip():
            break  # what follows a provenance clause elaborates on it
    text = " ".join(kept)
    if text and text[-1] not in ".!?":
        text += "."
    text = _fit_length(text)
    if text and text[-1] not in ".!?":
        text += "."
    if text[:1].islower():
        text = text[0].upper() + text[1:]
    return "" if summary_problems(text) else text


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
    return {"keep": out["keep"], "reject_reason": reason, "tags": tags, "summary": clean_summary(out.get("summary"))}


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
        self.data: dict[str, Any] = {"comments": {}, "videos": {}, "pending": [], "runs": [], "styles": {}}
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
        """The current-version verdict, else the newest legacy one (``PROMPT_B_LEGACY``)."""
        for version in (PROMPT_B_VERSION, *PROMPT_B_LEGACY):
            verdict = self.data["videos"].get(f"{version}:{vid}")
            if verdict is not None:
                return verdict
        return None

    def video_is_current(self, vid: str) -> bool:
        return f"{PROMPT_B_VERSION}:{vid}" in self.data["videos"]

    def set_video(self, vid: str, verdict: dict) -> None:
        self.data["videos"][f"{PROMPT_B_VERSION}:{vid}"] = verdict
        for version in PROMPT_B_LEGACY:
            self.data["videos"].pop(f"{version}:{vid}", None)

    def save(self) -> None:
        """Write the cache, dropping answers to prompt versions no longer used."""
        for section, versions in (("comments", (PROMPT_A_VERSION,)), ("videos", (PROMPT_B_VERSION, *PROMPT_B_LEGACY))):
            self.data[section] = {k: x for k, x in self.data[section].items() if k.split(":", 1)[0] in versions}
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
    """The structured answer: the tool call's input, else JSON in the text blocks."""
    texts = []
    for block in _get(message, "content") or []:
        kind = _get(block, "type")
        if kind == "tool_use" and _get(block, "name") == tool_name:
            return _get(block, "input")
        if kind == "text":
            texts.append(_get(block, "text") or "")
    text = "".join(texts).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    if text.startswith("{"):
        try:
            return json.loads(text)
        except ValueError:
            return None
    return None


def price(model: str) -> tuple[float, float]:
    return config.LLM_PRICES.get(model, max(config.LLM_PRICES.values()))


def cost(model: str, input_tokens: float, output_tokens: float, batch: bool) -> float:
    pin, pout = price(model)
    return (input_tokens * pin + output_tokens * pout) / 1e6 * (config.LLM_BATCH_DISCOUNT if batch else 1.0)


# Rough expected output tokens per request (tool JSON), for the estimate only.
_OUT_TOKENS = {"A": lambda n: 40 + 45 * n, "B": lambda n: 160}


def estimate(stage: str, model: str, requests: dict[str, dict], meta: dict[str, Any], batch: bool,
             style: str = "tool") -> float:
    """Estimated USD for ``requests``: ~3.5 characters per input token, generous output
    (plus ~300 thinking tokens per request for styles that let the model think)."""
    chars = sum(len(json.dumps(p, ensure_ascii=False)) for p in requests.values())
    out = sum(_OUT_TOKENS[stage](len(meta[c]) if isinstance(meta.get(c), list) else 1) for c in requests)
    if style in THINKING_STYLES:
        out += 300 * len(requests)
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

    def probe(self, stage: str, model: str, cid: str, user: str, meta: Any, ingest: Ingest) -> str | None:
        """Send one request, trying ``STYLES`` (the one remembered for ``model``
        first) until the API accepts one; returns that style, ``None`` if none works.

        Only a 400 (invalid request) moves on to the next style; other errors
        propagate. The answer is ingested like any other.
        """
        known = self.cache.data.setdefault("styles", {}).get(model)
        usage = self._usage(stage)
        for style in ([known] if known in STYLES else []) + [s for s in STYLES if s != known]:
            try:
                message = self.client.messages.create(**request_params(stage, model, user, style))
            except Exception as exc:  # noqa: BLE001
                if getattr(exc, "status_code", None) != 400:
                    raise
                log.warning("LLM stage %s: %s rejects output style %r: %s", stage, model, style, exc)
                continue
            usage.add(model, message, batch=False)
            if not ingest(cid, meta, message):
                usage.invalid += 1
            if known != style:
                log.info("LLM stage %s: using output style %r for %s", stage, style, model)
            self.cache.data["styles"][model] = style
            self.cache.save()
            return style
        return None

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
    """Subreddit wikis listing ``v`` on a content page (moderation/rules pages do not count)."""
    return len({(s.get("subreddit") or "").lower() for s in v.get("sources") or []
                if s.get("kind") == "wiki" and is_content_wiki_source(s)})


def _sum_score(v: dict) -> float:
    """Karma behind ``score`` (inverts ``rank_score`` after removing the wiki boost)."""
    rest = float(v.get("score") or 0) - sources_boost(v.get("sources")) - int(v.get("recommenders") or 0) * 10
    return max(0.0, math.expm1(max(0.0, rest / 5)))


def strip_llm(data: dict) -> dict:
    """Undo ``apply_verdicts``' count changes so carry-over sees the raw data; idempotent."""
    for v in data.get("videos", []):
        v.pop("llm_note", None)
        adj = v.pop("llm_rejected", None)
        if not adj:
            continue
        v["recommenders"] = int(v.get("recommenders") or 0) + int(adj.get("recommenders") or 0)
        v["score"] = round(float(v.get("score") or 0) + float(adj.get("score") or 0), 1)
        restored = list(adj.get("comments") or [])
        if restored:
            v["comments"] = sorted(list(v.get("comments") or []) + restored, key=lambda c: -(c.get("score") or 0))
        if adj.get("sources"):
            v["sources"] = list(v.get("sources") or []) + list(adj["sources"])
    return data


def comment_verdict(cache: LLMCache, vid: str, c: dict) -> bool | None:
    """True = counts as a recommendation, False = rejected, None = not classified."""
    verdict = cache.comment(vid, comment_key(c))
    if not verdict or "is_recommendation" not in verdict:
        return None
    return bool(verdict["is_recommendation"]) and verdict.get("sentiment") != "negative"


QUESTIONABLE_FLAGS = {"bro-science", "unsafe-form"}


def stage_a_drop(v: dict, cache: LLMCache) -> str | None:
    """Why stage A's verdicts drop ``v`` (``None``: keep). Wiki-listed videos are kept.

    * ``no-recommendation``: every saved excerpt was rejected and they account
      for all community recommenders.
    * ``anti-recommendation``: contested -- every accepted excerpt flags the
      video as bro-science/unsafe and another excerpt criticises it (e.g. a
      clip linked once as advice and once to rebut it).
    """
    if _n_wikis(v):
        return None
    vid = v["id"]
    comments = v.get("comments") or []
    verdicts = [comment_verdict(cache, vid, c) for c in comments]
    rejected = verdicts.count(False)
    if rejected and True not in verdicts and None not in verdicts and int(v.get("recommenders") or 0) <= rejected:
        return "no-recommendation"
    raw = [cache.comment(vid, comment_key(c)) or {} for c in comments]
    accepted = [r for r, ok in zip(raw, verdicts) if ok]
    if accepted and all(QUESTIONABLE_FLAGS & set(r.get("quality_flags") or []) for r in accepted) \
            and any(r.get("sentiment") == "negative" for r in raw):
        return "anti-recommendation"
    return None


# Rejection grounds that are about the video, not about a misattributed or
# off-topic excerpt: such videos are never kept by content alone.
HARD_REJECT_FLAGS = {"personal-clip", "event-footage", "joke", "self-promotion", "sales/affiliate", "bro-science",
                     "unsafe-form"}
RESCUE_TYPES = {"tutorial", "explainer", "follow-along", "program"}
# Training words in a title or channel name ("PowerMonkeyFitness"); the
# keyword tagger's title rules count too.
_FITNESS_WORDS_RE = re.compile(
    r"fitness|workout|training|strength|calisthenic|kettlebell|physio|athlet|mobility|stretch|yoga|pilates|"
    r"\b(?:fit|gym|coach\w*|exercises?|rehab\w*|barbell|dumbbells?|lifting|squats?|pistols?|lunges?|deadlifts?|"
    r"pull-?ups?|push-?ups?|planks?|handstands?)\b", re.IGNORECASE)


def rescuable(v: dict, cache: LLMCache) -> bool:
    """Whether stage B reviews ``v`` although stage A found no recommendation.

    Saved excerpts are sometimes misattributed (a thread's off-topic replies
    filed under a video linked elsewhere in it), so a ``no-recommendation``
    drop is reconsidered when no rejected excerpt points at a problem with the
    video itself: negative, own clip, event footage, joke, promotion,
    bro-science (``HARD_REJECT_FLAGS``).
    """
    if stage_a_drop(v, cache) != "no-recommendation":
        return False
    vid = v["id"]
    for c in v.get("comments") or []:
        verdict = cache.comment(vid, comment_key(c)) or {}
        if verdict.get("sentiment") == "negative" or HARD_REJECT_FLAGS & set(verdict.get("quality_flags") or []):
            return False
    return True


def needs_reask(cache: LLMCache, vid: str) -> bool:
    """A legacy stage-B answer that keeps the video but whose summary
    ``clean_summary`` cannot repair: ask again with the current prompt."""
    verdict = cache.video(vid)
    return bool(verdict and verdict.get("keep") is True and not cache.video_is_current(vid)
                and not clean_summary(verdict.get("summary")))


# Stage B sometimes files climbing knots, rappelling or gear under
# ``event-footage`` (prompt b1 told it to): such answers count as off-topic.
_OFF_TOPIC_RE = re.compile(r"\b(?:knots?|rappel\w*|abseil\w*|belay\w*|ropes?|rigging|harness\w*|carabiners?|gear|"
                           r"crampons?|anchors?)\b", re.IGNORECASE)
# Grounds on which stage B drops even a wiki-listed video: wiki demo pages
# also link people's own lift clips ("Movie on 22 12 13 at 5 28 PM"), FAQs
# link meet footage as examples ("Donny Shankle Nor Cal 2011 CJ3") and some
# pages link memes. A wiki listing only overrides the softer grounds
# (anti-recommendation, spam: judged from excerpts, not the video).
WIKI_DROP_REASONS = {"personal-clip", "non-fitness", "event-footage", "off-topic"}


def stage_b_drop(v: dict, verdict: dict | None) -> str | None:
    """Why stage B's ``verdict`` drops ``v`` (``None``: keep).

    Also a kept video whose summary describes a personal recording (stage B
    kept it but said "nothing shows it teaches anything") is a personal clip.
    """
    if not verdict:
        return None
    if not verdict.get("keep"):
        reason = verdict.get("reject_reason") or "non-fitness"
        if reason == "event-footage" and _OFF_TOPIC_RE.search(f"{v.get('title') or ''} {verdict.get('summary') or ''}"):
            reason = "off-topic"
        return reason
    if SUMMARY_PERSONAL_RE.search(verdict.get("summary") or ""):
        return "personal-clip"
    return None


def fitness_channels(videos: list[dict], cache: LLMCache) -> set[str]:
    """Channels with a video that is wiki-listed or has an accepted excerpt."""
    out = set()
    for v in videos:
        if v.get("channel") and (_n_wikis(v) or any(comment_verdict(cache, v["id"], c) for c in v.get("comments") or [])):
            out.add(v["channel"])
    return out


def kept_by_content(v: dict, cache: LLMCache, verdict: dict | None, channels: set[str] = frozenset()) -> bool:
    """A ``rescuable`` video stage B keeps as a tutorial/explainer/follow-along/program,
    whose title or channel name is about training or whose channel is in ``channels``."""
    if not (verdict and verdict.get("keep") and (verdict.get("tags") or {}).get("type") in RESCUE_TYPES):
        return False
    if not rescuable(v, cache):
        return False
    text = f"{v.get('title') or ''} {v.get('channel') or ''}"
    tags = tag_video(text)
    return bool(v.get("channel") in channels or tags.get("goal") or tags.get("body") or _FITNESS_WORDS_RE.search(text))


@dataclass
class Report:
    before: int = 0
    after: int = 0
    dropped: list[dict] = field(default_factory=list)
    rescued: list[dict] = field(default_factory=list)
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
               f"kept by content (stage A found no recommendation, stage B keeps it): {len(self.rescued)}",
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
        out += [f"kept by content {d['id']} {d['title']}" for d in self.rescued]
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
      when ``stage_a_drop`` says so: every saved excerpt was rejected and they
      account for all its community recommenders (e.g. a single-recommender
      video whose only comment is an anti-recommendation), or the video is
      contested (only bro-science-flagged support, plus criticism).
      Wiki-listed videos are moderator-curated and only dropped as a
      personal clip, non-fitness, event footage or off-topic
      (``stage_b_drop``, ``WIKI_DROP_REASONS``).
    * Exception: a ``no-recommendation`` video that is ``rescuable`` (only
      off-topic/misattributed excerpts) and that stage B keeps as a
      tutorial/explainer/follow-along/program stays when its title or channel
      is about training (``kept_by_content``), with ``llm_note:
      "kept-by-content"`` and at least one recommender.
    * Rejected excerpts (not a recommendation, or negative) are hidden and no
      longer count: one recommender each (keeping at least one per accepted
      excerpt) and their karma; ``sources`` entries pointing at them (forum /
      HN posts) are removed too. The change is stored in ``llm_rejected``.
    * Stage-B tags and ``summary`` (cleaned by ``clean_summary``, also for
      cached answers; omitted when nothing publishable is left) replace the
      keyword tags; invalid output keeps the
      keyword tags. Accepted videos lose ``needs_llm_check``, as do
      external ``sources`` whose excerpt was accepted.
    * Videos without verdicts (no API key, budget cap, pending batch) are kept
      unchanged.
    """
    report = report or Report()
    report.before = len(data.get("videos", []))
    report.tags_before = tag_coverage(data.get("videos", []))
    kept = []
    channels = fitness_channels(data.get("videos", []), cache)
    for v in data.get("videos", []):
        vid = v["id"]
        wikis = _n_wikis(v)
        b = cache.video(vid)
        valid_b = b if b and "keep" in b else None
        b_reason = stage_b_drop(v, valid_b)
        if b_reason and (not wikis or b_reason in WIKI_DROP_REASONS):
            report.dropped.append({"id": vid, "title": v.get("title", ""), "reason": b_reason})
            continue
        reason = stage_a_drop(v, cache)
        rescued = reason == "no-recommendation" and not wikis and kept_by_content(v, cache, valid_b, channels)
        if reason and not rescued:
            report.dropped.append({"id": vid, "title": v.get("title", ""), "reason": reason})
            continue

        comments = list(v.get("comments") or [])
        verdicts = [comment_verdict(cache, vid, c) for c in comments]
        rejected = [c for c, ok in zip(comments, verdicts) if ok is False]
        accepted = [c for c, ok in zip(comments, verdicts) if ok is True]
        if rejected:
            rec = int(v.get("recommenders") or 0)
            karma = sum(float(c.get("score") or 0) for c in rejected)
            new_rec = max(wikis + len(accepted), rec - len(rejected), 1 if rescued else 0)
            new_score = round(rank_score(new_rec, max(0.0, _sum_score(v) - karma)) + sources_boost(v.get("sources")), 1)
            new_score = min(new_score, float(v.get("score") or 0))
            rejected_links = {c.get("permalink") for c in rejected if c.get("permalink")}
            sources = list(v.get("sources") or [])
            gone = [s for s in sources if s.get("kind") not in ("wiki", "comment") and s.get("url") in rejected_links]
            v["llm_rejected"] = {"recommenders": rec - new_rec, "score": round(float(v.get("score") or 0) - new_score, 1),
                                 "comments": rejected}
            if gone:
                v["llm_rejected"]["sources"] = gone
                v["sources"] = [s for s in sources if s not in gone]
            v["recommenders"], v["score"] = new_rec, new_score
            v["comments"] = [c for c, ok in zip(comments, verdicts) if ok is not False]
            report.adjusted += 1

        if rescued:
            v["llm_note"] = "kept-by-content"
            report.rescued.append({"id": vid, "title": v.get("title", "")})
        if valid_b:
            v["tags"] = dict(valid_b["tags"])
            summary = clean_summary(valid_b.get("summary"))
            if summary:
                v["summary"] = summary
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


# -- publishing ----------------------------------------------------------------------

# Pipeline-internal fields: ``strip_llm`` needs them while the data is
# processed, the published ``videos.json`` must not carry them.
INTERNAL_FIELDS = ("llm_rejected", "llm_note")


def _recount_source_stats(v: dict, adj: dict | None) -> dict:
    """``source_stats`` after the LLM pass: rejected external excerpts no longer
    count, and a site none of whose ``sources`` entries is left is removed."""
    from .sources.merge import EXTERNAL_KINDS, SITES

    stats = {k: dict(x) for k, x in (v.get("source_stats") or {}).items()}
    rejected = (adj or {}).get("comments") or []
    kinds_left = {s.get("kind") for s in v.get("sources") or []}
    for kind in EXTERNAL_KINDS:
        if kind not in stats:
            continue
        gone = [c for c in rejected if c.get("site") == SITES[kind]]
        st = stats[kind]
        st["recommenders"] = max(0, int(st.get("recommenders") or 0) - len(gone))
        karma = sum(float(c.get("score") or 0) for c in gone)
        st["sum_score"] = round(max(0.0, float(st.get("sum_score") or 0) - karma), 1)
        if kind not in kinds_left or not st["recommenders"]:
            del stats[kind]
    if "reddit" in stats:
        reddit = stats["reddit"]
        external = sum(int(x.get("recommenders") or 0) for k, x in stats.items() if k != "reddit")
        reddit["recommenders"] = max(0, int(v.get("recommenders") or 0) - external)
        karma = sum(float(c.get("score") or 0) for c in rejected if c.get("site") in (None, "reddit.com"))
        reddit["sum_score"] = round(max(0.0, float(reddit.get("sum_score") or 0) - karma), 1)
    return stats


def publish(data: dict) -> dict:
    """The data as published in ``site/data/videos.json``: a copy without
    ``INTERNAL_FIELDS`` and with ``source_stats`` recounted after the LLM pass.

    Not reversible by ``strip_llm``: later runs start from the pre-LLM snapshot
    (``pipeline/base_videos.json``) and the verdict cache instead.
    """
    out = {k: x for k, x in data.items() if k != "videos"}
    videos = []
    for v in data.get("videos", []):
        v = dict(v)
        adj = v.get("llm_rejected")
        for key in INTERNAL_FIELDS:
            v.pop(key, None)
        if v.get("source_stats"):
            v["source_stats"] = _recount_source_stats(v, adj)
        videos.append(v)
    out["videos"] = videos
    return out


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
                req_a[cid] = prompt_a(v, todo)
                meta_a[cid] = [comment_key(c) for c in todo]
        budget = _submit(runner, "A", opts.classify_model, req_a, meta_a, ingest_a, opts, budget, report)

        # Stage B: videos stage A did not already drop, and those it drops for
        # lack of a recommendation that may still be kept by content.
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
            if (cache.video(v["id"]) is not None and not needs_reask(cache, v["id"])) or cid in pending:
                continue
            if stage_a_drop(v, cache) and not rescuable(v, cache):
                continue
            if len(req_b) >= opts.max_items:
                break
            ok = [c for c in v.get("comments") or [] if comment_verdict(cache, v["id"], c) is not False]
            req_b[cid] = prompt_b(v, ok)
            meta_b[cid] = v["id"]
        _submit(runner, "B", opts.model, req_b, meta_b, ingest_b, opts, budget, report)
        report.usage = runner.usage
        _log_run(cache, runner, opts)
        cache.save()
    data = apply_verdicts(data, cache, report)
    return data, report


def _submit(runner: Runner, stage: str, model: str, users: dict[str, str], meta: dict, ingest: Ingest,
            opts: Options, budget: float, report: Report) -> float:
    """Estimate, check the budget, probe the output style on one request, send the rest."""
    if not users:
        log.info("LLM stage %s: nothing new to send", stage)
        return budget
    style = runner.cache.data.get("styles", {}).get(model, "tool")
    batch = not opts.force_sync and len(users) > runner.sync_max
    requests = {cid: request_params(stage, model, u, style) for cid, u in users.items()}
    est = estimate(stage, model, requests, meta, batch, style)
    log.info("LLM stage %s: %d requests to %s via %s, estimated cost $%.2f (budget left $%.2f)", stage,
             len(requests), model, "Message Batches" if batch else "Messages", est, budget)
    if est > budget:
        msg = f"stage {stage}: estimated ${est:.2f} exceeds the remaining budget ${budget:.2f}; not sent"
        log.error("LLM %s", msg)
        report.skipped.append(msg)
        return budget
    try:
        first = next(iter(users))
        probed = runner.probe(stage, model, first, users[first], meta[first], ingest)
        if probed is None:
            raise RuntimeError(f"{model} accepted none of the output styles {STYLES}")
        if probed != style:
            requests = {cid: request_params(stage, model, u, probed) for cid, u in users.items()}
            est2 = estimate(stage, model, requests, meta, batch, probed)
            log.info("LLM stage %s: re-estimated for style %r: $%.2f", stage, probed, est2)
            if est2 > budget:
                raise RuntimeError(f"estimated ${est2:.2f} for style {probed!r} exceeds the remaining budget")
            est = est2
        del requests[first]
        runner.run(stage, model, requests, meta, ingest, batch=batch and len(requests) > runner.sync_max)
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
