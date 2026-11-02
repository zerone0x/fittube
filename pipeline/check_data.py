"""Data-quality check for the published ``site/data/videos.json``.

    python -m pipeline.check_data                      # checks site/data/videos.json
    python -m pipeline.check_data path/to/videos.json

Exits non-zero (listing every problem) when the file is not fit to publish:

* a ``summary`` with markup or braces, provenance ("recommended on r/...",
  commenters, wiki pages), meta talk about missing information or guesses
  (or only the title and channel: "Video titled ... from channel ..."),
  reception instead of content ("recommended as a brief ... guide"),
  a personal-clip description, a truncated or dangling ending
  (``llm.summary_problems``, the rule the LLM pass applies before publishing);
* pipeline-internal fields (``llm_rejected``, ``llm_note``);
* ``source_stats`` naming a site no ``sources`` entry points at;
* excerpts whose YouTube link lost the ``&`` before ``t=`` (``v=IDt=268``);
* no videos, records without an id, duplicate ids.

The refresh workflow runs it before committing regenerated data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import config
from .llm import INTERNAL_FIELDS, summary_problems

_GLUED_TIMESTAMP_RE = re.compile(r"(?:youtu\.be/|[?&]v=)[A-Za-z0-9_-]{11}t=\d")
_EXTERNAL_KINDS = ("hn", "forum")


def check(data: dict) -> list[str]:
    """Problems in ``data`` (a ``videos.json`` dict), one line each; ``[]``: fine."""
    problems: list[str] = []
    videos = data.get("videos") if isinstance(data, dict) else None
    if not isinstance(videos, list) or not videos:
        return ["no videos"]
    seen: set[str] = set()
    for i, v in enumerate(videos):
        vid = v.get("id") if isinstance(v, dict) else None
        if not vid:
            problems.append(f"video #{i}: no id")
            continue
        if vid in seen:
            problems.append(f"{vid}: duplicate id")
        seen.add(vid)
        for key in INTERNAL_FIELDS:
            if key in v:
                problems.append(f"{vid}: internal field {key!r}")
        if "summary" in v:
            for problem in summary_problems(v["summary"]):
                problems.append(f"{vid}: summary {problem}: {v['summary']!r}")
        kinds = {s.get("kind") for s in v.get("sources") or []}
        for kind in (v.get("source_stats") or {}):
            if kind in _EXTERNAL_KINDS and kind not in kinds:
                problems.append(f"{vid}: source_stats.{kind} without a {kind} source")
        for c in v.get("comments") or []:
            if _GLUED_TIMESTAMP_RE.search(c.get("text") or ""):
                problems.append(f"{vid}: excerpt with a glued timestamp: {c['text'][:80]!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", type=Path, default=config.OUTPUT_PATH)
    p.add_argument("--max-lines", type=int, default=200, help="problems printed at most (default: %(default)s)")
    args = p.parse_args(argv)
    try:
        data = json.loads(args.path.read_text())
    except (OSError, ValueError) as exc:
        print(f"check_data: cannot read {args.path}: {exc}", file=sys.stderr)
        return 2
    problems = check(data)
    videos = data.get("videos") or [] if isinstance(data, dict) else []
    summaries = sum(1 for v in videos if isinstance(v, dict) and v.get("summary"))
    if problems:
        for line in problems[: args.max_lines]:
            print(line)
        if len(problems) > args.max_lines:
            print(f"... and {len(problems) - args.max_lines} more")
        print(f"check_data: {args.path}: {len(problems)} problem(s) in {len(videos)} videos", file=sys.stderr)
        return 1
    print(f"check_data: {args.path}: OK ({len(videos)} videos, {summaries} summaries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
