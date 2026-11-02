"""FitTube data pipeline CLI.

    python -m pipeline.run                      # full run -> site/data/videos.json
    python -m pipeline.run --limit 50           # quick run: 1 page per subreddit, top 50 videos
    python -m pipeline.run --subreddits kettlebell flexibility --max-comments 300
    python -m pipeline.run --offline            # rebuild only from cached API responses
    python -m pipeline.run --offline-search     # cached search pages, but expand threads online
    python -m pipeline.run --merge-into site/data/videos.json
                                                # skip Reddit: merge HN/T-Nation into an existing file
    python -m pipeline.run --time-budget 90     # stop fetching after 90 min, build from the cache
    python -m pipeline.run --llm-only           # no crawling: LLM pass over the existing videos.json

Sources, in fetch order (so a time-boxed run gets the best signal first):
subreddit wikis -> posts -> replies of "recommend me videos" posts -> comment
search -> thread expansion. The result is merged with the previous output
(see carryover.py) unless --no-carry-over is given, then vetted and tagged by
the LLM pass (llm.py; needs ANTHROPIC_API_KEY, cached verdicts apply without).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from . import carryover, config, llm
from .build import build, write
from .fetch import ArcticShiftClient, fetch_all, fetch_threads, merge, pick_threads
from .reddit import posts as reddit_posts
from .reddit import wiki as reddit_wiki
from .sources import Mention
from .sources import hn, tnation
from .sources.merge import merge_external, strip_external


def collect_external(args, log: logging.Logger) -> list[Mention]:
    """Mentions from the non-Reddit sources; a failing source is logged and skipped."""
    offline = args.offline
    jobs = []
    if not args.no_hn:
        jobs.append(("hn", lambda: hn.collect(max_pages=args.hn_max_pages, offline=offline)))
    if not args.no_tnation:
        jobs.append(("tnation", lambda: tnation.collect(max_pages=args.tnation_max_pages, offline=offline)))
    mentions: list[Mention] = []
    for name, job in jobs:
        try:
            mentions.extend(job())
        except Exception:  # noqa: BLE001 - one broken source must not fail the run
            log.exception("source %s failed; continuing without it", name)
    return mentions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subreddits", nargs="+", default=config.SUBREDDITS, help="subreddits to scan")
    p.add_argument("--max-comments", type=int, default=config.MAX_COMMENTS_PER_SUB,
                   help="max comments fetched per subreddit (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="quick run: fetch one page per subreddit and keep only the top N videos")
    p.add_argument("--offline", action="store_true", help="use cached API responses only (no Arctic Shift requests)")
    p.add_argument("--offline-search", action="store_true",
                   help="use cached full-text search pages only (the slow part); thread expansion still runs online")
    p.add_argument("--max-threads", type=int, default=config.MAX_THREADS,
                   help="max threads to expand (0 disables; default: %(default)s)")
    p.add_argument("--thread-min-links", type=int, default=config.THREAD_MIN_LINKS,
                   help="expand threads with at least this many sampled YouTube-linking comments (default: %(default)s)")
    p.add_argument("--output", type=Path, default=config.OUTPUT_PATH)
    p.add_argument("--min-videos", type=int, default=0,
                   help="exit non-zero without writing if fewer videos than this were produced")
    p.add_argument("--no-hn", action="store_true", help="skip the Hacker News source")
    p.add_argument("--no-tnation", action="store_true", help="skip the T-Nation forum source")
    p.add_argument("--hn-max-pages", type=int, default=config.HN_MAX_PAGES,
                   help="max Algolia pages per HN query (default: %(default)s)")
    p.add_argument("--tnation-max-pages", type=int, default=config.TNATION_MAX_PAGES,
                   help="max T-Nation search pages (default: %(default)s)")
    p.add_argument("--merge-into", type=Path, default=None, metavar="VIDEOS_JSON",
                   help="skip Reddit: merge the non-Reddit sources into this existing videos.json "
                        "(a previous merge is stripped first) and write the result to --output")
    p.add_argument("--wiki-subreddits", nargs="*", default=config.WIKI_SUBREDDITS,
                   help="subreddits whose wikis are scanned (none to disable)")
    p.add_argument("--post-subreddits", nargs="*", default=None,
                   help="subreddits whose posts are scanned (default: --subreddits; none to disable)")
    p.add_argument("--max-posts", type=int, default=config.MAX_POSTS_PER_SUB,
                   help="max posts fetched per subreddit (default: %(default)s)")
    p.add_argument("--max-request-threads", type=int, default=config.MAX_REQUEST_THREADS,
                   help="max 'recommend me videos' posts whose replies are fetched (default: %(default)s)")
    p.add_argument("--time-budget", type=float, default=None, metavar="MINUTES",
                   help="stop making API requests after this many minutes and build from what is cached")
    p.add_argument("--previous", type=Path, default=None,
                   help="previous videos.json whose videos are carried over (default: --output)")
    p.add_argument("--no-carry-over", action="store_true",
                   help="replace the previous videos.json instead of merging with it")
    p.add_argument("--llm-only", action="store_true",
                   help="skip all sources: run the LLM pass over --output (or --previous) and write --output")
    p.add_argument("--no-llm", action="store_true", help="skip the LLM pass (keyword tags, no cached verdicts)")
    p.add_argument("--llm-max-items", type=int, default=config.LLM_MAX_ITEMS,
                   help="max LLM requests per stage (default: %(default)s)")
    p.add_argument("--llm-max-cost", type=float, default=config.LLM_MAX_COST,
                   help="max estimated LLM cost in USD per run (default: %(default)s)")
    p.add_argument("--llm-wait", type=float, default=config.LLM_WAIT_MINUTES, metavar="MINUTES",
                   help="minutes to wait for LLM batches before leaving them for the next run (default: %(default)s)")
    p.add_argument("--llm-sync", action="store_true", help="send LLM requests one by one instead of as a batch")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("pipeline")
    t0 = time.monotonic()

    if args.llm_only:
        source = args.previous or args.output
        data = llm.strip_llm(json.loads(source.read_text()))
        carryover.repair_excerpts(data["videos"])
        data = llm_pass(data, args, log)
        data["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return write_checked(data, args, log, t0)

    if args.merge_into:
        data = llm.strip_llm(json.loads(args.merge_into.read_text()))
        data = merge_external(data, collect_external(args, log))
        data = llm_pass(data, args, log)
        data["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write(data, args.output)
        log.info("merged into %s: %d videos -> %s in %.0fs", args.merge_into, len(data["videos"]),
                 args.output, time.monotonic() - t0)
        return 0

    deadline = t0 + args.time_budget * 60 if args.time_budget else None
    max_comments = config.PAGE_SIZE if args.limit else args.max_comments
    search_client = ArcticShiftClient(offline=args.offline or args.offline_search, deadline=deadline)
    thread_client = ArcticShiftClient(offline=args.offline, deadline=deadline)
    post_subs = args.subreddits if args.post_subreddits is None else args.post_subreddits

    # Highest-signal sources first: a time-boxed run must not starve them.
    wiki_ev: list[dict] = []
    try:
        wiki_ev = reddit_wiki.collect(thread_client, args.wiki_subreddits)
    except Exception:  # noqa: BLE001 - one broken source must not fail the run
        log.exception("wiki source failed; continuing without it")
    post_ev: list[dict] = []
    request_threads: list[tuple[str, int]] = []
    try:
        post_ev, request_threads = reddit_posts.collect(search_client, post_subs,
                                                        config.PAGE_SIZE if args.limit else args.max_posts)
    except Exception:  # noqa: BLE001
        log.exception("post source failed; continuing without it")
    log.info("wikis + posts: %d wiki lines, %d posts, %d request threads in %.0fs",
             len(wiki_ev), len(post_ev), len(request_threads), time.monotonic() - t0)

    request_threads = [] if args.limit else request_threads[: args.max_request_threads]
    replies = fetch_threads(thread_client, request_threads)

    searched = fetch_all(args.subreddits, max_comments, client=search_client)
    log.info("search: %d comments (%d requests, %d cache hits) in %.0fs",
             len(searched), search_client.requests_made, search_client.cache_hits, time.monotonic() - t0)

    done = {link for link, _ in request_threads}
    threads = [t for t in pick_threads(searched, min_links=args.thread_min_links,
                                       max_threads=0 if args.limit else args.max_threads) if t[0] not in done]
    comments = merge(post_ev, replies, searched, fetch_threads(thread_client, threads))
    all_subs = list(dict.fromkeys([*args.subreddits, *post_subs, *args.wiki_subreddits]))
    wanted = {s.lower() for s in all_subs}
    comments = [c for c in comments if (c.get("subreddit") or "").lower() in wanted] + wiki_ev
    log.info("total: %d comments/posts + %d wiki lines in %.0fs", len(comments) - len(wiki_ev), len(wiki_ev),
             time.monotonic() - t0)

    data = build(comments, sources=all_subs, limit=args.limit)
    if not args.limit:
        if not args.no_carry_over:
            # Carry over the previous Reddit data (its HN/T-Nation merge stripped
            # first), then merge the external sources into the combined result.
            previous = carryover.load(args.previous or args.output)
            data = carryover.merge_previous(data, strip_external(llm.strip_llm(previous)) if previous else None)
        data = merge_external(data, collect_external(args, log))
    data = llm_pass(data, args, log)
    return write_checked(data, args, log, t0)


def llm_pass(data: dict, args, log: logging.Logger) -> dict:
    """Run the LLM pass (unless --no-llm) and log its report."""
    if args.no_llm:
        return data
    opts = llm.Options(max_items=args.llm_max_items, max_cost=args.llm_max_cost, wait_minutes=args.llm_wait,
                       force_sync=args.llm_sync)
    try:
        data, report = llm.run_pass(data, llm.LLMCache(), llm.make_client(), opts)
    except Exception:  # noqa: BLE001 - keyword-tagged data is still worth writing
        log.exception("LLM pass failed; writing keyword-tagged data")
        return data
    lines = report.lines()
    for line in lines:
        log.info("LLM pass: %s", line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## LLM pass\n\n```\n" + "\n".join(lines) + "\n```\n")
    return data


def write_checked(data: dict, args, log: logging.Logger, t0: float) -> int:
    n = len(data["videos"])
    if n < args.min_videos:
        log.error("only %d videos (< --min-videos %d); not writing %s", n, args.min_videos, args.output)
        return 1
    write(data, args.output)
    log.info("wrote %d videos to %s in %.0fs", n, args.output, time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
