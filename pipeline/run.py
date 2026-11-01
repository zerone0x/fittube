"""FitTube data pipeline CLI.

    python -m pipeline.run                      # full run -> site/data/videos.json
    python -m pipeline.run --limit 50           # quick run: 1 page per subreddit, top 50 videos
    python -m pipeline.run --subreddits kettlebell flexibility --max-comments 300
    python -m pipeline.run --offline            # rebuild only from cached API responses
    python -m pipeline.run --offline-search     # cached search pages, but expand threads online
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import config
from .build import build, write
from .fetch import ArcticShiftClient, fetch_all, fetch_threads, merge, pick_threads


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
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("pipeline")
    t0 = time.monotonic()

    max_comments = config.PAGE_SIZE if args.limit else args.max_comments
    search_client = ArcticShiftClient(offline=args.offline or args.offline_search)
    searched = fetch_all(args.subreddits, max_comments, client=search_client)
    log.info("search: %d comments (%d requests, %d cache hits) in %.0fs",
             len(searched), search_client.requests_made, search_client.cache_hits, time.monotonic() - t0)

    threads = pick_threads(searched, min_links=args.thread_min_links,
                           max_threads=0 if args.limit else args.max_threads)
    thread_client = ArcticShiftClient(offline=args.offline)
    comments = merge(searched, fetch_threads(thread_client, threads))
    wanted = {s.lower() for s in args.subreddits}
    comments = [c for c in comments if (c.get("subreddit") or "").lower() in wanted]
    log.info("total: %d comments in %.0fs", len(comments), time.monotonic() - t0)

    data = build(comments, sources=args.subreddits, limit=args.limit)
    n = len(data["videos"])
    if n < args.min_videos:
        log.error("only %d videos (< --min-videos %d); not writing %s", n, args.min_videos, args.output)
        return 1
    write(data, args.output)
    log.info("wrote %d videos to %s in %.0fs", n, args.output, time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
