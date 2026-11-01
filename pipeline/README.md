# FitTube pipeline

Builds `site/data/videos.json` (see the data contract in the top-level README)
from Reddit comments, posts and subreddit wikis that link YouTube videos in
fitness subreddits.

Python 3.10+, stdlib + `requests`.

```bash
pip install -r pipeline/requirements.txt
python -m pipeline.run                 # full run (slow when uncached, see below)
python -m pipeline.run --limit 50      # quick run: 1 page per subreddit, top 50 videos
python -m pipeline.run --offline       # rebuild from cached responses only
python -m pipeline.run --offline-search --thread-min-links 1 --max-threads 2000
                                       # cached search pages + (cheap) expansion of every linked thread
python -m pipeline.run --time-budget 90
                                       # stop making API requests after 90 min, build from the cache
python -m pipeline.run --no-carry-over # replace videos.json instead of merging with it
python -m unittest                     # tests (run from the repo root)
```

## Stages

| Module | What it does |
| --- | --- |
| `config.py` | Subreddits, per-subreddit budget, thresholds, negative-context and bot patterns. |
| `reddit/wiki.py` | Subreddit wikis (`WIKI_SUBREDDITS`): all archived pages (first 100 via `?subreddit=`, the rest from `/wikis/list` + `?paths=` batches); one evidence item per page line linking YouTube. |
| `reddit/posts.py` | Submissions matching `query=youtube` (the `url=` filter times out), sampled over the same time windows as comments. Posts linking a video count like comments (post karma as score); questions and self-promotion do not. Replies of "recommend me videos" posts are fetched like threads. |
| `fetch.py` | Arctic Shift comment search (`body=youtube OR youtu.be`), one request at a time, backs off on the API's per-minute budget, caches every page under `pipeline/.cache/`. |
| `fetch.py` (threads) | Thread expansion: whole threads in which sampled comments already link YouTube are fetched with cheap `link_id` lookups, keeping comments that link YouTube. |
| `extract.py` | YouTube ids from every URL form (`watch?v=`, `youtu.be/`, `shorts/`, `embed/`, `m.`/`music.`, markdown links, escaped `\_`) plus `t=`/`start=` and `mm:ss`/`h:mm:ss` timestamps. |
| `build.py` | Drops bots (by name or body, e.g. the "Videos in this thread / Watch Playlist" bot)/deleted/score < 1, down-weights negative context, aggregates per video, thresholds, fetches oEmbed metadata (drops deleted/private videos), tags, scores, sorts, writes `sources`. |
| `carryover.py` | Merges the fresh build with the previous `videos.json` (see below). |
| `tagger.py` | Keyword rules mapping title + comments + subreddit priors + wiki page paths (e.g. `/wiki/exercises/handstand` -> skill, upper) onto the tag vocabularies. Self-contained so it can be swapped for an LLM tagger. |

### Adding a Reddit source

Modules in `pipeline/reddit/` expose `collect(client, subreddits, ...)` and
return *evidence*: dicts shaped like Arctic Shift comments (`id`, `author`,
`body`, `score`, `created_utc`, `subreddit`, `link_id`) plus a `kind`
(`"post"`, `"wiki"`; plain comments have none). `build.aggregate` handles all
kinds; a new kind needs a branch there and an entry in `build.source_entries`.
A source must log and skip a failing subreddit rather than raise, and
`run.py` additionally wraps each source so one broken source never fails the
run. Fetch through `ArcticShiftClient` (`search` for paged endpoints,
`get_json` otherwise) to get caching, backoff and the time budget for free.

## Ranking

`score = recommenders * 10 + log1p(sum_score) * 5 + WIKI_BOOST * wikis`, where
`recommenders` is the number of distinct non-negative comment/post authors plus
one per subreddit wiki listing the video (author `wiki:<subreddit>`),
`sum_score` is the summed comment and post karma (negative-context comments
count at 25%) and `wikis` is the number of distinct subreddit wikis listing the
video. `WIKI_BOOST = 30`, so a wiki listing is worth as much as four
independent recommenders: moderators curate these lists. A video is kept if it
is listed in a wiki, `recommenders >= 2` or its *comment* karma is `>= 10`
(post karma ranks but cannot qualify a video alone: posts linking a video are
mostly media shares such as meet footage or films, so a lone post needs
corroboration).

## Sampling and the API budget

Arctic Shift's full-text search is slow (~1-10 s per query) and the API
throttles clients by server time used (HTTP 422 "Timeout. Maybe slow down a
bit"). Retrying at the advertised per-minute reset keeps a client throttled,
so `fetch.py` rests 3x each query's duration after a success and backs off
exponentially (30 s, 60 s, 120 s, ...) on 422/429/5xx. In practice that is
1-3 pages of 100 comments per minute, so an uncached full run (10 subreddits x
`FITTUBE_MAX_COMMENTS`, default 1000) takes roughly an hour; cached reruns take
seconds.

Each subreddit's history is split into fixed 4-year windows and the
per-subreddit budget is spread across them, newest first. This samples every
era instead of only recent comments (whose karma has not settled yet), and
keeps cache keys stable: pages from closed windows never expire, so the weekly
GitHub Action (which persists `pipeline/.cache` with `actions/cache`) mostly
refetches the current window. If a query keeps timing out server-side the page
size is halved (down to 25) for that subreddit; a window that still fails is
skipped and the run continues.

## Hacker News and T-Nation sources

`pipeline/sources/` adds two non-Reddit sources. Each yields `Mention`s (one
post linking one video); `sources/merge.py` folds them into the Reddit build.

| Module | What it does |
| --- | --- |
| `sources/hn.py` | Algolia `search_by_date` for `youtube.com <kw>` / `youtu.be <kw>` comments (`HN_KEYWORDS`), paged with `created_at_i<cursor`, up to `HN_MAX_PAGES` x 1000 hits per query. Strict relevance: a fitness term (`sources.FITNESS_TERMS`) must appear within `HN_NEAR_CHARS` of the link. |
| `sources/tnation.py` | Discourse `search.json?q=youtube.com order:latest` (max 10 pages per query, so deeper pages re-query with `before:<date>`; `TNATION_MAX_PAGES` total), then `/t/<topic>/<post>.json` for the full post. Drops training-log / politics / physique-rating categories, log/challenge/journal topics, posts that read like the user's own clip ("my set", "form check", "here's me"...), quoted links, and -- outside the training categories -- posts with no fitness term. |
| `sources/merge.py` | Namespaced recommenders (`hn:<user>`, `tnation:<user>`), self-promotion filter (author name == channel, or T-Nation linking its own channel), thresholds for new videos, `sources` / `source_stats` / `needs_llm_check` fields (see the top-level README). Idempotent: a previous merge is stripped first. |

Videos already on the site take any relevant mention as corroboration; videos
found only on HN/T-Nation must pass the usual thresholds on their own
(at least 2 distinct recommenders; T-Nation likes only affect ranking) and are flagged
`needs_llm_check`. All mentions are flagged `needs_llm_check` in `sources`.

Both sources use a cached JSON client (`pipeline/.cache/hn`, `pipeline/.cache/tnation`;
search pages refresh after 6 days, posts after 30) with a descriptive
User-Agent and >= 1 s between T-Nation requests. A failing source is logged
and skipped; the run continues.

```bash
python -m pipeline.run --no-hn --no-tnation            # Reddit only
python -m pipeline.run --tnation-max-pages 10          # smaller forum crawl
python -m pipeline.run --merge-into site/data/videos.json
    # skip Reddit entirely: merge HN/T-Nation into an existing videos.json
    # (no Arctic Shift requests; Reddit data is taken as-is from the file)
```

`--merge-into` works from the committed file alone: a video's Reddit
recommenders are its `recommenders` count (namespaced external authors can
never coincide with them, so the union is a sum) and its Reddit karma is
recovered by inverting the ranking formula from `score`. It is how the
committed `site/data/videos.json` gained its HN/T-Nation data without
re-crawling Arctic Shift.

## Time budget, carry-over and CI

Sources are fetched best-signal first: wikis, posts, replies of request posts,
comment search, thread expansion. With `--time-budget MINUTES` the clients stop
making requests once the budget is spent (a backoff that would overrun it gives
up) and everything after that is served from the cache, including stale pages,
so the run still builds from everything fetched so far.

The result is then merged with the previous `videos.json` (`--previous`,
default `--output`; `--no-carry-over` disables it): previous videos are never
dropped just because a source was rate-limited or out of budget this run, only
when YouTube oEmbed reports them deleted/private. For videos in both, the
merge keeps the larger community recommender count and score (wiki boost
re-applied over the merged `sources`), and unions comments, timestamps,
subreddits, sources and list tags. Excerpts from body-detected bots are
stripped from carried records together with the recommender they counted as.

The weekly `refresh-data.yml` workflow restores `pipeline/.cache` with
`actions/cache/restore`, runs with `--time-budget 90`, and saves the cache with
`actions/cache/save` under `if: always()`, so even a failed or timed-out run
leaves a warmer cache for the next one. The defaults (`THREAD_MIN_LINKS = 1`,
`MAX_THREADS = 3000`) match the settings the committed data was built with.
