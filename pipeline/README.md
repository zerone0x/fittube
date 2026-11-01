# FitTube pipeline

Builds `site/data/videos.json` (see the data contract in the top-level README)
from Reddit comments that link YouTube videos in fitness subreddits.

Python 3.10+, stdlib + `requests`.

```bash
pip install -r pipeline/requirements.txt
python -m pipeline.run                 # full run (slow when uncached, see below)
python -m pipeline.run --limit 50      # quick run: 1 page per subreddit, top 50 videos
python -m pipeline.run --offline       # rebuild from cached responses only
python -m pipeline.run --offline-search --thread-min-links 1 --max-threads 2000
                                       # cached search pages + (cheap) expansion of every linked thread
python -m unittest                     # tests (run from the repo root)
```

## Stages

| Module | What it does |
| --- | --- |
| `config.py` | Subreddits, per-subreddit budget, thresholds, negative-context patterns. |
| `fetch.py` | Arctic Shift comment search (`body=youtube OR youtu.be`), one request at a time, backs off on the API's per-minute budget, caches every page under `pipeline/.cache/`. |
| `fetch.py` (threads) | Thread expansion: whole threads in which sampled comments already link YouTube are fetched with cheap `link_id` lookups, keeping comments that link YouTube. |
| `extract.py` | YouTube ids from every URL form (`watch?v=`, `youtu.be/`, `shorts/`, `embed/`, `m.`/`music.`, markdown links, escaped `\_`) plus `t=`/`start=` and `mm:ss`/`h:mm:ss` timestamps. |
| `build.py` | Drops bots/deleted/score < 1, down-weights negative context, aggregates per video, thresholds, fetches oEmbed metadata (drops deleted/private videos), tags, scores, sorts. |
| `tagger.py` | Keyword rules mapping title + comments + subreddit priors onto the tag vocabularies. Self-contained so it can be swapped for an LLM tagger. |

## Ranking

`score = recommenders * 10 + log1p(sum_score) * 5`, where `recommenders` is
the number of distinct non-negative comment authors and `sum_score` is the
summed comment karma (negative-context comments count at 25%). A video is kept
if `recommenders >= 2` or `sum_score >= 10`.

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
