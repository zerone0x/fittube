# FitTube pipeline

Builds `site/data/videos.json` (see the data contract in the top-level README)
from Reddit comments that link YouTube videos in fitness subreddits.

Python 3.10+, stdlib + `requests`.

```bash
pip install -r pipeline/requirements.txt
python -m pipeline.run                 # full run (~25-30 min uncached)
python -m pipeline.run --limit 50      # quick run: 1 page per subreddit, top 50 videos
python -m pipeline.run --offline       # rebuild from cached responses only
python -m unittest                     # tests (run from the repo root)
```

## Stages

| Module | What it does |
| --- | --- |
| `config.py` | Subreddits, per-subreddit budget, thresholds, negative-context patterns. |
| `fetch.py` | Arctic Shift comment search (`body=youtube OR youtu.be`), one request at a time, backs off on the API's per-minute budget, caches every page under `pipeline/.cache/`. |
| `extract.py` | YouTube ids from every URL form (`watch?v=`, `youtu.be/`, `shorts/`, `embed/`, `m.`/`music.`, markdown links, escaped `\_`) plus `t=`/`start=` and `mm:ss`/`h:mm:ss` timestamps. |
| `build.py` | Drops bots/deleted/score < 1, down-weights negative context, aggregates per video, thresholds, fetches oEmbed metadata (drops deleted/private videos), tags, scores, sorts. |
| `tagger.py` | Keyword rules mapping title + comments + subreddit priors onto the tag vocabularies. Self-contained so it can be swapped for an LLM tagger. |

## Ranking

`score = recommenders * 10 + log1p(sum_score) * 5`, where `recommenders` is
the number of distinct non-negative comment authors and `sum_score` is the
summed comment karma (negative-context comments count at 25%). A video is kept
if `recommenders >= 2` or `sum_score >= 10`.

## Sampling and the API budget

Arctic Shift's full-text search is slow (~5-10 s per query) and rate limited
per minute, so a run is bounded by the number of requests (~2 per minute).
Each subreddit's history is split into fixed 4-year windows and the
per-subreddit budget (`FITTUBE_MAX_COMMENTS`, default 500) is spread across
them, newest first. This samples every era instead of only recent comments
(whose karma has not settled yet), and keeps cache keys stable: pages from
closed windows never expire, so the weekly GitHub Action (which persists
`pipeline/.cache` with `actions/cache`) mostly refetches the current window.
