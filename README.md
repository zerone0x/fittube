# FitTube

A fitness video site in the spirit of Karpathy's HackerTube: instead of an engagement-driven feed, it shows YouTube videos that **fitness communities on Reddit actually recommend**.

## How it works

1. `pipeline/` pulls historical Reddit comments and posts containing YouTube links, plus the moderator-curated wikis, from fitness subreddits (via the public Arctic Shift API), aggregates them per video, filters noise, fetches metadata via YouTube oEmbed (no API key), tags videos, and writes `site/data/videos.json`. An LLM pass (Claude, run in GitHub Actions) then checks that each saved comment really recommends its video, drops non-fitness videos, personal lift clips, event footage and anti-recommendations, and writes the tags and a one-sentence summary (see `pipeline/README.md`). Without `ANTHROPIC_API_KEY` the keyword tagger and the cached LLM verdicts are used.
2. `site/` is a static, build-free frontend (HTML/CSS/vanilla JS) deployed to GitHub Pages.

## Data contract: `site/data/videos.json`

```json
{
  "generated_at": "2026-11-01T00:00:00Z",
  "sources": ["bodyweightfitness", "Fitness"],
  "videos": [
    {
      "id": "YouTube video id (11 chars)",
      "title": "string",
      "channel": "string",
      "thumbnail": "https://i.ytimg.com/vi/<id>/hqdefault.jpg",
      "recommenders": 7,
      "score": 123.4,
      "first_seen": "2015-03-02",
      "subreddits": ["bodyweightfitness"],
      "comments": [{"text": "excerpt <=280 chars", "score": 42, "permalink": "https://www.reddit.com/..."}],
      "timestamps": [{"t": 95, "label": "optional label"}],
      "summary": "optional one-sentence description written by the LLM pass",
      "tags": {
        "goal": ["strength|hypertrophy|fat-loss|mobility|rehab|endurance|skill"],
        "body": ["full|upper|lower|core|back|chest|shoulders|arms|legs|glutes|neck|hips"],
        "equipment": ["none|dumbbell|barbell|kettlebell|band|rings|bar|machine"],
        "level": "beginner|intermediate|advanced|any",
        "type": "tutorial|follow-along|explainer|program|motivation"
      },
      "sources": [
        {"kind": "wiki", "subreddit": "bodyweightfitness", "path": "/wiki/exercises/handstand", "url": "https://www.reddit.com/r/bodyweightfitness/wiki/exercises/handstand"},
        {"kind": "post", "subreddit": "flexibility", "url": "https://www.reddit.com/r/flexibility/comments/<id>/", "score": 28},
        {"kind": "comment", "subreddit": "Fitness", "count": 4}
      ]
    }
  ]
}
```

`recommenders` = number of distinct Reddit authors who linked the video (in comments or posts); each subreddit wiki listing the video counts as one more. Authors whose saved comment the LLM pass judged not to be a recommendation are not counted. `score` = ranking score computed by the pipeline (a wiki listing adds a fixed boost, see `pipeline/README.md`). `timestamps` come from `t=` params / `mm:ss` mentions in comments. Videos are sorted by `score` desc.

`sources` (optional) says where the recommendations come from. Entries share a `kind`:

| `kind` | Fields | Meaning |
| --- | --- | --- |
| `wiki` | `subreddit`, `path`, `url` | Listed in a moderator-curated subreddit wiki page |
| `post` | `subreddit`, `url`, `score` | A submission linking the video (top 3 by score) |
| `comment` | `subreddit`, `count` (optional) | Comments linking the video in that subreddit |
| `hn` / `forum` | `site`, `url`, `score` | Non-Reddit sources (Hacker News, forums) |

Consumers must ignore unknown kinds and fields. Records built before `sources` existed may lack it.

`summary` (optional) is a one-sentence, LLM-written description of the video itself (not of where it was recommended); consumers must treat it as plain text and cope with its absence (videos not yet reviewed by the LLM pass, or whose summary did not pass the quality check, have none). The excerpts the LLM pass judged not to be recommendations (and the `sources` entries pointing at them) are hidden from `comments`/`sources` and no longer counted in `recommenders`, `score` and `source_stats`. The published file carries no pipeline-internal fields (`llm_rejected`, `llm_note`); `python -m pipeline.check_data` enforces this and the summary rules before CI commits regenerated data.

### Non-Reddit sources (Hacker News, T-Nation)

The pipeline also merges YouTube links from Hacker News comments and the T-Nation forum (see `pipeline/README.md`). Recommenders are counted per source with namespaced authors (`hn:pg`, `tnation:alice`), so `recommenders` is the number of distinct people across all sources. These optional fields appear on videos with such mentions:

```json
{
  "sources": [{"kind": "hn|forum", "site": "news.ycombinator.com|t-nation.com", "url": "permalink to the post", "score": 3, "needs_llm_check": true}],
  "source_stats": {"reddit": {"recommenders": 7, "sum_score": 120.0}, "hn": {"recommenders": 1, "sum_score": 0}, "forum": {"recommenders": 2, "sum_score": 5}},
  "needs_llm_check": false
}
```

- `comments[]` / `timestamps[]` entries from these sites carry `"site"`; comment `permalink` then points to the HN item / T-Nation post and `score` may be `null` (HN does not expose comment points).
- `subreddits` is empty for videos found only outside Reddit; such videos have `needs_llm_check: true` (keyword-filtered evidence only, to be vetted by the LLM pass). The LLM pass sets it (and the flag on each `sources` entry whose excerpt it accepted) to `false`, and drops videos it rejects.
- The top-level `sources` list also names the sites (`news.ycombinator.com`, `t-nation.com`).
