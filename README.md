# FitTube

A fitness video site in the spirit of Karpathy's HackerTube: instead of an engagement-driven feed, it shows YouTube videos that **fitness communities on Reddit actually recommend**.

## How it works

1. `pipeline/` pulls historical Reddit comments containing YouTube links from fitness subreddits (via the public Arctic Shift API), aggregates them per video, filters noise, fetches metadata via YouTube oEmbed (no API key), tags videos, and writes `site/data/videos.json`.
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
      "tags": {
        "goal": ["strength|hypertrophy|fat-loss|mobility|rehab|endurance|skill"],
        "body": ["full|upper|lower|core|back|chest|shoulders|arms|legs|glutes|neck|hips"],
        "equipment": ["none|dumbbell|barbell|kettlebell|band|rings|bar|machine"],
        "level": "beginner|intermediate|advanced|any",
        "type": "tutorial|follow-along|explainer|program|motivation"
      }
    }
  ]
}
```

`recommenders` = number of distinct Reddit authors who linked the video. `score` = ranking score computed by the pipeline. `timestamps` come from `t=` params / `mm:ss` mentions in comments. Videos are sorted by `score` desc.

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
- `subreddits` is empty for videos found only outside Reddit; such videos have `needs_llm_check: true` (keyword-filtered evidence only, to be vetted by the LLM pass).
- The top-level `sources` list also names the sites (`news.ycombinator.com`, `t-nation.com`).
