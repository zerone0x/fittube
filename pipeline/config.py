"""Pipeline configuration. Everything tunable lives here."""

from __future__ import annotations

import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
CACHE_DIR = Path(os.environ.get("FITTUBE_CACHE_DIR", PIPELINE_DIR / ".cache"))
OUTPUT_PATH = REPO_ROOT / "site" / "data" / "videos.json"

# --- Sources -----------------------------------------------------------------

SUBREDDITS = [
    "bodyweightfitness",
    "Fitness",
    "weightroom",
    "naturalbodybuilding",
    "flexibility",
    "xxfitness",
    "overcominggravity",
    "physicaltherapy",
    "kettlebell",
    "StrongerByScience",
    # Added in #5. Verified to have data: yoga, climbing, powerlifting,
    # Stronglifts5x5, backpain; the rest may be sparse (empty is tolerated).
    "yoga",
    "climbing",
    "climbingtrainer",
    "powerlifting",
    "Stronglifts5x5",
    "backpain",
    "running",
    "advancedrunning",
    "C25K",
    "Posture",
    "fitness30plus",
    "over40fitness",
    "homegym",
    "crossfit",
    "bjj",
    "MuayThai",
    "swimming",
    "Kettleballs",
    "calisthenics",
]

ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"
ARCTIC_SHIFT_POSTS_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
ARCTIC_SHIFT_WIKIS_URL = "https://arctic-shift.photon-reddit.com/api/subreddits/wikis"
# Arctic Shift full-text search; one OR query covers both youtube.com and youtu.be links.
BODY_QUERY = "youtube OR youtu.be"
COMMENT_FIELDS = "id,author,body,score,created_utc,link_id,subreddit"
PAGE_SIZE = 100  # API maximum
# When a query hits the server-side timeout (slow 422), retry with half the page
# size, down to MIN_PAGE_SIZE: finding fewer matches scans fewer rows.
MIN_PAGE_SIZE = 25
SLOW_QUERY_SECONDS = 5.0

# Max comments fetched per subreddit. The budget is spread across time windows
# (see WINDOW_YEARS) so we sample every era, not only the newest comments.
MAX_COMMENTS_PER_SUB = int(os.environ.get("FITTUBE_MAX_COMMENTS", "1000"))

# Thread expansion: after the search pass, fetch whole threads in which at
# least THREAD_MIN_LINKS sampled comments link YouTube. These are cheap
# (indexed link_id lookups) compared to full-text search.
THREAD_MIN_LINKS = 1  # the committed data was built with 1 / 3000
MAX_THREADS = int(os.environ.get("FITTUBE_MAX_THREADS", "3000"))
THREAD_MAX_PAGES = 5
START_YEAR = 2010
WINDOW_YEARS = 4

# --- Reddit wikis and posts (pipeline/reddit/) ----------------------------------

# Moderator-curated subreddit wikis. A subreddit without an archived wiki is skipped.
WIKI_SUBREDDITS = [
    "bodyweightfitness", "Fitness", "flexibility", "xxfitness", "running", "yoga",
    "powerlifting", "climbingtrainer", "weightroom", "naturalbodybuilding",
]
# Paths with these segments are wiki plumbing, not content. Moderation and
# rules pages are not recommendation lists either: r/Fitness/wiki/moderation
# links memes ("Dog of Wisdom") as examples of what gets removed.
WIKI_SKIP_PATH_RE = (r"/wiki/(?:edit|revisions|settings|config|pages|discussions)(?:/|$)|automoderator|/wiki/config/|"
                     r"moderat(?:ion|ors?)|(?:^|[/_-])(?:rules|config)(?:[/_-]|$)")
WIKI_PATHS_PER_REQUEST = 50  # API maximum is 100; keep URLs short
# A wiki listing is a strong, curated signal: each subreddit wiki that lists a
# video counts as one recommender ("wiki:<subreddit>") plus WIKI_BOOST score
# (i.e. a wiki mention is worth 1 + WIKI_BOOST/10 = 4 recommenders in ranking).
WIKI_BOOST = 30.0

# Submissions: full-text search per subreddit, budget spread over WINDOW_YEARS windows.
POST_QUERY = "youtube"
POST_FIELDS = "id,author,title,selftext,url,score,num_comments,created_utc,subreddit"
MAX_POSTS_PER_SUB = int(os.environ.get("FITTUBE_MAX_POSTS", "300"))
# Post search on very active subreddits can hit the server-side timeout over
# and over; give up on a subreddit's posts sooner than on comment pages.
POST_MAX_RETRIES = 3
# "Recommend me videos" posts: their replies are fetched (cheap link_id lookups)
# and processed like normal comments.
REQUEST_POST_MIN_COMMENTS = 2
MAX_REQUEST_THREADS = int(os.environ.get("FITTUBE_MAX_REQUEST_THREADS", "400"))
# Posts sharing the author's own content are not recommendations.
SELF_PROMO_PATTERNS = [
    r"\bmy (?:new |first |latest |own )?(?:youtube )?(?:video|channel|vlog|vid)s?\b",
    r"\bi (?:made|created|filmed|uploaded|recorded|put together|started) (?:a|an|this|my)\b",
    r"\bsubscribe\b",
    r"\bcheck out my\b",
    r"\bform check\b",
    r"\bhere'?s me\b",
]

# Politeness: minimum seconds between requests, retries and backoff.
MIN_REQUEST_INTERVAL = 1.0
# After a successful search, wait REST_FACTOR x its duration (searches cost
# ~1-10 s of server time and the API budgets server time per client).
REST_FACTOR = 3.0
# On 422 "Timeout. Maybe slow down a bit" / 429 / 5xx: exponential backoff.
# Hammering at the advertised per-minute reset keeps the client throttled; a
# few minutes of silence clears it.
BASE_BACKOFF_SECONDS = 30
MAX_BACKOFF_SECONDS = 300
MAX_RETRIES = 7
REQUEST_TIMEOUT = 60
# Cached pages whose window ended more than this many days ago never expire;
# newer pages are refetched after CACHE_TTL_DAYS.
CACHE_TTL_DAYS = 6
IMMUTABLE_AFTER_DAYS = 60

USER_AGENT = "fittube-pipeline/0.1 (+https://github.com/zerone0x/fittube)"

# --- Filtering / ranking -----------------------------------------------------

MIN_COMMENT_SCORE = 1
# Videos linked as jokes, never as recommendations (rickrolls).
VIDEO_BLOCKLIST = {"dQw4w9WgXcQ", "oHg5SJYRHA0"}
IGNORED_AUTHORS = {"[deleted]", "automoderator", "[removed]"}
# Authors whose name looks like a bot are dropped too (see build.is_bot).
BOT_SUFFIXES = ("bot", "_bot", "-bot")
# Bots whose names look human are caught by their comment body instead, e.g.
# the "Videos in this thread: Watch Playlist" bot.
BOT_BODY_PATTERNS = [
    r"^\W*videos in this thread\b",
    r"\bwatch playlist\b.{0,40}\bvideo\s*\|\s*comment\b",
    r"\bi(?:'| a)m a bot\b",
    r"\bthis (?:comment|action) was (?:made|performed) (?:by a bot|automatically)\b",
]

# Comments matching any of these are treated as negative context and
# down-weighted (they still count, but much less).
NEGATIVE_PATTERNS = [
    r"\bdon'?t (?:watch|follow|listen|do this)\b",
    r"\bdo not (?:watch|follow|listen)\b",
    r"\bterrible form\b",
    r"\b(?:bad|awful|horrible|shit|crap|dangerous) form\b",
    r"\bthis is (?:wrong|bad|terrible|bullshit|garbage)\b",
    r"\bbro ?science\b",
    r"\bclickbait\b",
    r"\bscam\b",
    r"\bgarbage\b",
    r"\bavoid (?:this|him|her|them)\b",
    r"\bmisinformation\b",
    r"\bfake natty\b",
    r"\bnot recommend\b",
    r"\bwould(?:n'?t| not) recommend\b",
]
NEGATIVE_WEIGHT = 0.25

MIN_RECOMMENDERS = 2
MIN_SUM_SCORE = 10

TOP_COMMENTS = 3
EXCERPT_CHARS = 280
MAX_TIMESTAMPS = 10

OEMBED_URL = "https://www.youtube.com/oembed"
OEMBED_WORKERS = 4

# --- Non-Reddit sources (pipeline/sources/) ------------------------------------
# Their mentions are merged into the Reddit build (see sources/merge.py).
# Recommenders are namespaced per source ("hn:pg", "tnation:alice") so they
# never collide with Reddit usernames.

# Hacker News via the Algolia API (no key). One query per keyword and link form.
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={id}"
HN_KEYWORDS = [
    "exercise", "workout", "mobility", "stretching", "deadlift", "squat",
    "kettlebell", "physical therapy", "back pain", "posture", "zone 2",
    "running form", "strength training", "calisthenics", "yoga", "rehab",
]
HN_LINK_QUERIES = ["youtube.com", "youtu.be"]
HN_HITS_PER_PAGE = 1000  # API maximum
HN_MAX_PAGES = int(os.environ.get("FITTUBE_HN_MAX_PAGES", "3"))  # per keyword x link form
HN_MIN_REQUEST_INTERVAL = 0.5
# Strict relevance: a fitness term must appear within this many characters of
# the YouTube link in the comment text (HN threads are mostly off-topic).
HN_NEAR_CHARS = 250

# T-Nation forum (Discourse).
TNATION_BASE_URL = "https://t-nation.com"
TNATION_QUERY = "youtube.com order:latest"
TNATION_MAX_PAGES = int(os.environ.get("FITTUBE_TNATION_MAX_PAGES", "30"))
TNATION_MIN_REQUEST_INTERVAL = 1.0
# Category slugs whose posts are dropped: training logs are mostly users' own
# lift clips; politics/physique-rating threads are off-topic.
TNATION_DROP_CATEGORIES = {"training-logs", "politics-and-world-issues", "rate-my-physique-photos"}
# Posts in these categories count as-is; elsewhere (Main Forum lounge, Pharma,
# ...) the post or its topic title must mention a fitness term.
TNATION_TRUSTED_CATEGORIES = {"training", "over-35-lifter", "bodybuilding", "supplements-and-nutrition"}
# Topics that are personal logs/challenges outside the log category.
TNATION_DROP_TOPIC_PATTERN = r"\b(?:challenge|log|journal|journey|diary|progress)\b"
# A site's own channel linked from its own articles is promotion, not a recommendation.
EXTERNAL_SITE_CHANNELS = {"t-nation.com": ["Testosterone Nation", "T Nation", "T-Nation"]}

# Cache freshness for the forum/HN sources (offline runs ignore it).
EXTERNAL_SEARCH_TTL_DAYS = 6
EXTERNAL_POST_TTL_DAYS = 30

# --- LLM pass (pipeline/llm.py, issue #3) --------------------------------------
# Runs only when ANTHROPIC_API_KEY is set; cached verdicts in LLM_CACHE_PATH
# (committed) are applied on every run, with or without a key.

LLM_CACHE_PATH = Path(os.environ.get("FITTUBE_LLM_CACHE", PIPELINE_DIR / "llm_cache.json"))
# The data as it was before the LLM pass (after carry-over and the HN/T-Nation
# merge), committed: --llm-only runs and the next carry-over start from it, so
# a video the LLM pass drops can be reconsidered later instead of being lost.
BASE_PATH = Path(os.environ.get("FITTUBE_BASE", PIPELINE_DIR / "base_videos.json"))
# Stage B (vetting, tags, summary): FITTUBE_MODEL, default Sonnet. Stage A
# (is this excerpt a recommendation?) defaults to the same model: in a sample
# of the 40 top videos Haiku rejected ~25 of 120 excerpts that were plain
# helpful links (it wanted explicit praise, and the link is cut from most
# excerpts); set FITTUBE_CLASSIFY_MODEL=claude-haiku-5-5 for a cheaper run.
LLM_MODEL = os.environ.get("FITTUBE_MODEL") or "claude-sonnet-5-5"
LLM_CLASSIFY_MODEL = os.environ.get("FITTUBE_CLASSIFY_MODEL") or LLM_MODEL
# USD per million (input, output) tokens at list price, used for the cost
# estimate and report. Unknown models are priced like the most expensive one.
# The Message Batches API costs LLM_BATCH_DISCOUNT of that.
LLM_PRICES = {
    "claude-haiku-5-5": (1.0, 5.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-opus-5-5": (5.0, 25.0),
}
LLM_BATCH_DISCOUNT = 0.5
# Budget guard: at most this many requests per stage and this estimated cost
# (USD) per run; a stage that would exceed the remaining budget is skipped.
# A full run over ~900 videos is ~700 stage-A + ~900 stage-B requests.
LLM_MAX_ITEMS = int(os.environ.get("FITTUBE_LLM_MAX_ITEMS", "3000"))
LLM_MAX_COST = float(os.environ.get("FITTUBE_LLM_MAX_COST", "15"))
# Up to this many requests run synchronously (e.g. --limit runs); more go
# through the Message Batches API.
LLM_SYNC_MAX = 25
LLM_SYNC_WORKERS = 4
# Minutes to wait for batches; unfinished batches stay in the cache
# ("pending") and are collected by the next run.
LLM_WAIT_MINUTES = 30.0
LLM_POLL_SECONDS = 30
