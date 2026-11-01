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
]

ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"
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
THREAD_MIN_LINKS = 2
MAX_THREADS = int(os.environ.get("FITTUBE_MAX_THREADS", "600"))
THREAD_MAX_PAGES = 5
START_YEAR = 2010
WINDOW_YEARS = 4

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
IGNORED_AUTHORS = {"[deleted]", "automoderator", "[removed]"}
# Authors whose name looks like a bot are dropped too (see build.is_bot).
BOT_SUFFIXES = ("bot", "_bot", "-bot")

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
