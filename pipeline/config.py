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
START_YEAR = 2010
WINDOW_YEARS = 4

# Politeness: minimum seconds between requests, retries and backoff.
MIN_REQUEST_INTERVAL = 1.0
RESET_MARGIN_SECONDS = 6
MAX_RETRIES = 8
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
