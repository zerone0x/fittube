"""Extract YouTube video ids and timestamps from Reddit comment bodies."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlsplit

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Any URL that points at YouTube. We grab the whole URL and parse it properly
# afterwards. Stops at whitespace and characters that end markdown links.
_URL_RE = re.compile(
    r"(?:https?://)?(?:[a-z0-9-]+\.)*(?:youtube\.com|youtube-nocookie\.com|youtu\.be)/[^\s)\]<>\"'|]*",
    re.IGNORECASE,
)
_MD_LINK_RE = re.compile(r"\[([^\]]{1,200})\]\(\s*<?([^)\s>]+)>?[^)]*\)")
_PATH_ID_RE = re.compile(r"^/(?:shorts|embed|v|e|live)/([A-Za-z0-9_-]{11})(?:[/?#&]|$)")
_DURATION_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$", re.IGNORECASE)
# 1:23, 01:23, 1:02:03 -- not part of a larger number/ratio, not a clock time.
_TEXT_TS_RE = re.compile(
    r"(?<![\d:.,/])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:]|\s*(?:[ap]\.?m\b))",
    re.IGNORECASE,
)


_GENERIC_LABEL_RE = re.compile(r"\b(?:this|the|that|my|his|her|their|a|his) (?:video|vid|clip)\b|^(?:video|vid|link|here)\b"
                               r"|\bvideo as well\b", re.IGNORECASE)


@dataclass
class VideoRef:
    id: str
    timestamps: list[tuple[int, str]] = field(default_factory=list)


def clean_body(text: str) -> str:
    """Undo Reddit's HTML entities and markdown escapes (``\\_`` in urls)."""
    text = html.unescape(text or "")
    return re.sub(r"\\([_\\*~^&()\[\]#-])", r"\1", text)


def parse_duration(value: str) -> int | None:
    """Parse ``95``, ``95s``, ``1m35s``, ``1h2m3s`` into seconds."""
    value = (value or "").strip()
    if not value:
        return None
    m = _DURATION_RE.match(value)
    if not m or not any(m.groups()):
        return None
    h, mnt, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mnt * 60 + s


def parse_youtube_url(url: str) -> tuple[str, int | None] | None:
    """Return ``(video_id, start_seconds_or_None)`` for a YouTube URL."""
    url = url.strip().rstrip(".,;:!?*")
    if re.search(r"\?v%3d", url, re.IGNORECASE):  # fully percent-encoded query (seen in post URLs)
        url = unquote(url)
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    query = parse_qs(parts.query)
    # youtu.be/ID#t=30 and youtube.com/watch?v=ID#t=1m are both seen in the wild.
    frag = parse_qs(parts.fragment)

    vid = None
    if host == "youtu.be":
        candidate = parts.path.lstrip("/").split("/")[0]
        vid = candidate[:11] if len(candidate) >= 11 else None
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if parts.path.rstrip("/") in ("/watch", "") and "v" in query:
            vid = query["v"][0][:11]
        elif parts.path.startswith("/attribution_link") and "u" in query:
            return parse_youtube_url("https://youtube.com" + query["u"][0])
        else:
            m = _PATH_ID_RE.match(parts.path)
            vid = m.group(1) if m else None
    if not vid or not VIDEO_ID_RE.match(vid):
        return None

    start = None
    for source in (query, frag):
        for key in ("t", "start", "time_continue"):
            if key in source:
                start = parse_duration(source[key][0])
                if start is not None:
                    break
        if start is not None:
            break
    return vid, start


def text_timestamps(text: str) -> list[tuple[int, str]]:
    """Find ``mm:ss`` / ``h:mm:ss`` mentions in free text with a short label.

    The label is the rest of the line after the timestamp (or the text before
    it when nothing follows), trimmed to 80 chars.
    """
    out: list[tuple[int, str]] = []
    for line in text.splitlines():
        line_no_urls = _URL_RE.sub(" ", _MD_LINK_RE.sub(lambda m: m.group(1), line))
        matches = list(_TEXT_TS_RE.finditer(line_no_urls))
        for i, m in enumerate(matches):
            h, mnt, s = m.group(1), int(m.group(2)), int(m.group(3))
            if s >= 60 or (h is not None and mnt >= 60):
                continue
            seconds = (int(h) * 3600 if h else 0) + mnt * 60 + s
            end = matches[i + 1].start() if i + 1 < len(matches) else len(line_no_urls)
            label = line_no_urls[m.end():end]
            if not label.strip(" -–—:|*_>.,") and i == 0:
                label = line_no_urls[: m.start()]
            out.append((seconds, _clean_label(label)))
    return out


def _clean_label(label: str) -> str:
    label = re.sub(r"[*_`>#~]+", "", label)
    label = re.sub(r"\s+", " ", label).strip(" -–—:|()[].,;")
    label = re.sub(r"^(?:at|@|around|from)(?:\s+|$)|\s+(?:at|@|around|from)$", "", label, flags=re.IGNORECASE)
    if _GENERIC_LABEL_RE.search(label):
        return ""  # "in this video", "check out my video" ... add nothing
    if len(label) > 80:
        label = label[:79].rsplit(" ", 1)[0] + "…"
    return label


def extract_videos(body: str) -> list[VideoRef]:
    """Return the YouTube videos referenced by a comment, in order of appearance.

    Timestamps come from ``t=``/``start=`` url params and, when the comment
    links exactly one video, from ``mm:ss`` mentions in the text.
    """
    text = clean_body(body)
    md_labels: dict[str, str] = {}
    for m in _MD_LINK_RE.finditer(text):
        md_labels[m.group(2)] = m.group(1)

    refs: dict[str, VideoRef] = {}
    for m in _URL_RE.finditer(text):
        raw = m.group(0)
        parsed = parse_youtube_url(raw)
        if not parsed:
            continue
        vid, start = parsed
        ref = refs.setdefault(vid, VideoRef(vid))
        if start is not None:
            label = md_labels.get(raw, "")
            if label and _URL_RE.search(label):
                label = ""
            ref.timestamps.append((start, _clean_label(label)))

    if len(refs) == 1:
        ref = next(iter(refs.values()))
        ref.timestamps.extend(text_timestamps(text))

    for ref in refs.values():
        ref.timestamps = merge_timestamps(ref.timestamps)
    return list(refs.values())


def merge_timestamps(timestamps: list[tuple[int, str]], tolerance: int = 3) -> list[tuple[int, str]]:
    """Sort and merge timestamps within ``tolerance`` seconds, preferring labelled ones."""
    out: list[tuple[int, str]] = []
    for t, label in sorted(timestamps, key=lambda x: (x[0], not x[1])):
        if out and t - out[-1][0] <= tolerance:
            if label and not out[-1][1]:
                out[-1] = (out[-1][0], label)
            continue
        out.append((t, label))
    return out
