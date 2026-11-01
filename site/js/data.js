// Loading + normalising site/data/videos.json (see README "Data contract").
// Every optional field is defensively defaulted so a partial/older data file never breaks the UI.

export const FACETS = [
  { key: 'goal', label: 'Goal', values: ['strength', 'hypertrophy', 'fat-loss', 'mobility', 'rehab', 'endurance', 'skill'] },
  { key: 'body', label: 'Body', values: ['full', 'upper', 'lower', 'core', 'back', 'chest', 'shoulders', 'arms', 'legs', 'glutes', 'neck', 'hips'] },
  { key: 'equipment', label: 'Equipment', values: ['none', 'dumbbell', 'barbell', 'kettlebell', 'band', 'rings', 'bar', 'machine'] },
  { key: 'level', label: 'Level', values: ['beginner', 'intermediate', 'advanced', 'any'] },
  { key: 'type', label: 'Type', values: ['tutorial', 'follow-along', 'explainer', 'program', 'motivation'] },
];

const LABELS = {
  'fat-loss': 'Fat loss',
  full: 'Full body',
  upper: 'Upper body',
  lower: 'Lower body',
  none: 'No equipment',
  band: 'Bands',
  bar: 'Pull-up bar',
  'follow-along': 'Follow-along',
  any: 'All levels',
};

export function tagLabel(value) {
  if (LABELS[value]) return LABELS[value];
  const s = String(value).replace(/-/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// Hosts whose comment permalinks may be linked from the video modal.
export const COMMENT_HOSTS = ['reddit.com', 'news.ycombinator.com', 't-nation.com'];

const SITE_NAMES = { 'reddit.com': 'Reddit', 'news.ycombinator.com': 'Hacker News', 't-nation.com': 'T-Nation' };

// Display name for a permalink or hostname on one of COMMENT_HOSTS ('' if unknown).
export function siteName(u) {
  let host = str(u).toLowerCase();
  try {
    host = new URL(host).hostname;
  } catch {
    /* already a hostname */
  }
  const key = COMMENT_HOSTS.find((d) => host === d || host.endsWith('.' + d));
  return key ? SITE_NAMES[key] : '';
}

const ID_RE = /^[A-Za-z0-9_-]{11}$/;

function str(v, fallback = '') {
  return typeof v === 'string' ? v.trim() : v == null ? fallback : String(v);
}
function num(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}
function arr(v) {
  if (Array.isArray(v)) return v;
  return v == null || v === '' ? [] : [v];
}
function strList(v) {
  return arr(v).map((x) => str(x).toLowerCase()).filter(Boolean);
}

export function safeUrl(u, hosts) {
  try {
    const url = new URL(u);
    if (url.protocol !== 'https:') return null;
    if (hosts && !hosts.some((h) => url.hostname === h || url.hostname.endsWith('.' + h))) return null;
    return url.href;
  } catch {
    return null;
  }
}

function normalizeVideo(raw, index) {
  if (!raw || typeof raw !== 'object') return null;
  const id = str(raw.id);
  if (!ID_RE.test(id)) return null;
  const tags = raw.tags && typeof raw.tags === 'object' ? raw.tags : {};
  const comments = arr(raw.comments)
    .filter((c) => c && (typeof c === 'string' || typeof c.text === 'string'))
    .map((c) => (typeof c === 'string' ? { text: c } : c))
    .map((c) => ({
      text: str(c.text).slice(0, 600),
      score: Number.isFinite(Number(c.score)) ? Number(c.score) : null,
      permalink: safeUrl(c.permalink, COMMENT_HOSTS),
    }))
    .filter((c) => c.text);
  const timestamps = arr(raw.timestamps)
    .map((t) => (typeof t === 'number' ? { t } : t))
    .filter((t) => t && Number.isFinite(Number(t.t)) && Number(t.t) >= 0)
    .map((t) => ({ t: Math.floor(Number(t.t)), label: str(t.label) }));
  // de-dupe timestamps by second, keep order by time
  const seen = new Set();
  const uniqTs = timestamps
    .sort((a, b) => a.t - b.t)
    .filter((t) => (seen.has(t.t) ? false : (seen.add(t.t), true)));

  const level = str(Array.isArray(tags.level) ? tags.level[0] : tags.level).toLowerCase();
  const type = strList(tags.type);
  const v = {
    id,
    title: str(raw.title) || 'Untitled video',
    channel: str(raw.channel),
    thumbnail: safeUrl(raw.thumbnail, ['ytimg.com', 'youtube.com', 'ggpht.com']) || `https://i.ytimg.com/vi/${id}/hqdefault.jpg`,
    recommenders: Math.max(0, Math.round(num(raw.recommenders))),
    score: num(raw.score),
    first_seen: /^\d{4}-\d{2}-\d{2}/.test(str(raw.first_seen)) ? str(raw.first_seen).slice(0, 10) : '',
    subreddits: arr(raw.subreddits).map((s) => str(s).replace(/^\/?r\//i, '')).filter(Boolean),
    comments,
    // Non-Reddit sites (Hacker News, T-Nation) that recommended the video.
    sites: [...new Set([...arr(raw.sources).map((x) => x && siteName(x.site || x.url)), ...comments.map((c) => siteName(c.permalink))])]
      .filter((n) => n && n !== 'Reddit'),
    timestamps: uniqTs,
    tags: {
      goal: strList(tags.goal),
      body: strList(tags.body),
      equipment: strList(tags.equipment),
      level: level ? [level] : [],
      type,
    },
    order: index,
  };
  v.haystack = [v.title, v.channel, v.subreddits.join(' '), ...v.comments.map((c) => c.text)]
    .join(' \n ')
    .toLowerCase();
  return v;
}

export function normalizeData(json) {
  const list = Array.isArray(json) ? json : arr(json && json.videos);
  const byId = new Map();
  list.forEach((raw, i) => {
    const v = normalizeVideo(raw, i);
    if (v && !byId.has(v.id)) byId.set(v.id, v);
  });
  return {
    generatedAt: json && !Array.isArray(json) ? str(json.generated_at) : '',
    sources: json && !Array.isArray(json) ? arr(json.sources).map((s) => str(s)).filter(Boolean) : [],
    videos: Array.from(byId.values()),
    byId,
  };
}

export async function loadData(url = 'data/videos.json') {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status})`);
  return normalizeData(await res.json());
}
