// FitTube frontend entry point. Vanilla ES modules, no build step.

import { $, $$, h, icon, fmtTime, fmtNum, plural, clampInt, isTyping } from './js/util.js';
import { FACETS, tagLabel, loadData, siteName } from './js/data.js';
import { Player } from './js/player.js';
import { Session } from './js/session.js';
import { IntervalTimer } from './js/timer.js';

const PAGE_SIZE = 36;
const SORTS = ['score', 'recommenders', 'newest'];
const MOBILE = window.matchMedia('(max-width: 719px)');

const state = {
  q: '',
  sort: 'score',
  filters: Object.fromEntries(FACETS.map((f) => [f.key, new Set()])),
};

let data = { videos: [], byId: new Map(), sources: [], generatedAt: '' };
let results = [];
let renderedCount = 0;
let selected = -1;
let current = null; // video open in modal
let sessionPlay = null; // {index} while playing the session back to back
let lastFocus = null;
let loopA = null;
let loopB = null;

const el = {
  search: $('#search'),
  sort: $('#sort'),
  filters: $('#filters'),
  filtersToggle: $('#filters-toggle'),
  filterCount: $('#filter-count'),
  summary: $('#summary'),
  grid: $('#grid'),
  status: $('#status'),
  empty: $('#empty'),
  sentinel: $('#sentinel'),
  footerMeta: $('#footer-meta'),
  modal: $('#modal'),
  modalPanel: $('#modal .modal-panel'),
  drawer: $('#drawer'),
  help: $('#help'),
  timer: $('#timer'),
  toast: $('#toast'),
  sessionCount: $('#session-count'),
};

/* ------------------------------------------------------------------ URL state */

function readUrl() {
  const p = new URLSearchParams(location.search);
  state.q = (p.get('q') || '').slice(0, 200);
  state.sort = SORTS.includes(p.get('sort')) ? p.get('sort') : 'score';
  for (const f of FACETS) {
    const vals = (p.get(f.key) || '').split(',').map((s) => s.trim().toLowerCase()).filter((v) => f.values.includes(v));
    state.filters[f.key] = new Set(vals);
  }
  return p.get('v');
}

function buildUrl({ video } = {}) {
  const p = new URLSearchParams();
  if (state.q) p.set('q', state.q);
  for (const f of FACETS) {
    const set = state.filters[f.key];
    if (set.size) p.set(f.key, f.values.filter((v) => set.has(v)).join(','));
  }
  if (state.sort !== 'score') p.set('sort', state.sort);
  if (video) p.set('v', video);
  const qs = p.toString().replace(/%2C/g, ',');
  return location.pathname + (qs ? '?' + qs : '') + location.hash;
}

function syncUrl() {
  const url = buildUrl({ video: current && current.id });
  if (url !== location.pathname + location.search + location.hash) {
    history.replaceState(history.state, '', url);
  }
}

/* ------------------------------------------------------------------ filtering */

function activeFilterCount() {
  return FACETS.reduce((n, f) => n + state.filters[f.key].size, 0);
}

function matchesFacet(v, key) {
  const set = state.filters[key];
  if (!set.size) return true;
  const vals = v.tags[key];
  if (key === 'level' && vals.includes('any') && !set.has('any')) return true; // "all levels" fits any level filter
  return vals.some((x) => set.has(x));
}

function matchesQuery(v, tokens) {
  return tokens.every((t) => v.haystack.includes(t));
}

function queryTokens() {
  return state.q.toLowerCase().split(/\s+/).filter(Boolean);
}

function compare(a, b) {
  if (state.sort === 'recommenders') return b.recommenders - a.recommenders || b.score - a.score || a.order - b.order;
  if (state.sort === 'newest') return (b.first_seen || '').localeCompare(a.first_seen || '') || b.score - a.score || a.order - b.order;
  return b.score - a.score || b.recommenders - a.recommenders || a.order - b.order;
}

function applyFilters({ keepScroll = false } = {}) {
  const tokens = queryTokens();
  const base = data.videos.filter((v) => matchesQuery(v, tokens));
  results = base.filter((v) => FACETS.every((f) => matchesFacet(v, f.key))).sort(compare);
  updateFacetCounts(base);
  renderGrid();
  renderSummary();
  syncUrl();
  if (!keepScroll && window.scrollY > el.grid.offsetTop) {
    window.scrollTo({ top: Math.max(0, el.filters.closest('.toolbar').offsetTop - 70) });
  }
}

/* ------------------------------------------------------------------ filters UI */

function buildFilters() {
  el.filters.replaceChildren();
  for (const f of FACETS) {
    const present = new Set(data.videos.flatMap((v) => v.tags[f.key]));
    const values = f.values.filter((v) => present.has(v) || state.filters[f.key].has(v));
    if (!values.length) continue;
    const chips = h('div', { class: 'chips', role: 'group', 'aria-label': f.label });
    for (const value of values) {
      chips.append(
        h('button', {
          class: 'chip', type: 'button', 'aria-pressed': 'false',
          dataset: { facet: f.key, value },
          onclick: () => toggleFilter(f.key, value),
        }, tagLabel(value), h('span', { class: 'count' }, ''))
      );
    }
    el.filters.append(h('div', { class: 'filter-group' }, h('span', { class: 'filter-label' }, f.label), chips));
  }
}

function updateFacetCounts(base) {
  for (const f of FACETS) {
    const others = base.filter((v) => FACETS.every((g) => g.key === f.key || matchesFacet(v, g.key)));
    const counts = new Map();
    for (const v of others) {
      const vals = v.tags[f.key];
      for (const x of vals) counts.set(x, (counts.get(x) || 0) + 1);
    }
    for (const chip of $$(`.chip[data-facet="${f.key}"]`, el.filters)) {
      const n = counts.get(chip.dataset.value) || 0;
      const on = state.filters[f.key].has(chip.dataset.value);
      chip.setAttribute('aria-pressed', String(on));
      chip.classList.toggle('is-zero', n === 0);
      chip.lastChild.textContent = fmtNum(n);
    }
  }
  const n = activeFilterCount();
  el.filterCount.hidden = n === 0;
  el.filterCount.textContent = n;
}

function toggleFilter(key, value) {
  const set = state.filters[key];
  if (set.has(value)) set.delete(value);
  else set.add(value);
  applyFilters({ keepScroll: true });
}

function clearAll() {
  for (const f of FACETS) state.filters[f.key].clear();
  state.q = '';
  el.search.value = '';
  applyFilters();
}

function renderSummary() {
  const total = data.videos.length;
  el.summary.replaceChildren(results.length === total ? plural(total, 'video') : `${fmtNum(results.length)} of ${plural(total, 'video')}`);
  if (activeFilterCount() || state.q) {
    el.summary.append(' · ', h('button', { type: 'button', onclick: clearAll }, 'Clear filters'));
  }
}

function setFiltersOpen(open) {
  el.filters.hidden = !open;
  el.filtersToggle.setAttribute('aria-expanded', String(open));
}

/* ------------------------------------------------------------------ cards */

function cardTags(v) {
  const t = [];
  if (v.tags.level[0]) t.push([v.tags.level[0], 'tag tag-level']);
  for (const g of v.tags.goal.slice(0, 2)) t.push([g, 'tag']);
  if (v.tags.type[0]) t.push([v.tags.type[0], 'tag']);
  if (v.tags.equipment[0]) t.push([v.tags.equipment[0], 'tag']);
  return t.slice(0, 4).map(([val, cls]) => h('span', { class: cls }, tagLabel(val)));
}

function thumbImg(v, cls) {
  const img = h('img', { class: cls, src: v.thumbnail, alt: '', loading: 'lazy', decoding: 'async', width: 480, height: 270 });
  img.addEventListener('error', () => {
    const fallback = `https://i.ytimg.com/vi/${v.id}/mqdefault.jpg`;
    if (img.src !== fallback) img.src = fallback;
    else img.style.visibility = 'hidden';
  }, { passive: true });
  return img;
}

function recsEl(n) {
  return h('span', { class: 'recs', title: `${n} distinct people (Reddit, Hacker News, T-Nation) recommended this` }, icon('people'), plural(n, 'recommender'));
}

function renderCard(v, index) {
  const inSession = session.has(v.id);
  const meta = [recsEl(v.recommenders)];
  if (v.subreddits[0]) {
    meta.push(h('span', { class: 'dot', 'aria-hidden': 'true' }, '·'));
    meta.push(h('span', { class: 'card-subs', title: v.subreddits.map((s) => 'r/' + s).join(', ') }, `r/${v.subreddits[0]}${v.subreddits.length > 1 ? ` +${v.subreddits.length - 1}` : ''}`));
  } else if (v.sites[0]) {
    meta.push(h('span', { class: 'dot', 'aria-hidden': 'true' }, '·'));
    meta.push(h('span', { class: 'card-subs' }, v.sites.join(', ')));
  }
  const thumb = h('div', { class: 'thumb' },
    h('div', { class: 'thumb-placeholder', 'aria-hidden': 'true' }, icon('video')),
    thumbImg(v),
    state.sort === 'score' && !state.q && !activeFilterCount() ? h('span', { class: 'thumb-rank' }, `#${index + 1}`) : null,
    v.timestamps.length ? h('span', { class: 'thumb-badge' }, icon('clock'), plural(v.timestamps.length, 'timestamp')) : null,
  );
  const link = h('a', { class: 'card-link', href: buildUrl({ video: v.id }), dataset: { id: v.id } },
    thumb,
    h('div', { class: 'card-body' },
      h('h3', { class: 'card-title', title: v.title }, v.title),
      v.channel ? h('p', { class: 'card-channel' }, v.channel) : null,
      v.summary ? h('p', { class: 'card-summary' }, v.summary) : null,
      h('p', { class: 'card-meta' }, meta),
      h('div', { class: 'card-tags' }, cardTags(v)),
    ),
  );
  link.addEventListener('click', (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    select(index, { focus: false });
    sessionPlay = null;
    openVideo(v.id, { push: true });
  });
  link.addEventListener('focus', () => select(index, { focus: false, scroll: false }));
  const add = h('button', {
    class: 'card-add', type: 'button', 'aria-pressed': String(inSession),
    'aria-label': inSession ? `Remove “${v.title}” from session` : `Add “${v.title}” to session`,
    title: inSession ? 'Remove from session' : 'Add to session',
    onclick: () => toggleSession(v.id),
  }, icon(inSession ? 'check' : 'plus'));
  return h('article', { class: 'card', dataset: { id: v.id, index } }, link, add);
}

function refreshCardSessionState(id) {
  for (const card of $$(`.card[data-id="${CSS.escape(id)}"]`, el.grid)) {
    const v = data.byId.get(id);
    const on = session.has(id);
    const btn = card.querySelector('.card-add');
    btn.setAttribute('aria-pressed', String(on));
    btn.title = on ? 'Remove from session' : 'Add to session';
    btn.setAttribute('aria-label', on ? `Remove “${v.title}” from session` : `Add “${v.title}” to session`);
    btn.replaceChildren(icon(on ? 'check' : 'plus'));
  }
}

function renderGrid() {
  el.status.textContent = '';
  el.grid.replaceChildren();
  renderedCount = 0;
  selected = -1;
  el.empty.hidden = results.length > 0 || data.videos.length === 0;
  renderMore();
  // Re-observe so a sentinel that is already on screen fires again for the new result set.
  moreObserver.unobserve(el.sentinel);
  moreObserver.observe(el.sentinel);
}

function renderMore() {
  if (renderedCount >= results.length) return;
  const frag = document.createDocumentFragment();
  const end = Math.min(results.length, renderedCount + PAGE_SIZE);
  for (let i = renderedCount; i < end; i++) frag.append(renderCard(results[i], i));
  el.grid.append(frag);
  renderedCount = end;
}

const moreObserver = new IntersectionObserver((entries) => {
  if (entries.some((e) => e.isIntersecting)) renderMore();
}, { rootMargin: '800px 0px' });

function renderSkeleton() {
  const frag = document.createDocumentFragment();
  for (let i = 0; i < 8; i++) {
    frag.append(h('div', { class: 'card skeleton', 'aria-hidden': 'true' },
      h('div', { class: 'thumb' }), h('div', { class: 'sk-line', style: 'width:90%' }), h('div', { class: 'sk-line', style: 'width:55%' })));
  }
  el.grid.replaceChildren(frag);
}

/* ------------------------------------------------------------------ selection (j/k) */

function cards() {
  return $$('.card:not(.skeleton)', el.grid);
}

function select(index, { focus = true, scroll = true } = {}) {
  if (!results.length) return;
  index = Math.max(0, Math.min(results.length - 1, index));
  while (index >= renderedCount) renderMore();
  const list = cards();
  if (list[selected]) list[selected].classList.remove('is-selected');
  selected = index;
  const card = list[selected];
  if (!card) return;
  card.classList.add('is-selected');
  if (focus) card.querySelector('.card-link').focus({ preventScroll: true });
  if (scroll) card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  if (index >= renderedCount - 4) renderMore();
}

/* ------------------------------------------------------------------ modal */

const player = new Player($('#player'), {
  onEnded: () => {
    if (sessionPlay && sessionPlay.index < session.ids.length - 1) playSessionAt(sessionPlay.index + 1);
  },
  onLoop: () => renderLoopState(),
});

function setInert(on) {
  for (const node of [$('.topbar'), $('.hero'), $('.toolbar'), $('.main'), $('.footer')]) {
    if (on) node.setAttribute('inert', '');
    else node.removeAttribute('inert');
  }
}

function openVideo(id, { push = false, start = 0 } = {}) {
  const v = data.byId.get(id);
  if (!v) return;
  const wasOpen = !el.modal.hidden;
  current = v;
  if (!wasOpen) {
    lastFocus = document.activeElement;
    el.modal.hidden = false;
    document.body.classList.add('no-scroll');
    setInert(true);
    closeDrawer();
  }
  const url = buildUrl({ video: id });
  if (push && !wasOpen) history.pushState({ modal: true }, '', url);
  else history.replaceState(history.state, '', url);

  fillModal(v);
  el.modalPanel.scrollTop = 0;
  el.modal.scrollTop = 0;
  if (!wasOpen) el.modalPanel.focus({ preventScroll: true });
  document.title = `${v.title} · FitTube`;

  $('#player-fallback').hidden = true;
  player.load(v.id, start).catch(() => showPlayerFallback(v));
}

function showPlayerFallback(v) {
  const fb = $('#player-fallback');
  fb.style.backgroundImage = `url("${v.thumbnail}")`;
  fb.replaceChildren(h('div', {},
    h('p', {}, 'The YouTube player could not be loaded.'),
    h('a', { class: 'btn btn-primary', href: `https://www.youtube.com/watch?v=${v.id}`, target: '_blank', rel: 'noopener' }, icon('play'), 'Watch on YouTube')));
  fb.hidden = false;
}

function closeVideo({ fromHistory = false } = {}) {
  if (el.modal.hidden) return;
  player.pause();
  player.clearLoop();
  el.modal.hidden = true;
  document.body.classList.remove('no-scroll');
  setInert(false);
  current = null;
  sessionPlay = null;
  document.title = 'FitTube · Fitness videos people actually recommend';
  if (!fromHistory) {
    if (history.state && history.state.modal) history.back();
    else history.replaceState(null, '', buildUrl());
  }
  if (lastFocus && document.contains(lastFocus)) lastFocus.focus({ preventScroll: true });
}

function fillModal(v) {
  $('#m-title').textContent = v.title;
  const meta = [];
  if (v.channel) meta.push(h('span', {}, v.channel));
  meta.push(recsEl(v.recommenders));
  const via = [...v.subreddits.map((s) => `r/${s}`), ...v.sites];
  if (via.length) meta.push(h('span', {}, via.join(', ')));
  if (v.first_seen) meta.push(h('span', {}, `First recommended ${v.first_seen.slice(0, 4)}`));
  $('#m-meta').replaceChildren(...meta);
  $('#m-summary').textContent = v.summary;
  $('#m-summary').hidden = !v.summary;
  renderSources(v);

  $('#m-yt').href = `https://www.youtube.com/watch?v=${v.id}`;
  renderAddButton();

  const tags = [];
  for (const f of FACETS) for (const val of v.tags[f.key]) tags.push(h('span', { class: f.key === 'level' ? 'tag tag-level' : 'tag', title: f.label }, tagLabel(val)));
  $('#m-tags').replaceChildren(...tags);
  $('#m-tags').hidden = tags.length === 0;

  renderTimestamps(v);
  loopA = loopB = null;
  renderLoopState();

  const list = $('#m-comments');
  if (v.comments.length) {
    list.replaceChildren(...v.comments.map((c) => h('li', { class: 'comment' },
      h('p', {}, c.text),
      h('div', { class: 'comment-foot' },
        c.score != null ? h('span', { class: 'comment-score', title: siteName(c.permalink) ? `${siteName(c.permalink)} score` : 'Score' }, icon('arrowUp'), fmtNum(c.score)) : null,
        c.permalink ? h('a', { href: c.permalink, target: '_blank', rel: 'noopener nofollow' }, `View on ${siteName(c.permalink)}`) : null,
      ))));
  } else {
    list.replaceChildren(h('li', { class: 'comment-empty' }, `Recommended by ${plural(v.recommenders, 'person', 'people')}; no comment excerpts saved.`));
  }

  renderSessionNav();
}

// Source badges for Reddit wiki listings and posts (issue #5).
function sourceBadge(s) {
  let text;
  if (s.kind === 'wiki') text = `\u{1F4D6} r/${s.subreddit} wiki`;
  else if (s.kind === 'post') text = `\u{1F4DD} r/${s.subreddit} post${s.score != null ? ` \u00B7 ${plural(s.score, 'point')}` : ''}`;
  else return null; // comments: summarised by the subreddit list; other sites: shown in the meta line
  const title = s.kind === 'wiki' && s.path ? `Listed in r/${s.subreddit}${s.path}` : text;
  return s.url
    ? h('a', { class: `tag source-badge source-${s.kind}`, href: s.url, target: '_blank', rel: 'noopener nofollow', title }, text)
    : h('span', { class: `tag source-badge source-${s.kind}`, title }, text);
}

function renderSources(v) {
  const seen = new Set();
  const badges = v.sources
    .filter((s) => {
      const key = s.kind === 'wiki' ? `wiki:${s.subreddit.toLowerCase()}` : s.kind === 'post' ? s.url : `${s.kind}:${s.site}`;
      return seen.has(key) ? false : (seen.add(key), true);
    })
    .map(sourceBadge)
    .filter(Boolean)
    .slice(0, 8);
  $('#m-sources').replaceChildren(...badges);
  $('#m-sources').hidden = badges.length === 0;
}

function renderTimestamps(v) {
  const section = $('#m-ts-section');
  section.hidden = v.timestamps.length === 0;
  $('#m-timestamps').replaceChildren(...v.timestamps.map((ts) => h('li', { class: 'ts-item' },
    h('button', { class: 'ts-jump', type: 'button', onclick: () => player.seek(ts.t) },
      h('span', { class: 'mono' }, fmtTime(ts.t)), h('span', { class: 'ts-label' }, ts.label || 'Jump here')),
    h('button', {
      class: 'btn btn-soft btn-sm ts-loop', type: 'button', 'aria-pressed': 'false', dataset: { t: ts.t },
      title: 'Loop this moment for the chosen loop length',
      onclick: () => loopFrom(ts.t),
    }, icon('repeat'), 'Loop'),
  )));
}

function loopLen() {
  const input = $('#loop-len');
  const n = clampInt(input.value, 2, 600, 10);
  input.value = n;
  return n;
}

function loopFrom(t) {
  if (player.loop && player.loop.a === t) {
    player.clearLoop();
    return;
  }
  loopA = t;
  loopB = t + loopLen();
  player.setLoop(loopA, loopB);
  player.seek(loopA);
}

function renderLoopState() {
  $('#loop-a-val').textContent = loopA == null ? '--:--' : fmtTime(loopA);
  $('#loop-b-val').textContent = loopB == null ? '--:--' : fmtTime(loopB);
  const on = !!player.loop;
  const toggle = $('#loop-toggle');
  toggle.setAttribute('aria-pressed', String(on));
  toggle.textContent = on ? 'Looping' : 'Loop off';
  for (const b of $$('.ts-loop')) b.setAttribute('aria-pressed', String(on && Number(b.dataset.t) === player.loop.a));
  $('#loop-status').textContent = on
    ? `Looping ${fmtTime(player.loop.a)} → ${fmtTime(player.loop.b)} (${Math.round(player.loop.b - player.loop.a)} s)`
    : loopA != null && loopB == null ? `A set at ${fmtTime(loopA)}. Set B, or press Loop to repeat ${loopLen()} s.` : '';
}

$('#loop-a').addEventListener('click', () => {
  loopA = Math.floor(player.time());
  if (loopB != null && loopB <= loopA) loopB = null;
  if (player.loop) player.clearLoop();
  renderLoopState();
});
$('#loop-b').addEventListener('click', () => {
  const t = Math.ceil(player.time());
  if (loopA == null) loopA = Math.max(0, t - loopLen());
  loopB = t > loopA ? t : loopA + loopLen();
  player.setLoop(loopA, loopB);
  player.seek(loopA);
});
$('#loop-toggle').addEventListener('click', () => {
  if (player.loop) {
    player.clearLoop();
    return;
  }
  if (loopA == null) loopA = Math.floor(player.time());
  if (loopB == null || loopB <= loopA) loopB = loopA + loopLen();
  player.setLoop(loopA, loopB);
  player.seek(loopA);
});

function renderAddButton() {
  if (!current) return;
  const on = session.has(current.id);
  $('#m-add').replaceChildren(icon(on ? 'check' : 'plus'), on ? 'In my session' : 'Add to session');
  $('#m-add').classList.toggle('btn-primary', !on);
  $('#m-add').classList.toggle('btn-soft', on);
}
$('#m-add').addEventListener('click', () => current && toggleSession(current.id));
$('#m-timer').addEventListener('click', () => showTimer(true));

el.modal.addEventListener('click', (e) => {
  if (e.target.closest('[data-close]')) closeVideo();
});

window.addEventListener('popstate', () => {
  const v = readUrl();
  el.search.value = state.q;
  el.sort.value = state.sort;
  applyFilters({ keepScroll: true });
  if (v && data.byId.has(v)) openVideo(v);
  else closeVideo({ fromHistory: true });
});

/* ------------------------------------------------------------------ session */

const session = new Session(() => {
  renderSession();
  renderAddButton();
});

function toggleSession(id) {
  const v = data.byId.get(id);
  const added = session.toggle(id);
  refreshCardSessionState(id);
  toast(added ? `Added to session · ${session.ids.length} ${session.ids.length === 1 ? 'video' : 'videos'}` : `Removed “${v ? v.title : 'video'}”`);
}

function renderSession() {
  const n = session.ids.length;
  el.sessionCount.hidden = n === 0;
  el.sessionCount.textContent = n;
  $('#drawer-sub').textContent = n ? plural(n, 'video') : '';
  $('#session-empty').hidden = n > 0;
  $('#session-play').disabled = n === 0;
  $('#session-clear').hidden = n === 0;
  const list = $('#session-list');
  list.replaceChildren(...session.ids.map((id, i) => {
    const v = data.byId.get(id);
    if (!v) return null;
    return h('li', { class: 'session-item' + (sessionPlay && sessionPlay.index === i ? ' is-playing' : '') },
      h('span', { class: 'session-idx' }, i + 1),
      h('img', { class: 'session-thumb', src: `https://i.ytimg.com/vi/${v.id}/mqdefault.jpg`, alt: '', loading: 'lazy' }),
      h('button', { class: 'session-open', type: 'button', onclick: () => playSessionAt(i) },
        h('strong', {}, v.title), h('span', {}, v.channel || plural(v.recommenders, 'recommender'))),
      h('div', { class: 'session-btns' },
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Move up', disabled: i === 0, onclick: () => session.move(id, -1) }, icon('up')),
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Remove', onclick: () => { session.remove(id); refreshCardSessionState(id); } }, icon('x')),
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Move down', disabled: i === n - 1, onclick: () => session.move(id, 1) }, icon('down')),
      ));
  }));
  renderSessionNav();
}

function renderSessionNav() {
  const nav = $('#session-nav');
  if (!sessionPlay || !current) {
    nav.hidden = true;
    return;
  }
  const i = session.ids.indexOf(current.id);
  if (i < 0) {
    nav.hidden = true;
    return;
  }
  sessionPlay.index = i;
  nav.hidden = false;
  $('#sn-label').textContent = `My session · ${i + 1} of ${session.ids.length}`;
  $('#sn-prev').disabled = i === 0;
  $('#sn-next').disabled = i >= session.ids.length - 1;
}

function playSessionAt(i) {
  const id = session.ids[i];
  if (!id) return;
  sessionPlay = { index: i };
  openVideo(id, { push: el.modal.hidden });
  renderSession();
}

$('#sn-prev').addEventListener('click', () => sessionPlay && playSessionAt(sessionPlay.index - 1));
$('#sn-next').addEventListener('click', () => sessionPlay && playSessionAt(sessionPlay.index + 1));
$('#session-play').addEventListener('click', () => playSessionAt(0));
$('#session-clear').addEventListener('click', () => {
  const ids = [...session.ids];
  session.clear();
  ids.forEach(refreshCardSessionState);
  toast('Session cleared');
});

function openDrawer() {
  el.drawer.hidden = false;
  $('#session-btn').setAttribute('aria-expanded', 'true');
  $('.drawer-panel').focus({ preventScroll: true });
}
function closeDrawer() {
  if (el.drawer.hidden) return;
  el.drawer.hidden = true;
  $('#session-btn').setAttribute('aria-expanded', 'false');
}
$('#session-btn').addEventListener('click', () => (el.drawer.hidden ? openDrawer() : closeDrawer()));
el.drawer.addEventListener('click', (e) => {
  if (e.target.closest('[data-close-drawer]')) {
    closeDrawer();
    $('#session-btn').focus();
  }
});

/* ------------------------------------------------------------------ timer */

const timerUi = {
  display: $('.timer-display'),
  phase: $('#timer-phase'),
  clock: $('#timer-clock'),
  round: $('#timer-round'),
  bar: $('#timer-bar'),
  start: $('#timer-start'),
};
let lastPhase = null;

const timer = new IntervalTimer((t) => {
  const { work, rest, rounds } = t.config;
  const phaseNames = { idle: 'Ready', prep: 'Get ready', work: 'Work', rest: 'Rest', done: 'Done' };
  timerUi.display.dataset.phase = t.phase;
  timerUi.phase.textContent = phaseNames[t.phase];
  timerUi.clock.textContent = fmtTime(Math.ceil(Math.max(0, t.remaining)));
  timerUi.round.textContent = t.phase === 'idle'
    ? `${rounds} × ${work}s / ${rest}s`
    : t.phase === 'done' ? `${plural(rounds, 'round')} complete` : `Round ${t.round} of ${rounds}`;
  const pct = t.phase === 'idle' ? 0 : t.phase === 'done' ? 100 : (1 - t.remaining / t.total) * 100;
  timerUi.bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  timerUi.start.textContent = t.running ? 'Pause' : t.phase === 'idle' ? 'Start' : t.phase === 'done' ? 'Again' : 'Resume';
  if (lastPhase !== t.phase && (t.phase === 'work' || t.phase === 'rest' || t.phase === 'done')) {
    el.timer.classList.remove('is-flash');
    void el.timer.offsetWidth;
    el.timer.classList.add('is-flash');
  }
  lastPhase = t.phase;
});

function syncTimerInputs() {
  $('#t-work').value = timer.config.work;
  $('#t-rest').value = timer.config.rest;
  $('#t-rounds').value = timer.config.rounds;
  $('#t-sound').checked = timer.config.sound;
}
syncTimerInputs();

$('#timer-settings').addEventListener('change', () => {
  timer.configure({
    work: clampInt($('#t-work').value, 1, 3600, 40),
    rest: clampInt($('#t-rest').value, 0, 3600, 20),
    rounds: clampInt($('#t-rounds').value, 1, 99, 8),
    sound: $('#t-sound').checked,
  });
  syncTimerInputs();
});
$('#timer-settings').addEventListener('submit', (e) => e.preventDefault());
$('#timer-start').addEventListener('click', () => timer.toggle());
$('#timer-reset').addEventListener('click', () => timer.reset());
$('#timer-settings-btn').addEventListener('click', () => {
  const s = $('#timer-settings');
  s.hidden = !s.hidden;
  $('#timer-settings-btn').setAttribute('aria-expanded', String(!s.hidden));
});

function showTimer(show) {
  el.timer.hidden = !show;
  document.body.classList.toggle('timer-open', show);
  $('#timer-btn').setAttribute('aria-pressed', String(show));
  if (!show) timer.pause();
}
$('#timer-btn').addEventListener('click', () => showTimer(el.timer.hidden));
$('#timer-close').addEventListener('click', () => showTimer(false));

/* ------------------------------------------------------------------ help, toast, shuffle */

function openHelp() {
  el.help.hidden = false;
  $('.help-panel').focus({ preventScroll: true });
}
function closeHelp() {
  el.help.hidden = true;
}
$('#help-btn').addEventListener('click', openHelp);
el.help.addEventListener('click', (e) => {
  if (e.target.closest('[data-close-help]')) closeHelp();
});

let toastTimer = null;
function toast(msg) {
  el.toast.textContent = msg;
  el.toast.hidden = false;
  el.toast.style.animation = 'none';
  void el.toast.offsetWidth;
  el.toast.style.animation = '';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.toast.hidden = true), 2200);
}

/** Pick a random video from the top half of the current results (or everything if nothing matches). */
function shuffle() {
  const pool = (results.length ? results : [...data.videos].sort(compare));
  if (!pool.length) return;
  const top = pool.slice(0, Math.max(1, Math.ceil(pool.length / 2)));
  let pick = top[Math.floor(Math.random() * top.length)];
  if (current && top.length > 1 && pick.id === current.id) pick = top[(top.indexOf(pick) + 1) % top.length];
  sessionPlay = null;
  openVideo(pick.id, { push: el.modal.hidden });
}
$('#shuffle-btn').addEventListener('click', shuffle);

/* ------------------------------------------------------------------ inputs */

let searchTimer = null;
el.search.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.q = el.search.value.trim();
    applyFilters({ keepScroll: true });
  }, 120);
});
el.search.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    clearTimeout(searchTimer);
    state.q = el.search.value.trim();
    applyFilters({ keepScroll: true });
    if (results.length) select(0);
  }
});
el.sort.addEventListener('change', () => {
  state.sort = SORTS.includes(el.sort.value) ? el.sort.value : 'score';
  applyFilters();
});
el.filtersToggle.addEventListener('click', () => setFiltersOpen(el.filters.hidden));
$('#empty-clear').addEventListener('click', clearAll);

/* ------------------------------------------------------------------ keyboard */

document.addEventListener('keydown', (e) => {
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key;

  if (key === 'Escape') {
    if (!el.help.hidden) return closeHelp();
    if (isTyping(e.target)) {
      if (e.target === el.search) el.search.blur();
      return;
    }
    if (!el.modal.hidden) return closeVideo();
    if (!el.drawer.hidden) return closeDrawer();
    return;
  }
  if (isTyping(e.target)) return;

  if (!el.help.hidden) {
    if (key === '?') closeHelp();
    return;
  }

  const modalOpen = !el.modal.hidden;
  switch (key) {
    case '?':
      e.preventDefault();
      openHelp();
      return;
    case '/':
      e.preventDefault();
      if (modalOpen) closeVideo();
      closeDrawer();
      el.search.focus();
      el.search.select();
      return;
    case 't':
      showTimer(el.timer.hidden);
      return;
    case 'm':
      if (!modalOpen) el.drawer.hidden ? openDrawer() : closeDrawer();
      return;
    case 'r':
      shuffle();
      return;
    case 'a': {
      const target = modalOpen ? current : results[selected];
      if (target) toggleSession(target.id);
      return;
    }
  }

  if (modalOpen) {
    if (key === ' ') {
      e.preventDefault();
      player.toggle();
    }
    return;
  }
  if (!el.drawer.hidden) return;

  if (key === 'j') {
    select(selected < 0 ? 0 : selected + 1);
  } else if (key === 'k') {
    select(selected < 0 ? 0 : selected - 1);
  } else if (key === 'Enter' && selected >= 0 && !e.target.closest('a, button')) {
    e.preventDefault();
    sessionPlay = null;
    openVideo(results[selected].id, { push: true });
  }
});

/* ------------------------------------------------------------------ boot */

function renderFooter() {
  const parts = [`${plural(data.videos.length, 'video')}`];
  if (data.sources.length) parts.push(`from ${data.sources.map((s) => (s.includes('.') ? s : 'r/' + s)).join(', ')}`);
  let text = parts.join(' ');
  const d = data.generatedAt ? new Date(data.generatedAt) : null;
  if (d && !Number.isNaN(d.getTime())) text += ` · updated ${d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })}`;
  el.footerMeta.textContent = text;
}

async function boot() {
  const initialVideo = readUrl();
  el.search.value = state.q;
  el.sort.value = state.sort;
  setFiltersOpen(!MOBILE.matches);
  renderSkeleton();
  renderSession();

  try {
    data = await loadData();
  } catch (err) {
    console.warn(err);
    el.grid.replaceChildren();
    el.status.replaceChildren(
      h('p', {}, 'Couldn’t load the video list. Check your connection and try again.'),
      h('button', { class: 'btn btn-primary', type: 'button', onclick: () => location.reload() }, 'Retry'));
    return;
  }

  session.prune(new Set(data.byId.keys()));
  buildFilters();
  renderFooter();
  applyFilters({ keepScroll: true });
  renderSession();
  if (!data.videos.length) el.status.textContent = 'No videos yet. Check back soon.';
  if (initialVideo && data.byId.has(initialVideo)) openVideo(initialVideo);
}

boot();
