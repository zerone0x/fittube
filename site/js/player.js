// Thin wrapper around the YouTube IFrame Player API with A/B segment looping.

let apiPromise = null;

function loadApi() {
  if (window.YT && window.YT.Player) return Promise.resolve(window.YT);
  if (apiPromise) return apiPromise;
  apiPromise = new Promise((resolve, reject) => {
    const prev = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      if (typeof prev === 'function') prev();
      resolve(window.YT);
    };
    const s = document.createElement('script');
    s.src = 'https://www.youtube.com/iframe_api';
    s.async = true;
    s.onerror = () => {
      apiPromise = null;
      s.remove();
      reject(new Error('YouTube player could not be loaded'));
    };
    document.head.append(s);
    setTimeout(() => {
      if (window.YT && window.YT.Player) return;
      apiPromise = null;
      reject(new Error('YouTube player timed out'));
    }, 15000);
  });
  return apiPromise;
}

export class Player {
  /**
   * @param {HTMLElement} mount element that will be replaced by the iframe
   * @param {{onState?: (state:number)=>void, onEnded?: ()=>void, onLoop?: (loop)=>void}} hooks
   */
  constructor(mount, hooks = {}) {
    this.mount = mount;
    this.hooks = hooks;
    this.yt = null;
    this.ready = null;
    this.videoId = null;
    this.loop = null; // {a, b}
    this.playing = false;
    this._timer = null;
  }

  async load(videoId, start = 0) {
    this.videoId = videoId;
    this.clearLoop();
    const YT = await loadApi();
    if (this.videoId !== videoId) return; // superseded while loading
    if (!this.yt) {
      this.ready = new Promise((resolve) => {
        this.yt = new YT.Player(this.mount, {
          videoId,
          host: 'https://www.youtube-nocookie.com',
          playerVars: { autoplay: 1, playsinline: 1, rel: 0, modestbranding: 1, start: Math.floor(start) || 0, origin: location.origin },
          events: {
            onReady: () => resolve(),
            onStateChange: (e) => this._onState(e.data),
          },
        });
      });
      await this.ready;
    } else {
      await this.ready;
      this.yt.loadVideoById({ videoId, startSeconds: Math.floor(start) || 0 });
    }
  }

  _onState(state) {
    // -1 unstarted, 0 ended, 1 playing, 2 paused, 3 buffering, 5 cued
    this.playing = state === 1;
    if (this.hooks.onState) this.hooks.onState(state);
    if (state === 0) {
      if (this.loop) this.seek(this.loop.a, true);
      else if (this.hooks.onEnded) this.hooks.onEnded();
    }
  }

  _api() {
    return this.yt && typeof this.yt.getCurrentTime === 'function' ? this.yt : null;
  }

  time() {
    const p = this._api();
    return p ? p.getCurrentTime() || 0 : 0;
  }

  duration() {
    const p = this._api();
    return p ? p.getDuration() || 0 : 0;
  }

  seek(t, play = true) {
    const p = this._api();
    if (!p) return;
    p.seekTo(Math.max(0, t), true);
    if (play) p.playVideo();
  }

  toggle() {
    const p = this._api();
    if (!p) return;
    if (p.getPlayerState() === 1) p.pauseVideo();
    else p.playVideo();
  }

  pause() {
    const p = this._api();
    if (p) p.pauseVideo();
  }

  setLoop(a, b) {
    if (!(b > a)) return false;
    this.loop = { a, b };
    clearInterval(this._timer);
    this._timer = setInterval(() => {
      if (!this.loop) return;
      const t = this.time();
      if (t >= this.loop.b || t < this.loop.a - 1) this.seek(this.loop.a, true);
    }, 150);
    if (this.hooks.onLoop) this.hooks.onLoop(this.loop);
    return true;
  }

  clearLoop() {
    clearInterval(this._timer);
    this._timer = null;
    const had = !!this.loop;
    this.loop = null;
    if (had && this.hooks.onLoop) this.hooks.onLoop(null);
  }
}
