// Interval (work/rest x rounds) timer with WebAudio beeps. Uses wall-clock deltas so it stays
// accurate even when the tab is throttled.

import { storage } from './util.js';

const KEY = 'fittube.timer.v1';
const PREP = 5; // seconds of "get ready" before round 1

export class IntervalTimer {
  constructor(onTick) {
    this.onTick = onTick;
    const s = storage.get(KEY, {}) || {};
    this.config = {
      work: Number.isFinite(s.work) ? s.work : 40,
      rest: Number.isFinite(s.rest) ? s.rest : 20,
      rounds: Number.isFinite(s.rounds) ? s.rounds : 8,
      sound: s.sound !== false,
    };
    this.ctx = null;
    this.reset();
  }

  configure(patch) {
    Object.assign(this.config, patch);
    storage.set(KEY, this.config);
    if (this.phase === 'idle' || this.phase === 'done') this.reset();
  }

  /** phases: idle -> prep -> work -> rest -> work ... -> done */
  reset() {
    clearInterval(this._iv);
    this._iv = null;
    this.running = false;
    this.phase = 'idle';
    this.round = 0;
    this.remaining = this.config.work;
    this.total = this.config.work;
    this._lastWhole = null;
    this._emit();
  }

  start() {
    this._ensureAudio();
    if (this.phase === 'idle' || this.phase === 'done') {
      this.round = 1;
      this._enter('prep', PREP);
    }
    this.running = true;
    this._last = performance.now();
    clearInterval(this._iv);
    this._iv = setInterval(() => this._tick(), 100);
    this._emit();
  }

  pause() {
    this.running = false;
    clearInterval(this._iv);
    this._iv = null;
    this._emit();
  }

  toggle() {
    if (this.running) this.pause();
    else this.start();
  }

  _enter(phase, seconds) {
    this.phase = phase;
    this.total = seconds;
    this.remaining = seconds;
    this._lastWhole = Math.ceil(seconds);
  }

  _tick() {
    const now = performance.now();
    const dt = (now - this._last) / 1000;
    this._last = now;
    this.remaining -= dt;

    const whole = Math.ceil(this.remaining);
    if (whole !== this._lastWhole) {
      this._lastWhole = whole;
      if (whole > 0 && whole <= 3) this._beep(660, 0.09);
    }

    while (this.remaining <= 0 && this.running) {
      const carry = this.remaining;
      const { work, rest, rounds } = this.config;
      if (this.phase === 'prep') {
        this._enter('work', work);
        this._beep(990, 0.35);
      } else if (this.phase === 'work') {
        if (this.round >= rounds) {
          this._finish();
          return;
        }
        if (rest > 0) {
          this._enter('rest', rest);
          this._beep(440, 0.35);
        } else {
          this.round += 1;
          this._enter('work', work);
          this._beep(990, 0.35);
        }
      } else if (this.phase === 'rest') {
        this.round += 1;
        this._enter('work', work);
        this._beep(990, 0.35);
      }
      this.remaining += carry;
    }
    this._emit(true);
  }

  _finish() {
    this.pause();
    this.phase = 'done';
    this.remaining = 0;
    this._beep(880, 0.2);
    setTimeout(() => this._beep(1175, 0.45), 230);
    this._emit(true);
  }

  _ensureAudio() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    try {
      this.ctx = new AC();
    } catch {
      this.ctx = null;
    }
  }

  _beep(freq, dur) {
    if (!this.config.sound || !this.ctx) return;
    try {
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.35, t + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      osc.connect(gain).connect(this.ctx.destination);
      osc.start(t);
      osc.stop(t + dur + 0.02);
    } catch {
      /* audio is best-effort */
    }
  }

  _emit(phaseMaybeChanged = false) {
    if (this.onTick) this.onTick(this, phaseMaybeChanged);
  }
}
