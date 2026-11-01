// "My session": an ordered playlist of video ids persisted in localStorage.

import { storage } from './util.js';

const KEY = 'fittube.session.v1';

export class Session {
  constructor(onChange) {
    this.onChange = onChange || (() => {});
    const saved = storage.get(KEY, []);
    this.ids = Array.isArray(saved) ? saved.filter((x) => typeof x === 'string') : [];
    window.addEventListener('storage', (e) => {
      if (e.key !== KEY) return;
      const next = storage.get(KEY, []);
      this.ids = Array.isArray(next) ? next.filter((x) => typeof x === 'string') : [];
      this.onChange();
    });
  }

  /** Drop ids that no longer exist in the current data file. */
  prune(validIds) {
    const before = this.ids.length;
    this.ids = this.ids.filter((id) => validIds.has(id));
    if (this.ids.length !== before) this._save();
  }

  has(id) {
    return this.ids.includes(id);
  }

  toggle(id) {
    if (this.has(id)) this.remove(id);
    else this.add(id);
    return this.has(id);
  }

  add(id) {
    if (!this.has(id)) {
      this.ids.push(id);
      this._save();
    }
  }

  remove(id) {
    const i = this.ids.indexOf(id);
    if (i >= 0) {
      this.ids.splice(i, 1);
      this._save();
    }
  }

  move(id, delta) {
    const i = this.ids.indexOf(id);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= this.ids.length) return;
    [this.ids[i], this.ids[j]] = [this.ids[j], this.ids[i]];
    this._save();
  }

  clear() {
    this.ids = [];
    this._save();
  }

  _save() {
    storage.set(KEY, this.ids);
    this.onChange();
  }
}
