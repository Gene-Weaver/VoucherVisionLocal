// Shared state and a tiny event bus, so tabs can tell each other that data
// changed without reaching into each other's DOM.
//
// Events:
//   providers-changed   a key was saved, removed or tested (Keys tab)
//   models-changed      /v1/models data changed (e.g. a provider model list was loaded)
//   prompts-changed     a prompt was imported or deleted
//   use-prompt          {source, name}: select this prompt on the VVL tab
//   goto                {tab, target}: switch tabs (handled by app.js)
//   selection-changed   {ocr, parse}: the model choice on the VVL tab was saved
//   jobs-changed        the list of download / install jobs (jobs.js)
//   job-finished        one job that just ended (jobs.js)
//   engine-changed      the engine was installed, removed, started or stopped, or its settings changed

import { api } from './api.js';

export const bus = new EventTarget();

export function emit(name, detail = null) {
    bus.dispatchEvent(new CustomEvent(name, { detail }));
}

export function on(name, fn) {
    const handler = (ev) => fn(ev.detail);
    bus.addEventListener(name, handler);
    return () => bus.removeEventListener(name, handler);
}

// Cached GET /v1/app (paths, features, feature notes). Fetched once by app.js.
let appInfo = null;
export function setAppInfo(info) {
    appInfo = info;
}
export function getAppInfo() {
    return appInfo;
}

/** Feature flag from /v1/app (false when unknown). */
export function feature(name) {
    return Boolean(appInfo && appInfo.features && appInfo.features[name]);
}

/**
 * A reason the server gives for something being unavailable, as shown to the
 * user. Notes that talk about development stages are replaced by a neutral line.
 */
export function plainReason(text, fallback = 'Not available yet.') {
    if (typeof text !== 'string' || !text.trim()) return null;
    return /\bphase\b/i.test(text) ? fallback : text;
}

/** Human note for a disabled feature, if the backend sent one. */
export function featureNote(name) {
    return plainReason((appInfo && appInfo.feature_notes && appInfo.feature_notes[name]) || null);
}

// GET /v1/settings, shared by the VVL tab (enrichment) and the Engine tab (models folder).
let settingsPromise = null;
export function loadSettings({ force = false } = {}) {
    if (force || !settingsPromise) {
        settingsPromise = api.get('/v1/settings').catch((err) => {
            settingsPromise = null;
            throw err;
        });
    }
    return settingsPromise;
}

export function updateCachedSettings(settings) {
    settingsPromise = Promise.resolve(settings);
}

/** Request a tab switch; app.js owns the tab bar. */
export function gotoTab(tab, target = null) {
    emit('goto', { tab, target });
}
