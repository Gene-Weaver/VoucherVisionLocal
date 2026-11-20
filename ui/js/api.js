// The one HTTP client for the VoucherVisionLocal UI.
//
// - Session token: read once from ?token= in the address bar, kept in
//   sessionStorage only, and removed from the address bar with
//   history.replaceState (docs/API.md section 2, rule 4). It is sent as
//   X-VVL-Token on every request and never logged or written into the DOM.
// - Every failure becomes an ApiError, built from the backend's error
//   envelope ({error, error_code, message, request_id, details}) when there
//   is one.
// - Toasts and inline error boxes live here too, so every tab reports
//   problems the same way.

import { el, icon, stateBox, fmtBytes } from './dom.js';

const TOKEN_KEY = 'vvl.session-token';
let token = null;
let authFailedNotified = false;
let inflight = 0;

/** Pull ?token= out of the URL into sessionStorage, then strip it from the address bar. */
export function initToken() {
    const url = new URL(window.location.href);
    const fromUrl = url.searchParams.get('token');
    if (fromUrl) {
        try {
            window.sessionStorage.setItem(TOKEN_KEY, fromUrl);
        } catch (_) { /* storage blocked: keep it in memory for this page only */ }
        token = fromUrl;
        url.searchParams.delete('token');
        const clean = url.pathname + (url.searchParams.toString() ? `?${url.searchParams}` : '') + url.hash;
        window.history.replaceState(null, '', clean);
    } else {
        try {
            token = window.sessionStorage.getItem(TOKEN_KEY);
        } catch (_) {
            token = null;
        }
    }
    return Boolean(token);
}

export function hasToken() {
    return Boolean(token);
}

/** Number of requests in flight; the desktop screenshot run waits for this to reach 0. */
export function inflightCount() {
    return inflight;
}

export class ApiError extends Error {
    constructor({ status = 0, code = null, error = null, message = null, details = null, requestId = null, network = false, aborted = false } = {}) {
        super(error || message || (network ? 'Could not reach the VoucherVisionLocal server.' : `Request failed (HTTP ${status})`));
        this.name = 'ApiError';
        this.status = status;
        this.code = code;
        this.error = error;
        this.detailMessage = message;
        this.details = details;
        this.requestId = requestId;
        this.network = network;
        this.aborted = aborted;
    }

    /** One readable line, in the style of VoucherVisionGO's describeApiError(). */
    describe() {
        if (this.aborted) return 'The request took too long and was stopped.';
        if (this.network) return 'Could not reach the VoucherVisionLocal server. It may have stopped; restart the app.';
        let msg = this.error || this.detailMessage || `Request failed (HTTP ${this.status})`;
        if (this.detailMessage && this.error && this.detailMessage !== this.error) msg = `${this.error} — ${this.detailMessage}`;
        if (this.code) msg += ` [${this.code}]`;
        if (this.requestId) msg += ` (request ${this.requestId})`;
        return msg;
    }

    /** Validation messages carried in details (details.errors[] or other string values). */
    detailLines() {
        const d = this.details;
        if (!d) return [];
        if (Array.isArray(d)) return d.map(String);
        if (Array.isArray(d.errors)) {
            return d.errors.map((e) => (typeof e === 'string' ? e : (e && (e.message || e.msg)) || JSON.stringify(e)));
        }
        if (typeof d === 'object') {
            return Object.entries(d)
                .filter(([, v]) => ['string', 'number', 'boolean'].includes(typeof v))
                .map(([k, v]) => `${k}: ${v}`);
        }
        return [String(d)];
    }

    /** The message alone, for expected problems the user can act on (no code, no request id). */
    plain() {
        if (this.aborted || this.network) return this.describe();
        const d = this.details || {};
        if (this.code === 'INSUFFICIENT_DISK_SPACE' && Number.isFinite(d.required_bytes) && Number.isFinite(d.free_bytes)) {
            return `Not enough free disk space. This needs ${fmtBytes(d.required_bytes)} free, including a safety margin, and the models folder has ${fmtBytes(d.free_bytes)} free.`;
        }
        if (this.error && this.detailMessage && this.detailMessage !== this.error) return `${this.error} ${this.detailMessage}`;
        return this.error || this.detailMessage || `Request failed (HTTP ${this.status})`;
    }
}

/** A short message for the user; codes and request ids are left out for expected conflicts. */
export function plainError(err) {
    if (err instanceof ApiError) return ((err.status >= 400 && err.status < 500) || err.status === 507) ? err.plain() : err.describe();
    if (err && err.message) return err.message;
    return String(err);
}

export function describeError(err) {
    if (err instanceof ApiError) return err.describe();
    if (err && err.message) return err.message;
    return String(err);
}

async function parseError(response) {
    let payload = null;
    try {
        const raw = await response.text();
        payload = raw ? JSON.parse(raw) : null;
    } catch (_) {
        payload = null; // not JSON; the raw body is deliberately not shown (it may be remote content)
    }
    const requestId = (payload && payload.request_id) || response.headers.get('X-Request-ID');
    if (payload && typeof payload === 'object' && (payload.error || payload.message)) {
        return new ApiError({
            status: response.status,
            code: payload.error_code || null,
            error: typeof payload.error === 'string' ? payload.error : null,
            message: typeof payload.message === 'string' ? payload.message : null,
            details: payload.details || null,
            requestId,
        });
    }
    return new ApiError({ status: response.status, error: `${response.status} ${response.statusText || 'error'}`.trim(), requestId });
}

/**
 * request(method, path, {json, form, query, timeoutMs, signal, background})
 * Resolves with parsed JSON (or null for an empty body). Rejects with ApiError.
 * `background` marks a request the page makes by itself over and over (the
 * resource monitor); it is left out of inflightCount().
 */
export async function request(method, path, { json, form, query, timeoutMs = 60000, signal, background = false } = {}) {
    let url = path;
    if (query) {
        const qs = new URLSearchParams(query).toString();
        if (qs) url += (url.includes('?') ? '&' : '?') + qs;
    }
    const headers = { Accept: 'application/json' };
    if (token) headers['X-VVL-Token'] = token;
    let body;
    if (json !== undefined) {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify(json);
    } else if (form !== undefined) {
        body = form; // the browser sets the multipart boundary
    }

    const controller = new AbortController();
    const timer = timeoutMs ? setTimeout(() => controller.abort(), timeoutMs) : null;
    if (signal) {
        if (signal.aborted) controller.abort();
        else signal.addEventListener('abort', () => controller.abort(), { once: true });
    }
    if (!background) inflight += 1;
    let response;
    try {
        response = await fetch(url, {
            method, headers, body, signal: controller.signal, cache: 'no-store', credentials: 'same-origin',
        });
    } catch (err) {
        if (err && err.name === 'AbortError') throw new ApiError({ aborted: true });
        throw new ApiError({ network: true });
    } finally {
        if (timer) clearTimeout(timer);
        if (!background) inflight -= 1;
    }
    if (!response.ok) {
        const apiErr = await parseError(response);
        if (response.status === 401 && !authFailedNotified) {
            authFailedNotified = true;
            window.dispatchEvent(new CustomEvent('vvl:auth-failed', { detail: { message: apiErr.describe() } }));
        }
        throw apiErr;
    }
    if (response.status === 204) return null;
    const text = await response.text();
    if (!text) return null;
    try {
        return JSON.parse(text);
    } catch (_) {
        throw new ApiError({ status: response.status, error: 'The server sent a response that is not JSON.' });
    }
}

export const api = {
    get: (path, opts) => request('GET', path, opts),
    post: (path, json, opts = {}) => request('POST', path, { ...opts, json }),
    put: (path, json, opts = {}) => request('PUT', path, { ...opts, json }),
    del: (path, opts) => request('DELETE', path, opts),
    postForm: (path, form, opts = {}) => request('POST', path, { ...opts, form }),
};

// ---------------------------------------------------------------- toasts
let toastHost = null;

/** Small transient message. kind: 'info' | 'success' | 'error' | 'warning'. */
export function toast(message, kind = 'info', { timeout = kind === 'error' ? 9000 : 3500 } = {}) {
    if (!toastHost) {
        toastHost = document.getElementById('toasts');
        if (!toastHost) {
            toastHost = el('div', { id: 'toasts', class: 'toasts' });
            document.body.appendChild(toastHost);
        }
    }
    const iconName = { success: 'check', error: 'alert', warning: 'alert', info: 'info' }[kind] || 'info';
    const close = el('button', { type: 'button', class: 'toast-close', 'aria-label': 'Dismiss' }, icon('x', { size: 14 }));
    const node = el('div', { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : 'status' },
        icon(iconName, { size: 16 }), el('span', { class: 'toast-text' }, message), close);
    const remove = () => node.remove();
    close.addEventListener('click', remove);
    toastHost.appendChild(node);
    if (timeout) setTimeout(remove, timeout);
    return remove;
}

export function toastError(err, prefix = '') {
    return toast(prefix ? `${prefix}: ${describeError(err)}` : describeError(err), 'error');
}

/** Replace `container` with an inline error box (message + optional details + retry). */
export function inlineError(container, err, { title = 'Something went wrong', retry = null, compact = false } = {}) {
    const detail = el('div', null, describeError(err));
    if (err instanceof ApiError) {
        const lines = err.detailLines();
        if (lines.length) detail.appendChild(el('ul', { class: 'error-lines' }, lines.map((l) => el('li', null, l))));
    }
    const box = stateBox('error', title, detail, { retry, compact });
    if (container) {
        while (container.firstChild) container.removeChild(container.firstChild);
        container.appendChild(box);
    }
    return box;
}
