// DOM helpers shared by every tab.
//
// Rule for this UI: untrusted text (model names, prompt text, file names, OCR
// output, error messages) is only ever inserted as text. `el()` builds nodes
// with textContent and setAttribute, so nothing here parses HTML and no
// escaping helper is needed.

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Build an element.
 *   el('button', {class: 'button', type: 'button', onclick: fn, 'aria-label': 'Save'}, 'Save')
 * Attribute keys:
 *   class / className  -> className
 *   text               -> textContent
 *   dataset            -> object copied onto element.dataset
 *   on<event>          -> addEventListener(event, fn)
 *   hidden / disabled / checked / selected / value / readOnly -> set as properties
 *   anything else      -> setAttribute (null/undefined/false are skipped)
 * Children: strings and numbers become text nodes; null/false are skipped; arrays are flattened.
 */
export function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    applyAttrs(node, attrs);
    append(node, children);
    return node;
}

const PROPS = new Set(['hidden', 'disabled', 'checked', 'selected', 'value', 'readOnly', 'multiple', 'indeterminate', 'open']);

function applyAttrs(node, attrs) {
    if (!attrs) return;
    for (const [key, val] of Object.entries(attrs)) {
        if (val === undefined || val === null) continue;
        if (key === 'class' || key === 'className') {
            node.className = Array.isArray(val) ? val.filter(Boolean).join(' ') : String(val);
        } else if (key === 'text') {
            node.textContent = String(val);
        } else if (key === 'dataset') {
            for (const [dk, dv] of Object.entries(val)) {
                if (dv !== undefined && dv !== null) node.dataset[dk] = String(dv);
            }
        } else if (key.startsWith('on') && typeof val === 'function') {
            node.addEventListener(key.slice(2).toLowerCase(), val);
        } else if (PROPS.has(key)) {
            node[key] = val;
        } else if (val === false) {
            continue;
        } else if (val === true) {
            node.setAttribute(key, '');
        } else {
            node.setAttribute(key, String(val));
        }
    }
}

export function append(node, children) {
    for (const child of children.flat(Infinity)) {
        if (child === null || child === undefined || child === false) continue;
        if (child instanceof Node) node.appendChild(child);
        else node.appendChild(document.createTextNode(String(child)));
    }
    return node;
}

export function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
    return node;
}

export function replace(node, ...children) {
    clear(node);
    return append(node, children);
}

// ---------------------------------------------------------------- icons
// Small stroke icons (Feather-style geometry, drawn by hand here). Built with
// createElementNS, so no markup parsing is involved.
const ICONS = {
    sun: [['circle', { cx: 12, cy: 12, r: 5 }], ['path', { d: 'M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42' }]],
    moon: [['path', { d: 'M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z' }]],
    computer: [['rect', { x: 2, y: 3, width: 20, height: 14, rx: 2 }], ['path', { d: 'M8 21h8M12 17v4' }]],
    shield: [['path', { d: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z' }], ['path', { d: 'M9 12l2 2 4-4' }]],
    cloud: [['path', { d: 'M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z' }]],
    key: [['circle', { cx: 7.5, cy: 15.5, r: 4.5 }], ['path', { d: 'M10.7 12.3 21 2M16 7l3 3M18 5l2 2' }]],
    eye: [['path', { d: 'M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z' }], ['circle', { cx: 12, cy: 12, r: 3 }]],
    eyeOff: [['path', { d: 'M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19M14.12 14.12a3 3 0 1 1-4.24-4.24M1 1l22 22' }]],
    check: [['path', { d: 'M20 6 9 17l-5-5' }]],
    chevronDown: [['path', { d: 'm6 9 6 6 6-6' }]],
    x: [['path', { d: 'M18 6 6 18M6 6l12 12' }]],
    alert: [['path', { d: 'M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z' }], ['path', { d: 'M12 9v4M12 17h.01' }]],
    info: [['circle', { cx: 12, cy: 12, r: 10 }], ['path', { d: 'M12 16v-4M12 8h.01' }]],
    refresh: [['path', { d: 'M23 4v6h-6M1 20v-6h6' }], ['path', { d: 'M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15' }]],
    download: [['path', { d: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3' }]],
    external: [['path', { d: 'M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14 21 3' }]],
    folder: [['path', { d: 'M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z' }]],
    cpu: [['rect', { x: 4, y: 4, width: 16, height: 16, rx: 2 }], ['rect', { x: 9, y: 9, width: 6, height: 6 }], ['path', { d: 'M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3' }]],
    gpu: [['rect', { x: 2, y: 6, width: 20, height: 12, rx: 2 }], ['circle', { cx: 9, cy: 12, r: 3 }], ['path', { d: 'M16 10h3M16 14h3' }]],
    memory: [['rect', { x: 3, y: 7, width: 18, height: 10, rx: 1 }], ['path', { d: 'M7 7v10M11 7v10M15 7v10M3 20h18' }]],
    disk: [['path', { d: 'M22 12H2M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z' }], ['path', { d: 'M6 16h.01M10 16h.01' }]],
    pin: [['path', { d: 'M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z' }], ['circle', { cx: 12, cy: 10, r: 3 }]],
    trash: [['path', { d: 'M3 6h18M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M10 11v6M14 11v6M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2' }]],
    file: [['path', { d: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z' }], ['path', { d: 'M14 2v6h6' }]],
    terminal: [['path', { d: 'M4 17l6-6-6-6M12 19h8' }]],
    plus: [['path', { d: 'M12 5v14M5 12h14' }]],
    star: [['path', { d: 'M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z' }]],
    play: [['path', { d: 'M5 3l14 9-14 9V3z' }]],
    stop: [['rect', { x: 5, y: 5, width: 14, height: 14, rx: 2 }]],
    copy: [['rect', { x: 9, y: 9, width: 13, height: 13, rx: 2 }], ['path', { d: 'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1' }]],
    settings: [['path', { d: 'M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6' }]],
};

export function icon(name, { size = 18, label = null, className = 'icon' } = {}) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', String(size));
    svg.setAttribute('height', String(size));
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('class', className);
    if (label) {
        svg.setAttribute('role', 'img');
        svg.setAttribute('aria-label', label);
    } else {
        svg.setAttribute('aria-hidden', 'true');
        svg.setAttribute('focusable', 'false');
    }
    for (const [tag, attrs] of ICONS[name] || []) {
        const child = document.createElementNS(SVG_NS, tag);
        for (const [k, v] of Object.entries(attrs)) child.setAttribute(k, String(v));
        svg.appendChild(child);
    }
    return svg;
}

// ---------------------------------------------------------------- external links
/** True only for an absolute https: URL (parsed, never prefix-matched). */
export function isHttpsUrl(url) {
    try {
        return new URL(String(url)).protocol === 'https:';
    } catch (_) {
        return false;
    }
}

/** Open a page in the system browser (desktop) or a new tab (browser). https only. */
export function openExternal(url) {
    if (!isHttpsUrl(url)) return false;
    const bridge = window.vvlDesktop;
    if (bridge && typeof bridge.openExternal === 'function') {
        bridge.openExternal(String(url));
        return true;
    }
    window.open(String(url), '_blank', 'noopener,noreferrer');
    return true;
}

/** An <a> for an external https page; anything else renders as plain text. */
export function externalLink(label, url, { className = 'external-link' } = {}) {
    if (!isHttpsUrl(url)) return el('span', null, label);
    return el('a', {
        class: className,
        href: url,
        target: '_blank',
        rel: 'noopener noreferrer',
        onclick: (ev) => {
            ev.preventDefault();
            openExternal(url);
        },
    }, label, el('span', { class: 'visually-hidden' }, ' (opens in your web browser)'), icon('external', { size: 13, className: 'icon icon-inline' }));
}

// ---------------------------------------------------------------- formatting
export function fmtBytes(n, { digits = 1 } = {}) {
    const v = Number(n);
    if (!Number.isFinite(v) || v < 0) return '—';
    if (v < 1000) return `${v} B`;
    const units = ['KB', 'MB', 'GB', 'TB', 'PB'];
    let x = v;
    let i = -1;
    do {
        x /= 1000;
        i += 1;
    } while (x >= 1000 && i < units.length - 1);
    return `${x.toFixed(x >= 100 ? 0 : digits)} ${units[i]}`;
}

// Memory (RAM, graphics memory) is shown in binary units, the way operating
// systems and hardware labels report it: a "48 GB" card, "16 GB" of RAM.
export function fmtMemory(n, { digits = 1 } = {}) {
    const v = Number(n);
    if (!Number.isFinite(v) || v < 0) return '—';
    const gb = v / (1024 ** 3);
    if (gb >= 1) return `${gb.toFixed(gb >= 100 ? 0 : digits).replace(/\.0$/, '')} GB`;
    return `${Math.round(v / (1024 ** 2))} MB`;
}

/** A file or disk size with the exact byte count beside it: "35.6 GB (35,590,000,000 bytes)". */
export function fmtBytesExact(n) {
    const v = Number(n);
    if (!Number.isFinite(v) || v < 0) return '—';
    return `${fmtBytes(v)} (${v.toLocaleString('en-US')} bytes)`;
}

/** Seconds as a short duration: "45 s", "3 min 20 s", "2 h 5 min". */
export function fmtDuration(seconds) {
    const s = Math.round(Number(seconds));
    if (!Number.isFinite(s) || s < 0) return '—';
    if (s < 60) return `${s} s`;
    if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
    return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}

/** Elapsed time as a clock: "0:07", "12:40", "1:02:03". */
export function fmtClock(seconds) {
    const s = Math.max(0, Math.floor(Number(seconds) || 0));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = String(s % 60).padStart(2, '0');
    return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

export function fmtDateTime(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' });
}

// ---------------------------------------------------------------- files and clipboard
export function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: filename, class: 'visually-hidden' });
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        a.remove();
        URL.revokeObjectURL(url);
    }, 200);
}

export function downloadText(text, filename, type = 'text/plain;charset=utf-8') {
    downloadBlob(new Blob([text], { type }), filename);
}

export function base64ToBlob(b64, type = 'image/jpeg') {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type });
}

export async function copyText(text) {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (_) { /* fall through to the textarea method */ }
    const ta = el('textarea', { class: 'visually-hidden', 'aria-hidden': 'true' });
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
        ok = document.execCommand('copy');
    } catch (_) {
        ok = false;
    }
    ta.remove();
    return ok;
}

// ---------------------------------------------------------------- small widgets
/** Loading / empty / error placeholder. kind: 'loading' | 'empty' | 'error'. */
export function stateBox(kind, title, detail = null, { retry = null, compact = false } = {}) {
    const box = el('div', {
        class: ['state-box', `state-${kind}`, compact ? 'state-compact' : ''],
        role: kind === 'error' ? 'alert' : 'status',
    });
    if (kind === 'loading') box.appendChild(el('span', { class: 'spinner', 'aria-hidden': 'true' }));
    else box.appendChild(icon(kind === 'error' ? 'alert' : 'info', { size: 20 }));
    const body = el('div', { class: 'state-body' }, el('div', { class: 'state-title' }, title));
    if (detail) body.appendChild(el('div', { class: 'state-detail' }, detail));
    if (retry) {
        body.appendChild(el('button', { type: 'button', class: 'button-secondary state-retry', onclick: retry }, 'Try again'));
    }
    box.appendChild(body);
    return box;
}

export function badge(text, kind = 'neutral', { title = null } = {}) {
    return el('span', { class: `badge badge-${kind}`, title }, text);
}

/** Simple label/value definition list from [[label, value], ...]; value may be a Node. */
export function defList(rows, className = 'kv') {
    const dl = el('dl', { class: className });
    for (const [k, v] of rows) {
        if (v === undefined) continue;
        dl.appendChild(el('dt', null, k));
        dl.appendChild(el('dd', null, v === null || v === '' ? '—' : v));
    }
    return dl;
}

/** Modal dialog with focus handling. Returns {close}. */
export function openModal({ title, body, className = '', labelledBy = null, onClose = null }) {
    const previous = document.activeElement;
    const titleId = labelledBy || `modal-title-${Math.random().toString(36).slice(2)}`;
    const closeBtn = el('button', { type: 'button', class: 'modal-close', 'aria-label': 'Close' }, icon('x', { size: 18 }));
    const dialog = el('div', {
        class: ['modal', className],
        role: 'dialog',
        'aria-modal': 'true',
        'aria-labelledby': titleId,
        tabindex: '-1',
    },
    el('div', { class: 'modal-header' }, title ? el('h3', { id: titleId }, title) : null, closeBtn),
    el('div', { class: 'modal-body' }, body));
    const overlay = el('div', { class: 'modal-overlay' }, dialog);

    function close() {
        document.removeEventListener('keydown', onKey, true);
        overlay.remove();
        if (previous && typeof previous.focus === 'function') previous.focus();
        if (onClose) onClose();
    }
    function onKey(ev) {
        if (ev.key === 'Escape') {
            ev.stopPropagation();
            close();
        } else if (ev.key === 'Tab') {
            const focusables = dialog.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
            if (!focusables.length) return;
            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            if (ev.shiftKey && document.activeElement === first) {
                ev.preventDefault();
                last.focus();
            } else if (!ev.shiftKey && document.activeElement === last) {
                ev.preventDefault();
                first.focus();
            }
        }
    }
    closeBtn.addEventListener('click', close);
    overlay.addEventListener('click', (ev) => {
        if (ev.target === overlay) close();
    });
    document.addEventListener('keydown', onKey, true);
    document.body.appendChild(overlay);
    dialog.focus();
    return { close, dialog };
}

/** Yes/No confirmation built on openModal. Resolves true/false. */
export function confirmDialog({ title, message, confirmLabel = 'Confirm', danger = false }) {
    return new Promise((resolve) => {
        let answered = false;
        const yes = el('button', { type: 'button', class: danger ? 'button button-danger' : 'button' }, confirmLabel);
        const no = el('button', { type: 'button', class: 'button-secondary' }, 'Cancel');
        const m = openModal({
            title,
            className: 'modal-confirm',
            body: [el('p', null, message), el('div', { class: 'modal-actions' }, no, yes)],
            onClose: () => {
                if (!answered) resolve(false);
            },
        });
        yes.addEventListener('click', () => {
            answered = true;
            m.close();
            resolve(true);
        });
        no.addEventListener('click', () => {
            answered = true;
            m.close();
            resolve(false);
        });
        no.focus();
    });
}

/**
 * A dialog whose confirm button runs an async action. While the action runs the
 * buttons are disabled; if it throws, `describe(err)` is shown inside the dialog
 * and the dialog stays open so the user can read it. Resolves the action's
 * result, or null when the dialog is dismissed.
 *   body: a Node, a string, or an array of them.
 */
export function actionDialog({ title, body, confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false, action, describe = (err) => (err && err.message) || String(err), confirmDisabled = false }) {
    return new Promise((resolve) => {
        let settled = false;
        let running = false;
        const yes = el('button', { type: 'button', class: danger ? 'button button-danger' : 'button', disabled: confirmDisabled }, confirmLabel);
        const no = el('button', { type: 'button', class: 'button-secondary' }, cancelLabel);
        const problem = el('div', { class: 'modal-problem', role: 'alert', hidden: true });
        const content = Array.isArray(body) ? body : [body];
        const m = openModal({
            title,
            className: 'modal-confirm',
            body: [content.map((c) => (typeof c === 'string' ? el('p', null, c) : c)), problem, el('div', { class: 'modal-actions' }, no, yes)],
            onClose: () => {
                // Closed while the action runs (Escape): its outcome is still reported below.
                if (!settled && !running) {
                    settled = true;
                    resolve(null);
                }
            },
        });
        yes.addEventListener('click', async () => {
            if (running) return;
            running = true;
            yes.disabled = true;
            no.disabled = true;
            problem.hidden = true;
            try {
                const result = await action();
                running = false;
                settled = true;
                m.close();
                resolve(result === undefined ? true : result);
            } catch (err) {
                if (!m.dialog.isConnected) {
                    settled = true;
                    resolve(null);
                    return;
                }
                replace(problem, icon('alert', { size: 18 }), el('span', null, describe(err)));
                problem.hidden = false;
                yes.disabled = false;
                no.disabled = false;
                no.focus();
            } finally {
                running = false;
            }
        });
        no.addEventListener('click', () => m.close());
        (confirmDisabled ? no : yes).focus();
    });
}

/** Safe wrappers for per-viewer conveniences (never secrets). */
export const prefs = {
    get(key, fallback = null) {
        try {
            const raw = window.localStorage.getItem(`vvl.${key}`);
            return raw === null ? fallback : JSON.parse(raw);
        } catch (_) {
            return fallback;
        }
    },
    set(key, value) {
        try {
            window.localStorage.setItem(`vvl.${key}`, JSON.stringify(value));
        } catch (_) { /* storage unavailable: the preference just is not remembered */ }
    },
};
