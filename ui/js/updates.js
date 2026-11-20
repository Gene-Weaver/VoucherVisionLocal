// App updates, from the desktop shell (window.vvlDesktop.update).
//
// The shell checks for, downloads and installs new versions. This page only
// shows what it reports and asks it to restart into a downloaded update. In a
// browser, or in an older shell without `update`, there is nothing to watch and
// the state says so ('unavailable').
//
// UpdateState (from the shell):
//   {status: 'disabled' | 'checking' | 'none' | 'downloading' | 'ready' | 'error',
//    version, percent, error, reason}
// plus, made here: status 'unavailable' when there is no updater at all.

const UNAVAILABLE = { status: 'unavailable', version: null, percent: null, error: null, reason: null };
const STATUSES = new Set(['disabled', 'checking', 'none', 'downloading', 'ready', 'error']);

let current = { ...UNAVAILABLE };
let started = false;
let unsubscribe = null;
let substitute = null; // a stand-in updater for the desktop screenshot scenarios (see startUpdates)
const listeners = new Set();

function updater() {
    const bridge = window.vvlDesktop;
    const u = substitute || (bridge && bridge.update);
    return u && typeof u.getState === 'function' && typeof u.install === 'function' ? u : null;
}

function normalize(raw) {
    if (!raw || typeof raw !== 'object' || !STATUSES.has(raw.status)) return { ...UNAVAILABLE };
    const pct = Number(raw.percent);
    return {
        status: raw.status,
        version: typeof raw.version === 'string' && raw.version ? raw.version : null,
        percent: raw.percent === null || raw.percent === undefined || !Number.isFinite(pct) ? null : Math.max(0, Math.min(100, Math.round(pct))),
        error: typeof raw.error === 'string' && raw.error ? raw.error : null,
        reason: typeof raw.reason === 'string' && raw.reason ? raw.reason : null,
    };
}

function set(raw) {
    current = normalize(raw);
    for (const fn of listeners) {
        try {
            fn(current);
        } catch (err) {
            console.error('[vvl] update listener failed', err);
        }
    }
}

/**
 * Start following the shell's update state (again). Safe to call more than once.
 * `standIn` replaces the shell's updater: the desktop bridge is frozen, so the
 * screenshot scenarios (desktop/scenarios/setup-page.js) pass a fake one here
 * through window.vvlUi.useUpdater to show the update states.
 */
export function startUpdates(standIn = undefined) {
    if (standIn !== undefined) substitute = standIn || null;
    if (unsubscribe) {
        try { unsubscribe(); } catch (_) { /* the shell may be gone */ }
        unsubscribe = null;
    }
    started = true;
    const u = updater();
    if (!u) {
        set(UNAVAILABLE);
        return;
    }
    if (typeof u.onState === 'function') {
        try {
            const off = u.onState((state) => set(state));
            if (typeof off === 'function') unsubscribe = off;
        } catch (err) {
            console.warn('[vvl] update events unavailable', err);
        }
    }
    Promise.resolve().then(() => u.getState()).then(set, () => set(UNAVAILABLE));
}

export function updateState() {
    return current;
}

/** Call `fn(state)` now and on every change. Returns a function that stops it. */
export function onUpdate(fn) {
    if (!started) startUpdates();
    listeners.add(fn);
    fn(current);
    return () => listeners.delete(fn);
}

/**
 * Ask the shell to restart into the downloaded update. The shell asks the user
 * first when a batch is running. Resolves false when there is nothing to do or
 * the user declined.
 */
export async function installUpdate() {
    const u = updater();
    if (!u || current.status !== 'ready') return false;
    const ui = window.vvlUi;
    const batchRunning = Boolean(ui && typeof ui.batchRunning === 'function' && ui.batchRunning());
    try {
        return (await u.install({ batchRunning })) !== false;
    } catch (err) {
        console.warn('[vvl] update install failed', err);
        return false;
    }
}

/**
 * The update status in a few words, for the Setup page.
 *   {kind: 'good' | 'info' | 'working' | 'ready' | 'bad', text, tip}
 */
export function describeUpdate(state = current) {
    const v = state.version;
    switch (state.status) {
        case 'none':
            return { kind: 'good', text: 'Up to date', tip: 'The app checks for a new version when it starts.' };
        case 'checking':
            return { kind: 'working', text: 'Checking for updates…', tip: null };
        case 'downloading':
            return {
                kind: 'working',
                text: `Downloading ${v || 'an update'}${state.percent === null ? '' : ` (${state.percent}%)`}`,
                tip: 'You can keep working. The update is installed when the app restarts.',
            };
        case 'ready':
            return { kind: 'ready', text: `${v || 'An update'} is ready`, tip: 'Restarting takes a few seconds. Downloads continue where they stopped.' };
        case 'error':
            return { kind: 'bad', text: `Could not check for updates${state.error ? `: ${state.error}` : ''}`, tip: 'The app tries again the next time it starts.' };
        case 'disabled': {
            const packaged = Boolean(window.vvlDesktop && window.vvlDesktop.packaged);
            if (/develop/i.test(state.reason || '') || (!packaged && (!state.reason || /not started/i.test(state.reason)))) {
                return { kind: 'info', text: 'Updates are off in development builds', tip: 'Update a source checkout with git pull.' };
            }
            if (/not started/i.test(state.reason || '')) return { kind: 'info', text: 'Updates are checked after the app starts', tip: null };
            return { kind: 'info', text: `Updates are off${state.reason ? `: ${state.reason}` : ''}`, tip: null };
        }
        default:
            return window.vvlDesktop
                ? { kind: 'info', text: 'Updates are off in development builds', tip: 'This desktop app does not install updates by itself. Update a source checkout with git pull.' }
                : { kind: 'info', text: 'Updates come with the desktop app', tip: 'In a web browser, update VoucherVisionLocal where it is installed.' };
    }
}
