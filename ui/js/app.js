// App shell: session-token gate, the top tab bar, lazy tab loading, theme.

import { api, ApiError, describeError, initToken, inflightCount, toast } from './api.js';
import { clear, el, icon, replace, prefs, stateBox } from './dom.js';
import { getAppInfo, loadSettings, on, setAppInfo } from './state.js';
import { allJobs, isActive, mountIndicator, refreshJobs } from './jobs.js';
import { mountMonitor } from './monitor.js';
import { installUpdate, onUpdate, startUpdates } from './updates.js';

const TABS = [
    { id: 'vvl', label: 'VVL', title: 'Process specimen images', module: './tabs/vvl.js' },
    { id: 'models', label: 'Models', title: 'Local and API models', module: './tabs/models.js' },
    { id: 'keys', label: 'Keys', title: 'API keys and tokens', module: './tabs/keys.js' },
    { id: 'prompts', label: 'Prompts', title: 'Prompt templates', module: './tabs/prompts.js' },
    { id: 'engine', label: 'Engine', title: 'This computer and the local engine', module: './tabs/engine.js' },
    // No button in the tab bar: opened from the version number in the header, by #setup, and on first run.
    { id: 'setup', label: 'Setup', title: 'Set up VoucherVisionLocal', module: './tabs/setup.js', hidden: true },
];

const loaded = new Map(); // tab id -> Promise<controller>
let activeTab = null;

// Exposed for the desktop shell's screenshot run (no secrets here).
window.vvlUi = {
    inflight: () => inflightCount(),
    activeTab: () => activeTab,
    activeJobs: () => allJobs().filter(isActive).length,
    // True while the VVL tab is processing images (the VVL tab replaces this once it loads).
    batchRunning: () => false,
    // Screenshot scenarios only: follow a stand-in updater instead of the shell's (ui/js/updates.js).
    useUpdater: (standIn) => startUpdates(standIn),
    tabState: (id) => {
        const panel = document.getElementById(`panel-${id}`);
        return panel ? panel.dataset.state || 'idle' : 'missing';
    },
};

// ---------------------------------------------------------------- theme
function currentTheme() {
    return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

function applyTheme(theme, { persist = true } = {}) {
    document.documentElement.dataset.theme = theme;
    const btn = document.getElementById('themeToggle');
    if (btn) {
        const dark = theme === 'dark';
        btn.setAttribute('aria-pressed', String(dark));
        btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
        btn.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
    }
    if (persist) prefs.set('theme', theme);
}

function setupTheme() {
    const btn = document.getElementById('themeToggle');
    btn.appendChild(icon('sun', { size: 20, className: 'icon icon-sun' }));
    btn.appendChild(icon('moon', { size: 20, className: 'icon icon-moon' }));
    applyTheme(currentTheme(), { persist: false });
    btn.addEventListener('click', () => {
        const next = currentTheme() === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        // Also remember it in settings, best effort (the local preference already applied).
        api.put('/v1/settings', { ui: { theme: next } }).catch(() => {});
    });
}

// ---------------------------------------------------------------- sticky header and tab bar
/**
 * The header and the tab bar stay at the top while the page scrolls. Their
 * height goes into --topbar-h, which the style sheet uses so that in-page
 * jumps, focus moves and other sticky parts stop below them.
 */
function setupTopBar() {
    const bar = document.getElementById('topBar');
    if (!bar) return;
    const measure = () => document.documentElement.style.setProperty('--topbar-h', `${Math.ceil(bar.getBoundingClientRect().height)}px`);
    measure();
    if (typeof ResizeObserver === 'function') new ResizeObserver(measure).observe(bar);
    else window.addEventListener('resize', measure);
    const onScroll = () => bar.classList.toggle('is-scrolled', window.scrollY > 2);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
}

// ---------------------------------------------------------------- fatal screens
function fatal(title, paragraphs, { retry = false } = {}) {
    document.getElementById('tabBar').hidden = true;
    const main = document.getElementById('main');
    const box = el('section', { class: 'fatal', role: 'alert', 'aria-labelledby': 'fatalTitle' },
        icon('alert', { size: 36 }),
        el('h2', { id: 'fatalTitle' }, title),
        paragraphs.map((p) => el('p', null, p)),
        retry ? el('button', { type: 'button', class: 'button', onclick: () => window.location.reload() }, 'Try again') : null);
    replace(main, box);
}

function fatalNoToken() {
    fatal('This page needs its session link', [
        'VoucherVisionLocal only answers requests that carry the session token it created when it started.',
        'Open the app from the VoucherVisionLocal desktop window, or open the full link printed by "vvl serve" in the terminal (it ends with ?token=...).',
    ]);
}

function fatalBadToken() {
    fatal('The session token was not accepted', [
        'The server running on this computer did not accept this page’s session token. This usually means the server was restarted and created a new token.',
        'Open the new link printed by "vvl serve", or restart the desktop app.',
    ]);
}

// ---------------------------------------------------------------- tabs
function buildTabBar() {
    const bar = document.getElementById('tabBar');
    const list = el('div', { class: 'nav-tabs', role: 'tablist', 'aria-label': 'VoucherVisionLocal sections' });
    for (const t of TABS.filter((x) => !x.hidden)) {
        list.appendChild(el('button', {
            type: 'button',
            role: 'tab',
            id: `tab-${t.id}`,
            class: 'nav-tab',
            'aria-controls': `panel-${t.id}`,
            'aria-selected': 'false',
            tabindex: '-1',
            title: t.title,
            dataset: { tab: t.id },
            onclick: () => activate(t.id, null, { focus: false, updateHash: true }),
        }, t.label));
    }
    list.addEventListener('keydown', (ev) => {
        const tabs = [...list.querySelectorAll('[role="tab"]')];
        const i = tabs.indexOf(document.activeElement);
        if (i < 0) return;
        let next = null;
        if (ev.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
        else if (ev.key === 'ArrowLeft') next = tabs[(i - 1 + tabs.length) % tabs.length];
        else if (ev.key === 'Home') next = tabs[0];
        else if (ev.key === 'End') next = tabs[tabs.length - 1];
        if (next) {
            ev.preventDefault();
            activate(next.dataset.tab, null, { focus: true, updateHash: true });
        }
    });
    bar.appendChild(list);

    const main = document.getElementById('main');
    clear(main);
    for (const t of TABS) {
        main.appendChild(el('section', {
            id: `panel-${t.id}`,
            class: 'tab-panel',
            role: 'tabpanel',
            'aria-labelledby': t.hidden ? null : `tab-${t.id}`,
            'aria-label': t.hidden ? t.title : null,
            tabindex: '0',
            hidden: true,
            dataset: { state: 'idle' },
        }));
    }
}

function parseHash() {
    const raw = decodeURIComponent((window.location.hash || '').replace(/^#/, ''));
    const [tab, ...rest] = raw.split('/');
    const known = TABS.some((t) => t.id === tab);
    return { tab: known ? tab : 'vvl', target: known && rest.length ? rest.join('/') : null };
}

async function activate(id, target = null, { focus = false, updateHash = false } = {}) {
    const def = TABS.find((t) => t.id === id) || TABS[0];
    activeTab = def.id;
    for (const t of TABS) {
        const tabBtn = document.getElementById(`tab-${t.id}`);
        const panel = document.getElementById(`panel-${t.id}`);
        const on = t.id === def.id;
        if (tabBtn) {
            tabBtn.setAttribute('aria-selected', String(on));
            tabBtn.tabIndex = on ? 0 : -1;
            tabBtn.classList.toggle('active', on);
        }
        panel.hidden = !on;
    }
    // The Setup page has no tab button; the version button in the header stands for it.
    const versionBtn = document.getElementById('appVersion');
    if (versionBtn) {
        if (def.id === 'setup') versionBtn.setAttribute('aria-current', 'page');
        else versionBtn.removeAttribute('aria-current');
    }
    const tabBtnNow = document.getElementById(`tab-${def.id}`);
    if (focus && tabBtnNow) tabBtnNow.focus();
    if (updateHash) {
        const hash = `#${def.id}${target ? `/${encodeURIComponent(target)}` : ''}`;
        if (window.location.hash !== hash) window.history.replaceState(null, '', hash);
    }
    document.title = `${def.label === 'VVL' ? 'VoucherVisionLocal' : `${def.label} — VoucherVisionLocal`}`;

    const panel = document.getElementById(`panel-${def.id}`);
    if (!loaded.has(def.id)) {
        panel.dataset.state = 'loading';
        replace(panel, stateBox('loading', `Loading ${def.label}…`));
        loaded.set(def.id, import(def.module).then((mod) => mod.mount(panel)).catch((err) => {
            loaded.delete(def.id);
            panel.dataset.state = 'error';
            replace(panel, el('div', { class: 'state-box state-error', role: 'alert' },
                icon('alert', { size: 20 }),
                el('div', { class: 'state-body' },
                    el('div', { class: 'state-title' }, `The ${def.label} tab could not be loaded`),
                    el('div', { class: 'state-detail' }, describeError(err)))));
            throw err;
        }));
    }
    try {
        const controller = await loaded.get(def.id);
        if (activeTab === def.id && controller && typeof controller.show === 'function') controller.show(target);
    } catch (err) {
        console.error(`[vvl] tab ${def.id} failed to load`, err);
    }
}

function renderHeaderInfo(info) {
    const ver = document.getElementById('appVersion');
    if (ver && info && info.version) {
        ver.textContent = `v${info.version}`;
        ver.setAttribute('aria-label', `Version ${info.version}: setup and updates`);
    }
}

// The version number opens the Setup page; "Update ready" appears beside it once the
// desktop shell has downloaded a new version (ui/js/updates.js).
function setupHeaderButtons() {
    const ver = document.getElementById('appVersion');
    if (ver) ver.addEventListener('click', () => activate('setup', null, { updateHash: true }));
    const pill = document.getElementById('updateReady');
    if (!pill) return;
    pill.addEventListener('click', async () => {
        pill.disabled = true;
        try {
            await installUpdate();
        } finally {
            pill.disabled = false;
        }
    });
    onUpdate((state) => {
        const ready = state.status === 'ready';
        pill.hidden = !ready;
        if (!ready) return;
        const label = state.version ? `Version ${state.version} is downloaded. Restart to update.` : 'An update is downloaded. Restart to update.';
        replace(pill, icon('refresh', { size: 13 }), el('span', null, 'Update ready'));
        pill.title = label;
        pill.setAttribute('aria-label', label);
    });
}

/**
 * First run: nothing is set up yet (no answer to "How will you run models?", no
 * OpenRouter key, no engine). Any failure counts as "not first run".
 */
async function isFirstRun() {
    try {
        const settings = await loadSettings();
        if (!settings || !settings.setup || settings.setup.usage !== null) return false;
        const [providers, engine] = await Promise.all([
            api.get('/v1/providers', { timeoutMs: 15000 }),
            api.get('/v1/engine', { timeoutMs: 30000 }),
        ]);
        const openRouter = ((providers && providers.providers) || []).find((p) => p.id === 'openrouter');
        if (openRouter && openRouter.configured) return false;
        return !(engine && engine.managed && engine.managed.installed);
    } catch (_) {
        return false;
    }
}

// ---------------------------------------------------------------- boot
async function boot() {
    setupTheme();
    setupTopBar();
    if (!initToken()) {
        fatalNoToken();
        return;
    }
    window.addEventListener('vvl:auth-failed', () => fatalBadToken());

    let info;
    try {
        info = await api.get('/v1/app', { timeoutMs: 15000 });
    } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
            fatalBadToken();
        } else if (err instanceof ApiError && err.network) {
            fatal('VoucherVisionLocal is not responding', [
                `Nothing answered at ${window.location.origin}. The server may have stopped or may still be starting.`,
                'If you started it with "vvl serve", check that terminal for errors.',
            ], { retry: true });
        } else {
            fatal('VoucherVisionLocal could not start its interface', [describeError(err)], { retry: true });
        }
        return;
    }
    setAppInfo(info);
    renderHeaderInfo(info);
    setupHeaderButtons();

    // Theme from settings only when this viewer has no saved preference yet.
    if (prefs.get('theme') === null) {
        loadSettings().then((s) => {
            if (s && s.ui && (s.ui.theme === 'dark' || s.ui.theme === 'light')) applyTheme(s.ui.theme, { persist: false });
        }).catch(() => {});
    }

    buildTabBar();
    document.getElementById('tabBar').hidden = false;
    // Downloads and installs keep running on the server; pick them up again after a reload.
    mountIndicator(document.getElementById('jobIndicator'));
    refreshJobs();
    mountMonitor(document.getElementById('resourceMonitor'));
    on('goto', ({ tab, target }) => activate(tab, target, { updateHash: true }));
    window.addEventListener('hashchange', () => {
        const { tab, target } = parseHash();
        activate(tab, target);
    });
    const noHash = !(window.location.hash || '').replace(/^#/, '');
    const firstRun = noHash && await isFirstRun();
    if (activeTab !== null) return; // a tab was clicked while that was checked
    const { tab, target } = parseHash();
    if (firstRun && !(window.location.hash || '').replace(/^#/, '')) activate('setup', null, { updateHash: true });
    else activate(tab, target, { updateHash: true });
}

window.addEventListener('unhandledrejection', (ev) => {
    // Keep failures visible instead of silent; tabs already show their own errors.
    if (ev.reason instanceof ApiError) {
        ev.preventDefault();
        console.warn('[vvl] unhandled API error:', ev.reason.describe());
    }
});

if (getAppInfo() === null) boot().catch((err) => {
    console.error('[vvl] boot failed', err);
    toast(describeError(err), 'error');
});
