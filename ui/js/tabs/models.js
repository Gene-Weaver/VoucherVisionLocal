// Models tab: two half-width tabs. "Local models" (images stay here) and
// "OpenRouter models" (images are sent to the provider), kept visually apart.

import { api, inlineError, plainError, toast, toastError } from '../api.js';
import {
    el, icon, replace, stateBox, badge, externalLink, fmtBytes, fmtBytesExact, defList, actionDialog,
} from '../dom.js';
import { emit, on, gotoTab } from '../state.js';
import { activeJob, getJob, isActive, jobError, jobFor, progressView, refreshJobs, trackJob } from '../jobs.js';
import { mountOpenRouterGallery } from './openrouter_gallery.js';
import { mountModelsFolder } from '../models_folder.js';

const FIT_KIND = { fits_gpu: 'good', fits_split: 'info', cpu_only: 'warn', too_large: 'bad', unknown: 'neutral' };
const ROLE_LABEL = { ocr: 'OCR', parse: 'Parsing' };
const DOWNLOAD_KIND = 'model_download';

const SUBTABS = [
    { id: 'local', icon: 'shield', title: 'Local models', subtitle: 'Images stay on this computer. Nothing is uploaded.' },
    { id: 'remote', icon: 'cloud', title: 'OpenRouter models', subtitle: 'Images are sent to the provider, using your own key.' },
];

function plannedCard(family) {
    return el('div', { class: 'card family-card is-planned' },
        el('div', { class: 'card-header' }, el('h3', null, family.label), badge('Coming soon', 'neutral')),
        el('div', { class: 'family-meta' }, [family.developer, family.license].filter(Boolean).join(' · ')),
        family.summary ? el('p', { class: 'family-summary' }, family.summary) : null);
}

/** Where a build's files are kept: <models folder>/models/<org>/<repo>. */
function buildFolder(modelsDir, build) {
    if (!modelsDir) return null;
    const sep = modelsDir.includes('\\') && !modelsDir.includes('/') ? '\\' : '/';
    const base = modelsDir.replace(/[\\/]+$/, '');
    return [base, 'models', ...String(build.hf_repo || '').split('/').filter(Boolean)].join(sep);
}

/** What a build row should show, from the model list and the job list together. */
function buildView(build) {
    const job = jobFor(DOWNLOAD_KIND, build.id) || getJob(build.download_job_id);
    if (isActive(job)) return { state: job.state === 'queued' ? 'queued' : 'downloading', job };
    const state = build.install_state || 'not_installed';
    // The model list is re-read when a job ends; until then trust the finished job.
    if (state === 'downloading' || state === 'queued') {
        return { state: job && job.state === 'done' ? 'installed' : (Number(build.installed_bytes) > 0 ? 'partial' : 'not_installed'), job };
    }
    return { state, job };
}

export async function mount(panel) {
    const localBody = el('div');
    const remoteBody = el('div');
    let gallery = null;          // the OpenRouter gallery, made when the provider is listed
    let activeTab = 'local';

    // ------------------------------------------------------------ local models
    let families = [];
    let modelsDir = null;
    let engineInstalled = null; // null = not known
    const rows = new Map(); // build id -> {build, key, status, action, progressRow, progressCell, view}
    const engineNotice = el('div');

    function findBuild(id) {
        for (const f of families) {
            const b = (f.builds || []).find((x) => x.id === id);
            if (b) return b;
        }
        return null;
    }

    function renderEngineNotice() {
        replace(engineNotice, engineInstalled === false
            ? el('div', { class: 'notice notice-warn' }, icon('alert', { size: 18 }),
                el('span', { class: 'notice-text' }, 'The engine that runs local models is not installed yet. Models can be downloaded now and used once it is installed.'),
                el('button', { type: 'button', class: 'button-secondary button-small', onclick: () => gotoTab('engine', 'engine') }, 'Go to the Engine tab'))
            : null);
    }

    async function confirmDownload(build, { resume = false } = {}) {
        const fit = build.fit || {};
        const ahead = activeJob(DOWNLOAD_KIND);
        const folder = buildFolder(modelsDir, build);
        const space = el('span', null, 'Checking…');
        const have = Number(build.installed_bytes) || 0;
        const body = [
            defList([
                ['Model', build.label],
                ['Download size', fmtBytesExact(build.total_size_bytes)],
                resume && have > 0 ? ['Already downloaded', `${fmtBytes(have)}; ${fmtBytes(Math.max(0, build.total_size_bytes - have))} left`] : ['', undefined],
                ['Free disk space', space],
                ['Saved to', folder || 'The models folder'],
                ['Fit on this computer', el('span', null, badge(fit.label || 'Not checked', FIT_KIND[fit.verdict] || 'neutral'), fit.detail ? el('div', { class: 'fit-detail' }, fit.detail) : null)],
            ], 'kv kv-dialog'),
            fit.verdict === 'too_large' ? el('div', { class: 'notice notice-warn' }, icon('alert', { size: 18 }), el('span', null, 'This model is too large to run on this computer. You can still download it.')) : null,
            ahead ? el('p', { class: 'help-text' }, `${ahead.label || 'Another model'} is downloading now. This model will start when the downloads ahead of it finish.`) : null,
            el('p', { class: 'help-text' }, 'The download can be canceled and resumed later.'),
        ];
        // Free space is read fresh, so the number shown is current.
        api.get('/v1/system', { query: { refresh: '1' }, timeoutMs: 60000 }).then((report) => {
            const disk = (report && report.disk) || {};
            space.textContent = Number.isFinite(disk.free_bytes)
                ? `${fmtBytes(disk.free_bytes)} free at ${disk.checked_path || disk.models_dir || 'the models folder'}`
                : 'Unknown';
        }).catch(() => {
            space.textContent = 'Unknown';
        });
        const started = await actionDialog({
            title: resume ? `Resume downloading ${build.label}?` : `Download ${build.label}?`,
            body,
            confirmLabel: ahead ? 'Add to the download queue' : (resume ? 'Resume download' : `Download ${fmtBytes(build.total_size_bytes)}`),
            describe: plainError,
            action: async () => {
                const res = await api.post(`/v1/models/local/${encodeURIComponent(build.id)}/download`, { confirm_size_bytes: build.total_size_bytes }, { timeoutMs: 60000 });
                return (res && res.job) || true;
            },
        });
        if (started && started.id) {
            build.install_state = started.state === 'queued' ? 'queued' : 'downloading';
            build.download_job_id = started.id;
            trackJob(started);
        } else if (started) {
            refreshJobs();
        }
    }

    async function cancelDownload(build) {
        try {
            const res = await api.post(`/v1/models/local/${encodeURIComponent(build.id)}/download/cancel`, {});
            if (res && res.job) trackJob(res.job);
            const wasQueued = res && res.job && res.job.state === 'canceled' && !(Number(res.job.bytes_done) > 0);
            toast(wasQueued ? `${build.label} removed from the download queue.`
                : `Download of ${build.label} canceled. It can be resumed later.`, 'info');
        } catch (err) {
            toast(plainError(err), 'error');
            refreshJobs();
        }
    }

    async function removeBuild(build, { partial = false } = {}) {
        const removed = await actionDialog({
            title: partial ? `Delete the partial download of ${build.label}?` : `Remove ${build.label}?`,
            body: [
                partial
                    ? `The ${fmtBytes(build.installed_bytes)} downloaded so far is deleted. Downloading again starts from the beginning.`
                    : `This deletes the model files from this computer and frees about ${fmtBytes(build.installed_bytes || build.total_size_bytes)}. You can download the model again later.`,
            ],
            confirmLabel: partial ? 'Delete' : 'Remove',
            danger: true,
            describe: (err) => (err && err.status === 409
                ? `${plainError(err)} You can stop it on the Engine tab.`
                : plainError(err)),
            action: () => api.del(`/v1/models/local/${encodeURIComponent(build.id)}`, { timeoutMs: 120000 }),
        });
        if (removed) {
            toast(`${build.label} removed${Number.isFinite(removed.freed_bytes) ? `; ${fmtBytes(removed.freed_bytes)} freed` : ''}.`, 'success');
            emit('models-changed');
            load({ quiet: true });
        }
    }

    function smallButton(label, iconName, onclick, { danger = false, disabled = false, describedBy = null, primary = false } = {}) {
        return el('button', {
            type: 'button', class: [primary ? 'button' : 'button-secondary', 'button-small', danger ? 'button-danger' : ''],
            disabled, onclick, 'aria-describedby': describedBy,
        }, iconName ? icon(iconName, { size: 13 }) : null, label);
    }

    /** Fill the status and action cells of one build row for its current state. */
    function renderRow(row) {
        const { build } = row;
        const view = buildView(build);
        // Downloads queue up, so nothing here is blocked by another download.
        const queueNote = (job) => {
            const n = Number(job && job.queue_position);
            return Number.isInteger(n) && n > 0 ? (n === 1 ? 'Next in line' : `Number ${n} in line`) : 'Waiting in line';
        };
        // The key says when the cells must be rebuilt; progress numbers update in place.
        const key = [view.state, view.job ? `${view.job.id}:${view.job.state}:${view.job.queue_position ?? ''}` : '', build.installed_bytes].join('|');
        if (key !== row.key) {
            row.key = key;
            row.view = null;
            let status;
            let actions;
            let detail = null;
            if (view.state === 'queued') {
                status = [badge('Queued', 'neutral'), el('div', { class: 'cell-note' }, queueNote(view.job))];
                actions = [smallButton('Cancel', 'x', () => cancelDownload(build), { danger: true })];
            } else if (view.state === 'downloading') {
                status = badge(view.job.state === 'verifying' ? 'Verifying' : 'Downloading', 'info');
                actions = [smallButton('Cancel', 'x', () => cancelDownload(build), { danger: true })];
                row.view = progressView(`Download progress for ${build.label}`);
                detail = row.view.node;
            } else if (view.state === 'installed') {
                status = [badge('Installed', 'good'), el('div', { class: 'cell-note' }, `${fmtBytes(build.installed_bytes || build.total_size_bytes)} on disk`)];
                actions = [smallButton('Remove', 'trash', () => removeBuild(build), { danger: true })];
            } else if (view.state === 'partial') {
                status = [badge('Paused', 'warn'), el('div', { class: 'cell-note' }, `${fmtBytes(build.installed_bytes)} of ${fmtBytes(build.total_size_bytes)} downloaded`)];
                actions = [
                    smallButton('Resume download', 'download', () => confirmDownload(build, { resume: true }), { primary: true }),
                    smallButton('Delete', 'trash', () => removeBuild(build, { partial: true }), { danger: true }),
                ];
            } else if (view.state === 'error') {
                const problem = jobError(view.job);
                status = badge('Download failed', 'bad');
                actions = [
                    smallButton('Retry', 'refresh', () => confirmDownload(build, { resume: Number(build.installed_bytes) > 0 })),
                    Number(build.installed_bytes) > 0 ? smallButton('Delete', 'trash', () => removeBuild(build, { partial: true }), { danger: true }) : null,
                ];
                detail = el('div', { class: 'job-problem', role: 'alert' }, icon('alert', { size: 18 }),
                    el('div', null, el('div', { class: 'job-problem-title' }, view.job ? problem.message : 'The last download did not finish.'),
                        problem.hint && view.job ? el('div', { class: 'job-problem-hint' }, problem.hint) : null));
            } else {
                status = badge('Not installed', 'neutral');
                actions = [smallButton('Download', 'download', () => confirmDownload(build))];
            }
            replace(row.status, status);
            replace(row.action, el('div', { class: 'row-actions' }, actions));
            replace(row.progressCell, detail);
            row.progressRow.hidden = !detail;
        }
        if (row.view && view.job) row.view.update(view.job);
    }

    function buildRows(build) {
        const fit = build.fit || {};
        const status = el('td', { class: 'build-status' });
        const action = el('td', { class: 'build-action' });
        const progressCell = el('td', { colspan: '6' });
        const progressRow = el('tr', { class: 'build-detail-row', hidden: true }, progressCell);
        const main = el('tr', { class: build.recommended ? 'is-recommended' : null },
            el('th', { scope: 'row', class: 'build-name' }, el('strong', null, build.label),
                build.recommended ? el('div', { class: 'recommended-mark' }, icon('star', { size: 13 }), 'Recommended for this computer') : null),
            el('td', null, fmtBytes(build.total_size_bytes)),
            el('td', null, build.quant || '—'),
            el('td', null, badge(fit.label || 'Not checked', FIT_KIND[fit.verdict] || 'neutral'), fit.detail ? el('div', { class: 'fit-detail' }, fit.detail) : null),
            status, action);
        const row = { build, key: null, status, action, progressRow, progressCell, view: null };
        rows.set(build.id, row);
        renderRow(row);
        return [main, progressRow];
    }

    function familyCard(family) {
        const roles = (family.roles || []).map((r) => ROLE_LABEL[r] || r).join(' and ');
        return el('div', { class: 'card family-card' },
            el('div', { class: 'card-header' }, el('h3', null, family.label), roles ? badge(roles, 'good') : null,
                el('span', { class: 'family-meta' }, [family.developer, family.license].filter(Boolean).join(' · ')),
                el('span', { class: 'spacer' }),
                family.homepage ? externalLink('Model page', family.homepage) : null),
            family.summary ? el('p', { class: 'family-summary' }, family.summary) : null,
            (family.builds || []).length
                ? el('div', { class: 'table-wrap' }, el('table', { class: 'data build-table' },
                    el('thead', null, el('tr', null, ['Model', 'Download size', 'Quantization', 'Fit on this computer', 'Status', ''].map((t) => el('th', { scope: 'col' }, t || el('span', { class: 'visually-hidden' }, 'Actions'))))),
                    el('tbody', null, family.builds.map(buildRows))))
                : el('p', { class: 'help-text' }, 'No builds listed.'));
    }

    function renderLocal() {
        rows.clear();
        const ready = families.filter((f) => f.status !== 'planned');
        const planned = families.filter((f) => f.status === 'planned');
        replace(localBody,
            engineNotice,
            families.length ? null : stateBox('empty', 'No local models are listed'),
            ready.map(familyCard),
            planned.length ? el('div', { class: 'planned-grid' }, planned.map(plannedCard)) : null);
        renderEngineNotice();
    }

    async function loadEngineState() {
        try {
            const engine = await api.get('/v1/engine', { timeoutMs: 30000 });
            engineInstalled = Boolean(engine && engine.managed && engine.managed.installed);
            if (engine && engine.models_dir) modelsDir = engine.models_dir;
        } catch (_) {
            engineInstalled = null; // unknown: say nothing rather than guess
        }
        renderEngineNotice();
    }

    let loading = null;
    let loadedAt = 0;
    function load({ quiet = false } = {}) {
        if (loading) return loading;
        loading = (async () => {
            if (!quiet) {
                panel.dataset.state = 'loading';
                replace(localBody, stateBox('loading', 'Loading local models…'));
            }
            try {
                const [data] = await Promise.all([api.get('/v1/models'), refreshJobs(), loadEngineState()]);
                families = (data.local && data.local.families) || [];
                renderLocal();
                const openRouter = ((data.api && data.api.providers) || []).find((p) => p.kind === 'openrouter');
                if (openRouter && !gallery) gallery = mountOpenRouterGallery(remoteBody, openRouter);
                if (!openRouter) replace(remoteBody, stateBox('empty', 'OpenRouter is not listed'));
                if (gallery && activeTab === 'remote') gallery.load({ quiet: gallery.loaded() });
                panel.dataset.state = 'ready';
                loadedAt = Date.now();
            } catch (err) {
                if (quiet && families.length) {
                    toastError(err, 'The model list could not be refreshed');
                } else {
                    inlineError(localBody, err, { title: 'Models could not be loaded', retry: () => load() });
                    panel.dataset.state = 'error';
                }
            } finally {
                loading = null;
            }
        })();
        return loading;
    }

    on('jobs-changed', () => {
        for (const row of rows.values()) renderRow(row);
    });
    on('job-finished', (job) => {
        if (job.kind === 'engine_install') {
            loadEngineState();
            if (job.state === 'done') load({ quiet: true });
            return;
        }
        if (job.kind !== DOWNLOAD_KIND) return;
        const build = findBuild(job.target);
        const name = (build && build.label) || job.label || 'The model';
        if (job.state === 'done') toast(`${name} is downloaded and verified.`, 'success');
        else if (job.state === 'error') toast(`${name} did not download: ${jobError(job).message}`, 'error');
        emit('models-changed');
        load({ quiet: true });
    });
    on('engine-changed', () => load({ quiet: true }));

    // One container in the existing style; its header is the two tabs.
    // The models folder and the space the downloaded models use, above the model list.
    const folderHost = el('div');
    const folderSection = el('section', { class: 'mf-section', 'aria-label': 'Models folder' },
        el('h3', { class: 'mf-section-title' }, 'Models folder'), folderHost);
    mountModelsFolder(folderHost);
    const bodies = { local: el('div', { class: 'privacy-body', role: 'tabpanel', id: 'models-panel-local' }, folderSection, localBody),
        remote: el('div', { class: 'privacy-body', role: 'tabpanel', id: 'models-panel-remote', hidden: true }, remoteBody) };
    const tabButtons = {};
    const section = el('section', { class: 'privacy-section is-local' });
    function showTab(id, { focus = false } = {}) {
        activeTab = id;
        section.className = `privacy-section is-${id}`;
        for (const tab of SUBTABS) {
            const on = tab.id === id;
            tabButtons[tab.id].setAttribute('aria-selected', String(on));
            tabButtons[tab.id].tabIndex = on ? 0 : -1;
            bodies[tab.id].hidden = !on;
        }
        if (focus) tabButtons[id].focus();
        if (id === 'remote' && gallery) gallery.load({ quiet: gallery.loaded() });
    }
    for (const tab of SUBTABS) {
        tabButtons[tab.id] = el('button', {
            type: 'button', role: 'tab', class: `privacy-tab is-${tab.id}`, id: `models-tab-${tab.id}`, 'aria-controls': `models-panel-${tab.id}`,
            onclick: () => showTab(tab.id),
            onkeydown: (e) => {
                if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
                e.preventDefault();
                showTab(tab.id === 'local' ? 'remote' : 'local', { focus: true });
            },
        }, icon(tab.icon, { size: 34 }), el('span', { class: 'privacy-tab-text' }, el('span', { class: 'privacy-tab-title' }, tab.title), el('span', { class: 'privacy-tab-sub' }, tab.subtitle)));
        bodies[tab.id].setAttribute('aria-labelledby', `models-tab-${tab.id}`);
    }
    replace(section, el('div', { class: 'privacy-tabs', role: 'tablist', 'aria-label': 'Where models run' }, SUBTABS.map((t) => tabButtons[t.id])),
        bodies.local, bodies.remote);
    replace(panel, section);
    showTab('local');

    let stale = false;
    on('providers-changed', () => {
        stale = true;
    });
    await load();

    return {
        show(target) {
            if (stale) {
                stale = false;
                load();
            } else if (Date.now() - loadedAt > 2000) {
                load({ quiet: true });
            }
            // A link to a provider (e.g. from the Keys tab) opens the OpenRouter tab.
            if (target) showTab(target === 'local' ? 'local' : 'remote');
        },
    };
}
