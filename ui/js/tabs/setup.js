// Setup page: a short walk-through of what VoucherVisionLocal needs on this
// computer. It has no button in the tab bar; the version number in the header
// opens it, and so does the first start of the app.
//
//   Step 1  How will you run models?  OpenRouter | On this computer | Both
//           (saved as settings.setup.usage)
//   Step 2  A checklist of only the steps that choice needs, each with one action.
//
// Everything here reuses the routes and helpers the other tabs use. Apart from
// the label detector, which downloads by itself at first start, nothing is
// installed or downloaded without a click.

import { api, plainError, toast } from '../api.js';
import { el, icon, replace, externalLink, fmtBytes, fmtMemory } from '../dom.js';
import { emit, getAppInfo, gotoTab, loadSettings, on, updateCachedSettings } from '../state.js';
import { activeJob, allJobs, isActive, jobError, progressView, refreshJobs, trackJob } from '../jobs.js';
import { mountModelsFolder } from '../models_folder.js';
import { describeUpdate, installUpdate, onUpdate } from '../updates.js';

const CHOICES = [
    {
        id: 'openrouter', icons: ['cloud'], title: 'OpenRouter',
        text: 'Cloud models through one key.',
        facts: ['Pay only for what you use', 'Nothing large to download'],
        tip: 'Images are sent to the model’s provider through OpenRouter, with your own key.',
    },
    {
        id: 'local', icons: ['computer'], title: 'On this computer',
        text: 'Private, and free to run.',
        facts: ['Needs a capable graphics card', 'Large downloads (5 to 40 GB)'],
        tip: 'Images never leave this computer. Models run in the llama.cpp engine.',
    },
    {
        id: 'both', icons: ['cloud', 'computer'], title: 'Both',
        text: 'Pick either one for each job.',
        facts: ['The most choice', 'Set up both'],
        tip: 'Set up OpenRouter and local models, then choose per run on the VVL tab.',
    },
];

const STATUS_LABEL = { done: 'Done', todo: 'To do', working: 'Working', problem: 'Needs attention' };
const ENGINE_KIND = 'engine_install';
const DOWNLOAD_KIND = 'model_download';
const DETECTOR_KIND = 'detector_download';
const LOW_DISK_BYTES = 20e9;

/** How well local models will run, in plain words. {status, text} */
function verdict(report) {
    const cap = (report && report.capability) || {};
    switch (cap.tier) {
        case 'workstation':
        case 'high':
            return { good: true, short: 'Good fit', text: 'This computer runs local models well, including large ones.' };
        case 'mid':
            return { good: true, short: 'Good fit', text: 'This computer runs small and medium local models well.' };
        case 'entry':
            return { good: false, short: 'Small models only', text: 'This computer can run small local models. Larger ones will be slow.' };
        case 'cpu_only':
            return { good: false, short: 'No usable graphics card', text: 'No graphics card that local models can use was found. They would run slowly; OpenRouter may suit this computer better.' };
        default:
            return { good: false, short: 'Not checked', text: 'How well local models will run here could not be checked.' };
    }
}

function lastJob(kind) {
    const mine = allJobs().filter((j) => j.kind === kind);
    return mine.find(isActive) || mine[mine.length - 1] || null;
}

/** The build to suggest: a recommended one, from the Qwen3-VL family when it has one. */
function suggestedBuild(families) {
    const ready = (families || []).filter((f) => f.status !== 'planned');
    const recommended = ready.flatMap((f) => (f.builds || []).filter((b) => b.recommended).map((b) => ({ family: f, build: b })));
    const qwen = recommended.find((r) => /^qwen3-?vl/i.test(r.family.id || '') || /^qwen3-vl/i.test(r.family.label || ''));
    return (qwen || recommended[0] || {}).build || null;
}

function tipMark(text) {
    if (!text) return null;
    return el('span', { class: 'setup-tip', tabindex: '0', role: 'img', 'aria-label': text, 'data-tip': text }, icon('info', { size: 14 }));
}

export async function mount(panel) {
    const info = getAppInfo() || {};

    // ------------------------------------------------------------ data
    const data = {
        usage: undefined,     // settings.setup.usage once loaded (null = not chosen)
        settings: null,
        providers: null,      // GET /v1/providers -> providers[]
        system: null,         // GET /v1/system
        engine: null,         // GET /v1/engine
        families: null,       // GET /v1/models -> local.families
        detector: null,       // GET /v1/detector
    };
    const failed = {};        // source -> error, when its last load failed
    const ui = {
        keyBusy: false, keyOpen: false, keyProblem: null,
        engineBusy: false, engineProblem: null,
        modelBusy: false, modelProblem: null,
        detectorBusy: false, detectorProblem: null,
        folderOpen: false, choosing: null,
    };
    let update = { status: 'unavailable' };

    const wantsCloud = () => data.usage === 'openrouter' || data.usage === 'both';
    const wantsLocal = () => data.usage === 'local' || data.usage === 'both';
    const openRouter = () => (data.providers || []).find((p) => p.id === 'openrouter') || null;

    async function loadSource(name, fn) {
        try {
            data[name] = await fn();
            delete failed[name];
        } catch (err) {
            failed[name] = err;
        }
    }
    const loaders = {
        settings: () => loadSource('settings', async () => {
            const s = await loadSettings({ force: true });
            data.usage = (s && s.setup && s.setup.usage) || null;
            return s;
        }),
        providers: () => loadSource('providers', async () => ((await api.get('/v1/providers', { timeoutMs: 15000 })).providers || [])),
        system: (refresh = false) => loadSource('system', () => api.get('/v1/system', { query: refresh ? { refresh: '1' } : undefined, timeoutMs: 120000 })),
        engine: () => loadSource('engine', () => api.get('/v1/engine', { timeoutMs: 30000 })),
        detector: () => loadSource('detector', () => api.get('/v1/detector', { timeoutMs: 15000 })),
        models: () => loadSource('families', async () => {
            const models = await api.get('/v1/models', { timeoutMs: 60000 });
            return (models && models.local && models.local.families) || [];
        }),
    };

    async function loadAll() {
        await loaders.settings();
        const work = [loaders.providers(), loaders.system(), loaders.detector(), refreshJobs()];
        if (wantsLocal()) work.push(loaders.engine(), loaders.models());
        await Promise.all(work);
    }

    // ------------------------------------------------------------ persistent parts
    // Kept across re-renders, so typing and progress bars are not reset.
    const keyInput = el('input', {
        type: 'password', id: 'setupOpenRouterKey', placeholder: 'Paste your OpenRouter key (starts with sk-or-)',
        autocomplete: 'off', spellcheck: 'false', autocapitalize: 'off', 'aria-label': 'OpenRouter API key',
    });
    keyInput.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') saveKey();
    });
    const keyToggle = el('button', { type: 'button', class: 'icon-button', 'aria-label': 'Show what you typed', 'aria-pressed': 'false', title: 'Show or hide what you typed' }, icon('eye', { size: 16 }));
    keyToggle.addEventListener('click', () => {
        const show = keyInput.type === 'password';
        keyInput.type = show ? 'text' : 'password';
        keyToggle.setAttribute('aria-pressed', String(show));
        keyToggle.setAttribute('aria-label', show ? 'Hide what you typed' : 'Show what you typed');
        replace(keyToggle, icon(show ? 'eyeOff' : 'eye', { size: 16 }));
    });
    const folderHost = el('div', { class: 'setup-folder-card' });
    let folderMounted = false;
    const engineProgress = progressView('Engine install progress');
    const modelProgress = progressView('Model download progress');
    const detectorProgress = progressView('Label detector download progress');

    // ------------------------------------------------------------ actions
    async function choose(usage) {
        if (ui.choosing || usage === data.usage) return;
        const before = data.usage;
        ui.choosing = usage;
        data.usage = usage;
        render();
        try {
            const saved = await api.put('/v1/settings', { setup: { usage } });
            const { warnings: _w, ...settings } = saved || {};
            updateCachedSettings(settings);
            data.settings = settings;
            if (wantsLocal() && (!data.engine || !data.families)) await Promise.all([loaders.engine(), loaders.models()]);
        } catch (err) {
            data.usage = before;
            toast(`The choice was not saved: ${plainError(err)}`, 'error');
        } finally {
            ui.choosing = null;
            render();
        }
    }

    async function saveKey() {
        const value = keyInput.value.trim();
        keyInput.value = '';
        keyInput.type = 'password';
        if (!value) {
            toast('Paste a key first.', 'warning');
            return;
        }
        ui.keyBusy = true;
        ui.keyProblem = null;
        render();
        try {
            await api.put('/v1/keys/openrouter', { fields: { api_key: value } });
            const result = await api.post('/v1/keys/openrouter/test', {}, { timeoutMs: 30000 });
            const p = openRouter();
            if (p) Object.assign(p, { configured: true, last_test: result });
            if (result && result.ok) {
                ui.keyOpen = false;
                toast('OpenRouter key saved. It works.', 'success');
            }
            emit('providers-changed');
            emit('models-changed');
        } catch (err) {
            ui.keyProblem = plainError(err);
        } finally {
            ui.keyBusy = false;
            await loaders.providers();
            render();
        }
    }

    async function testKey() {
        ui.keyBusy = true;
        ui.keyProblem = null;
        render();
        try {
            const result = await api.post('/v1/keys/openrouter/test', {}, { timeoutMs: 30000 });
            const p = openRouter();
            if (p) p.last_test = result;
        } catch (err) {
            ui.keyProblem = plainError(err);
        } finally {
            ui.keyBusy = false;
            render();
        }
    }

    async function installEngine(variantId = null) {
        ui.engineBusy = true;
        ui.engineProblem = null;
        render();
        try {
            const res = await api.post('/v1/engine/install', variantId ? { variant: variantId } : {}, { timeoutMs: 60000 });
            if (res && res.job) trackJob(res.job);
        } catch (err) {
            ui.engineProblem = plainError(err);
        } finally {
            ui.engineBusy = false;
            await loaders.engine();
            render();
        }
    }

    async function installDetector() {
        ui.detectorBusy = true;
        ui.detectorProblem = null;
        render();
        try {
            const res = await api.post('/v1/detector/install', {}, { timeoutMs: 60000 });
            if (res && res.job) trackJob(res.job);
        } catch (err) {
            ui.detectorProblem = plainError(err);
        } finally {
            ui.detectorBusy = false;
            await loaders.detector();
            render();
        }
    }

    async function downloadModel(build) {
        ui.modelBusy = true;
        ui.modelProblem = null;
        render();
        try {
            const res = await api.post(`/v1/models/local/${encodeURIComponent(build.id)}/download`,
                { confirm_size_bytes: build.total_size_bytes }, { timeoutMs: 60000 });
            if (res && res.job) trackJob(res.job);
        } catch (err) {
            ui.modelProblem = plainError(err);
        } finally {
            ui.modelBusy = false;
            render();
        }
    }

    // ------------------------------------------------------------ checklist items
    // Each item: {id, status, title, text, tip, action, extra, path, facts, required, ok}
    // Every `required` item must be `ok` before "You're ready" shows.
    function button(label, onclick, { primary = true, iconName = null, disabled = false } = {}) {
        return el('button', { type: 'button', class: primary ? 'button' : 'button-secondary', onclick, disabled },
            iconName ? icon(iconName, { size: 15 }) : null, label);
    }

    function loadingItem(id, title, source, retry) {
        if (failed[source]) {
            return { id, status: 'problem', title, text: 'This could not be checked.', tip: plainError(failed[source]),
                action: button('Try again', async () => { await retry(); render(); }, { primary: false }), required: true };
        }
        return { id, status: 'working', title, text: 'Checking…', required: true };
    }

    function appItem() {
        const ready = update.status === 'ready';
        return {
            id: 'app', status: 'done', title: 'VoucherVisionLocal',
            text: ready ? `Version ${info.version}. A new version is ready; restart to update.` : `Version ${info.version || 'unknown'} is installed.`,
            tip: info.packaged ? 'The installed desktop app.' : 'Running from a source checkout.',
        };
    }

    function pythonItem() {
        return {
            id: 'python', status: 'done', title: 'Python environment',
            text: info.python ? `Python ${info.python} is working.` : 'Python is working.',
            tip: info.packaged ? 'Installed and kept up to date by the app.' : 'The project environment, managed by uv.',
        };
    }

    function detectorItem() {
        const title = 'Label detector';
        if (!data.detector) return loadingItem('detector', title, 'detector', loaders.detector);
        const d = data.detector;
        const job = activeJob(DETECTOR_KIND);
        const size = fmtBytes(d.size_bytes);
        const tip = 'Finds the labels on each sheet, so only they are sent for OCR (the LeafMachine3 archival detector). '
            + `It runs on this computer with ${d.runtime}.`;
        if (job || ui.detectorBusy || d.state === 'downloading') {
            if (job) detectorProgress.update(job);
            return { id: 'detector', status: 'working', title, text: `Downloading the label detector (${size})…`,
                tip: `${tip} Until it is ready, whole images are used instead of label collages.`, extra: job ? detectorProgress.node : null, required: true };
        }
        if (ui.detectorProblem) {
            return { id: 'detector', status: 'problem', title, text: ui.detectorProblem, action: button('Try again', installDetector), required: true };
        }
        if (d.state === 'installed' && d.load_error) {
            return { id: 'detector', status: 'problem', title, text: 'The label detector is downloaded but could not be loaded.', tip: d.load_error, required: true };
        }
        if (d.state === 'installed') {
            return { id: 'detector', status: 'done', title, text: `Ready (${d.runtime}).`, tip };
        }
        if (d.state === 'error') {
            const e = jobError(d.job);
            return { id: 'detector', status: 'problem', title, text: `The download stopped: ${e.message}`, tip: e.hint || tip,
                action: button('Try again', installDetector), required: true };
        }
        return { id: 'detector', status: 'todo', title, text: `Download the label detector (${size}). Until then, whole images are used instead of label collages.`,
            tip, action: button('Download', installDetector), required: true };
    }

    function keyItem() {
        const title = 'OpenRouter key';
        if (!data.providers) return loadingItem('key', title, 'providers', loaders.providers);
        const p = openRouter();
        if (!p) return { id: 'key', status: 'problem', title, text: 'OpenRouter is not listed in this version.', required: true };
        const masked = ((p.fields || []).find((f) => f.saved && f.masked) || {}).masked;
        const test = p.last_test;
        const getKey = p.key_url ? externalLink('Get a key', p.key_url) : null;
        const form = el('div', { class: 'setup-key' },
            el('div', { class: 'key-input-row' }, keyInput, keyToggle,
                button(ui.keyBusy ? 'Checking…' : 'Save and check', saveKey, { disabled: ui.keyBusy })),
            el('div', { class: 'setup-key-links' }, getKey,
                el('span', { class: 'help-text' }, 'Checking a key is free. It is kept only on this computer.')));
        keyInput.disabled = ui.keyBusy;

        if (ui.keyBusy) return { id: 'key', status: 'working', title, text: 'Saving and checking the key…', extra: form, required: true };
        if (ui.keyProblem) {
            return { id: 'key', status: 'problem', title, text: `The key was not checked: ${ui.keyProblem}`, extra: form, required: true };
        }
        if (!p.configured) {
            return { id: 'key', status: 'todo', title, text: 'Paste your OpenRouter key.', tip: 'One key reaches Claude, GPT, Gemini, Qwen and hundreds of other models. OpenRouter bills your account for what you use.', extra: form, required: true };
        }
        const shown = masked ? ` (${masked})` : '';
        if (test && test.status === 'invalid_key') {
            return { id: 'key', status: 'problem', title, text: `OpenRouter did not accept the saved key${shown}. Paste a new one.`, extra: form, required: true };
        }
        if (test && !test.ok) {
            return { id: 'key', status: 'problem', title, text: test.message || 'The key could not be checked.', tip: 'Check the internet connection, then try again.',
                action: button('Check again', testKey, { primary: false }), required: true };
        }
        return {
            id: 'key', status: 'done', title,
            text: test && test.ok ? `Key saved and working${shown}.` : `Key saved${shown}.`,
            action: ui.keyOpen
                ? button('Cancel', () => { ui.keyOpen = false; render(); }, { primary: false })
                : button('Replace', () => { ui.keyOpen = true; render(); keyInput.focus(); }, { primary: false }),
            extra: ui.keyOpen ? form : null,
            required: true,
        };
    }

    function pickItem() {
        const models = (data.settings && data.settings.openrouter && data.settings.openrouter.models) || [];
        const go = () => gotoTab('models', 'remote');
        if (models.length) {
            return { id: 'pick', status: 'done', title: 'Pick models', text: `${models.length} OpenRouter model${models.length === 1 ? '' : 's'} added.`,
                action: button('Change', go, { primary: false }), required: true };
        }
        return { id: 'pick', status: 'todo', title: 'Pick models', text: 'Choose which OpenRouter models to use.',
            tip: 'The Models tab lists every model with its price per 1,000 images.', action: button('Pick models', go), required: true };
    }

    function systemItem() {
        const title = 'Check this computer';
        if (!data.system) {
            return { ...loadingItem('system', title, 'system', () => loaders.system(true)), required: false };
        }
        const r = data.system;
        const v = verdict(r);
        const cap = r.capability || {};
        const disk = r.disk || {};
        const mem = r.memory || {};
        const gpus = Array.isArray(r.gpus) ? r.gpus : [];
        const facts = [];
        if (gpus.length) {
            const g = gpus[0];
            const each = gpus.length > 1 && gpus.every((x) => x.vram_total_bytes === g.vram_total_bytes && x.name === g.name);
            facts.push(each ? `${gpus.length}× ${g.name} (${fmtMemory(g.vram_total_bytes)} each)`
                : gpus.map((x) => `${x.name}${Number.isFinite(x.vram_total_bytes) ? ` (${fmtMemory(x.vram_total_bytes)})` : ''}`).join(', '));
        } else {
            facts.push('No graphics card found');
        }
        if (Number.isFinite(mem.total_bytes)) facts.push(`${fmtMemory(mem.total_bytes)} RAM`);
        const lowDisk = Number.isFinite(disk.free_bytes) && disk.free_bytes < LOW_DISK_BYTES;
        if (Number.isFinite(disk.free_bytes)) facts.push(`${fmtBytes(disk.free_bytes)} free for models`);
        let text = v.text;
        if (lowDisk) text += ' The models folder’s drive is nearly full.';
        return {
            id: 'system', status: v.good && !lowDisk ? 'done' : 'problem', title, text,
            tip: [cap.summary, cap.backend_reason].filter(Boolean).join('. ') || null,
            facts,
            action: button('Check again', async () => {
                data.system = null;
                render();
                await loaders.system(true);
                render();
            }, { primary: false }),
            required: false,
        };
    }

    function folderItem() {
        const dir = (data.settings && data.settings.models_dir) || (data.engine && data.engine.models_dir) || '';
        const synced = data.system && data.system.disk && data.system.disk.is_synced_folder;
        if (ui.folderOpen && !folderMounted) {
            folderMounted = true;
            mountModelsFolder(folderHost, {
                onSaved: async () => {
                    await Promise.all([loaders.settings(), loaders.system(true), loaders.engine(), loaders.models()]);
                    render();
                },
            });
        }
        return {
            id: 'folder', status: synced ? 'problem' : 'done', title: 'Model folder',
            text: synced ? 'This folder is inside a synced folder such as Dropbox. Choose another one.' : 'Downloaded models are kept here.',
            path: ui.folderOpen ? null : dir,
            tip: 'Model files are large. Choose a drive with plenty of free space, outside any synced folder.',
            action: button(ui.folderOpen ? 'Close' : 'Change', () => { ui.folderOpen = !ui.folderOpen; render(); }, { primary: false }),
            extra: ui.folderOpen ? folderHost : null,
            required: false,
        };
    }

    function engineItem() {
        const title = 'llama.cpp engine';
        if (!data.engine) return loadingItem('engine', title, 'engine', loaders.engine);
        const e = data.engine;
        const m = e.managed || {};
        const rec = e.recommended_variant;
        const job = activeJob(ENGINE_KIND);
        const details = button('Details', () => gotoTab('engine', 'engine'), { primary: false });
        if (job || ui.engineBusy) {
            if (job) engineProgress.update(job);
            return { id: 'engine', status: 'working', title, text: job ? 'Installing the engine…' : 'Starting the install…',
                tip: `${job && job.label ? `${job.label}. ` : ''}After the download, the engine is tested on this computer.`, extra: job ? engineProgress.node : null, required: true };
        }
        if (ui.engineProblem) {
            return { id: 'engine', status: 'problem', title, text: ui.engineProblem,
                action: button('Try again', () => installEngine(m.installed && m.variant ? m.variant.id : null)), required: true };
        }
        if (m.installed) {
            const variant = m.variant && m.variant.label ? m.variant.label : null;
            if (m.pinned_version && m.version !== m.pinned_version) {
                return { id: 'engine', status: 'todo', title, text: `Engine update available: ${m.pinned_version}.`,
                    tip: `Installed: llama.cpp ${m.version || 'unknown'}${variant ? ` (${variant})` : ''}. The update is the version this release of VoucherVisionLocal was tested with.`,
                    action: button('Update', () => installEngine(m.variant ? m.variant.id : null), { iconName: 'download' }), required: true, ok: true };
            }
            return { id: 'engine', status: 'done', title, text: m.version ? `Installed, version ${m.version}.` : 'Installed.',
                tip: variant ? `${variant}. It starts by itself when a local model is used.` : null, action: details, required: true };
        }
        const last = lastJob(ENGINE_KIND);
        if (last && last.state === 'error') {
            const problem = jobError(last);
            const nextId = problem.details && problem.details.next_variant;
            const next = nextId ? (e.variants || []).find((v) => v.id === nextId) : null;
            return { id: 'engine', status: 'problem', title,
                text: problem.code === 'ENGINE_SELF_TEST_FAILED' ? 'That engine build cannot run on this computer.' : 'The install did not finish.',
                tip: [problem.message, problem.hint].filter(Boolean).join(' '),
                action: next ? button(`Install ${next.label}`, () => installEngine(next.id), { iconName: 'download' })
                    : button('Try again', () => installEngine(last.target || null)),
                required: true };
        }
        if (!rec) {
            return { id: 'engine', status: 'problem', title, text: e.unavailable_reason || 'No engine build fits this computer.', action: details, required: true };
        }
        return { id: 'engine', status: 'todo', title, text: 'Install the program that runs models on this computer.',
            tip: `${rec.label}${rec.reason ? `. ${rec.reason}` : ''}`,
            action: button(`Install (${fmtBytes(rec.download_size_bytes)})`, () => installEngine(rec.id), { iconName: 'download' }), required: true };
    }

    function modelItem() {
        const title = 'A model';
        if (!data.families) return loadingItem('model', title, 'families', loaders.models);
        const builds = data.families.flatMap((f) => f.builds || []);
        const installed = builds.filter((b) => b.install_state === 'installed');
        const other = el('button', { type: 'button', class: 'link-button', onclick: () => gotoTab('models', 'local') }, 'Choose a different model');
        const job = activeJob(DOWNLOAD_KIND);
        if (job || ui.modelBusy) {
            if (job) modelProgress.update(job);
            return { id: 'model', status: 'working', title, text: job ? `Downloading ${job.label || 'a model'}…` : 'Starting the download…',
                tip: 'The download can be paused and resumed on the Models tab.', extra: job ? modelProgress.node : null, required: true };
        }
        if (installed.length) {
            const more = installed.length - 1;
            return { id: 'model', status: 'done', title, text: `${installed[0].label}${more ? ` and ${more} more` : ''} ${more ? 'are' : 'is'} downloaded.`,
                action: button('Models tab', () => gotoTab('models', 'local'), { primary: false }), required: true };
        }
        const build = suggestedBuild(data.families);
        if (!build) {
            return { id: 'model', status: 'todo', title, text: 'Choose a model to download.', action: button('Choose a model', () => gotoTab('models', 'local')), required: true };
        }
        const fit = build.fit || {};
        const size = fmtBytes(build.total_size_bytes);
        const partial = build.install_state === 'partial' || Number(build.installed_bytes) > 0;
        const last = lastJob(DOWNLOAD_KIND);
        const failedJob = last && last.state === 'error' && last.target === build.id ? jobError(last) : null;
        const problem = ui.modelProblem || (failedJob && [failedJob.message, failedJob.hint].filter(Boolean).join(' '));
        return {
            id: 'model', status: problem ? 'problem' : 'todo', title,
            text: problem || `Recommended for this computer: ${build.label}.`,
            tip: [`${size} download`, fit.label, fit.detail].filter(Boolean).join('. '),
            action: button(partial ? 'Resume download' : `Download (${size})`, () => downloadModel(build), { iconName: 'download' }),
            extra: el('div', { class: 'setup-item-links' }, other),
            required: true,
        };
    }

    function items() {
        // The app, Python and the label detector are needed whatever the choice, so they show before it.
        const list = [appItem(), pythonItem(), detectorItem()];
        if (wantsCloud()) list.push(keyItem(), pickItem());
        if (wantsLocal()) list.push(systemItem(), folderItem(), engineItem(), modelItem());
        return list;
    }

    // ------------------------------------------------------------ rendering
    const updateLine = el('div', { class: 'setup-update', role: 'status' });
    const choicesHost = el('div', { class: 'setup-choices', role: 'radiogroup', 'aria-label': 'How will you run models?' });
    const checklistHost = el('div');
    const localHint = () => {
        if (!data.system) return null;
        const v = verdict(data.system);
        return el('span', { class: ['setup-choice-hint', v.good ? 'is-good' : 'is-warn'] }, icon(v.good ? 'check' : 'alert', { size: 13 }), `This computer: ${v.short.toLowerCase()}`);
    };

    function renderUpdate() {
        const d = describeUpdate(update);
        const lead = d.kind === 'working' ? el('span', { class: 'spinner spinner-small', 'aria-hidden': 'true' })
            : icon({ good: 'check', ready: 'download', bad: 'alert' }[d.kind] || 'info', { size: 15 });
        replace(updateLine,
            el('span', { class: 'setup-version' }, `Version ${info.version || 'unknown'}`),
            el('span', { class: 'setup-dot', 'aria-hidden': 'true' }, '·'),
            el('span', { class: `setup-update-state is-${d.kind}` }, lead, d.text),
            d.kind === 'ready' ? el('button', { type: 'button', class: 'button button-small', onclick: () => installUpdate() }, 'Restart to update') : null,
            tipMark(d.tip));
    }

    function renderChoices() {
        replace(choicesHost, CHOICES.map((c) => {
            const on = data.usage === c.id;
            return el('button', {
                type: 'button', role: 'radio', class: ['setup-choice', on ? 'is-selected' : ''], 'aria-checked': String(on),
                disabled: Boolean(ui.choosing), dataset: { usage: c.id }, onclick: () => choose(c.id),
            },
            el('span', { class: 'setup-choice-icons' }, c.icons.map((n) => icon(n, { size: 26 }))),
            el('span', { class: 'setup-choice-title' }, c.title, on ? icon('check', { size: 16, className: 'icon setup-choice-check' }) : null),
            el('span', { class: 'setup-choice-text' }, c.text),
            el('span', { class: 'setup-choice-facts' }, c.facts.map((f) => el('span', null, f))),
            c.id === 'local' ? localHint() : null,
            el('span', { class: 'visually-hidden' }, ` ${c.tip}`));
        }));
    }

    function statusIcon(status, n) {
        const label = STATUS_LABEL[status];
        if (status === 'working') return el('span', { class: 'setup-status is-working', role: 'img', 'aria-label': label }, el('span', { class: 'spinner', 'aria-hidden': 'true' }));
        if (status === 'done') return el('span', { class: 'setup-status is-done', role: 'img', 'aria-label': label }, icon('check', { size: 16 }));
        if (status === 'problem') return el('span', { class: 'setup-status is-problem', role: 'img', 'aria-label': label }, icon('alert', { size: 15 }));
        return el('span', { class: 'setup-status is-todo', role: 'img', 'aria-label': label }, el('span', { 'aria-hidden': 'true' }, String(n)));
    }

    function renderChecklist() {
        if (data.usage === undefined) {
            replace(checklistHost, el('p', { class: 'help-text' }, 'Loading…'));
            return;
        }
        const chosen = data.usage !== null;
        const list = items();
        const required = list.filter((i) => i.required);
        const ready = chosen && required.length > 0 && required.every((i) => i.ok || (i.ok === undefined && i.status === 'done'));
        const doneCount = list.filter((i) => i.status === 'done').length;
        const keepFocus = document.activeElement === keyInput;
        replace(checklistHost,
            chosen ? el('p', { class: 'setup-progress' }, `${doneCount} of ${list.length} done`) : null,
            el('ol', { class: 'setup-list' }, list.map((item, i) => el('li', { class: ['setup-item', `is-${item.status}`], dataset: { step: item.id } },
                statusIcon(item.status, i + 1),
                el('div', { class: 'setup-item-body' },
                    el('div', { class: 'setup-item-title' }, item.title, tipMark(item.tip)),
                    el('div', { class: 'setup-item-text' }, item.text),
                    item.path ? el('code', { class: 'setup-path' }, item.path) : null,
                    item.facts && item.facts.length ? el('div', { class: 'setup-facts' }, item.facts.map((f) => el('span', null, f))) : null,
                    item.extra || null),
                item.action ? el('div', { class: 'setup-item-action' }, item.action) : null))),
            chosen ? null : el('p', { class: 'setup-empty' }, 'Choose an option above to see the rest of what is needed.'),
            ready
                ? el('div', { class: 'setup-ready', role: 'status' },
                    el('span', { class: 'setup-ready-icon', 'aria-hidden': 'true' }, icon('check', { size: 22 })),
                    el('div', { class: 'setup-ready-text' },
                        el('strong', null, 'You’re ready'),
                        el('span', null, 'Everything this choice needs is set up.')),
                    el('button', { type: 'button', class: 'button', onclick: () => gotoTab('vvl') }, 'Go to VVL', icon('play', { size: 14 })))
                : null);
        if (keepFocus && keyInput.isConnected) keyInput.focus();
    }

    function render() {
        renderUpdate();
        renderChoices();
        renderChecklist();
    }

    const step = (number, title, ...body) => el('section', { class: 'step-container setup-step', 'aria-label': title },
        el('div', { class: 'step-header' }, el('span', { class: 'step-number', 'aria-hidden': 'true' }, number), el('h3', null, title)),
        body);

    replace(panel, el('div', { class: 'setup-page' },
        el('header', { class: 'setup-hero' },
            el('img', { src: '/ui/img/logo.png', alt: '', class: 'setup-logo', width: '96', height: '50' }),
            el('div', null, el('h2', { class: 'setup-title' }, 'Set up VoucherVisionLocal'), updateLine)),
        step('1', 'How will you run models?', choicesHost,
            el('p', { class: 'help-text setup-change-later' }, 'You can change this at any time.')),
        step('2', 'Get ready', checklistHost)));

    // ------------------------------------------------------------ events
    let lastJobsKey = '';
    on('jobs-changed', (jobs) => {
        const key = (jobs || []).filter((j) => [ENGINE_KIND, DOWNLOAD_KIND, DETECTOR_KIND].includes(j.kind)).map((j) => `${j.id}:${j.state}`).join('|');
        if (key !== lastJobsKey) {
            lastJobsKey = key;
            render();
            return;
        }
        const e = activeJob(ENGINE_KIND);
        const d = activeJob(DOWNLOAD_KIND);
        const det = activeJob(DETECTOR_KIND);
        if (e) engineProgress.update(e);
        if (d) modelProgress.update(d);
        if (det) detectorProgress.update(det);
    });
    on('job-finished', async (job) => {
        if (job && job.kind === DETECTOR_KIND) {
            await loaders.detector();
            render();
            return;
        }
        if (!wantsLocal()) return;
        if (job.kind === ENGINE_KIND) await loaders.engine();
        await loaders.models();
        render();
    });
    on('providers-changed', async () => {
        await loaders.providers();
        render();
    });
    on('models-changed', async () => {
        await loaders.settings();
        if (wantsLocal()) await loaders.models();
        render();
    });
    on('engine-changed', async () => {
        if (!wantsLocal()) return;
        await loaders.engine();
        render();
    });
    onUpdate((state) => {
        update = state;
        if (data.usage !== undefined) render();
        else renderUpdate();
    });

    panel.dataset.state = 'loading';
    render();
    await loadAll();
    render();
    panel.dataset.state = 'ready';

    let shownAt = Date.now();
    return {
        async show() {
            // Coming back from the Models or Keys tab: pick up what changed there.
            if (Date.now() - shownAt < 1500) return;
            shownAt = Date.now();
            await loadAll();
            render();
        },
    };
}
