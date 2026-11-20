// Engine tab: what this computer can run, where models are stored, and the
// engine that runs local models (status, builds, settings, log).

import { api, inlineError, plainError, toast, toastError } from '../api.js';
import {
    el, icon, replace, stateBox, badge, fmtBytes, fmtBytesExact, fmtMemory, fmtDateTime, defList, actionDialog, copyText, prefs,
} from '../dom.js';
import { emit, on, gotoTab, loadSettings, updateCachedSettings } from '../state.js';
import { activeJob, isActive, jobError, jobFor, progressView, refreshJobs, trackJob } from '../jobs.js';
import { mountModelsFolder } from '../models_folder.js';

const TIER_LABEL = { cpu_only: 'CPU only', entry: 'Entry', mid: 'Mid-range', high: 'High-end', workstation: 'Workstation' };
const BACKEND_LABEL = { cuda: 'NVIDIA CUDA', metal: 'Apple Metal', vulkan: 'Vulkan', cpu: 'CPU', rocm: 'AMD ROCm', sycl: 'Intel SYCL' };
const STATE_LABEL = {
    not_installed: ['Not installed', 'neutral'], stopped: ['Stopped', 'neutral'], starting: ['Starting', 'info'],
    loading_model: ['Loading model', 'info'], ready: ['Ready', 'good'], stopping: ['Stopping', 'info'], error: ['Error', 'bad'],
};
const STATE_TEXT = {
    not_installed: 'The engine that runs local models is not installed.',
    stopped: 'The engine is installed and not running. It starts by itself when a local model is used.',
    starting: 'The engine is starting.',
    loading_model: 'The model is being loaded into memory. Large models take a minute or more.',
    ready: 'The engine is running and ready for requests.',
    stopping: 'The engine is stopping.',
    error: 'The engine stopped because of a problem.',
};
const TRANSITIONAL = new Set(['starting', 'loading_model', 'stopping']);
const LOG_STATES = new Set(['starting', 'loading_model', 'ready', 'stopping', 'error']);
const INSTALL_KIND = 'engine_install';
const POLL_FAST_MS = 1000;
const POLL_SLOW_MS = 5000;
const LOG_POLL_MS = 1500;
const LOG_KEEP = 2000;
const KIND_LABEL = { discrete: 'Dedicated graphics card', integrated: 'Integrated graphics', unified: 'Apple Silicon (shared memory)' };

function known(v) {
    return v !== null && v !== undefined && v !== '';
}

function meter(used, total, label) {
    if (!Number.isFinite(used) || !Number.isFinite(total) || total <= 0) return null;
    const pct = Math.max(0, Math.min(100, Math.round((used / total) * 100)));
    const fill = el('div', { class: 'meter-fill' });
    fill.style.width = `${pct}%`;
    return el('div', { class: 'meter', role: 'meter', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(pct), 'aria-label': label }, fill);
}

function sysCard(iconName, title, main, ...subs) {
    return el('div', { class: 'card' },
        el('div', { class: 'sys-card-title' }, icon(iconName, { size: 18 }), title),
        el('div', { class: 'sys-main' }, known(main) ? main : 'Unknown'),
        subs.filter(Boolean).map((s) => (s instanceof Node ? s : el('div', { class: 'sys-sub' }, s))));
}

function section(title, action = null) {
    const body = el('div');
    const node = el('section', { 'aria-label': title },
        el('div', { class: 'section-head' }, el('h2', { class: 'section-title' }, title), action),
        body);
    return { node, body };
}

export async function mount(panel) {
    let systemReport = null;
    let engineHooks = null; // set once the engine panel exists
    const engineReady = () => engineHooks;

    // ------------------------------------------------------------ system report
    const recheck = el('button', { type: 'button', class: 'button-secondary', onclick: () => loadSystem(true) }, icon('refresh', { size: 14 }), 'Re-check');
    const sys = section('This computer', recheck);

    function renderSystem(r) {
        const os = r.os || {};
        const cpu = r.cpu || {};
        const mem = r.memory || {};
        const disk = r.disk || {};
        const cap = r.capability || {};
        const gpus = Array.isArray(r.gpus) ? r.gpus : [];
        const acc = r.accelerators || {};

        const cards = [
            sysCard('computer', 'Operating system', os.name || os.system,
                [os.arch, known(os.kernel) ? `kernel ${os.kernel}` : null].filter(Boolean).join(' · '),
                os.is_wsl ? 'Running inside Windows Subsystem for Linux' : null),
            sysCard('cpu', 'Processor', cpu.model,
                known(cpu.physical_cores) || known(cpu.logical_cores)
                    ? `${known(cpu.physical_cores) ? `${cpu.physical_cores} cores` : ''}${known(cpu.physical_cores) && known(cpu.logical_cores) ? ', ' : ''}${known(cpu.logical_cores) ? `${cpu.logical_cores} threads` : ''}` : null),
            sysCard('memory', 'Memory (RAM)', known(mem.total_bytes) ? fmtMemory(mem.total_bytes) : null,
                meter(mem.total_bytes - mem.available_bytes, mem.total_bytes, 'Memory in use'),
                known(mem.available_bytes) ? `${fmtMemory(mem.available_bytes)} available` : null),
            ...gpus.map((g, i) => sysCard('gpu', gpus.length > 1 ? `Graphics card ${known(g.index) ? g.index : i}` : 'Graphics card', g.name,
                known(g.vram_total_bytes) ? `${fmtMemory(g.vram_total_bytes)} memory` : 'Memory size unknown',
                meter(g.vram_total_bytes - g.vram_free_bytes, g.vram_total_bytes, 'Graphics memory in use'),
                known(g.vram_free_bytes) ? `${fmtMemory(g.vram_free_bytes)} free` : null,
                [KIND_LABEL[g.kind] || null, known(g.driver_version) ? `driver ${g.driver_version}` : null].filter(Boolean).join(' · '))),
            gpus.length ? null : sysCard('gpu', 'Graphics card', 'None found', 'Models will run on the processor, which is slower.'),
            sysCard('disk', 'Disk space for models', known(disk.free_bytes) ? `${fmtBytes(disk.free_bytes)} free` : null,
                meter(disk.total_bytes - disk.free_bytes, disk.total_bytes, 'Disk space in use'),
                known(disk.total_bytes) ? `of ${fmtBytes(disk.total_bytes)} at ${disk.checked_path || disk.models_dir || ''}` : null,
                known(disk.models_dir) ? `Models folder: ${disk.models_dir}${disk.exists === false ? ' (not created yet)' : ''}` : null),
        ].filter(Boolean);

        const accRows = Object.entries(acc).filter(([, v]) => v && typeof v === 'object').map(([name, v]) => el('tr', null,
            el('td', null, BACKEND_LABEL[name] || name),
            el('td', null, v.available === true ? badge('Available', 'good') : (v.available === false ? badge('Not available', 'neutral') : badge('Unknown', 'warn'))),
            el('td', null, v.detail || '')));
        const warnings = Array.isArray(r.warnings) ? r.warnings : [];
        const failed = (Array.isArray(r.probes) ? r.probes : []).filter((p) => p && p.ok === false);

        replace(sys.body,
            el('div', { class: 'sys-summary' }, icon('check', { size: 26 }),
                el('div', { class: 'sys-summary-text' },
                    el('div', { class: 'sys-summary-title' }, cap.summary || 'System checked'),
                    el('div', { class: 'sys-summary-sub' },
                        known(cap.recommended_backend) ? `Recommended engine type: ${BACKEND_LABEL[cap.recommended_backend] || cap.recommended_backend}. ` : '',
                        cap.backend_reason || '')),
                known(cap.tier) ? badge(`${TIER_LABEL[cap.tier] || cap.tier} tier`, 'good') : null),
            warnings.length ? el('ul', { class: 'warning-list' }, warnings.map((w) => el('li', { class: 'notice notice-warn' }, icon('alert', { size: 18 }), el('span', null, (w && w.message) || String(w))))) : null,
            el('div', { class: 'sys-grid' }, cards),
            accRows.length ? el('div', { class: 'card' }, el('div', { class: 'sys-card-title' }, 'Acceleration'),
                el('div', { class: 'table-wrap' }, el('table', { class: 'data' }, el('tbody', null, accRows)))) : null,
            failed.length ? el('p', { class: 'help-text' }, `Some checks could not run: ${failed.map((p) => p.name).join(', ')}.`) : null,
            el('p', { class: 'help-text' }, `Checked ${fmtDateTime(r.generated_at)}`),
            el('details', { class: 'raw' }, el('summary', null, 'Technical details (JSON)'), el('pre', { class: 'json-content' }, JSON.stringify(r, null, 2))));
    }

    async function loadSystem(refresh = false) {
        recheck.disabled = true;
        replace(sys.body, stateBox('loading', refresh ? 'Checking this computer again…' : 'Checking this computer…'));
        try {
            const report = await api.get('/v1/system', { query: refresh ? { refresh: '1' } : undefined, timeoutMs: 120000 });
            if (!report || typeof report !== 'object') throw new Error('The system report was empty.');
            systemReport = report;
            renderSystem(report);
            if (engineReady()) engineReady().refreshFromSystem();
        } catch (err) {
            inlineError(sys.body, err, { title: 'The system check failed', retry: () => loadSystem(refresh) });
        } finally {
            recheck.disabled = false;
        }
    }

    // ------------------------------------------------------------ models folder
    // The same card (path, Browse, Save, and the space-used bar) is also shown above the
    // local models on the Models tab: ui/js/models_folder.js.
    const folder = section('Models folder');
    function loadFolder() {
        mountModelsFolder(folder.body, { onSaved: () => { loadSystem(true); loadEngine(); } });
    }

    // ------------------------------------------------------------ engine
    const engine = section('Local engine');
    engine.node.id = 'engineSection';
    let data = null;            // last GET /v1/engine
    let localBuilds = [];       // builds from GET /v1/models (labels, what is installed)
    let pollTimer = null;
    let busyAction = false;     // a start / stop / uninstall request is in flight

    const statusCard = el('div', { class: 'card' });
    const buildsCard = el('div', { class: 'card' });
    const settingsCard = el('div', { class: 'card' });
    const logCard = el('div', { class: 'card' });
    let engineMounted = false;
    let statusKey = null;
    let buildsKey = null;
    const installViews = new Map(); // variant id -> progress view

    const state = () => (data && data.managed && data.managed.state) || 'not_installed';
    const tabVisible = () => !panel.hidden && !document.hidden;
    const modelLabel = (modelId) => {
        const b = localBuilds.find((x) => x.model_id === modelId);
        return b ? b.label : modelId;
    };
    const variantLabel = (id) => {
        const v = ((data && data.variants) || []).find((x) => x.id === id);
        return v ? v.label : id;
    };

    // ---- status and start / stop
    const modelSelect = el('select', { id: 'engineModel' });
    let modelSelectKey = null;

    function syncModelSelect() {
        const installed = localBuilds.filter((b) => b.install_state === 'installed');
        const key = installed.map((b) => b.model_id).join('|');
        if (key === modelSelectKey) return installed;
        modelSelectKey = key;
        const keep = modelSelect.value;
        replace(modelSelect, installed.map((b) => el('option', { value: b.model_id }, b.label)));
        const loaded = data && data.managed && data.managed.loaded_model;
        const wanted = [keep, loaded, prefs.get('engineModel', null)].find((id) => id && installed.some((b) => b.model_id === id));
        if (wanted) modelSelect.value = wanted;
        return installed;
    }
    modelSelect.addEventListener('change', () => {
        prefs.set('engineModel', modelSelect.value);
        statusKey = null;
        renderStatus();
    });

    async function engineAction(path, body, failTitle) {
        if (busyAction) return;
        busyAction = true;
        statusKey = null;
        renderStatus();
        try {
            const next = await api.post(path, body, { timeoutMs: 120000 });
            if (next && typeof next === 'object') data = next;
            emit('engine-changed');
        } catch (err) {
            toast(`${failTitle}: ${plainError(err)}`, 'error');
        } finally {
            busyAction = false;
            statusKey = null;
            renderAll();
            schedulePoll(300);
        }
    }

    function gpuNames(indexes) {
        if (!Array.isArray(indexes) || !indexes.length) return null;
        const gpus = (systemReport && Array.isArray(systemReport.gpus)) ? systemReport.gpus : [];
        return indexes.map((i) => {
            const g = gpus.find((x) => x.index === i);
            return g && g.name ? `Card ${i}: ${g.name}` : `Card ${i}`;
        }).join(', ');
    }

    function renderStatus() {
        if (!data) return;
        const m = data.managed || {};
        const st = state();
        const installed = syncModelSelect();
        const key = JSON.stringify([m, modelSelectKey, modelSelect.value, busyAction, data.available, data.unavailable_reason,
            (data.settings || {}).idle_stop_minutes, Boolean(systemReport), data.plan]);
        if (key === statusKey) return;
        statusKey = key;

        const [label, kind] = STATE_LABEL[st] || [st, 'neutral'];
        const transitional = TRANSITIONAL.has(st);
        const running = Boolean(m.running) || st === 'ready' || st === 'starting' || st === 'loading_model';
        const rows = [];
        if (m.installed) {
            rows.push(['Engine', `llama.cpp ${m.version || ''}${m.variant && m.variant.label ? ` · ${m.variant.label}` : ''}`.trim()]);
            rows.push(['Runs on', BACKEND_LABEL[m.backend] || m.backend || null]);
        }
        const engines = Array.isArray(m.engines) ? m.engines : [];
        if (running && engines.length > 1) {
            // Several engines: one line each (which model, on which card, how many images at a time).
            for (const e of engines) {
                const where = m.backend === 'cpu' ? 'processor' : gpuNames(e.gpu_devices);
                const parts = [e.model_id ? modelLabel(e.model_id) : null, where,
                    known(e.parallel) ? `${e.parallel} image${e.parallel === 1 ? '' : 's'} at a time` : null,
                    e.state && e.state !== 'ready' ? 'loading' : null].filter(Boolean);
                rows.push([`Engine ${e.number}`, parts.join(' · ')]);
            }
            rows.push(['Context size', known(m.context_size) ? `${Number(m.context_size).toLocaleString('en-US')} tokens per request` : null]);
            rows.push(['In progress', m.active_requests ? `${m.active_requests} request${m.active_requests === 1 ? '' : 's'}` : null]);
        } else if (running) {
            rows.push(['Model', m.loaded_model ? modelLabel(m.loaded_model) : null]);
            rows.push(['Graphics card', m.backend === 'cpu' ? 'None (processor only)' : gpuNames(m.gpu_devices)]);
            rows.push(['Context size', known(m.context_size) ? `${Number(m.context_size).toLocaleString('en-US')} tokens per request` : null]);
            rows.push(['Requests at once', known(m.parallel) ? `${m.parallel}${m.active_requests ? ` (${m.active_requests} in progress)` : ''}` : null]);
            rows.push(['Started', m.started_at ? fmtDateTime(m.started_at) : null]);
            const idleMinutes = (data.settings || {}).idle_stop_minutes;
            rows.push(['Stops when idle', m.idle_stop_at ? `at ${fmtDateTime(m.idle_stop_at)} if nothing uses it`
                : (idleMinutes ? `after ${idleMinutes} minute${idleMinutes === 1 ? '' : 's'} without a request` : 'Never')]);
        }

        // While the engine is still starting, some details are not known yet: leave those lines out.
        const shownRows = rows.filter(([, value]) => known(value));

        const choice = modelSelect.value;
        const sameLoaded = st === 'ready' && m.loaded_model === choice;
        const startLabel = st === 'ready' && m.loaded_model && m.loaded_model !== choice ? 'Switch model' : 'Start';
        let hint = null;
        if (!m.installed) hint = 'Install an engine build below to run local models.';
        else if (!installed.length) hint = 'No local model is downloaded yet.';
        const start = el('button', {
            type: 'button', class: 'button', id: 'engineStart',
            disabled: busyAction || !m.installed || !installed.length || sameLoaded || st === 'stopping' || st === 'starting' || st === 'loading_model',
            onclick: () => engineAction('/v1/engine/start', { model_id: modelSelect.value }, 'The engine did not start'),
        }, icon('play', { size: 14 }), startLabel);
        const stop = el('button', {
            type: 'button', class: 'button-secondary', id: 'engineStop',
            disabled: busyAction || !running || st === 'stopping',
            onclick: () => engineAction('/v1/engine/stop', {}, 'The engine did not stop'),
        }, icon('stop', { size: 14 }), 'Stop');

        const err = m.last_error;
        const tail = err && Array.isArray(err.log_tail) ? err.log_tail.filter((l) => typeof l === 'string') : [];
        replace(statusCard,
            el('div', { class: 'engine-status', role: 'status' },
                el('span', { class: 'engine-state' }, transitional ? el('span', { class: 'spinner spinner-small', 'aria-hidden': 'true' }) : null, badge(label, kind)),
                el('span', { class: 'engine-state-text' }, STATE_TEXT[st] || '')),
            data.available === false && data.unavailable_reason ? el('div', { class: 'notice notice-warn' }, icon('alert', { size: 18 }), el('span', null, data.unavailable_reason)) : null,
            shownRows.length ? defList(shownRows, 'kv engine-kv') : null,
            el('div', { class: 'engine-actions' },
                el('label', { for: 'engineModel', class: 'engine-model-label' }, 'Model'),
                installed.length ? modelSelect : el('span', { class: 'help-text' }, 'None downloaded'),
                start, stop,
                !installed.length && m.installed ? el('button', { type: 'button', class: 'link-button', onclick: () => gotoTab('models') }, 'Download a model') : null),
            hint ? el('p', { class: 'help-text' }, hint) : null,
            m.installed ? el('p', { class: 'help-text' },
                'Engines are arranged automatically: a model runs on every graphics card it fits on, and the OCR and parsing models stay loaded together when both fit.',
                data.plan && data.plan.summary ? ` With the current selection: ${data.plan.summary}` : '') : null,
            err ? el('div', { class: 'job-problem', role: 'alert' }, icon('alert', { size: 18 }),
                el('div', { class: 'job-problem-body' },
                    el('div', { class: 'job-problem-title' }, err.message || 'The engine stopped with an error.'),
                    el('div', { class: 'job-problem-hint' }, [err.at ? fmtDateTime(err.at) : null, known(err.exit_code) ? `exit code ${err.exit_code}` : null].filter(Boolean).join(' · ')),
                    tail.length ? el('details', { class: 'raw' }, el('summary', null, 'Last engine output'), el('pre', { class: 'logs logs-small', tabindex: '0' }, tail.join('\n'))) : null)) : null);
    }

    // ---- builds: install and uninstall
    async function confirmInstall(variant) {
        const sep = (data.models_dir || '').includes('\\') ? '\\' : '/';
        const started = await actionDialog({
            title: 'Install the engine?',
            body: [
                defList([
                    ['Build', variant.label],
                    ['Runs on', BACKEND_LABEL[variant.backend] || variant.backend],
                    ['Download size', fmtBytesExact(variant.download_size_bytes)],
                    ['Saved to', data.models_dir ? `${data.models_dir.replace(/[\\/]+$/, '')}${sep}engine` : 'The models folder'],
                ], 'kv kv-dialog'),
                variant.compatible === false ? el('div', { class: 'notice notice-warn' }, icon('alert', { size: 18 }), el('span', null, `This build is not expected to work here. ${variant.reason || ''}`)) : null,
                el('p', { class: 'help-text' }, 'After the download, the engine is tested on this computer before it is used.'),
            ],
            confirmLabel: `Install ${fmtBytes(variant.download_size_bytes)}`,
            describe: plainError,
            action: async () => {
                const res = await api.post('/v1/engine/install', { variant: variant.id }, { timeoutMs: 60000 });
                return (res && res.job) || true;
            },
        });
        if (started && started.id) trackJob(started);
        if (started) loadEngine({ quiet: true });
    }

    async function confirmUninstall(variant) {
        const next = await actionDialog({
            title: `Uninstall ${variant.label}?`,
            body: ['The engine files are deleted. Downloaded models are kept. You can install the engine again later.'],
            confirmLabel: 'Uninstall',
            danger: true,
            describe: plainError,
            action: () => api.post('/v1/engine/uninstall', { variant: variant.id }, { timeoutMs: 120000 }),
        });
        if (next) {
            if (typeof next === 'object') data = next;
            toast(`${variant.label} uninstalled.`, 'success');
            emit('engine-changed');
            emit('models-changed');
            renderAll();
        }
    }

    function renderBuilds() {
        if (!data) return;
        const variants = Array.isArray(data.variants) ? data.variants : [];
        const rec = data.recommended_variant;
        const job = activeJob(INSTALL_KIND);
        const st = state();
        const engineBusy = st !== 'not_installed' && st !== 'stopped' && st !== 'error';
        const lastJobs = variants.map((v) => jobFor(INSTALL_KIND, v.id));
        const key = JSON.stringify([variants, rec && rec.id, job && `${job.id}:${job.state}`, engineBusy,
            lastJobs.map((j) => (j ? `${j.id}:${j.state}` : ''))]);
        if (key !== buildsKey) {
            buildsKey = key;
            installViews.clear();
            const blockReason = job ? 'An engine install is running.' : (engineBusy ? 'Stop the engine before installing or uninstalling a build.' : null);
            const body = [];
            variants.forEach((v, i) => {
                const mine = lastJobs[i];
                const installing = isActive(mine);
                const isRec = rec && rec.id === v.id;
                const reasonId = `variant-block-${v.id}`;
                let action;
                if (installing) {
                    action = el('span', { class: 'cell-note' }, mine.state === 'verifying' ? 'Verifying…' : 'Installing…');
                } else if (v.installed) {
                    action = el('button', { type: 'button', class: 'button-secondary button-small button-danger', disabled: Boolean(blockReason), 'aria-describedby': blockReason ? reasonId : null, onclick: () => confirmUninstall(v) }, icon('trash', { size: 13 }), 'Uninstall');
                } else {
                    action = el('button', { type: 'button', class: [isRec && !variants.some((x) => x.installed) ? 'button' : 'button-secondary', 'button-small'], disabled: Boolean(blockReason), 'aria-describedby': blockReason ? reasonId : null, onclick: () => confirmInstall(v) }, icon('download', { size: 13 }), 'Install');
                }
                body.push(el('tr', null,
                    el('th', { scope: 'row', class: 'build-name' }, el('strong', null, v.label),
                        isRec ? el('div', { class: 'recommended-mark' }, icon('star', { size: 13 }), 'Recommended for this computer') : null),
                    el('td', { class: 'nowrap' }, BACKEND_LABEL[v.backend] || v.backend || '—'),
                    el('td', { class: 'nowrap' }, fmtBytes(v.download_size_bytes)),
                    el('td', null,
                        v.installed ? badge('Installed', 'good') : (v.compatible ? badge('Compatible', 'neutral') : badge('Not compatible', 'bad')),
                        v.reason ? el('div', { class: 'fit-detail' }, v.reason) : null),
                    el('td', { class: 'build-action' }, el('div', { class: 'row-actions' }, action,
                        blockReason && !installing ? el('span', { class: 'visually-hidden', id: reasonId }, blockReason) : null))));
                let detail = null;
                if (installing) {
                    const view = progressView(`Install progress for ${v.label}`);
                    installViews.set(v.id, view);
                    detail = view.node;
                } else if (mine && mine.state === 'error' && !v.installed && !variants.some((x) => x.installed)) {
                    const problem = jobError(mine);
                    const nextId = problem.details && problem.details.next_variant;
                    const next = nextId ? variants.find((x) => x.id === nextId) : null;
                    const output = problem.details && problem.details.output_tail;
                    const outputText = Array.isArray(output) ? output.join('\n') : (typeof output === 'string' ? output : '');
                    detail = el('div', { class: 'job-problem', role: 'alert' }, icon('alert', { size: 18 }),
                        el('div', { class: 'job-problem-body' },
                            el('div', { class: 'job-problem-title' }, problem.code === 'ENGINE_SELF_TEST_FAILED' ? 'This build cannot run on this computer.' : 'The install did not finish.'),
                            el('div', { class: 'job-problem-hint' }, problem.message),
                            problem.hint ? el('div', { class: 'job-problem-hint' }, problem.hint) : null,
                            next && !next.installed ? el('div', { class: 'job-problem-actions' },
                                el('button', { type: 'button', class: 'button button-small', disabled: Boolean(blockReason), onclick: () => confirmInstall(next) }, icon('download', { size: 13 }), `Install ${next.label} instead`)) : null,
                            outputText.trim() ? el('details', { class: 'raw' }, el('summary', null, 'Test output'), el('pre', { class: 'logs logs-small', tabindex: '0' }, outputText)) : null));
                }
                if (detail) body.push(el('tr', { class: 'build-detail-row' }, el('td', { colspan: '5' }, detail)));
            });
            replace(buildsCard,
                el('div', { class: 'sys-card-title' }, icon('download', { size: 16 }), 'Engine builds for this computer'),
                rec
                    ? el('p', { class: 'help-text' }, `Recommended: ${rec.label} (${fmtBytes(rec.download_size_bytes)} download).${rec.reason && !rec.reason.startsWith(rec.label) ? ` ${rec.reason}` : ''}`)
                    : el('p', { class: 'help-text' }, data.unavailable_reason || 'No build is recommended for this computer.'),
                blockReason && !job ? el('p', { class: 'help-text' }, blockReason) : null,
                variants.length
                    ? el('div', { class: 'table-wrap' }, el('table', { class: 'data build-table' },
                        el('thead', null, el('tr', null, ['Build', 'Runs on', 'Download size', 'On this computer', ''].map((t) => el('th', { scope: 'col' }, t || el('span', { class: 'visually-hidden' }, 'Actions'))))),
                        el('tbody', null, body)))
                    : null);
        }
        for (const [id, view] of installViews) view.update(jobFor(INSTALL_KIND, id));
    }

    // ---- settings
    const form = {
        variant: el('select', { id: 'engVariant' }),
        ctxMode: el('select', { id: 'engCtxMode' }, el('option', { value: 'auto' }, 'Automatic'), el('option', { value: 'custom' }, 'Custom')),
        ctx: el('input', { type: 'number', id: 'engCtx', min: '8192', max: '262144', step: '1024', inputmode: 'numeric', 'aria-label': 'Context size in tokens' }),
        parallel: el('input', { type: 'number', id: 'engParallel', min: '1', max: '16', step: '1', inputmode: 'numeric', placeholder: 'Automatic' }),
        staged: el('input', { type: 'checkbox', id: 'engStaged' }),
        gpuAuto: el('input', { type: 'radio', name: 'engGpuMode', id: 'engGpuAuto', value: 'auto' }),
        gpuPick: el('input', { type: 'radio', name: 'engGpuMode', id: 'engGpuPick', value: 'pick' }),
        gpuList: el('div', { class: 'gpu-choices' }),
        idle: el('input', { type: 'number', id: 'engIdle', min: '0', max: '10080', step: '1', inputmode: 'numeric' }),
        save: el('button', { type: 'button', class: 'button' }, 'Save settings'),
        problem: el('div', { class: 'modal-problem', role: 'alert', hidden: true }),
    };
    let formDirty = false;
    let formKey = null;
    let gpuWanted = 'auto';

    function gpuChecks() {
        return [...form.gpuList.querySelectorAll('input[type="checkbox"]')];
    }

    function renderGpuChoices() {
        const gpus = (systemReport && Array.isArray(systemReport.gpus)) ? systemReport.gpus.filter((g) => Number.isInteger(g.index)) : [];
        const picked = Array.isArray(gpuWanted) ? gpuWanted : [];
        replace(form.gpuList, gpus.map((g) => {
            const id = `engGpu${g.index}`;
            return el('label', { class: 'check-row', for: id },
                el('input', { type: 'checkbox', id, value: String(g.index), checked: picked.includes(g.index), disabled: !form.gpuPick.checked, onchange: markDirty }),
                `Card ${g.index}: ${g.name || 'Graphics card'}${known(g.vram_total_bytes) ? ` (${fmtMemory(g.vram_total_bytes)})` : ''}`);
        }));
        // Cards saved in the settings that this computer no longer reports stay selected, so saving does not drop them silently.
        for (const i of picked.filter((x) => !gpus.some((g) => g.index === x))) {
            const id = `engGpu${i}`;
            form.gpuList.appendChild(el('label', { class: 'check-row', for: id },
                el('input', { type: 'checkbox', id, value: String(i), checked: true, disabled: !form.gpuPick.checked, onchange: markDirty }), `Card ${i} (not found now)`));
        }
        form.gpuPick.disabled = !form.gpuList.childElementCount;
    }

    function markDirty() {
        formDirty = true;
        form.problem.hidden = true;
    }

    function syncFormEnabled() {
        form.ctx.disabled = form.ctxMode.value !== 'custom';
        for (const box of gpuChecks()) box.disabled = !form.gpuPick.checked;
    }

    function fillForm() {
        if (!data || !data.settings) return;
        const s = data.settings;
        const variants = Array.isArray(data.variants) ? data.variants : [];
        const key = JSON.stringify([s, variants.map((v) => [v.id, v.label])]);
        if (formDirty || key === formKey) return;
        formKey = key;
        const options = [el('option', { value: 'auto' }, 'Automatic (the recommended build)'), ...variants.map((v) => el('option', { value: v.id }, v.label))];
        if (s.variant !== 'auto' && !variants.some((v) => v.id === s.variant)) options.push(el('option', { value: s.variant }, s.variant));
        replace(form.variant, options);
        form.variant.value = s.variant || 'auto';
        form.ctxMode.value = s.context_size === 'auto' ? 'auto' : 'custom';
        form.ctx.value = s.context_size === 'auto' ? '' : String(s.context_size);
        form.parallel.value = s.parallel === 'auto' || s.parallel == null ? '' : String(s.parallel);
        form.staged.checked = Boolean(s.staged_batches);
        gpuWanted = Array.isArray(s.gpu_devices) ? s.gpu_devices : 'auto';
        form.gpuAuto.checked = gpuWanted === 'auto';
        form.gpuPick.checked = gpuWanted !== 'auto';
        form.idle.value = String(s.idle_stop_minutes);
        renderGpuChoices();
        syncFormEnabled();
    }

    function readForm() {
        const whole = (input, low, high, name) => {
            const n = Number(input.value);
            if (input.value.trim() === '' || !Number.isInteger(n) || n < low || n > high) {
                input.focus();
                throw new Error(`${name} must be a whole number from ${low.toLocaleString('en-US')} to ${high.toLocaleString('en-US')}.`);
            }
            return n;
        };
        const out = { variant: form.variant.value || 'auto' };
        out.context_size = form.ctxMode.value === 'custom' ? whole(form.ctx, 8192, 262144, 'Context size') : 'auto';
        out.parallel = form.parallel.value.trim() === '' ? 'auto' : whole(form.parallel, 1, 16, 'Requests at once');
        out.staged_batches = form.staged.checked;
        if (form.gpuPick.checked) {
            const picked = gpuChecks().filter((b) => b.checked).map((b) => Number(b.value));
            if (!picked.length) throw new Error('Choose at least one graphics card, or select Automatic.');
            out.gpu_devices = picked;
        } else {
            out.gpu_devices = 'auto';
        }
        out.idle_stop_minutes = whole(form.idle, 0, 10080, 'Minutes before stopping');
        return out;
    }

    async function saveSettings() {
        let values;
        try {
            values = readForm();
        } catch (err) {
            replace(form.problem, icon('alert', { size: 18 }), el('span', null, err.message));
            form.problem.hidden = false;
            return;
        }
        form.problem.hidden = true;
        form.save.disabled = true;
        try {
            const saved = await api.put('/v1/settings', { engine: values });
            const { warnings, ...settings } = saved || {};
            updateCachedSettings(settings);
            formDirty = false;
            formKey = null;
            if (data && settings.engine) data.settings = settings.engine;
            fillForm();
            toast('Engine settings saved. They apply the next time the engine starts.', 'success');
            emit('engine-changed');
        } catch (err) {
            replace(form.problem, icon('alert', { size: 18 }), el('span', null, plainError(err)));
            form.problem.hidden = false;
        } finally {
            form.save.disabled = false;
        }
    }

    for (const node of [form.variant, form.ctx, form.parallel, form.idle]) node.addEventListener('input', markDirty);
    form.staged.addEventListener('change', markDirty);
    form.ctxMode.addEventListener('change', () => {
        markDirty();
        syncFormEnabled();
        if (form.ctxMode.value === 'custom' && !form.ctx.value) form.ctx.value = '16384';
    });
    for (const radio of [form.gpuAuto, form.gpuPick]) {
        radio.addEventListener('change', () => {
            markDirty();
            syncFormEnabled();
        });
    }
    form.save.addEventListener('click', saveSettings);

    function field(labelText, forId, control, help = null) {
        return el('div', { class: 'form-field' }, el('label', { for: forId }, labelText), control, help ? el('p', { class: 'help-text' }, help) : null);
    }

    replace(settingsCard,
        el('div', { class: 'sys-card-title' }, icon('settings', { size: 16 }), 'Engine settings'),
        el('p', { class: 'help-text' }, 'Changes apply the next time the engine starts.'),
        el('div', { class: 'form-grid' },
            field('Engine build', 'engVariant', form.variant, 'Automatic uses the build recommended for this computer.'),
            field('Context size', 'engCtxMode', el('div', { class: 'row' }, form.ctxMode, form.ctx),
                'How much text and image data fits in one request, in tokens. Automatic suits the chosen model.'),
            field('Requests at once, per engine', 'engParallel', form.parallel, 'Leave blank for Automatic: as many as graphics memory allows, up to 4. More uses more graphics memory.'),
            field('Stop when idle after (minutes)', 'engIdle', form.idle, 'Frees graphics memory when nothing has used the engine for this long. 0 keeps it running.'),
            el('fieldset', { class: 'form-field form-fieldset' }, el('legend', null, 'Graphics cards'),
                el('label', { class: 'check-row', for: 'engGpuAuto' }, form.gpuAuto, 'Automatic (the card with the most free memory, or all cards for a large model)'),
                el('label', { class: 'check-row', for: 'engGpuPick' }, form.gpuPick, 'Only these cards:'),
                form.gpuList),
            el('div', { class: 'form-field' },
                el('label', { class: 'check-row', for: 'engStaged' }, form.staged, 'In batches, run all OCR first, then all parsing'),
                el('p', { class: 'help-text' }, 'This always happens when the OCR and parsing models do not fit in graphics memory together.'))),
        form.problem,
        el('div', { class: 'engine-actions' }, form.save));

    // ---- log console
    const logView = el('pre', { class: 'logs', role: 'log', 'aria-label': 'Engine log', 'aria-live': 'off', tabindex: '0' });
    const logEmpty = el('span', { class: 'logs-empty' }, 'No engine output yet.');
    logView.appendChild(logEmpty);
    let logCursor = null;
    let logTimer = null;
    let logFetching = false;
    let logLines = 0;

    function appendLogs(lines) {
        if (!lines.length) return;
        const pinned = logView.scrollTop + logView.clientHeight >= logView.scrollHeight - 12;
        if (logEmpty.parentNode) logEmpty.remove();
        for (const line of lines) {
            logView.appendChild(el('span', { class: 'log-line' }, `${typeof line.text === 'string' ? line.text : ''}\n`));
            logLines += 1;
        }
        while (logLines > LOG_KEEP && logView.firstChild) {
            logView.removeChild(logView.firstChild);
            logLines -= 1;
        }
        if (pinned) logView.scrollTop = logView.scrollHeight;
    }

    async function fetchLogs() {
        if (logFetching) return;
        logFetching = true;
        try {
            const query = logCursor === null ? { limit: '500' } : { after: String(logCursor), limit: '500' };
            const res = await api.get('/v1/engine/logs', { query, timeoutMs: 15000 });
            const lines = (res && Array.isArray(res.lines)) ? res.lines : [];
            if (res && Number.isFinite(res.next)) {
                if (logCursor !== null && res.next < logCursor) logCursor = null; // the server restarted its log
                else logCursor = res.next;
            }
            appendLogs(lines);
        } catch (_) {
            // The status panel already reports a server that cannot be reached.
        } finally {
            logFetching = false;
        }
    }

    function scheduleLogs() {
        if (logTimer) clearTimeout(logTimer);
        logTimer = null;
        if (!tabVisible() || !LOG_STATES.has(state())) return;
        logTimer = setTimeout(async () => {
            logTimer = null;
            await fetchLogs();
            scheduleLogs();
        }, LOG_POLL_MS);
    }

    replace(logCard,
        el('div', { class: 'log-head' },
            el('div', { class: 'sys-card-title' }, icon('terminal', { size: 16 }), 'Engine log'),
            el('span', { class: 'spacer' }),
            el('button', {
                type: 'button', class: 'button-secondary button-small',
                onclick: async () => toast((await copyText(logView.textContent === logEmpty.textContent ? '' : logView.textContent)) ? 'Log copied.' : 'Copy failed.', 'info'),
            }, icon('copy', { size: 13 }), 'Copy'),
            el('button', {
                type: 'button', class: 'button-secondary button-small',
                onclick: () => {
                    replace(logView, logEmpty);
                    logLines = 0;
                },
            }, icon('x', { size: 13 }), 'Clear')),
        logView);

    engineHooks = {
        refreshFromSystem() {
            statusKey = null;
            renderStatus();
            if (!formDirty) renderGpuChoices();
            syncFormEnabled();
        },
    };

    // ---- loading and polling
    function renderAll() {
        if (!data) return;
        if (!engineMounted) {
            engineMounted = true;
            replace(engine.body, statusCard, buildsCard, settingsCard, logCard);
        }
        renderStatus();
        renderBuilds();
        fillForm();
    }

    function schedulePoll(ms = null) {
        if (pollTimer) clearTimeout(pollTimer);
        pollTimer = null;
        if (!tabVisible()) return;
        const st = state();
        let wait = ms;
        if (wait === null) {
            if (TRANSITIONAL.has(st) || activeJob(INSTALL_KIND)) wait = POLL_FAST_MS;
            else if (st === 'ready') wait = POLL_SLOW_MS;
        }
        if (wait === null) return; // idle: nothing to watch
        pollTimer = setTimeout(() => {
            pollTimer = null;
            loadEngine({ quiet: true });
        }, wait);
    }

    let engineLoading = null;
    function loadEngine({ quiet = false } = {}) {
        if (engineLoading) return engineLoading;
        engineLoading = (async () => {
            if (!quiet && !engineMounted) replace(engine.body, stateBox('loading', 'Loading engine status…', null, { compact: true }));
            const before = state();
            try {
                const next = await api.get('/v1/engine', { timeoutMs: 30000 });
                if (!next || typeof next !== 'object') throw new Error('The engine status was empty.');
                data = next;
                renderAll();
                if (state() !== before) {
                    fetchLogs(); // catch the last lines of a start, stop or crash
                    if (!TRANSITIONAL.has(state())) emit('engine-changed');
                }
            } catch (err) {
                if (!engineMounted) {
                    inlineError(engine.body, err, { title: 'The engine status could not be loaded', retry: () => loadEngine() });
                }
            } finally {
                engineLoading = null;
                schedulePoll();
                scheduleLogs();
            }
        })();
        return engineLoading;
    }

    async function loadLocalBuilds() {
        try {
            const models = await api.get('/v1/models', { timeoutMs: 30000 });
            localBuilds = ((models.local && models.local.families) || []).flatMap((f) => f.builds || []);
            statusKey = null;
            renderStatus();
        } catch (_) {
            // The Start control then lists no models; the Models tab shows the real error.
        }
    }

    on('jobs-changed', () => {
        if (engineMounted) renderBuilds();
    });
    on('job-finished', (job) => {
        if (job.kind === INSTALL_KIND) {
            if (job.state === 'done') toast('The engine is installed and passed its test.', 'success');
            emit('models-changed');
            loadEngine({ quiet: true });
        } else {
            loadLocalBuilds();
        }
    });
    on('models-changed', loadLocalBuilds);
    document.addEventListener('visibilitychange', () => {
        if (tabVisible()) loadEngine({ quiet: true });
    });

    replace(panel, sys.node, folder.node, engine.node);
    await Promise.all([loadSystem(), loadFolder(), loadEngine(), loadLocalBuilds(), refreshJobs()]);
    renderAll();
    fetchLogs();
    panel.dataset.state = 'ready';
    return {
        show(target) {
            loadEngine({ quiet: true });
            loadLocalBuilds();
            refreshJobs();
            if (target === 'engine') engine.node.scrollIntoView({ block: 'start' });
        },
    };
}
