// The "Models folder" card: where downloaded models are kept, and a bar of the
// disk space the downloaded models use, one segment per model, colored and
// grouped by model family. Shown on the Engine tab and above the local models
// on the Models tab; the copies stay in step through the 'models-changed' event.

import { api, toast, toastError } from './api.js';
import { el, icon, replace, fmtBytes } from './dom.js';
import { emit, on, loadSettings, updateCachedSettings } from './state.js';

const OTHER_COLOR = '#8a8f98';
let cardCount = 0;
let tipNode = null;

function tip() {
    if (!tipNode) {
        tipNode = el('div', { class: 'mf-tip', role: 'tooltip', hidden: true });
        document.body.appendChild(tipNode);
    }
    return tipNode;
}

function showTip(target, lines) {
    const node = tip();
    replace(node, lines.map((line, i) => el('div', { class: i ? 'mf-tip-sub' : 'mf-tip-title' }, line)));
    node.hidden = false;
    const box = target.getBoundingClientRect();
    const width = node.offsetWidth;
    // Set through the style object: the page's security policy refuses style="" attributes.
    node.style.left = `${Math.max(8, Math.min(window.innerWidth - width - 8, box.left + box.width / 2 - width / 2))}px`;
    node.style.top = `${Math.max(8, box.top - node.offsetHeight - 8)}px`;
}

function hideTip() {
    if (tipNode) tipNode.hidden = true;
}

/** Downloaded (or partly downloaded) builds, grouped by family in registry order. */
function downloaded(families) {
    const groups = [];
    for (const family of families || []) {
        const builds = (family.builds || []).filter((b) => Number(b.installed_bytes) > 0);
        if (!builds.length) continue;
        groups.push({
            id: family.id, label: family.label, developer: family.developer,
            color: /^#[0-9a-fA-F]{6}$/.test(family.brand_color || '') ? family.brand_color : OTHER_COLOR,
            bytes: builds.reduce((sum, b) => sum + Number(b.installed_bytes), 0),
            builds,
        });
    }
    return groups;
}

function usageView(families, disk) {
    const groups = downloaded(families);
    const total = groups.reduce((sum, g) => sum + g.bytes, 0);
    const count = groups.reduce((sum, g) => sum + g.builds.length, 0);
    const diskTotal = Number(disk && disk.total_bytes);
    const diskFree = Number(disk && disk.free_bytes);
    const knownDisk = Number.isFinite(diskTotal) && diskTotal > 0 && Number.isFinite(diskFree);
    const head = el('div', { class: 'mf-head' },
        el('span', { class: 'mf-head-main' }, total
            ? `${count} downloaded model${count === 1 ? '' : 's'} use ${fmtBytes(total)}`
            : 'No models are downloaded yet'),
        knownDisk ? el('span', { class: 'mf-head-sub' },
            `${total ? `${((total / diskTotal) * 100).toFixed(total / diskTotal < 0.1 ? 1 : 0)}% of this ${fmtBytes(diskTotal)} drive · ` : ''}${fmtBytes(diskFree)} free`) : null);
    if (!total) return [head];

    const bar = el('div', { class: 'mf-bar', role: 'img', 'aria-label': `Disk space used by each downloaded model, ${fmtBytes(total)} in all` });
    for (const group of groups) {
        const part = el('div', { class: 'mf-group' });
        part.style.flexGrow = String(group.bytes);
        for (const build of group.builds) {
            const bytes = Number(build.installed_bytes);
            const partial = build.install_state !== 'installed';
            const seg = el('div', { class: ['mf-seg', partial ? 'is-partial' : ''], tabindex: '0',
                'aria-label': `${build.label}, ${fmtBytes(bytes)}` });
            seg.style.flexGrow = String(bytes);
            seg.style.setProperty('--mf-color', group.color);
            const lines = [build.label, [fmtBytes(bytes), build.params ? `${build.params} parameters` : null, build.quant].filter(Boolean).join(' · '),
                [group.label, group.developer].filter(Boolean).join(' · ') + (partial ? ' · download not finished' : '')];
            seg.addEventListener('mouseenter', () => showTip(seg, lines));
            seg.addEventListener('focus', () => showTip(seg, lines));
            seg.addEventListener('mouseleave', hideTip);
            seg.addEventListener('blur', hideTip);
            part.appendChild(seg);
        }
        bar.appendChild(part);
    }
    const legend = el('div', { class: 'mf-legend' }, groups.map((g) => {
        const dot = el('span', { class: 'mf-dot' });
        dot.style.setProperty('--mf-color', g.color);
        return el('span', { class: 'mf-legend-item' }, dot, `${g.label} `, el('span', { class: 'mf-legend-size' }, `${fmtBytes(g.bytes)} · ${g.builds.length}`));
    }));
    let drive = null;
    if (knownDisk) {
        const used = Math.max(0, diskTotal - diskFree);
        const models = el('div', { class: 'mf-drive-models' });
        models.style.width = `${Math.min(100, (total / diskTotal) * 100)}%`;
        const other = el('div', { class: 'mf-drive-other' });
        other.style.width = `${Math.max(0, Math.min(100, ((used - total) / diskTotal) * 100))}%`;
        drive = el('div', { class: 'mf-drive', role: 'img',
            'aria-label': `Whole drive: models ${fmtBytes(total)}, other files ${fmtBytes(Math.max(0, used - total))}, free ${fmtBytes(diskFree)}` },
        el('div', { class: 'mf-drive-bar' }, models, other),
        el('div', { class: 'mf-drive-key' },
            el('span', null, el('span', { class: 'mf-key is-models' }), `Models ${fmtBytes(total)}`),
            el('span', null, el('span', { class: 'mf-key is-other' }), `Other files ${fmtBytes(Math.max(0, used - total))}`),
            el('span', null, el('span', { class: 'mf-key is-free' }), `Free ${fmtBytes(diskFree)}`)));
    }
    return [head, bar, legend, drive];
}

/**
 * Put the Models folder card into `host`.
 * `onSaved` runs after a new folder is saved (the Engine tab reloads its own cards).
 */
export function mountModelsFolder(host, { onSaved = null } = {}) {
    cardCount += 1;
    const inputId = `modelsDir${cardCount}`;
    const input = el('input', { type: 'text', id: inputId, spellcheck: 'false', autocomplete: 'off' });
    const warnings = el('div');
    const usage = el('div', { class: 'mf-usage' });
    const save = el('button', { type: 'button', class: 'button', onclick: saveFolder }, 'Save');
    const bridge = window.vvlDesktop;
    const browse = bridge && typeof bridge.pickFolder === 'function'
        ? el('button', {
            type: 'button', class: 'button-secondary',
            onclick: async () => {
                try {
                    const picked = await bridge.pickFolder();
                    if (picked) input.value = picked;
                } catch (_) {
                    toast('The folder picker could not be opened.', 'error');
                }
            },
        }, icon('folder', { size: 14 }), 'Browse…')
        : null;

    function showWarnings(list) {
        replace(warnings, (list || []).map((w) => el('div', { class: 'notice notice-warn', role: 'alert' }, icon('alert', { size: 18 }), el('span', null, (w && w.message) || String(w)))));
    }

    async function saveFolder() {
        const value = input.value.trim();
        if (!value) {
            toast('Enter the full path of a folder.', 'warning');
            return;
        }
        save.disabled = true;
        try {
            const saved = await api.put('/v1/settings', { models_dir: value });
            const { warnings: list, ...settings } = saved || {};
            updateCachedSettings(settings);
            input.value = settings.models_dir || value;
            showWarnings(list);
            toast('Models folder saved.', 'success');
            emit('models-changed');
            if (onSaved) onSaved();
        } catch (err) {
            toastError(err, 'The models folder was not saved');
        } finally {
            save.disabled = false;
        }
    }

    async function loadPath() {
        try {
            const settings = await loadSettings({ force: true });
            // Do not overwrite a path the user is typing.
            if (document.activeElement !== input) input.value = (settings && settings.models_dir) || '';
        } catch (_) {
            // The card still works; the usage bar reports its own problems.
        }
    }

    let loading = false;
    async function loadUsage() {
        if (loading) return;
        loading = true;
        try {
            const [models, system] = await Promise.all([
                api.get('/v1/models', { timeoutMs: 60000 }),
                api.get('/v1/system', { query: { refresh: '1' }, timeoutMs: 60000 }).catch(() => null),
            ]);
            replace(usage, usageView((models.local && models.local.families) || [], system && system.disk));
        } catch (_) {
            replace(usage, el('p', { class: 'help-text' }, 'The space used by models could not be read.'));
        } finally {
            loading = false;
        }
    }

    replace(host, el('div', { class: 'card mf-card' },
        el('label', { for: inputId }, 'Where downloaded models are kept'),
        el('p', { class: 'help-text' }, 'Model files are large. Choose a drive with plenty of free space, outside any synced folder such as Dropbox or OneDrive.'),
        el('div', { class: 'row row-grow' }, input, browse, save),
        warnings,
        usage));
    const refresh = () => { loadPath(); loadUsage(); };
    on('models-changed', refresh);
    on('job-finished', (job) => { if (job && job.kind === 'model_download') loadUsage(); });
    refresh();
    return { refresh };
}
