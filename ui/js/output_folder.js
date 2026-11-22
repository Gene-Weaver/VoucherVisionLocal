// Where runs are saved (docs/API.md, "Saved results"): the "Results folder" card
// on the Setup page, and helpers shared with the VVL and Viewer tabs.

import { api, toast, toastError } from './api.js';
import { el, icon, infoTip, replace } from './dom.js';
import { emit, updateCachedSettings } from './state.js';

let cardCount = 0;

export const OUTPUT_TYPES = [
    { key: 'json', label: 'JSON', required: true, tip: 'Full result per image (always saved).' },
    { key: 'xlsx', label: 'Excel (.xlsx)', required: false, tip: 'Combined spreadsheet of parsed fields.' },
    { key: 'label_collage', label: 'Collage', required: false, tip: 'Cropped label collage JPEG.' },
    { key: 'input_image', label: 'Input image', required: false, tip: 'Copy of the processed image.' },
    { key: 'ocr_txt', label: 'OCR text', required: false, tip: 'Plain text file of the final OCR.' },
];

/** GET /v1/output: {dir, default_dir, is_default, exists, output_types, folders, ...}. */
export function loadOutput() {
    return api.get('/v1/output');
}

export function formatOutputSummary(outputTypes) {
    const types = outputTypes || {};
    const names = ['JSON'];
    for (const item of OUTPUT_TYPES) {
        if (!item.required && types[item.key] !== false) names.push(item.label.replace(' (.xlsx)', ''));
    }
    return names.join(', ');
}

/** Open a folder in Finder / Explorer / the file manager. False when unavailable. */
export async function openFolder(path) {
    const bridge = window.vvlDesktop;
    if (!bridge || typeof bridge.openFolder !== 'function') return false;
    try {
        return Boolean(await bridge.openFolder(path));
    } catch (_) {
        return false;
    }
}

export function canOpenFolders() {
    const bridge = window.vvlDesktop;
    return Boolean(bridge && typeof bridge.openFolder === 'function');
}

export function canPickFolders() {
    const bridge = window.vvlDesktop;
    return Boolean(bridge && typeof bridge.pickFolder === 'function');
}

export async function pickFolder() {
    const bridge = window.vvlDesktop;
    if (!bridge || typeof bridge.pickFolder !== 'function') return null;
    return bridge.pickFolder();
}

/** A button that opens `path`, or null where folders cannot be opened (a plain browser). */
export function openFolderButton(path, label = 'Open folder') {
    if (!canOpenFolders() || !path) return null;
    return el('button', {
        type: 'button', class: 'button-secondary button-small',
        onclick: async () => {
            if (!(await openFolder(path))) toast('Could not open folder.', 'error');
        },
    }, icon('folder', { size: 13 }), label);
}

/** The name a run's folder gets, as the server builds it (vvl/outputs.py). */
export function folderPreview(name, append) {
    const clean = (name || '').trim();
    if (append) return clean || 'Select or name a folder';
    return clean ? `${clean}_YYYYMMDD-HH-MM-SS` : 'YYYYMMDD-HH-MM-SS';
}

/**
 * Put the Results folder + Saved outputs card into `host` (used on Setup).
 */
export function mountOutputFolder(host, { onSaved = null } = {}) {
    cardCount += 1;
    const prefix = `outputCfg${cardCount}`;
    const inputId = `${prefix}Dir`;
    const input = el('input', { type: 'text', id: inputId, spellcheck: 'false', autocomplete: 'off' });
    const note = el('p', { class: 'help-text' });
    const save = el('button', { type: 'button', class: 'button', onclick: () => saveFolder(input.value.trim()) }, 'Save');
    const useDefault = el('button', { type: 'button', class: 'button-secondary', onclick: () => saveFolder(null) }, 'Use default');
    const browse = canPickFolders()
        ? el('button', {
            type: 'button', class: 'button-secondary',
            onclick: async () => {
                try {
                    const picked = await pickFolder();
                    if (picked) {
                        input.value = picked;
                        await saveFolder(picked);
                    }
                } catch (_) {
                    toast('Could not open folder picker.', 'error');
                }
            },
        }, icon('folder', { size: 14 }), 'Browse')
        : null;

    const checkboxes = {};
    const typeChecks = OUTPUT_TYPES.map((item) => {
        const id = `${prefix}_${item.key}`;
        const box = el('input', {
            type: 'checkbox',
            id,
            checked: true,
            disabled: item.required,
            onchange: () => saveTypes(),
        });
        checkboxes[item.key] = box;
        return el('label', { class: ['output-type-chip', item.required ? 'is-required' : ''], for: id, title: item.tip },
            box,
            el('span', null, item.label),
            item.required ? el('span', { class: 'output-req-tag' }, 'Required') : null);
    });

    function show(output) {
        if (document.activeElement !== input) input.value = (output && output.dir) || '';
        useDefault.hidden = Boolean(output && output.is_default);
        note.textContent = output && output.is_default ? 'Using default folder.' : `Default: ${(output && output.default_dir) || ''}`;
        const types = (output && output.output_types) || {};
        for (const item of OUTPUT_TYPES) {
            if (!item.required && checkboxes[item.key]) {
                checkboxes[item.key].checked = types[item.key] !== false;
            }
        }
    }

    async function saveFolder(value) {
        if (value !== null && !value) {
            toast('Enter a folder path.', 'warning');
            return;
        }
        save.disabled = true;
        useDefault.disabled = true;
        try {
            const saved = await api.put('/v1/settings', { output_dir: value });
            const { warnings: _warnings, ...settings } = saved || {};
            updateCachedSettings(settings);
            show(await loadOutput());
            toast('Results folder saved.', 'success');
            emit('output-changed');
            if (onSaved) onSaved();
        } catch (err) {
            toastError(err, 'Could not save results folder');
        } finally {
            save.disabled = false;
            useDefault.disabled = false;
        }
    }

    async function saveTypes() {
        const output_types = {};
        for (const item of OUTPUT_TYPES) {
            if (!item.required) output_types[item.key] = Boolean(checkboxes[item.key].checked);
        }
        try {
            const saved = await api.put('/v1/settings', { output_types });
            const { warnings: _warnings, ...settings } = saved || {};
            updateCachedSettings(settings);
            emit('output-changed');
            if (onSaved) onSaved();
        } catch (err) {
            toastError(err, 'Could not save output options');
            await refresh();
        }
    }

    async function refresh() {
        try {
            show(await loadOutput());
        } catch (_) {
            note.textContent = 'Could not load results folder settings.';
        }
    }

    replace(host, el('div', { class: 'card mf-card' },
        el('div', { class: 'mf-header-row' },
            el('label', { for: inputId }, 'Results folder'),
            infoTip('Where runs are saved. Each run creates a dated folder here unless you add to an existing one.', 'start')),
        el('div', { class: 'row row-grow' }, input, browse, save, useDefault),
        note,
        el('div', { class: 'output-types-block' },
            el('div', { class: 'mf-header-row' },
                el('span', { class: 'output-types-heading' }, 'Saved outputs'),
                infoTip('JSON is always saved. Turn other folders on or off as needed.', 'start')),
            el('div', { class: 'output-types-row' }, typeChecks))));
    refresh();
    return { refresh };
}
