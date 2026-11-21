// Where runs are saved (docs/API.md, "Saved results"): the "Results folder" card
// on the Setup page, and small helpers the VVL tab uses for each run's folder.

import { api, toast, toastError } from './api.js';
import { el, icon, infoTip, replace } from './dom.js';
import { emit, updateCachedSettings } from './state.js';

let cardCount = 0;

/** GET /v1/output: {dir, default_dir, is_default, exists, synced_folder, subfolders}. */
export function loadOutput() {
    return api.get('/v1/output');
}

/** Open a folder in Finder / Explorer / the file manager. False when that is not possible here. */
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

/** A button that opens `path`, or null where folders cannot be opened (a plain browser). */
export function openFolderButton(path, label = 'Open folder') {
    if (!canOpenFolders() || !path) return null;
    return el('button', {
        type: 'button', class: 'button-secondary button-small',
        onclick: async () => {
            if (!(await openFolder(path))) toast('The folder could not be opened.', 'error');
        },
    }, icon('folder', { size: 13 }), label);
}

/** The name a run's folder gets, as the server builds it (vvl/outputs.py). */
export function folderPreview(name, append) {
    const clean = (name || '').trim();
    if (append) return clean || '…';
    return clean ? `${clean}_YYYYMMDD-HH-MM-SS` : 'YYYYMMDD-HH-MM-SS';
}

/**
 * Put the Results folder card into `host`: the folder every run is saved in.
 * `onSaved` runs after a change is saved.
 */
export function mountOutputFolder(host, { onSaved = null } = {}) {
    cardCount += 1;
    const inputId = `outputDir${cardCount}`;
    const input = el('input', { type: 'text', id: inputId, spellcheck: 'false', autocomplete: 'off' });
    const note = el('p', { class: 'help-text' });
    const save = el('button', { type: 'button', class: 'button', onclick: () => saveFolder(input.value.trim()) }, 'Save');
    const useDefault = el('button', { type: 'button', class: 'button-secondary', onclick: () => saveFolder(null) }, 'Use default');
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
        }, icon('folder', { size: 14 }), 'Browse')
        : null;

    function show(output) {
        if (document.activeElement !== input) input.value = (output && output.dir) || '';
        useDefault.hidden = Boolean(output && output.is_default);
        note.textContent = output && output.is_default ? 'This is the default folder.' : `Default: ${(output && output.default_dir) || ''}`;
    }

    async function saveFolder(value) {
        if (value !== null && !value) {
            toast('Enter the full path of a folder.', 'warning');
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
            toastError(err, 'The results folder was not saved');
        } finally {
            save.disabled = false;
            useDefault.disabled = false;
        }
    }

    async function refresh() {
        try {
            show(await loadOutput());
        } catch (_) {
            note.textContent = 'The results folder could not be read.';
        }
    }

    replace(host, el('div', { class: 'card mf-card' },
        el('div', { class: 'mf-header-row' },
            el('label', { for: inputId }, 'Folder'),
            infoTip('Every run makes its own folder in here, with json, xlsx, label_collage, input_image and ocr_txt inside.', 'start')),
        el('div', { class: 'row row-grow' }, input, browse, save, useDefault),
        note));
    refresh();
    return { refresh };
}
