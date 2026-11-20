// Prompts tab: built-in and user prompt templates, details, import, delete.

import { api, ApiError, describeError, inlineError, toast, toastError } from '../api.js';
import { el, icon, replace, clear, stateBox, badge, defList, confirmDialog } from '../dom.js';
import { emit, gotoTab } from '../state.js';

const NAME_RE = /^[A-Za-z0-9._-]+\.ya?ml$/;

export async function mount(panel) {
    let prompts = [];
    let current = null; // {source, name}
    let detailSeq = 0;

    const list = el('div', { class: 'prompt-list' });
    const detail = el('div', { class: 'prompt-detail' });
    const importErrors = el('div', { class: 'import-errors' });
    const importInput = el('input', { type: 'file', accept: '.yaml,.yml', class: 'visually-hidden', id: 'promptImportInput', tabindex: '-1' });
    const importButton = el('button', { type: 'button', class: 'button-secondary', onclick: () => importInput.click() }, icon('plus', { size: 14 }), 'Import a prompt (.yaml)');

    function key(p) {
        return `${p.source}/${p.name}`;
    }

    function renderList() {
        clear(list);
        if (!prompts.length) {
            list.appendChild(stateBox('empty', 'No prompts yet', 'Import a YAML prompt to get started.', { compact: true }));
            return;
        }
        for (const [source, title] of [['user', 'Your prompts'], ['builtin', 'Built-in prompts']]) {
            const group = prompts.filter((p) => p.source === source);
            if (!group.length) continue;
            list.appendChild(el('div', { class: 'prompt-list-group' }, title));
            for (const p of group) {
                const selected = current && key(current) === key(p);
                list.appendChild(el('button', {
                    type: 'button', class: ['prompt-item', selected ? 'selected' : ''], 'aria-current': selected ? 'true' : null,
                    onclick: () => select(p.source, p.name),
                },
                el('span', { class: 'prompt-item-name' }, p.prompt_name || p.name, p.valid === false ? badge('Invalid', 'bad') : null),
                el('span', { class: 'prompt-item-meta' },
                    [p.name, p.prompt_version ? `version ${p.prompt_version}` : null, Number.isFinite(p.n_fields) ? `${p.n_fields} fields` : null].filter(Boolean).join(' · ')),
                p.prompt_author || p.prompt_author_institution
                    ? el('span', { class: 'prompt-item-meta' }, [p.prompt_author, p.prompt_author_institution].filter(Boolean).join(', ')) : null,
                p.prompt_description ? el('span', { class: 'prompt-item-meta' }, p.prompt_description.length > 110 ? `${p.prompt_description.slice(0, 110)}…` : p.prompt_description) : null));
            }
        }
    }

    function rulesTable(d) {
        const fields = d.fields || [];
        if (!fields.length) return el('p', { class: 'help-text' }, 'This prompt lists no fields.');
        const groups = [];
        for (const f of fields) {
            const name = f.group || 'Other';
            let g = groups.find((x) => x.name === name);
            if (!g) groups.push(g = { name, fields: [] });
            g.fields.push(f);
        }
        const showGroups = groups.length > 1 || groups[0].name !== 'Other';
        const tbody = el('tbody');
        for (const g of groups) {
            if (showGroups) tbody.appendChild(el('tr', { class: 'group-row' }, el('th', { colspan: '2', scope: 'colgroup' }, g.name)));
            for (const f of g.fields) tbody.appendChild(el('tr', null, el('td', null, f.name), el('td', null, f.rule || '—')));
        }
        return el('div', { class: 'table-wrap' }, el('table', { class: 'data rules-table' },
            el('thead', null, el('tr', null, el('th', { scope: 'col' }, 'Field'), el('th', { scope: 'col' }, 'Rule'))), tbody));
    }

    function renderDetail(d) {
        const user = d.source === 'user';
        replace(detail,
            el('div', { class: 'card' },
                el('div', { class: 'prompt-detail-header' },
                    el('div', null, el('h2', null, d.prompt_name || d.name), el('div', { class: 'prompt-file' }, d.name)),
                    badge(user ? 'Yours' : 'Built-in', user ? 'info' : 'good'),
                    d.valid === false ? badge('Invalid', 'bad') : null,
                    el('span', { class: 'spacer' }),
                    el('button', { type: 'button', class: 'button', disabled: d.valid === false, onclick: () => usePrompt(d) }, 'Use this prompt'),
                    user ? el('button', { type: 'button', class: 'button-secondary button-danger', onclick: () => removePrompt(d) }, icon('trash', { size: 14 }), 'Delete') : null),
                d.valid === false ? el('div', { class: 'notice notice-warn', role: 'alert' }, icon('alert', { size: 18 }), el('span', null, d.error || 'This prompt is not valid.')) : null,
                defList([
                    ['Author', d.prompt_author || null],
                    ['Institution', d.prompt_author_institution || null],
                    ['Version', d.prompt_version || null],
                    ['Written for', d.llm || null],
                    ['Fields', String((d.fields || []).length)],
                ]),
                d.prompt_description ? el('p', { class: 'prompt-text family-summary' }, d.prompt_description) : null),
            d.instructions ? el('div', { class: 'card' }, el('div', { class: 'card-header' }, el('h3', null, 'Instructions')), el('p', { class: 'prompt-text' }, d.instructions)) : null,
            d.json_formatting_instructions ? el('div', { class: 'card' }, el('div', { class: 'card-header' }, el('h3', null, 'Formatting instructions')), el('p', { class: 'prompt-text' }, d.json_formatting_instructions)) : null,
            el('div', { class: 'card' }, el('div', { class: 'card-header' }, el('h3', null, 'Fields and rules')), rulesTable(d)),
            el('div', { class: 'card' }, el('details', { class: 'raw' }, el('summary', null, 'Raw YAML'), el('pre', { class: 'json-content' }, d.yaml_text || ''))));
    }

    async function select(source, name) {
        current = { source, name };
        renderList();
        const seq = detailSeq += 1;
        replace(detail, stateBox('loading', `Loading ${name}…`));
        try {
            const d = await api.get(`/v1/prompts/${encodeURIComponent(source)}/${encodeURIComponent(name)}`);
            if (seq === detailSeq) renderDetail(d);
        } catch (err) {
            if (seq === detailSeq) inlineError(detail, err, { title: `${name} could not be loaded`, retry: () => select(source, name) });
        }
    }

    function usePrompt(d) {
        emit('use-prompt', { source: d.source, name: d.name });
        toast(`${d.name} is selected for processing.`, 'success');
        gotoTab('vvl');
    }

    async function removePrompt(d) {
        const yes = await confirmDialog({ title: `Delete ${d.name}?`, message: 'This removes the prompt file from your prompts folder.', confirmLabel: 'Delete', danger: true });
        if (!yes) return;
        try {
            await api.del(`/v1/prompts/user/${encodeURIComponent(d.name)}`);
            toast(`${d.name} was deleted.`, 'success');
            current = null;
            emit('prompts-changed');
            await load();
        } catch (err) {
            toastError(err, `Could not delete ${d.name}`);
        }
    }

    importInput.addEventListener('change', async () => {
        const file = importInput.files[0];
        importInput.value = '';
        clear(importErrors);
        if (!file) return;
        if (!NAME_RE.test(file.name)) {
            replace(importErrors, stateBox('error', 'This file name cannot be used',
                'Use only letters, digits, ".", "_" and "-", ending in .yaml or .yml.', { compact: true }));
            return;
        }
        importButton.disabled = true;
        try {
            const form = new FormData();
            form.append('file', file, file.name);
            const summary = await api.postForm('/v1/prompts/import', form);
            toast(`${file.name} was imported.`, 'success');
            emit('prompts-changed');
            current = { source: 'user', name: (summary && summary.name) || file.name };
            await load();
        } catch (err) {
            const lines = err instanceof ApiError ? err.detailLines() : [];
            replace(importErrors, stateBox('error', `${file.name} was not imported`,
                el('div', null, describeError(err), lines.length ? el('ul', { class: 'error-lines' }, lines.map((l) => el('li', null, l))) : null), { compact: true }));
        } finally {
            importButton.disabled = false;
        }
    });

    async function load() {
        panel.dataset.state = 'loading';
        replace(list, stateBox('loading', 'Loading prompts…', null, { compact: true }));
        try {
            const data = await api.get('/v1/prompts');
            prompts = (data && data.prompts) || [];
            if (current && !prompts.some((p) => key(p) === key(current))) current = null;
            if (!current && prompts.length) current = { source: prompts[0].source, name: prompts[0].name };
            renderList();
            if (current) await select(current.source, current.name);
            else replace(detail, stateBox('empty', 'No prompt selected'));
            panel.dataset.state = 'ready';
        } catch (err) {
            prompts = [];
            inlineError(list, err, { title: 'Prompts could not be loaded', retry: load, compact: true });
            replace(detail);
            panel.dataset.state = 'error';
        }
    }

    replace(panel, el('div', { class: 'prompts-layout' },
        el('div', { class: 'prompt-list-panel' }, importButton, importInput, importErrors, list),
        detail));
    await load();

    return {
        show(target) {
            if (!target) return;
            const slash = target.indexOf('/');
            if (slash < 1) return;
            const source = target.slice(0, slash);
            const name = target.slice(slash + 1);
            if (!current || current.source !== source || current.name !== name) select(source, name);
        },
    };
}
