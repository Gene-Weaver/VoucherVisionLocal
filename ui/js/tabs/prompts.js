// Prompts tab: two half-width sub-tabs following the Models tab pattern.
//   Catalog   browse built-in and custom prompts, inspect details, import, delete
//   Designer  create, edit, validate, export, and compare custom prompts

import { api, ApiError, describeError, inlineError, toast, toastError } from '../api.js';
import { el, icon, replace, clear, stateBox, badge, defList, confirmDialog } from '../dom.js';
import { emit, gotoTab } from '../state.js';
import { mountPromptDesigner } from './prompt_designer.js';

const NAME_RE = /^[A-Za-z0-9_-]+\.ya?ml$/;

const SUBTABS = [
    { id: 'catalog', icon: 'file', title: 'Catalog', subtitle: 'Browse and manage prompts' },
    { id: 'designer', icon: 'edit', title: 'Designer', subtitle: 'Create, edit, and compare prompts' },
];

export async function mount(panel) {
    let prompts = [];
    let userDir = '';
    let current = null; // {source, name}
    let detailSeq = 0;
    let activeSubtab = 'catalog';

    const list = el('div', { class: 'prompt-list' });
    const detail = el('div', { class: 'prompt-detail' });
    const importErrors = el('div', { class: 'import-errors' });
    const importInput = el('input', { type: 'file', accept: '.yaml,.yml', class: 'visually-hidden', id: 'promptImportInput', tabindex: '-1' });
    const importButton = el('button', { type: 'button', class: 'button-secondary', onclick: () => importInput.click() }, icon('plus', { size: 14 }), 'Import YAML');

    const designerHost = el('div');
    let designer = null;

    function key(p) {
        return `${p.source}/${p.name}`;
    }

    function renderList() {
        clear(list);
        if (!prompts.length) {
            list.appendChild(stateBox('empty', 'No prompts yet', null, { compact: true }));
            return;
        }
        for (const [source, title] of [['user', 'Custom'], ['builtin', 'Built-in']]) {
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
                    [Number.isFinite(p.n_fields) ? `${p.n_fields} fields` : null, p.prompt_version ? `v${p.prompt_version}` : null].filter(Boolean).join(' · '))));
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

    function openInDesigner(d, { copy = false } = {}) {
        showSubtab('designer');
        if (designer) designer.openPrompt(d.source, d.name, { copy });
    }

    function renderDetail(d) {
        const user = d.source === 'user';
        const author = [d.prompt_author, d.prompt_author_institution].filter(Boolean).join(', ');
        const metaRows = [
            author ? ['Author', author] : null,
            d.prompt_version ? ['Version', d.prompt_version] : null,
            ['Fields', String((d.fields || []).length)],
        ].filter(Boolean);
        const checklist = Array.isArray(d.checklist) ? d.checklist : [];
        const defaults = d.field_default_values && typeof d.field_default_values === 'object' ? Object.entries(d.field_default_values) : [];
        const skipReview = Array.isArray(d.review_not_required) ? d.review_not_required : [];
        const flags = Array.isArray(d.custom_flags) ? d.custom_flags : [];

        replace(detail,
            el('div', { class: 'card' },
                el('div', { class: 'prompt-detail-header' },
                    el('div', null, el('h2', null, d.prompt_name || d.name), el('div', { class: 'prompt-file' }, d.name)),
                    badge(user ? 'Custom' : 'Built-in', user ? 'info' : 'good'),
                    d.valid === false ? badge('Invalid', 'bad') : null,
                    el('span', { class: 'spacer' }),
                    el('button', { type: 'button', class: 'button', disabled: d.valid === false, onclick: () => usePrompt(d) }, 'Use prompt'),
                    user
                        ? el('button', { type: 'button', class: 'button-secondary', onclick: () => openInDesigner(d, { copy: false }) }, icon('edit', { size: 14 }), 'Edit')
                        : el('button', { type: 'button', class: 'button-secondary', onclick: () => openInDesigner(d, { copy: true }) }, icon('copy', { size: 14 }), 'Copy to Designer'),
                    user ? el('button', { type: 'button', class: 'button-secondary button-danger', onclick: () => removePrompt(d) }, icon('trash', { size: 14 }), 'Delete') : null),
                d.valid === false ? el('div', { class: 'notice notice-warn', role: 'alert' }, icon('alert', { size: 18 }), el('span', null, d.error || 'Invalid prompt.')) : null,
                defList(metaRows),
                d.prompt_description ? el('p', { class: 'prompt-text family-summary' }, d.prompt_description) : null),
            d.instructions ? el('div', { class: 'card' }, el('div', { class: 'card-header' }, el('h3', null, 'Instructions')), el('p', { class: 'prompt-text' }, d.instructions)) : null,
            d.json_formatting_instructions ? el('div', { class: 'card' }, el('div', { class: 'card-header' }, el('h3', null, 'Formatting')), el('p', { class: 'prompt-text' }, d.json_formatting_instructions)) : null,
            el('div', { class: 'card' }, el('div', { class: 'card-header' }, el('h3', null, 'Fields')), rulesTable(d)),
            checklist.length ? el('div', { class: 'card' },
                el('div', { class: 'card-header' }, el('h3', null, 'Checklist')),
                el('p', { class: 'prompt-text' }, checklist.join('\n'))) : null,
            (defaults.length || skipReview.length || flags.length) ? el('div', { class: 'card' },
                el('div', { class: 'card-header' }, el('h3', null, 'Editor options')),
                defList([
                    defaults.length ? ['Default values', defaults.map(([k, v]) => `${k}: ${v.join(', ')}`).join(' · ')] : null,
                    skipReview.length ? ['Skip review', skipReview.join(', ')] : null,
                    flags.length ? ['Custom flags', flags.join(', ')] : null,
                ].filter(Boolean))) : null,
            el('div', { class: 'card' }, el('details', { class: 'raw' }, el('summary', null, 'YAML'), el('pre', { class: 'json-content' }, d.yaml_text || ''))));
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
        toast(`Selected ${d.prompt_name || d.name}.`, 'success');
        gotoTab('vvl');
    }

    async function removePrompt(d) {
        const yes = await confirmDialog({ title: `Delete ${d.name}?`, message: 'This cannot be undone.', confirmLabel: 'Delete', danger: true });
        if (!yes) return;
        try {
            await api.del(`/v1/prompts/user/${encodeURIComponent(d.name)}`);
            toast('Deleted.', 'success');
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
                'Use only letters, numbers, "_" and "-", ending in .yaml or .yml.', { compact: true }));
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
            userDir = (data && data.user_dir) || '';
            if (current && !prompts.some((p) => key(p) === key(current))) current = null;
            if (!current && prompts.length) current = { source: prompts[0].source, name: prompts[0].name };
            renderList();
            if (current) await select(current.source, current.name);
            else replace(detail, stateBox('empty', 'No prompt selected'));
            if (designer && activeSubtab === 'designer') await designer.refresh();
            panel.dataset.state = 'ready';
        } catch (err) {
            prompts = [];
            inlineError(list, err, { title: 'Prompts could not be loaded', retry: load, compact: true });
            replace(detail);
            panel.dataset.state = 'error';
        }
    }

    // Two half-width sub-tabs: Catalog | Designer
    const catalogBody = el('div', { class: 'prompts-layout' },
        el('div', { class: 'prompt-list-panel' }, importButton, importInput, importErrors, list),
        detail);
    const bodies = {
        catalog: el('div', { class: 'privacy-body', role: 'tabpanel', id: 'prompts-panel-catalog' }, catalogBody),
        designer: el('div', { class: 'privacy-body', role: 'tabpanel', id: 'prompts-panel-designer', hidden: true }, designerHost),
    };
    const tabButtons = {};
    const shell = el('section', { class: 'privacy-section prompts-shell is-catalog' });

    function showSubtab(id, { focus = false } = {}) {
        activeSubtab = id;
        shell.className = `privacy-section prompts-shell is-${id}`;
        for (const tab of SUBTABS) {
            const onTab = tab.id === id;
            tabButtons[tab.id].setAttribute('aria-selected', String(onTab));
            tabButtons[tab.id].tabIndex = onTab ? 0 : -1;
            bodies[tab.id].hidden = !onTab;
        }
        if (focus) tabButtons[id].focus();
        if (id === 'designer' && designer) designer.refresh();
    }

    for (const tab of SUBTABS) {
        tabButtons[tab.id] = el('button', {
            type: 'button', role: 'tab', class: `privacy-tab is-${tab.id}`, id: `prompts-tab-${tab.id}`, 'aria-controls': `prompts-panel-${tab.id}`,
            onclick: () => showSubtab(tab.id),
            onkeydown: (e) => {
                if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
                e.preventDefault();
                showSubtab(tab.id === 'catalog' ? 'designer' : 'catalog', { focus: true });
            },
        }, icon(tab.icon, { size: 32 }), el('span', { class: 'privacy-tab-text' }, el('span', { class: 'privacy-tab-title' }, tab.title), el('span', { class: 'privacy-tab-sub' }, tab.subtitle)));
        bodies[tab.id].setAttribute('aria-labelledby', `prompts-tab-${tab.id}`);
    }

    designer = mountPromptDesigner(designerHost, {
        getPrompts: () => prompts,
        getUserDir: () => userDir,
        onSaved: async ({ source, name }) => {
            current = { source, name };
            emit('prompts-changed');
            await load();
        },
        onUse: (d) => usePrompt(d),
    });

    replace(shell,
        el('div', { class: 'privacy-tabs', role: 'tablist', 'aria-label': 'Prompts sections' }, SUBTABS.map((t) => tabButtons[t.id])),
        bodies.catalog,
        bodies.designer);
    replace(panel, shell);
    showSubtab('catalog');
    await load();

    return {
        show(target) {
            if (!target) return;
            if (target === 'designer') {
                showSubtab('designer');
                return;
            }
            const slash = target.indexOf('/');
            if (slash < 1) return;
            const source = target.slice(0, slash);
            const name = target.slice(slash + 1);
            showSubtab('catalog');
            if (!current || current.source !== source || current.name !== name) select(source, name);
        },
    };
}
