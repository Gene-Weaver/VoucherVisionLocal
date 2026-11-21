// Prompt Designer: create, edit, validate, export, and compare custom prompts.
// Built-in prompts are read-only; editing one starts from a uniquely named copy.

import { api, plainError, toast, toastError } from '../api.js';
import {
    el, icon, infoTip, replace, clear, stateBox, badge, openModal, confirmDialog, downloadText, copyText,
} from '../dom.js';

const PROMPT_NAME_RE = /^[A-Za-z0-9_-]+$/;
const PROMPT_NAME_RULE = 'Letters, numbers, "_" and "-" only. No spaces.';
const FIELD_RE = /^[A-Za-z][A-Za-z0-9_]*$/;
const CAT_RE = /^[A-Z][A-Z0-9_]*$/;
const REQUIRED_CATS = ['GEOGRAPHY', 'TAXONOMY', 'COLLECTING'];
const MAX_CATS = 6;
const CHECKLIST_TAGS = ['VoucherVision', 'OCR', 'Patterns', 'Dates', 'Collectors', 'Taxonomy', 'Geography'];
const CANONICAL_FORMATTING = 'This section provides rules for formatting each JSON value organized by the JSON key. Correct minor typos introduced by OCR errors.';

const STEPS = [
    { id: 'info', num: '1', label: 'Info' },
    { id: 'prompt', num: '2', label: 'Prompt & Fields' },
    { id: 'mapping', num: '3', label: 'Mapping' },
    { id: 'checklist', num: '4', label: 'Checklist' },
    { id: 'editor', num: '5', label: 'Editor' },
];

// ---------------------------------------------------------------- helpers

function stem(name) {
    return String(name || '').replace(/\.ya?ml$/i, '');
}

function parseChecklistItem(raw) {
    const cleaned = String(raw || '').trim().replace(/^\d+\.\s*/, '');
    const match = cleaned.match(/^\[([^\]]+)\]\s*-\s*(.*)$/);
    if (match) return { tag: match[1].trim(), text: match[2].trim() };
    return { tag: '', text: cleaned };
}

function formatChecklistItem(item, index) {
    const text = (item.text || '').trim();
    const tag = (item.tag || '').trim();
    if (!text && !tag) return '';
    return tag ? `${index + 1}. [${tag}] - ${text}` : `${index + 1}. ${text}`;
}

function detailToData(d) {
    const rules = (d.fields || []).map((f) => ({ name: f.name || '', rule: f.rule || '', _mappedKey: f.name || '' }));
    const ruleSet = new Set(rules.map((r) => r.name));

    const rawMap = (d.mapping && typeof d.mapping === 'object') ? d.mapping : {};
    const rawMapUpper = {};
    const catOrder = [...REQUIRED_CATS];
    for (const [k, v] of Object.entries(rawMap)) {
        const up = String(k).trim().toUpperCase();
        if (up && CAT_RE.test(up)) {
            if (!rawMapUpper[up]) rawMapUpper[up] = [];
            if (Array.isArray(v)) rawMapUpper[up].push(...v);
            if (!catOrder.includes(up) && catOrder.length < MAX_CATS) catOrder.push(up);
        }
    }

    const assigned = new Set();
    const mapping = catOrder.map((category) => {
        const list = Array.isArray(rawMapUpper[category]) ? rawMapUpper[category] : [];
        const fields = [];
        for (const f of list) {
            if (ruleSet.has(f) && !assigned.has(f)) {
                assigned.add(f);
                fields.push(f);
            }
        }
        return { category, fields };
    });

    // Guarantee every rule belongs to a category (default to COLLECTING).
    const collecting = mapping.find((m) => m.category === 'COLLECTING') || mapping[0];
    for (const r of rules) {
        if (!assigned.has(r.name)) {
            assigned.add(r.name);
            collecting.fields.push(r.name);
        }
    }

    const defaults = {};
    const rawDefaults = (d.field_default_values && typeof d.field_default_values === 'object') ? d.field_default_values : {};
    for (const [k, vals] of Object.entries(rawDefaults)) {
        if (ruleSet.has(k) && Array.isArray(vals)) {
            const clean = vals.map((v) => String(v).trim()).filter(Boolean);
            if (clean.length) defaults[k] = [...new Set(clean)];
        }
    }

    const skip = (Array.isArray(d.review_not_required) ? d.review_not_required : [])
        .filter((k) => ruleSet.has(k));

    return {
        prompt_author: d.prompt_author || '',
        prompt_author_institution: d.prompt_author_institution || '',
        prompt_name: d.prompt_name || stem(d.name),
        prompt_version: d.prompt_version || 'v-1-0',
        prompt_description: d.prompt_description || '',
        LLM: d.llm || '',
        instructions: d.instructions || '',
        json_formatting_instructions: CANONICAL_FORMATTING,
        rules,
        mapping,
        checklist: (Array.isArray(d.checklist) ? d.checklist : []).map(parseChecklistItem),
        field_default_values: defaults,
        review_not_required: [...new Set(skip)],
        custom_flags: (Array.isArray(d.custom_flags) ? d.custom_flags : []).map((x) => String(x)),
    };
}

function starterData(promptName) {
    return {
        prompt_author: 'Author',
        prompt_author_institution: '',
        prompt_name: promptName || 'Custom_Prompt',
        prompt_version: 'v-1-0',
        prompt_description: 'Custom extraction prompt.',
        LLM: '',
        instructions: '1. Refactor the unstructured OCR text into a JSON dictionary using the rules below.\n2. Return only a valid JSON object.',
        json_formatting_instructions: CANONICAL_FORMATTING,
        rules: [
            { name: 'catalogNumber', rule: 'Barcode or accession identifier.', _mappedKey: 'catalogNumber' },
            { name: 'scientificName', rule: 'Full scientific name of the specimen.', _mappedKey: 'scientificName' },
            { name: 'collectedBy', rule: 'Primary collector name.', _mappedKey: 'collectedBy' },
            { name: 'country', rule: 'Country of collection.', _mappedKey: 'country' },
        ],
        mapping: [
            { category: 'GEOGRAPHY', fields: ['catalogNumber', 'country'] },
            { category: 'TAXONOMY', fields: ['scientificName'] },
            { category: 'COLLECTING', fields: ['collectedBy'] },
        ],
        checklist: [
            { tag: 'VoucherVision', text: 'Accept fields that do not require review' },
            { tag: 'OCR', text: 'Review OCR disagreements' },
        ],
        field_default_values: { country: ['no data'] },
        review_not_required: [],
        custom_flags: [],
    };
}

function dataToPayload(data) {
    const mapping = {};
    for (const m of data.mapping) mapping[m.category] = [...m.fields];
    const checklist = data.checklist
        .map((item, idx) => formatChecklistItem(item, idx))
        .filter(Boolean);
    const fieldDefaultValues = {};
    for (const [k, vals] of Object.entries(data.field_default_values)) {
        const clean = vals.map((v) => v.trim()).filter(Boolean);
        if (clean.length) fieldDefaultValues[k] = clean;
    }
    return {
        prompt_author: data.prompt_author,
        prompt_author_institution: data.prompt_author_institution,
        prompt_name: data.prompt_name,
        prompt_version: data.prompt_version,
        prompt_description: data.prompt_description,
        LLM: data.LLM,
        instructions: data.instructions,
        json_formatting_instructions: CANONICAL_FORMATTING,
        rules: data.rules.map((r) => ({ name: r.name, rule: r.rule })),
        mapping,
        checklist,
        field_default_values: fieldDefaultValues,
        review_not_required: [...data.review_not_required],
        custom_flags: data.custom_flags.map((f) => f.trim()).filter(Boolean),
    };
}

function categoryOf(data, fieldName) {
    const hit = data.mapping.find((m) => m.fields.includes(fieldName));
    return hit ? hit.category : 'COLLECTING';
}

function moveFieldToCategory(data, fieldName, targetCat) {
    for (const m of data.mapping) m.fields = m.fields.filter((f) => f !== fieldName);
    const dest = data.mapping.find((m) => m.category === targetCat) || data.mapping[0];
    if (dest && !dest.fields.includes(fieldName)) dest.fields.push(fieldName);
}

function renameFieldReferences(data, oldName, newName) {
    if (!oldName || !newName || oldName === newName) return;
    for (const m of data.mapping) {
        m.fields = m.fields.map((f) => (f === oldName ? newName : f));
    }
    if (Object.prototype.hasOwnProperty.call(data.field_default_values, oldName)) {
        const vals = data.field_default_values[oldName];
        delete data.field_default_values[oldName];
        data.field_default_values[newName] = vals;
    }
    data.review_not_required = data.review_not_required.map((f) => (f === oldName ? newName : f));
}

function removeFieldReferences(data, fieldName) {
    for (const m of data.mapping) m.fields = m.fields.filter((f) => f !== fieldName);
    delete data.field_default_values[fieldName];
    data.review_not_required = data.review_not_required.filter((f) => f !== fieldName);
}

function sanitizePromptName(text) {
    return String(text || 'Custom_Prompt')
        .trim()
        .replace(/[^A-Za-z0-9_-]+/g, '_')
        .replace(/_+/g, '_')
        .replace(/^[_-]+|[_-]+$/g, '') || 'Custom_Prompt';
}

function filenameForPromptName(promptName) {
    const clean = String(promptName || '').trim();
    return clean ? `${clean}.yaml` : '';
}

function suggestUniqueNames(baseStem, allPrompts) {
    const stemSet = new Set(allPrompts.map((p) => stem(p.name).toLowerCase()));
    const titleSet = new Set(allPrompts.map((p) => (p.prompt_name || '').trim().toLowerCase()));
    const cleanBase = sanitizePromptName(baseStem);
    const taken = (candidate) => {
        const low = candidate.toLowerCase();
        return stemSet.has(low) || titleSet.has(low);
    };
    if (!taken(cleanBase)) {
        return { filename: `${cleanBase}.yaml`, promptName: cleanBase };
    }
    for (let i = 1; i < 500; i += 1) {
        const suffix = i === 1 ? '_custom' : `_custom_${i}`;
        const pname = `${cleanBase}${suffix}`;
        if (!taken(pname)) {
            return { filename: `${pname}.yaml`, promptName: pname };
        }
    }
    const fallback = `${cleanBase}_${Date.now()}`;
    return { filename: `${fallback}.yaml`, promptName: fallback };
}

function validateDraft(draft, allPrompts) {
    const errors = [];
    if (!draft) return { valid: false, errors };
    const d = draft.data;
    const excludeFile = draft.source === 'user' ? draft.filename.toLowerCase() : null;
    const pname = d.prompt_name.trim();

    if (!pname) {
        errors.push({ section: 'info', message: 'Prompt name is required.' });
    } else if (draft.source !== 'builtin') {
        if (!PROMPT_NAME_RE.test(pname)) {
            errors.push({ section: 'info', message: `Invalid prompt name. ${PROMPT_NAME_RULE}` });
        } else {
            const targetLow = pname.toLowerCase();
            const dupFile = allPrompts.find((p) => stem(p.name).toLowerCase() === targetLow
                && !(p.source === 'user' && excludeFile && p.name.toLowerCase() === excludeFile));
            if (dupFile) errors.push({ section: 'info', message: `File name '${pname}.yaml' is already used.` });
            const dupName = allPrompts.find((p) => (p.prompt_name || '').trim().toLowerCase() === targetLow
                && !(p.source === 'user' && excludeFile && p.name.toLowerCase() === excludeFile));
            if (dupName) errors.push({ section: 'info', message: `Prompt name '${pname}' is already used by ${dupName.name}.` });
        }
    }
    if (!d.prompt_version.trim()) errors.push({ section: 'info', message: 'Version is required.' });
    if (!d.prompt_author.trim()) errors.push({ section: 'info', message: 'Author is required.' });
    if (!d.prompt_description.trim()) errors.push({ section: 'info', message: 'Description is required.' });

    if (!d.instructions.trim()) errors.push({ section: 'prompt', message: 'Instructions are required.' });
    if (!d.rules.length) {
        errors.push({ section: 'prompt', message: 'Add at least one field.' });
    } else {
        const seen = new Set();
        for (const r of d.rules) {
            const key = r.name.trim();
            if (!FIELD_RE.test(key)) {
                errors.push({ section: 'prompt', message: `Invalid field name '${r.name || '(blank)'}'. Use letters, digits, underscores.` });
            } else if (seen.has(key.toLowerCase())) {
                errors.push({ section: 'prompt', message: `Duplicate field name '${key}'.` });
            } else {
                seen.add(key.toLowerCase());
            }
        }
    }

    const cats = d.mapping.map((m) => m.category);
    for (const req of REQUIRED_CATS) {
        if (!cats.includes(req)) errors.push({ section: 'mapping', message: `Required category ${req} is missing.` });
    }
    if (cats.length > MAX_CATS) errors.push({ section: 'mapping', message: `Maximum ${MAX_CATS} mapping categories allowed.` });
    const mapped = new Set();
    for (const m of d.mapping) {
        if (!CAT_RE.test(m.category)) errors.push({ section: 'mapping', message: `Category '${m.category}' must be ALL CAPS.` });
        for (const f of m.fields) mapped.add(f);
    }
    const unmapped = d.rules.map((r) => r.name.trim()).filter((k) => k && !mapped.has(k));
    if (unmapped.length) errors.push({ section: 'mapping', message: `Unmapped field(s): ${unmapped.join(', ')}.` });

    return { valid: errors.length === 0, errors };
}

// Simple word-level diff (LCS on whitespace-separated tokens, bounded for responsiveness).
function wordDiffNodes(leftText, rightText) {
    const a = String(leftText || '').split(/(\s+)/);
    const b = String(rightText || '').split(/(\s+)/);
    if (a.length * b.length > 160000) {
        return [leftText === rightText ? leftText : el('span', { class: 'pd-diff-del' }, leftText || '—')];
    }
    const m = a.length;
    const n = b.length;
    const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
    for (let i = m - 1; i >= 0; i -= 1) {
        for (let j = n - 1; j >= 0; j -= 1) {
            dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
    }
    const out = [];
    let i = 0;
    let j = 0;
    while (i < m && j < n) {
        if (a[i] === b[j]) {
            out.push(a[i]);
            i += 1;
            j += 1;
        } else if (dp[i + 1][j] >= dp[i][j + 1]) {
            out.push(el('span', { class: 'pd-diff-del' }, a[i]));
            i += 1;
        } else {
            out.push(el('span', { class: 'pd-diff-ins' }, b[j]));
            j += 1;
        }
    }
    while (i < m) { out.push(el('span', { class: 'pd-diff-del' }, a[i])); i += 1; }
    while (j < n) { out.push(el('span', { class: 'pd-diff-ins' }, b[j])); j += 1; }
    return out.length ? out : ['—'];
}

// ---------------------------------------------------------------- main mount

export function mountPromptDesigner(host, { getPrompts, getUserDir, onSaved, onUse }) {
    let draft = null;          // { source: 'builtin' | 'user', filename, data, dirty }
    let compareRef = null;     // { source, name }
    let compareData = null;    // normalized data of compare prompt
    let mode = 'edit';         // 'edit' | 'compare'
    let activeStep = 'info';
    let showIssues = false;
    let diffOnly = false;
    let ruleFilter = '';
    let saving = false;

    const toolbar = el('div', { class: 'pd-toolbar' });
    const banner = el('div');
    const issuesHost = el('div');
    const workspace = el('div');

    // Shared datalist for checklist tags
    const tagListId = 'pdChecklistTags';
    const tagList = el('datalist', { id: tagListId }, CHECKLIST_TAGS.map((t) => el('option', { value: t })));

    replace(host, toolbar, banner, issuesHost, workspace, tagList);

    const isReadOnly = () => !draft || draft.source === 'builtin';

    function markDirty() {
        if (!draft || isReadOnly()) return;
        draft.dirty = true;
        renderChrome();
    }

    async function fetchPromptDetail(source, name) {
        return api.get(`/v1/prompts/${encodeURIComponent(source)}/${encodeURIComponent(name)}`);
    }

    async function loadActive(source, name) {
        replace(workspace, stateBox('loading', `Loading ${name}…`));
        try {
            const d = await fetchPromptDetail(source, name);
            draft = { source: d.source, filename: d.name, data: detailToData(d), dirty: false };
            showIssues = false;
            renderAll();
        } catch (err) {
            replace(workspace, stateBox('error', `Could not load ${name}`, plainError(err)));
        }
    }

    async function loadCompare(source, name) {
        compareRef = { source, name };
        try {
            const d = await fetchPromptDetail(source, name);
            compareData = detailToData(d);
        } catch (err) {
            compareData = null;
            toastError(err, `Could not load ${name}`);
        }
        if (mode === 'compare') renderWorkspace();
    }

    // ------------------------------------------------------------ copy / new modal
    function openCopyModal({ baseData, baseStem, title = 'Copy prompt', submitLabel = 'Create copy' }) {
        const all = getPrompts();
        const suggested = suggestUniqueNames(baseStem, all);

        const sourceSelect = !baseData ? el('select', { id: 'pdNewBase' },
            el('option', { value: 'builtin/SLTPvM_full.yaml' }, 'SLTPvM_full (recommended)'),
            el('option', { value: '__starter__' }, 'Minimal starter (4 fields)'),
            all.filter((p) => !(p.source === 'builtin' && p.name === 'SLTPvM_full.yaml')).map((p) =>
                el('option', { value: `${p.source}/${p.name}` }, `${p.prompt_name || p.name} (${p.source === 'user' ? 'Custom' : 'Built-in'})`))) : null;

        const nameInput = el('input', {
            type: 'text', id: 'pdCopyName', value: suggested.promptName, autocomplete: 'off', spellcheck: 'false',
            placeholder: 'My_Custom-Prompt',
        });
        const filePreviewVal = el('span', null, suggested.filename);
        const filePreview = el('div', { class: 'pd-filename-preview' },
            el('span', { class: 'pd-filename-preview-label' }, 'File name:'),
            filePreviewVal);
        const errBox = el('div', { class: 'modal-problem', role: 'alert', hidden: true });
        const cancelBtn = el('button', { type: 'button', class: 'button-secondary' }, 'Cancel');
        const submitBtn = el('button', { type: 'button', class: 'button' }, submitLabel);

        function checkOverlap() {
            const pname = nameInput.value.trim();
            const fname = filenameForPromptName(pname);
            filePreviewVal.textContent = fname || '—';
            const msgs = [];
            if (!pname) {
                msgs.push('Prompt name is required.');
            } else if (!PROMPT_NAME_RE.test(pname)) {
                msgs.push(PROMPT_NAME_RULE);
            } else {
                const low = pname.toLowerCase();
                if (all.some((p) => stem(p.name).toLowerCase() === low || (p.prompt_name || '').trim().toLowerCase() === low)) {
                    msgs.push(`'${pname}' is already used. Pick a different name.`);
                }
            }
            if (msgs.length) {
                replace(errBox, icon('alert', { size: 16 }), el('span', null, msgs.join(' ')));
                errBox.hidden = false;
                submitBtn.disabled = true;
            } else {
                errBox.hidden = true;
                submitBtn.disabled = false;
            }
            return msgs.length === 0;
        }

        nameInput.addEventListener('input', checkOverlap);

        const body = el('div', { class: 'pd-modal-form' },
            sourceSelect ? el('div', { class: 'pd-field' }, el('label', { for: 'pdNewBase' }, 'Start from'), sourceSelect) : null,
            el('div', { class: 'pd-field' },
                el('label', { for: 'pdCopyName' }, 'Prompt name'),
                nameInput,
                el('span', { class: 'pd-field-hint' }, PROMPT_NAME_RULE)),
            filePreview,
            errBox,
            el('div', { class: 'modal-actions' }, cancelBtn, submitBtn));

        const modal = openModal({ title, body, className: 'modal-confirm' });
        cancelBtn.addEventListener('click', () => modal.close());

        submitBtn.addEventListener('click', async () => {
            if (!checkOverlap()) return;
            submitBtn.disabled = true;
            try {
                const pname = nameInput.value.trim();
                let cloned;
                if (baseData) {
                    cloned = structuredClone(baseData);
                } else if (sourceSelect.value === '__starter__') {
                    cloned = starterData(pname);
                } else {
                    const [src, ...rest] = sourceSelect.value.split('/');
                    const fetched = await fetchPromptDetail(src, rest.join('/'));
                    cloned = detailToData(fetched);
                }
                cloned.prompt_name = pname;
                cloned.json_formatting_instructions = CANONICAL_FORMATTING;
                if (!cloned.prompt_author.trim()) cloned.prompt_author = 'Author';
                if (!cloned.prompt_version.trim()) cloned.prompt_version = 'v-1-0';
                if (!cloned.prompt_description.trim()) cloned.prompt_description = 'Custom extraction prompt.';
                if (!cloned.instructions.trim()) cloned.instructions = 'Refactor the OCR text into a valid JSON object using the rules below.';
                const fname = filenameForPromptName(pname);
                const created = await api.post('/v1/prompts/user', { name: fname, prompt: dataToPayload(cloned) });
                draft = { source: 'user', filename: created.name, data: detailToData(created), dirty: false };
                modal.close();
                toast(`Created ${created.name}.`, 'success');
                if (onSaved) await onSaved({ source: 'user', name: created.name });
                renderAll();
            } catch (err) {
                replace(errBox, icon('alert', { size: 16 }), el('span', null, plainError(err)));
                errBox.hidden = false;
                submitBtn.disabled = false;
            }
        });

        checkOverlap();
        nameInput.focus();
        nameInput.select();
    }

    // ------------------------------------------------------------ save / export / preview
    async function saveDraft() {
        if (!draft || isReadOnly() || saving) return;
        const { valid } = validateDraft(draft, getPrompts());
        if (!valid) {
            showIssues = true;
            renderChrome();
            toast('Fix validation issues before saving.', 'warning');
            return;
        }
        saving = true;
        renderChrome();
        try {
            const updated = await api.put(`/v1/prompts/user/${encodeURIComponent(draft.filename)}`, { prompt: dataToPayload(draft.data) });
            draft = { source: 'user', filename: updated.name, data: detailToData(updated), dirty: false };
            toast(`Saved ${updated.name}.`, 'success');
            if (onSaved) await onSaved({ source: 'user', name: updated.name });
        } catch (err) {
            toastError(err, 'Save failed');
        } finally {
            saving = false;
            renderAll();
        }
    }

    async function exportYaml() {
        if (!draft) return;
        const { valid } = validateDraft(draft, getPrompts());
        if (!valid) {
            showIssues = true;
            renderChrome();
            toast('Fix validation issues before exporting.', 'warning');
            return;
        }
        try {
            const res = await api.post('/v1/prompts/render', { prompt: dataToPayload(draft.data) });
            const fname = filenameForPromptName(draft.data.prompt_name) || draft.filename;
            downloadText(res.yaml_text, fname, 'text/yaml;charset=utf-8');
        } catch (err) {
            toastError(err, 'Export failed');
        }
    }

    async function previewYaml() {
        if (!draft) return;
        try {
            const res = await api.post('/v1/prompts/render', { prompt: dataToPayload(draft.data) });
            const fname = filenameForPromptName(draft.data.prompt_name) || draft.filename;
            const copyBtn = el('button', {
                type: 'button', class: 'button-secondary button-small',
                onclick: async () => toast((await copyText(res.yaml_text)) ? 'Copied.' : 'Copy failed.', 'info'),
            }, icon('copy', { size: 13 }), 'Copy');
            const dlBtn = el('button', {
                type: 'button', class: 'button button-small',
                onclick: () => downloadText(res.yaml_text, fname, 'text/yaml;charset=utf-8'),
            }, icon('download', { size: 13 }), 'Export YAML');
            openModal({
                title: fname,
                body: [
                    el('div', { class: 'json-controls' }, copyBtn, dlBtn),
                    el('pre', { class: 'json-content pd-yaml-pre' }, res.yaml_text),
                ],
            });
        } catch (err) {
            toastError(err, 'Cannot render YAML');
        }
    }

    // ------------------------------------------------------------ toolbar & chrome
    function promptSelectElement(currentValue, onChange, ariaLabel) {
        const all = getPrompts();
        const userList = all.filter((p) => p.source === 'user');
        const builtinList = all.filter((p) => p.source === 'builtin');
        const sel = el('select', { class: 'pd-select', 'aria-label': ariaLabel, onchange: () => onChange(sel.value) },
            userList.length ? el('optgroup', { label: 'Custom' }, userList.map((p) =>
                el('option', { value: `user/${p.name}` }, p.prompt_name || p.name))) : null,
            builtinList.length ? el('optgroup', { label: 'Built-in' }, builtinList.map((p) =>
                el('option', { value: `builtin/${p.name}` }, p.prompt_name || p.name))) : null);
        if (currentValue) sel.value = currentValue;
        return sel;
    }

    function buildActiveSelect(activeVal) {
        const sel = promptSelectElement(activeVal, async (val) => {
            if (draft && draft.dirty) {
                const ok = await confirmDialog({
                    title: 'Discard unsaved changes?',
                    message: `You have unsaved edits in ${draft.filename}.`,
                    confirmLabel: 'Discard',
                    danger: true,
                });
                if (!ok) {
                    sel.value = activeVal;
                    return;
                }
            }
            const [src, ...rest] = val.split('/');
            loadActive(src, rest.join('/'));
        }, 'Active prompt');
        return sel;
    }

    function renderChrome() {
        const all = getPrompts();
        const check = draft ? validateDraft(draft, all) : { valid: true, errors: [] };
        const activeVal = draft ? `${draft.source}/${draft.filename}` : '';

        const editBtn = el('button', { type: 'button', 'aria-pressed': String(mode === 'edit'), onclick: () => { mode = 'edit'; renderAll(); } }, 'Edit');
        const cmpBtn = el('button', {
            type: 'button', 'aria-pressed': String(mode === 'compare'),
            onclick: async () => {
                mode = 'compare';
                if (!compareRef && all.length) {
                    const fallback = all.find((p) => `${p.source}/${p.name}` !== activeVal) || all[0];
                    await loadCompare(fallback.source, fallback.name);
                }
                renderAll();
            },
        }, icon('columns', { size: 13 }), ' Compare');

        const newBtn = el('button', {
            type: 'button', class: 'button-secondary button-small',
            onclick: () => openCopyModal({ baseData: null, baseStem: 'Custom_Prompt', title: 'New custom prompt', submitLabel: 'Create' }),
        }, icon('plus', { size: 13 }), 'New');

        const statusBtn = draft
            ? el('button', {
                type: 'button', class: 'pd-status-btn',
                title: check.valid ? 'Prompt is valid' : 'Click to view validation issues',
                onclick: () => { showIssues = !showIssues; renderChrome(); },
            }, check.valid ? badge('Valid', 'good') : badge(`${check.errors.length} issue${check.errors.length === 1 ? '' : 's'}`, 'warn'))
            : null;

        const userDir = getUserDir ? getUserDir() : '';
        const saveTip = userDir
            ? `Saved in ${userDir}. Kept across app updates and reinstalls.`
            : 'Saved in your user folder. Kept across app updates.';

        const rightActions = [];
        if (draft) {
            if (isReadOnly()) {
                rightActions.push(el('button', {
                    type: 'button', class: 'button button-small',
                    onclick: () => openCopyModal({ baseData: draft.data, baseStem: stem(draft.filename) }),
                }, icon('copy', { size: 13 }), 'Copy to edit'));
            } else {
                rightActions.push(
                    el('button', {
                        type: 'button', class: 'button button-small',
                        disabled: saving || !check.valid,
                        onclick: saveDraft,
                    }, icon('save', { size: 13 }), saving ? 'Saving…' : 'Save', draft.dirty ? el('span', { class: 'pd-dirty-dot', title: 'Unsaved changes' }) : null),
                    infoTip(saveTip, 'end'),
                    el('button', {
                        type: 'button', class: 'button-secondary button-small',
                        onclick: () => openCopyModal({ baseData: draft.data, baseStem: stem(draft.filename), title: 'Duplicate prompt', submitLabel: 'Duplicate' }),
                    }, icon('copy', { size: 13 }), 'Duplicate'),
                    el('button', {
                        type: 'button', class: 'button-secondary button-small',
                        disabled: !check.valid || draft.dirty,
                        title: draft.dirty ? 'Save changes first' : 'Select this prompt on the VVL tab',
                        onclick: () => onUse && onUse({ source: 'user', name: draft.filename, prompt_name: draft.data.prompt_name }),
                    }, 'Use'),
                );
            }
            rightActions.push(
                el('button', { type: 'button', class: 'button-secondary button-small', disabled: !check.valid, onclick: exportYaml }, icon('download', { size: 13 }), 'Export YAML'),
                el('button', { type: 'button', class: 'button-secondary button-small', disabled: !check.valid, onclick: previewYaml }, 'YAML'),
            );
        }

        // In Compare mode, Active & Compare selectors live in their aligned column header below.
        const leftTools = mode === 'edit'
            ? [
                el('div', { class: 'pd-toolbar-group' }, el('label', null, 'Active'), buildActiveSelect(activeVal), newBtn),
                el('div', { class: 'org-seg', role: 'group', 'aria-label': 'Designer view' }, editBtn, cmpBtn),
                statusBtn,
            ]
            : [
                el('div', { class: 'org-seg', role: 'group', 'aria-label': 'Designer view' }, editBtn, cmpBtn),
                newBtn,
                statusBtn,
            ];

        replace(toolbar,
            ...leftTools,
            el('span', { class: 'spacer' }),
            el('div', { class: 'pd-toolbar-group' }, rightActions));

        if (draft && isReadOnly()) {
            replace(banner, el('div', { class: 'pd-readonly-bar' },
                icon('info', { size: 17 }),
                el('span', null, 'Built-in prompt (read-only). Make a copy to customize it.'),
                el('span', { class: 'spacer' }),
                el('button', {
                    type: 'button', class: 'button button-small',
                    onclick: () => openCopyModal({ baseData: draft.data, baseStem: stem(draft.filename) }),
                }, icon('copy', { size: 13 }), 'Copy to edit')));
        } else {
            clear(banner);
        }

        if (draft && !check.valid && showIssues) {
            replace(issuesHost, el('div', { class: 'pd-issues', role: 'alert' },
                el('strong', null, 'Fix before saving:'),
                el('ul', null, check.errors.map((e) => el('li', null, e.message)))));
        } else {
            clear(issuesHost);
        }
    }

    // ------------------------------------------------------------ edit sections
    function fieldInput(label, value, { required = false, multiline = false, rows = 3, mono = false, tip = null, hint = null, badgeText = null, readonly = false, onInput }) {
        const ro = readonly || isReadOnly();
        const ctrl = multiline
            ? el('textarea', { rows: String(rows), readOnly: ro, oninput: () => onInput && onInput(ctrl.value) }, value)
            : el('input', { type: 'text', value, readOnly: ro, class: mono ? 'pd-rule-key' : null, oninput: () => onInput && onInput(ctrl.value) });
        if (multiline) ctrl.value = value;
        return el('div', { class: ['pd-field', multiline ? 'pd-span-2' : ''] },
            el('label', null,
                label,
                required ? el('span', { class: 'pd-req' }, '*') : null,
                badgeText ? badge(badgeText, 'neutral') : null,
                infoTip(tip, 'start')),
            ctrl,
            hint ? el('span', { class: 'pd-field-hint' }, hint) : null);
    }

    function renderInfoStep() {
        const d = draft.data;
        const fileBadge = badge(filenameForPromptName(d.prompt_name) || draft.filename, 'neutral');
        return el('div', { class: 'pd-card' },
            el('div', { class: 'pd-card-head' }, el('h3', null, 'Versioning & Info'), fileBadge),
            el('div', { class: 'pd-grid-2' },
                fieldInput('Prompt name', d.prompt_name, {
                    required: true,
                    mono: true,
                    hint: PROMPT_NAME_RULE,
                    tip: 'Matches the YAML file name. Must be unique across all prompts.',
                    onInput: (v) => {
                        d.prompt_name = v;
                        fileBadge.textContent = filenameForPromptName(v) || '—';
                        markDirty();
                    },
                }),
                fieldInput('Version', d.prompt_version, { required: true, tip: 'e.g. v-1-0', onInput: (v) => { d.prompt_version = v; markDirty(); } }),
                fieldInput('Author', d.prompt_author, { required: true, onInput: (v) => { d.prompt_author = v; markDirty(); } }),
                fieldInput('Institution', d.prompt_author_institution, { onInput: (v) => { d.prompt_author_institution = v; markDirty(); } }),
                fieldInput('Target LLM', d.LLM, { tip: 'Recommended model types for this prompt.', onInput: (v) => { d.LLM = v; markDirty(); } }),
                fieldInput('Description', d.prompt_description, { required: true, multiline: true, rows: 3, onInput: (v) => { d.prompt_description = v; markDirty(); } })));
    }

    function renderPromptStep() {
        const d = draft.data;
        const ro = isReadOnly();
        const cats = d.mapping.map((m) => m.category);

        const searchInput = el('input', {
            type: 'text', class: 'pd-search', placeholder: 'Filter fields…', value: ruleFilter, 'aria-label': 'Filter fields',
            oninput: () => { ruleFilter = searchInput.value; renderRulesList(); },
        });

        const rulesHost = el('div', { class: 'pd-rules-list' });

        function renderRulesList() {
            const q = ruleFilter.trim().toLowerCase();
            const counts = new Map();
            for (const r of d.rules) {
                const k = r.name.trim().toLowerCase();
                counts.set(k, (counts.get(k) || 0) + 1);
            }
            const rows = d.rules.map((r, idx) => ({ r, idx })).filter(({ r }) => !q || `${r.name} ${r.rule}`.toLowerCase().includes(q));
            replace(rulesHost, rows.length ? rows.map(({ r, idx }) => {
                const keyTrim = r.name.trim();
                const invalid = !FIELD_RE.test(keyTrim) || (counts.get(keyTrim.toLowerCase()) || 0) > 1;
                const keyInput = el('input', {
                    type: 'text', class: 'pd-rule-key', value: r.name, readOnly: ro, placeholder: 'camelCaseField', 'aria-label': 'Field name',
                });
                keyInput.addEventListener('change', () => {
                    const prevMapped = r._mappedKey || r.name;
                    const next = keyInput.value.trim();
                    r.name = next;
                    if (FIELD_RE.test(next) && !d.rules.some((other, oi) => oi !== idx && other.name.toLowerCase() === next.toLowerCase())) {
                        renameFieldReferences(d, prevMapped, next);
                        if (!d.mapping.some((m) => m.fields.includes(next))) moveFieldToCategory(d, next, 'COLLECTING');
                        r._mappedKey = next;
                    }
                    markDirty();
                    renderRulesList();
                });

                const catSel = el('select', { disabled: ro, 'aria-label': `Category for ${r.name}` },
                    cats.map((c) => el('option', { value: c }, c)));
                catSel.value = categoryOf(d, r.name);
                catSel.addEventListener('change', () => {
                    moveFieldToCategory(d, r.name, catSel.value);
                    markDirty();
                });

                const ruleArea = el('textarea', { rows: '2', readOnly: ro, placeholder: 'Rule for extracting this field…', 'aria-label': `Rule for ${r.name}` });
                ruleArea.value = r.rule;
                ruleArea.addEventListener('input', () => { r.rule = ruleArea.value; markDirty(); });

                const upBtn = el('button', {
                    type: 'button', class: 'pd-mini-btn', title: 'Move up', 'aria-label': 'Move up', disabled: ro || idx === 0,
                    onclick: () => { const [x] = d.rules.splice(idx, 1); d.rules.splice(idx - 1, 0, x); markDirty(); renderRulesList(); },
                }, icon('arrowUp', { size: 12 }));
                const dnBtn = el('button', {
                    type: 'button', class: 'pd-mini-btn', title: 'Move down', 'aria-label': 'Move down', disabled: ro || idx === d.rules.length - 1,
                    onclick: () => { const [x] = d.rules.splice(idx, 1); d.rules.splice(idx + 1, 0, x); markDirty(); renderRulesList(); },
                }, icon('arrowDown', { size: 12 }));
                const delBtn = el('button', {
                    type: 'button', class: 'pd-mini-btn is-danger', title: 'Remove field', 'aria-label': 'Remove field', disabled: ro || d.rules.length <= 1,
                    onclick: () => {
                        const [removed] = d.rules.splice(idx, 1);
                        removeFieldReferences(d, removed._mappedKey || removed.name);
                        removeFieldReferences(d, removed.name);
                        markDirty();
                        renderWorkspace();
                    },
                }, icon('x', { size: 13 }));

                return el('div', { class: ['pd-rule-row', invalid ? 'is-invalid' : ''] },
                    el('div', { class: 'pd-order-btns' }, upBtn, dnBtn),
                    keyInput,
                    catSel,
                    ruleArea,
                    delBtn);
            }) : el('p', { class: 'help-text' }, 'No matching fields.'));
        }

        renderRulesList();

        return [
            el('div', { class: 'pd-card' },
                el('div', { class: 'pd-card-head' }, el('h3', null, 'LLM Instructions')),
                el('div', { class: 'pd-grid-2' },
                    fieldInput('Instructions', d.instructions, { required: true, multiline: true, rows: 7, tip: 'General instructions sent to the model before field rules.', onInput: (v) => { d.instructions = v; markDirty(); } }),
                    fieldInput('Formatting instructions', CANONICAL_FORMATTING, { multiline: true, rows: 2, readonly: true, badgeText: 'Read-only', tip: 'Standard formatting instructions required for all prompts.' }))),
            el('div', { class: 'pd-card' },
                el('div', { class: 'pd-rules-head' },
                    el('h3', null, `Fields (${d.rules.length})`),
                    infoTip('Use camelCase keys. Each field is automatically synced with Mapping.'),
                    el('span', { class: 'spacer' }),
                    searchInput,
                    !ro ? el('button', {
                        type: 'button', class: 'button button-small',
                        onclick: () => {
                            let n = d.rules.length + 1;
                            let name = `customField${n}`;
                            while (d.rules.some((x) => x.name.toLowerCase() === name.toLowerCase())) { n += 1; name = `customField${n}`; }
                            d.rules.push({ name, rule: '', _mappedKey: name });
                            moveFieldToCategory(d, name, 'COLLECTING');
                            ruleFilter = '';
                            markDirty();
                            renderWorkspace();
                        },
                    }, icon('plus', { size: 13 }), 'Add field') : null),
                rulesHost),
        ];
    }

    function renderMappingStep() {
        const d = draft.data;
        const ro = isReadOnly();
        const catNames = d.mapping.map((m) => m.category);

        const newCatInput = el('input', { type: 'text', placeholder: 'NEW_CATEGORY', maxLength: '32', disabled: ro || d.mapping.length >= MAX_CATS, 'aria-label': 'New category name' });
        const addCatBtn = el('button', {
            type: 'button', class: 'button-secondary button-small', disabled: ro || d.mapping.length >= MAX_CATS,
            onclick: () => {
                const raw = newCatInput.value.trim().toUpperCase().replace(/\s+/g, '_');
                if (!CAT_RE.test(raw)) {
                    toast('Use ALL CAPS letters, digits, and underscores.', 'warning');
                    return;
                }
                if (d.mapping.some((m) => m.category === raw)) {
                    toast(`Category ${raw} already exists.`, 'warning');
                    return;
                }
                d.mapping.push({ category: raw, fields: [] });
                newCatInput.value = '';
                markDirty();
                renderWorkspace();
            },
        }, icon('plus', { size: 13 }), 'Add category');

        newCatInput.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); addCatBtn.click(); } });

        return el('div', { class: 'pd-card' },
            el('div', { class: 'pd-card-head' },
                el('h3', null, `Mapping (${d.mapping.length}/${MAX_CATS})`),
                infoTip('Every field appears in one VVGO-Editor category. GEOGRAPHY, TAXONOMY, and COLLECTING are required; add up to 3 custom ALL-CAPS categories.')),
            !ro ? el('div', { class: 'pd-cat-add' }, newCatInput, addCatBtn) : null,
            el('div', { class: 'pd-cat-grid' }, d.mapping.map((m, ci) => {
                const required = REQUIRED_CATS.includes(m.category);
                return el('div', { class: 'pd-cat-card' },
                    el('div', { class: 'pd-cat-head' },
                        el('span', { class: 'pd-cat-name' }, m.category),
                        badge(String(m.fields.length), 'neutral'),
                        required ? badge('Required', 'good') : null,
                        el('span', { class: 'spacer' }),
                        !required && !ro ? el('button', {
                            type: 'button', class: 'pd-mini-btn is-danger', title: 'Remove category (moves fields to COLLECTING)', 'aria-label': `Remove ${m.category}`,
                            onclick: () => {
                                const col = d.mapping.find((x) => x.category === 'COLLECTING');
                                for (const f of m.fields) if (col && !col.fields.includes(f)) col.fields.push(f);
                                d.mapping.splice(ci, 1);
                                markDirty();
                                renderWorkspace();
                            },
                        }, icon('x', { size: 12 })) : null),
                    el('div', { class: 'pd-cat-list' }, m.fields.length ? m.fields.map((fname, fi) => {
                        const moveSel = el('select', { disabled: ro, 'aria-label': `Move ${fname}` },
                            catNames.map((c) => el('option', { value: c }, c)));
                        moveSel.value = m.category;
                        moveSel.addEventListener('change', () => {
                            moveFieldToCategory(d, fname, moveSel.value);
                            markDirty();
                            renderWorkspace();
                        });
                        return el('div', { class: 'pd-cat-item' },
                            el('span', { class: 'pd-cat-item-key' }, fname),
                            !ro ? el('button', {
                                type: 'button', class: 'pd-mini-btn', disabled: fi === 0, title: 'Move up', 'aria-label': 'Move up',
                                onclick: () => { const [x] = m.fields.splice(fi, 1); m.fields.splice(fi - 1, 0, x); markDirty(); renderWorkspace(); },
                            }, icon('arrowUp', { size: 11 })) : null,
                            !ro ? el('button', {
                                type: 'button', class: 'pd-mini-btn', disabled: fi === m.fields.length - 1, title: 'Move down', 'aria-label': 'Move down',
                                onclick: () => { const [x] = m.fields.splice(fi, 1); m.fields.splice(fi + 1, 0, x); markDirty(); renderWorkspace(); },
                            }, icon('arrowDown', { size: 11 })) : null,
                            moveSel);
                    }) : el('span', { class: 'help-text' }, 'No fields')));
            })));
    }

    function renderChecklistStep() {
        const d = draft.data;
        const ro = isReadOnly();
        return el('div', { class: 'pd-card' },
            el('div', { class: 'pd-card-head' },
                el('h3', null, `Checklist (${d.checklist.length})`),
                infoTip('Editor workflow steps. Automatically numbered as "1. [Tag] - Description" on export.'),
                el('span', { class: 'spacer' }),
                !ro ? el('button', {
                    type: 'button', class: 'button button-small',
                    onclick: () => { d.checklist.push({ tag: '', text: '' }); markDirty(); renderWorkspace(); },
                }, icon('plus', { size: 13 }), 'Add step') : null),
            el('div', { class: 'pd-check-list' }, d.checklist.length ? d.checklist.map((item, idx) => {
                const tagIn = el('input', {
                    type: 'text', value: item.tag, list: tagListId, readOnly: ro, placeholder: 'Tag (optional)', 'aria-label': `Tag for step ${idx + 1}`,
                    oninput: () => { item.tag = tagIn.value; markDirty(); },
                });
                const textIn = el('input', {
                    type: 'text', value: item.text, readOnly: ro, placeholder: 'Step description', 'aria-label': `Step ${idx + 1}`,
                    oninput: () => { item.text = textIn.value; markDirty(); },
                });
                return el('div', { class: 'pd-check-row' },
                    el('span', { class: 'pd-check-num' }, `${idx + 1}.`),
                    tagIn,
                    textIn,
                    el('div', { class: 'pd-check-order' },
                        el('button', {
                            type: 'button', class: 'pd-mini-btn', disabled: ro || idx === 0, title: 'Move up', 'aria-label': 'Move up',
                            onclick: () => { const [x] = d.checklist.splice(idx, 1); d.checklist.splice(idx - 1, 0, x); markDirty(); renderWorkspace(); },
                        }, icon('arrowUp', { size: 12 })),
                        el('button', {
                            type: 'button', class: 'pd-mini-btn', disabled: ro || idx === d.checklist.length - 1, title: 'Move down', 'aria-label': 'Move down',
                            onclick: () => { const [x] = d.checklist.splice(idx, 1); d.checklist.splice(idx + 1, 0, x); markDirty(); renderWorkspace(); },
                        }, icon('arrowDown', { size: 12 }))),
                    el('button', {
                        type: 'button', class: 'pd-mini-btn is-danger', disabled: ro, title: 'Remove step', 'aria-label': 'Remove step',
                        onclick: () => { d.checklist.splice(idx, 1); markDirty(); renderWorkspace(); },
                    }, icon('x', { size: 13 })));
            }) : el('p', { class: 'help-text' }, 'No checklist steps yet.')));
    }

    let selectedDefaultField = '';

    function renderEditorStep() {
        const d = draft.data;
        const ro = isReadOnly();
        const allFieldNames = d.rules.map((r) => r.name).filter((k) => FIELD_RE.test(k));
        if (selectedDefaultField && !allFieldNames.includes(selectedDefaultField)) selectedDefaultField = '';

        const addDefaultSel = el('select', {
            id: 'pdDefaultFieldSel', disabled: ro || !allFieldNames.length,
            onchange: () => { selectedDefaultField = addDefaultSel.value; syncAddBtn(); },
        },
        el('option', { value: '' }, 'Select a field…'),
        allFieldNames.map((f) => el('option', { value: f }, f)));
        addDefaultSel.value = selectedDefaultField;

        const addDefaultVal = el('input', {
            type: 'text', id: 'pdDefaultValInput', disabled: ro || !allFieldNames.length,
            placeholder: 'e.g. not present, no data',
            oninput: () => syncAddBtn(),
        });

        const addDefaultBtn = el('button', {
            type: 'button', class: 'button button-small', disabled: true,
            onclick: submitDefault,
        }, icon('plus', { size: 13 }), 'Add default value');

        function syncAddBtn() {
            addDefaultBtn.disabled = ro || !addDefaultSel.value || !addDefaultVal.value.trim();
        }

        function submitDefault() {
            const fname = addDefaultSel.value;
            const val = addDefaultVal.value.trim();
            if (!fname || !val) return;
            if (!Array.isArray(d.field_default_values[fname])) d.field_default_values[fname] = [];
            if (!d.field_default_values[fname].includes(val)) d.field_default_values[fname].push(val);
            selectedDefaultField = fname;
            addDefaultVal.value = '';
            markDirty();
            renderWorkspace();
        }

        addDefaultVal.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter') { ev.preventDefault(); submitDefault(); }
        });

        const defaultEntries = Object.entries(d.field_default_values);

        return [
            // Default values
            el('div', { class: 'pd-card' },
                el('div', { class: 'pd-card-head' },
                    el('h3', null, 'Default values'),
                    infoTip('Choose a field and add quick-fill default values offered in VVGO-Editor.')),
                !ro ? el('div', { class: 'pd-default-adder' },
                    el('div', { class: 'pd-field' },
                        el('label', { for: 'pdDefaultFieldSel' }, 'Field to add a default value for'),
                        addDefaultSel),
                    el('div', { class: 'pd-field' },
                        el('label', { for: 'pdDefaultValInput' }, 'Default value'),
                        addDefaultVal),
                    addDefaultBtn) : null,
                el('div', { class: 'pd-defaults-list' }, defaultEntries.length ? defaultEntries.map(([fname, vals]) => {
                    const valInput = el('input', { type: 'text', placeholder: 'Add value…', disabled: ro, 'aria-label': `Add default value for ${fname}` });
                    const addVal = () => {
                        const v = valInput.value.trim();
                        if (!v) return;
                        if (!vals.includes(v)) vals.push(v);
                        valInput.value = '';
                        markDirty();
                        renderWorkspace();
                    };
                    valInput.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); addVal(); } });
                    return el('div', { class: 'pd-default-row' },
                        el('div', { class: 'pd-default-head' },
                            el('span', { class: 'pd-default-key' }, fname),
                            el('span', { class: 'spacer' }),
                            !ro ? el('button', {
                                type: 'button', class: 'pd-mini-btn is-danger', title: 'Remove field defaults', 'aria-label': `Remove defaults for ${fname}`,
                                onclick: () => { delete d.field_default_values[fname]; markDirty(); renderWorkspace(); },
                            }, icon('x', { size: 12 })) : null),
                        el('div', { class: 'pd-chips' },
                            vals.map((v, vi) => el('span', { class: 'pd-chip' }, v,
                                !ro ? el('button', {
                                    type: 'button', 'aria-label': `Remove ${v}`,
                                    onclick: () => {
                                        vals.splice(vi, 1);
                                        if (!vals.length) delete d.field_default_values[fname];
                                        markDirty();
                                        renderWorkspace();
                                    },
                                }, '×') : null)),
                            !ro ? el('span', { class: 'pd-chip-add' }, valInput,
                                el('button', { type: 'button', class: 'button-secondary button-small', onclick: addVal }, 'Add')) : null));
                }) : el('p', { class: 'help-text' }, 'No default values set.'))),

            // Review not required
            el('div', { class: 'pd-card' },
                el('div', { class: 'pd-card-head' },
                    el('h3', null, `Skip review (${d.review_not_required.length})`),
                    infoTip('Fields marked as not requiring manual review in VVGO-Editor.')),
                el('div', { class: 'pd-skip-grid' }, allFieldNames.map((fname) => {
                    const on = d.review_not_required.includes(fname);
                    return el('button', {
                        type: 'button', class: 'pd-skip-chip', 'aria-pressed': String(on), disabled: ro,
                        onclick: () => {
                            if (on) d.review_not_required = d.review_not_required.filter((x) => x !== fname);
                            else d.review_not_required.push(fname);
                            markDirty();
                            renderWorkspace();
                        },
                    }, fname);
                }))),

            // Custom flags
            el('div', { class: 'pd-card' },
                el('div', { class: 'pd-card-head' },
                    el('h3', null, `Custom flags (${d.custom_flags.length})`),
                    infoTip('Custom review flags shown in VVGO-Editor (e.g. "(Team) Flag for review").'),
                    el('span', { class: 'spacer' }),
                    !ro ? el('button', {
                        type: 'button', class: 'button-secondary button-small',
                        onclick: () => { d.custom_flags.push(''); markDirty(); renderWorkspace(); },
                    }, icon('plus', { size: 13 }), 'Add flag') : null),
                el('div', { class: 'pd-flag-list' }, d.custom_flags.length ? d.custom_flags.map((flag, fi) => {
                    const inp = el('input', {
                        type: 'text', value: flag, readOnly: ro, placeholder: '(Tag) Flag description', 'aria-label': `Custom flag ${fi + 1}`,
                        oninput: () => { d.custom_flags[fi] = inp.value; markDirty(); },
                    });
                    return el('div', { class: 'pd-flag-row' }, inp,
                        !ro ? el('button', {
                            type: 'button', class: 'pd-mini-btn is-danger', title: 'Remove flag', 'aria-label': 'Remove flag',
                            onclick: () => { d.custom_flags.splice(fi, 1); markDirty(); renderWorkspace(); },
                        }, icon('x', { size: 13 })) : null);
                }) : el('p', { class: 'help-text' }, 'No custom flags.'))),
        ];
    }

    function renderEditMode() {
        const check = validateDraft(draft, getPrompts());
        const badSections = new Set(check.errors.map((e) => e.section));
        const counts = {
            info: null,
            prompt: String(draft.data.rules.length),
            mapping: `${draft.data.mapping.length}/${MAX_CATS}`,
            checklist: String(draft.data.checklist.length),
            editor: String(Object.keys(draft.data.field_default_values).length + draft.data.review_not_required.length + draft.data.custom_flags.length),
        };

        const stepBar = el('div', { class: 'pd-steps', role: 'tablist', 'aria-label': 'Prompt sections' },
            STEPS.map((s) => el('button', {
                type: 'button', role: 'tab', class: ['pd-step-btn', badSections.has(s.id) ? 'has-issue' : ''],
                'aria-selected': String(activeStep === s.id),
                onclick: () => { activeStep = s.id; renderWorkspace(); },
            }, `${s.num}. ${s.label}`, counts[s.id] ? el('span', { class: 'pd-step-count' }, `(${counts[s.id]})`) : null)));

        let body;
        if (activeStep === 'info') body = renderInfoStep();
        else if (activeStep === 'prompt') body = renderPromptStep();
        else if (activeStep === 'mapping') body = renderMappingStep();
        else if (activeStep === 'checklist') body = renderChecklistStep();
        else body = renderEditorStep();

        replace(workspace, stepBar, body);
    }

    // ------------------------------------------------------------ compare / diff mode
    function renderCompareMode() {
        const ro = isReadOnly();
        const activeVal = draft ? `${draft.source}/${draft.filename}` : '';
        const cmpVal = compareRef ? `${compareRef.source}/${compareRef.name}` : '';
        const cmpSel = promptSelectElement(cmpVal, (val) => {
            const [src, ...rest] = val.split('/');
            loadCompare(src, rest.join('/'));
        }, 'Compare prompt');
        const activeSel = buildActiveSelect(activeVal);

        const diffOnlyBox = el('input', {
            type: 'checkbox', id: 'pdDiffOnly', checked: diffOnly,
            onchange: () => { diffOnly = diffOnlyBox.checked; renderWorkspace(); },
        });

        function buildColumnHeader(badgeEl) {
            return el('div', { class: 'pd-diff-header-grid' },
                el('div', { class: 'pd-diff-col-meta' },
                    badgeEl,
                    el('label', { class: 'check-row', for: 'pdDiffOnly' }, diffOnlyBox, 'Differences only')),
                el('div', { class: 'pd-diff-col-picker' },
                    el('label', null, 'Compare (read-only)'),
                    cmpSel),
                el('div', { class: 'pd-diff-col-picker' },
                    el('label', null, ro ? 'Active (read-only)' : 'Active (editable)'),
                    activeSel));
        }

        if (!compareData) {
            replace(workspace,
                buildColumnHeader(badge('Loading…', 'neutral')),
                stateBox('loading', 'Loading comparison prompt…'));
            return;
        }

        const L = compareData;
        const R = draft.data;
        let diffCount = 0;

        function textRow(label, leftVal, rightVal, { multiline = false, readonly = false, onChange, onCopy } = {}) {
            const cellRo = ro || readonly;
            const changed = (leftVal || '').trim() !== (rightVal || '').trim();
            if (changed) diffCount += 1;
            if (diffOnly && !changed) return null;

            const ctrl = multiline
                ? el('textarea', { rows: String(Math.min(10, Math.max(2, Math.ceil((rightVal || '').length / 75)))), readOnly: cellRo })
                : el('input', { type: 'text', value: rightVal || '', readOnly: cellRo });
            if (multiline) ctrl.value = rightVal || '';
            if (!cellRo && onChange) ctrl.addEventListener('input', () => { onChange(ctrl.value); markDirty(); });

            return el('div', { class: ['pd-diff-row', changed ? 'is-changed' : ''] },
                el('div', { class: 'pd-diff-key' }, label, changed ? badge('Changed', 'warn') : null),
                el('div', { class: 'pd-diff-left' }, changed ? wordDiffNodes(leftVal, rightVal) : (leftVal || '—')),
                el('div', { class: 'pd-diff-right' },
                    ctrl,
                    changed && !cellRo && onCopy ? el('div', { class: 'pd-diff-right-actions' },
                        el('button', { type: 'button', class: 'link-button', onclick: () => { onCopy(); markDirty(); renderWorkspace(); } }, '← Use left')) : null));
        }

        // Section 1: Info
        const infoRows = [
            textRow('prompt_name', L.prompt_name, R.prompt_name, { onChange: (v) => { R.prompt_name = v; }, onCopy: () => { R.prompt_name = L.prompt_name; } }),
            textRow('prompt_version', L.prompt_version, R.prompt_version, { onChange: (v) => { R.prompt_version = v; }, onCopy: () => { R.prompt_version = L.prompt_version; } }),
            textRow('prompt_author', L.prompt_author, R.prompt_author, { onChange: (v) => { R.prompt_author = v; }, onCopy: () => { R.prompt_author = L.prompt_author; } }),
            textRow('institution', L.prompt_author_institution, R.prompt_author_institution, { onChange: (v) => { R.prompt_author_institution = v; }, onCopy: () => { R.prompt_author_institution = L.prompt_author_institution; } }),
            textRow('LLM', L.LLM, R.LLM, { onChange: (v) => { R.LLM = v; }, onCopy: () => { R.LLM = L.LLM; } }),
            textRow('description', L.prompt_description, R.prompt_description, { multiline: true, onChange: (v) => { R.prompt_description = v; }, onCopy: () => { R.prompt_description = L.prompt_description; } }),
        ].filter(Boolean);

        // Section 2: Instructions & Rules
        const promptRows = [
            textRow('instructions', L.instructions, R.instructions, { multiline: true, onChange: (v) => { R.instructions = v; }, onCopy: () => { R.instructions = L.instructions; } }),
            textRow('formatting', CANONICAL_FORMATTING, CANONICAL_FORMATTING, { multiline: true, readonly: true }),
        ].filter(Boolean);

        const leftMap = new Map(L.rules.map((r) => [r.name, r]));
        const rightMap = new Map(R.rules.map((r) => [r.name, r]));
        const unionKeys = [...new Set([...R.rules.map((r) => r.name), ...L.rules.map((r) => r.name)])];
        const fieldRows = unionKeys.map((key) => {
            const lr = leftMap.get(key);
            const rr = rightMap.get(key);
            const status = !lr ? 'added' : (!rr ? 'missing' : ((lr.rule || '').trim() !== (rr.rule || '').trim() ? 'changed' : 'same'));
            if (status !== 'same') diffCount += 1;
            if (diffOnly && status === 'same') return null;

            let rightCell;
            if (rr) {
                const ta = el('textarea', { rows: '2', readOnly: ro });
                ta.value = rr.rule;
                ta.addEventListener('input', () => { rr.rule = ta.value; markDirty(); });
                rightCell = el('div', { class: 'pd-diff-right' },
                    ta,
                    status === 'changed' && !ro ? el('div', { class: 'pd-diff-right-actions' },
                        el('button', { type: 'button', class: 'link-button', onclick: () => { rr.rule = lr.rule; markDirty(); renderWorkspace(); } }, '← Use left')) : null);
            } else {
                rightCell = el('div', { class: 'pd-diff-right' },
                    el('span', { class: 'help-text' }, 'Not in active prompt'),
                    !ro ? el('div', { class: 'pd-diff-right-actions' },
                        el('button', {
                            type: 'button', class: 'button-secondary button-small',
                            onclick: () => {
                                R.rules.push({ name: key, rule: lr.rule, _mappedKey: key });
                                const leftCat = categoryOf(L, key);
                                const hasCat = R.mapping.some((m) => m.category === leftCat);
                                moveFieldToCategory(R, key, hasCat ? leftCat : 'COLLECTING');
                                markDirty();
                                renderWorkspace();
                            },
                        }, icon('plus', { size: 12 }), 'Add to active')) : null);
            }

            const statusBadge = status === 'added' ? badge('Added', 'good')
                : (status === 'missing' ? badge('Only in compare', 'bad')
                    : (status === 'changed' ? badge('Changed', 'warn') : null));

            return el('div', { class: ['pd-diff-row', status !== 'same' ? `is-${status}` : ''] },
                el('div', { class: 'pd-diff-key' }, key, statusBadge),
                el('div', { class: 'pd-diff-left' }, lr ? (status === 'changed' ? wordDiffNodes(lr.rule, rr.rule) : (lr.rule || '—')) : '—'),
                rightCell);
        }).filter(Boolean);

        // Section 3: Mapping
        const allCats = [...new Set([...R.mapping.map((m) => m.category), ...L.mapping.map((m) => m.category)])];
        const mapRows = allCats.map((cat) => {
            const lm = (L.mapping.find((m) => m.category === cat) || {}).fields || [];
            const rm = (R.mapping.find((m) => m.category === cat) || {}).fields || [];
            const lText = lm.join(', ');
            const rText = rm.join(', ');
            const changed = lText !== rText;
            if (changed) diffCount += 1;
            if (diffOnly && !changed) return null;
            return el('div', { class: ['pd-diff-row', changed ? 'is-changed' : ''] },
                el('div', { class: 'pd-diff-key' }, cat, changed ? badge('Changed', 'warn') : null),
                el('div', { class: 'pd-diff-left' }, changed ? wordDiffNodes(lText, rText) : (lText || '—')),
                el('div', { class: 'pd-diff-left' }, rText || '—'));
        }).filter(Boolean);

        // Section 4 & 5: Checklist & Editor options summary rows
        const lCheck = L.checklist.map((c, i) => formatChecklistItem(c, i)).filter(Boolean).join('\n');
        const rCheck = R.checklist.map((c, i) => formatChecklistItem(c, i)).filter(Boolean).join('\n');
        const lSkip = L.review_not_required.join(', ');
        const rSkip = R.review_not_required.join(', ');
        const lFlags = L.custom_flags.join('\n');
        const rFlags = R.custom_flags.join('\n');
        const lDefs = Object.entries(L.field_default_values).map(([k, v]) => `${k}: ${v.join(', ')}`).join('\n');
        const rDefs = Object.entries(R.field_default_values).map(([k, v]) => `${k}: ${v.join(', ')}`).join('\n');

        const editorRows = [
            textRow('checklist', lCheck, rCheck, {
                multiline: true,
                onChange: (v) => { R.checklist = v.split('\n').map((l) => l.trim()).filter(Boolean).map(parseChecklistItem); },
                onCopy: () => { R.checklist = structuredClone(L.checklist); },
            }),
            (() => {
                const changed = lDefs !== rDefs;
                if (changed) diffCount += 1;
                if (diffOnly && !changed) return null;
                return el('div', { class: ['pd-diff-row', changed ? 'is-changed' : ''] },
                    el('div', { class: 'pd-diff-key' }, 'defaults', changed ? badge('Changed', 'warn') : null),
                    el('div', { class: 'pd-diff-left' }, lDefs || '—'),
                    el('div', { class: 'pd-diff-right' },
                        el('div', { class: 'pd-diff-left' }, rDefs || '—'),
                        changed && !ro ? el('div', { class: 'pd-diff-right-actions' },
                            el('button', {
                                type: 'button', class: 'link-button',
                                onclick: () => {
                                    const validKeys = new Set(R.rules.map((r) => r.name));
                                    R.field_default_values = {};
                                    for (const [k, v] of Object.entries(L.field_default_values)) {
                                        if (validKeys.has(k)) R.field_default_values[k] = [...v];
                                    }
                                    markDirty();
                                    renderWorkspace();
                                },
                            }, '← Copy matching defaults')) : null));
            })(),
            (() => {
                const changed = lSkip !== rSkip;
                if (changed) diffCount += 1;
                if (diffOnly && !changed) return null;
                return el('div', { class: ['pd-diff-row', changed ? 'is-changed' : ''] },
                    el('div', { class: 'pd-diff-key' }, 'skip review', changed ? badge('Changed', 'warn') : null),
                    el('div', { class: 'pd-diff-left' }, lSkip || '—'),
                    el('div', { class: 'pd-diff-right' },
                        el('div', { class: 'pd-diff-left' }, rSkip || '—'),
                        changed && !ro ? el('div', { class: 'pd-diff-right-actions' },
                            el('button', {
                                type: 'button', class: 'link-button',
                                onclick: () => {
                                    const validKeys = new Set(R.rules.map((r) => r.name));
                                    R.review_not_required = L.review_not_required.filter((k) => validKeys.has(k));
                                    markDirty();
                                    renderWorkspace();
                                },
                            }, '← Copy matching fields')) : null));
            })(),
            textRow('custom_flags', lFlags, rFlags, {
                multiline: true,
                onChange: (v) => { R.custom_flags = v.split('\n').map((l) => l.trim()).filter(Boolean); },
                onCopy: () => { R.custom_flags = [...L.custom_flags]; },
            }),
        ].filter(Boolean);

        function diffSection(title, rows) {
            if (!rows.length) return null;
            return el('section', { class: 'pd-diff-section' },
                el('div', { class: 'pd-diff-section-head' }, title),
                rows);
        }

        const sections = [
            diffSection('1. Info', infoRows),
            diffSection('2. Instructions', promptRows),
            diffSection('2. Fields', fieldRows),
            diffSection('3. Mapping', mapRows),
            diffSection('4–5. Checklist & Editor', editorRows),
        ].filter(Boolean);

        replace(workspace,
            buildColumnHeader(badge(diffCount ? `${diffCount} difference${diffCount === 1 ? '' : 's'}` : 'Identical', diffCount ? 'warn' : 'good')),
            sections.length ? sections : stateBox('empty', 'Prompts are identical', null, { compact: true }));
    }

    function renderWorkspace() {
        if (!draft) {
            replace(workspace, stateBox('empty', 'Select a prompt or click New to start.'));
            return;
        }
        if (mode === 'compare') renderCompareMode();
        else renderEditMode();
    }

    function renderAll() {
        renderChrome();
        renderWorkspace();
    }

    async function ensureLoaded() {
        const all = getPrompts();
        if (!draft && all.length) {
            const preferred = all.find((p) => p.source === 'user')
                || all.find((p) => p.name === 'SLTPvM_full.yaml')
                || all[0];
            await loadActive(preferred.source, preferred.name);
        } else {
            renderAll();
        }
    }

    async function openPrompt(source, name, { copy = false } = {}) {
        mode = 'edit';
        await loadActive(source, name);
        if (copy && draft) {
            openCopyModal({ baseData: draft.data, baseStem: stem(draft.filename) });
        }
    }

    return {
        refresh: ensureLoaded,
        openPrompt,
    };
}
