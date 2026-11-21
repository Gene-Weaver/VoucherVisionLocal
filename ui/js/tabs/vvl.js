// VVL tab: choose models, enrichment and a prompt, process images, view results.
// Modeled on the VoucherVisionGO page; model lists come from GET /v1/models.

import { api, describeError, inlineError, toast, toastError } from '../api.js';
import {
    el, icon, infoTip, clear, replace, stateBox, prefs, openModal, copyText, downloadBlob, downloadText,
    base64ToBlob, externalLink, fmtClock,
} from '../dom.js';
import { on, emit, feature, featureNote, loadSettings, updateCachedSettings, gotoTab, plainReason } from '../state.js';
import { tabStrip, modeToggle } from '../widgets.js';
import { parseCsv, toCsv } from '../lib/csv.js';
import { makeZip } from '../lib/zip.js';
import { createMap } from '../lib/leaflet-map.js';
import { mountGenSettings } from './gen_settings.js';

// A local model can take minutes (the first request also loads it), and a PDF
// has many pages, so processing requests are never cut off by a timer here.
const NO_TIMEOUT = 0;
const ENGINE_POLL_MS = 1000;
const HISTORY_LIMIT = 50;
const MAX_OCR_ENGINES = 4;
const URL_NAMES_PER_REQUEST = 5000; // the most URLs POST /v1/url-names takes at once
const IMAGE_NAME_RE = /\.(jpe?g|png|tiff?|webp|bmp|gif|heic|heif|jp2)$/i;
const BASE64_RE = /^[A-Za-z0-9+/=\s]+$/;
const EXAMPLE_URLS = [
    ['Jouvea pilosa', 'https://quod.lib.umich.edu/cgi/i/image/api/image/herb00ic:1122841:MICH-V-1122841/full/res:0/0/native.jpg'],
    ['Quercus oxyodon', 'https://medialib.naturalis.nl/file/id/L.3800382/format/large'],
    ['Liriodendron tulipifera', 'https://quod.lib.umich.edu/cgi/i/image/api/image/herb00ic:1500329:MICH-V-1500329/full/res:0/0/native.jpg'],
];

function step(number, title, ...body) {
    const actions = el('div', { class: 'step-header-actions' });
    const node = el('section', { class: 'step-container', 'aria-label': title },
        el('div', { class: 'step-header' }, el('span', { class: 'step-number', 'aria-hidden': 'true' }, number), el('h3', null, title), actions),
        body);
    return { node, actions };
}

function isHttpUrl(text) {
    try {
        const u = new URL(text);
        return u.protocol === 'https:' || u.protocol === 'http:';
    } catch (_) {
        return false;
    }
}

function stem(name) {
    const base = String(name || 'result').split(/[\\/]/).pop();
    const dot = base.lastIndexOf('.');
    return (dot > 0 ? base.slice(0, dot) : base) || 'result';
}

/**
 * File names for image URLs, as a Map of url -> name. The names come from the
 * server's URL name parser (POST /v1/url-names), the same one the downloader
 * uses; a name is never worked out from a URL in this page. One request covers
 * a whole list.
 */
async function namesForUrls(urls) {
    const names = new Map();
    for (let i = 0; i < urls.length; i += URL_NAMES_PER_REQUEST) {
        const res = await api.post('/v1/url-names', { urls: urls.slice(i, i + URL_NAMES_PER_REQUEST) }, { timeoutMs: 30000 });
        for (const n of (res && Array.isArray(res.names) ? res.names : [])) {
            if (n && typeof n.url === 'string' && typeof n.name === 'string' && n.name) names.set(n.url, n.name);
        }
    }
    return names;
}

/** Entry names for a zip file: a name used twice gets _2, _3, ... so nothing is overwritten. */
function uniqueNames(names) {
    const seen = new Map();
    return names.map((name) => {
        const count = (seen.get(name.toLowerCase()) || 0) + 1;
        seen.set(name.toLowerCase(), count);
        if (count === 1) return name;
        const dot = name.lastIndexOf('.');
        return dot > 0 ? `${name.slice(0, dot)}_${count}${name.slice(dot)}` : `${name}_${count}`;
    });
}

function hasRecord(data) {
    return Boolean(data && data.formatted_json && typeof data.formatted_json === 'object' && Object.keys(data.formatted_json).length);
}

function coordsOf(data) {
    if (!hasRecord(data)) return null;
    const lat = parseFloat(data.formatted_json.decimalLatitude);
    const lon = parseFloat(data.formatted_json.decimalLongitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
    return { lat, lon };
}

export async function mount(panel) {
    // ------------------------------------------------------------ state
    let selectable = { ocr: [], parse: [] };
    let apiProviders = [];          // GET /v1/models -> api.providers (for the [API] groups)
    let installState = new Map();   // local model id -> install_state (from GET /v1/models local.families)
    let selection = { ocr: [], parse: null };
    let prompts = [];
    let promptName = prefs.get('prompt', null);
    let busy = false;
    let engineParallel = null;   // how many images the local engines work on at once
    let enginePlan = null;       // GET /v1/engine -> plan (mode, staged, concurrency)
    let localLabels = new Map(); // local model id -> label
    const sliders = [];          // batch concurrency controls
    const history = [];
    const runButtons = [];

    function setBusy(value) {
        busy = value;
        for (const b of runButtons) b.disabled = value;
    }
    // The desktop shell asks before restarting for an update while images are being processed.
    if (window.vvlUi) window.vvlUi.batchRunning = () => busy;

    function runButton(label, handler) {
        const b = el('button', { type: 'button', class: 'button', onclick: handler }, label);
        runButtons.push(b);
        return b;
    }

    // ------------------------------------------------------------ step 1: models
    const pickers = {
        ocr: { title: 'OCR', tip: 'Reads text from specimen images (up to 4 models).', input: 'checkbox', list: el('div', { class: 'model-list' }), thinking: null },
        parse: { title: 'Parsing', tip: 'Extracts structured fields from the OCR text.', input: 'radio', list: el('div', { class: 'model-list' }), thinking: null },
    };
    for (const [role, p] of Object.entries(pickers)) {
        const select = el('select', { id: `${role}ThinkingLevel`, onchange: () => prefs.set(`thinking.${role}`, select.value) });
        p.thinking = { select, node: el('div', { class: 'thinking-level-control', hidden: true }, el('label', { for: select.id }, 'Thinking'), select) };
        // The models chosen now, shown at the right of the card title (easy to spot two OCR models).
        p.chosen = el('span', { class: 'picker-chosen', 'aria-live': 'polite' });
        p.card = el('div', { class: 'settings-card' },
            el('h3', { class: 'picker-title' },
                el('span', { class: 'picker-title-main' }, p.title, infoTip(p.tip, 'start')),
                p.chosen),
            p.thinking.node, p.list);
    }
    // Per-step model settings (reasoning, output length, ...) under each card's model list.
    const genSettings = mountGenSettings({ ocr: pickers.ocr.card, parse: pickers.parse.card });
    const ocrOnly = modeToggle({ id: 'ocrOnly', label: 'OCR only', tip: 'Transcribe text only; skip field parsing.' });
    const notebookMode = modeToggle({ id: 'notebookMode', label: 'Notebook mode', tip: 'Output Markdown directly without cropping labels.' });
    const skipCollage = modeToggle({ id: 'skipLabelCollage', label: 'Skip label crop', tip: 'Send the full image instead of cropping labels first.' });
    const modelWarning = el('div', { class: 'model-warning', 'aria-live': 'polite' });
    const step1 = step(1, 'Models',
        el('div', { class: 'ocr-llm-row' }, pickers.ocr.card, pickers.parse.card),
        modelWarning,
        el('div', { class: 'mode-toggles' }, ocrOnly.node, notebookMode.node, skipCollage.node));
    // Show the two model lists at full height (no scrolling inside them), or at the usual fixed height.
    const expandLists = el('button', { type: 'button', class: 'button-secondary button-small list-expand', 'aria-pressed': 'false' }, icon('chevronDown', { size: 14 }));
    function setListsExpanded(on) {
        step1.node.classList.toggle('lists-expanded', on);
        expandLists.setAttribute('aria-pressed', String(on));
        const label = on ? 'Collapse lists' : 'Expand lists';
        expandLists.title = label;
        expandLists.setAttribute('aria-label', label);
    }
    expandLists.addEventListener('click', () => {
        const on = expandLists.getAttribute('aria-pressed') !== 'true';
        prefs.set('modelListsExpanded', on);
        setListsExpanded(on);
    });
    setListsExpanded(Boolean(prefs.get('modelListsExpanded', false)));
    step1.actions.appendChild(expandLists);
    step1.actions.appendChild(el('button', { type: 'button', class: 'button-secondary button-small', onclick: () => gotoTab('models') },
        icon('download', { size: 14 }), 'Get models'));

    const isLocal = (id) => typeof id === 'string' && id.startsWith('local:');
    const needsParse = () => !ocrOnly.input.checked && !notebookMode.input.checked;

    /** The local models this run would use: {ocr: [ids], parse: id | null, all: [distinct ids]}. */
    function localModelsInUse() {
        const ocr = selection.ocr.filter(isLocal);
        const parse = needsParse() && isLocal(selection.parse) ? selection.parse : null;
        return { ocr, parse, all: [...new Set([...ocr, ...(parse ? [parse] : [])])] };
    }

    function updateModelWarning() {
        const used = localModelsInUse();
        // Only worth saying when the chosen models cannot be loaded together.
        const takeTurns = used.all.length > 1 && enginePlan && enginePlan.mode === 'exclusive';
        replace(modelWarning, takeTurns
            ? el('div', { class: 'notice notice-warn' }, icon('alert', { size: 18 }),
                el('span', null, 'Models will load one at a time.',
                    infoTip('Not enough GPU memory to keep both loaded. Batches run all OCR first, then parsing.')))
            : null);
    }

    /** How many requests OpenRouter runs at once (1 to 4), when a chosen model is on OpenRouter; else null. */
    function openRouterLimit() {
        const ids = [...selection.ocr, ...(needsParse() && selection.parse ? [selection.parse] : [])];
        if (!ids.some((id) => typeof id === 'string' && id.startsWith('api:openrouter:'))) return null;
        const provider = apiProviders.find((x) => x.id === 'openrouter');
        const n = provider && Number(provider.concurrency);
        return Number.isInteger(n) && n > 0 ? n : 4;
    }

    /** Batch runs should not start more requests at once than the local engine has slots, or than OpenRouter is set to run. */
    function applyConcurrencyLimit() {
        const local = localModelsInUse().all.length > 0;
        const hosted = openRouterLimit();
        const limits = [];
        if (local && Number.isFinite(engineParallel)) limits.push(engineParallel);
        if (hosted) limits.push(hosted);
        const cap = limits.length ? Math.min(...limits) : null;
        for (const item of sliders) {
            const limited = cap !== null;
            if (limited && !item.touched && Number(item.slider.value) !== cap) {
                item.slider.value = String(Math.max(1, Math.min(Number(item.slider.max), cap)));
                item.value.textContent = item.slider.value;
            } else if (!limited && !item.touched) {
                item.slider.value = String(Math.min(8, prefs.get('concurrency', 4)));
                item.value.textContent = item.slider.value;
            }
            const over = limited && Number(item.slider.value) > cap;
            item.note.hidden = !limited;
            if (limited) {
                const who = hosted && cap === hosted ? `OpenRouter limit: ${cap}` : `Engine limit: ${cap}`;
                replace(item.note, who, infoTip(over ? 'Extra requests wait automatically.' : 'Maximum concurrent requests for the current model.'));
            } else {
                clear(item.note);
            }
        }
    }

    function selectionChanged() {
        updateModelWarning();
        applyConcurrencyLimit();
    }
    for (const toggle of [ocrOnly, notebookMode]) toggle.input.addEventListener('change', selectionChanged);

    async function loadEngineInfo() {
        try {
            const engine = await api.get('/v1/engine', { timeoutMs: 30000 });
            enginePlan = engine && engine.plan ? engine.plan : null;
            const parallel = enginePlan && Number(enginePlan.concurrency);
            engineParallel = Number.isInteger(parallel) && parallel > 0 ? parallel : null;
        } catch (_) {
            enginePlan = null;
            engineParallel = null; // unknown: leave the batch controls alone
        }
        updateModelWarning();
        applyConcurrencyLimit();
        return enginePlan;
    }

    function entry(role, id) {
        return selectable[role].find((e) => e.model_id === id) || null;
    }

    function selectedLevels(role) {
        const ids = role === 'ocr' ? selection.ocr : [selection.parse];
        const sets = ids.map((id) => entry(role, id)).filter(Boolean)
            .map((e) => (e.capabilities && Array.isArray(e.capabilities.thinking_levels) ? e.capabilities.thinking_levels : []))
            .filter((levels) => levels.length);
        if (!sets.length) return [];
        return sets[0].filter((level) => sets.every((s) => s.includes(level)));
    }

    function updateThinking(role) {
        const { select, node } = pickers[role].thinking;
        const levels = selectedLevels(role);
        node.hidden = levels.length === 0;
        if (!levels.length) return;
        const wanted = levels.includes(select.value) ? select.value : prefs.get(`thinking.${role}`, null);
        replace(select, levels.map((level) => el('option', { value: level }, level.charAt(0).toUpperCase() + level.slice(1))));
        select.value = levels.includes(wanted) ? wanted : levels[levels.length - 1];
    }

    // One save at a time, always of the latest choice: two saves on their way
    // together could reach the server in either order and leave the older one saved.
    let saving = false;
    let saveAgain = false;
    async function saveSelection() {
        if (!selection.ocr.length || !selection.parse) return;
        if (saving) {
            saveAgain = true;
            return;
        }
        saving = true;
        try {
            do {
                saveAgain = false;
                try {
                    const res = await api.put('/v1/selection', { ocr: selection.ocr, parse: selection.parse });
                    // The answer must not undo a newer choice made while it was on its way.
                    if (!saveAgain && res && res.selection) selection = { ocr: [...res.selection.ocr], parse: res.selection.parse };
                } catch (err) {
                    if (!saveAgain) toastError(err, 'The model choice was not saved');
                }
            } while (saveAgain && selection.ocr.length && selection.parse);
        } finally {
            saving = false;
        }
        emit('selection-changed', { ocr: [...selection.ocr], parse: selection.parse });
        loadEngineInfo();   // the engine layout follows the saved selection
        genSettings.refresh();
    }

    // The demo engine stays in the code (and the API) but is not offered in the pickers.
    const SHOW_DEMO = false;
    // Groups for API providers that cannot run yet (kept in the code, not shown).
    const SHOW_UNAVAILABLE_API = false;
    const familyOpen = new Map();   // `${role}:${family}` -> open, kept across re-renders

    /** Selectable entries the pickers show: local models only once downloaded; the demo engine only with SHOW_DEMO. */
    function visible(role) {
        return selectable[role].filter((e) => (SHOW_DEMO || e.group !== 'Demo')
            && (e.group !== 'Local' || installState.get(e.model_id) === 'installed'));
    }

    /** Entries grouped by model family: the primary family first, then the list order; API providers last. */
    function familiesFor(role) {
        const families = [];
        for (const item of visible(role)) {
            const id = item.family_id || item.provider_label || item.group;
            let f = families.find((x) => x.id === id);
            if (!f) {
                families.push(f = {
                    id, label: item.family_label || item.provider_label || item.group, api: item.group === 'API',
                    demo: item.group === 'Demo', color: item.family_color || null, primary: Boolean(item.family_primary), items: [],
                });
            }
            f.items.push(item);
        }
        // API providers whose model list is not loaded yet still get a (closed) group.
        for (const provider of SHOW_UNAVAILABLE_API ? apiProviders : []) {
            if (!families.some((f) => f.id === provider.id)) {
                families.push({ id: provider.id, label: provider.label, api: true, demo: false, color: provider.brand_color || null,
                    primary: false, items: [], configured: Boolean(provider.configured),
                    reason: plainReason(provider.unavailable_reason) || null });
            }
        }
        const rank = (f) => (f.demo ? 0 : f.primary ? 1 : f.api ? 3 : 2);
        // A family is shown only when it has a model to offer: a downloaded local model,
        // or an API model that can run.
        const offered = families.filter((f) => f.items.length && (!f.api || SHOW_UNAVAILABLE_API || f.items.some((i) => i.runnable)));
        return offered.map((f, i) => ({ f, i })).sort((x, y) => rank(x.f) - rank(y.f) || x.i - y.i).map((x) => x.f);
    }

    /** Short names of the chosen models, for the card title. An OCR name can be clicked to deselect that model. */
    function renderChosen(role) {
        const ids = role === 'ocr' ? selection.ocr : (selection.parse ? [selection.parse] : []);
        const short = (id) => {
            const item = entry(role, id);
            const label = item ? item.label : id.split(':').pop();
            return label.includes(': ') ? label.split(': ').slice(1).join(': ') : label;   // drop an owner prefix
        };
        const p = pickers[role];
        p.chosen.classList.toggle('is-many', ids.length > 1);
        p.chosen.title = '';
        if (!ids.length) {
            replace(p.chosen, 'None');
            return;
        }
        // Parsing has exactly one model, so there is nothing to deselect there.
        const removable = role === 'ocr' && ids.length > 1;
        replace(p.chosen, ids.map((id, i) => [i ? ', ' : null, removable
            ? el('button', { type: 'button', class: 'picker-chosen-name', title: `Remove ${short(id)}`, 'aria-label': `Remove ${short(id)}`,
                onclick: () => {
                    selection.ocr = selection.ocr.filter((x) => x !== id);
                    selectionChanged();
                    saveSelection();
                    renderPicker('ocr');
                } }, short(id))
            : el('span', role === 'ocr' ? { title: 'At least one OCR model must stay selected.' } : null, short(id))]));
    }

    function renderPicker(role) {
        const p = pickers[role];
        renderChosen(role);
        clear(p.list);
        const families = familiesFor(role);
        if (!families.length) {
            p.list.appendChild(stateBox('empty', 'No models yet', 'Click Get models to add one.', { compact: true }));
            return;
        }
        const isChecked = (item) => (role === 'ocr' ? selection.ocr.includes(item.model_id) : selection.parse === item.model_id);
        for (const f of families) {
            const key = `${role}:${f.id}`;
            const open = familyOpen.has(key) ? familyOpen.get(key) : f.primary || f.demo;
            const reasons = new Set(f.items.filter((i) => !i.runnable).map((i) => plainReason(i.unavailable_reason) || 'Unavailable'));
            // One reason for every unavailable model in the family is said once, not on each line.
            const common = reasons.size === 1 ? [...reasons][0] : null;
            const ready = f.items.filter((i) => i.runnable);
            const chosen = f.items.filter((i) => isChecked(i) && i.runnable);
            const summaryNote = chosen.length
                ? chosen.map((i) => i.label).join(', ')
                : (ready.length ? `${ready.length} ready` : (f.reason || common || 'Unavailable'));
            const details = el('details', { class: ['model-family', f.api ? 'is-api' : ''], open,
                ontoggle: () => familyOpen.set(key, details.open) },
            el('summary', { class: 'model-family-summary' },
                el('span', { class: 'model-family-name' }, f.api ? `${f.label} · Cloud` : f.label),
                el('span', { class: ['model-family-note', chosen.length ? 'is-selected' : ''] }, summaryNote)));
            // Set through the style object: the page's security policy refuses style="" attributes.
            if (f.color && /^#[0-9a-fA-F]{6}$/.test(f.color)) details.style.setProperty('--family-color', f.color);
            const box = el('div', { class: 'model-family-body', role: 'group', 'aria-label': `${f.label} models` });
            if (!f.items.length) {
                // The summary line already gives the reason; say what the user can do instead.
                box.appendChild(el('div', { class: 'model-group-note' }, f.api && !f.configured
                    ? 'Add a key for this provider on the Keys tab to see its models here.'
                    : 'Its models will be listed here.'));
            }
            // Models that can run come first, so they are not buried under the ones that cannot.
            const ordered = [...ready, ...f.items.filter((i) => !i.runnable)];
            let noted = false;
            for (const item of ordered) {
                if (!item.runnable && common && !noted) {
                    noted = true;
                    box.appendChild(el('div', { class: 'model-group-note' }, common));
                }
                const input = el('input', {
                    type: p.input, name: role === 'ocr' ? 'engines' : 'llm_model', value: item.model_id,
                    checked: isChecked(item) && item.runnable, disabled: !item.runnable,
                    onchange: () => {
                        if (role === 'ocr') {
                            if (input.checked && p.list.querySelectorAll('input:checked').length > MAX_OCR_ENGINES) {
                                input.checked = false;
                                toast(`Choose at most ${MAX_OCR_ENGINES} OCR engines.`, 'warning');
                                return;
                            }
                            selection.ocr = [...p.list.querySelectorAll('input:checked')].map((n) => n.value);
                            if (!selection.ocr.length) toast('Choose at least one OCR engine.', 'warning');
                        } else {
                            selection.parse = input.value;
                        }
                        updateThinking(role);
                        selectionChanged();
                        saveSelection();
                        // Refresh the "Selected:" lines, keeping keyboard focus on this choice.
                        const value = input.value;
                        renderPicker(role);
                        const again = [...p.list.querySelectorAll('input')].find((n) => n.value === value);
                        if (again && document.activeElement !== again) again.focus({ preventScroll: true });
                    },
                });
                const reason = item.runnable ? null : (plainReason(item.unavailable_reason) || 'Not available');
                box.appendChild(el('label', { class: ['model-option', item.runnable ? '' : 'is-disabled'], title: reason },
                    input,
                    el('span', null, item.label),
                    reason && !common ? el('span', { class: 'model-option-reason' }, reason) : null));
            }
            details.appendChild(box);
            p.list.appendChild(details);
        }
        updateThinking(role);
    }

    async function loadModels() {
        for (const p of Object.values(pickers)) replace(p.list, stateBox('loading', 'Loading models…', null, { compact: true }));
        try {
            const data = await api.get('/v1/models');
            selectable = {
                ocr: (data.selectable && data.selectable.ocr) || [],
                parse: (data.selectable && data.selectable.parse) || [],
            };
            const sel = data.selection || {};
            const runnable = (role, id) => Boolean(entry(role, id) && entry(role, id).runnable);
            selection = {
                ocr: (Array.isArray(sel.ocr) ? sel.ocr : []).filter((id) => runnable('ocr', id)),
                parse: runnable('parse', sel.parse) ? sel.parse : null,
            };
            apiProviders = (data.api && Array.isArray(data.api.providers)) ? data.api.providers : [];
            installState = new Map(((data.local && data.local.families) || [])
                .flatMap((f) => (f.builds || []).map((b) => [b.model_id || `local:${b.id}`, b.install_state])));
            // With nothing usable saved, start from the first model that can run, primary family first.
            const firstRunnable = (role) => {
                const shown = familiesFor(role).flatMap((f) => f.items).filter((e) => e.runnable);
                return shown[0] ? shown[0].model_id : null;
            };
            if (!selection.ocr.length) selection.ocr = [firstRunnable('ocr')].filter(Boolean);
            if (!selection.parse) selection.parse = firstRunnable('parse');
            localLabels = new Map([...selectable.ocr, ...selectable.parse].filter((e) => isLocal(e.model_id)).map((e) => [e.model_id, e.label]));
            renderPicker('ocr');
            renderPicker('parse');
            selectionChanged();
            genSettings.refresh();
        } catch (err) {
            selectable = { ocr: [], parse: [] };
            selection = { ocr: [], parse: null };
            inlineError(pickers.ocr.list, err, { title: 'The model list could not be loaded', retry: loadModels, compact: true });
            clear(pickers.parse.list);
        }
    }

    // ------------------------------------------------------------ step 2: enrichment
    const wfoBox = el('input', { type: 'checkbox', id: 'includeWfo' });
    const elevBox = el('input', { type: 'checkbox', id: 'includeCop90' });
    const wfoNote = el('p', { class: 'option-note', id: 'includeWfoNote', hidden: true });
    const elevNote = el('p', { class: 'option-note', id: 'includeCop90Note', hidden: true });
    const wfoCard = el('div', { class: 'option-card' },
        el('h4', null, 'World Flora Online', infoTip('Validates species names against World Flora Online.')),
        el('label', { for: 'includeWfo' }, wfoBox, 'Include WFO validation'), wfoNote);
    const elevCard = el('div', { class: 'option-card' },
        el('h4', null, 'Copernicus GLO-90 Elevation', infoTip('Estimates elevation from coordinates using Copernicus GLO-90.')),
        el('label', { for: 'includeCop90' }, elevBox, 'Derive elevation from coordinates'), elevNote);
    const enrichStatus = el('div');
    const step2 = step(2, 'Enrichment', el('div', { class: 'options-row' }, wfoCard, elevCard), enrichStatus);
    let wfoSource = 'vvgo';

    function setupEnrichmentToggle(box, card, note, featureName) {
        const enabled = feature(featureName);
        box.disabled = !enabled;
        card.classList.toggle('is-disabled', !enabled);
        const text = enabled ? null : (featureNote(featureName) || 'Not available');
        note.hidden = !text;
        note.textContent = text || '';
        if (text) box.setAttribute('aria-describedby', note.id);
        if (!enabled) box.checked = false;
        return enabled;
    }

    async function saveEnrichment(patch) {
        try {
            const saved = await api.put('/v1/settings', { enrichment: patch });
            if (saved) {
                const { warnings, ...settings } = saved;
                updateCachedSettings(settings);
            }
        } catch (err) {
            toastError(err, 'The enrichment setting was not saved');
        }
    }

    async function loadEnrichment() {
        clear(enrichStatus);
        const wfoOn = setupEnrichmentToggle(wfoBox, wfoCard, wfoNote, 'enrichment_wfo');
        const elevOn = setupEnrichmentToggle(elevBox, elevCard, elevNote, 'enrichment_elevation');
        if (elevOn) elevBox.checked = true;
        try {
            const settings = await loadSettings();
            const enr = (settings && settings.enrichment) || {};
            if (enr.wfo_source && enr.wfo_source !== 'off') wfoSource = enr.wfo_source;
            if (wfoOn) wfoBox.checked = Boolean(enr.wfo_source && enr.wfo_source !== 'off');
            if (elevOn) elevBox.checked = enr.elevation !== false;
        } catch (err) {
            inlineError(enrichStatus, err, { title: 'Saved enrichment settings could not be loaded', retry: loadEnrichment, compact: true });
        }
    }
    wfoBox.addEventListener('change', () => saveEnrichment({ wfo_source: wfoBox.checked ? wfoSource : 'off' }));
    elevBox.addEventListener('change', () => saveEnrichment({ elevation: elevBox.checked }));

    // ------------------------------------------------------------ step 3: prompt
    const promptCurrent = el('span', { class: 'prompt-current-name' }, '—');
    const promptDetails = el('button', { type: 'button', class: 'link-button', hidden: true }, 'Details');
    const promptButtons = el('div', { class: 'template-buttons-container', role: 'group', 'aria-label': 'Prompts' });
    const step3 = step(3, 'Prompt',
        el('div', { class: 'prompt-current' }, promptCurrent, promptDetails),
        promptButtons);
    step3.actions.appendChild(el('button', { type: 'button', class: 'button-secondary button-small', onclick: () => loadPrompts() }, icon('refresh', { size: 14 }), 'Refresh'));

    function selectedPrompt() {
        return prompts.find((p) => p.name === promptName && p.valid !== false) || null;
    }

    function renderPrompts() {
        clear(promptButtons);
        if (!prompts.length) {
            promptButtons.appendChild(stateBox('empty', 'No prompts found', null, { compact: true }));
        }
        for (const p of prompts) {
            const user = p.source === 'user';
            promptButtons.appendChild(el('button', {
                type: 'button',
                class: ['template-button', user ? 'user-generated' : '', p.name === promptName ? 'selected' : ''],
                'aria-pressed': String(p.name === promptName),
                disabled: p.valid === false,
                title: p.valid === false ? (p.error || 'Invalid prompt') : `${p.name}${user ? ' (custom)' : ''}`,
                onclick: () => setPrompt(p.name),
            }, stem(p.name)));
        }
        const sel = selectedPrompt();
        promptCurrent.textContent = sel ? stem(sel.name) : 'None';
        promptDetails.hidden = !sel;
    }

    function setPrompt(name) {
        promptName = name;
        prefs.set('prompt', name);
        renderPrompts();
    }
    promptDetails.addEventListener('click', () => {
        const sel = selectedPrompt();
        if (sel) gotoTab('prompts', `${sel.source}/${sel.name}`);
    });

    async function loadPrompts() {
        replace(promptButtons, stateBox('loading', 'Loading prompts…', null, { compact: true }));
        try {
            const data = await api.get('/v1/prompts');
            prompts = (data && data.prompts) || [];
            if (!selectedPrompt()) {
                const valid = prompts.filter((p) => p.valid !== false);
                const fallback = valid.find((p) => p.name === 'SLTPvM_geolocate.yaml') || valid[0];
                promptName = fallback ? fallback.name : null;
            }
            renderPrompts();
        } catch (err) {
            prompts = [];
            promptCurrent.textContent = '—';
            promptDetails.hidden = true;
            inlineError(promptButtons, err, { title: 'Prompts could not be loaded', retry: loadPrompts, compact: true });
        }
    }

    // ------------------------------------------------------------ request building
    /** Shared form fields as [name, value] pairs, or null (after telling the user why). */
    function requestFields() {
        if (!selection.ocr.length) {
            toast('Choose at least one OCR engine in step 1.', 'warning');
            return null;
        }
        const needsParse = !ocrOnly.input.checked && !notebookMode.input.checked;
        if (needsParse && !selection.parse) {
            toast('Choose a parsing model in step 1.', 'warning');
            return null;
        }
        const prompt = selectedPrompt();
        if (needsParse && !prompt) {
            toast('Choose a prompt template in step 3.', 'warning');
            return null;
        }
        const fields = selection.ocr.map((id) => ['engines', id]);
        if (selection.parse) fields.push(['llm_model', selection.parse]);
        if (prompt) fields.push(['prompt', prompt.name]);
        if (ocrOnly.input.checked) fields.push(['ocr_only', 'true']);
        if (notebookMode.input.checked) fields.push(['notebook_mode', 'true']);
        if (skipCollage.input.checked) fields.push(['skip_label_collage', 'true']);
        fields.push(['include_wfo', String(wfoBox.checked && !wfoBox.disabled)]);
        fields.push(['include_cop90', String(elevBox.checked && !elevBox.disabled)]);
        if (!pickers.ocr.thinking.node.hidden) fields.push(['ocr_thinking_level', pickers.ocr.thinking.select.value]);
        if (!pickers.parse.thinking.node.hidden) fields.push(['llm_thinking_level', pickers.parse.thinking.select.value]);
        return fields;
    }

    function formFor(fields, source) {
        const form = new FormData();
        for (const [k, v] of fields) form.append(k, v);
        if (source.file) form.append('file', source.file, source.file.name);
        else form.append('image_url', source.url);
        return form;
    }

    // ------------------------------------------------------------ step 5: results
    const historySelect = el('select', { id: 'resultHistorySelect', onchange: () => showResult(history[Number(historySelect.value)]) });
    const historyBar = el('div', { class: 'result-history-bar', hidden: true }, el('label', { for: 'resultHistorySelect' }, 'History:'), historySelect);
    const resultsArea = el('div', { class: 'results', 'aria-live': 'polite' },
        stateBox('empty', 'No results yet'));
    const coordStatus = el('div', { class: 'coord-status invalid' });
    const mapHolder = el('div', { class: 'map-holder' });
    const mapSection = el('div', { class: 'global-map-section', hidden: true }, coordStatus, mapHolder);
    const map = createMap(mapHolder);
    const step5 = step(5, 'Results', historyBar, resultsArea, mapSection);

    function fieldTable(obj) {
        const table = el('table', { class: 'field-table' });
        for (const [key, val] of Object.entries(obj)) {
            const empty = val === null || val === undefined || val === '';
            const text = empty ? '—' : (typeof val === 'object' ? JSON.stringify(val) : String(val));
            table.appendChild(el('tr', null, el('th', { scope: 'row' }, key), el('td', { class: empty ? 'field-empty' : null }, text)));
        }
        return table;
    }

    function section(title, body) {
        return el('div', { class: 'formatted-section' }, title ? el('h4', null, title) : null, body);
    }

    function formattedPanel(data) {
        const out = [];
        if (hasRecord(data)) out.push(section(null, fieldTable(data.formatted_json)));
        if (typeof data.formatted_md === 'string' && data.formatted_md.trim()) {
            out.push(section('Markdown', el('pre', { class: 'formatted-value' }, data.formatted_md)));
        }
        return out.length ? out : el('p', { class: 'help-text' }, 'None.');
    }

    function ocrPanel(data) {
        const text = typeof data.ocr === 'string' ? data.ocr : (data.ocr ? JSON.stringify(data.ocr, null, 2) : '');
        return text.trim() ? el('pre', { class: 'formatted-value' }, text) : el('p', { class: 'help-text' }, 'None.');
    }

    function enrichmentPanel(data) {
        const out = [];
        if (data.WFO_info && typeof data.WFO_info === 'object' && Object.keys(data.WFO_info).length) {
            out.push(section('World Flora Online', fieldTable(data.WFO_info)));
        }
        if (data.COP90_elevation_m !== undefined && data.COP90_elevation_m !== null && data.COP90_elevation_m !== '') {
            out.push(section('Elevation', el('p', null, el('strong', null, `${data.COP90_elevation_m} m`))));
        }
        return out.length ? out : el('p', { class: 'help-text' }, 'None.');
    }

    function usagePanel(data) {
        const rows = [];
        const num = (v) => (Number.isFinite(Number(v)) ? Number(v).toLocaleString('en-US') : '—');
        const cost = (v) => (Number(v) > 0 ? `$${Number(v).toFixed(6)}` : '—');
        if (data.ocr_info && typeof data.ocr_info === 'object') {
            for (const [model, info] of Object.entries(data.ocr_info)) {
                if (info && typeof info === 'object') rows.push(['OCR', model, info.tokens_in, info.tokens_out, info.thinking_tokens, info.total_cost]);
            }
        }
        const p = data.parsing_info;
        if (p && typeof p === 'object' && p.model) rows.push(['Parsing', p.model, p.input, p.output, p.thinking_tokens, p.total_cost]);
        if (!rows.length) return el('p', { class: 'help-text' }, 'None.');
        return el('div', { class: 'table-wrap' }, el('table', { class: 'data' },
            el('thead', null, el('tr', null, ['Stage', 'Model', 'In', 'Out', 'Thinking', 'Cost'].map((t) => el('th', { scope: 'col' }, t)))),
            el('tbody', null, rows.map((r) => el('tr', null, el('td', null, r[0]), el('td', null, r[1]), el('td', null, num(r[2])), el('td', null, num(r[3])), el('td', null, num(r[4])), el('td', null, cost(r[5])))))));
    }

    function collageOf(data) {
        const b64 = data.collage_info && data.collage_info.base64image_text_collage;
        if (typeof b64 !== 'string' || !b64 || !BASE64_RE.test(b64)) return null;
        const format = /^(jpeg|png|webp)$/.test(String(data.collage_image_format || '')) ? data.collage_image_format : 'jpeg';
        return { b64, type: `image/${format}`, ext: format === 'jpeg' ? 'jpg' : format };
    }

    function displayJson(data) {
        return JSON.stringify(data, (key, value) => (key === 'base64image_text_collage' && typeof value === 'string' && value.length > 200
            ? `<image data, ${value.length.toLocaleString('en-US')} characters, included in Download JSON>` : value), 2);
    }

    function zoomImage(src, title) {
        openModal({ title, className: 'modal-image', body: el('img', { src, alt: title }) });
    }

    function showResult(item) {
        if (!item) return;
        const { data, base } = item;
        const md = typeof data.formatted_md === 'string' ? data.formatted_md.trim() : '';
        const controls = el('div', { class: 'json-controls' },
            el('button', { type: 'button', class: 'button-secondary', onclick: async () => toast((await copyText(JSON.stringify(data, null, 2))) ? 'Copied.' : 'Copy failed.', 'info') }, 'Copy'),
            el('button', { type: 'button', class: 'button-secondary', onclick: () => downloadText(JSON.stringify(data, null, 2), `${base}.json`, 'application/json') }, icon('download', { size: 14 }), 'JSON'),
            md ? el('button', { type: 'button', class: 'button-secondary', onclick: () => downloadText(md, `${base}.md`, 'text/markdown;charset=utf-8') }, icon('download', { size: 14 }), 'Markdown') : null);
        const tabs = tabStrip({
            label: 'Result views', className: 'result-tab', listClass: 'result-tabs', panelClass: 'result-tab-content',
            tabs: [
                { id: 'formatted', label: 'Fields', panel: el('div', null, formattedPanel(data)) },
                { id: 'ocr', label: 'OCR', panel: el('div', null, ocrPanel(data)) },
                { id: 'enrichment', label: 'Enrichment', panel: el('div', null, enrichmentPanel(data)) },
                { id: 'usage', label: 'Usage', panel: el('div', null, usagePanel(data)) },
                { id: 'raw', label: 'Raw', panel: el('pre', { class: 'json-content' }, displayJson(data)) },
            ],
        });
        const collage = collageOf(data);
        const imageBox = el('div', { class: 'image-result-container' }, el('h4', null, 'Label crop'));
        if (collage) {
            const src = `data:${collage.type};base64,${collage.b64.replace(/\s+/g, '')}`;
            imageBox.append(
                el('button', { type: 'button', class: 'image-button', 'aria-label': 'Enlarge image', onclick: () => zoomImage(src, 'Label crop') },
                    el('img', { class: 'processed-image', src, alt: 'Label crop' })),
                el('div', { class: 'image-actions' }, el('button', {
                    type: 'button', class: 'button-secondary',
                    onclick: () => downloadBlob(base64ToBlob(collage.b64.replace(/\s+/g, ''), collage.type), `${base}_collage.${collage.ext}`),
                }, icon('download', { size: 14 }), 'Download')));
        } else {
            imageBox.appendChild(el('p', { class: 'no-image-msg' }, 'None'));
        }
        replace(resultsArea,
            data.engine_notice ? el('div', { class: 'notice notice-demo', role: 'alert' }, icon('alert', { size: 20 }), el('span', null, String(data.engine_notice))) : null,
            el('div', { class: 'split-results-container' },
                el('div', { class: 'json-result-container' }, controls, tabs.list, tabs.panels),
                imageBox));

        const coords = coordsOf(data);
        mapSection.hidden = false;
        coordStatus.className = `coord-status ${coords ? 'valid' : 'invalid'}`;
        if (coords) {
            const text = `${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)}`;
            replace(coordStatus, icon('pin', { size: 16 }), el('span', { class: 'coord-text' }, text),
                el('button', { type: 'button', class: 'link-button', onclick: async () => toast((await copyText(text)) ? 'Copied.' : 'Copy failed.', 'info') }, 'Copy'),
                externalLink('OpenStreetMap', `https://www.openstreetmap.org/?mlat=${coords.lat}&mlon=${coords.lon}#map=10/${coords.lat}/${coords.lon}`));
            map.setPoint(coords.lat, coords.lon, data.formatted_json.scientificName || text);
        } else {
            replace(coordStatus, 'No coordinates');
            map.clearPoint();
        }
    }

    function renderHistory(index) {
        replace(historySelect, history.map((h, i) => el('option', { value: String(i) }, `${h.time} — ${h.label}`)));
        historySelect.value = String(index);
        historyBar.hidden = history.length < 2;
    }

    /**
     * Add a response (single image or a PDF with pages) to the history and show it.
     * Downloads are named after the file name in the response; `fileName` is used
     * only when the response has none.
     */
    function addResult(label, data, fileName = label) {
        const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const entries = Array.isArray(data.pages)
            ? data.pages.map((page, i) => ({
                time, label: `${data.source_pdf || label} — page ${i + 1}`,
                base: stem(page.filename || `${stem(data.source_pdf || fileName)}__page_${i + 1}`),
                data: { ...page, engine_notice: page.engine_notice || data.engine_notice },
            }))
            : [{ time, label, base: stem(data.filename || fileName), data }];
        if (!entries.length) {
            replace(resultsArea, stateBox('empty', 'The document had no pages to process'));
            return;
        }
        history.unshift(...entries);
        history.length = Math.min(history.length, HISTORY_LIMIT);
        renderHistory(0);
        showResult(history[0]);
    }

    /** One line saying what a local-model run is waiting for, from the engine's state. */
    function stageText(engine, used, seen) {
        const m = (engine && engine.managed) || {};
        const name = localLabels.get(m.loaded_model) || 'the model';
        switch (m.state) {
            case 'starting':
                return 'Starting the engine…';
            case 'loading_model':
                return `Loading ${name} into memory. Large models take a minute or more…`;
            case 'stopping':
                return used.all.length > 1 ? 'Switching to the next model…' : 'Restarting the engine…';
            case 'ready':
                if (!m.active_requests) return seen.worked ? 'Finishing up…' : 'Preparing the image…';
                seen.worked = true;
                if (!used.parse) return 'Reading the label…';
                // One image, one model for both steps: the engine's log shows which step it is on.
                if (seen.steps === 1) return 'Reading the label…';
                if (seen.steps >= 2) return 'Parsing the text into fields…';
                if (used.parse && used.ocr.length && !used.ocr.includes(used.parse)) {
                    return m.loaded_model === used.parse ? 'Parsing the text into fields…' : 'Reading the label…';
                }
                if (!used.ocr.length) return 'Parsing the text into fields…';
                return 'Reading the label, then parsing the text into fields…';
            case 'not_installed':
                return 'The engine is not installed.';
            default:
                return 'Preparing the image…';
        }
    }

    /**
     * Watch a run in progress: a clock, and for local models the engine's state.
     * onStage(text) is called only when the line changes. Returns stop().
     */
    function watchRun({ onTick, onStage, single = false }) {
        const started = Date.now();
        const used = localModelsInUse();
        // Steps can be told apart only for one image going through one local model.
        const countSteps = single && selection.ocr.length === 1 && used.ocr.length === 1 && (!used.parse || used.parse === used.ocr[0]);
        const seen = { worked: false, steps: 0 };
        let logCursor = null;
        let stopped = false;
        let last = null;
        const say = (text) => {
            if (text !== last) {
                last = text;
                onStage(text);
            }
        };
        say(used.all.length ? 'Preparing the image…' : 'Working…');
        onTick(0);
        const clock = setInterval(() => onTick((Date.now() - started) / 1000), 1000);
        let timer = null;
        const readSteps = async () => {
            // Each request the engine begins is one line in its log: the first is OCR, the second parsing.
            const query = logCursor === null ? { limit: '1' } : { after: String(logCursor), limit: '500' };
            const res = await api.get('/v1/engine/logs', { query, timeoutMs: 10000 });
            if (logCursor !== null && res && Array.isArray(res.lines)) {
                seen.steps += res.lines.filter((l) => l && typeof l.text === 'string' && /launch_slot_.*processing task/.test(l.text)).length;
            }
            if (res && Number.isFinite(res.next)) logCursor = res.next;
        };
        const poll = async () => {
            timer = null;
            try {
                if (countSteps) await readSteps().catch(() => {});
                const engine = await api.get('/v1/engine', { timeoutMs: 10000 });
                if (!stopped) say(stageText(engine, used, seen));
            } catch (_) { /* the run itself reports a server that stopped */ }
            if (!stopped) timer = setTimeout(poll, ENGINE_POLL_MS);
        };
        if (used.all.length) {
            if (countSteps) readSteps().catch(() => {}); // remember where the log is now
            timer = setTimeout(poll, 400);
        }
        return () => {
            stopped = true;
            clearInterval(clock);
            if (timer) clearTimeout(timer);
        };
    }

    /**
     * source: {file} or {url}. For a URL, `label` is its name from the URL name
     * parser (or the URL itself when that is not known yet); the result is then
     * listed under the file name the server reports for the image it fetched.
     */
    async function runSingle(source, label, timeoutMs) {
        if (busy) return;
        const fields = requestFields();
        if (!fields) return;
        setBusy(true);
        const stage = el('div', { class: 'vv-loader-stage', role: 'status' });
        const clock = el('span', { class: 'vv-loader-time', 'aria-hidden': 'true' });
        replace(resultsArea, el('div', { class: 'vv-loader' }, el('span', { class: 'spinner', 'aria-hidden': 'true' }),
            el('div', { class: 'vv-loader-text' }, el('div', { class: 'vv-loader-title' }, `Processing ${label}`), stage),
            clock));
        const stopWatching = watchRun({
            single: !/\.pdf$/i.test(label),
            onTick: (seconds) => {
                clock.textContent = fmtClock(seconds);
            },
            onStage: (text) => {
                stage.textContent = text;
            },
        });
        mapSection.hidden = true;
        step5.node.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        try {
            const data = await api.postForm(source.file ? '/process' : '/process-url', formFor(fields, source), { timeoutMs });
            if (!data || typeof data !== 'object') throw new Error('The server sent an empty response.');
            const fetchedName = source.url && typeof data.filename === 'string' && data.filename ? data.filename : null;
            addResult(fetchedName || label, data, source.url ? (source.fileName || 'image') : label);
        } catch (err) {
            inlineError(resultsArea, err, { title: `Could not process ${label}` });
        } finally {
            stopWatching();
            setBusy(false);
        }
    }

    // ------------------------------------------------------------ step 4: inputs
    const previewArea = el('div', { class: 'step4-preview-area' }, el('p', { class: 'no-image-msg' }, 'No image'));
    let previewUrl = null;

    function showPreview(file) {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = null;
        if (!file) {
            replace(previewArea, el('p', { class: 'no-image-msg' }, 'No image'));
            return;
        }
        if (!(file.type || '').startsWith('image/')) {
            replace(previewArea, el('p', { class: 'no-image-msg' }, file.name));
            return;
        }
        previewUrl = URL.createObjectURL(file);
        const img = el('img', { src: previewUrl, alt: `Preview of ${file.name}`, onclick: () => zoomImage(previewUrl, file.name) });
        img.addEventListener('error', () => replace(previewArea, el('p', { class: 'no-image-msg' }, `${file.name} (no preview)`)));
        replace(previewArea, img);
    }

    // Image URL
    const urlInput = el('input', { type: 'url', id: 'imageUrl', placeholder: 'https://…', 'aria-label': 'Image URL', autocomplete: 'off', spellcheck: 'false' });
    let namingUrl = false;
    const urlRun = runButton('Run', async () => {
        const url = urlInput.value.trim();
        if (!isHttpUrl(url)) {
            toast('Enter a valid image URL.', 'warning');
            urlInput.focus();
            return;
        }
        if (busy || namingUrl) return;
        namingUrl = true;
        let fileName = null;
        try {
            fileName = (await namesForUrls([url])).get(url) || null;
        } catch (_) {
            fileName = null; // the run still works; the result carries the file name
        } finally {
            namingUrl = false;
        }
        runSingle({ url, fileName }, fileName || url, NO_TIMEOUT);
    });
    urlInput.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') urlRun.click();
    });
    const urlPanel = el('div', null,
        el('div', { class: 'form-group' }, urlInput,
            el('div', { class: 'example-urls' }, el('span', null, 'Try:'),
                EXAMPLE_URLS.map(([name, url]) => el('button', { type: 'button', class: 'link-button', onclick: () => { urlInput.value = url; } }, name)))),
        urlRun);

    // File upload
    const fileInput = el('input', { type: 'file', id: 'fileInput', accept: 'image/*,.heic,.heif,.tif,.tiff', onchange: () => showPreview(fileInput.files[0]) });
    const filePanel = el('div', null,
        el('div', { class: 'form-group' }, fileInput),
        runButton('Run', () => {
            const file = fileInput.files[0];
            if (!file) {
                toast('Select an image first.', 'warning');
                return;
            }
            runSingle({ file }, file.name, NO_TIMEOUT);
        }));

    // PDF
    const pdfInput = el('input', { type: 'file', id: 'pdfInput', accept: '.pdf,application/pdf', onchange: () => showPreview(pdfInput.files[0]) });
    const pdfPanel = el('div', null,
        el('div', { class: 'form-group' }, el('label', { for: 'pdfInput' }, 'PDF file', infoTip('Each page runs as a separate image.')), pdfInput),
        runButton('Run', () => {
            const file = pdfInput.files[0];
            if (!file) {
                toast('Select a PDF first.', 'warning');
                return;
            }
            runSingle({ file }, file.name, NO_TIMEOUT);
        }));

    // ---- batches (URL list and folder share one runner)
    function batchUi(idPrefix) {
        const slider = el('input', { type: 'range', id: `${idPrefix}Concurrency`, min: '1', max: '8', value: String(Math.min(8, prefs.get('concurrency', 4))) });
        const value = el('span', { class: 'concurrency-value' }, slider.value);
        const note = el('p', { class: 'help-text concurrency-note', id: `${idPrefix}ConcurrencyNote`, hidden: true });
        slider.setAttribute('aria-describedby', note.id);
        const item = { slider, value, note, touched: false };
        sliders.push(item);
        slider.addEventListener('input', () => {
            value.textContent = slider.value;
            item.touched = true;
            // Remembered for other models; a choice made for the local engine's limit is not.
            if (!localModelsInUse().all.length && !openRouterLimit()) prefs.set('concurrency', Number(slider.value));
            applyConcurrencyLimit();
        });
        const saveCsv = el('input', { type: 'checkbox', id: `${idPrefix}SaveCsv`, checked: true });
        const bar = el('div', { class: 'progress-bar' });
        const text = el('span', { class: 'progress-text' });
        const stage = el('span', { class: 'progress-stage' });
        const count = el('span', { class: 'progress-count' });
        const downloads = el('div', { class: 'progress-download-options' });
        const stop = el('button', { type: 'button', class: 'button-secondary button-danger button-small', hidden: true }, 'Stop');
        const progress = el('div', { class: 'batch-progress', hidden: true },
            el('div', { class: 'progress-track', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100' }, bar),
            el('div', { class: 'progress-stats' }, el('span', { class: 'progress-status', role: 'status' }, text, stage), el('span', { class: 'row' }, count, stop)),
            downloads);
        const results = el('div', { class: 'batch-results' });
        return {
            slider, saveCsv, bar, text, stage, count, downloads, stop, progress, results,
            controls: (run) => [
                el('div', { class: 'form-group' }, el('label', { class: 'check-row', for: saveCsv.id }, saveCsv, 'Auto-save CSV')),
                el('div', { class: 'batch-controls' },
                    el('div', { class: 'concurrency-selector' }, el('label', { for: slider.id }, 'Parallel:'), slider, value),
                    run),
                note,
                progress, results,
            ],
        };
    }

    function batchRows(results) {
        const fields = [];
        for (const r of results) {
            if (r.success && hasRecord(r.response)) {
                for (const k of Object.keys(r.response.formatted_json)) if (!fields.includes(k)) fields.push(k);
            }
        }
        const summary = results.map((r) => ({
            Source: r.source, Status: r.success ? 'Success' : 'Failed', Error: r.success ? '' : r.error,
            'OCR Engine': r.success && r.response.ocr_info && typeof r.response.ocr_info === 'object' ? Object.keys(r.response.ocr_info).join('+') : '',
            'Formatted JSON': r.success && hasRecord(r.response) ? r.response.formatted_json : '',
        }));
        const detailed = results.filter((r) => r.success && hasRecord(r.response)).map((r) => ({ Source: r.source, ...r.response.formatted_json }));
        return { fields, summary, detailed };
    }

    /** A zip with one entry per result, named after each result's file name. */
    function zipOf(list, extension, textOf) {
        const names = uniqueNames(list.map((r) => `${stem(r.fileName)}.${extension}`));
        return makeZip(list.map((r, i) => ({ name: names[i], text: textOf(r) })));
    }

    function batchDownloads(ui, results, tag) {
        const { fields, summary, detailed } = batchRows(results);
        const ok = results.filter((r) => r.success);
        const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
        const detailedCsv = () => downloadText(toCsv(['Source', ...fields], detailed), `vvl_${tag}_results_${stamp}.csv`, 'text/csv;charset=utf-8');
        const button = (label, enabled, fn) => el('button', { type: 'button', class: 'button-secondary button-small', disabled: !enabled, onclick: fn }, icon('download', { size: 13 }), label);
        const mds = ok.filter((r) => typeof r.response.formatted_md === 'string' && r.response.formatted_md.trim());
        replace(ui.downloads,
            button('Summary CSV', results.length > 0, () => downloadText(toCsv(['Source', 'Status', 'Error', 'OCR Engine', 'Formatted JSON'], summary), `vvl_${tag}_summary_${stamp}.csv`, 'text/csv;charset=utf-8')),
            button('Results CSV', detailed.length > 0, detailedCsv),
            button('Results JSON', detailed.length > 0, () => downloadBlob(zipOf(ok.filter((r) => hasRecord(r.response)), 'json', (r) => JSON.stringify(r.response.formatted_json, null, 2)), `vvl_${tag}_json_${stamp}.zip`)),
            button('Full JSON', ok.length > 0, () => downloadBlob(zipOf(ok, 'json', (r) => JSON.stringify(r.response, null, 2)), `vvl_${tag}_full_json_${stamp}.zip`)),
            mds.length ? button('Markdown', true, () => downloadBlob(zipOf(mds, 'md', (r) => r.response.formatted_md), `vvl_${tag}_markdown_${stamp}.zip`)) : null);
        if (ui.saveCsv.checked && detailed.length) detailedCsv();
    }

    /**
     * items: [{name, fileName, source, url? | file?}]. `name` is what the progress
     * rows show; `fileName` names the downloads. For a URL both are replaced by
     * the file name in the response once the image has been fetched.
     */
    async function runBatch(items, ui, tag) {
        if (busy) return;
        const fields = requestFields();
        if (!fields) return;
        setBusy(true);
        const concurrency = Number(ui.slider.value) || 1;
        // Two passes when the engine says so (models that cannot be loaded together,
        // or the Engine setting), and only when there is a parsing step with a local model.
        const plan = localModelsInUse().all.length && needsParse() ? await loadEngineInfo() : null;
        const staged = Boolean(plan && plan.staged);
        let phase = '';
        let read = 0;
        const results = new Array(items.length);
        const controller = new AbortController();
        let done = 0;
        let notice = null;
        clear(ui.results);
        clear(ui.downloads);
        ui.progress.hidden = false;
        ui.stop.hidden = false;
        ui.stop.onclick = () => controller.abort();
        const update = () => {
            const total = staged ? items.length * 2 : items.length;
            const pct = total ? Math.round(((staged ? done * 2 + Math.max(0, read - done) : done) / total) * 100) : 100;
            ui.bar.style.width = `${pct}%`;
            ui.bar.parentElement.setAttribute('aria-valuenow', String(pct));
            ui.count.textContent = `${done}/${items.length}`;
            const running = staged && phase === 'Reading labels' ? `Reading labels: ${read + done} of ${items.length}…`
                : (staged && phase === 'Parsing' ? `Parsing: ${done} of ${items.length}…` : `Processing ${done} of ${items.length}…`);
            ui.text.textContent = controller.signal.aborted ? 'Stopped' : (done === items.length ? 'Processing complete' : running);
        };
        update();
        let clockText = '';
        let stageLine = '';
        const showStage = () => {
            ui.stage.textContent = [stageLine, clockText].filter(Boolean).join(' · ');
        };
        const stopWatching = watchRun({
            onTick: (seconds) => {
                clockText = fmtClock(seconds);
                showStage();
            },
            onStage: (line) => {
                // Only the engine's own states are worth showing beside a batch counter.
                stageLine = /engine|Loading|Switching/.test(line) ? line : '';
                showStage();
            },
        });

        const record = (i, result) => {
            results[i] = result;
            done += 1;
            update();
            const detail = result.success
                ? ((hasRecord(result.response) && result.response.formatted_json.scientificName) || 'Done')
                : result.error;
            ui.results.appendChild(el('div', { class: ['result-item', result.success ? 'success' : 'error'] },
                el('span', { class: 'result-item-name' }, result.name),
                el('span', { class: 'result-item-detail' }, String(detail)),
                result.success ? el('button', { type: 'button', class: 'link-button', onclick: () => addResult(result.name, result.response, result.fileName) }, 'View') : null));
        };
        const finished = (i, response) => {
            const item = items[i];
            const fetchedName = item.url && typeof response.filename === 'string' && response.filename ? response.filename : null;
            if (response.engine_notice) notice = String(response.engine_notice);
            return {
                name: fetchedName || item.name, fileName: fetchedName || item.fileName || `image_${i + 1}`,
                source: item.source, success: true, response,
            };
        };
        const failed = (i, err) => ({
            name: items[i].name, fileName: items[i].fileName || `image_${i + 1}`, source: items[i].source,
            success: false, error: describeError(err),
        });

        /** Run `task(i)` for every index in `todo`, `concurrency` at a time. */
        async function pass(todo, task) {
            let cursor = 0;
            async function worker() {
                while (cursor < todo.length && !controller.signal.aborted) {
                    const i = todo[cursor];
                    cursor += 1;
                    await task(i);
                }
            }
            await Promise.all(Array.from({ length: Math.min(concurrency, todo.length) }, worker));
        }

        // One request per image, start to finish.
        const onePass = (i) => api.postForm(items[i].file ? '/process' : '/process-url', formFor(fields, items[i]),
            { timeoutMs: NO_TIMEOUT, signal: controller.signal }).then((response) => {
            if (!response || typeof response !== 'object') throw new Error('The server sent an empty response.');
            record(i, finished(i, response));
        }).catch((err) => {
            if (!controller.signal.aborted) record(i, failed(i, err));
        });

        // Two passes: OCR for every image, then parsing for every image, so the
        // engine swaps models once per batch instead of twice per image.
        async function twoPasses() {
            const tokens = new Map();
            const finishFields = Object.fromEntries(fields.filter(([k]) => ['llm_model', 'prompt', 'include_wfo', 'include_cop90', 'llm_thinking_level'].includes(k)));
            phase = 'Reading labels';
            update();
            await pass(items.map((_, i) => i), (i) => api.postForm(items[i].file ? '/process' : '/process-url',
                formFor([...fields, ['stage', 'ocr']], items[i]), { timeoutMs: NO_TIMEOUT, signal: controller.signal }).then((response) => {
                if (!response || typeof response !== 'object') throw new Error('The server sent an empty response.');
                if (response.staged && response.stage_token) {
                    tokens.set(i, { token: response.stage_token, filename: response.filename });
                    read += 1;
                    update();
                } else {
                    record(i, finished(i, response));     // already complete (nothing to parse)
                }
            }).catch((err) => {
                if (!controller.signal.aborted) record(i, failed(i, err));
            }));
            if (controller.signal.aborted) return;
            phase = 'Parsing';
            update();
            await pass([...tokens.keys()], (i) => api.post('/v1/process/finish', { ...finishFields, stage_token: tokens.get(i).token },
                { timeoutMs: NO_TIMEOUT, signal: controller.signal }).then((response) => {
                if (!response || typeof response !== 'object') throw new Error('The server sent an empty response.');
                record(i, finished(i, response));
            }).catch((err) => {
                if (!controller.signal.aborted) record(i, failed(i, err));
            }));
        }

        try {
            if (staged) await twoPasses();
            else await pass(items.map((_, i) => i), onePass);
        } finally {
            stopWatching();
            stageLine = '';
            showStage();
            ui.stop.hidden = true;
            update();
            const finished = results.filter(Boolean);
            if (notice) ui.results.prepend(el('div', { class: 'notice notice-demo', role: 'alert' }, icon('alert', { size: 20 }), el('span', null, notice)));
            batchDownloads(ui, finished, tag);
            setBusy(false);
        }
    }

    // Batch URLs
    const urlBatch = batchUi('batchUrl');
    const urlListInput = el('input', { type: 'file', id: 'batchUrlFileInput', accept: '.txt,.csv,text/plain,text/csv' });
    const urlColumn = el('input', { type: 'text', id: 'csvUrlColumn', value: 'url' });
    const urlListPreview = el('div', { class: 'file-list' });
    let urlItems = [];
    let urlListSeq = 0;

    async function readUrlList() {
        const seq = urlListSeq += 1;
        urlItems = [];
        clear(urlListPreview);
        const file = urlListInput.files[0];
        if (!file) return;
        try {
            const text = await file.text();
            let urls;
            if (/\.csv$/i.test(file.name)) {
                const { headers, records } = parseCsv(text);
                let column = urlColumn.value.trim() || 'url';
                if (!headers.includes(column)) {
                    const guess = headers.find((h) => /url|link|image/i.test(h));
                    if (!guess) throw new Error(`Column "${column}" was not found. Columns in this file: ${headers.join(', ') || 'none'}.`);
                    column = guess;
                    urlColumn.value = guess;
                }
                urls = records.map((r) => (r[column] || '').trim());
            } else {
                urls = text.split(/\r?\n/).map((line) => line.trim());
            }
            const valid = urls.filter((u) => u && isHttpUrl(u));
            const skipped = urls.filter((u) => u && !isHttpUrl(u)).length;
            // One request names the whole list. If it fails, each row shows its URL until its image is fetched.
            let names = new Map();
            try {
                names = await namesForUrls([...new Set(valid)]);
            } catch (_) {
                names = new Map();
            }
            if (seq !== urlListSeq) return; // another list was chosen meanwhile
            urlItems = valid.map((url) => ({ name: names.get(url) || url, fileName: names.get(url) || null, source: url, url }));
            replace(urlListPreview,
                el('div', null, el('strong', null, `${valid.length} URL${valid.length === 1 ? '' : 's'} found`), skipped ? ` (${skipped} line${skipped === 1 ? '' : 's'} skipped)` : ''),
                urlItems.slice(0, 5).map((item) => el('div', null, item.fileName ? el('strong', null, item.fileName) : null, item.fileName ? ' — ' : '', item.url)),
                valid.length > 5 ? el('div', null, `… and ${valid.length - 5} more`) : null);
        } catch (err) {
            if (seq !== urlListSeq) return;
            inlineError(urlListPreview, err, { title: 'The list could not be read', compact: true });
        }
    }
    urlListInput.addEventListener('change', readUrlList);
    urlColumn.addEventListener('change', readUrlList);
    const batchUrlPanel = el('div', null,
        el('div', { class: 'form-group' }, el('label', { for: 'batchUrlFileInput' }, 'URL file (.txt or .csv)', infoTip('One URL per line (.txt), or a column of URLs (.csv).')),
            urlListInput, urlListPreview),
        el('div', { class: 'form-group' }, el('label', { for: 'csvUrlColumn' }, 'CSV URL column'), urlColumn),
        urlBatch.controls(runButton('Run', () => {
            if (!urlItems.length) {
                toast('Select a URL file first.', 'warning');
                return;
            }
            runBatch(urlItems, urlBatch, 'urls');
        })));

    // Batch folder
    const folderBatch = batchUi('batchFolder');
    const folderInput = el('input', { type: 'file', id: 'imageFolderInput', multiple: true, webkitdirectory: true, 'aria-label': 'Choose a folder of images' });
    const folderList = el('div', { class: 'file-list' });
    let folderFiles = [];

    function setFolderFiles(files) {
        folderFiles = files.filter((f) => (f.type || '').startsWith('image/') || IMAGE_NAME_RE.test(f.name))
            .sort((a, b) => a.name.localeCompare(b.name, 'en', { numeric: true }));
        replace(folderList,
            el('div', null, el('strong', null, `${folderFiles.length} image${folderFiles.length === 1 ? '' : 's'} selected`)),
            folderFiles.slice(0, 8).map((f) => el('div', null, f.name)),
            folderFiles.length > 8 ? el('div', null, `… and ${folderFiles.length - 8} more`) : null);
        showPreview(folderFiles[0] || null);
    }

    function readEntry(entryNode) {
        return new Promise((resolve) => {
            if (entryNode.isFile) {
                entryNode.file((f) => resolve([f]), () => resolve([]));
            } else if (entryNode.isDirectory) {
                const reader = entryNode.createReader();
                const all = [];
                const pump = () => reader.readEntries(async (batch) => {
                    if (!batch.length) {
                        resolve((await Promise.all(all.map(readEntry))).flat());
                        return;
                    }
                    all.push(...batch);
                    pump();
                }, () => resolve([]));
                pump();
            } else {
                resolve([]);
            }
        });
    }

    const dropZone = el('div', { class: 'folder-drop-zone', id: 'imageFolderDropZone' },
        el('p', null, 'Drop a folder or images here'), folderInput, folderList);
    folderInput.addEventListener('change', () => setFolderFiles([...folderInput.files]));
    for (const name of ['dragenter', 'dragover']) {
        dropZone.addEventListener(name, (ev) => {
            ev.preventDefault();
            dropZone.classList.add('drag-over');
        });
    }
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', async (ev) => {
        ev.preventDefault();
        dropZone.classList.remove('drag-over');
        const entries = [...(ev.dataTransfer.items || [])].map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
        const files = entries.length ? (await Promise.all(entries.map(readEntry))).flat() : [...ev.dataTransfer.files];
        setFolderFiles(files);
    });
    const batchFolderPanel = el('div', null, dropZone,
        folderBatch.controls(runButton('Run', () => {
            if (!folderFiles.length) {
                toast('Select a folder first.', 'warning');
                return;
            }
            runBatch(folderFiles.map((file) => ({ name: file.name, fileName: file.name, source: file.webkitRelativePath || file.name, file })), folderBatch, 'folder');
        })));

    const inputTabs = tabStrip({
        label: 'Image source',
        initial: prefs.get('inputTab', 'url'),
        onChange: (id) => prefs.set('inputTab', id),
        tabs: [
            { id: 'url', label: 'URL', panel: urlPanel },
            { id: 'file', label: 'Image', panel: filePanel },
            { id: 'batchUrls', label: 'URL list', panel: batchUrlPanel },
            { id: 'batchFolder', label: 'Folder', panel: batchFolderPanel },
            { id: 'pdf', label: 'PDF', panel: pdfPanel },
        ],
    });
    const step4 = step(4, 'Input',
        el('div', { class: 'step4-split' },
            el('div', { class: 'step4-left' }, inputTabs.list, inputTabs.panels),
            el('div', { class: 'step4-right' }, el('h4', null, 'Preview'), previewArea)));

    // ------------------------------------------------------------ assemble
    replace(panel, step1.node, step2.node, step3.node, step4.node, step5.node);
    on('models-changed', loadModels);
    // A finished download adds a model to the pickers.
    on('job-finished', (job) => {
        if (job && job.kind === 'model_download' && job.state === 'done') loadModels();
    });
    on('engine-changed', () => {
        loadEngineInfo();
        loadModels();
    });
    on('prompts-changed', loadPrompts);
    on('use-prompt', ({ name }) => {
        promptName = name;
        prefs.set('prompt', name);
        if (prompts.some((p) => p.name === name)) renderPrompts();
        else loadPrompts();
    });
    await Promise.all([loadModels(), loadEnrichment(), loadPrompts(), loadEngineInfo()]);
    panel.dataset.state = 'ready';

    return {
        show() {
            map.resize();
        },
    };
}
