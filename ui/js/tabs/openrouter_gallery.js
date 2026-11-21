// The OpenRouter model gallery on the Models tab: a filter rail and a sortable
// table (docs/API.md section 14). Models added here appear in the OCR and
// parsing pickers on the VVL tab.

import { api, inlineError, toastError } from '../api.js';
import { el, icon, infoTip, replace, stateBox, badge } from '../dom.js';
import { emit, gotoTab } from '../state.js';

const OWNER_COLORS = {
    anthropic: '#D97757', openai: '#10A37F', google: '#4796E3', mistralai: '#FA520F', qwen: '#615CED',
    meta: '#0668E1', 'meta-llama': '#0668E1', 'x-ai': '#8a8f98', deepseek: '#4D6BFE', 'z-ai': '#2f6fed',
    moonshotai: '#16a3b7', nvidia: '#76B900', amazon: '#FF9900', cohere: '#39594D', microsoft: '#00A4EF',
    xiaomi: '#FF6900', tencent: '#0052D9', perplexity: '#20808D',
};
const COST_STEPS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, null];
const AGES = [['any', 'Any'], ['3', '3 months'], ['6', '6 months'], ['12', '1 year'], ['24', '2 years']];
const ROLES = [['ocr', 'OCR'], ['parse', 'Parsing'], ['both', 'Both']];
const PAGE = 100;

const usd = (v) => {
    if (v === 0) return 'Free';
    if (v < 0.01) return '<$0.01';
    if (v < 100) return `$${v.toFixed(2)}`;
    return `$${Math.round(v).toLocaleString('en-US')}`;
};
const perM = (v) => (v === 0 ? 'Free' : `$${v < 1 ? v.toFixed(3).replace(/0$/, '') : v.toFixed(2)}`);
const ctx = (n) => (!n ? '—' : n >= 1e6 ? `${(n / 1e6).toFixed(1).replace(/\.0$/, '')}M` : `${Math.round(n / 1000)}K`);

function dot(owner) {
    const node = el('span', { class: 'org-dot' });
    // Set through the style object: the page's security policy refuses style="" attributes.
    if (OWNER_COLORS[owner]) node.style.setProperty('--owner', OWNER_COLORS[owner]);
    return node;
}

export function mountOpenRouterGallery(host, provider) {
    const state = {
        role: 'ocr', q: '', owners: new Set(), cost: COST_STEPS.length - 1, age: 'any', sort: 'cost',
        reasoning: false, structured: false, hideFree: true, limit: PAGE,
    };
    let data = null;       // GET /v1/openrouter/catalog
    let account = null;    // GET /v1/openrouter/account
    let byId = new Map();
    let results = el('div');
    const mine = el('div', { class: 'org-mine' });
    const head = el('div', { class: 'org-head' });

    const now = () => (data && data.fetched_at) || Date.now() / 1000;
    const months = (m) => (m.created ? (now() - m.created) / (30.44 * 86400) : null);
    const released = (m) => (m.created
        ? new Date(m.created * 1000).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }) : 'Unknown');
    function ago(m) {
        const n = months(m);
        if (n === null) return '';
        if (n < 1) return 'this month';
        if (n < 12) return `${Math.round(n)} mo ago`;
        return `${(n / 12).toFixed(1).replace(/\.0$/, '')} yr ago`;
    }
    /** {typical, high} USD per 1,000 images for the role being viewed. */
    function cost(m) {
        if (state.role === 'both') {
            return { typical: m.cost_1k.ocr.typical + m.cost_1k.parse.typical, high: m.cost_1k.ocr.high + m.cost_1k.parse.high };
        }
        return m.cost_1k[state.role];
    }
    const eligible = (m) => (state.role === 'parse' ? true : m.vision);
    const privacyKnown = () => data.models.some((m) => m.private !== null && m.private !== undefined);

    function filtered() {
        const q = state.q.trim().toLowerCase();
        const max = COST_STEPS[state.cost];
        const maxAge = state.age === 'any' ? null : Number(state.age);
        const out = data.models.filter((m) => {
            if (!eligible(m)) return false;
            if (state.owners.size && !state.owners.has(m.owner)) return false;
            if (data.private_only && m.private !== true) return false;
            if (q && !`${m.name} ${m.id} ${m.owner_label}`.toLowerCase().includes(q)) return false;
            if (max !== null && cost(m).typical > max) return false;
            if (maxAge !== null && (months(m) === null || months(m) > maxAge)) return false;
            if (state.reasoning && !m.reasoning) return false;
            if (state.structured && !m.structured_output) return false;
            if (state.hideFree && m.free) return false;
            return true;
        });
        const by = {
            cost: (a, b) => cost(a).typical - cost(b).typical || a.name.localeCompare(b.name),
            cost_desc: (a, b) => cost(b).typical - cost(a).typical,
            new: (a, b) => (b.created || 0) - (a.created || 0),
            old: (a, b) => (a.created || Infinity) - (b.created || Infinity),
            name: (a, b) => a.name.localeCompare(b.name),
            context: (a, b) => (b.context_length || 0) - (a.context_length || 0),
        }[state.sort];
        return out.sort(by);
    }

    function owners() {
        const map = new Map();
        for (const m of data.models) {
            if (!eligible(m)) continue;
            const o = map.get(m.owner) || { id: m.owner, label: m.owner_label, count: 0 };
            o.count += 1;
            map.set(m.owner, o);
        }
        return [...map.values()].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
    }

    // ------------------------------------------------------------ changes
    function accept(payload) {
        data = payload;
        byId = new Map(data.models.map((m) => [m.id, m]));
    }
    async function changeModels(body) {
        try {
            accept(await api.put('/v1/openrouter/models', body, { timeoutMs: 30000 }));
            renderHead();
            renderMine();
            refresh();
            emit('models-changed');
        } catch (err) {
            toastError(err, 'OpenRouter models');
        }
    }
    async function saveSetting(change, apply) {
        try {
            await api.put('/v1/settings', { openrouter: change });
            apply();
            emit('models-changed');
        } catch (err) {
            toastError(err, 'OpenRouter setting');
        }
    }

    // ------------------------------------------------------------- pieces
    function addButton(m) {
        const on = data.added.includes(m.id);
        return el('button', {
            type: 'button', class: ['org-add', on ? 'is-added' : ''], 'aria-pressed': String(on),
            title: on ? 'Remove from the OCR and parsing lists' : 'Show this model in the OCR and parsing lists on the VVL tab',
            onclick: () => changeModels(on ? { remove: [m.id] } : { add: [m.id] }),
        }, on ? '✓ Added' : '+ Add');
    }
    function tags(m) {
        return el('span', { class: 'org-tags' },
            el('span', { class: ['org-tag', m.vision ? 'is-img' : ''] }, m.vision ? 'Vision' : 'Text'),
            m.private ? el('span', { class: 'org-tag is-private', title: `Zero-retention: ${(m.private_providers || []).join(', ')}` },
                icon('shield', { size: 11 }), el('span', { class: 'visually-hidden' }, 'Private')) : null,
            m.reasoning ? el('span', { class: 'org-tag', title: 'Supports thinking levels' }, 'Thinking') : null,
            m.structured_output ? el('span', { class: 'org-tag', title: 'Strict JSON output' }, 'JSON') : null,
            m.free ? el('span', { class: 'org-tag is-free', title: 'Rate-limited (~20/min)' }, 'Free') : null);
    }
    function check(key, label, title) {
        return el('label', { class: 'org-check', title },
            el('input', { type: 'checkbox', checked: state[key], onchange: (e) => { state[key] = e.target.checked; state.limit = PAGE; refresh(); } }), label);
    }

    function renderHead() {
        const [low, high] = data.concurrency_range || [1, 4];
        const levels = [];
        for (let n = low; n <= high; n += 1) levels.push(n);
        let credit = null;
        if (account && account.ok) {
            const left = account.limit_remaining_usd;
            credit = el('span', { class: 'org-credit' }, Number.isFinite(left) ? `${usd(left)} left` : `${usd(account.usage_usd || 0)} used`);
        } else if (account && account.configured && account.message) {
            credit = el('span', { class: 'org-credit is-bad' }, account.message);
        }
        replace(head,
            el('h3', null, provider.label),
            data.configured ? badge('Connected', 'good') : badge('No key', 'warn'),
            el('button', { type: 'button', class: 'link-button', onclick: () => gotoTab('keys', provider.id) }, data.configured ? 'Key' : 'Add key'),
            credit,
            el('span', { class: 'spacer' }),
            el('span', { class: 'org-conc' }, 'Parallel', infoTip('Concurrent requests sent during a batch.'),
                el('span', { class: 'org-seg', role: 'group', 'aria-label': 'Parallel requests' }, levels.map((n) =>
                    el('button', { type: 'button', 'aria-pressed': String(n === data.concurrency),
                        onclick: () => saveSetting({ concurrency: n }, () => { data.concurrency = n; renderHead(); }) }, String(n))))),
            el('button', { type: 'button', class: 'button-secondary button-small', title: 'Refresh model list and prices', onclick: () => load({ refresh: true }) },
                icon('refresh', { size: 13 }), 'Refresh'));
    }

    function renderMine() {
        replace(mine, el('span', { class: 'org-mine-title' }, `Added (${data.added.length})`),
            data.added.length
                ? data.added.map((id) => {
                    const m = byId.get(id);
                    const name = m ? m.name : id;
                    const uses = [];
                    if ((data.selection.ocr || []).includes(`api:openrouter:${id}`)) uses.push('OCR');
                    if (data.selection.parse === `api:openrouter:${id}`) uses.push('Parsing');
                    return el('span', { class: 'org-pill' }, dot(m ? m.owner : null), name,
                        el('small', null, uses.length ? uses.join(' · ') : (!m ? 'unlisted' : (m.vision ? 'OCR · Parsing' : 'Parsing'))),
                        el('button', { type: 'button', 'aria-label': `Remove ${name}`, title: 'Remove', onclick: () => changeModels({ remove: [id] }) }, '×'));
                })
                : el('span', { class: 'help-text' }, 'None added.'));
    }

    function rail() {
        const text = () => (COST_STEPS[state.cost] === null ? 'Any' : `≤ ${usd(COST_STEPS[state.cost])}`);
        const out = el('output', null, text());
        const known = privacyKnown();
        return el('aside', { class: 'org-rail', 'aria-label': 'Filters' },
            el('h4', null, 'Role'),
            el('div', { class: 'org-stack' }, ROLES.map(([id, label]) => el('label', { class: 'org-check' },
                el('input', { type: 'radio', name: 'openRouterRole', checked: state.role === id, onchange: () => { state.role = id; state.limit = PAGE; build(); } }),
                label))),
            el('h4', null, 'Search'),
            el('input', { type: 'text', placeholder: 'Search…', value: state.q, 'aria-label': 'Search models',
                oninput: (e) => { state.q = e.target.value; state.limit = PAGE; refresh(); } }),
            el('h4', null, 'Maker'),
            el('div', { class: 'org-owners' }, owners().map((o) => el('label', null,
                el('input', { type: 'checkbox', checked: state.owners.has(o.id),
                    onchange: (e) => { if (e.target.checked) state.owners.add(o.id); else state.owners.delete(o.id); state.limit = PAGE; refresh(); } }),
                dot(o.id), el('span', null, o.label), el('span', { class: 'org-owner-count' }, o.count)))),
            el('h4', null, 'Privacy'),
            el('label', { class: ['org-check', 'org-private', known ? '' : 'is-disabled'] },
                el('input', { type: 'checkbox', checked: Boolean(data.private_only), disabled: !known,
                    onchange: (e) => { const on = e.target.checked; saveSetting({ private_only: on }, () => { data.private_only = on; state.limit = PAGE; refresh(); }); } }),
                icon('shield', { size: 15 }), 'Zero-retention only',
                infoTip(known ? 'Only use hosts that keep no copy of your images or text.' : 'Zero-retention list unavailable.', 'start')),
            el('h4', null, 'Max cost / 1K'),
            el('div', { class: 'org-cost' },
                el('input', { type: 'range', min: '0', max: String(COST_STEPS.length - 1), step: '1', value: String(state.cost), 'aria-label': 'Max cost per 1,000 images',
                    oninput: (e) => { state.cost = Number(e.target.value); out.textContent = text(); state.limit = PAGE; refresh(); } }), out),
            el('h4', null, 'Released'),
            el('select', { 'aria-label': 'Release date', onchange: (e) => { state.age = e.target.value; state.limit = PAGE; refresh(); } },
                AGES.map(([v, label]) => el('option', { value: v, selected: v === state.age }, label))),
            el('h4', null, 'Features'),
            el('div', { class: 'org-stack' }, check('reasoning', 'Thinking'), check('structured', 'JSON'), check('hideFree', 'Hide free')));
    }

    function table(list) {
        const shown = list.slice(0, state.limit);
        const cols = [['add', '', ''], ['name', 'Model', ''], ['tags', 'Features', ''], ['cost', 'Est. / 1K', 'num'],
            ['price', '$/1M', 'num'], ['context', 'Context', 'num'], ['new', 'Released', 'num']];
        const next = { name: 'name', cost: state.sort === 'cost' ? 'cost_desc' : 'cost', context: 'context', new: state.sort === 'new' ? 'old' : 'new' };
        const active = { name: 'name', cost: 'cost', cost_desc: 'cost', context: 'context', new: 'new', old: 'new' }[state.sort];
        const dir = ['cost', 'old', 'name'].includes(state.sort) ? 'ascending' : 'descending';
        return [el('div', { class: 'table-wrap' }, el('table', { class: 'data org-table' },
            el('thead', null, el('tr', null, cols.map(([id, label, cls]) => el('th', { scope: 'col', class: cls || null, 'aria-sort': active === id ? dir : null },
                next[id]
                    ? el('button', { type: 'button', onclick: () => { state.sort = next[id]; refresh(); } }, label, active === id ? (dir === 'ascending' ? ' ▲' : ' ▼') : '')
                    : (label || el('span', { class: 'visually-hidden' }, 'Add')),
                id === 'cost' ? infoTip('Estimated cost per 1,000 images (typical to long labels; excludes thinking).', 'end') : null)))),
            el('tbody', null, shown.map((m) => {
                const c = cost(m);
                return el('tr', { class: data.added.includes(m.id) ? 'is-added' : null },
                    el('td', { class: 'nowrap' }, addButton(m)),
                    el('td', { title: m.description || null }, el('div', { class: 'org-name' }, m.name), el('div', { class: 'org-sub' }, dot(m.owner), ' ', m.owner_label)),
                    el('td', { class: 'nowrap' }, tags(m)),
                    el('td', { class: 'num' }, el('b', null, usd(c.typical)), c.typical ? el('div', { class: 'org-sub' }, `–${usd(c.high)}`) : null),
                    el('td', { class: 'num' }, m.free ? 'Free' : `${perM(m.price_in)} · ${perM(m.price_out)}`),
                    el('td', { class: 'num' }, ctx(m.context_length)),
                    el('td', { class: 'num' }, released(m)));
            })))),
        shown.length < list.length
            ? el('button', { type: 'button', class: 'button-secondary org-more', onclick: () => { state.limit += 150; refresh(); } }, `Show more (${list.length - shown.length} left)`)
            : null];
    }

    function refresh() {
        const list = filtered();
        const narrowed = state.owners.size || state.q || COST_STEPS[state.cost] !== null || state.age !== 'any' || state.reasoning || state.structured;
        replace(results,
            el('p', { class: 'org-count' }, el('b', null, list.length), ` model${list.length === 1 ? '' : 's'}`,
                list.length > state.limit ? ` · first ${state.limit}` : '',
                narrowed ? [' · ', el('button', { type: 'button', class: 'link-button', onclick: () => {
                    state.owners.clear(); state.q = ''; state.cost = COST_STEPS.length - 1; state.age = 'any'; state.reasoning = false; state.structured = false; build();
                } }, 'Clear')] : null),
            list.length ? table(list) : el('div', { class: 'org-empty' }, 'No matching models.'));
    }

    function build() {
        results = el('div', { class: 'org-results' });
        replace(host, el('div', { class: 'card org-card' }, head, mine,
            data.error ? el('div', { class: 'notice notice-warn' }, icon('alert', { size: 18 }), el('span', null, `${data.error} Showing the last list that was saved.`)) : null,
            el('div', { class: 'org-layout' }, rail(), results)));
        renderHead();
        renderMine();
        refresh();
    }

    async function loadAccount() {
        try {
            account = await api.get('/v1/openrouter/account', { timeoutMs: 30000 });
        } catch (_) {
            account = null;
        }
        if (data) renderHead();
    }

    let loading = null;
    function load({ refresh: fetchNew = false, quiet = false } = {}) {
        if (loading) return loading;
        loading = (async () => {
            if (!quiet) replace(host, stateBox('loading', 'Loading OpenRouter’s model list…'));
            try {
                accept(await api.get('/v1/openrouter/catalog', { query: fetchNew ? { refresh: '1' } : {}, timeoutMs: 90000 }));
                if (!data.models.length) {
                    replace(host, stateBox('error', 'OpenRouter’s model list could not be loaded', data.error || null, { retry: () => load({ refresh: true }) }));
                    return;
                }
                build();
                if (data.configured) loadAccount();
                else account = null;
            } catch (err) {
                inlineError(host, err, { title: 'OpenRouter’s model list could not be loaded', retry: () => load() });
            } finally {
                loading = null;
            }
        })();
        return loading;
    }

    return { load, loaded: () => Boolean(data) };
}
