// Keys tab: one card per credential. Key values are write-only: they go from
// the input straight into one request and are never kept in a variable that
// outlives it, in web storage, or in the DOM after saving.

import { api, inlineError, toast, toastError } from '../api.js';
import { el, icon, infoTip, replace, stateBox, badge, externalLink, confirmDialog, fmtDateTime } from '../dom.js';
import { emit, getAppInfo } from '../state.js';

const GROUPS = [
    ['model_provider', 'AI models'],
    ['enrichment', 'Enrichment'],
    ['downloads', 'Downloads'],
];
const TEST_KIND = { ok: 'good', invalid_key: 'bad', not_configured: 'warn', network_error: 'warn', error: 'bad' };
const TEST_LABEL = { ok: 'Working', invalid_key: 'Invalid', not_configured: 'Not set', network_error: 'Offline', error: 'Failed' };

export async function mount(panel) {
    const info = getAppInfo() || {};
    const privateFile = (info.paths && info.paths.private_file) || 'PRIVATE.yaml';
    const body = el('div');
    const cards = new Map();

    function keyCard(initial) {
        const card = el('section', { class: 'card key-card', id: `key-card-${initial.id}`, tabindex: '-1', 'aria-label': initial.label });
        let provider = initial;
        let busy = null;

        function render() {
            const p = provider;
            const inputs = [];
            const singleField = (p.fields || []).length === 1;
            const fieldNodes = (p.fields || []).map((f) => {
                const id = `key-${p.id}-${f.name}`;
                const input = el('input', {
                    type: 'password', id, name: f.name, placeholder: f.placeholder || 'Paste key…',
                    autocomplete: 'off', spellcheck: 'false', autocapitalize: 'off', disabled: Boolean(busy),
                    'aria-label': f.label || f.name,
                });
                inputs.push(input);
                const toggle = el('button', { type: 'button', class: 'icon-button', 'aria-label': 'Show', 'aria-pressed': 'false', title: 'Show/hide' }, icon('eye', { size: 16 }));
                toggle.addEventListener('click', () => {
                    const show = input.type === 'password';
                    input.type = show ? 'text' : 'password';
                    toggle.setAttribute('aria-pressed', String(show));
                    toggle.setAttribute('aria-label', show ? 'Hide' : 'Show');
                    replace(toggle, icon(show ? 'eyeOff' : 'eye', { size: 16 }));
                });
                input.addEventListener('keydown', (ev) => {
                    if (ev.key === 'Enter') save();
                });
                return el('div', { class: 'key-field' },
                    singleField ? null : el('label', { for: id }, f.label || f.name,
                        f.saved ? badge('Saved', 'good') : badge('Not set', 'neutral'),
                        f.saved && f.masked ? el('span', { class: 'key-mask' }, f.masked) : null),
                    el('div', { class: 'key-input-row' }, input, toggle));
            });
            const anySaved = (p.fields || []).some((f) => f.saved);
            const firstMasked = ((p.fields || []).find((f) => f.saved && f.masked) || {}).masked;

            async function save() {
                const fields = {};
                for (const input of inputs) {
                    if (input.value.trim()) fields[input.name] = input.value.trim();
                    input.value = '';
                    input.type = 'password';
                }
                if (!Object.keys(fields).length) {
                    toast('Paste a key first.', 'warning');
                    return;
                }
                await act('save', () => api.put(`/v1/keys/${encodeURIComponent(p.id)}`, { fields }), 'Saved.');
            }

            async function remove() {
                const yes = await confirmDialog({ title: `Remove ${p.label} key?`, message: 'Deletes the saved key from this computer.', confirmLabel: 'Remove', danger: true });
                if (yes) await act('remove', () => api.del(`/v1/keys/${encodeURIComponent(p.id)}`), 'Removed.');
            }

            async function test() {
                busy = 'test';
                render();
                try {
                    const result = await api.post(`/v1/keys/${encodeURIComponent(p.id)}/test`, {}, { timeoutMs: 30000 });
                    provider = { ...provider, last_test: result };
                } catch (err) {
                    toastError(err, `${p.label} test`);
                } finally {
                    busy = null;
                    render();
                }
            }

            const t = p.last_test;
            replace(card,
                el('div', { class: 'card-header' }, el('h3', null, p.label),
                    p.configured ? badge('Saved', 'good') : badge('Not set', 'neutral'),
                    firstMasked ? el('span', { class: 'key-mask-inline' }, firstMasked) : null,
                    el('span', { class: 'spacer' }),
                    p.sends_images
                        ? el('span', { class: 'sends-note is-remote privacy-badge-tip' }, icon('cloud', { size: 14 }), 'Cloud', infoTip('Images are sent to this provider when used.', 'end'))
                        : el('span', { class: 'sends-note is-local privacy-badge-tip' }, icon('shield', { size: 14 }), 'No images', infoTip('Never receives your specimen images.', 'end'))),
                p.purpose ? el('p', { class: 'key-purpose' }, p.purpose) : null,
                el('div', { class: 'key-links' },
                    p.key_url ? externalLink('Get key', p.key_url) : null,
                    p.docs_url ? externalLink('Docs', p.docs_url) : null),
                fieldNodes,
                el('div', { class: 'key-actions' },
                    el('button', { type: 'button', class: 'button', disabled: Boolean(busy), onclick: save }, busy === 'save' ? 'Saving…' : 'Save'),
                    el('button', { type: 'button', class: 'button-secondary', disabled: Boolean(busy) || !p.configured, onclick: test }, busy === 'test' ? 'Testing…' : 'Test'),
                    el('button', { type: 'button', class: 'button-secondary button-danger', disabled: Boolean(busy) || !anySaved, onclick: remove }, icon('trash', { size: 14 }), 'Remove')),
                t ? el('div', { class: 'key-test', role: 'status' },
                    badge(TEST_LABEL[t.status] || (t.ok ? 'Working' : 'Failed'), TEST_KIND[t.status] || (t.ok ? 'good' : 'bad')),
                    t.message ? el('span', null, t.message) : null,
                    t.tested_at ? el('span', { class: 'key-test-time' }, fmtDateTime(t.tested_at)) : null) : null);
        }

        async function act(kind, call, doneMessage) {
            busy = kind;
            render();
            try {
                const updated = await call();
                if (updated && updated.id) provider = updated;
                toast(doneMessage, 'success');
                emit('providers-changed');
                emit('models-changed');
            } catch (err) {
                toastError(err, provider.label);
            } finally {
                busy = null;
                render();
            }
        }

        render();
        return card;
    }

    async function load() {
        panel.dataset.state = 'loading';
        replace(body, stateBox('loading', 'Loading keys…'));
        cards.clear();
        try {
            const data = await api.get('/v1/providers');
            const providers = (data && data.providers) || [];
            if (!providers.length) {
                replace(body, stateBox('empty', 'No services are listed'));
            } else {
                const known = new Set(GROUPS.map(([role]) => role));
                const groups = [...GROUPS, ['', 'Other']];
                replace(body, groups.map(([role, title]) => {
                    const list = providers.filter((p) => (role ? p.role === role : !known.has(p.role)));
                    if (!list.length) return null;
                    return el('section', { 'aria-label': title }, el('h2', { class: 'section-title' }, title),
                        el('div', { class: 'card-grid' }, list.map((p) => {
                            const card = keyCard(p);
                            cards.set(p.id, card);
                            return card;
                        })));
                }));
            }
            panel.dataset.state = 'ready';
        } catch (err) {
            inlineError(body, err, { title: 'Keys could not be loaded', retry: load });
            panel.dataset.state = 'error';
        }
    }

    replace(panel,
        el('div', { class: 'keys-privacy' }, icon('shield', { size: 22 }),
            el('div', null,
                el('p', null,
                    el('strong', null, 'Keys stay on this computer.'),
                    ' All keys are optional.',
                    infoTip(`Saved locally in ${privateFile}. Saved keys aren't shown again—paste a new key to replace.`, 'start')))),
        body);
    await load();

    return {
        show(target) {
            const card = target ? cards.get(target) : null;
            if (!card) return;
            for (const c of cards.values()) c.classList.remove('is-highlighted');
            card.classList.add('is-highlighted');
            card.scrollIntoView({ block: 'center' });
            card.focus({ preventScroll: true });
        },
    };
}
